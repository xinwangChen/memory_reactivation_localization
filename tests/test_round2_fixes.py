"""Regression cases for cached input identity, streaming and implicit schedules."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
import yaml

from pb2023_eeg.artifacts.checkpoints import load_model, save_model
from pb2023_eeg.artifacts.exports import copy_verified_exports, describe_export
from pb2023_eeg.artifacts.provenance import data_identity, digest, file_sha256
from pb2023_eeg.config import parse_request
from pb2023_eeg.contracts import PredictionSpec
from pb2023_eeg.data.dataset import LagDecodingDataset
from pb2023_eeg.data.spatial import build_graph
from pb2023_eeg.evaluation.export import export_target_outputs
from pb2023_eeg.evaluation.predict import iter_predictions
from pb2023_eeg.models.factory import build_model
from pb2023_eeg.training.common import build_optimizer, validate_resume_config
from pb2023_eeg.training.trainer import fit_target_decoder
from pb2023_eeg.workflows import evaluate
from regression.helpers_smoke import TargetPipelineSmokeTest


def config(channels=2, times=8):
    return {"seed": 7, "task": "choice", "data": {}, "train": {}, "model": {
        "encoder": {"type": "cnn", "in_channels": channels, "latent_dim": 4,
                    "num_blocks": 1, "kernel_size": 3, "dropout": 0., "expansion_ratio": 1},
        "attention_mode": "gated_scorer", "classifier_num_layers": 1,
        "resolved_time_point_count": times}}


@pytest.fixture
def native():
    fixture = TargetPipelineSmokeTest()
    fixture.setUp()
    yield fixture
    fixture.tearDown()


def dataset(native, **kwargs):
    return LagDecodingDataset(native.records, native.splits, native.normalization,
        split="val", task="choice", lags=[1], balance_cells=False, **kwargs)


def loader(ds):
    return DataLoader(ds, batch_size=8, sampler=list(range(8)))


def export_args(ds, output):
    return dict(model=build_model(config(64, 40)["model"]).eval(), loader=loader(ds),
        task="choice", device=torch.device("cpu"), amp=False, output_dir=output,
        split="val", attention_indices=torch.arange(40), time_s=ds.time_s,
        progress_bar_enabled=False)


def replace_npz(path, values):
    # Replacing the path also covers readers with an already-open archive fd.
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    replacement = Path(path).with_suffix(".replacement")
    replacement.write_bytes(stream.getvalue())
    replacement.replace(path)


def change_eeg(native):
    path = Path(native.records[0].npz_path)
    with np.load(path) as archive:
        values = {key: archive[key] for key in archive.files}
    values["X"] += 3
    replace_npz(path, values)


@pytest.mark.parametrize("cache", ["memory", "offline", "worker_pickle"])
def test_cached_dataset_cannot_certify_or_reuse_replaced_eeg(tmp_path, native, cache):
    kwargs = {} if cache == "memory" else {"offline_cache_dir": tmp_path / "cache"}
    old = dataset(native, **kwargs)
    args = export_args(old, tmp_path / "export")
    export_target_outputs(**args)
    original = {p.name: p.read_bytes() for p in args["output_dir"].iterdir()}
    if cache == "worker_pickle":
        old = pickle.loads(pickle.dumps(old))
        next(iter(loader(old)))
        args["loader"] = loader(old)
    change_eeg(native)
    with pytest.raises(ValueError, match="changed since loading"):
        data_identity(args["loader"])
    with pytest.raises(ValueError, match="changed since loading"):
        export_target_outputs(**args)
    assert original == {p.name: p.read_bytes() for p in args["output_dir"].iterdir()}
    fresh = dataset(native, **kwargs)
    expected = describe_export(export_target_outputs, **{**args, "loader": loader(fresh)})
    with pytest.raises(ValueError, match="parameters|identity|match"):
        copy_verified_exports(args["output_dir"], tmp_path / "reuse", decoder="target",
            splits=["val"], input_kinds=["real"], expected_parameters={("val", "real"): expected})
    export_target_outputs(**{**args, "loader": loader(fresh), "output_dir": tmp_path / "fresh"})


@pytest.mark.parametrize("source", ["normalization", "metadata", "split"])
def test_cached_metadata_is_bound_to_loaded_version(native, source):
    ds = dataset(native)
    paths = {"normalization": native.normalization, "metadata": native.records[0].meta_path,
             "split": native.splits}
    path = Path(paths[source])
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed since loading"):
        data_identity(loader(ds))
    data_identity(loader(dataset(native)))


def test_content_check_does_not_trust_unchanged_stat_values(native, monkeypatch):
    from pb2023_eeg.utils import source_files
    ds = dataset(native)
    next(iter(loader(ds)))
    path = str(native.records[0].npz_path)
    stamp = ds.store.source_files.stamps[path]
    original = source_files._stamp
    monkeypatch.setattr(source_files, "_stamp", lambda p: stamp if str(p) == path else original(p))
    change_eeg(native)
    with pytest.raises(ValueError, match="changed since loading"):
        data_identity(loader(ds))


@pytest.mark.parametrize("cache", [False, True])
def test_lazy_reads_reject_replacements_after_metadata_loading(tmp_path, native, cache):
    ds = dataset(native, offline_cache_dir=tmp_path / "cache" if cache else None)
    change_eeg(native)
    with pytest.raises(ValueError, match="changed since loading"):
        next(iter(loader(ds)))


def test_export_rechecks_sources_after_prediction_before_publish(tmp_path, native, monkeypatch):
    ds = dataset(native)
    args = export_args(ds, tmp_path / "export")
    export_target_outputs(**args)
    original = {p.name: p.read_bytes() for p in args["output_dir"].iterdir()}
    forward = args["model"].forward
    changed = False
    def mutate(*a, **kw):
        nonlocal changed
        result = forward(*a, **kw)
        if not changed:
            changed = True
            change_eeg(native)
        return result
    monkeypatch.setattr(args["model"], "forward", mutate)
    with pytest.raises(ValueError, match="changed since loading"):
        export_target_outputs(**args)
    assert original == {p.name: p.read_bytes() for p in args["output_dir"].iterdir()}


@pytest.mark.parametrize("outer", [torch.enable_grad, torch.no_grad, torch.inference_mode])
def test_suspended_prediction_stream_preserves_caller_autograd_and_restores_flags(outer):
    model = build_model(config()["model"])
    next(m for m in model.modules() if isinstance(m, torch.nn.Conv1d)).eval()
    flags = [m.training for m in model.modules()]
    batches = [{"eeg": torch.randn(2, 2, 8)}] * 2
    with outer():
        modes = torch.is_grad_enabled(), torch.is_inference_mode_enabled()
        stream = iter_predictions(model, batches, PredictionSpec("choice"))
        next(stream)
        assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == modes
        if modes == (True, False):
            unrelated = torch.nn.Linear(2, 1)
            unrelated(torch.ones(1, 2)).sum().backward()
            assert unrelated.weight.grad is not None
        stream.close()
        assert (torch.is_grad_enabled(), torch.is_inference_mode_enabled()) == modes
    assert [m.training for m in model.modules()] == flags


def test_prediction_error_restores_context_and_model_flags():
    model = build_model(config()["model"]).train()
    stream = iter_predictions(model, [{"eeg": torch.randn(2, 2, 8)}, {"eeg": None}], PredictionSpec("choice"))
    next(stream)
    with pytest.raises(ValueError, match="eeg"):
        next(stream)
    assert model.training and torch.is_grad_enabled() and not torch.is_inference_mode_enabled()


@pytest.fixture
def prediction_files(tmp_path):
    cfg = config()
    model = build_model(cfg["model"]).eval()
    checkpoint, inputs, output = (tmp_path / name for name in ("model.ckpt", "inputs.npz", "outputs.npz"))
    save_model(checkpoint, model, config=cfg)
    eeg = np.random.default_rng(43).normal(size=(8, 2, 8)).astype(np.float32)
    np.savez_compressed(inputs, eeg=eeg)
    output.write_bytes(b"previous predictions")
    return cfg, model, checkpoint, inputs, output, eeg


@pytest.mark.parametrize("phase", ["read", "predict", "publish"])
def test_prediction_refuses_replaced_npz_and_preserves_output(prediction_files, monkeypatch, phase):
    _, _, checkpoint, inputs, output, eeg = prediction_files
    changed = False
    def mutate():
        nonlocal changed
        if not changed:
            changed = True
            replace_npz(inputs, {"eeg": eeg + 3})
    if phase == "read":
        with np.load(inputs) as data:
            cls = type(data)
        read = cls.__getitem__
        def patched(self, key):
            value = read(self, key)
            if key == "eeg":
                mutate()
            return value
        monkeypatch.setattr(cls, "__getitem__", patched)
    elif phase == "predict":
        predict = evaluate.iter_predictions
        def patched(*a, **kw):
            mutate()
            yield from predict(*a, **kw)
        monkeypatch.setattr(evaluate, "iter_predictions", patched)
    else:
        save = np.savez_compressed
        def patched(path, **values):
            result = save(path, **values)
            if isinstance(path, Path) and path.name.startswith(".prediction-"):
                mutate()
            return result
        monkeypatch.setattr(np, "savez_compressed", patched)
    with pytest.raises(ValueError, match="changed since loading"):
        evaluate.main([str(checkpoint), "--input", str(inputs), "--output", str(output)])
    assert output.read_bytes() == b"previous predictions"
    assert not list(output.parent.glob(".prediction-*"))


def test_prediction_checks_the_open_file_even_if_original_path_is_restored(prediction_files, monkeypatch):
    _, _, checkpoint, inputs, output, eeg = prediction_files
    original_open = Path.open
    opens = 0
    def swap_while_opening(path, *a, **kw):
        nonlocal opens
        if path == inputs and (a[0] if a else kw.get("mode")) == "rb":
            opens += 1
            if opens == 2:  # after initial hashing, before opening the NPZ reader
                saved = path.with_suffix(".original")
                path.rename(saved)
                replace_npz(path, {"eeg": eeg + 3})
                stream = original_open(path, *a, **kw)
                saved.replace(path)
                return stream
        return original_open(path, *a, **kw)
    # np.load opens paths via builtins.open. Supplying an already-open file also
    # makes the producer/consumer handoff explicit for this regression.
    load = np.load
    def patched_load(path, *a, **kw):
        if path == inputs:
            return load(swap_while_opening(inputs, "rb"), *a, **kw)
        return load(path, *a, **kw)
    monkeypatch.setattr(Path, "open", swap_while_opening)
    monkeypatch.setattr(np, "load", patched_load)
    with pytest.raises(ValueError, match="changed|input version"):
        evaluate.main([str(checkpoint), "--input", str(inputs), "--output", str(output)])
    assert output.read_bytes() == b"previous predictions"


@pytest.mark.parametrize("alias", ["attention_time", "train_time"])
@pytest.mark.parametrize("verified_axis", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_predict_restores_saved_window_without_cropping_eeg(prediction_files, alias, verified_axis, dtype):
    cfg, model, checkpoint, inputs, output, eeg = prediction_files
    times = (np.arange(8) / 10).astype(dtype)
    cfg["train"].update({alias + "_start_s": .2, alias + "_end_s": .4})
    if verified_axis:
        cfg["model"]["resolved_time_axis_sha256"] = hashlib.sha256(times.astype(np.float64).tobytes()).hexdigest()
    save_model(checkpoint, model, config=cfg)
    np.savez_compressed(inputs, eeg=eeg, time_s=times)
    evaluate.main([str(checkpoint), "--input", str(inputs), "--output", str(output), "--batch-size", "8"])
    # Match training's exact comparisons after promoting the stored float32 axis.
    indices = torch.from_numpy(np.flatnonzero((times.astype(np.float64) >= .2) & (times.astype(np.float64) <= .4)))
    with torch.inference_mode():
        expected = model(torch.from_numpy(eeg), "choice", attention_indices=indices)["pooled_logits"].numpy()
    with np.load(output) as data:
        np.testing.assert_array_equal(data["pooled_logits"], expected)
        metadata = json.loads(str(data["metadata_json"]))
        assert metadata["prediction"]["attention_indices"] == indices.tolist()
        assert metadata["input_sha256"] == file_sha256(inputs)


def test_time_axis_binding_survives_save_load_and_rejects_conflicts(prediction_files):
    cfg, model, checkpoint, _, _, _ = prediction_files
    key = "resolved_time_axis_sha256"
    cfg["model"][key] = "a" * 64
    save_model(checkpoint, model, config=cfg)
    loaded = load_model(checkpoint)
    assert loaded.config["model"][key] == loaded.spec.model_config[key] == "a" * 64
    wrong = deepcopy(loaded.config)
    wrong["model"][key] = "b" * 64
    before = checkpoint.read_bytes()
    with pytest.raises(ValueError, match="time axis"):
        save_model(checkpoint, loaded.model, config=wrong)
    with pytest.raises(ValueError, match="time axis"):
        load_model(checkpoint, config=wrong)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("axis", [None, "wrong_hash", "wrong_shape", "nan", "unordered", "empty_support"])
def test_predict_rejects_unverifiable_attention_support(prediction_files, axis):
    cfg, model, checkpoint, inputs, output, eeg = prediction_files
    cfg["train"].update(attention_time_start_s=.2, attention_time_end_s=.4)
    times = np.arange(8, dtype=np.float64) / 10
    if axis == "wrong_hash":
        cfg["model"]["resolved_time_axis_sha256"] = "0" * 64
    save_model(checkpoint, model, config=cfg)
    if axis == "wrong_shape": times = times[:-1]
    if axis == "nan": times[0] = np.nan
    if axis == "unordered": times = times[::-1]
    if axis == "empty_support": times = times + 10
    np.savez_compressed(inputs, eeg=eeg, **({} if axis is None else {"time_s": times}))
    with pytest.raises(ValueError, match="time_s|time axis"):
        evaluate.main([str(checkpoint), "--input", str(inputs), "--output", str(output)])
    assert output.read_bytes() == b"previous predictions"


@pytest.mark.parametrize("legacy_implicit", [False, True])
def test_resuming_extended_training_keeps_original_cosine_duration(tmp_path, legacy_implicit):
    cfg = config()
    cfg["train"] = dict(iterations=4, learning_rate=.01, min_learning_rate=.001,
        lr_schedule="warmup_cosine", validation_interval_iterations=1, log_interval_iterations=1,
        patience_iterations=100, amp=False, progress_bar=False, loss_type="ce",
        checkpoint_metric="val_pooled_accuracy", checkpoint_mode="max")
    model = build_model(cfg["model"])
    batch = {"eeg": torch.randn(1, 2, 8).repeat(4, 1, 1), "target": torch.tensor([0, 1, 0, 1]),
             "subject_id": torch.zeros(4, dtype=torch.long), "current_stimulus": torch.ones(4, dtype=torch.long)}
    def fit(settings, resume=None):
        return fit_target_decoder(model, [batch], [batch], "choice", torch.arange(8), settings,
            tmp_path, torch.device("cpu"), resume_from=resume)
    fit(cfg)
    checkpoint = tmp_path / "last.ckpt"
    saved = torch.load(checkpoint, weights_only=True)
    assert saved["config"]["train"]["lr_schedule_iterations"] == 4
    assert "lr_schedule_iterations" not in cfg["train"]  # caller request is unchanged
    if legacy_implicit:
        for path in (checkpoint, tmp_path / "best_val.ckpt"):
            payload = torch.load(path, weights_only=True)
            payload["config"]["train"].pop("lr_schedule_iterations")
            torch.save(payload, path)
    extended = deepcopy(cfg)
    extended["train"]["iterations"] = 8
    rows = fit(extended, checkpoint)
    assert [r["iteration"] for r in rows] == [5, 6, 7, 8]
    assert all(r["learning_rate"] == .001 for r in rows)
    assert torch.load(checkpoint, weights_only=True)["config"]["train"]["lr_schedule_iterations"] == 4
    changed = deepcopy(extended)
    changed["train"].update(iterations=12, lr_schedule_iterations=12)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    with pytest.raises(ValueError, match="lr_schedule_iterations"):
        fit(changed, checkpoint)
    assert before == {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}


def test_unresolved_resume_cannot_silently_change_schedule():
    old = config()
    old["train"].update(iterations=100, lr_schedule="warmup_cosine")
    new = deepcopy(old)
    new["train"]["iterations"] = 200
    with pytest.raises(ValueError, match="lr_schedule_iterations"):
        validate_resume_config(old, new)
    new["train"]["lr_schedule_iterations"] = 100
    validate_resume_config(old, new)


def test_relative_electrode_path_resolves_from_yaml(tmp_path, monkeypatch):
    folder = tmp_path / "config"
    folder.mkdir()
    electrodes = folder / "electrodes.npz"
    np.savez(electrodes, channel_names=np.array(["a", "b", "c"]),
             positions_m=np.eye(3), sphere_origin_m=np.zeros(3), sphere_radius_m=np.array(1.),
             coord_frame=np.array("head"), channel_index=np.arange(3), template=np.array("probe"))
    cfg = config()
    cfg["preprocessing"] = {"graph_laplacian": {"enabled": True, "electrodes_npz": "electrodes.npz"}}
    source = folder / "train.yaml"
    source.write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)
    parsed = parse_request(source).config["preprocessing"]["graph_laplacian"]["electrodes_npz"]
    assert parsed == str(electrodes)
    build_graph(Path(parsed), 2)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_nesterov_requires_bool_in_yaml_and_optimizer_api(tmp_path, value):
    cfg = config()
    cfg["train"].update(optimizer="sgd", learning_rate=.01, nesterov=value)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="nesterov must be a boolean"):
        parse_request(path)
    with pytest.raises(ValueError, match="nesterov must be a boolean"):
        build_optimizer(torch.nn.Linear(2, 1).parameters(), cfg["train"])


@pytest.mark.parametrize("value", [False, True])
def test_real_nesterov_boolean_preserves_optimizer_behavior(tmp_path, value):
    cfg = config()
    cfg["train"].update(optimizer="sgd", learning_rate=.01, momentum=.9, nesterov=value)
    path = tmp_path / "train.yaml"
    path.write_text(yaml.safe_dump(cfg))
    optimizer = build_optimizer(torch.nn.Linear(2, 1).parameters(), parse_request(path).config["train"])
    assert optimizer.param_groups[0]["nesterov"] is value
