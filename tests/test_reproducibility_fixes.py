"""Regression cases from the 2026-11-14 reproducibility audit."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

from memory_reactivation.artifacts.checkpoints import load_model, save_model
from memory_reactivation.artifacts.exports import copy_verified_exports, describe_export
from memory_reactivation.artifacts.provenance import data_identity, digest
from memory_reactivation.config import parse_request
from memory_reactivation.contracts import PredictionSpec
from memory_reactivation.data.dataset import GlobalBatchSampler, LagDecodingDataset
from memory_reactivation.evaluation.export import export_target_outputs
from memory_reactivation.evaluation.predict import iter_predictions
from memory_reactivation.models.baselines.spatial import SpatialTrialDecoder
from memory_reactivation.models.factory import build_model
from memory_reactivation.workflows import evaluate, report
from memory_reactivation.workflows import train as train_workflow
from regression.helpers_smoke import TargetPipelineSmokeTest


def config(channels=2, times=8):
    return {"task": "choice", "data": {}, "train": {}, "model": {
        "encoder": {"type": "cnn", "in_channels": channels, "latent_dim": 4, "num_blocks": 1,
                    "kernel_size": 3, "dropout": 0., "expansion_ratio": 1},
        "attention_mode": "gated_scorer", "classifier_num_layers": 1,
        "resolved_time_point_count": times}}


def test_source_drift_rejects_checkpoint_without_relabelling_or_replacing_it(tmp_path):
    source = Path(__file__).resolve().parents[1] / "memory_reactivation"
    shutil.copytree(source, tmp_path / "memory_reactivation", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    code = r'''
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
import torch
import memory_reactivation.training.step as step
from memory_reactivation.training.common import save_checkpoint
from memory_reactivation.artifacts.provenance import implementation_identity, runtime_implementation_identity
root = Path(sys.argv[1])
assert Path(step.__file__).is_relative_to(root)
model = torch.nn.Linear(2, 2)
optimizer = torch.optim.Adam(model.parameters())
file = root / 'last.ckpt'
save_checkpoint(file, model, optimizer, 0, {}, {})
original = file.read_bytes()
original_code = step.target_step.__code__
original_hash = torch.load(file, weights_only=True)['implementation_sha256']
path = Path(step.__file__)
text = path.read_text()
assert text.count('    return StepResult(loss, {') == 1
path.write_text(text.replace('    return StepResult(loss, {', '    loss = loss * 2\n    return StepResult(loss, {'))
assert step.target_step.__code__ is original_code
assert original_hash != implementation_identity()['sha256']
try:
    save_checkpoint(file, model, optimizer, 1, {}, {})
except RuntimeError as error:
    assert 'sources changed' in str(error)
else:
    raise AssertionError('source drift was accepted')
assert file.read_bytes() == original
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=tmp_path, check=True,
                   capture_output=True, text=True)


@pytest.mark.parametrize("verified", [False, True])
def test_resume_requires_import_time_identity_even_when_the_disk_hash_matches(tmp_path, monkeypatch, verified):
    from memory_reactivation.artifacts.provenance import runtime_implementation_identity
    identity = runtime_implementation_identity()
    payload = {"implementation_sha256": identity["sha256"]}
    if verified:
        payload["implementation_identity_schema"] = identity["schema"]
    checkpoint = tmp_path / "last.ckpt"
    torch.save(payload, checkpoint)
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump(config()))
    class PassedResumeIdentity(Exception):
        pass
    def reached_data_setup(*args, **kwargs):
        raise PassedResumeIdentity()
    monkeypatch.setattr(train_workflow, "configure_cross_validation", reached_data_setup)
    with pytest.raises(PassedResumeIdentity if verified else ValueError):
        train_workflow.main(["--config", str(cfg), "--task", "choice", "--resume-from", str(checkpoint)])


def test_transformer_save_load_uses_bound_spec_and_preserves_rng(tmp_path):
    cfg = config()
    del cfg["model"]["resolved_time_point_count"]
    cfg["model"]["encoder"] = {"type": "window_transformer", "in_channels": "auto", "latent_dim": 4,
        "num_layers": 1, "window_size": 3, "num_heads": 2, "ffn_ratio": 1, "dropout": 0.}
    model = build_model(cfg["model"], time_point_count=8, input_channels=2).eval()
    before = torch.get_rng_state().clone()
    file = tmp_path / "model.ckpt"
    save_model(file, model, config=cfg)
    loaded = load_model(file, config=cfg)
    assert torch.equal(torch.get_rng_state(), before)
    assert loaded.spec.time_point_count == 8
    assert loaded.spec.input_channels == 2
    eeg = torch.randn(2, 2, 8)
    for name, expected in model(eeg, "choice").items():
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(loaded.model(eeg, "choice")[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("boundary", ["save", "load"])
def test_conflicting_config_and_model_spec_are_rejected(tmp_path, boundary):
    cfg = config()
    model = build_model({**cfg["model"], "attention_gate_activation": "tanh_plus_one"})
    path = tmp_path / "conflicting.ckpt"
    if boundary == "save":
        path.write_bytes(b"previous file")
        with pytest.raises(ValueError, match="conflicts"):
            save_model(path, model, config=cfg)
        assert path.read_bytes() == b"previous file"
    else:
        torch.save({"config": cfg, "model_spec": model._pb_model_spec, "model_state": model.state_dict()}, path)
        with pytest.raises(ValueError, match="conflicts"):
            load_model(path)


@pytest.mark.parametrize("field,value", [("input_channels", 7), ("time_point_count", 9),
    ("embedding_capacity", 3), ("schema", "invalid"), ("time_point_count", True)])
def test_inconsistent_or_invalid_saved_model_specs_fail(tmp_path, field, value):
    cfg = config()
    model = build_model(cfg["model"])
    spec = deepcopy(model._pb_model_spec)
    spec[field] = value
    file = tmp_path / "invalid.ckpt"
    torch.save({"config": cfg, "model_state": model.state_dict(), "model_spec": spec}, file)
    with pytest.raises(ValueError):
        load_model(file)


@pytest.fixture
def native_dataset():
    fixture = TargetPipelineSmokeTest()
    fixture.setUp()
    try:
        yield LagDecodingDataset(fixture.records, fixture.splits, fixture.normalization,
            split="val", task="choice", lags=[1], balance_cells=False)
    finally:
        fixture.tearDown()


def test_different_sampler_subsets_cannot_reuse_each_others_exports(tmp_path, native_dataset):
    dataset = native_dataset
    model = build_model(config(64, 40)["model"]).eval()
    left = DataLoader(dataset, batch_size=8, sampler=list(range(8)))
    right = DataLoader(dataset, batch_size=8, sampler=list(range(8, 16)))
    arguments = dict(model=model, loader=left, task="choice", device=torch.device("cpu"), amp=False,
        output_dir=tmp_path / "left", split="val", attention_indices=torch.arange(40),
        time_s=dataset.time_s, progress_bar_enabled=False)
    export_target_outputs(**arguments)
    expected = describe_export(export_target_outputs, **{**arguments, "loader": right})
    destination = tmp_path / "destination"
    destination.mkdir()
    (destination / "keep.txt").write_text("preserve")
    with pytest.raises(ValueError, match="data_sha256"):
        copy_verified_exports(arguments["output_dir"], destination, decoder="target", splits=["val"],
            input_kinds=["real"], expected_parameters={("val", "real"): expected})
    assert list(destination.iterdir()) == [destination / "keep.txt"]


def test_sampler_identity_covers_order_batching_and_native_epoch_without_rng_consumption(native_dataset):
    dataset = native_dataset
    loaders = [DataLoader(dataset, batch_size=bs, sampler=indices) for bs, indices in
               [(8, list(range(8))), (8, list(reversed(range(8)))), (4, list(range(8)))]]
    identities = [digest(data_identity(loader)) for loader in loaders]
    assert len(set(identities)) == 3
    sampler = GlobalBatchSampler(dataset, batch_size=8, shuffle=True, seed=7)
    loader = DataLoader(dataset, batch_sampler=sampler)
    rng = torch.get_rng_state().clone()
    first = digest(data_identity(loader))
    assert first == digest(data_identity(loader))
    sampler.set_epoch(1)
    assert digest(data_identity(loader)) != first
    assert torch.equal(rng, torch.get_rng_state())


def test_dataset_explicit_identity_does_not_hide_sampler_differences():
    dataset = TensorDataset(torch.arange(8))
    dataset.export_identity = {"dataset": "fixed-fixture"}
    a = DataLoader(dataset, sampler=[0, 1])
    b = DataLoader(dataset, sampler=[2, 3])
    assert data_identity(a) != data_identity(b)


def test_unverifiable_random_sampler_is_rejected_without_advancing_rng(native_dataset):
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="Unverifiable export sampler"):
        data_identity(DataLoader(native_dataset, batch_size=8, shuffle=True))
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.fixture
def baseline_checkpoint(tmp_path):
    options = dict(spatial_filters=3, dropout=0., activation="elu", spatial_max_norm=1., classifier_max_norm=.25)
    model = SpatialTrialDecoder(2, 8, 2, options)
    cfg = {"task": "choice", "model": {"encoder": options}}
    native = {"config": cfg, "fold": 0}
    (tmp_path / "identity.json").write_text(json.dumps({"identity": native, "identity_sha256": digest(native)}))
    (tmp_path / "resolved_model.json").write_text(json.dumps({"type": "spatial_only", **model.encoder.resolved_config()}))
    file = tmp_path / "baseline.ckpt"
    torch.save({"schema": "baseline.checkpoint.v1", "identity_sha256": digest(native), "cv_fold": 0,
                "model_state": model.state_dict()}, file)
    return file


def test_baseline_task_binding_covers_library_and_cli(baseline_checkpoint, tmp_path):
    loaded = load_model(baseline_checkpoint)
    batch = {"eeg": torch.zeros(2, 2, 8)}
    assert next(iter_predictions(loaded.model, [batch], PredictionSpec("choice"))).outputs["pooled_logits"].shape == (2, 2)
    with pytest.raises(ValueError, match="trained task"):
        next(iter_predictions(loaded.model, [batch], PredictionSpec("pitch")))
    inputs, output = tmp_path / "input.npz", tmp_path / "output.npz"
    np.savez(inputs, eeg=batch["eeg"].numpy())
    with pytest.raises(ValueError, match="trained task"):
        evaluate.main([str(baseline_checkpoint), "--input", str(inputs), "--output", str(output), "--task", "pitch"])
    assert not output.exists()


def test_baseline_prediction_accepts_and_validates_shared_time_axis(baseline_checkpoint, tmp_path):
    eeg = np.random.default_rng(7).normal(size=(3, 2, 8)).astype(np.float32)
    inputs = tmp_path / "input.npz"
    outputs = []
    for with_axis in (False, True):
        output = tmp_path / f"output-{with_axis}.npz"
        axis = {"time_s": np.arange(8, dtype=np.float64) / 100} if with_axis else {}
        np.savez(inputs, eeg=eeg, **axis)
        evaluate.main([str(baseline_checkpoint), "--input", str(inputs), "--output", str(output),
                       "--batch-size", "2"])
        with np.load(output) as saved:
            outputs.append(saved["pooled_logits"].copy())
    np.testing.assert_array_equal(*outputs)
    before = output.read_bytes()
    np.savez(inputs, eeg=eeg, time_s=np.zeros(8))
    with pytest.raises(ValueError, match="increasing"):
        evaluate.main([str(baseline_checkpoint), "--input", str(inputs), "--output", str(output)])
    assert output.read_bytes() == before


def test_report_missing_run_does_not_create_empty_success_directory(tmp_path):
    run = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="run directory"):
        report.main(["--run-dir", str(run), "--splits", "val"])
    assert not run.exists()


@pytest.mark.parametrize("damage", ["missing", "empty", "gaussian", "later_split", "activation"])
def test_report_checks_all_required_inputs_before_writing_figures(tmp_path, damage):
    for stem in ("attention_weights", "trial_metrics", "time_accuracy"):
        (tmp_path / f"val_{stem}.csv").write_text("example\n1\n")
    plots, splits = {}, ["val"]
    if damage == "missing": (tmp_path / "val_trial_metrics.csv").unlink()
    if damage == "empty": (tmp_path / "val_trial_metrics.csv").write_text("example\n")
    if damage == "gaussian": plots = {"gaussian_baseline": {"enabled": True}}
    if damage == "later_split": splits.append("test")
    if damage == "activation": plots = {"activation": {"enabled": True}}
    with pytest.raises((ValueError, FileNotFoundError)):
        report.plot_target_run(tmp_path, splits=splits, plot_config=plots)
    assert not (tmp_path / "figures").exists()


@pytest.mark.parametrize("override,field", [
    ({"train": {"attention_mass_penality_weight": .1}}, "train.attention_mass_penality_weight"),
    ({"model": {"attention_gate_activaton": "tanh_plus_one"}}, "model.attention_gate_activaton"),
    ({"data": {"balance_train_cell": True}}, "data.balance_train_cell"),
    ({"train": {"near_full_noise": {"probablity": .1}}}, "train.near_full_noise.probablity"),
    ({"plots": {"gaussian_baseline": {"enable": True}}}, "plots.gaussian_baseline.enable"),
    ({"plots": {"event_markers": {"markers": [{"time_s": 0., "lable": "onset"}]}}}, "plots.event_markers.markers[].lable"),
])
def test_nested_unknown_fields_fail_including_overrides(tmp_path, override, field):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config()))
    with pytest.raises(ValueError) as error:
        parse_request(path, override)
    assert field in str(error.value)


def test_npz_prediction_loads_each_input_once_and_keeps_sample_order(tmp_path, monkeypatch):
    cfg = config()
    model = build_model(cfg["model"]).eval()
    checkpoint = tmp_path / "model.ckpt"
    save_model(checkpoint, model, config=cfg)
    eeg = np.arange(160, dtype=np.float32).reshape(10, 2, 8) / 100.
    inputs, output = tmp_path / "inputs.npz", tmp_path / "predictions.npz"
    np.savez_compressed(inputs, eeg=eeg, trial0=np.arange(10))
    with np.load(inputs) as archive:
        cls = type(archive)
    original = cls.__getitem__
    counts = {}
    def read(self, key):
        counts[key] = counts.get(key, 0) + 1
        return original(self, key)
    with monkeypatch.context() as patch:
        patch.setattr(cls, "__getitem__", read)
        evaluate.main([str(checkpoint), "--input", str(inputs), "--output", str(output), "--batch-size", "2"])
    assert counts == {"eeg": 1, "trial0": 1}
    with np.load(output) as actual:
        np.testing.assert_array_equal(actual["trial0"], np.arange(10))
        expected = model(torch.from_numpy(eeg), "choice")["pooled_logits"].detach().numpy()
        np.testing.assert_allclose(actual["pooled_logits"], expected, rtol=1e-6, atol=1e-6)
