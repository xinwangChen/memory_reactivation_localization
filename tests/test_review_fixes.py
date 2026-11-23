"""Regressions for checkpoint rewinds, export identity and ignored configuration."""

from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from memory_reactivation.artifacts.exports import _model_sha256
from memory_reactivation.config import parse_request
from memory_reactivation.data.configured import build_lag_dataset
from memory_reactivation.data.dataset import normalize_lags
from memory_reactivation.training.common import load_checkpoint, recover_best_checkpoint
from memory_reactivation.workflows import reexport, train
from regression.helpers_smoke import TargetPipelineSmokeTest


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    base = tmp_path_factory.mktemp("review-fixes")
    fixture = TargetPipelineSmokeTest()
    fixture.setUp()
    config = dict(
        seed=42,
        data=dict(manifest_csv=str(fixture.manifest), split_csv=str(fixture.splits),
                  normalization_json=str(fixture.normalization), experiments=[1], lags=[1],
                  balance_train_cells=False, balance_eval_cells=False, num_workers=0,
                  offline_cache_dir=str(base / "cache")),
        model=dict(
            encoder=dict(type="window_transformer", in_channels="auto", latent_dim=8,
                         num_layers=1, window_size=3, num_heads=2, ffn_ratio=2, dropout=0.,
                         attention_pattern="centered_local", position_encoding_mode="sinusoidal_qk",
                         boundary_mode="noise_pad", boundary_noise_seed=42),
            attention_mode="gated_scorer", attention_gate_activation="tanh_plus_one",
            classifier_num_layers=1, classifier_dropout=0., time_no_class_enabled=True),
        train=dict(iterations=3, validation_interval_iterations=1, log_interval_iterations=1,
                   batch_size=16, learning_rate=.001, lr_schedule="constant", amp=False,
                   progress_bar=False, checkpoint_metric="val_loss", checkpoint_mode="min",
                   loss_type="ce", time_loss_weight=.5, pooled_loss_weight=1.,
                   train_time_start_s=-.1, train_time_end_s=.29,
                   attention_time_start_s=-.1, attention_time_end_s=.29,
                   trajectory_checkpoint_iterations=[1]),
        plots=dict(enabled=False, splits=["val"], gaussian_baseline=dict(enabled=False)),
        swanlab=dict(enabled=False),
        run=dict(base_output_dir=str(base / "runs"), base_experiment_name="review"))
    filename = base / "config.yaml"
    filename.write_text(yaml.safe_dump(config))
    try:
        with patch.dict(os.environ, PB2023_SKIP_CODE_SNAPSHOT="1", PB2023_SWANLAB_ENABLED="0"):
            train.main(["--config", str(filename), "--task", "choice"])
            run = next((base / "runs").rglob("best_val.ckpt")).parent
            assert load_checkpoint(run / "best_val.ckpt")["iteration"] == 3
            yield run, config
    finally:
        fixture.tearDown()


def resume_run(trained_run, tmp_path, iterations):
    run, config = trained_run
    clone = tmp_path / "run"
    shutil.copytree(run, clone)
    settings = deepcopy(config)
    settings["train"]["iterations"] = iterations
    filename = tmp_path / "config.yaml"
    filename.write_text(yaml.safe_dump(settings))
    args = ["--config", str(filename), "--task", "choice",
            "--resume-from", str(clone / "trajectory_iter000001.ckpt")]
    return clone, args


@pytest.mark.parametrize("iterations", [1, 2])
def test_rewind_publishes_only_weights_from_the_resumed_trajectory(trained_run, tmp_path, iterations):
    clone, args = resume_run(trained_run, tmp_path, iterations)
    original_rows = csv_rows(clone / "metrics.csv")
    train.main(args)
    best, last = [load_checkpoint(clone / name) for name in ("best_val.ckpt", "last.ckpt")]
    assert best["iteration"] <= iterations
    assert last["iteration"] == iterations
    rows = csv_rows(clone / "metrics.csv")
    assert [int(float(row["iteration"])) for row in rows] == list(range(1, iterations + 1))
    assert [row["val_loss"] for row in rows] == [row["val_loss"] for row in original_rows[:iterations]]
    assert best["metrics"]["val_loss"] == min(float(row["val_loss"]) for row in rows)
    manifest = json.loads((clone / "val_export_manifest.json").read_text())
    from memory_reactivation.artifacts.checkpoints import load_model
    expected_model = load_model(clone / "best_val.ckpt").model
    assert manifest["parameters"]["model_state_sha256"] == _model_sha256(expected_model)


def test_rejected_resume_preserves_existing_logs_and_checkpoints(trained_run, tmp_path):
    clone, args = resume_run(trained_run, tmp_path, iterations=3)
    best = load_checkpoint(clone / "best_val.ckpt")
    best["config"]["seed"] = 999
    torch.save(best, clone / "best_val.ckpt")
    names = ("best_val.ckpt", "last.ckpt", "metrics.csv", "train_metrics.csv")
    before = {name: (clone / name).read_bytes() for name in names}
    with pytest.raises(ValueError, match="config.seed"):
        train.main(args)
    assert {name: (clone / name).read_bytes() for name in names} == before


def test_resume_rejects_budget_before_selected_checkpoint(trained_run, tmp_path):
    clone, args = resume_run(trained_run, tmp_path, iterations=2)
    args[-1] = str(clone / "last.ckpt")
    names = ("best_val.ckpt", "last.ckpt", "metrics.csv", "train_metrics.csv")
    before = {name: (clone / name).read_bytes() for name in names}
    with pytest.raises(ValueError, match="exceeds train.iterations"):
        train.main(args)
    assert {name: (clone / name).read_bytes() for name in names} == before


def test_explicit_cli_lag_still_overrides_an_empty_yaml_default(trained_run, tmp_path):
    clone, args = resume_run(trained_run, tmp_path, iterations=1)
    filename = tmp_path / "config.yaml"
    config = yaml.safe_load(filename.read_text())
    config["data"]["lags"] = []
    filename.write_text(yaml.safe_dump(config))
    train.main([*args, "--lag", "1"])
    last = load_checkpoint(clone / "last.ckpt")
    assert last["iteration"] == 1
    assert last["config"]["data"]["lags"] == [1]


@pytest.mark.parametrize("change", ["task", "lag", "noise_seed", "gate", "attention_window"])
def test_reexport_rejects_snapshot_drift_before_touching_exports(trained_run, tmp_path, change):
    run, _ = trained_run
    clone = tmp_path / "run"
    shutil.copytree(run, clone)
    filename = clone / "config_snapshot.yaml"
    config = yaml.safe_load(filename.read_text())
    if change == "task":
        config["task"] = "pitch"
    elif change == "lag":
        config["lag"] = 0
        config["data"]["lags"] = [0]
    elif change == "noise_seed":
        for key in ("encoder", "resolved_encoder"):
            config["model"][key]["boundary_noise_seed"] = 43
    elif change == "gate":
        config["model"]["attention_gate_activation"] = "sigmoid"
    else:
        config["train"]["attention_time_start_s"] = 0.
    filename.write_text(yaml.safe_dump(config))
    before = {p.name: p.read_bytes() for p in clone.glob("val_*")}
    with pytest.raises(ValueError, match="differs|conflicts"):
        reexport.main(["--run-dir", str(clone), "--splits", "val"])
    assert {p.name: p.read_bytes() for p in clone.glob("val_*")} == before


def test_reexport_checks_model_spec_even_if_embedded_and_adjacent_configs_agree(trained_run, tmp_path):
    run, _ = trained_run
    clone = tmp_path / "run"
    shutil.copytree(run, clone)
    filename = clone / "config_snapshot.yaml"
    config = yaml.safe_load(filename.read_text())
    best = load_checkpoint(clone / "best_val.ckpt")
    for mapping in (config, best["config"]):
        for key in ("encoder", "resolved_encoder"):
            mapping["model"][key]["boundary_noise_seed"] = 43
    filename.write_text(yaml.safe_dump(config))
    torch.save(best, clone / "best_val.ckpt")
    with pytest.raises(ValueError, match="forward semantics"):
        reexport.main(["--run-dir", str(clone), "--splits", "val"])


@pytest.mark.parametrize("value", [1.5, -.5, True, False, "1", 1., None])
def test_invalid_lags_are_rejected_before_building_data(tmp_path, value):
    config = dict(data={"lags": [value]}, model={}, train={})
    filename = tmp_path / "config.yaml"
    filename.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="lags"):
        parse_request(filename)
    with pytest.raises(ValueError, match="lags"):
        train.resolve_single_lag(config["data"], None)
    with pytest.raises(ValueError, match="lags"):
        train.resolve_single_lag({}, value)
    with pytest.raises(ValueError, match="lags"):
        build_lag_dataset(config["data"], task="choice", split="val", seed=42)


def test_lag_normalization_preserves_integer_inputs_and_existing_multiple_lag_support():
    assert normalize_lags([np.int64(2), 1, 2]) == [1, 2]
    assert train.resolve_single_lag({"lags": [2, 2]}, None) == 2
    assert train.resolve_single_lag({"lags": [2]}, 0) == 0


@pytest.mark.parametrize("options", [[12, 18], [], [12.5], [True], "12"])
def test_target_rejects_unsupported_window_options_at_configuration_entry(tmp_path, options):
    config = dict(data={"dynamic_time_window_sample_options": options}, model={}, train={})
    filename = tmp_path / "config.yaml"
    filename.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="dynamic_time_window_sample_options"):
        parse_request(filename)
    with pytest.raises(ValueError, match="dynamic_time_window_sample_options"):
        build_lag_dataset(config["data"], task="choice", split="train", seed=42)


def test_single_window_option_is_applied_to_training_validation_and_reexport(trained_run, tmp_path, monkeypatch):
    _, config = trained_run
    settings = deepcopy(config)
    settings["data"]["dynamic_time_window_sample_options"] = [12]
    settings["run"]["base_output_dir"] = str(tmp_path / "window-runs")
    settings["train"].update(iterations=1)
    filename = tmp_path / "window.yaml"
    filename.write_text(yaml.safe_dump(settings))
    for split in ("train", "val"):
        dataset = build_lag_dataset(settings["data"], task="choice", split=split, seed=42)
        assert dataset.store.dynamic_time_window_sample_options == (12,)
        dataset.set_epoch(5)
        assert dataset[0]["eeg"].shape[-1] == 12
    train.main(["--config", str(filename), "--task", "choice"])
    run = next((tmp_path / "window-runs").rglob("best_val.ckpt")).parent
    saved = yaml.safe_load((run / "config_snapshot.yaml").read_text())
    assert saved["model"]["resolved_time_point_count"] == 12
    monkeypatch.setattr(reexport, "plot_target_conditions", lambda *a, **k: None)
    reexport.main(["--run-dir", str(run), "--splits", "val", "--input-kinds", "real",
                   "--reuse-existing-kinds", "real", "--controls-output-dir", str(tmp_path / "reuse")])
    assert (tmp_path / "reuse/exports/val_trial_metrics.csv").read_bytes() == (run / "val_trial_metrics.csv").read_bytes()


@pytest.mark.parametrize("mode, candidate, best, future", [
    ("max", .6, .8, .9), ("max", .6, .8, .8),
    ("min", .8, .6, .5), ("min", .8, .6, .6)])
def test_unavailable_historical_best_is_rejected_without_replacing_future_weights(tmp_path, mode, candidate, best, future):
    resume = tmp_path / "trajectory.ckpt"
    current_best = tmp_path / "best_val.ckpt"
    payload = dict(iteration=3, config={"train": {}}, metrics={"score": candidate},
                   extra_state={"best_score": best})
    torch.save(payload, resume)
    torch.save({**payload, "iteration": 8, "metrics": {"score": future}}, current_best)
    before = current_best.read_bytes()
    with pytest.raises(ValueError, match="newer than the resume step"):
        recover_best_checkpoint(resume, tmp_path, metric="score", mode=mode)
    assert current_best.read_bytes() == before
    torch.save({**payload, "iteration": 2, "metrics": {"score": best}}, current_best)
    before = current_best.read_bytes()
    assert not recover_best_checkpoint(resume, tmp_path, metric="score", mode=mode)
    assert current_best.read_bytes() == before


def test_rewind_before_selection_warmup_removes_ineligible_best(tmp_path):
    resume = tmp_path / "trajectory.ckpt"
    payload = dict(iteration=1, config={"train": {}}, metrics={"score": .5},
                   extra_state={"best_score": -float("inf")})
    torch.save(payload, resume)
    torch.save({**payload, "iteration": 3}, tmp_path / "best_val.ckpt")
    assert recover_best_checkpoint(resume, tmp_path, metric="score", mode="max")
    assert not (tmp_path / "best_val.ckpt").exists()
    assert load_checkpoint(resume)["iteration"] == 1
