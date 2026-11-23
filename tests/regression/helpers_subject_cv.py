"""Subject-CV population, normalization leakage and full CLI regression checks."""
from __future__ import annotations

import copy
import importlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from memory_reactivation.data.configured import (
    build_lag_dataset, configure_cross_validation, validate_data_protocol,
    validate_subject_cv_datasets,
)
from memory_reactivation.data.manifest import (
    SubjectRecord, load_split_map, read_csv_rows, subject_split_membership,
    write_block_cv_splits, write_csv_rows, write_subject_cv_splits,
)
from memory_reactivation.data.normalization import (
    compute_normalization_stats, load_normalization_stats, validate_normalization_identity,
)
from memory_reactivation.data.spatial import GraphLaplacian, configure_spatial_preprocessing
from memory_reactivation.data.subject_conditioning import configure_training_subject_conditioning
from memory_reactivation.models.subject_conditioning import SharedResidualSubjectEmbedding
from memory_reactivation.training.common import validate_checkpoint_cv_identity
from memory_reactivation.utils.config import load_yaml, save_yaml


def _record(root, index, *, channels=2, trials=20):
    subject = f"Sub{index:02d}"
    npz, meta = root / f"{subject}.npz", root / f"{subject}_meta.csv"
    trial = np.arange(trials)
    x = np.random.default_rng(index).normal(size=(trials, channels, 9)).astype(np.float32) + index * 7
    np.savez(npz, X=x, trial0=trial, y_pitch=trial % 5 + 1, y_choice=trial % 2,
             y_motor=trial % 2 + 1, valid_for_history=np.ones(trials),
             time_s=np.arange(9) / 100, fsample=[100])
    write_csv_rows(meta, [{"trial0": int(t), "block": int(t // 10), "valid_eeg": 1,
                           "pitch": int(t % 5 + 1), "choice": int(t % 2),
                           "motor": int(t % 2 + 1)} for t in trial])
    return SubjectRecord(1, subject, f"Exp1_{subject}", str(npz), str(meta), trials)


def _bundle(root, *, channels=2):
    records = [_record(root, i, channels=channels, trials=10 + i * 2) for i in range(1, 5)]
    cv = root / "cv-subject"
    write_csv_rows(cv / "manifest.csv", [asdict(record) for record in records])
    cache = {}
    for split in write_subject_cv_splits(records, cv, n_folds=2, seed=42):
        compute_normalization_stats(records, split, split.parent / "normalization.json",
                                    mode="train_global", moments_cache=cache)
    assert len(cache) == 4
    return cv, records


def _config(root, cv):
    return {
        "seed": 42, "cv_fold": 0, "cv_folds": 2, "cv_strategy": "subject", "cv_split_dir": str(cv),
        "run": {"base_output_dir": str(root / "runs")}, "swanlab": {"enabled": False},
        "data": {"experiments": [1], "lags": [0], "normalization_mode": "train_global",
                 "balance_train_cells": False, "balance_eval_cells": False, "num_workers": 0},
        "model": {"encoder": {"in_channels": "auto", "latent_dim": 4, "num_blocks": 1,
                              "kernel_size": 3, "dilations": [1]},
                  "subject_conditioning": {"enabled": True, "mode": "shared_plus_residual",
                                           "embedding_dim": 4, "condition_dropout": 0.1},
                  "fusion": {"enabled": False}},
        "train": {"iterations": 1, "validation_interval_iterations": 1, "batch_size": 10,
                  "learning_rate": 0.001, "amp": False, "progress_bar": False,
                  "checkpoint_metric": "val_loss", "checkpoint_mode": "min"},
        "plots": {"enabled": False, "splits": ["val"]},
    }


def test_thirty_subjects_disjoint_24_6_deterministic_across_experiments(tmp_path):
    records = [_record(tmp_path, i, trials=4) for i in range(1, 31)]
    # A second experiment for the same person must follow the same assignment.
    records.append(replace(records[0], exp=2, subject_uid="Exp2_Sub01"))
    paths = write_subject_cv_splits(records, tmp_path / "cv", seed=42)
    reordered = write_subject_cv_splits(records[::-1], tmp_path / "reordered", seed=42)
    other_seed = write_subject_cv_splits(records, tmp_path / "other", seed=17)
    seen = []
    for path, other in zip(paths, reordered):
        membership = subject_split_membership(records, load_split_map(path))
        assert len(membership["train"]) == 24 and len(membership["val"]) == 6
        assert not set(membership["train"]) & set(membership["val"])
        key = lambda row: (row["subject_uid"], row["block"])
        assert sorted(read_csv_rows(path), key=key) == sorted(read_csv_rows(other), key=key)
        assert path.read_bytes() == other.read_bytes()
        seen.extend(membership["val"])
    assert sorted(seen) == [f"Sub{i:02d}" for i in range(1, 31)]
    assert load_split_map(paths[0]) != load_split_map(other_seed[0])
    split = load_split_map(paths[0])
    split["Exp2_Sub01", 0] = "train" if split["Exp1_Sub01", 0] == "val" else "val"
    with pytest.raises(ValueError, match="participants overlap"):
        subject_split_membership(records, split)
    missing = load_split_map(paths[0])
    del missing["Exp2_Sub01", 0]
    # Leave another block for this UID so missing-block validation is exercised.
    missing["Exp2_Sub01", 99] = "train"
    with pytest.raises(ValueError, match="assign all blocks"):
        subject_split_membership(records, missing)


def test_global_stats_pooled_variance_and_no_heldout_eeg_reads(tmp_path):
    cv, records = _bundle(tmp_path)
    split = cv / "fold_0/block_splits.csv"
    members = subject_split_membership(records, load_split_map(split))
    pieces = []
    for record in records:
        if record.subject in members["train"]:
            with np.load(record.npz_path) as archive:
                pieces.append(archive["X"].transpose(1, 0, 2).reshape(2, -1))
        else:
            Path(record.npz_path).unlink()  # Fitting must succeed without validation EEG.
    expected = np.concatenate(pieces, axis=1)
    out = tmp_path / "global.json"
    compute_normalization_stats(records, split, out, mode="train_global")
    stats = load_normalization_stats(out)
    assert set(stats) == {record.subject_uid for record in records}
    for values in stats.values():
        np.testing.assert_allclose(values["mean"], expected.mean(axis=1, dtype=np.float64), atol=1e-6)
        np.testing.assert_allclose(values["std"], expected.std(axis=1, dtype=np.float64), atol=1e-6)
    validate_normalization_identity(records, split, out, mode="train_global")
    raw = json.loads(out.read_text())
    assert raw["fit_subjects"] == members["train"]
    assert raw["fit_observation_count_per_channel"] == expected.shape[1]
    with pytest.raises(ValueError, match="identity differs"):
        validate_normalization_identity(records, cv / "fold_1/block_splits.csv", out, mode="train_global")
    with pytest.raises(ValueError, match="mode differs"):
        validate_normalization_identity(records, split, out, mode="per_subject")


def test_actual_datasets_unseen_namespace_and_fold_reresolution(tmp_path):
    cv, _ = _bundle(tmp_path)
    config = _config(tmp_path, cv)
    configure_cross_validation(config)
    data = config["data"]
    datasets = {split: build_lag_dataset(data, task="choice", split=split, seed=42)
                for split in ("train", "val")}
    validate_subject_cv_datasets(config, datasets)
    capacity = configure_training_subject_conditioning(config["model"], datasets)
    ids = config["model"]["subject_conditioning"]["resolved_training_subject_ids"]
    assert len(ids) == 2
    val_ids = {int(datasets["val"].participant_indices[e.subject_index]) for e in datasets["val"].examples}
    assert not set(ids) & val_ids
    emb = SharedResidualSubjectEmbedding(capacity, 4, ids).eval()
    with torch.no_grad():
        emb.weight.fill_(999.)
    result = emb(torch.tensor(sorted(val_ids)))
    torch.testing.assert_close(result, emb.shared_condition.expand(len(val_ids), -1))
    wrong = copy.deepcopy(config)
    wrong["model"]["subject_conditioning"]["mode"] = "individual"
    with pytest.raises(ValueError, match="shared_plus_residual"):
        validate_subject_cv_datasets(wrong, datasets)
    with pytest.raises(ValueError, match="Actual train"):
        validate_subject_cv_datasets(config, {"train": datasets["val"], "val": datasets["train"]})
    old_val = config["cv_val_subjects"]
    configure_cross_validation(config, cv_fold=1)
    assert config["cv_train_subjects"] == old_val
    with pytest.raises(ValueError, match="strategy"):
        configure_cross_validation(_config(tmp_path, cv), cv_strategy="block")
    with pytest.raises(ValueError, match="requires normalization_mode"):
        validate_data_protocol({**data, "normalization_mode": "per_subject"})


@pytest.mark.parametrize("strategy", ["block", "subject"])
def test_protocol_validation_accepts_legacy_string_experiment_numbers(tmp_path, strategy):
    cv, records = _bundle(tmp_path)
    config = _config(tmp_path, cv)
    if strategy == "block":
        cv = tmp_path / "cv-block"
        write_csv_rows(cv / "manifest.csv", [asdict(record) for record in records])
        for split in write_block_cv_splits(records, cv, n_folds=2, seed=42):
            compute_normalization_stats(records, split, split.parent / "normalization.json")
        config.update(cv_strategy="block", cv_split_dir=str(cv))
        config["data"]["normalization_mode"] = "per_subject"
    config["data"]["experiments"] = ["1"]
    configure_cross_validation(config)
    membership = validate_data_protocol(config["data"])
    assert membership == ({"train": config["cv_train_subjects"], "val": config["cv_val_subjects"]}
                          if strategy == "subject" else None)


@pytest.mark.parametrize("module_name", ["train_target_decode", "train_time_decode"])
def test_all_training_entrypoints_reject_undeclared_subject_protocol(tmp_path, monkeypatch, module_name):
    """A fixed-path config must not silently treat subject artifacts as block data."""
    cv, _ = _bundle(tmp_path)
    config = _config(tmp_path, cv)
    for key in ("cv_fold", "cv_folds", "cv_strategy", "cv_split_dir"):
        config.pop(key)
    config["data"].pop("normalization_mode")
    config["data"].update(manifest_csv=str(cv / "manifest.csv"),
                          split_csv=str(cv / "fold_0/block_splits.csv"),
                          normalization_json=str(cv / "fold_0/normalization.json"))
    config["model"]["subject_conditioning"]["mode"] = "individual"
    path = tmp_path / "undeclared-subject-config.yaml"
    save_yaml(path, config)
    entry = importlib.import_module(f"scripts.{module_name}")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(entry, "PROJECT_ROOT", source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PB2023_SWANLAB_ENABLED", "0")
    monkeypatch.setattr(sys, "argv", [module_name, "--config", str(path), "--task", "choice"])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with pytest.raises(ValueError, match="CV strategy differs"):
            entry.main()
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("mode", ["replace", "concat"])
def test_graph_global_stats_and_cache_are_fold_specific(tmp_path, mode):
    cv, records = _bundle(tmp_path, channels=64)
    config = _config(tmp_path, cv)
    config["preprocessing"] = {"graph_laplacian": {"enabled": True, "mode": mode}}
    configure_cross_validation(config)
    configure_spatial_preprocessing(config)
    data = config["data"]
    graph = GraphLaplacian.load(Path(data["graph_laplacian_json"]), data["graph_laplacian_sha256"])
    selected = config["cv_train_subjects"]
    transformed = []
    for record in records:
        if record.subject in selected:
            with np.load(record.npz_path) as archive:
                x = archive["X"]
            gx = graph.apply(x)
            transformed.append(gx if mode == "replace" else np.concatenate([x, gx], axis=1))
    expected = np.concatenate(transformed)
    stats = load_normalization_stats(Path(data["normalization_json"]), graph_sha256=graph.sha256, graph_mode=mode)
    np.testing.assert_allclose(stats[records[0].subject_uid]["mean"], expected.mean(axis=(0, 2), dtype=np.float64), atol=1e-6)
    np.testing.assert_allclose(stats[records[0].subject_uid]["std"], expected.std(axis=(0, 2), dtype=np.float64), atol=1e-6)
    original = Path(data["normalization_json"])
    original_mtime = original.stat().st_mtime_ns
    configure_spatial_preprocessing(config)
    assert Path(config["data"]["normalization_json"]) == original
    assert original.stat().st_mtime_ns == original_mtime
    configure_cross_validation(config, cv_fold=1)
    configure_spatial_preprocessing(config)
    assert Path(config["data"]["normalization_json"]) != original
    with pytest.raises(ValueError, match="mode"):
        load_normalization_stats(original, graph_sha256=graph.sha256, graph_mode="concat" if mode == "replace" else "replace")


@pytest.mark.parametrize("module_name", ["train_target_decode", "train_time_decode"])
@pytest.mark.parametrize("graph_mode", [None, "replace", "concat"])
def test_subject_cv_cli_train_reexport_and_resume_from_local_artifacts(tmp_path, monkeypatch, module_name, graph_mode):
    cv, _ = _bundle(tmp_path, channels=64 if graph_mode else 2)
    config = _config(tmp_path, cv)
    if graph_mode:
        config["preprocessing"] = {"graph_laplacian": {"enabled": True, "mode": graph_mode}}
    source = tmp_path / "source"
    source.mkdir()
    entry = importlib.import_module(f"scripts.{module_name}")
    monkeypatch.setattr(entry, "PROJECT_ROOT", source)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PB2023_SWANLAB_ENABLED", "0")
    path = tmp_path / "config.yaml"
    save_yaml(path, config)
    argv = [module_name, "--config", str(path), "--task", "choice", "--cv-strategy", "subject"]
    monkeypatch.setattr(sys, "argv", argv)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        entry.main()
        run = next((tmp_path / "runs").glob("*/run-cv-fold0-of-2_*"))
        assert "subject-cv" in run.parent.name
        payload = torch.load(run / "last.ckpt", map_location="cpu", weights_only=False)
        snapshot = load_yaml(run / "config_snapshot.yaml")
        assert payload["cv_strategy"] == "subject"
        assert len(payload["cv_train_subjects"]) == len(payload["cv_val_subjects"]) == 2
        assert snapshot["data"]["normalization_mode"] == "train_global"
        assert len(snapshot["model"]["subject_conditioning"]["resolved_training_subject_ids"]) == 2
        assert any("shared_condition" in key for key in payload["model_state"])
        for key in ("cv_strategy", "cv_train_subjects"):
            wrong = dict(payload)
            wrong[key] = "block" if key == "cv_strategy" else []
            with pytest.raises(ValueError, match="CV"):
                validate_checkpoint_cv_identity(wrong, snapshot)
        exporter = importlib.import_module("scripts.reexport_target_outputs" if module_name == "train_target_decode" else "scripts.reexport_time_outputs")
        monkeypatch.setattr(exporter, "plot_target_run", lambda *a, **kw: None)
        shutil.rmtree(cv)  # Export/resume must use frozen artifacts, not original paths.
        monkeypatch.setattr(sys, "argv", ["reexport", "--run-dir", str(run)])
        exporter.main()
        config["train"]["iterations"] = 2
        save_yaml(path, config)
        monkeypatch.setattr(sys, "argv", [*argv, "--resume-from", str(run / "last.ckpt")])
        entry.main()
        assert torch.load(run / "last.ckpt", map_location="cpu", weights_only=False)["iteration"] == 2
        with pytest.raises(ValueError, match="different cv_strategy"):
            configure_cross_validation({"data": {}, "cv_strategy": "block"}, resume_from=run / "last.ckpt")
    finally:
        torch.set_num_threads(old_threads)
