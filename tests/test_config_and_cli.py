from pathlib import Path
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from pb2023_eeg.config import DataContext, parse_request, resolve_request
from pb2023_eeg.models.factory import build_model
from pb2023_eeg.artifacts.checkpoints import save_model
from pb2023_eeg.contracts import PredictionSpec
from pb2023_eeg.evaluation.predict import iter_predictions


@pytest.mark.parametrize("task", ["choice", "pitch"])
@pytest.mark.parametrize("cv", ["block", "subject"])
def test_registered_recipes_resolve_with_explicit_data_context(task, cv):
    file = Path(__file__).resolve().parents[1] / "configs" / f"{task}-{cv}.yaml"
    request = parse_request(file, {"task": task})
    resolved = resolve_request(request, DataContext(64, np.arange(275) * .01,
                               {"pitch": tuple(range(5)), "choice": (0, 1), "motor": (0, 1)},
                               training_subject_ids=(0, 2, 5), embedding_capacity=6))
    assert resolved.model_spec.input_channels == 64
    assert resolved.model_spec.time_point_count == 275
    assert resolved.config["train"] == request.config["train"]


def test_config_paths_overrides_and_type_errors(tmp_path, monkeypatch):
    file = tmp_path / "request.yaml"
    raw = {"seed": 17, "data": {"manifest_csv": "inputs/manifest.csv"},
           "model": {"encoder": {"in_channels": "auto"}}, "train": {"amp": False}}
    file.write_text(yaml.safe_dump(raw))
    monkeypatch.chdir("/tmp")
    request = parse_request(file, {"seed": 19})
    assert request.config["seed"] == 19
    assert request.config["data"]["manifest_csv"] == str(tmp_path / "inputs/manifest.csv")
    for override in ({"seed": True}, {"seed": 2.1}, {"train": {"amp": "false"}}, {"typo": {}}):
        with pytest.raises(ValueError):
            parse_request(file, override)
    with pytest.raises(ValueError, match="increasing"):
        resolve_request(request, DataContext(2, [0., 0.], {}))


def test_source_snapshot_keeps_artifact_code_and_excludes_run_data(tmp_path):
    from pb2023_eeg.utils.run import snapshot_project_source
    source = tmp_path / "project"
    for name in ("pb2023_eeg/artifacts/checkpoints.py", "configs/example.yaml", "runs/old/huge.csv", ".venv/bin/python"):
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name)
    output = source / "runs/new"
    output.mkdir(parents=True)
    frozen = snapshot_project_source(source, output)
    assert (frozen / "pb2023_eeg/artifacts/checkpoints.py").is_file()
    assert (frozen / "configs/example.yaml").is_file()
    assert not (frozen / "runs").exists()
    assert not (frozen / ".venv").exists()


def test_prediction_cli_preserves_attention_support_across_unequal_batches(tmp_path):
    model_config = {"encoder": {"type": "cnn", "in_channels": 2, "latent_dim": 4,
                               "num_blocks": 1, "kernel_size": 3, "dropout": 0., "expansion_ratio": 1},
                    "attention_mode": "scorer", "attention_dim": 4, "pool_valid_context_only": True,
                    "resolved_time_point_count": 8, "classifier_num_layers": 1, "time_no_class_enabled": True}
    model = build_model(model_config, time_point_count=8).eval()
    checkpoint, inputs, output = (tmp_path / name for name in ("model.ckpt", "input.npz", "output.npz"))
    save_model(checkpoint, model, config={"task": "choice", "model": model_config})
    eeg = torch.randn(5, 2, 8)
    np.savez(inputs, eeg=eeg.numpy(), record_uid=np.array(["Exp1_Sub01"] * 5), trial0=np.arange(5), lag=np.ones(5, dtype=np.int64))
    command = [sys.executable, "-m", "pb2023_eeg", "predict", str(checkpoint), "--input", str(inputs),
               "--output", str(output), "--batch-size", "2"]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    with np.load(output) as actual:
        assert actual["time_logits"].shape == (5, 8, 3)
        np.testing.assert_allclose(actual["pooled_logits"], model(eeg, "choice")["pooled_logits"].detach().numpy(), atol=1e-6)
        metadata = json.loads(str(actual["metadata_json"]))
        assert len(metadata["prediction"]["attention_valid_mask"]) == 8
        assert "attention_valid_mask" not in actual
    with pytest.raises(ValueError, match="integers"):
        list(iter_predictions(model, [{"eeg": eeg}], PredictionSpec("choice", attention_indices=(0., 1.))))
