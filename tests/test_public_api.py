from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from pb2023_eeg.artifacts.checkpoints import load_model, save_model
from pb2023_eeg.artifacts.exports import describe_export
from pb2023_eeg.artifacts.provenance import digest, model_semantics
from pb2023_eeg.contracts import PredictionSpec
from pb2023_eeg.data.transforms import time_shuffle_eeg
from pb2023_eeg.evaluation.export import export_target_outputs
from pb2023_eeg.evaluation.predict import iter_predictions
from pb2023_eeg.models.factory import build_model


def configuration():
    return {"task": "choice", "model": {
        "encoder": {"type": "cnn", "in_channels": 2, "latent_dim": 4, "num_blocks": 1,
                    "kernel_size": 3, "dropout": 0.0, "expansion_ratio": 1},
        "attention_mode": "gated_scorer", "attention_dim": 4,
        "classifier_num_layers": 1, "time_no_class_enabled": True,
        "resolved_time_point_count": 8}}


def test_loading_preserves_rng_and_all_heads_and_rejects_conflicting_semantics(tmp_path):
    config = configuration()
    original = build_model(config["model"], time_point_count=8).eval()
    file = tmp_path / "legacy.ckpt"
    torch.save({"config": config, "model_state": original.state_dict()}, file)
    before = torch.get_rng_state().clone()
    loaded = load_model(file)
    assert torch.equal(before, torch.get_rng_state())
    assert loaded.model.attention_gate_activation == "sigmoid"
    assert set(loaded.model.class_heads) == {"choice", "pitch", "motor"}
    eeg = torch.randn(2, 2, 8)
    for key, value in original(eeg, "choice").items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(loaded.model(eeg, "choice")[key], value, rtol=0, atol=0)
    other = deepcopy(config)
    other["model"]["attention_gate_activation"] = "tanh_plus_one"
    with pytest.raises(ValueError, match="conflicts"):
        load_model(file, config=other)
    with pytest.raises(ValueError, match="checksum"):
        load_model(file, expected_sha256="0" * 64)
    del config["model"]
    with pytest.raises(ValueError, match="config.model"):
        save_model(tmp_path / "invalid.ckpt", original, config=config)


def test_export_identity_binds_parameter_free_forward_semantics_and_data():
    config = configuration()["model"]
    first = build_model(config, time_point_count=8)
    second = build_model({**config, "attention_gate_activation": "tanh_plus_one"}, time_point_count=8)
    second.load_state_dict(first.state_dict(), strict=True)
    assert digest(model_semantics(first)) != digest(model_semantics(second))
    batch = {"eeg": torch.randn(2, 2, 8), "target": torch.tensor([0, 1])}
    args = dict(loader=[batch], task="choice", time_s=torch.arange(8), attention_indices=torch.arange(8),
                output_dir=Path("unused"), split="val", device=torch.device("cpu"), amp=False)
    a = describe_export(export_target_outputs, model=first, **args)
    b = describe_export(export_target_outputs, model=second, **args)
    assert a["model_state_sha256"] == b["model_state_sha256"]
    assert a["model_semantics_sha256"] != b["model_semantics_sha256"]
    batch["eeg"].add_(1)
    c = describe_export(export_target_outputs, model=first, **args)
    assert c["data_sha256"] != a["data_sha256"]


def test_prediction_preserves_native_dtype_no_class_and_training_flags():
    model = build_model(configuration()["model"], time_point_count=8)
    model.train()
    model.encoder.eval()
    flags = [m.training for m in model.modules()]
    raw = {"eeg": torch.randn(3, 2, 8), "trial0": torch.arange(3),
           "record_uid": ["Exp1_Sub01"] * 3, "lag": torch.ones(3, dtype=torch.long)}
    batches = list(iter_predictions(model, [raw], PredictionSpec("choice", score_modes={"time": "sigmoid", "pooled": "softmax"})))
    assert [m.training for m in model.modules()] == flags
    assert batches[0].outputs["time_logits"].shape == (3, 8, 3)
    assert batches[0].outputs["pooled_logits"].shape == (3, 2)
    np.testing.assert_allclose(batches[0].outputs["pooled_scores"].sum(-1), 1, rtol=1e-6)
    assert batches[0].outputs["time_logits"].dtype == np.float32
    assert batches[0].identity["record_uid"].tolist() == raw["record_uid"]


@pytest.mark.parametrize("bad", [True, 1.2, "3"])
def test_shuffle_rejects_identity_truncation(bad):
    with pytest.raises(ValueError, match="seed"):
        time_shuffle_eeg(torch.zeros(2, 2, 8), seed=bad, record_id=torch.zeros(2, dtype=torch.long),
                         trial0=torch.arange(2), lag=torch.ones(2, dtype=torch.long))


def test_shuffle_is_invariant_to_batch_order_and_size():
    eeg = torch.randn(4, 2, 8)
    keys = {"record_id": torch.tensor([0, 1, 0, 1]), "trial0": torch.arange(4), "lag": torch.ones(4, dtype=torch.long)}
    before = torch.get_rng_state().clone()
    whole = time_shuffle_eeg(eeg, seed=17, **keys)
    pieces = torch.cat([time_shuffle_eeg(eeg[i:i+1], seed=17, **{k: v[i:i+1] for k, v in keys.items()}) for i in range(4)])
    torch.testing.assert_close(whole, pieces, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())


def test_installed_package_has_no_old_source_dependency(tmp_path):
    project = Path(__file__).resolve().parents[1]
    code = """
import pathlib, sys
import pb2023_eeg
from pb2023_eeg.workflows import train, reexport
from pb2023_eeg.artifacts.checkpoints import load_model
root = pathlib.Path(sys.argv[1])
for name, module in tuple(sys.modules.items()):
    if name == 'pb2023_eeg' or name.startswith('pb2023_eeg.'):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(root), (name, module.__file__)
assert not any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(project / "pb2023_eeg")], cwd=tmp_path, check=True)
