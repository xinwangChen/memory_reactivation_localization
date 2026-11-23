from pathlib import Path

import pytest
import torch

from pb2023_eeg.training.common import atomic_torch_save, load_checkpoint, save_checkpoint


def test_atomic_checkpoint_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "best_val.ckpt"
    atomic_torch_save({"iteration": 1, "weights": torch.arange(3)}, path)
    atomic_torch_save({"iteration": 2, "weights": torch.arange(4)}, path)
    restored = load_checkpoint(path)
    assert restored["iteration"] == 2
    torch.testing.assert_close(restored["weights"], torch.arange(4))
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["serialization", "sync", "replacement"])
def test_interrupted_save_preserves_previous_checkpoint(tmp_path, monkeypatch, existing, failure):
    path = tmp_path / "last.ckpt"
    previous = None
    if existing:
        atomic_torch_save({"iteration": 7}, path)
        previous = path.read_bytes()

    def fail_write(payload, handle):
        handle.write(b"incomplete checkpoint")
        raise OSError("simulated write failure")

    def fail(*args, **kwargs):
        raise OSError("simulated storage failure")

    target = {
        "serialization": "pb2023_eeg.training.common.torch.save",
        "sync": "pb2023_eeg.training.common.os.fsync",
        "replacement": "pb2023_eeg.training.common.os.replace",
    }[failure]
    monkeypatch.setattr(target, fail_write if failure == "serialization" else fail)
    with pytest.raises(OSError):
        atomic_torch_save({"iteration": 8}, path)

    if existing:
        assert path.read_bytes() == previous
        assert load_checkpoint(path)["iteration"] == 7
    else:
        assert not path.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_training_checkpoint_records_actual_selection_identity(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "best_val.ckpt"
    save_checkpoint(
        path, model, optimizer, 0, {"train": {"checkpoint_metric": "val_fusion_accuracy"}},
        {"val_pooled_accuracy": 0.75}, iteration=20,
        selection_metric="val_pooled_accuracy", selection_mode="max",
    )
    checkpoint = load_checkpoint(path)
    assert checkpoint["selection_metric"] == "val_pooled_accuracy"
    assert checkpoint["selection_mode"] == "max"
    assert checkpoint["selection_score"] == 0.75
    # The original config remains available, including an automatically resolved alias.
    assert checkpoint["config"]["train"]["checkpoint_metric"] == "val_fusion_accuracy"


@pytest.mark.parametrize('saved_enabled,current_enabled', [(False, True), (True, True), (True, False), (False, False)])
def test_scaler_resume_supports_device_precision_transitions(tmp_path, saved_enabled, current_enabled):
    from pb2023_eeg.training.common import load_training_checkpoint
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    saved = torch.amp.GradScaler('cpu', enabled=saved_enabled, init_scale=16.)
    current = torch.amp.GradScaler('cpu', enabled=current_enabled, init_scale=128.)
    path = tmp_path / 'last.ckpt'
    config = {'train': {'amp': True}}
    save_checkpoint(path, model, optimizer, 0, config, {}, scaler=saved)
    load_training_checkpoint(path, model=model, optimizer=optimizer, scaler=current, expected_config=config)
    expected_scale = (16. if saved_enabled else 128.) if current_enabled else 1.
    assert current.get_scale() == expected_scale
    optimizer.zero_grad()
    current.scale(model(torch.ones(2, 2)).square().mean()).backward()
    current.step(optimizer)
    current.update()
    assert all(torch.isfinite(p).all() for p in model.parameters())
