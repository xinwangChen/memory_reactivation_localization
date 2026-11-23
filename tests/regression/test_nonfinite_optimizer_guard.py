import copy

import pytest
import torch

from pb2023_eeg.training.target import run_target_epoch


class TinyDecoder(torch.nn.Module):
    fusion_enabled = False

    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
        self.bad_forward = False

    def forward(self, eeg, task, subject_id=None, attention_indices=None, fusion_indices=None):
        logits = self.logits[None].expand(eeg.shape[0], -1) * 1.0
        if self.bad_forward:
            logits = logits * float("nan")
        return {"time_logits": logits[:, None, :].expand(-1, eeg.shape[-1], -1),
                "pooled_logits": logits,
                "attention_weights": torch.ones_like(eeg[:, 0]) / eeg.shape[-1]}


def run(kind, model, optimizer, scaler, batches=1):
    common = dict(model=model, loader=[{"eeg": torch.zeros(2, 1, 3),
                  "target": torch.tensor([0, 1])}] * batches, task="choice",
                  loss_weights={"time": 1.0, "pooled": 1.0}, label_smoothing=0.0,
                  device=torch.device("cpu"), amp=False, optimizer=optimizer,
                  scaler=scaler, grad_clip_norm=1.0, progress_bar_enabled=False)
    if kind == "target":
        return run_target_epoch(**common, attention_indices=torch.arange(3), top_k=1, top_ratio=1.0)
    return run_time_decode_epoch(**common, time_indices=torch.arange(3), sample_ratio=1.0)


@pytest.mark.parametrize("kind", ["target"])
@pytest.mark.parametrize("bad", [float("inf"), float("nan")])
def test_bad_unscaled_gradient_preserves_parameters_and_optimizer_state(kind, bad):
    model = TinyDecoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    run(kind, model, optimizer, scaler)
    before = copy.deepcopy(model.state_dict())
    state = copy.deepcopy(optimizer.state_dict())
    model.logits.register_hook(lambda gradient: torch.full_like(gradient, bad))
    with pytest.raises(FloatingPointError, match="Non-finite gradients"):
        run(kind, model, optimizer, scaler)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    for key, values in optimizer.state_dict()["state"].items():
        for name, value in values.items():
            torch.testing.assert_close(value, state["state"][key][name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("kind", ["target"])
def test_nonfinite_unscaled_loss_fails_before_backward(kind):
    model = TinyDecoder()
    model.bad_forward = True
    before = model.logits.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        run(kind, model, optimizer, torch.amp.GradScaler("cpu", enabled=False))
    torch.testing.assert_close(model.logits, before, rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("kind", ["target"])
def test_enabled_scaler_skips_overflow_then_recovers(kind):
    model = TinyDecoder()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128.)
    triggered = False
    def overflow_once(gradient):
        nonlocal triggered
        if not triggered:
            triggered = True
            return torch.full_like(gradient, torch.inf)
        return gradient
    model.logits.register_hook(overflow_once)
    before = model.logits.detach().clone()
    metrics = run(kind, model, optimizer, scaler, batches=2)
    assert metrics["nonfinite_grad_batches"] == 1
    assert scaler.get_scale() == 64.
    assert torch.isfinite(model.logits).all()
    assert not torch.equal(model.logits, before)
    assert all(int(values["step"]) == 1 for values in optimizer.state.values())
