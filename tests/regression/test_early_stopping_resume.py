"""Resuming a terminal checkpoint must not extend the optimization trajectory."""

from copy import deepcopy

import pytest
import torch

from pb2023_eeg.training.common import load_checkpoint
from pb2023_eeg.training.target import fit_target_decoder
from pb2023_eeg.workflows.train import build_model as build_target


@pytest.fixture(params=["target"])
def training(request):
    torch.manual_seed(7)
    model_config = dict(
        encoder=dict(type="cnn", in_channels=2, latent_dim=4, num_blocks=1,
                     kernel_size=3, expansion_ratio=1, dropout=0.),
        classifier_num_layers=1, classifier_dropout=0., attention_mode="scorer",
        attention_dim=4, attention_dropout=0., fusion=dict(enabled=False))
    metric = "val_pooled_accuracy" if request.param == "target" else "val_time_accuracy"
    config = dict(model=model_config, data={}, train=dict(
        iterations=6, validation_interval_iterations=2, log_interval_iterations=1,
        learning_rate=.001, lr_schedule="constant", amp=False, progress_bar=False,
        loss_type="ce", time_loss_weight=1., pooled_loss_weight=1., fusion_loss_weight=0.,
        checkpoint_metric=metric, checkpoint_mode="max", patience_iterations=2))
    build, fit = ((build_target, fit_target_decoder) if request.param == "target"
                  else (build_time, fit_time_decoder))
    model = build(model_config, time_point_count=8)
    # Identical inputs with balanced labels give constant 0.5 validation accuracy,
    # while the optimizer still has nonzero gradients to update.
    batch = dict(eeg=torch.randn(1, 2, 8).repeat(4, 1, 1), target=torch.tensor([0, 1, 0, 1]),
                 subject_id=torch.zeros(4, dtype=torch.long), current_stimulus=torch.ones(4, dtype=torch.long))

    def run(output, *, resume=None, settings=None, train_loader=None, val_loader=None):
        return fit(model, [batch] if train_loader is None else train_loader,
                   [batch] if val_loader is None else val_loader, "choice", torch.arange(8),
                   config if settings is None else settings, output, torch.device("cpu"), resume_from=resume)

    return model, config, run


def assert_state_equal(model, saved):
    assert set(model.state_dict()) == set(saved)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved[name], rtol=0, atol=0)


def test_resume_after_early_stop_preserves_checkpoint_metrics_and_parameters(tmp_path, training):
    model, _, run = training
    run(tmp_path)
    saved = load_checkpoint(tmp_path / "last.ckpt")
    assert saved["iteration"] == 4
    assert saved["extra_state"]["validations_without_improvement"] == 1
    files = {name: (tmp_path / name).read_bytes() for name in
             ("last.ckpt", "best_val.ckpt", "metrics.csv", "train_metrics.csv")}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10)
    torch.rand(10)

    class NoDataReads:
        def __iter__(self):
            raise AssertionError("A stopped resume must not read training or validation data")

    assert run(tmp_path, resume=tmp_path / "last.ckpt",
               train_loader=NoDataReads(), val_loader=NoDataReads()) == []
    assert_state_equal(model, saved["model_state"])
    assert torch.equal(torch.get_rng_state(), saved["rng_states"]["torch_cpu"])
    for name, original in files.items():
        assert (tmp_path / name).read_bytes() == original


def test_nonterminal_checkpoint_still_resumes_to_the_same_early_stop(tmp_path, training):
    model, _, run = training
    run(tmp_path)
    completed = load_checkpoint(tmp_path / "last.ckpt")
    best = load_checkpoint(tmp_path / "best_val.ckpt")
    assert best["iteration"] == 2 and best["extra_state"]["validations_without_improvement"] == 0
    history = run(tmp_path, resume=tmp_path / "best_val.ckpt")
    assert [row["iteration"] for row in history] == [4]
    assert load_checkpoint(tmp_path / "last.ckpt")["iteration"] == completed["iteration"]
    assert_state_equal(model, completed["model_state"])


def test_iteration_limit_can_be_extended_when_early_stop_was_not_reached(tmp_path, training):
    _, config, run = training
    short = deepcopy(config)
    short["train"].update(iterations=2, patience_iterations=10)
    run(tmp_path, settings=short)
    assert load_checkpoint(tmp_path / "last.ckpt")["extra_state"]["validations_without_improvement"] == 0
    extended = deepcopy(short)
    extended["train"]["iterations"] = 4
    history = run(tmp_path, settings=extended, resume=tmp_path / "last.ckpt")
    assert [row["iteration"] for row in history] == [4]
    assert load_checkpoint(tmp_path / "last.ckpt")["iteration"] == 4
