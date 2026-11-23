"""Numerical and CPU integration coverage for configurable target focal BCE."""
from __future__ import annotations

import csv
import math
from copy import deepcopy

import pytest
import torch
import torch.nn.functional as F

from memory_reactivation.training.common import load_checkpoint, validate_resume_config
from memory_reactivation.training.target import (
    attention_topk_clean_time_loss,
    classification_branch_loss,
    export_target_outputs,
    fit_target_decoder,
    focal_bce_loss_values,
    normalize_loss_type,
    resolve_focal_params,
    resolve_focal_train_config,
    score_mode_for_loss_type,
)
from memory_reactivation.workflows.train import build_model
from scripts import reexport_target_outputs
from memory_reactivation.training import trainer as target_training


@pytest.mark.parametrize("classes,smoothing,sigma,pos_weight", [
    (2, 0.0, None, None), (2, 0.1, None, 1.7), (5, 0.1, 0.7, None),
])
def test_gamma_zero_preserves_bce_loss_and_gradient(classes, smoothing, sigma, pos_weight):
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(4, classes, generator=generator, dtype=torch.float64, requires_grad=True)
    other = logits.detach().clone().requires_grad_(True)
    target = torch.tensor([0, 1, 0, 1])
    options = dict(label_smoothing=smoothing, ordinal_sigma=sigma, bce_pos_weight=pos_weight,
                   sample_weight=torch.tensor([0.25, 2.0, 1.0, 0.0], dtype=torch.float64))
    ordinary = classification_branch_loss(logits, target, "bce_one_vs_rest", **options)
    focal = classification_branch_loss(other, target, "focal_bce", focal_gamma=0.0, **options)
    torch.testing.assert_close(focal, ordinary, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(focal, other)[0],
                               torch.autograd.grad(ordinary, logits)[0], rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("branch", ["pooled", "time"])
def test_gamma_zero_matches_bce_with_autocast_and_positive_weight(dtype, branch):
    target = torch.tensor([0, 1])
    logits = torch.tensor([[.3, -.7], [-1.2, .8]], dtype=dtype, requires_grad=True)
    other = logits.detach().clone().requires_grad_(True)
    weights = torch.tensor([.5, 2.])
    with torch.autocast('cpu', dtype=torch.bfloat16):
        if branch == 'pooled':
            expected = classification_branch_loss(logits, target, 'bce', bce_pos_weight=1.7,
                                                  sample_weight=weights)
            actual = classification_branch_loss(other, target, 'focal_bce', focal_gamma=0,
                                                bce_pos_weight=1.7, sample_weight=weights)
        else:
            options = dict(target=target, time_indices=torch.arange(2), attention_weights=torch.ones(2, 2),
                           clean_time_mask=None, top_ratio=1., bce_pos_weight=1.7, sample_weight=weights)
            expected = attention_topk_clean_time_loss(logits[:, None].expand(-1, 2, -1),
                                                     loss_type='bce', **options)[0]
            actual = attention_topk_clean_time_loss(other[:, None].expand(-1, 2, -1),
                                                   loss_type='focal_bce', focal_gamma=0, **options)[0]
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.autograd.grad(actual, other)[0],
                               torch.autograd.grad(expected, logits)[0], atol=0, rtol=0)


def test_focal_dispatch_rejects_none_gamma_instead_of_silently_using_bce():
    with pytest.raises(ValueError, match='focal_gamma'):
        classification_branch_loss(torch.zeros(2, 2), torch.tensor([0, 1]), 'focal_bce', focal_gamma=None)


@pytest.mark.parametrize('gamma', [0., .5, 2.])
@pytest.mark.parametrize('alpha', [None, .25, 1.])
@pytest.mark.parametrize('soft_targets', [False, True])
def test_focal_loss_and_gradients_match_torchvision_reference(gamma, alpha, soft_targets):
    reference = pytest.importorskip('torchvision.ops').sigmoid_focal_loss
    generator = torch.Generator().manual_seed(37)
    logits = (torch.randn(3, 4, 2, generator=generator, dtype=torch.float64) * 3).requires_grad_(True)
    other = logits.detach().clone().requires_grad_(True)
    targets = torch.rand(3, 4, 2, generator=generator, dtype=torch.float64)
    if not soft_targets:
        targets = (targets > .5).double()
    expected = reference(logits, targets, gamma=gamma, alpha=-1 if alpha is None else alpha)
    actual = focal_bce_loss_values(other, targets, gamma=gamma, alpha=alpha)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), other)[0],
                               torch.autograd.grad(expected.sum(), logits)[0], atol=1e-12, rtol=1e-9)


def test_focal_suppresses_easy_examples_and_preserves_trial_weighting():
    z = math.log(9.0)
    logits = torch.tensor([[z, -z], [0.0, 0.0], [-z, z]], dtype=torch.float64)
    targets = torch.tensor([0, 0, 0])
    # Both binary terms have p_t = 0.9, 0.5, 0.1 respectively.
    bce = F.binary_cross_entropy_with_logits(
        logits, F.one_hot(targets, 2).double(), reduction="none",
    ).mean(dim=-1)
    factors = torch.tensor([0.01, 0.25, 0.81], dtype=torch.float64)
    weights = torch.tensor([1., 3., 0.], dtype=torch.float64)
    actual = classification_branch_loss(logits, targets, "focal_bce", sample_weight=weights)
    torch.testing.assert_close(actual, (weights * factors * bce).sum() / weights.sum())


def test_focal_factor_remains_in_gradient_graph():
    logits = torch.tensor([[0.4, -0.3], [-0.7, 1.2]], dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [0.1, 0.9]], dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda z: focal_bce_loss_values(z, targets, gamma=1.3, alpha=0.3).sum(),
        (logits,), eps=1e-6, atol=1e-5,
    )


@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_extreme_logits_have_finite_loss_and_gradients(gamma, dtype):
    logits = torch.tensor([[1000., -1000.], [-1000., 1000.], [30., -30.]], dtype=dtype,
                          requires_grad=True)
    targets = torch.tensor([[1., 0.], [1., 0.], [1., 0.]], dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = focal_bce_loss_values(logits, targets, gamma=gamma).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_alpha_weights_binary_terms_and_is_optional():
    logits = torch.zeros(2, 2)
    targets = torch.eye(2)
    plain = focal_bce_loss_values(logits, targets, gamma=0)
    balanced = focal_bce_loss_values(logits, targets, gamma=0, alpha=0.25)
    expected_weights = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    torch.testing.assert_close(balanced, plain * expected_weights)
    # alpha is not a weight attached only to the Choice label 1 trial.
    torch.testing.assert_close(balanced[0].mean(), balanced[1].mean())


@pytest.mark.parametrize("key,value", [
    ("focal_gamma", -1), ("focal_gamma", float("nan")), ("focal_gamma", float("inf")),
    ("focal_gamma", True), ("pooled_focal_gamma", None), ("time_focal_gamma", "bad"),
    ("focal_alpha", -0.1), ("focal_alpha", 1.1), ("focal_alpha", float("nan")),
    ("focal_alpha", float("inf")), ("focal_alpha", False), ("fusion_focal_alpha", "bad"),
])
def test_invalid_focal_config_is_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        resolve_focal_params({key: value})


def test_branch_overrides_defaults_snapshots_and_export_mode():
    config = dict(loss_type="bce", pooled_loss_type="focal", focal_gamma=1.5, focal_alpha=0.3,
                  pooled_focal_gamma=0, pooled_focal_alpha=None)
    resolved = resolve_focal_train_config(config)
    assert config['pooled_loss_type'] == 'focal'  # Caller is not mutated.
    assert resolved['pooled_loss_type'] == 'focal_bce'
    assert resolved['pooled_focal_gamma'] == 0.
    assert resolved['pooled_focal_alpha'] is None
    assert 'time_focal_gamma' not in resolved
    assert resolve_focal_params(resolved)['time'] == dict(focal_gamma=1.5, focal_alpha=0.3)
    defaults = resolve_focal_train_config(dict(loss_type="bce", pooled_loss_type="focal_bce"))
    assert defaults['pooled_focal_gamma'] == 2.
    assert defaults['pooled_focal_alpha'] is None
    assert resolve_focal_train_config(dict(loss_type="ce")) == dict(loss_type="ce")
    for alias in ('focal', 'focal-bce', 'sigmoid_focal', 'focal_bce_one_vs_rest'):
        assert normalize_loss_type(alias) == 'focal_bce'
        assert score_mode_for_loss_type(alias) == 'sigmoid'
        assert reexport_target_outputs.score_mode_for_loss_type(alias) == 'sigmoid'


@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_time_focal_honors_no_class_and_excludes_unsupervised_positions(gamma):
    logits = torch.zeros(2, 4, 3, requires_grad=True)
    clean = torch.tensor([[True, False, True, False], [True, False, False, False]])
    covered = torch.tensor([[False, True, False, False], [False, True, False, False]])
    options = dict(target=torch.tensor([0, 1]), time_indices=torch.arange(4),
                   attention_weights=torch.ones(2, 4), clean_time_mask=clean,
                   covered_time_mask=covered, no_class_index=2, top_ratio=1.0,
                   sample_weight=torch.tensor([1.0, 2.0]))
    result = attention_topk_clean_time_loss(logits, loss_type="focal_bce", focal_gamma=gamma, **options)
    assert torch.equal(result[4], clean | covered)
    if gamma == 0:
        expected = attention_topk_clean_time_loss(logits, loss_type="bce", **options)[0]
        torch.testing.assert_close(result[0], expected, rtol=0, atol=0)
    result[0].backward()
    assert torch.equal(logits.grad[~(clean | covered)], torch.zeros_like(logits.grad[~(clean | covered)]))
    assert (logits.grad[:, 1, 2] < 0).all()  # Covered positions target no-class.
    assert (logits.grad[:, 1, :2] > 0).all()


@pytest.mark.parametrize("encoder_type", ["cnn", "window_transformer"])
@pytest.mark.parametrize("task", ["choice", "pitch"])
def test_cpu_fit_checkpoint_reload_and_real_gaussian_exports(tmp_path, encoder_type, task):
    torch.manual_seed(11)
    encoder = dict(type=encoder_type, in_channels=3, latent_dim=8, dropout=0.0)
    if encoder_type == 'cnn':
        encoder.update(num_blocks=1, kernel_size=3, expansion_ratio=1)
    else:
        encoder.update(num_layers=1, window_size=3, num_heads=2, ffn_ratio=2,
                       attention_pattern='centered_local', position_encoding_mode='sinusoidal_qk')
    model_config = dict(encoder=encoder, attention_mode='gated_scorer', attention_dim=8,
                        time_no_class_enabled=True, classifier_num_layers=1, classifier_dropout=0.,
                        subject_conditioning=dict(enabled=True, embedding_dim=4), fusion=dict(enabled=True))
    model = build_model(model_config, n_subjects=2, time_point_count=8)
    targets = [0, 1, 1, 0] if task == 'choice' else [0, 2, 4, 1]
    class_count = 2 if task == 'choice' else 5
    batch = dict(eeg=torch.randn(4, 3, 8), target=torch.tensor(targets),
                 sample_weight=torch.tensor([1., .5, 2., 1.]), subject_id=torch.tensor([0, 0, 1, 1]),
                 record_id=torch.tensor([0, 0, 1, 1]), exp=torch.ones(4, dtype=torch.long),
                 trial0=torch.arange(4), lag=torch.ones(4, dtype=torch.long),
                 current_stimulus=torch.ones(4, dtype=torch.long))
    config = dict(model=model_config, data={}, train=dict(
        iterations=2, validation_interval_iterations=1, log_interval_iterations=1,
        learning_rate=1e-3, weight_decay=0., time_loss_weight=.2, pooled_loss_weight=1.,
        fusion_loss_weight=.1, loss_type='bce', time_loss_type='bce', pooled_loss_type='focal_bce',
        pooled_focal_gamma=1., fusion_loss_type='focal_bce', fusion_focal_gamma=1.5,
        checkpoint_metric='val_pooled_accuracy', checkpoint_mode='max', amp=False, progress_bar=False,
    ))
    history = fit_target_decoder(model, [batch], [batch], task, torch.arange(8), config, tmp_path,
                                 torch.device('cpu'))
    assert len(history) == 2
    assert history[-1]['pooled_focal_gamma'] == 1.
    assert history[-1]['fusion_focal_gamma'] == 1.5
    assert 'time_focal_gamma' not in history[-1]
    assert all(row['nonfinite_grad_batches'] == 0 for row in history)
    checkpoint = load_checkpoint(tmp_path / 'best_val.ckpt', map_location='cpu')
    assert checkpoint['config']['train']['pooled_focal_alpha'] is None
    assert checkpoint['selection_metric'] == 'val_pooled_accuracy'
    assert (tmp_path / 'last.ckpt').is_file()
    restored = build_model(checkpoint['config']['model'], n_subjects=2, time_point_count=8)
    restored.load_state_dict(checkpoint['model_state'])
    changed = {**checkpoint['config'], 'train': {**checkpoint['config']['train'], 'pooled_focal_gamma': 2.}}
    with pytest.raises(ValueError, match='pooled_focal_gamma'):
        validate_resume_config(changed, checkpoint['config'])
    restored.eval()
    for kind in ('real', 'gaussian'):
        export_target_outputs(
            restored, [batch], task, torch.device('cpu'), False, tmp_path, 'val', torch.arange(8),
            [i * .01 for i in range(8)], progress_bar_enabled=False, input_kind=kind, gaussian_seed=17,
            time_score_mode=score_mode_for_loss_type('bce'), pooled_score_mode=score_mode_for_loss_type('focal_bce'),
            fusion_score_mode=score_mode_for_loss_type('focal_bce'),
        )
        prefix = 'val' if kind == 'real' else 'val_gaussian'
        with (tmp_path / f'{prefix}_trial_metrics.csv').open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 4
        assert all(r['time_output_class_count'] == str(class_count + 1)
                   and r['pooled_output_class_count'] == str(class_count) for r in rows)
        if kind == 'real':
            accuracy = sum(int(r['pooled_correct']) for r in rows) / len(rows)
            assert accuracy == checkpoint['metrics']['val_pooled_accuracy']


def test_focal_resume_matches_uninterrupted_training_with_dropout(tmp_path, monkeypatch):
    class TinyTarget(torch.nn.Module):
        fusion_enabled = False

        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 4)
            self.dropout = torch.nn.Dropout(.3)
            self.head = torch.nn.Linear(4, 2)

        def forward(self, eeg, task, attention_indices, subject_id=None):
            hidden = self.dropout(self.projection(eeg.transpose(1, 2)).tanh())
            time_logits = self.head(hidden)
            return dict(time_logits=time_logits, pooled_logits=time_logits.mean(dim=1),
                        attention_weights=eeg.new_full((eeg.shape[0], eeg.shape[2]), 1 / eeg.shape[2]))

    generator = torch.Generator().manual_seed(73)
    batches = [dict(eeg=torch.randn(4, 3, 4, generator=generator), target=torch.tensor([0, 1, 1, 0]),
                    subject_id=torch.zeros(4, dtype=torch.long), current_stimulus=torch.ones(4, dtype=torch.long),
                    sample_weight=torch.tensor([.5, 1., 2., 1.])) for _ in range(2)]
    config = dict(data={}, train=dict(iterations=4, validation_interval_iterations=1,
                  log_interval_iterations=1, learning_rate=1e-3, weight_decay=.1, amp=False, progress_bar=False,
                  time_loss_weight=.2, pooled_loss_weight=1., loss_type='bce', pooled_loss_type='focal_bce',
                  pooled_focal_gamma=1.5, checkpoint_metric='val_pooled_accuracy', checkpoint_mode='max'))
    torch.manual_seed(71)
    initial = deepcopy(TinyTarget().state_dict())
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    full.mkdir()
    interrupted.mkdir()

    def fit(model, directory, resume=None):
        return fit_target_decoder(model, batches, batches, 'choice', torch.arange(4), config,
                                  directory, torch.device('cpu'), resume_from=resume)

    model = TinyTarget()
    model.load_state_dict(initial)
    torch.manual_seed(89)
    fit(model, full)

    model = TinyTarget()
    model.load_state_dict(initial)
    torch.manual_seed(89)
    original_append = target_training.append_metrics_csv

    def interrupt_after_second_checkpoint(path, row):
        if path.name == 'train_metrics.csv' and int(row['iteration']) == 3:
            raise InterruptedError('synthetic interruption')
        original_append(path, row)

    with monkeypatch.context() as patcher:
        patcher.setattr(target_training, 'append_metrics_csv', interrupt_after_second_checkpoint)
        with pytest.raises(InterruptedError, match='synthetic interruption'):
            fit(model, interrupted)
    assert load_checkpoint(interrupted / 'last.ckpt')['iteration'] == 2
    # Model creation consumes RNG; the checkpoint loader must restore the training RNG afterwards.
    resumed = TinyTarget()
    fit(resumed, interrupted, interrupted / 'last.ckpt')
    expected = load_checkpoint(full / 'last.ckpt')
    actual = load_checkpoint(interrupted / 'last.ckpt')
    assert actual['iteration'] == expected['iteration'] == 4
    for key, value in expected['model_state'].items():
        torch.testing.assert_close(actual['model_state'][key], value, rtol=0, atol=0)
    for name in ('metrics.csv', 'train_metrics.csv'):
        assert (full / name).read_text() == (interrupted / name).read_text()
