"""Post-gate softmax uses the actual weights and preserves pooling support."""
from copy import deepcopy
from memory_reactivation.training import step as step_module
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from memory_reactivation.models.encoder import EEGEncoder
from memory_reactivation.models.target_decoder import GatedTimeAttentionPooling, TargetAttentionDecoder
from memory_reactivation.training.common import validate_resume_config
from memory_reactivation.training.target import export_target_outputs, run_target_epoch
from memory_reactivation.training import trainer as training
from memory_reactivation.workflows.train import build_model


def make_model(activation='tanh_plus_one', combination='multiply', **kwargs):
    return TargetAttentionDecoder(
        EEGEncoder(in_channels=3, latent_dim=6, num_blocks=1, kernel_size=3, dropout=0.),
        attention_mode='gated_scorer', attention_gate_activation=activation,
        attention_gate_combination=combination, attention_effective_normalization='softmax',
        classifier_num_layers=1, time_no_class_enabled=True, **kwargs)


def make_batch(n_times=9):
    return dict(eeg=torch.randn(2, 3, n_times), target=torch.tensor([0, 1]),
        subject_id=torch.tensor([0, 1]), current_stimulus=torch.tensor([0, 0]),
        record_id=torch.tensor([0, 1]), trial0=torch.tensor([0, 0]),
        lag=torch.tensor([1, 1]), exp=torch.tensor([1, 1]))


@pytest.mark.parametrize('activation', ['sigmoid', 'tanh', 'tanh_plus_one'])
@pytest.mark.parametrize('combination', ['multiply', 'add'])
@pytest.mark.parametrize('mask_kind', ['none', 'shared', 'per_trial'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.bfloat16])
def test_softmax_after_combination_pooling_and_backward(activation, combination, mask_kind, dtype):
    torch.manual_seed(728)
    pool = GatedTimeAttentionPooling(6, gate_activation=activation, gate_combination=combination,
                                   effective_normalization='softmax').to(dtype)
    with torch.no_grad():
        pool.gate_proj.weight.normal_(std=.8)
    latent = torch.randn(2, 5, 6, dtype=dtype, requires_grad=True)
    valid = torch.tensor([False, True, True, False, True])
    valid = None if mask_kind == 'none' else torch.stack([valid, valid.roll(1)]) if mask_kind == 'per_trial' else valid
    pooled, weights, _, gate, effective = pool(latent, valid_mask=valid)
    combined = weights * gate if combination == 'multiply' else weights + gate
    logits = combined.float() if dtype == torch.bfloat16 else combined
    if valid is not None:
        logits = logits.masked_fill(~valid, -float('inf'))
    expected = logits.softmax(-1).to(dtype)
    torch.testing.assert_close(effective, expected)
    torch.testing.assert_close(effective.sum(-1), torch.ones(2, dtype=dtype))
    torch.testing.assert_close(pooled, (expected[..., None] * latent).sum(1), rtol=.02 if dtype == torch.bfloat16 else 1e-7, atol=.01 if dtype == torch.bfloat16 else 1e-7)
    grad = torch.autograd.grad(pooled.square().sum(), latent, retain_graph=True)[0]
    reference_pooled = torch.bmm(expected[:, None], latent).squeeze(1)
    reference_grad = torch.autograd.grad(reference_pooled.square().sum(), latent)[0]
    torch.testing.assert_close(grad, reference_grad)
    assert torch.isfinite(grad).all()
    if valid is not None:
        assert torch.count_nonzero(effective.masked_select(~valid.expand_as(effective))) == 0
        assert torch.count_nonzero(grad.masked_select(~valid[..., None].expand_as(grad))) == 0


@pytest.mark.parametrize('masked', [False, True])
def test_zero_combined_weights_become_uniform_on_valid_support(masked):
    pool = GatedTimeAttentionPooling(3, gate_activation='tanh_plus_one', gate_init_bias=-1000.,
                                   effective_normalization='softmax')
    valid = torch.tensor([True, False, True, False, True]) if masked else None
    x = torch.randn(2, 5, 3, requires_grad=True)
    pooled, _, _, gate, effective = pool(x, valid_mask=valid)
    assert torch.count_nonzero(gate) == 0
    expected = valid.float() / valid.sum() if masked else torch.full((5,), .2)
    torch.testing.assert_close(effective, expected.expand(2, -1))
    pooled.sum().backward()
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize('activation', ['sigmoid', 'tanh', 'tanh_plus_one'])
@pytest.mark.parametrize('combination', ['multiply', 'add'])
def test_export_and_plot_identify_positive_final_weights(tmp_path, monkeypatch, activation, combination):
    from matplotlib.figure import Figure
    from memory_reactivation.workflows.report import _plot_attention
    model = make_model(activation, combination).eval()
    with torch.no_grad():
        model.attention.gate_proj.weight.zero_(); model.attention.gate_proj.weight[0, 0] = 1.
        model.attention.gate_proj.bias.zero_()
    latent = torch.zeros(2, 3, 6)
    latent[:, :, 0] = torch.tensor([-3., 0., 3.])
    batch = make_batch(3)
    with patch.object(model.encoder, 'forward_features', return_value={'latent': latent}):
        out = model(batch['eeg'], 'choice')
        export_target_outputs(model, [batch], 'choice', torch.device('cpu'), False,
            tmp_path, 'val', torch.arange(3), [-.01, 0., .01], progress_bar_enabled=False)
    assert 'attention_magnitude_weights' not in out
    with (tmp_path / 'val_attention_weights.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    actual = torch.tensor([float(r['effective_attention_weight']) for r in rows]).reshape(2, 3)
    torch.testing.assert_close(actual, out['effective_attention_weights'])
    assert bool((actual >= 0).all())
    assert all(r['attention_effective_normalization'] == 'softmax' for r in rows)
    assert all('absolute_effective_attention_weight' not in r for r in rows)
    assert all('signed_effective_attention_mass' not in r for r in rows)
    if activation != 'sigmoid' or combination != 'multiply':
        assert all(r['attention_mass_basis'] == 'nonnegative' for r in rows)
    curves = {}
    def save(fig, name, *args, **kwargs):
        curves[Path(name).name] = [np.asarray(line.get_ydata()) for line in fig.axes[0].lines
                                 if line.get_label().startswith('real EEG')]
    monkeypatch.setattr(Figure, 'savefig', save)
    _plot_attention('val', rows, tmp_path)
    combined = curves['03_effective_attention_val.png']
    assert len(combined) == 1  # No duplicate signed/absolute curves after softmax.
    np.testing.assert_allclose(combined[0], actual.mean(0).numpy(), rtol=1e-6)


def test_softmax_respects_builder_and_resume_identity():
    base = {'model': {'encoder': {'in_channels': 3, 'latent_dim': 6, 'num_blocks': 1, 'kernel_size': 3},
             'attention_mode': 'gated_scorer', 'attention_gate_activation': 'tanh_plus_one'}}
    softmax = deepcopy(base)
    softmax['model']['attention_effective_normalization'] = 'softmax'
    model = build_model(softmax['model'])
    restored = build_model(softmax['model']); restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.attention.effective_normalization == 'softmax'
    for mode in ['none', 'one_plus_mass']:
        other = deepcopy(base); other['model']['attention_effective_normalization'] = mode
        with pytest.raises(ValueError, match='attention_effective_normalization'):
            validate_resume_config(other, softmax)


def test_softmax_loss_and_topq_consume_final_weights():
    from memory_reactivation.training.augmentation import EEGInputAugmenter
    model = make_model(pool_valid_context_only=True, attention_score_centering=True, attention_gate_centering=True)
    augmenter = EEGInputAugmenter(sample_interval_s=.01, probability=1., side='left',
        min_duration_s=.02, max_left_duration_s=.02)
    captured = []
    hook = model.register_forward_hook(lambda _m, _i, out: captured.append(out))
    with patch.object(step_module, 'attention_mask_penalty', wraps=training.attention_mask_penalty) as mask:
        metrics = run_target_epoch(model, [make_batch()], 'choice', torch.arange(9),
            {'time': .5, 'pooled': 1., 'attention_mask': .7}, top_k=2, top_ratio=.8,
            label_smoothing=0., device=torch.device('cpu'), amp=False,
            optimizer=torch.optim.AdamW(model.parameters(), lr=2e-5),
            scaler=torch.amp.GradScaler('cuda', enabled=False), eeg_augmenter=augmenter,
            noise_supervision_mode='all_positions', progress_bar_enabled=False)
        torch.testing.assert_close(mask.call_args.args[0], captured[0]['effective_attention_weights'])
        torch.testing.assert_close(captured[0]['time_selection_attention_weights'], captured[0]['effective_attention_weights'])
    hook.remove()
    assert metrics['nonfinite_grad_batches'] == 0
