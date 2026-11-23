"""Gate explorations retain the historical model and handle signed contributions."""
from copy import deepcopy
from pb2023_eeg.training import step as step_module
import csv
from itertools import product
from unittest.mock import patch

import numpy as np
import pytest
import torch

from pb2023_eeg.models.encoder import EEGEncoder
from pb2023_eeg.models.target_decoder import GatedTimeAttentionPooling, TargetAttentionDecoder
from pb2023_eeg.training import target as target_training
from pb2023_eeg.training.augmentation import EEGInputAugmenter
from pb2023_eeg.training.common import validate_resume_config
from pb2023_eeg.workflows.reexport import resolve_model_config_for_checkpoint
from pb2023_eeg.workflows.train import build_model


MODES = [
    ('sigmoid', 'multiply', False),
    ('sigmoid', 'add', False),
    ('sigmoid', 'multiply', True),
    ('sigmoid', 'add', True),
    ('tanh', 'add', False),
    ('tanh', 'multiply', False),
    ('tanh_plus_one', 'multiply', False),
]


def model_for(activation='sigmoid', combination='multiply', **kwargs):
    encoder = EEGEncoder(in_channels=3, latent_dim=6, num_blocks=1,
                         kernel_size=3, dropout=0., expansion_ratio=1)
    return TargetAttentionDecoder(encoder, attention_mode='gated_scorer',
        attention_gate_activation=activation, attention_gate_combination=combination,
        classifier_num_layers=1, **kwargs).eval()


def batch_for(n_times=9):
    return dict(eeg=torch.randn(2, 3, n_times), target=torch.tensor([0, 1]),
        subject_id=torch.tensor([0, 1]), current_stimulus=torch.tensor([0, 0]),
        record_id=torch.tensor([0, 1]), trial0=torch.tensor([0, 0]), lag=torch.tensor([1, 1]),
        exp=torch.tensor([1, 1]))


@pytest.mark.parametrize('activation,combination,centered', MODES)
def test_all_seven_modes_use_post_softmax_gate_and_raw_pooling(activation, combination, centered):
    torch.manual_seed(623)
    model = model_for(activation, combination, attention_gate_centering=centered,
                      attention_score_centering=True, time_no_class_enabled=True)
    with torch.no_grad():
        model.attention.score[-1].weight.normal_()
        model.attention.gate_proj.weight.normal_()
        model.attention.gate_proj.bias.fill_(.2)
    latent = torch.randn(2, 9, 6) + torch.arange(6)[None, None]
    indices = torch.tensor([1, 4, 7])
    raw = latent[:, indices]
    with patch.object(model.encoder, 'forward_features', return_value={'latent': latent}):
        result = model(torch.zeros(2, 3, 9), 'choice', attention_indices=indices)
    gate_input = raw - latent.mean(1, keepdim=True) if centered else raw
    logits = model.attention.gate_proj(gate_input).squeeze(-1)
    gate = torch.sigmoid(logits) if activation == 'sigmoid' else torch.tanh(logits)
    if activation == 'tanh_plus_one':
        gate = gate + 1
    a = torch.softmax(result['attention_scores'], -1)
    effective = a * gate if combination == 'multiply' else a + gate
    torch.testing.assert_close(result['attention_weights'], a)
    torch.testing.assert_close(result['attention_gate'], gate)
    torch.testing.assert_close(result['effective_attention_weights'], effective)
    torch.testing.assert_close(result['pooled_time_latent'], (effective[..., None] * raw).sum(1))
    torch.testing.assert_close(result['time_logits'], model.class_heads['choice'](latent))
    if activation == 'tanh':
        torch.testing.assert_close(result['time_selection_attention_weights'], effective.abs())
    else:
        assert 'attention_magnitude_weights' not in result


@pytest.mark.parametrize('activation,combination', list(product(('sigmoid', 'tanh', 'tanh_plus_one'), ('multiply', 'add'))))
@pytest.mark.parametrize('per_trial', [False, True])
def test_additive_gate_cannot_reactivate_excluded_tokens(activation, combination, per_trial):
    pool = GatedTimeAttentionPooling(6, gate_activation=activation, gate_combination=combination)
    latent = torch.randn(2, 5, 6, requires_grad=True)
    valid = torch.tensor([False, True, True, False, True])
    if per_trial:
        valid = torch.stack([valid, valid.roll(1)])
    pooled, weights, _, _, effective = pool(latent, valid_mask=valid)
    invalid = ~valid.expand_as(weights)
    assert torch.count_nonzero(effective.masked_select(invalid)) == 0
    pooled.sum().backward()
    assert torch.count_nonzero(latent.grad.masked_select(invalid[..., None].expand_as(latent))) == 0


def test_zero_bias_neutral_additive_tanh_and_multiplicative_tanh_plus_one():
    latent = torch.randn(2, 5, 6)
    for activation, combination in [('tanh', 'add'), ('tanh_plus_one', 'multiply')]:
        pool = GatedTimeAttentionPooling(6, gate_activation=activation,
                                        gate_combination=combination, gate_init_bias=0.)
        pooled, weights, _, _, effective = pool(latent)
        torch.testing.assert_close(effective, weights, rtol=0, atol=0)
        torch.testing.assert_close(pooled, latent.mean(1))
    legacy = GatedTimeAttentionPooling(6)
    assert legacy.gate_proj.bias.item() == 2.
    _, weights, _, gate, effective = legacy(latent)
    torch.testing.assert_close(gate, torch.full_like(gate, torch.sigmoid(torch.tensor(2.)).item()))
    torch.testing.assert_close(effective, weights * gate, rtol=0, atol=0)


@pytest.mark.parametrize('combination', ['multiply', 'add'])
def test_strong_negative_contribution_is_selected_and_penalized_toward_zero(combination):
    model = model_for('tanh', combination, attention_gate_init_bias=0.)
    with torch.no_grad():
        model.attention.gate_proj.weight.zero_()
        model.attention.gate_proj.weight[0, 0] = 1.
    latent = torch.zeros(1, 3, 6)
    latent[0, :, 0] = torch.tensor([-3., .1, .05])
    latent.requires_grad_()
    with patch.object(model.encoder, 'forward_features', return_value={'latent': latent}):
        result = model(torch.zeros(1, 3, 3), 'choice')
    signed = result['effective_attention_weights']
    magnitude = result['attention_magnitude_weights']
    assert signed[0, 0] < 0
    assert result['time_selection_attention_weights'].argmax(1).item() == 0
    covered = torch.tensor([[True, False, False]])
    loss = target_training.attention_mask_penalty(magnitude, torch.arange(3), covered)
    assert loss.item() > 0
    gradient = torch.autograd.grad(loss, latent)[0]
    # Gradient descent moves the negative gate logit upward, toward zero contribution.
    assert gradient[0, 0, 0] < 0
    relative = target_training.attention_mask_penalty(magnitude, torch.arange(3), covered,
                                                     mode='relative_effective_mass')
    assert 0 <= relative.item() <= 1


@pytest.mark.parametrize('activation,combination,centered', MODES)
def test_builder_reexport_and_state_layout_remain_compatible(activation, combination, centered):
    cfg = dict(encoder=dict(in_channels=3, latent_dim=6, num_blocks=1),
        attention_mode='gated_scorer', attention_gate_activation=activation,
        attention_gate_combination=combination, attention_gate_centering=centered)
    model = build_model(cfg).eval()
    legacy_cfg = {k: v for k, v in cfg.items() if not k.startswith('attention_gate_')}
    assert set(model.state_dict()) == set(build_model(legacy_cfg).state_dict())
    restored = build_model(resolve_model_config_for_checkpoint(cfg, model.state_dict())).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    eeg = torch.randn(2, 3, 9)
    original, reexported = model(eeg, 'choice'), restored(eeg, 'choice')
    for key in original:
        torch.testing.assert_close(original[key], reexported[key], rtol=0, atol=0)


def test_resume_defaults_are_compatible_and_exploration_changes_are_rejected():
    old = {'model': {'attention_mode': 'gated_scorer'}}
    explicit = deepcopy(old)
    explicit['model'].update(attention_gate_activation='sigmoid', attention_gate_combination='multiply')
    validate_resume_config(old, explicit)
    validate_resume_config(explicit, old)
    for key, value in [('attention_gate_activation', 'tanh'), ('attention_gate_activation', 'tanh_plus_one'),
                       ('attention_gate_combination', 'add')]:
        changed = deepcopy(old); changed['model'][key] = value
        validate_resume_config(changed, deepcopy(changed))
        for a, b in [(old, changed), (changed, old)]:
            with pytest.raises(ValueError, match=key):
                validate_resume_config(a, b)


@pytest.mark.parametrize('key', ['attention_gate_activation', 'attention_gate_combination'])
@pytest.mark.parametrize('value', [None, True, 1, [], 'unknown'])
def test_invalid_gate_options_are_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        build_model({'encoder': {'in_channels': 3, 'latent_dim': 6, 'num_blocks': 1},
                     'attention_mode': 'gated_scorer', key: value})


@pytest.mark.parametrize('attention_mode', ['scorer', 'query', 'gated_query'])
def test_nondefault_gate_options_cannot_be_silently_ignored(attention_mode):
    for options in [dict(attention_gate_activation='tanh'), dict(attention_gate_combination='add')]:
        with pytest.raises(ValueError, match='require gated_scorer'):
            build_model(dict(encoder=dict(in_channels=3, latent_dim=6, num_blocks=1),
                             attention_mode=attention_mode, **options))


@pytest.mark.parametrize('combination', ['multiply', 'add'])
@pytest.mark.parametrize('amp', [False, True])
def test_signed_training_routes_magnitudes_to_auxiliary_losses(combination, amp):
    torch.manual_seed(527)
    model = model_for('tanh', combination, attention_gate_init_bias=-2., time_no_class_enabled=True)
    data = batch_for()
    augmenter = EEGInputAugmenter(sample_interval_s=.01, probability=1., side='left',
                                  min_duration_s=.02, max_left_duration_s=.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    mask = target_training.attention_mask_penalty
    smooth = target_training.attention_smoothness_penalty
    with patch.object(step_module, 'attention_mask_penalty', wraps=mask) as mask_spy, \
         patch.object(step_module, 'attention_smoothness_penalty', wraps=smooth) as smooth_spy:
        metrics = target_training.run_target_epoch(model, [data], 'choice', torch.arange(9),
            {'time': .5, 'pooled': 1., 'attention_mask': .7, 'attention_smoothness': .01,
             'null_attention': .01, 'att_time_consistency': .01},
            top_k=2, top_ratio=.8, label_smoothing=0., device=torch.device('cpu'), amp=amp,
            optimizer=optimizer, scaler=torch.amp.GradScaler('cuda', enabled=False),
            eeg_augmenter=augmenter, noise_supervision_mode='all_positions', progress_bar_enabled=False)
        assert mask_spy.call_args.args[0].min().item() >= 0
        assert smooth_spy.call_args.args[0].min().item() >= 0
    assert metrics['attention_mask_loss'] > 0
    assert metrics['nonfinite_grad_batches'] == 0
    assert np.isfinite(metrics['loss'])
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('combination', ['multiply', 'add'])
def test_signed_exports_keep_both_sign_and_non_cancelling_mass(tmp_path, combination):
    model = model_for('tanh', combination, attention_gate_init_bias=0.)
    with torch.no_grad():
        model.attention.gate_proj.weight.zero_(); model.attention.gate_proj.weight[0, 0] = 1.
    latent = torch.zeros(2, 3, 6)
    latent[:, :, 0] = torch.tensor([-3., 3., 0.])
    data = batch_for(3)
    with patch.object(model.encoder, 'forward_features', return_value={'latent': latent}):
        target_training.export_target_outputs(model, [data], 'choice', torch.device('cpu'), False,
            tmp_path, 'val', torch.arange(3), [-.01, 0., .01], progress_bar_enabled=False)
    with (tmp_path / 'val_attention_weights.csv').open() as handle:
        rows = list(csv.DictReader(handle))[:3]
    signed = np.array([float(r['effective_attention_weight']) for r in rows])
    absolute = np.array([float(r['absolute_effective_attention_weight']) for r in rows])
    normalized = np.array([float(r['normalized_effective_attention_weight']) for r in rows])
    assert signed.min() < 0
    np.testing.assert_allclose(absolute, np.abs(signed))
    np.testing.assert_allclose(normalized, absolute / absolute.sum())
    assert normalized.sum() == pytest.approx(1.)
    assert float(rows[0]['effective_attention_mass']) == pytest.approx(absolute.sum())
    assert float(rows[0]['signed_effective_attention_mass']) == pytest.approx(signed.sum())
    assert rows[0]['attention_mass_basis'] == 'absolute'
    assert rows[0]['attention_gate_combination_stage'] == 'post_softmax'
    assert all(float(r['time_selection_attention_weight']) == pytest.approx(abs(float(r['effective_attention_weight']))) for r in rows)


def test_signed_mass_and_gate_plots_do_not_clip_negative_values(tmp_path, monkeypatch):
    from matplotlib.figure import Figure
    from pb2023_eeg.workflows.report import _plot_attention, _subject_time_summary

    rows = [dict(input_kind='real', subject_id=0, record_id=0, trial0=0, lag=1,
        time_index=i, time_s=i * .01, attention_weight=1 / 3,
        attention_gate=gate, effective_attention_weight=weight,
        absolute_effective_attention_weight=abs(weight), effective_attention_mass=1.,
        normalized_effective_attention_weight=abs(weight))
        for i, (gate, weight) in enumerate([(-.9, -.3), (.9, .3), (-.6, -.4)])]
    captured = {}

    def save(fig, filename, *args, **kwargs):
        captured[str(filename).split('/')[-1]] = (fig.axes[0].get_ylim(),
            [line.get_ydata() for line in fig.axes[0].lines])

    monkeypatch.setattr(Figure, 'savefig', save)
    summary = _subject_time_summary(rows)
    assert [r['mean_absolute_effective_attention'] for r in summary] == pytest.approx([.3, .3, .4])
    _plot_attention('val', rows, tmp_path)
    assert captured['03_effective_attention_val.png'][0][0] < -.4
    assert captured['02_attention_gate_val.png'][0][0] < -.9
    np.testing.assert_allclose(captured['03_effective_attention_val.png'][1][0], [-.3, .3, -.4])
    np.testing.assert_allclose(captured['03_effective_attention_val.png'][1][1], [.3, .3, .4])


def test_curve_diagnostics_use_trialwise_absolute_mass_before_averaging():
    from pb2023_eeg.workflows.report import _attention_curve_diagnostics, _subject_time_summary

    rows = []
    # Opposite trial contributions cancel at each time, but both trials have mass .6.
    for trial, weights in enumerate([[-.3, .3], [.3, -.3]]):
        for time_index, weight in enumerate(weights):
            rows.append(dict(input_kind='real', subject_id=0, record_id=0, trial0=trial,
                lag=1, time_index=time_index, time_s=time_index * .01,
                attention_weight=.5, attention_gate=2 * weight,
                effective_attention_weight=weight, absolute_effective_attention_weight=abs(weight),
                normalized_effective_attention_weight=.5))
    diagnostic = _attention_curve_diagnostics('val', _subject_time_summary(rows))[0]
    assert diagnostic['mean_effective_attention_mass'] == pytest.approx(.6)
    assert diagnostic['mean_signed_effective_attention_mass'] == pytest.approx(0.)
    assert diagnostic['attention_mass_basis'] == 'absolute'


@pytest.mark.parametrize('activation,bias', [('tanh', 0.), ('tanh_plus_one', -20.), ('sigmoid', -20.)])
def test_fp16_zero_gate_consistency_loss_and_gradients_are_finite(activation, bias):
    torch.manual_seed(659)
    model = model_for(activation, 'multiply', attention_gate_init_bias=bias,
                      time_no_class_enabled=True).half()
    latent = torch.randn(2, 5, 6, dtype=torch.float16)
    batch = batch_for(5)
    with patch.object(model.encoder, 'forward_features', return_value={'latent': latent}):
        metrics = target_training.run_target_epoch(model, [batch], 'choice', torch.arange(5),
            {'time': .5, 'pooled': 1., 'att_time_consistency': .1},
            top_k=3, top_ratio=.8, label_smoothing=0., device=torch.device('cpu'), amp=False,
            optimizer=torch.optim.SGD(model.parameters(), lr=2e-5),
            scaler=torch.amp.GradScaler('cuda', enabled=False), progress_bar_enabled=False)
    assert np.isfinite(metrics['loss'])
    assert metrics['nonfinite_grad_batches'] == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
