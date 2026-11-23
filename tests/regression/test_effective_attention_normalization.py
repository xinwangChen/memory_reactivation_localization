"""Effective-weight scaling preserves suppression and reaches every consumer."""
from copy import deepcopy
from pb2023_eeg.training import step as step_module
import csv
from unittest.mock import patch

import pytest
import torch

from pb2023_eeg.models.encoder import EEGEncoder
from pb2023_eeg.models.target_decoder import GatedTimeAttentionPooling, TargetAttentionDecoder
from pb2023_eeg.training.common import validate_resume_config
from pb2023_eeg.training.target import export_target_outputs, run_target_epoch
from pb2023_eeg.training import trainer as training
from pb2023_eeg.workflows.train import build_model


def make_model(**options):
    encoder = EEGEncoder(in_channels=3, latent_dim=6, num_blocks=1,
                         kernel_size=3, dropout=0., expansion_ratio=1)
    return TargetAttentionDecoder(encoder, attention_mode='gated_scorer',
        attention_gate_activation='tanh_plus_one', classifier_num_layers=1,
        time_no_class_enabled=True, **options)


def make_batch():
    return dict(eeg=torch.randn(2, 3, 9), target=torch.tensor([0, 1]),
        subject_id=torch.tensor([0, 1]), current_stimulus=torch.tensor([0, 0]),
        record_id=torch.tensor([0, 1]), trial0=torch.tensor([0, 0]),
        lag=torch.tensor([1, 1]), exp=torch.tensor([1, 1]))


@pytest.mark.parametrize('activation', ['sigmoid', 'tanh_plus_one'])
@pytest.mark.parametrize('combination', ['multiply', 'add'])
@pytest.mark.parametrize('dtype', [torch.float64, torch.float32, torch.bfloat16])
@pytest.mark.parametrize('mask_kind', ['none', 'shared', 'per_trial'])
def test_actual_pooling_and_gradients_follow_one_plus_mass(activation, combination, dtype, mask_kind):
    torch.manual_seed(118)
    pool = GatedTimeAttentionPooling(6, gate_activation=activation, gate_combination=combination,
                                   effective_normalization='one_plus_mass').to(dtype)
    with torch.no_grad():
        pool.gate_proj.weight.normal_(std=.4)
    latent = torch.randn(2, 5, 6, dtype=dtype, requires_grad=True)
    valid = torch.tensor([False, True, True, False, True])
    valid = None if mask_kind == 'none' else torch.stack([valid, valid.roll(1)]) if mask_kind == 'per_trial' else valid
    pooled, weights, _, gate, effective = pool(latent, valid_mask=valid)
    raw = weights * gate if combination == 'multiply' else weights + gate
    if valid is not None:
        raw = raw.masked_fill(~valid, 0)
    reference = raw.float() if dtype == torch.bfloat16 else raw
    expected = (reference / (1 + reference.sum(-1, keepdim=True))).to(dtype)
    torch.testing.assert_close(effective, expected)
    torch.testing.assert_close(pooled, torch.bmm(expected[:, None], latent).squeeze(1))
    actual_grad = torch.autograd.grad(pooled.square().sum(), latent, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(torch.bmm(expected[:, None], latent).square().sum(), latent)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.isfinite(actual_grad).all()
    if valid is not None:
        assert torch.count_nonzero(effective.masked_select(~valid.expand_as(effective))) == 0


def test_zero_gate_stays_zero_with_finite_backward():
    pool = GatedTimeAttentionPooling(3, gate_activation='tanh_plus_one', gate_init_bias=-1000,
                                   effective_normalization='one_plus_mass')
    latent = torch.randn(2, 7, 3, requires_grad=True)
    pooled, _, _, _, effective = pool(latent)
    assert torch.count_nonzero(effective) == 0
    assert torch.count_nonzero(pooled) == 0
    pooled.sum().backward()
    assert torch.isfinite(latent.grad).all()


@pytest.mark.parametrize('centered', [False, True])
def test_raw_values_time_head_and_topq_support_remain_aligned(centered):
    model = make_model(attention_score_centering=True, attention_gate_centering=centered,
        attention_effective_normalization='one_plus_mass', pool_valid_context_only=True).double().eval()
    with torch.no_grad():
        model.attention.gate_proj.weight.normal_()
    latent = torch.randn(2, 9, 6, dtype=torch.float64)
    with patch.object(model.encoder, 'forward_features', return_value={'latent': latent}):
        out = model(torch.zeros(2, 3, 9, dtype=torch.float64), 'choice')
    pre = out['pre_normalization_effective_attention_weights']
    expected = pre / (1 + pre.sum(-1, keepdim=True))
    torch.testing.assert_close(out['effective_attention_weights'], expected)
    torch.testing.assert_close(out['time_selection_attention_weights'], expected)
    torch.testing.assert_close(out['pooled_time_latent'], (expected[..., None] * latent).sum(1))
    torch.testing.assert_close(out['time_logits'], model.class_heads['choice'](latent))


@pytest.mark.parametrize('value', [None, True, False, 'log_softmax', 'unknown', 1])
def test_invalid_normalization_is_rejected(value):
    with pytest.raises(ValueError, match='attention_effective_normalization'):
        make_model(attention_effective_normalization=value)


def test_signed_or_non_gated_normalization_is_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        GatedTimeAttentionPooling(3, gate_activation='tanh', effective_normalization='one_plus_mass')
    with pytest.raises(ValueError, match='requires gated_scorer'):
        TargetAttentionDecoder(EEGEncoder(in_channels=3, latent_dim=6, num_blocks=1, kernel_size=3),
            attention_mode='scorer', attention_effective_normalization='one_plus_mass')


def test_builder_resume_identity_and_reexport_roundtrip():
    config = {'encoder': {'in_channels': 3, 'latent_dim': 6, 'num_blocks': 1, 'kernel_size': 3},
              'attention_mode': 'gated_scorer', 'attention_gate_activation': 'tanh_plus_one'}
    old = {'model': config}
    disabled = deepcopy(old)
    disabled['model']['attention_effective_normalization'] = 'none'
    validate_resume_config(old, disabled)
    enabled = deepcopy(old)
    enabled['model']['attention_effective_normalization'] = 'one_plus_mass'
    model = build_model(enabled['model'])
    assert model.attention.effective_normalization == 'one_plus_mass'
    restored = build_model(enabled['model'])
    restored.load_state_dict(model.state_dict(), strict=True)
    with pytest.raises(ValueError, match='attention_effective_normalization'):
        validate_resume_config(old, enabled)


def test_exports_store_actual_weights_and_pre_scaling_provenance(tmp_path):
    model = make_model(attention_effective_normalization='one_plus_mass').eval()
    batch = make_batch()
    out = model(batch['eeg'], 'choice')
    export_target_outputs(model, [batch], 'choice', torch.device('cpu'), False,
        tmp_path, 'val', torch.arange(9), list(range(9)), progress_bar_enabled=False)
    with (tmp_path / 'val_attention_weights.csv').open() as f:
        rows = list(csv.DictReader(f))
    actual = torch.tensor([float(r['effective_attention_weight']) for r in rows]).reshape(2, 9)
    pre = torch.tensor([float(r['pre_normalization_effective_attention_weight']) for r in rows]).reshape(2, 9)
    torch.testing.assert_close(actual, out['effective_attention_weights'])
    torch.testing.assert_close(pre, out['pre_normalization_effective_attention_weights'])
    torch.testing.assert_close(actual.sum(-1), pre.sum(-1) / (1 + pre.sum(-1)))
    assert all(r['attention_effective_normalization'] == 'one_plus_mass' for r in rows)
    assert float(rows[0]['pre_normalization_effective_attention_mass']) == pytest.approx(pre[0].sum().item())


def test_training_auxiliary_losses_consume_actual_scaled_weights():
    from pb2023_eeg.training.augmentation import EEGInputAugmenter
    model = make_model(attention_effective_normalization='one_plus_mass')
    augmenter = EEGInputAugmenter(sample_interval_s=.01, probability=1., side='left',
        min_duration_s=.02, max_left_duration_s=.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    captured = []
    hook = model.register_forward_hook(lambda _m, _i, out: captured.append(out))
    with patch.object(step_module, 'attention_mask_penalty', wraps=training.attention_mask_penalty) as spy:
        metrics = run_target_epoch(model, [make_batch()], 'choice', torch.arange(9),
            {'time': .5, 'pooled': 1., 'attention_mask': .7}, top_k=2, top_ratio=.8,
            label_smoothing=0., device=torch.device('cpu'), amp=False, optimizer=optimizer,
            scaler=torch.amp.GradScaler('cuda', enabled=False), eeg_augmenter=augmenter,
            noise_supervision_mode='all_positions', progress_bar_enabled=False)
        torch.testing.assert_close(spy.call_args.args[0], captured[0]['effective_attention_weights'])
    hook.remove()
    assert metrics['nonfinite_grad_batches'] == 0
    assert torch.isfinite(torch.tensor(metrics['loss']))
