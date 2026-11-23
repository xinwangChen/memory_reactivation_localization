"""One-sided mass gradients, full-noise routing, AMP and training/resume wiring."""
from copy import deepcopy
import csv
import math

import pytest
import torch
from torch import nn
from regression.helpers_smoke import target_test_config

from memory_reactivation.training.attention_regularization import (
    attention_mass_penalty,
    normalize_attention_mass_penalty_config,
)
from memory_reactivation.training.augmentation import TemporalCropNoiseResult
from memory_reactivation.training.common import load_checkpoint, validate_resume_config
from memory_reactivation.training import trainer as training
from memory_reactivation.training.swanlab_logger import target_swanlab_metrics
from memory_reactivation.workflows.train import build_model


@pytest.mark.parametrize('value', [None, True, False, 'bad', -.1, float('nan'), float('inf'), 10 ** 400])
def test_invalid_weight_is_rejected(value):
    with pytest.raises(ValueError, match='attention_mass_penalty_weight'):
        normalize_attention_mass_penalty_config({'attention_mass_penalty_weight': value})


def test_default_is_disabled_and_weight_is_independent():
    assert normalize_attention_mass_penalty_config({}) == {'weight': 0.}
    assert normalize_attention_mass_penalty_config({'attention_mass_penalty_weight': '0.7',
        'attention_mask_penalty_weight': .3, 'clean_attention_penalty_weight': .5}) == {'weight': .7}


def test_exact_per_trial_loss_and_gradients_above_at_and_below_one():
    weights = torch.tensor([[.125] * 4, [.25] * 4, [.375] * 4], dtype=torch.float64, requires_grad=True)
    rng = torch.get_rng_state()
    loss = attention_mass_penalty(weights)
    assert loss.item() == pytest.approx(.25 / 3)
    loss.backward()
    expected = torch.tensor([[0.] * 4, [0.] * 4, [1 / 3] * 4], dtype=torch.float64)
    torch.testing.assert_close(weights.grad, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_trials_cannot_cancel_excess_mass_and_signed_modes_use_magnitude():
    weights = torch.tensor([[0., 0.], [1., -1.]], requires_grad=True)
    loss = attention_mass_penalty(weights)
    assert loss.item() == .5  # Mean masses equal 1, but the second trial exceeds 1.
    loss.backward()
    torch.testing.assert_close(weights.grad, torch.tensor([[0., 0.], [1., -1.]]))


@pytest.mark.parametrize('per_trial', [False, True])
def test_only_valid_pooling_support_contributes(per_trial):
    weights = torch.tensor([[.75, 100., .75], [.75, 100., .75]], requires_grad=True)
    valid = torch.tensor([[True, False, True], [False, False, True]]) if per_trial else torch.tensor([True, False, True])
    loss = attention_mass_penalty(weights, valid_mask=valid)
    assert loss.item() == pytest.approx(.125 if per_trial else .25)
    loss.backward()
    expected = torch.tensor([[.5, 0., .5], [0., 0., 0.]]) if per_trial else torch.tensor([[.5, 0., .5]] * 2)
    torch.testing.assert_close(weights.grad, expected, rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(0, 5), (2, 0), (0, 0)])
def test_empty_support_has_finite_graph_connected_zero(shape):
    weights = torch.empty(shape, requires_grad=True)
    loss = attention_mass_penalty(weights)
    assert loss.item() == 0.
    loss.backward()
    assert weights.grad is not None and weights.grad.numel() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_reduction_and_square_do_not_overflow_in_half_precision(device, dtype):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    weights = torch.full((275, 256), 1.9, dtype=dtype, device=device).T.requires_grad_()
    valid = torch.ones(275, dtype=torch.bool, device=device)
    valid[:25] = False
    rng = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
    loss = attention_mass_penalty(weights, valid_mask=valid)
    expected = (float(weights[0, 0]) * 250 - 1) ** 2
    assert torch.isfinite(loss) and loss.item() == pytest.approx(expected, rel=1e-5)
    assert loss.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    loss.backward()
    assert torch.isfinite(weights.grad).all() and torch.count_nonzero(weights.grad[:, :25]) == 0
    assert torch.equal(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state(), rng)
    # Masked-out half-precision batch sums must never be used even on a zero path.
    weights.grad = None
    zero = attention_mass_penalty(weights, valid_mask=torch.zeros_like(valid))
    assert zero.item() == 0.
    zero.backward()
    assert torch.count_nonzero(weights.grad) == 0


@pytest.mark.parametrize('problem', ['rank', 'dtype', 'mask_shape', 'mask_batch', 'mask_dtype'])
def test_invalid_tensor_layouts_are_rejected(problem):
    weights = torch.ones(2, 3)
    valid = None
    if problem == 'rank': weights = weights[:, :, None]
    elif problem == 'dtype': weights = weights.long()
    elif problem == 'mask_shape': valid = torch.ones(2, dtype=torch.bool)
    elif problem == 'mask_batch': valid = torch.ones(3, 3, dtype=torch.bool)
    elif problem == 'mask_dtype': valid = torch.ones(3)
    with pytest.raises(ValueError, match='attention mass'):
        attention_mass_penalty(weights, valid_mask=valid)


def test_masked_double_precision_gradient_passes_gradcheck():
    weights = torch.tensor([[.5, .75, -100.], [.125, .25, -100.]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda w: attention_mass_penalty(w,
        valid_mask=torch.tensor([True, True, False])), (weights,))


class Probe(nn.Module):
    fusion_enabled = False

    def __init__(self, valid=None):
        super().__init__()
        self.weights = nn.Parameter(torch.tensor([[.125] * 6, [.5] * 6, [.25] * 6, [.25] * 6, [.25] * 6]))
        self.time = nn.Parameter(torch.zeros(5, 6, 3))
        self.pooled = nn.Parameter(torch.zeros(5, 2))
        self.register_buffer('valid', valid)

    def time_class_count(self, task): return 3
    def time_no_class_index(self, task): return 2

    def forward(self, eeg, attention_indices, **kwargs):
        n = eeg.shape[0]
        weights = self.weights[:n].index_select(1, attention_indices)
        if self.valid is not None: weights = weights.masked_fill(~self.valid, 0.)
        return dict(time_logits=self.time[:n], pooled_logits=self.pooled[:n],
            attention_weights=torch.ones_like(weights) / weights.shape[1],
            effective_attention_weights=weights, attention_valid_mask=self.valid)


class AllKindsAugmentation(nn.Module):
    def forward(self, eeg, return_mask=False):
        assert eeg.shape[0] == 5
        covered = torch.zeros(5, 6, dtype=torch.bool, device=eeg.device)
        covered[1, :2] = True   # local replacement
        covered[3, :5] = True   # near-full replacement
        covered[4, :] = True    # full replacement
        return TemporalCropNoiseResult(eeg, ~covered, covered,
            torch.tensor([False, False, False, True, True], device=eeg.device),
            torch.tensor([False, False, True, False, False], device=eeg.device),
            torch.tensor([0, 0, 1, 0, 0], device=eeg.device))


def batch(n=5):
    return dict(eeg=torch.randn(n, 2, 6), target=torch.tensor([0, 1, 0, 1, 0])[:n],
        sample_weight=torch.tensor([1., 5., 10., 20., 40.])[:n])


def epoch(model, batches=None, training_mode=True, weight=.7, **options):
    return training.run_target_epoch(model, batches or [batch()], 'choice',
        options.pop('attention_indices', torch.arange(6)),
        options.pop('loss_weights', {'time': .5, 'pooled': 1., 'attention_mask': .7, 'attention_mass': weight}),
        None, options.pop('top_ratio', .2), 0., torch.device('cpu'), False,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.) if training_mode else None,
        scaler=torch.amp.GradScaler('cpu', enabled=False) if training_mode else None,
        noise_supervision_mode='all_positions', progress_bar_enabled=False, **options)


def test_all_augmentation_routes_participate_without_class_or_topq_weights():
    before_model = Probe()
    after_model = deepcopy(before_model)
    examples = [batch()]
    opts = dict(eeg_augmenter=AllKindsAugmentation(), near_full_noise_time_loss_scale=.25)
    before = epoch(before_model, examples, weight=0., **opts)
    after = epoch(after_model, examples, weight=.7, **opts)
    expected = (0. + 4. + .25 + .25 + .25) / 5
    assert after['attention_mass_loss'] == pytest.approx(expected)
    assert after['loss'] - before['loss'] == pytest.approx(.7 * expected)
    expected_gradient = torch.tensor([[0.] * 6, [.56] * 6, [.14] * 6, [.14] * 6, [.14] * 6])
    torch.testing.assert_close(after_model.weights.grad - before_model.weights.grad, expected_gradient)
    torch.testing.assert_close(before_model.time.grad, after_model.time.grad, rtol=0, atol=0)
    torch.testing.assert_close(before_model.pooled.grad, after_model.pooled.grad, rtol=0, atol=0)
    for key in ['time_loss', 'pooled_loss', 'attention_mask_loss', 'trial_loss_sample_count', 'near_full_noise_sample_count']:
        assert before[key] == after[key]
    assert after['trial_loss_sample_count'] == 3 and after['near_full_noise_sample_count'] == 2
    assert after['clean_attention_loss'] == 0.


def test_entire_full_noise_batch_keeps_mass_gradients_with_all_classification_excluded():
    class FullNoise(nn.Module):
        def forward(self, eeg, return_mask=False):
            covered = torch.ones(eeg.shape[0], eeg.shape[-1], dtype=torch.bool)
            return TemporalCropNoiseResult(eeg, ~covered, covered,
                torch.ones(eeg.shape[0], dtype=torch.bool))

    model = Probe()
    result = epoch(model, [batch()], eeg_augmenter=FullNoise(), near_full_noise_time_loss_scale=0.,
        loss_weights={'time': .5, 'pooled': 1., 'attention_mask': 0., 'attention_mass': .7})
    assert result['trial_loss_sample_count'] == 0 and result['near_full_noise_sample_count'] == 5
    assert math.isnan(result['pooled_accuracy'])
    assert result['time_loss'] == result['pooled_loss'] == 0.
    assert result['attention_mass_loss'] == pytest.approx(.95)
    assert result['loss'] == pytest.approx(.7 * .95)
    expected = torch.tensor([[0.] * 6, [.56] * 6, [.14] * 6, [.14] * 6, [.14] * 6])
    torch.testing.assert_close(model.weights.grad, expected)
    assert torch.count_nonzero(model.time.grad) == torch.count_nonzero(model.pooled.grad) == 0


def test_clean_and_mass_penalties_keep_independent_coefficients_and_add_gradients_once():
    examples = [batch()]
    models = [Probe() for _ in range(3)]
    outputs = [epoch(model, examples,
        loss_weights={'time': 0., 'pooled': 0., 'clean_attention': clean, 'attention_mass': mass},
        clean_attention_penalty_probability=1.)
        for model, (clean, mass) in zip(models, [(0.3, 0.), (0., .7), (.3, .7)])]
    clean_only, mass_only, combined = outputs
    assert combined['clean_attention_loss'] == pytest.approx(1.65)
    assert combined['attention_mass_loss'] == pytest.approx(.95)
    assert combined['loss'] == pytest.approx(.3 * 1.65 + .7 * .95)
    assert combined['loss'] == pytest.approx(clean_only['loss'] + mass_only['loss'])
    torch.testing.assert_close(models[2].weights.grad, models[0].weights.grad + models[1].weights.grad)


def test_actual_rf_aware_crop_uses_pooling_support_instead_of_clean_or_topq_support():
    class LocalNoise(nn.Module):
        def forward(self, eeg, return_mask=False):
            covered = torch.zeros(eeg.shape[0], eeg.shape[-1], dtype=torch.bool)
            covered[:, 4] = True
            return TemporalCropNoiseResult(eeg, ~covered, covered)

    model_cfg = dict(encoder=dict(in_channels=2, latent_dim=4, num_blocks=1, kernel_size=3,
        expansion_ratio=1), attention_mode='gated_scorer', attention_gate_activation='tanh_plus_one',
        attention_gate_init_bias=1., pool_valid_context_only=True,
        classifier_num_layers=1, classifier_dropout=0.)
    model = build_model(model_cfg, time_point_count=9)
    captured = []
    handle = model.register_forward_hook(lambda _m, _i, out: captured.append(out))
    example = dict(eeg=torch.randn(2, 2, 9), target=torch.tensor([0, 1]))
    result = training.run_target_epoch(model, [example], 'choice', torch.tensor([0, 1, 2, 4, 5, 7, 8]),
        {'time': .5, 'pooled': 1., 'attention_mass': .7}, None, .2, 0., torch.device('cpu'), False,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.),
        scaler=torch.amp.GradScaler('cpu', enabled=False), eeg_augmenter=LocalNoise(),
        noise_supervision_mode='rf_aware', progress_bar_enabled=False)
    handle.remove()
    support = captured[0]['attention_valid_mask']
    assert not support[0] and not support[-1] and support[1:-1].all()
    assert result['selected_time_count_per_sample'] < int(support.sum())
    # Constant tanh+1 gate and softmax on any valid support give M=1+tanh(1).
    # Removing noisy/halo or non-Top-q points from this sum would give a smaller M.
    assert result['attention_mass_loss'] == pytest.approx(math.tanh(1.) ** 2, rel=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('per_trial', [False, True])
def test_cpu_valid_support_can_be_applied_to_cuda_weights(per_trial):
    valid = torch.tensor([[True, False, True], [True, False, False]]) if per_trial else torch.tensor([True, False, True])
    weights = torch.full((2, 3), .75, device='cuda', requires_grad=True)
    loss = attention_mass_penalty(weights, valid_mask=valid)
    assert loss.item() == pytest.approx(.125 if per_trial else .25)
    loss.backward()
    assert weights.grad.is_cuda and torch.isfinite(weights.grad).all()
    assert weights.grad[:, 1].count_nonzero() == 0


def test_epoch_cropped_support_masks_and_unequal_batches_use_trial_mean():
    model = Probe(valid=torch.tensor([True, False, True]))
    result = epoch(model, [batch(), batch(1)], attention_indices=torch.tensor([1, 3, 5]),
        loss_weights={'time': 0., 'pooled': 0., 'attention_mass': 2.})
    # Cropped valid sums are .25, 1, .5, .5, .5; no trial exceeds the ceiling.
    assert result['attention_mass_loss'] == result['loss'] == 0.
    model = Probe()
    result = epoch(model, [batch(), batch(1)], loss_weights={'time': 0., 'pooled': 0., 'attention_mass': 2.})
    assert result['attention_mass_loss'] == pytest.approx((4 + .25 * 3) / 6)
    assert result['loss'] == pytest.approx(2 * (4 + .25 * 3) / 6)


@pytest.mark.parametrize('validation', [False, True])
def test_disabled_and_validation_paths_do_not_evaluate_new_penalty(monkeypatch, validation):
    def must_not_run(*args, **kwargs): raise AssertionError('Unexpected mass penalty computation')
    monkeypatch.setattr('memory_reactivation.training.step.attention_mass_penalty', must_not_run)
    examples = [batch()]
    rng = torch.get_rng_state()
    result = epoch(Probe(), examples, training_mode=not validation, weight=.7 if validation else 0.)
    assert result['attention_mass_loss'] == 0.
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('activation,combination,normalization', [
    ('sigmoid', 'multiply', 'none'), ('sigmoid', 'add', 'none'),
    ('tanh', 'multiply', 'none'), ('tanh', 'add', 'none'),
    ('tanh_plus_one', 'multiply', 'none'), ('tanh_plus_one', 'multiply', 'one_plus_mass'),
    ('tanh_plus_one', 'multiply', 'softmax'),
])
def test_penalty_reads_final_effective_weights(activation, combination, normalization):
    cfg = dict(encoder=dict(in_channels=2, latent_dim=4, num_blocks=1, kernel_size=3),
        attention_mode='gated_scorer', attention_gate_activation=activation,
        attention_gate_combination=combination, attention_effective_normalization=normalization,
        attention_gate_init_bias=-1. if activation == 'tanh' else 1.,
        time_no_class_enabled=True, classifier_num_layers=1, classifier_dropout=0.)
    model = build_model(cfg, time_point_count=6)
    captured = []
    handle = model.register_forward_hook(lambda _m, _i, out: captured.append(out))
    result = epoch(model)
    handle.remove()
    masses = captured[0]['effective_attention_weights'].detach().double().abs().sum(-1).tolist()
    expected = sum(max(0., m - 1) ** 2 for m in masses) / len(masses)
    assert result['attention_mass_loss'] == pytest.approx(expected, abs=1e-7, rel=1e-5)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_mass_gradient_reaches_gate_scorer_and_encoder_without_changing_forward():
    torch.manual_seed(421)
    cfg = dict(encoder=dict(in_channels=2, latent_dim=4, num_blocks=1, kernel_size=3),
        attention_mode='gated_scorer', attention_gate_activation='tanh_plus_one',
        attention_score_centering=True, attention_gate_init_bias=1.,
        time_no_class_enabled=True, classifier_num_layers=1, classifier_dropout=0.)
    model = build_model(cfg, time_point_count=6)
    with torch.no_grad(): model.attention.gate_proj.weight.normal_(0., .2)
    examples = [batch()]
    before = deepcopy(model.state_dict())
    result = epoch(model, examples, loss_weights={'time': 0., 'pooled': 0., 'attention_mass': 1.})
    assert result['attention_mass_loss'] > 0.
    for name in ['attention.gate_proj.weight', 'attention.score.4.weight', 'encoder.spatial_projection.0.weight']:
        parameter = dict(model.named_parameters())[name]
        assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
    for name, value in before.items(): torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize('task', ['choice', 'pitch'])
@pytest.mark.parametrize('protocol', ['block', 'subject'])
def test_target_variants_fit_full275_and_record_loss_config_and_checkpoint(tmp_path, task, protocol):
    config = target_test_config(task, protocol)
    config['model']['subject_conditioning']['resolved_training_subject_ids'] = list(range(24))
    config['train'].update(iterations=2, validation_interval_iterations=1, log_interval_iterations=1,
        lr_schedule_iterations=2, warmup_iterations=0, progress_bar=False, amp=False,
        attention_mass_penalty_weight=.3, clean_attention_penalty_weight=0.)
    config['train']['temporal_crop_noise']['sample_interval_s'] = .01
    model = build_model(config['model'], n_subjects=30, time_point_count=275, input_channels=2)
    ids = torch.tensor([24, 25]) if protocol == 'subject' else torch.tensor([0, 1])
    example = dict(eeg=torch.randn(2, 2, 275), target=torch.tensor([0, 1]), subject_id=ids,
        current_stimulus=torch.tensor([0, 0]), exp=torch.ones(2, dtype=torch.long), trial0=torch.arange(2))
    history = training.fit_target_decoder(model, [example], [example], task, torch.arange(275),
        config, tmp_path, torch.device('cpu'))
    assert len(history) == 2
    for row in history:
        assert row['attention_mass_penalty_weight'] == .3
        assert row['train_attention_mass_loss'] > 0. and math.isfinite(row['train_loss'])
        assert row['val_attention_mass_loss'] == row['train_clean_attention_loss'] == 0.
    for filename in ['metrics.csv', 'train_metrics.csv']:
        rows = list(csv.DictReader((tmp_path / filename).open()))
        assert float(rows[-1]['attention_mass_penalty_weight']) == .3
        assert float(rows[-1]['train_attention_mass_loss']) > 0.
    saved = load_checkpoint(tmp_path / 'last.ckpt')
    assert saved['config']['train']['attention_mass_penalty_weight'] == .3
    assert saved['selection_metric'] == 'val_pooled_accuracy'


def test_resume_identity_keeps_old_defaults_and_rejects_enabled_weight_changes():
    legacy = {'train': {}}
    validate_resume_config(legacy, {'train': {'attention_mass_penalty_weight': 0.}})
    enabled = {'train': {'attention_mass_penalty_weight': .3}}
    validate_resume_config(enabled, {'train': {'attention_mass_penalty_weight': '0.3'}})
    for before, after in [(legacy, enabled), (enabled, legacy),
            (enabled, {'train': {'attention_mass_penalty_weight': .7}})]:
        with pytest.raises(ValueError, match='attention_mass_penalty_weight'):
            validate_resume_config(before, after)


@pytest.mark.parametrize('encoder_type', ['cnn', 'window_transformer'])
def test_active_mass_penalty_resumes_exactly(tmp_path, monkeypatch, encoder_type):
    encoder = dict(type=encoder_type, in_channels=2, latent_dim=4, dropout=.1)
    if encoder_type == 'cnn': encoder.update(num_blocks=1, kernel_size=3, expansion_ratio=1)
    else: encoder.update(num_layers=1, window_size=3, num_heads=2, ffn_ratio=2,
        attention_pattern='centered_local', position_encoding_mode='sinusoidal_qk')
    model_cfg = dict(encoder=encoder, attention_mode='gated_scorer', attention_dim=4,
        attention_gate_activation='tanh_plus_one', attention_score_centering=True,
        time_no_class_enabled=True, attention_dropout=.1, classifier_num_layers=1, classifier_dropout=.1)
    config = dict(model=model_cfg, train=dict(iterations=4, validation_interval_iterations=1,
        log_interval_iterations=1, learning_rate=1e-3, lr_schedule='constant', weight_decay=.1,
        amp=False, progress_bar=False, time_loss_type='bce_one_vs_rest', pooled_loss_type='focal_bce',
        pooled_focal_gamma=1., time_loss_weight=.5, pooled_loss_weight=1., attention_mask_penalty_weight=.7,
        attention_topk_ratio=.8, checkpoint_metric='val_pooled_accuracy', checkpoint_mode='max',
        attention_mass_penalty_weight=.3,
        temporal_crop_noise=dict(enabled=True, geometry='random_blocks', sample_interval_s=.01,
            probability=.8, min_blocks=1, max_blocks=1, min_block_duration_s=.02,
            min_total_duration_s=.02, max_total_duration_s=.1, supervision_mode='all_positions'),
        near_full_noise=dict(enabled=True, probability=.1, coverage_ratio=.95, time_loss_scale=.25),
        augmentation_pipeline={'enabled': True, 'roll_max_shift_s': .1}))
    generator = torch.Generator().manual_seed(73)
    batches = [dict(eeg=torch.randn(8, 2, 20, generator=generator), target=torch.arange(8) % 2) for _ in range(3)]
    def new_model(): return build_model(deepcopy(model_cfg), time_point_count=20)
    torch.manual_seed(71)
    initial = deepcopy(new_model().state_dict())
    def fit(model, directory, resume=None):
        return training.fit_target_decoder(model, batches, batches, 'choice', torch.arange(20), config,
            directory, torch.device('cpu'), resume_from=resume)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    full.mkdir(); interrupted.mkdir()
    for directory in (full, interrupted):
        model = new_model(); model.load_state_dict(initial); torch.manual_seed(89)
        if directory == full:
            history = fit(model, directory)
            assert all(row['train_attention_mass_loss'] > 0. for row in history)
        else:
            append = training.append_metrics_csv
            def stop(path, row):
                if path.name == 'train_metrics.csv' and int(row['iteration']) == 3:
                    raise InterruptedError('synthetic interruption')
                append(path, row)
            with monkeypatch.context() as patcher:
                patcher.setattr(training, 'append_metrics_csv', stop)
                with pytest.raises(InterruptedError): fit(model, directory)
    checkpoint = interrupted / 'last.ckpt'
    assert load_checkpoint(checkpoint)['iteration'] == 2
    fit(new_model(), interrupted, checkpoint)
    expected, actual = load_checkpoint(full / 'last.ckpt'), load_checkpoint(checkpoint)
    for key, value in expected['model_state'].items():
        torch.testing.assert_close(actual['model_state'][key], value, rtol=0, atol=0)
    assert torch.equal(expected['rng_states']['torch_cpu'], actual['rng_states']['torch_cpu'])
    for filename in ['metrics.csv', 'train_metrics.csv']:
        assert (full / filename).read_bytes() == (interrupted / filename).read_bytes()




def test_raw_loss_reaches_swanlab():
    result = target_swanlab_metrics({'train_attention_mass_loss': .2, 'val_attention_mass_loss': 0.})
    assert result['train/loss/attention_mass'] == .2 and result['val/loss/attention_mass'] == 0.
