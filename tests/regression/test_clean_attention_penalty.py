"""Analytical gradients, mask routing, old-config compatibility and real fit wiring."""
from copy import deepcopy
from pathlib import Path
import csv
import math

import pytest
import torch
from torch import nn
import yaml

from pb2023_eeg.training.attention_regularization import (
    clean_attention_penalty,
    normalize_clean_attention_penalty_config,
)
from pb2023_eeg.training.augmentation import TemporalCropNoiseResult
from pb2023_eeg.training.common import load_checkpoint, validate_resume_config
from pb2023_eeg.training import trainer as training
from pb2023_eeg.training.swanlab_logger import target_swanlab_metrics
from pb2023_eeg.workflows.train import build_model


@pytest.mark.parametrize('name', ['weight', 'probability'])
@pytest.mark.parametrize('value', [None, True, False, 'bad', -0.1, float('nan'), float('inf'), 10 ** 400])
def test_invalid_configuration_is_rejected(name, value):
    with pytest.raises(ValueError, match='clean_attention_penalty_' + name):
        normalize_clean_attention_penalty_config({'clean_attention_penalty_' + name: value})


def test_configuration_defaults_and_bounds():
    assert normalize_clean_attention_penalty_config({}) == {'weight': 0., 'probability': .1}
    assert normalize_clean_attention_penalty_config({
        'clean_attention_penalty_weight': 2., 'clean_attention_penalty_probability': 0.,
    }) == {'weight': 2., 'probability': 0.}
    with pytest.raises(ValueError, match='probability'):
        normalize_clean_attention_penalty_config({'clean_attention_penalty_probability': 1.01})


def test_cropped_support_roll_and_noise_masks_have_exact_gradients():
    weights = torch.tensor([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]], requires_grad=True)
    indices = torch.tensor([1, 3, 5])
    covered = torch.zeros(3, 6, dtype=torch.bool)
    covered[1, :5] = True
    covered[2, 1] = True
    valid = torch.tensor([True, False, True])
    rng = torch.get_rng_state()
    loss, candidates, selected = clean_attention_penalty(weights, indices, covered,
        probability=1., valid_mask=valid, roll_mask=torch.tensor([True, False, False]))
    expected = torch.tensor([[False, False, False], [False, False, True], [False, False, True]])
    assert torch.equal(candidates, expected) and torch.equal(selected, expected)
    assert loss.item() == pytest.approx((.6 + .9) / 3)
    loss.backward()
    torch.testing.assert_close(weights.grad, expected.float() / 3, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), rng)


def test_random_sampling_is_per_position_and_not_renormalized():
    weights = torch.full((3, 275), .01, requires_grad=True)
    torch.manual_seed(321)
    expected = torch.rand(3, 275) < .1
    expected_rng = torch.get_rng_state()
    torch.manual_seed(321)
    loss, candidates, selected = clean_attention_penalty(weights, torch.arange(275), None)
    assert candidates.all() and torch.equal(selected, expected)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert 0 < int(selected.sum()) < 275 * 3
    assert loss.item() == pytest.approx(int(expected.sum()) * .01 / 3)
    loss.backward()
    torch.testing.assert_close(weights.grad, selected.float() / 3, rtol=0, atol=0)


@pytest.mark.parametrize('empty_kind', ['covered', 'invalid', 'roll', 'empty_batch', 'empty_time', 'probability_zero'])
def test_zero_paths_are_differentiable_and_do_not_consume_rng(empty_kind):
    shape = (0, 5) if empty_kind == 'empty_batch' else (2, 0) if empty_kind == 'empty_time' else (2, 5)
    weights = torch.full(shape, .2, requires_grad=True)
    options = {}
    covered = torch.ones(shape, dtype=torch.bool) if empty_kind == 'covered' else None
    if empty_kind == 'invalid': options['valid_mask'] = torch.zeros(shape[1], dtype=torch.bool)
    if empty_kind == 'roll': options['roll_mask'] = torch.ones(shape[0], dtype=torch.bool)
    if empty_kind == 'probability_zero': options['probability'] = 0.
    rng = torch.get_rng_state()
    loss, _, selected = clean_attention_penalty(weights, torch.arange(shape[1]), covered, **options)
    assert loss.item() == 0. and not selected.any()
    loss.backward()
    assert weights.grad is not None and torch.count_nonzero(weights.grad) == 0
    assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('empty_kind', ['covered', 'invalid', 'roll', 'probability_zero'])
def test_empty_candidates_do_not_sum_large_excluded_fp16_values(device, empty_kind):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    # A long additive-gate curve can have finite weights but an overflowing FP16
    # batch sum. Transpose also checks that the zero path supports strided inputs.
    weights = torch.full((275, 256), 1.9, device=device, dtype=torch.float16).T.requires_grad_()
    assert torch.isfinite(weights).all() and not torch.isfinite(weights.sum())
    covered = torch.ones_like(weights, dtype=torch.bool) if empty_kind == 'covered' else None
    options = {}
    if empty_kind == 'invalid': options['valid_mask'] = torch.zeros(275, dtype=torch.bool, device=device)
    if empty_kind == 'roll': options['roll_mask'] = torch.ones(256, dtype=torch.bool, device=device)
    if empty_kind == 'probability_zero': options['probability'] = 0.
    rng = torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state()
    loss, _, selected = clean_attention_penalty(weights, torch.arange(275, device=device), covered, **options)
    assert loss.dtype == weights.dtype and loss.item() == 0. and not selected.any()
    loss.backward()
    assert weights.grad is not None and torch.count_nonzero(weights.grad) == 0
    assert torch.equal(torch.cuda.get_rng_state() if device == 'cuda' else torch.get_rng_state(), rng)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_reduction_and_per_trial_valid_support(dtype):
    weights = torch.full((2, 275), 1 / 275, dtype=dtype, requires_grad=True)
    valid = torch.ones_like(weights, dtype=torch.bool)
    valid[1, :100] = False
    loss, candidates, selected = clean_attention_penalty(weights, torch.arange(275), None,
        probability=1., valid_mask=valid)
    assert int(candidates.sum()) == 450 and torch.equal(candidates, selected)
    assert loss.item() == pytest.approx(float(weights[0, 0]) * 450 / 2, rel=.004)
    loss.backward()
    assert torch.isfinite(weights.grad).all()
    assert torch.count_nonzero(weights.grad[~valid]) == 0


@pytest.mark.parametrize('problem', ['index_dtype', 'index_shape', 'index_count', 'index_negative',
                                    'index_range', 'covered_shape', 'covered_dtype', 'valid_shape', 'roll_shape'])
def test_mask_alignment_errors_are_reported(problem):
    weights = torch.ones(2, 3)
    indices = torch.arange(3)
    covered = torch.zeros(2, 3, dtype=torch.bool)
    options = {}
    if problem == 'index_dtype': indices = indices.float()
    elif problem == 'index_shape': indices = indices[None]
    elif problem == 'index_count': indices = indices[:2]
    elif problem == 'index_negative': indices = torch.tensor([-1, 0, 1])
    elif problem == 'index_range': indices = torch.tensor([0, 1, 3])
    elif problem == 'covered_shape': covered = covered[:1]
    elif problem == 'covered_dtype': covered = covered.float()
    elif problem == 'valid_shape': options['valid_mask'] = torch.ones(2, dtype=torch.bool)
    elif problem == 'roll_shape': options['roll_mask'] = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match='clean attention'):
        clean_attention_penalty(weights, indices, covered, **options)


class Probe(nn.Module):
    fusion_enabled = False

    def __init__(self, valid=None):
        super().__init__()
        self.strength = nn.Parameter(torch.arange(1, 19).reshape(3, 6).float() / 100)
        self.time = nn.Parameter(torch.zeros(3, 6, 3))
        self.pooled = nn.Parameter(torch.zeros(3, 2))
        self.register_buffer('valid', valid)

    def time_class_count(self, task): return 3
    def time_no_class_index(self, task): return 2

    def forward(self, eeg, attention_indices, **kwargs):
        batch_size = eeg.shape[0]
        weights = self.strength[:batch_size].index_select(1, attention_indices)
        if self.valid is not None: weights = weights.masked_fill(~self.valid, 0)
        return dict(time_logits=self.time[:batch_size], pooled_logits=self.pooled[:batch_size],
                    attention_weights=torch.ones_like(weights) / weights.shape[1],
                    effective_attention_weights=weights, attention_valid_mask=self.valid)


class MixedAugmentation(nn.Module):
    def forward(self, eeg, return_mask=False):
        covered = torch.zeros(3, 6, dtype=torch.bool, device=eeg.device)
        covered[1, :5] = True
        covered[2, 1] = True
        return TemporalCropNoiseResult(eeg, ~covered, covered,
            torch.tensor([False, True, False], device=eeg.device),
            torch.tensor([True, False, False], device=eeg.device),
            torch.tensor([1, 0, 0], device=eeg.device))


def batch(n=3):
    return dict(eeg=torch.zeros(n, 2, 6), target=torch.tensor([0, 1, 0])[:n],
                sample_weight=torch.tensor([1., 10., 100.])[:n])


def epoch(model, batches=None, training_mode=True, weight=2., probability=1., **options):
    return training.run_target_epoch(model, batches or [batch()], 'choice',
        options.pop('attention_indices', torch.arange(6)),
        options.pop('loss_weights', {'time': .5, 'pooled': 1., 'attention_mask': .7, 'clean_attention': weight}),
        None, .5, 0., torch.device('cpu'), False,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.) if training_mode else None,
        scaler=torch.amp.GradScaler('cpu', enabled=False) if training_mode else None,
        noise_supervision_mode=options.pop('noise_supervision_mode', 'all_positions'), progress_bar_enabled=False,
        clean_attention_penalty_probability=probability, **options)


def test_actual_training_masks_labels_near_full_noise_and_independent_weight():
    baseline = Probe(valid=torch.tensor([True, False, True]))
    actual = deepcopy(baseline)
    options = dict(eeg_augmenter=MixedAugmentation(), attention_indices=torch.tensor([1, 3, 5]),
                   near_full_noise_time_loss_scale=.25)
    before = epoch(baseline, weight=0., **options)
    after = epoch(actual, **options)
    expected_loss = float((actual.strength[1, 5] + actual.strength[2, 5]).detach()) / 3
    assert after['clean_attention_loss'] == pytest.approx(expected_loss)
    assert after['loss'] - before['loss'] == pytest.approx(2 * expected_loss)
    expected_gradient = torch.zeros(3, 6)
    expected_gradient[1:, 5] = 2 / 3
    torch.testing.assert_close(actual.strength.grad - baseline.strength.grad, expected_gradient)
    torch.testing.assert_close(actual.time.grad, baseline.time.grad, rtol=0, atol=0)
    torch.testing.assert_close(actual.pooled.grad, baseline.pooled.grad, rtol=0, atol=0)
    for key in ['time_loss', 'pooled_loss', 'attention_mask_loss', 'trial_loss_sample_count']:
        assert before[key] == after[key]
    assert after['clean_attention_candidate_count_per_sample'] == pytest.approx(2 / 3)
    assert after['clean_attention_selected_count_per_sample'] == pytest.approx(2 / 3)
    assert after['clean_attention_selected_fraction'] == 1.


@pytest.mark.parametrize('disabled', ['weight', 'probability', 'validation'])
def test_disabled_and_validation_paths_never_sample(monkeypatch, disabled):
    model = Probe()
    def must_not_run(*args, **kwargs): raise AssertionError('Unexpected clean attention sampling')
    monkeypatch.setattr('pb2023_eeg.training.step.clean_attention_penalty', must_not_run)
    rng = torch.get_rng_state()
    result = epoch(model, weight=0. if disabled == 'weight' else 2.,
                   probability=0. if disabled == 'probability' else .1,
                   training_mode=disabled != 'validation')
    assert result['clean_attention_loss'] == result['clean_attention_selected_fraction'] == 0.
    assert torch.equal(torch.get_rng_state(), rng)


def test_epoch_reduction_weights_unequal_batches_by_trial_count():
    model = Probe()
    result = epoch(model, batches=[batch(), batch(1)],
        loss_weights={'time': 0., 'pooled': 0., 'attention_mask': 0., 'clean_attention': 2.})
    expected = float((model.strength.sum() + model.strength[0].sum()).detach()) / 4
    assert result['clean_attention_loss'] == pytest.approx(expected)
    assert result['loss'] == pytest.approx(2 * expected)
    assert result['clean_attention_candidate_count_per_sample'] == 6.


@pytest.mark.parametrize('activation,normalization', [
    ('tanh_plus_one', 'none'), ('tanh_plus_one', 'one_plus_mass'),
    ('tanh_plus_one', 'softmax'), ('tanh', 'none'),
])
def test_actual_final_weights_and_signed_magnitude_feed_the_penalty(activation, normalization):
    config = dict(encoder=dict(in_channels=2, latent_dim=4, num_blocks=1, kernel_size=3),
        attention_mode='gated_scorer', attention_gate_activation=activation,
        attention_gate_init_bias=-1., attention_effective_normalization=normalization,
        time_no_class_enabled=True, classifier_num_layers=1, classifier_dropout=0.)
    model = build_model(config, time_point_count=6)
    captured = []
    handle = model.register_forward_hook(lambda _m, _i, out: captured.append(out))
    result = epoch(model)
    handle.remove()
    expected = captured[0]['effective_attention_weights'].detach().abs().sum(-1).mean()
    assert result['clean_attention_loss'] == pytest.approx(float(expected))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('task', ['choice', 'pitch'])
@pytest.mark.parametrize('protocol', ['block', 'subject'])
def test_four_selected_baseline_models_fit_and_record_the_new_configuration(tmp_path, task, protocol):
    path = Path(__file__).resolve().parents[2] / 'configs' / 'noise-pad' / f'{task}-{protocol}.yaml'
    config = yaml.safe_load(path.read_text())
    config['model']['subject_conditioning']['resolved_training_subject_ids'] = list(range(24))
    config['train'].update(iterations=2, validation_interval_iterations=1, log_interval_iterations=1,
        lr_schedule_iterations=2, warmup_iterations=0,
        progress_bar=False, amp=False, clean_attention_penalty_weight=.2,
        clean_attention_penalty_probability=.1)
    # Production loaders expose dataset.time_s; this list-based fixture supplies
    # the equivalent sampling interval explicitly for the existing augmenter.
    config['train']['temporal_crop_noise']['sample_interval_s'] = .01
    model = build_model(config['model'], n_subjects=30, time_point_count=275, input_channels=2)
    ids = torch.tensor([24, 25]) if protocol == 'subject' else torch.tensor([0, 1])
    example = dict(eeg=torch.randn(2, 2, 275), target=torch.tensor([0, 1]), subject_id=ids,
                   current_stimulus=torch.tensor([0, 0]), exp=torch.ones(2, dtype=torch.long), trial0=torch.arange(2))
    history = training.fit_target_decoder(model, [example], [example], task, torch.arange(275),
        config, tmp_path, torch.device('cpu'))
    assert len(history) == 2
    for row in history:
        assert row['clean_attention_penalty_weight'] == .2 and row['clean_attention_penalty_probability'] == .1
        assert row['train_clean_attention_loss'] >= 0. and math.isfinite(row['train_loss'])
        assert row['val_clean_attention_loss'] == row['val_clean_attention_selected_fraction'] == 0.
        assert row['train_clean_attention_candidate_count_per_sample'] <= 275
    rows = list(csv.DictReader((tmp_path / 'metrics.csv').open()))
    assert float(rows[-1]['clean_attention_penalty_probability']) == .1
    saved = load_checkpoint(tmp_path / 'last.ckpt')
    assert saved['config']['train']['clean_attention_penalty_weight'] == .2
    assert saved['config']['train']['clean_attention_penalty_probability'] == .1


def test_resume_identity_distinguishes_enabled_options_but_preserves_disabled_defaults():
    legacy = {'train': {}}
    for options in ({'clean_attention_penalty_weight': 0.},
                    {'clean_attention_penalty_probability': .7},
                    {'clean_attention_penalty_weight': 2., 'clean_attention_penalty_probability': 0.}):
        validate_resume_config(legacy, {'train': options})
    enabled = {'train': {'clean_attention_penalty_weight': .2}}
    explicit = deepcopy(enabled)
    explicit['train']['clean_attention_penalty_probability'] = .1
    validate_resume_config(enabled, explicit)
    for key, value in [('clean_attention_penalty_probability', .2), ('clean_attention_penalty_weight', .3)]:
        changed = deepcopy(explicit); changed['train'][key] = value
        with pytest.raises(ValueError, match=key): validate_resume_config(enabled, changed)
    with pytest.raises(ValueError, match='clean_attention_penalty'): validate_resume_config(legacy, enabled)


@pytest.mark.parametrize('encoder_type', ['cnn', 'window_transformer'])
def test_active_sampling_resumes_with_identical_model_rng_and_csv(tmp_path, monkeypatch, encoder_type):
    encoder = dict(type=encoder_type, in_channels=2, latent_dim=4, dropout=.1)
    if encoder_type == 'cnn':
        encoder.update(num_blocks=1, kernel_size=3, expansion_ratio=1)
    else:
        encoder.update(num_layers=1, window_size=3, num_heads=2, ffn_ratio=2,
            attention_pattern='centered_local', position_encoding_mode='sinusoidal_qk')
    model_config = dict(encoder=encoder, attention_mode='gated_scorer', attention_dim=4,
        attention_gate_activation='tanh_plus_one', attention_score_centering=True,
        time_no_class_enabled=True, attention_dropout=.1, classifier_num_layers=1, classifier_dropout=.1)
    config = dict(model=model_config, train=dict(iterations=4, validation_interval_iterations=1,
        log_interval_iterations=1, learning_rate=1e-3, lr_schedule='constant', weight_decay=.1,
        amp=False, progress_bar=False, time_loss_type='bce_one_vs_rest', pooled_loss_type='focal_bce',
        pooled_focal_gamma=1., time_loss_weight=.5, pooled_loss_weight=1., attention_mask_penalty_weight=.7,
        attention_topk_ratio=.8, checkpoint_metric='val_pooled_accuracy', checkpoint_mode='max',
        clean_attention_penalty_weight=.2, clean_attention_penalty_probability=.1,
        temporal_crop_noise=dict(enabled=True, geometry='random_blocks', sample_interval_s=.01,
            probability=.8, min_blocks=1, max_blocks=1, min_block_duration_s=.02,
            min_total_duration_s=.02, max_total_duration_s=.1, supervision_mode='all_positions'),
        near_full_noise=dict(enabled=True, probability=.1, coverage_ratio=.95, time_loss_scale=.25),
        augmentation_pipeline={'enabled': True, 'roll_max_shift_s': .1}))
    generator = torch.Generator().manual_seed(73)
    batches = [dict(eeg=torch.randn(8, 2, 20, generator=generator), target=torch.arange(8) % 2)
               for _ in range(3)]
    def new_model(): return build_model(deepcopy(model_config), time_point_count=20)
    torch.manual_seed(71)
    initial = deepcopy(new_model().state_dict())
    def fit(model, directory, resume=None):
        return training.fit_target_decoder(model, batches, batches, 'choice', torch.arange(20),
            config, directory, torch.device('cpu'), resume_from=resume)
    full, interrupted = tmp_path / 'full', tmp_path / 'interrupted'
    full.mkdir(); interrupted.mkdir()
    for directory in (full, interrupted):
        model = new_model(); model.load_state_dict(initial); torch.manual_seed(89)
        if directory == full:
            history = fit(model, directory)
            assert all(r['train_clean_attention_selected_count_per_sample'] > 0 for r in history)
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


def test_rf_aware_uses_the_existing_contaminated_halo(monkeypatch):
    from pb2023_eeg.training.augmentation import classify_temporal_coverage
    config = dict(encoder=dict(in_channels=2, latent_dim=4, num_blocks=1, kernel_size=3),
        attention_mode='gated_scorer', attention_gate_activation='tanh_plus_one',
        classifier_num_layers=1, classifier_dropout=0.)
    model = build_model(config, time_point_count=6)
    class LocalNoise(nn.Module):
        def forward(self, eeg, return_mask=False):
            covered = torch.zeros(3, 6, dtype=torch.bool)
            covered[:, 2] = True
            return TemporalCropNoiseResult(eeg, ~covered, covered)
    captured = []
    real_penalty = training.clean_attention_penalty
    def spy(weights, indices, contaminated, **kwargs):
        captured.append(contaminated.clone())
        return real_penalty(weights, indices, contaminated, **kwargs)
    monkeypatch.setattr('pb2023_eeg.training.step.clean_attention_penalty', spy)
    epoch(model, eeg_augmenter=LocalNoise(), noise_supervision_mode='rf_aware')
    covered = torch.zeros(3, 6, dtype=torch.bool); covered[:, 2] = True
    rf = training.target_time_receptive_field_layers(model)
    expected = ~classify_temporal_coverage(covered, rf).clean_time_mask
    assert expected.sum() > covered.sum() and torch.equal(captured[0], expected)




def test_new_diagnostics_reach_swanlab():
    values = target_swanlab_metrics(dict(train_clean_attention_loss=.02,
        train_clean_attention_candidate_count_per_sample=200.,
        train_clean_attention_selected_count_per_sample=20., train_clean_attention_selected_fraction=.1))
    assert values['train/loss/clean_attention'] == .02
    assert values['train/attention/clean_candidate_count'] == 200.
    assert values['train/attention/clean_selected_count'] == 20.
    assert values['train/attention/clean_selected_fraction'] == .1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cuda_sampling_keeps_rng_and_gradients_on_device(dtype):
    weights = torch.full((4, 275), 1 / 275, device='cuda', dtype=dtype, requires_grad=True)
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    loss, candidates, selected = clean_attention_penalty(weights, torch.arange(275, device='cuda'), None)
    after = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(cuda_rng)
    repeated, _, mask = clean_attention_penalty(weights, torch.arange(275, device='cuda'), None)
    assert torch.equal(selected, mask) and torch.equal(loss, repeated)
    assert torch.equal(torch.cuda.get_rng_state(), after) and torch.equal(torch.get_rng_state(), cpu_rng)
    assert candidates.device == selected.device == weights.device
    loss.backward()
    assert torch.isfinite(weights.grad).all()
    assert torch.count_nonzero(weights.grad[~selected]) == 0
