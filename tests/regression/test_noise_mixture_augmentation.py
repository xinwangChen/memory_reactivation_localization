"""Trial-wise mixture routing, original-data semantics and reproducibility."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from memory_reactivation.training.augmentation import (
    EEGInputAugmenter, NearFullNoiseAugmenter, TemporalCropNoiseResult,
    _replace_with_noise, build_eeg_augmenter, build_target_eeg_augmenter,
    normalize_noise_mode_probabilities,
)
from memory_reactivation.training.common import validate_resume_config

SHARED = 'time_shuffle'
INDEPENDENT = 'time_shuffle_independent'
GAUSSIAN = 'sample_channel'
MIXTURES = [
    {SHARED: .5, INDEPENDENT: .5},
    {INDEPENDENT: .5, GAUSSIAN: .5},
    {SHARED: .5, GAUSSIAN: .5},
    {SHARED: 1/3, GAUSSIAN: 1/3, INDEPENDENT: 1/3},
]


def replace(eeg, mask, probabilities):
    return _replace_with_noise(eeg, mask, noise_mode='mixture',
                               noise_mode_probabilities=probabilities, min_noise_std=1e-4)


@pytest.mark.parametrize('probabilities', [None, {}, [], {None: 1.}, {'mixture': 1.}, {'bad': 1.},
    {SHARED: -.1, GAUSSIAN: 1.1}, {SHARED: float('nan')}, {SHARED: float('inf')},
    {SHARED: True}, {SHARED: 10**400}, {SHARED: None}, {SHARED: 'bad'},
    {SHARED: 0.}, {SHARED: .9}, {SHARED: .5, INDEPENDENT: .49},
    {SHARED: .5, INDEPENDENT: .51}])
def test_invalid_mixture_probabilities_are_rejected(probabilities):
    with pytest.raises(ValueError, match='noise_mode_probabilities'):
        EEGInputAugmenter(sample_interval_s=.01, noise_mode='mixture', noise_mode_probabilities=probabilities)
    with pytest.raises(ValueError, match='noise_mode_probabilities'):
        NearFullNoiseAugmenter(noise_mode='mixture', noise_mode_probabilities=probabilities)


def test_probabilities_cannot_be_silently_ignored_in_a_single_mode():
    with pytest.raises(ValueError, match='requires noise_mode=mixture'):
        build_eeg_augmenter(dict(enabled=True, sample_interval_s=.01, noise_mode=GAUSSIAN,
                                 noise_mode_probabilities=MIXTURES[-1]))


def test_normalization_is_order_independent_and_does_not_alias_user_mapping():
    probabilities = deepcopy(MIXTURES[-1])
    augmenter = NearFullNoiseAugmenter(noise_mode='mixture', noise_mode_probabilities=probabilities)
    assert augmenter.noise_mode_probabilities == probabilities
    assert list(augmenter.noise_mode_probabilities) == sorted(probabilities)
    probabilities[SHARED] = 0.
    assert augmenter.noise_mode_probabilities[SHARED] == 1/3
    assert normalize_noise_mode_probabilities(GAUSSIAN) is None


@pytest.mark.parametrize('probabilities', MIXTURES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_mixture_preserves_uncovered_data_shape_dtype_and_input(probabilities, dtype, noncontiguous):
    eeg = torch.randn(8, 3, 30).to(dtype)
    if noncontiguous:
        eeg = eeg.transpose(1, 2).contiguous().transpose(1, 2)
        assert not eeg.is_contiguous()
    before = eeg.clone()
    mask = torch.zeros(8, 30, dtype=torch.bool)
    mask[1, [0, 2, 6, 7, 21, 29]] = True
    mask[2:, 3:25] = True
    old_mask = mask.clone()
    actual = replace(eeg, mask, probabilities)
    assert actual.shape == eeg.shape and actual.dtype == dtype and actual.device == eeg.device
    assert torch.isfinite(actual).all() and torch.equal(eeg, before) and torch.equal(mask, old_mask)
    kept = (~mask)[:, None, :].expand_as(eeg)
    assert torch.equal(actual[kept], eeg[kept])
    if GAUSSIAN not in probabilities:
        assert torch.equal(actual.sort(dim=2).values, eeg.sort(dim=2).values)


def test_routing_chooses_once_per_active_trial_and_uses_original_whole_batch():
    torch.manual_seed(512)
    eeg = torch.arange(6*3*20, dtype=torch.float32).reshape(6, 3, 20)
    mask = torch.zeros(6, 20, dtype=torch.bool)
    mask[1:, [0, 1, 5, 8, 13, 19]] = True
    choices = torch.tensor([0, 1, 2, 0, 2])
    probabilities = MIXTURES[-1]
    original = _replace_with_noise
    with patch('torch.multinomial', return_value=choices) as draw:
        with patch('memory_reactivation.training.augmentation._replace_with_noise', wraps=original) as spy:
            actual = replace(eeg, mask, probabilities)
    draw.assert_called_once()
    assert draw.call_args.args[1] == 5 and draw.call_args.kwargs['replacement'] is True
    calls = [call for call in spy.call_args_list if call.kwargs['noise_mode'] != 'mixture']
    assert len(calls) == 3
    union = torch.zeros_like(mask)
    for index, mode in enumerate(sorted(probabilities)):
        call = next(call for call in calls if call.kwargs['noise_mode'] == mode)
        assert call.args[0] is eeg  # No sequential corruption or subgroup statistics.
        expected = mask.clone()
        selected = torch.cat((torch.tensor([-1]), choices))
        expected &= (selected == index)[:, None]
        assert torch.equal(call.args[1], expected)
        assert not (union & expected).any()
        union |= expected
    assert torch.equal(union, mask)
    # The shared row still moves complete spatial vectors; the independent row does not.
    assert torch.equal(actual[2, 1] - 20, actual[2, 0])
    assert not torch.equal(actual[3, 1] - 20, actual[3, 0])


def test_batch_channel_mixture_estimates_statistics_from_original_full_batch():
    eeg = torch.arange(4*2*12, dtype=torch.float32).reshape(4, 2, 12)
    mask = torch.ones(4, 12, dtype=torch.bool)
    # Sorted candidates are batch_channel, standard_normal; only one row uses batch_channel.
    with patch('torch.multinomial', return_value=torch.tensor([0, 1, 1, 1])):
        with patch('torch.randn_like', side_effect=lambda values: torch.zeros_like(values)):
            actual = replace(eeg, mask, {'batch_channel': .5, 'standard_normal': .5})
    assert torch.equal(actual[0], eeg.mean(dim=(0, 2))[:, None].expand_as(eeg[0]))
    assert torch.equal(actual[1:], torch.zeros_like(actual[1:]))


@pytest.mark.parametrize('probabilities', MIXTURES)
def test_observed_modes_follow_configured_probabilities_per_trial(probabilities):
    # Unique time codes let us independently identify permutations vs Gaussian output.
    torch.manual_seed(321)
    n = 4096
    eeg = (torch.arange(n)[:, None, None]*1000 + torch.arange(3)[None, :, None]*100
           + torch.arange(20)[None, None, :]).float()
    actual = replace(eeg, torch.ones(n, 20, dtype=torch.bool), probabilities)
    permuted = (eeg.sort(dim=2).values == actual.sort(dim=2).values).all(dim=2).all(dim=1)
    shared = permuted & (actual[:, 1:] - actual[:, :1] == torch.tensor([100, 200])[None, :, None]).all(dim=2).all(dim=1)
    observed = {SHARED: shared.float().mean().item(), INDEPENDENT: (permuted & ~shared).float().mean().item(),
                GAUSSIAN: (~permuted).float().mean().item()}
    for mode in (SHARED, INDEPENDENT, GAUSSIAN):
        assert abs(observed[mode] - probabilities.get(mode, 0.)) < .025


@pytest.mark.parametrize('probabilities', MIXTURES)
def test_rng_restore_and_mapping_reordering_reproduce_mixed_output(probabilities):
    eeg = torch.randn(16, 3, 50)
    mask = torch.ones(16, 50, dtype=torch.bool)
    torch.manual_seed(51)
    state = torch.get_rng_state()
    first = replace(eeg, mask, probabilities)
    final = torch.get_rng_state()
    assert not torch.equal(first, replace(eeg, mask, probabilities))
    torch.set_rng_state(state)
    assert torch.equal(first, replace(eeg, mask, dict(reversed(list(probabilities.items())))))
    assert torch.equal(final, torch.get_rng_state())


@pytest.mark.parametrize('mode', [GAUSSIAN, 'batch_channel', 'standard_normal', SHARED, INDEPENDENT])
def test_one_hot_mixture_exactly_preserves_single_mode_output_and_rng(mode):
    eeg = torch.randn(5, 3, 40)
    mask = torch.zeros(5, 40, dtype=torch.bool)
    mask[1:, [0, 1, 4, 8, 19, 39]] = True
    torch.manual_seed(17)
    reference = _replace_with_noise(eeg, mask, noise_mode=mode, min_noise_std=1e-4)
    state = torch.get_rng_state()
    torch.manual_seed(17)
    actual = replace(eeg, mask, {mode: 1.})
    assert torch.equal(reference, actual) and torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('ordinary_enabled', [False, True])
@pytest.mark.parametrize('near_probability', [0., .5, 1.])
def test_factory_inherits_same_mixture_and_keeps_original_near_clean_segment(ordinary_enabled, near_probability):
    config = dict(enabled=ordinary_enabled, geometry='random_blocks', sample_interval_s=.01, probability=.8,
                  min_blocks=1, max_blocks=2, min_block_duration_s=.05, min_total_duration_s=.1,
                  max_total_duration_s=.5, noise_mode='mixture', noise_mode_probabilities=MIXTURES[-1])
    aug = build_target_eeg_augmenter(config, near_full_noise=dict(enabled=True, probability=near_probability, coverage_ratio=.95))
    if not ordinary_enabled and not near_probability:
        assert aug is None
        return
    assert aug.noise_mode_probabilities == MIXTURES[-1]
    if isinstance(aug, NearFullNoiseAugmenter) and ordinary_enabled:
        assert aug.ordinary_augmenter.noise_mode_probabilities == aug.noise_mode_probabilities
    eeg = torch.randn(64, 3, 275)
    result = aug(eeg, return_mask=True)
    kept = result.valid_time_mask[:, None, :].expand_as(eeg)
    assert torch.equal(result.eeg[kept], eeg[kept])
    if result.near_full_noise_mask is not None:
        near = result.near_full_noise_mask
        assert (result.covered_time_mask[near].sum(dim=1) == 261).all()
        for trial in near.nonzero().flatten():
            clean = result.valid_time_mask[trial].nonzero().flatten()
            assert clean.numel() == 14 and int(clean[-1] - clean[0]) == 13


def test_near_mixture_rebuilds_from_original_not_ordinary_output():
    class Ordinary(nn.Module):
        def forward(self, eeg, return_mask=False):
            mask = torch.ones(eeg.shape[0], eeg.shape[-1], dtype=torch.bool)
            return TemporalCropNoiseResult(torch.full_like(eeg, 99999), ~mask, mask)
    eeg = torch.randn(32, 2, 275)
    result = NearFullNoiseAugmenter(Ordinary(), probability=1., coverage_ratio=.95,
        noise_mode='mixture', noise_mode_probabilities=MIXTURES[0])(eeg, return_mask=True)
    assert torch.equal(result.eeg.sort(dim=2).values, eeg.sort(dim=2).values)
    assert torch.equal(result.eeg[result.valid_time_mask[:, None, :].expand_as(eeg)],
                       eeg[result.valid_time_mask[:, None, :].expand_as(eeg)])


def test_disabled_and_empty_mixture_do_not_draw_random_numbers():
    eeg = torch.randn(3, 2, 20)
    for aug in [EEGInputAugmenter(sample_interval_s=.01, probability=0., noise_mode='mixture', noise_mode_probabilities=MIXTURES[-1]),
                NearFullNoiseAugmenter(probability=0., noise_mode='mixture', noise_mode_probabilities=MIXTURES[-1])]:
        state = torch.get_rng_state()
        assert torch.equal(aug(eeg), eeg) and torch.equal(torch.get_rng_state(), state)
    state = torch.get_rng_state()
    assert torch.equal(replace(eeg, torch.zeros(3, 20, dtype=torch.bool), MIXTURES[-1]), eeg)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize('probabilities', MIXTURES)
def test_mixture_backward_is_finite_without_mutating_original(probabilities):
    eeg = torch.randn(12, 3, 40, requires_grad=True)
    original = eeg.detach().clone()
    mask = torch.zeros(12, 40, dtype=torch.bool)
    mask[1:, :30] = True
    replace(eeg, mask, probabilities).square().sum().backward()
    assert torch.equal(eeg.detach(), original) and torch.isfinite(eeg.grad).all()
    if GAUSSIAN not in probabilities:
        torch.testing.assert_close(eeg.grad, 2*original, rtol=0, atol=0)


def test_resume_accepts_mapping_order_but_rejects_changing_modes_or_probabilities():
    old = dict(train=dict(temporal_crop_noise=dict(enabled=True, noise_mode='mixture', noise_mode_probabilities=MIXTURES[-1]),
                          near_full_noise=dict(enabled=True, probability=.1)))
    reordered = deepcopy(old)
    reordered['train']['temporal_crop_noise']['noise_mode_probabilities'] = dict(reversed(list(MIXTURES[-1].items())))
    validate_resume_config(old, reordered)
    changed = deepcopy(old)
    changed['train']['temporal_crop_noise']['noise_mode_probabilities'] = MIXTURES[0]
    with pytest.raises(ValueError, match='noise_mode_probabilities'):
        validate_resume_config(old, changed)
    changed['train']['temporal_crop_noise'] = dict(enabled=True, noise_mode=GAUSSIAN)
    with pytest.raises(ValueError, match='noise_mode'):
        validate_resume_config(old, changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('probabilities', MIXTURES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_cuda_mixture_rng_locality_and_backward(probabilities, dtype):
    eeg = torch.randn(12, 3, 40, device='cuda', dtype=dtype, requires_grad=True)
    mask = torch.zeros(12, 40, dtype=torch.bool, device='cuda')
    mask[1:, 2:35] = True
    state = torch.cuda.get_rng_state()
    first = replace(eeg, mask, probabilities)
    final = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    second = replace(eeg, mask, probabilities)
    assert torch.equal(first, second) and torch.equal(final, torch.cuda.get_rng_state())
    kept = (~mask)[:, None, :].expand_as(eeg)
    assert torch.equal(first[kept], eeg[kept])
    first.float().square().mean().backward()
    assert torch.isfinite(eeg.grad).all()
