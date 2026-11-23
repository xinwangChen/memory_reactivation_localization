"""Shared/independent time shuffle, exact locality and training integration."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from pb2023_eeg.training.augmentation import (
    EEGInputAugmenter,
    NearFullNoiseAugmenter,
    _replace_with_noise,
    build_eeg_augmenter,
    build_target_eeg_augmenter,
)
from pb2023_eeg.training.common import validate_resume_config


SHUFFLE_MODES = ['time_shuffle', 'time_shuffle_independent']


def shuffle(eeg, covered, mode='time_shuffle'):
    return _replace_with_noise(eeg, covered, noise_mode=mode, min_noise_std=1e-4)


def assert_covered_values_are_permuted(original, actual, covered):
    assert actual.shape == original.shape and actual.dtype == original.dtype
    assert actual.device == original.device
    keep = (~covered)[:, None, :].expand_as(original)
    torch.testing.assert_close(actual[keep], original[keep], rtol=0, atol=0)
    for trial in range(original.shape[0]):
        indices = covered[trial].nonzero().flatten()
        if indices.numel() == 0:
            continue
        before = original[trial, :, indices]
        after = actual[trial, :, indices]
        torch.testing.assert_close(before.sort(dim=-1).values, after.sort(dim=-1).values, rtol=0, atol=0)


def assert_covered_vectors_are_permuted(original, actual, covered):
    """Use a unique first-channel time code to identify the moved spatial vectors."""
    assert_covered_values_are_permuted(original, actual, covered)
    for trial in range(original.shape[0]):
        indices = covered[trial].nonzero().flatten()
        if indices.numel() == 0:
            continue
        before = original[trial, :, indices]
        after = actual[trial, :, indices]
        # Every entire spatial vector must appear once, with no channel-wise mixing.
        matches = after[0, :, None] == before[0, None, :]
        assert (matches.sum(dim=0) == 1).all() and (matches.sum(dim=1) == 1).all()
        torch.testing.assert_close(after, before[:, matches.long().argmax(dim=1)], rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize('noncontiguous', [False, True])
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_shuffle_preserves_exact_values_and_uncovered_positions(dtype, noncontiguous, mode):
    eeg = (torch.arange(3)[:, None, None] * 60
           + torch.arange(3)[None, :, None] * 15
           + torch.arange(12)[None, None, :]).to(dtype)
    if noncontiguous:
        eeg = eeg.transpose(1, 2).contiguous().transpose(1, 2)
        assert not eeg.is_contiguous()
    before = eeg.clone()
    covered = torch.zeros(3, 12, dtype=torch.bool)
    covered[1, [1, 3, 5, 6, 10]] = True
    covered[2] = True
    mask_before = covered.clone()
    torch.manual_seed(7)
    with patch('torch.randn_like', side_effect=AssertionError('No Gaussian draws in time_shuffle')):
        actual = shuffle(eeg, covered, mode)
    assert_covered_values_are_permuted(eeg, actual, covered)
    if mode == 'time_shuffle':
        assert_covered_vectors_are_permuted(eeg, actual, covered)
    else:
        assert not torch.equal(actual[2, 0] - before[2, 0, 0], actual[2, 1] - before[2, 1, 0])
    assert not torch.equal(actual[2], eeg[2])
    torch.testing.assert_close(eeg, before, rtol=0, atol=0)
    assert torch.equal(covered, mask_before)


@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_shuffles_are_independent_per_trial_and_reproducible_from_rng_state(mode):
    eeg = torch.arange(32, dtype=torch.float32).expand(8, 3, -1).clone()
    covered = torch.ones(8, 32, dtype=torch.bool)
    torch.manual_seed(59)
    state = torch.get_rng_state()
    first = shuffle(eeg, covered, mode)
    second = shuffle(eeg, covered, mode)
    assert not torch.equal(first, second)
    assert not torch.equal(first[0], first[1])
    torch.set_rng_state(state)
    assert torch.equal(first, shuffle(eeg, covered, mode))
    if mode == 'time_shuffle':
        assert torch.equal(first[:, :1].expand_as(first), first)
    else:
        assert not torch.equal(first[0, 0], first[0, 1])


@pytest.mark.parametrize('count', [0, 1])
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_empty_and_singleton_coverage_leave_values_unchanged(count, mode):
    eeg = torch.randn(2, 3, 8)
    covered = torch.zeros(2, 8, dtype=torch.bool)
    covered[:, :count] = True
    actual = shuffle(eeg, covered, mode)
    assert torch.equal(actual, eeg) and actual.data_ptr() != eeg.data_ptr()


@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_shuffle_backward_preserves_gradient_to_each_original_value(mode):
    eeg = torch.randn(3, 2, 20, requires_grad=True)
    original = eeg.detach().clone()
    covered = torch.zeros(3, 20, dtype=torch.bool)
    covered[0] = True
    covered[1, [0, 1, 5, 6, 7, 15, 19]] = True
    shuffle(eeg, covered, mode).square().sum().backward()
    torch.testing.assert_close(eeg.grad, 2 * original, rtol=0, atol=0)
    torch.testing.assert_close(eeg.detach(), original, rtol=0, atol=0)


@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_every_six_point_coverage_mask_preserves_its_source_set(mode):
    covered = (torch.arange(64)[:, None] & (1 << torch.arange(6)[None, :])) != 0
    eeg = torch.arange(64 * 3 * 6, dtype=torch.float32).reshape(64, 3, 6)
    actual = shuffle(eeg, covered, mode)
    assert_covered_values_are_permuted(eeg, actual, covered)
    if mode == 'time_shuffle':
        assert_covered_vectors_are_permuted(eeg, actual, covered)


@pytest.mark.parametrize('geometry', ['edge', 'circular_block', 'random_blocks'])
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_ordinary_builder_preserves_shuffle_contract(geometry, mode):
    config = dict(enabled=True, geometry=geometry, sample_interval_s=.01,
                  probability=1., min_duration_s=.05, max_left_duration_s=.1,
                  max_right_duration_s=.1, block_duration_s=.1,
                  min_total_duration_s=.1, max_total_duration_s=.15,
                  min_blocks=1, max_blocks=2, noise_mode=mode)
    augmenter = build_eeg_augmenter(config)
    eeg = torch.arange(4 * 3 * 30, dtype=torch.float32).reshape(4, 3, 30)
    torch.manual_seed(93)
    result = augmenter(eeg, return_mask=True)
    assert result.covered_time_mask.any()
    assert torch.equal(result.valid_time_mask, ~result.covered_time_mask)
    assert result.near_full_noise_mask is None
    assert_covered_values_are_permuted(eeg, result.eeg, result.covered_time_mask)
    if mode == 'time_shuffle':
        assert_covered_vectors_are_permuted(eeg, result.eeg, result.covered_time_mask)


@pytest.mark.parametrize('ordinary_enabled', [False, True])
@pytest.mark.parametrize('near_probability', [0., .5, 1.])
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_combined_builder_inherits_mode_and_near_trials_use_original_eeg(ordinary_enabled, near_probability, mode):
    config = dict(enabled=ordinary_enabled, sample_interval_s=.01, geometry='circular_block',
                  probability=1., block_duration_s=.2, noise_mode=mode)
    augmenter = build_target_eeg_augmenter(config, near_full_noise=dict(
        enabled=True, probability=near_probability, coverage_ratio=.95,
    ))
    if near_probability == 0 and not ordinary_enabled:
        assert augmenter is None
        return
    assert augmenter.noise_mode == mode
    if isinstance(augmenter, NearFullNoiseAugmenter) and ordinary_enabled:
        assert augmenter.ordinary_augmenter.noise_mode == mode
    eeg = torch.arange(8 * 3 * 275, dtype=torch.float32).reshape(8, 3, 275)
    torch.manual_seed(12)
    result = augmenter(eeg, return_mask=True)
    assert_covered_values_are_permuted(eeg, result.eeg, result.covered_time_mask)
    if mode == 'time_shuffle':
        assert_covered_vectors_are_permuted(eeg, result.eeg, result.covered_time_mask)
    near = result.near_full_noise_mask
    if near is not None:
        assert (result.covered_time_mask[near].sum(dim=1) == 261).all()
        if near_probability == 1.:
            assert near.all()
        if near_probability == .5:
            assert near.any() and (~near).any()
        for trial in near.nonzero().flatten():
            kept = result.valid_time_mask[trial].nonzero().flatten()
            assert len(kept) == 14 and int(kept[-1] - kept[0]) == 13


@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_disabled_shuffle_preserves_data_masks_and_rng(mode):
    eeg = torch.randn(3, 2, 10)
    for augmenter in (EEGInputAugmenter(sample_interval_s=.01, probability=0., noise_mode=mode),
                      NearFullNoiseAugmenter(probability=0., noise_mode=mode)):
        state = torch.get_rng_state()
        result = augmenter(eeg, return_mask=True)
        assert torch.equal(result.eeg, eeg) and not result.covered_time_mask.any()
        assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize('ordinary_enabled', [False, True])
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_resuming_with_a_different_replacement_mode_is_rejected(ordinary_enabled, mode):
    old = dict(train=dict(temporal_crop_noise=dict(enabled=ordinary_enabled, noise_mode='sample_channel'),
                          near_full_noise=dict(enabled=True)))
    changed = deepcopy(old)
    changed['train']['temporal_crop_noise']['noise_mode'] = mode
    with pytest.raises(ValueError, match='noise_mode'):
        validate_resume_config(old, changed)


@pytest.mark.parametrize('old_mode,new_mode', [tuple(SHUFFLE_MODES), tuple(reversed(SHUFFLE_MODES))])
def test_resuming_cannot_switch_between_shared_and_independent_shuffle(old_mode, new_mode):
    old = dict(train=dict(temporal_crop_noise=dict(enabled=True, noise_mode=old_mode)))
    changed = deepcopy(old)
    changed['train']['temporal_crop_noise']['noise_mode'] = new_mode
    with pytest.raises(ValueError, match='noise_mode'):
        validate_resume_config(old, changed)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_cuda_shuffle_preserves_values_and_backward(mode, dtype):
    eeg = torch.arange(3 * 2 * 32, device='cuda', dtype=dtype).reshape(3, 2, 32).requires_grad_()
    covered = torch.zeros(3, 32, dtype=torch.bool, device='cuda')
    covered[0] = True
    covered[1, 5:25] = True
    actual = shuffle(eeg, covered, mode)
    assert_covered_values_are_permuted(eeg, actual, covered)
    if mode == 'time_shuffle':
        assert_covered_vectors_are_permuted(eeg, actual, covered)
    actual.float().square().mean().backward()
    assert eeg.grad is not None and torch.isfinite(eeg.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
@pytest.mark.parametrize('mode', SHUFFLE_MODES)
def test_cuda_rng_restore_reproduces_masked_shuffle(mode):
    eeg = torch.arange(4 * 3 * 275, dtype=torch.float32, device='cuda').reshape(4, 3, 275)
    covered = torch.zeros(4, 275, dtype=torch.bool, device='cuda')
    covered[1, [0, 1, 2, 130, 131, 132, 274]] = True
    covered[2, :261] = True
    covered[3] = True
    torch.cuda.manual_seed(52)
    state = torch.cuda.get_rng_state()
    first = shuffle(eeg, covered, mode)
    final_state = torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state)
    second = shuffle(eeg, covered, mode)
    assert torch.equal(first, second)
    assert torch.equal(final_state, torch.cuda.get_rng_state())
