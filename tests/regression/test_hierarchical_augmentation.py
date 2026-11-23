"""Hierarchical branch probabilities, roll supervision and deterministic resume."""
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch import nn

from memory_reactivation.training.augmentation import (
    HierarchicalEEGAugmenter, TemporalCropNoiseResult, build_target_eeg_augmenter,
    normalize_augmentation_pipeline_config,
)
from memory_reactivation.training.common import load_checkpoint, validate_resume_config
from memory_reactivation.training.target import (
    fit_target_decoder, run_target_epoch, target_augmentation_diagnostic_totals,
    finalize_target_augmentation_diagnostics,
)
from memory_reactivation.workflows.train import build_model


MIXTURE = {'time_shuffle': .5, 'sample_channel': .25, 'time_shuffle_independent': .25}


def ordinary(probability=.8, **updates):
    result = dict(enabled=True, sample_interval_s=.01, geometry='random_blocks', probability=probability,
        min_blocks=1, max_blocks=1, min_block_duration_s=.02, min_total_duration_s=.05,
        max_total_duration_s=.2, supervision_mode='all_positions', noise_mode='mixture',
        noise_mode_probabilities=MIXTURE.copy())
    result.update(updates)
    return result


def augmenter(probability=.8, *, near=.1, roll=.2, left=.5, maximum=1., **updates):
    return build_target_eeg_augmenter(ordinary(probability, **updates),
        near_full_noise=dict(enabled=True, probability=near, coverage_ratio=.95, time_loss_scale=.25),
        augmentation_pipeline=dict(enabled=True, roll_probability=roll,
            roll_max_shift_s=maximum, roll_left_probability=left))


@pytest.mark.parametrize('options', [None, {'enabled': False}, {'enabled': False, 'roll_probability': .9}])
def test_disabled_pipeline_keeps_legacy_values_masks_and_rng(options):
    config = ordinary()
    near = dict(enabled=True, probability=.3, coverage_ratio=.95)
    legacy = build_target_eeg_augmenter(config, near_full_noise=near)
    candidate = build_target_eeg_augmenter(config, near_full_noise=near, augmentation_pipeline=options)
    eeg = torch.randn(20, 3, 40)
    torch.manual_seed(73); expected = legacy(eeg, return_mask=True); state = torch.get_rng_state()
    torch.manual_seed(73); actual = candidate(eeg, return_mask=True)
    for field in ('eeg', 'covered_time_mask', 'valid_time_mask', 'near_full_noise_mask'):
        assert torch.equal(getattr(expected, field), getattr(actual, field))
    assert torch.equal(state, torch.get_rng_state()) and actual.roll_mask is None


@pytest.mark.parametrize('config', [[], True, {'enabled': 'true'}, {'unknown': 1},
    {'roll_probability': -1}, {'roll_probability': 1.1}, {'roll_probability': True},
    {'roll_probability': float('nan')}, {'roll_max_shift_s': 0}, {'roll_max_shift_s': -1},
    {'roll_max_shift_s': float('inf')}, {'roll_max_shift_s': None}, {'roll_max_shift_s': 10**400},
    {'roll_left_probability': 2}, {'roll_left_probability': True}, {'roll_left_probability': 'invalid'}])
def test_invalid_pipeline_parameters_are_rejected(config):
    with pytest.raises(ValueError, match='augmentation_pipeline'):
        normalize_augmentation_pipeline_config(config)


def test_enabled_pipeline_requires_an_enabled_ordinary_config_and_resolvable_roll():
    with pytest.raises(ValueError, match='temporal_crop_noise.enabled'):
        build_target_eeg_augmenter(None, augmentation_pipeline={'enabled': True})
    with pytest.raises(ValueError, match='shorter than one EEG sample'):
        augmenter(maximum=.001)
    with pytest.raises(ValueError, match='at least two time points'):
        augmenter(1., roll=1.)(torch.ones(2, 3, 1))


@pytest.mark.parametrize('updates,field', [
    ({'enabled': 'false'}, 'enabled'), ({'enabled': 1}, 'enabled'),
    ({'probability': True}, 'probability'), ({'probability': False}, 'probability'),
    ({'probability': 10**400}, 'probability'),
    ({'sample_interval_s': True}, 'sample_interval_s'),
    ({'sample_interval_s': None, 'sample_rate_hz': True}, 'sample_rate_hz'),
])
def test_pipeline_rejects_coerced_global_gate_and_clock_fields(updates, field):
    with pytest.raises(ValueError, match='temporal_crop_noise.' + field):
        build_target_eeg_augmenter(ordinary(**updates), augmentation_pipeline={'enabled': True})


@pytest.mark.parametrize('clock', [
    {'sample_interval_s': .01},
    {'sample_interval_s': None, 'sample_rate_hz': 100.},
])
def test_pipeline_rejects_a_clock_that_would_exceed_the_physical_roll_limit(clock):
    with pytest.raises(ValueError, match='disagree.*time_s'):
        build_target_eeg_augmenter(ordinary(**clock), augmentation_pipeline={'enabled': True},
            time_s=torch.arange(275, dtype=torch.float64) * .02)


@pytest.mark.parametrize('time_s', [
    [], [0.], [[0., .01], [.02, .03]], [0., .01, .03, .04],
    [.02, .01, 0.], [0., .01, .01], [0., float('nan')],
    [0., float('inf')], [0.+0.j, .01+0.j], 'invalid',
])
def test_pipeline_rejects_invalid_or_irregular_physical_time_axes(time_s):
    with pytest.raises(ValueError, match='time_s'):
        build_target_eeg_augmenter(ordinary(), augmentation_pipeline={'enabled': True}, time_s=time_s)


@pytest.mark.parametrize('sample_rate_hz', [100., 1000.])
@pytest.mark.parametrize('infer_interval', [False, True])
def test_pipeline_accepts_regular_float32_time_axes_despite_rounding(sample_rate_hz, infer_interval):
    interval = 1. / sample_rate_hz
    time_s = 2. + torch.arange(275, dtype=torch.float32) * interval
    config = ordinary(sample_interval_s=None if infer_interval else interval)
    aug = build_target_eeg_augmenter(config, augmentation_pipeline={'enabled': True}, time_s=time_s)
    assert aug.sample_interval_s == pytest.approx(interval, rel=1e-4)


def test_pipeline_rejects_overflow_in_finite_duration_to_sample_conversion():
    with pytest.raises(ValueError, match='augmentation_pipeline.roll_max_shift_s'):
        augmenter(maximum=1e308)


def test_disabled_pipeline_keeps_legacy_global_gate_and_clock_coercions():
    config = ordinary(enabled='false', probability=True)
    time_s = torch.arange(275, dtype=torch.float64) * .02
    legacy = build_target_eeg_augmenter(config, time_s=time_s)
    disabled = build_target_eeg_augmenter(config, time_s=time_s,
        augmentation_pipeline={'enabled': False})
    assert legacy.probability == disabled.probability == 1.
    assert legacy.sample_interval_s == disabled.sample_interval_s == .01


def test_zero_global_probability_is_identity_and_consumes_no_rng():
    aug = augmenter(0.)
    eeg = torch.randn(5, 3, 275)
    state = torch.get_rng_state()
    result = aug(eeg, return_mask=True)
    assert torch.equal(result.eeg, eeg) and result.eeg.data_ptr() != eeg.data_ptr()
    assert torch.equal(state, torch.get_rng_state())
    assert not result.roll_mask.any() and not result.near_full_noise_mask.any()
    assert not result.covered_time_mask.any() and result.valid_time_mask.all()


@pytest.mark.parametrize('left', [0., .5, 1.])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_roll_is_shared_across_channels_bounded_nonzero_and_not_noise(left, dtype):
    eeg = (torch.arange(512 * 3 * 275, dtype=torch.float64) % 1000).reshape(512, 3, 275).to(dtype)
    assert torch.isfinite(eeg).all()
    original = eeg.clone()
    aug = augmenter(1., roll=1., left=left)
    torch.manual_seed(21)
    result = aug(eeg, return_mask=True)
    assert result.roll_mask.all() and not result.near_full_noise_mask.any()
    assert not result.covered_time_mask.any() and result.valid_time_mask.all()
    assert result.roll_shifts.abs().min() >= 1 and result.roll_shifts.abs().max() <= 100
    if left in (0., 1.):
        assert ((result.roll_shifts < 0) == bool(left)).all()
    else:
        assert .4 < (result.roll_shifts < 0).float().mean() < .6
    for i, shift in enumerate(result.roll_shifts):
        assert torch.equal(result.eeg[i], torch.roll(original[i], int(shift), dims=-1))
    assert torch.equal(eeg, original) and result.eeg.dtype == eeg.dtype


def test_hierarchical_probabilities_mixture_routing_and_no_double_local_gate():
    torch.manual_seed(31)
    eeg = torch.randn(12000, 2, 40)
    aug = augmenter()
    original_draw = torch.multinomial
    choices = []

    def draw(weights, count, **kwargs):
        assert weights.tolist() == [.25, .5, .25]
        result = original_draw(weights, count, **kwargs)
        choices.append(result)
        return result

    with patch('torch.multinomial', side_effect=draw):
        result = aug(eeg, return_mask=True)
    roll, near, covered = result.roll_mask, result.near_full_noise_mask, result.covered_time_mask
    replacement = covered.any(dim=1)
    local = replacement & ~near
    clean = ~roll & ~replacement
    assert not (roll & replacement).any() and not (roll & near).any()
    for mask, expected in ((roll,.16),(near,.064),(local,.576),(clean,.2)):
        assert float(mask.float().mean()) == pytest.approx(expected, abs=.012)
    assert float(near.sum() / replacement.sum()) == pytest.approx(.1, abs=.015)
    assert len(choices) == 1 and len(choices[0]) == int(replacement.sum())
    frequencies = torch.bincount(choices[0], minlength=3) / len(choices[0])
    torch.testing.assert_close(frequencies, torch.tensor([.25,.5,.25]), rtol=0, atol=.02)
    assert (covered[near].sum(dim=1) == 38).all()
    kept = (~covered & replacement[:,None])[:,None,:].expand_as(eeg)
    assert torch.equal(result.eeg[kept], eeg[kept])
    assert torch.equal(result.eeg[clean], eeg[clean])
    assert (result.roll_shifts[~roll] == 0).all()


def test_local_branch_executes_after_one_global_probability_draw():
    torch.manual_seed(27)
    eeg = torch.randn(4000, 2, 40)
    result = augmenter(.3, near=0., roll=0.)(eeg, return_mask=True)
    assert float(result.covered_time_mask.any(dim=1).float().mean()) == pytest.approx(.3, abs=.02)


def test_replacement_reads_original_full_batch_once_after_both_coverages():
    from memory_reactivation.training import augmentation as module
    eeg = torch.randn(64, 3, 40)
    original = module._replace_with_noise
    calls = []

    def replace(source, covered, **kwargs):
        if kwargs['noise_mode'] == 'mixture':
            calls.append((source, covered.clone()))
        return original(source, covered, **kwargs)

    with patch.object(module, '_replace_with_noise', side_effect=replace):
        result = augmenter(1., near=.5, roll=0.)(eeg, return_mask=True)
    assert len(calls) == 1 and calls[0][0] is eeg
    assert torch.equal(calls[0][1], result.covered_time_mask)
    assert result.near_full_noise_mask.any() and (~result.near_full_noise_mask).any()
    for selected in result.valid_time_mask[result.near_full_noise_mask]:
        indices = selected.nonzero().flatten()
        assert int(indices[-1] - indices[0] + 1) == len(indices)


def test_roll_rng_state_can_be_restored_exactly():
    eeg = torch.randn(32, 3, 275)
    aug = augmenter()
    state = torch.get_rng_state()
    expected = aug(eeg, return_mask=True); after = torch.get_rng_state()
    torch.set_rng_state(state)
    actual = aug(eeg, return_mask=True)
    for name in ('eeg','covered_time_mask','near_full_noise_mask','roll_mask','roll_shifts'):
        assert torch.equal(getattr(actual,name),getattr(expected,name))
    assert torch.equal(after,torch.get_rng_state())


class BranchProbe(nn.Module):
    fusion_enabled = True

    def __init__(self):
        super().__init__()
        self.time = nn.Parameter(torch.zeros(3, 20, 3))
        self.pooled = nn.Parameter(torch.zeros(3, 2))
        self.fusion = nn.Parameter(torch.zeros(3, 2))
        self.gates = nn.Parameter(torch.zeros(3, 20))

    def time_class_count(self, task): return 3
    def time_no_class_index(self, task): return 2

    def forward(self, eeg, **kwargs):
        weights = self.gates.sigmoid() / 20
        return dict(time_logits=self.time,pooled_logits=self.pooled,fusion_logits=self.fusion,
            attention_weights=weights,effective_attention_weights=weights)


class MixedBranches(nn.Module):
    def forward(self, eeg, return_mask=False):
        covered = torch.zeros(3,20,dtype=torch.bool,device=eeg.device)
        covered[1,:19] = True
        covered[2,5:10] = True
        return TemporalCropNoiseResult(eeg,~covered,covered,
            torch.tensor([False,True,False],device=eeg.device),
            torch.tensor([True,False,False],device=eeg.device),
            torch.tensor([-1,0,0],device=eeg.device))


def test_mixed_losses_keep_roll_classification_and_time_but_zero_its_attention_penalty():
    model = BranchProbe()
    batch = dict(eeg=torch.randn(3,2,20),target=torch.tensor([0,1,1]))
    metrics = run_target_epoch(model,[batch],'choice',torch.arange(20),
        {'time':.5,'pooled':1.,'fusion':1.,'attention_mask':.7},None,1.,0.,torch.device('cpu'),False,
        eeg_augmenter=MixedBranches(),optimizer=torch.optim.SGD(model.parameters(),lr=0.),
        scaler=torch.amp.GradScaler('cpu',enabled=False),near_full_noise_time_loss_scale=.25,
        noise_supervision_mode='all_positions',progress_bar_enabled=False)
    assert model.pooled.grad[0].abs().sum() > 0 and model.fusion.grad[0].abs().sum() > 0
    assert model.pooled.grad[1].abs().sum() == model.fusion.grad[1].abs().sum() == 0
    assert (model.time.grad[0,:,0] < 0).all() and (model.time.grad[0,:,2] > 0).all()
    assert float(model.time.grad[1].abs().sum() / model.time.grad[0].abs().sum()) == pytest.approx(.25)
    assert model.gates.grad[0].abs().sum() == 0
    assert model.gates.grad[1,:19].sum() > 0 and model.gates.grad[2,5:10].sum() > 0
    assert metrics['trial_loss_sample_count'] == 2
    assert metrics['roll_fraction'] == pytest.approx(1/3)
    assert metrics['replacement_trial_fraction'] == pytest.approx(2/3)
    assert metrics['near_full_noise_fraction'] == metrics['local_noise_fraction'] == pytest.approx(1/3)
    assert metrics['near_full_noise_fraction_among_replacements'] == .5
    assert metrics['augmented_trial_fraction'] == 1.
    assert metrics['roll_left_fraction'] == 1.


def test_diagnostics_count_full_input_transformations_outside_attention_crop():
    covered = torch.zeros(2,20,dtype=torch.bool);covered[0,:2]=True
    totals = target_augmentation_diagnostic_totals(torch.zeros(2,20,3),torch.ones(2,10)/10,
        torch.arange(10,20),covered,configured_topk_count=5,roll_mask=torch.tensor([False,True]))
    stats = finalize_target_augmentation_diagnostics(totals)
    assert totals['replacement_trial_count'] == 1 and stats['augmented_trial_fraction'] == 1.
    assert stats['covered_effective_attention_mass_fraction'] == 0.


def test_resume_identity_handles_disabled_defaults_and_rejects_pipeline_changes():
    legacy = dict(train=dict(temporal_crop_noise=ordinary()))
    disabled = deepcopy(legacy);disabled['train']['augmentation_pipeline']={'enabled':False,'roll_probability':.9}
    validate_resume_config(legacy,disabled)
    enabled = deepcopy(legacy);enabled['train']['augmentation_pipeline']={'enabled':True}
    with pytest.raises(ValueError,match='augmentation_pipeline'):validate_resume_config(legacy,enabled)
    explicit=deepcopy(enabled);explicit['train']['augmentation_pipeline']=normalize_augmentation_pipeline_config({'enabled':True})
    validate_resume_config(enabled,explicit)
    for key,value in [('enabled',False),('roll_probability',0.),('roll_max_shift_s',.5),('roll_left_probability',0.)]:
        changed=deepcopy(explicit);changed['train']['augmentation_pipeline'][key]=value
        with pytest.raises(ValueError,match='augmentation_pipeline'):validate_resume_config(explicit,changed)




def test_new_branch_rates_are_mapped_to_swanlab():
    from memory_reactivation.training.swanlab_logger import target_swanlab_metrics
    result=target_swanlab_metrics({'train_roll_fraction':.16,'train_local_noise_fraction':.576,
        'train_near_full_noise_fraction_among_replacements':.1})
    assert result['train/augmentation/roll_fraction']==.16
    assert result['train/augmentation/local_noise_fraction']==.576
    assert result['train/augmentation/near_full_noise_fraction_among_replacements']==.1


@pytest.mark.parametrize('problem',['covered','near','zero_shift','nonroll_shift'])
def test_inconsistent_roll_metadata_is_rejected_before_training(problem):
    class Invalid(MixedBranches):
        def forward(self,eeg,return_mask=False):
            result=super().forward(eeg,return_mask=return_mask)
            if problem=='covered':result.covered_time_mask[0,0]=True
            elif problem=='near':result.near_full_noise_mask[0]=True
            elif problem=='zero_shift':result.roll_shifts[0]=0
            else:result.roll_shifts[2]=1
            return result
    model=BranchProbe()
    with pytest.raises(ValueError,match='Roll|roll'):
        run_target_epoch(model,[{'eeg':torch.randn(3,2,20),'target':torch.tensor([0,1,1])}],
            'choice',torch.arange(20),{'time':1.,'pooled':1.,'attention_mask':1.},None,1.,0.,torch.device('cpu'),False,
            eeg_augmenter=Invalid(),optimizer=torch.optim.SGD(model.parameters(),lr=0.),
            scaler=torch.amp.GradScaler('cpu',enabled=False),noise_supervision_mode='all_positions',progress_bar_enabled=False)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_cuda_hierarchy_keeps_rng_devices_and_input_gradients(dtype):
    eeg=torch.randn(64,3,275,device='cuda',dtype=dtype,requires_grad=True)
    aug=augmenter()
    cpu_state=torch.get_rng_state()
    state=torch.cuda.get_rng_state()
    first=aug(eeg,return_mask=True);after=torch.cuda.get_rng_state()
    torch.cuda.set_rng_state(state);second=aug(eeg,return_mask=True)
    for field in ('eeg','covered_time_mask','near_full_noise_mask','roll_mask','roll_shifts'):
        assert torch.equal(getattr(first,field),getattr(second,field))
    assert torch.equal(after,torch.cuda.get_rng_state())
    assert torch.equal(cpu_state,torch.get_rng_state())
    first.eeg.float().square().mean().backward()
    assert torch.isfinite(eeg.grad).all() and first.eeg.device==eeg.device


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
@pytest.mark.parametrize('encoder_type',['cnn','window_transformer'])
def test_cuda_amp_training_and_validation_use_the_expected_branches(encoder_type):
    torch.manual_seed(91)
    encoder=dict(type=encoder_type,in_channels=2,latent_dim=4,dropout=0.)
    if encoder_type=='cnn':encoder.update(num_blocks=1,kernel_size=3,expansion_ratio=1)
    else:encoder.update(num_layers=1,window_size=3,num_heads=2,ffn_ratio=2,
        attention_pattern='centered_local',position_encoding_mode='sinusoidal_qk')
    model=build_model(dict(encoder=encoder,attention_mode='gated_scorer',attention_dim=4,
        time_no_class_enabled=True,classifier_num_layers=1,classifier_dropout=0.),time_point_count=275).cuda()
    batch=dict(eeg=torch.randn(32,2,275,device='cuda'),target=torch.arange(32,device='cuda')%2,
        subject_id=torch.arange(32,device='cuda')//16,current_stimulus=torch.ones(32,device='cuda',dtype=torch.long))
    aug=augmenter(1.,near=.5,roll=.5)
    kwargs=dict(model=model,loader=[batch,batch],task='choice',attention_indices=torch.arange(275,device='cuda'),
        loss_weights={'time':.5,'pooled':1.,'attention_mask':.7},top_k=None,top_ratio=.8,label_smoothing=0.,
        device=torch.device('cuda'),amp=True,eeg_augmenter=aug,noise_supervision_mode='all_positions',
        near_full_noise_time_loss_scale=.25,progress_bar_enabled=False)
    result=run_target_epoch(**kwargs,optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4),scaler=torch.amp.GradScaler('cuda'))
    assert result['roll_sample_count']>0 and result['near_full_noise_sample_count']>0
    assert torch.isfinite(torch.tensor(result['loss'])) and all(torch.isfinite(p).all() for p in model.parameters())
    validation=run_target_epoch(**kwargs)
    assert validation['augmented_trial_fraction']==validation['near_full_noise_fraction']==0.


@pytest.mark.parametrize('encoder_type',['cnn','window_transformer'])
def test_cpu_fit_and_resume_reproduce_with_all_hierarchical_branches(tmp_path,monkeypatch,encoder_type):
    from memory_reactivation.training import trainer as module
    encoder=dict(type=encoder_type,in_channels=2,latent_dim=4,dropout=.1)
    if encoder_type=='cnn':encoder.update(num_blocks=1,kernel_size=3,expansion_ratio=1)
    else:encoder.update(num_layers=1,window_size=3,num_heads=2,ffn_ratio=2,
        attention_pattern='centered_local',position_encoding_mode='sinusoidal_qk')
    model_config=dict(encoder=encoder,attention_mode='gated_scorer',attention_dim=4,
        time_no_class_enabled=True,attention_dropout=.1,classifier_num_layers=1,classifier_dropout=.1,
        fusion=dict(enabled=True))
    config=dict(model=model_config,train=dict(iterations=4,validation_interval_iterations=1,
        log_interval_iterations=1,learning_rate=1e-3,lr_schedule='constant',weight_decay=.1,
        amp=False,progress_bar=False,time_loss_type='bce_one_vs_rest',pooled_loss_type='focal_bce',
        pooled_focal_gamma=1.,time_loss_weight=.5,pooled_loss_weight=1.,fusion_loss_weight=.5,
        attention_mask_penalty_weight=.7,attention_topk_ratio=.8,
        checkpoint_metric='val_pooled_accuracy',checkpoint_mode='max',
        temporal_crop_noise=ordinary(),near_full_noise=dict(enabled=True,probability=.3,coverage_ratio=.95,time_loss_scale=.25),
        augmentation_pipeline={'enabled':True}))
    generator=torch.Generator().manual_seed(73)
    batches=[dict(eeg=torch.randn(16,2,20,generator=generator),target=torch.arange(16)%2,
        subject_id=torch.arange(16)%2,current_stimulus=torch.ones(16,dtype=torch.long)) for _ in range(3)]
    def new_model():return build_model(deepcopy(model_config),time_point_count=20)
    torch.manual_seed(71);initial=deepcopy(new_model().state_dict())
    def fit(model,directory,resume=None):
        return fit_target_decoder(model,batches,batches,'choice',torch.arange(20),config,directory,torch.device('cpu'),resume_from=resume)
    full,interrupted=tmp_path/'full',tmp_path/'interrupted';full.mkdir();interrupted.mkdir()
    for directory in (full,interrupted):
        model=new_model();model.load_state_dict(initial);torch.manual_seed(89)
        if directory==full:
            history=fit(model,directory)
            assert sum(row['train_roll_sample_count'] for row in history)>0
            assert sum(row['train_near_full_noise_sample_count'] for row in history)>0
            assert all(row['val_augmented_trial_fraction']==0. for row in history)
        else:
            append=module.append_metrics_csv
            def stop(path,row):
                if path.name=='train_metrics.csv' and int(row['iteration'])==3:raise InterruptedError('synthetic interruption')
                append(path,row)
            with monkeypatch.context() as patcher:
                patcher.setattr(module,'append_metrics_csv',stop)
                with pytest.raises(InterruptedError):fit(model,directory)
    checkpoint=interrupted/'last.ckpt';assert load_checkpoint(checkpoint)['iteration']==2
    fit(new_model(),interrupted,checkpoint)
    expected,actual=load_checkpoint(full/'last.ckpt'),load_checkpoint(checkpoint)
    for key,value in expected['model_state'].items():torch.testing.assert_close(actual['model_state'][key],value,rtol=0,atol=0)
    assert torch.equal(expected['rng_states']['torch_cpu'],actual['rng_states']['torch_cpu'])
    for name in ('metrics.csv','train_metrics.csv'):assert (full/name).read_bytes()==(interrupted/name).read_bytes()
