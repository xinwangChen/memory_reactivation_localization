"""Noise-pad semantics and integration, adapted from the original Target tests."""
from copy import deepcopy
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from memory_reactivation.artifacts.checkpoints import load_model, save_model
from memory_reactivation.artifacts.exports import describe_export, copy_verified_exports
from memory_reactivation.config import parse_request
from memory_reactivation.contracts import PredictionSpec
from memory_reactivation.evaluation.export import export_target_outputs
from memory_reactivation.evaluation.predict import iter_predictions
from memory_reactivation.models.encoder import RightLocalTransformerBlock, build_eeg_encoder
from memory_reactivation.models.adapters import boundary_noise_forward_kwargs, forward_batch
from memory_reactivation.models.factory import build_model as build_target
from memory_reactivation.training.common import load_checkpoint, validate_resume_config
from memory_reactivation.training.trainer import run_target_epoch
from memory_reactivation.training import trainer as training
from memory_reactivation.workflows import evaluate

def encoder_config(**overrides):
    config = dict(type="window_transformer", in_channels=3, latent_dim=8,
                  num_blocks=1, window_size=3, num_heads=2, dropout=0.,
                  attention_pattern="centered_local", position_encoding_mode="sinusoidal_qk",
                  boundary_mode="noise_pad", boundary_noise_seed=42)
    config.update(overrides)
    return config



def model_config(**overrides):
    config = dict(encoder=encoder_config(), classifier_num_layers=1, classifier_dropout=0.,
                  attention_mode="gated_scorer", attention_dim=4, attention_dropout=0.,
                  attention_score_centering=True, attention_gate_activation="tanh_plus_one",
                  time_no_class_enabled=True, time_head_mode="independent_linear",
                  fusion=dict(enabled=False))
    config.update(overrides)
    return config



def batch(n=4, t=11):
    return dict(eeg=torch.randn(n, 3, t), target=torch.arange(n) % 2,
                sample_weight=torch.ones(n), subject_id=torch.zeros(n, dtype=torch.long),
                record_id=torch.zeros(n, dtype=torch.long), exp=torch.ones(n, dtype=torch.long),
                trial0=torch.arange(n), lag=torch.ones(n, dtype=torch.long),
                current_stimulus=torch.ones(n, dtype=torch.long))



def activate(encoder):
    with torch.no_grad():
        for block in encoder.temporal_blocks:
            nn.init.normal_(block.attention.out_proj.weight, std=.15)
            nn.init.normal_(block.ffn[-2].weight, std=.15)


def test_real_heads_and_pool_support_exclude_padding():
    model = build_target(model_config(pool_valid_context_only=True), time_point_count=11).train()
    activate(model.encoder)
    data = batch()
    seen = []
    head_hook = model.class_heads['choice'].register_forward_pre_hook(
        lambda _, args: seen.append(args[0].shape[1]))
    def attention_input(_, args, kwargs):
        assert args[0].shape == (4, 11, 8)
        assert kwargs['score_latent'].shape == (4, 11, 8)
        torch.testing.assert_close(kwargs['score_latent'].mean(1), torch.zeros(4, 8), atol=1e-6, rtol=0)
    attention_hook = model.attention.register_forward_pre_hook(attention_input, with_kwargs=True)
    outputs = model(data['eeg'], 'choice', **boundary_noise_forward_kwargs(model, data))
    assert seen == [11] and outputs['time_logits'].shape == (4, 11, 3)
    assert outputs['effective_attention_weights'].shape == (4, 11)
    assert outputs['attention_valid_mask'].all()
    assert outputs['pooled_logits'].shape == (4, 2)
    outputs['time_logits'].square().mean().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.encoder.spatial_projection[0].weight.grad.abs().sum() > 0
    head_hook.remove()
    attention_hook.remove()


def test_exports_and_streaming_controls_reuse_trial_padding(tmp_path):
    model = build_target(model_config(), time_point_count=11).eval()
    activate(model.encoder)
    data = batch()
    expected_noise = boundary_noise_forward_kwargs(model, data)['boundary_noise_eeg']
    seen = []
    hook = model.encoder.spatial_projection.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone()) if args[0].shape[-1] == 2 else None)
    real_arguments = None
    for kind in ['real', 'gaussian', 'time_shuffle', 'time_roll']:
        arguments = dict(model=model, loader=[data], task='choice', device=torch.device('cpu'), amp=False,
            output_dir=tmp_path/'exports', split='val', time_s=torch.arange(11)/100,
            progress_bar_enabled=False, input_kind=kind, gaussian_seed=43,
            time_shuffle_seed=7, time_roll_shift_samples=3, attention_indices=torch.arange(11))
        if kind == 'real':
            real_arguments = arguments
        description = describe_export(export_target_outputs, **arguments)
        assert description['boundary_noise']['evaluation_seed'] == 42
        export_target_outputs(**arguments)
        prefix = 'val_' if kind == 'real' else f'val_{kind}_'
        with (tmp_path/'exports'/f'{prefix}time_accuracy.csv').open() as stream:
            assert len({row['time_index'] for row in csv.DictReader(stream)}) == 11
        with (tmp_path/'exports'/f'{prefix}attention_weights.csv').open() as stream:
            assert len(list(csv.DictReader(stream))) == 4 * 11
        metadata = json.loads((tmp_path/'exports'/f'{prefix}export_manifest.json').read_text())
        assert metadata['parameters']['boundary_noise']['distribution'] == 'standard_normal'
        prediction = next(iter_predictions(model, [data], PredictionSpec(
            task='choice', input_kind=kind, control_seed=7, roll_shift=3)))
        np.testing.assert_array_equal(prediction.identity['exp'], data['exp'].numpy())
        assert prediction.metadata['boundary_noise'] == description['boundary_noise']
        assert prediction.outputs['time_logits'].shape[1] == 11
    assert len(seen) == 8 and all(torch.equal(pad, expected_noise) for pad in seen)
    hook.remove()
    model.encoder.boundary_noise_seed = 43
    assert describe_export(export_target_outputs, **arguments) != description
    new_parameters = describe_export(export_target_outputs, **real_arguments)
    with pytest.raises(ValueError):
        copy_verified_exports(tmp_path/'exports', tmp_path/'reuse', decoder='target', splits=['val'], input_kinds=['real'],
                              expected_parameters={('val', 'real'): new_parameters})


@pytest.mark.parametrize('missing', ['exp', 'subject_id', 'trial0'])
def test_prediction_requires_stable_physical_trial_identity(missing):
    model = build_target(model_config(), time_point_count=11).eval()
    data = batch()
    del data[missing]
    with pytest.raises(ValueError, match='stable trial metadata'):
        next(iter_predictions(model, [data], PredictionSpec(task='choice')))


@pytest.mark.parametrize('field', ['exp', 'subject_id', 'trial0'])
@pytest.mark.parametrize('bad', ['float', 'bool', 'negative'])
def test_trial_metadata_cannot_be_promoted_from_invalid_types(field, bad):
    model = build_target(model_config(), time_point_count=11).eval()
    data = batch()
    data[field] = data[field].float() if bad == 'float' else data[field].bool() if bad == 'bool' else -torch.ones(4, dtype=torch.long)
    with pytest.raises(ValueError, match='nonnegative integers'):
        boundary_noise_forward_kwargs(model, data)


def test_config_checkpoint_and_prediction_cli_round_trip(tmp_path):
    config = dict(task='choice', seed=42, data={}, model=model_config(), train={})
    config_path = tmp_path/'request.yaml'
    config_path.write_text(yaml.safe_dump(config))
    request = parse_request(config_path)
    model = build_target(request.config['model'], time_point_count=11).eval()
    activate(model.encoder)
    checkpoint = tmp_path/'model.ckpt'
    save_model(checkpoint, model, config=config)
    loaded = load_model(checkpoint)
    assert loaded.spec.model_config['encoder']['boundary_noise_seed'] == 42
    data = batch()
    with torch.inference_mode():
        expected = forward_batch(model, data, task='choice')
        actual = forward_batch(loaded.model, data, task='choice')
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    inputs = {key: value.numpy() for key, value in data.items()}
    source, output = tmp_path/'inputs.npz', tmp_path/'predictions.npz'
    np.savez(source, **inputs)
    evaluate.main([str(checkpoint), '--input', str(source), '--output', str(output), '--batch-size', '2'])
    with np.load(output) as archive:
        np.testing.assert_allclose(archive['pooled_logits'], expected['pooled_logits'].numpy(), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(archive['exp'], inputs['exp'])
        assert json.loads(str(archive['metadata_json']))['prediction']['boundary_noise']['evaluation_seed'] == 42
    # Invalid inputs must not replace a previously published prediction.
    saved = output.read_bytes()
    del inputs['exp']
    np.savez(source, **inputs)
    with pytest.raises(ValueError, match='stable trial metadata'):
        evaluate.main([str(checkpoint), '--input', str(source), '--output', str(output)])
    assert output.read_bytes() == saved
    changed = deepcopy(config)
    changed['model']['encoder']['boundary_noise_seed'] = 43
    with pytest.raises(ValueError):
        load_model(checkpoint, config=changed)


def test_matched_null_training_forwards_share_one_sampled_pad():
    model = build_target(model_config(), time_point_count=11)
    activate(model.encoder)
    data = batch()
    seen = []
    hook = model.encoder.spatial_projection.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone()) if args[0].shape[-1] == 2 else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    result = run_target_epoch(model=model, loader=[data], task='choice', attention_indices=torch.arange(11),
        loss_weights=dict(time=.5, pooled=1., null_attention=.1, real_noise_margin=.1),
        top_k=None, top_ratio=.8, label_smoothing=0., device=torch.device('cpu'), amp=False,
        progress_bar_enabled=False, loss_type='bce_one_vs_rest', optimizer=optimizer,
        scaler=torch.amp.GradScaler('cpu', enabled=False))
    hook.remove()
    assert np.isfinite(result['loss']) and len(seen) == 3
    assert all(torch.equal(value, seen[0]) for value in seen[1:])



@pytest.mark.parametrize("window", [1, 3, 5, 7])
@pytest.mark.parametrize("time", [1, 2, 11, 275])
def test_full_windows_preserve_every_original_real_to_real_mask_entry(window, time):
    kw = dict(channels=8, window_size=window, time_point_count=time, num_heads=2,
              dropout=0., attention_pattern="centered_local")
    old = RightLocalTransformerBlock(**kw)
    new = RightLocalTransformerBlock(**kw, boundary_mode="noise_pad")
    radius = window // 2
    assert new.attention_mask.shape == (time, time + 2 * radius)
    assert torch.equal(new.attention_mask[:, radius:radius + time], old.attention_mask)
    assert torch.equal((~new.attention_mask).sum(1), torch.full((time,), window))
    torch.testing.assert_close(new.key_position_encoding[:, radius:radius + time], old.position_encoding,
                               rtol=0, atol=0)
    torch.testing.assert_close(new.position_encoding, old.position_encoding, rtol=0, atol=0)



@pytest.mark.parametrize("position_mode", ["none", "sinusoidal_qk"])
def test_padding_only_provides_kv_and_only_changes_the_matching_edge(position_mode):
    torch.manual_seed(7)
    block = RightLocalTransformerBlock(channels=8, window_size=3, time_point_count=11,
        num_heads=2, dropout=0., attention_pattern="centered_local",
        position_encoding_mode=position_mode, boundary_mode="noise_pad").eval()
    with torch.no_grad():
        nn.init.normal_(block.attention.out_proj.weight, std=.15)
    x, noise = torch.randn(2, 8, 11), torch.randn(2, 8, 2)
    seen = []
    handle = block.attention.register_forward_pre_hook(lambda _, args: seen.append(tuple(a.shape for a in args[:3])))
    baseline = block(x, boundary_latent=noise)
    handle.remove()
    assert seen == [(torch.Size([2, 11, 8]), torch.Size([2, 13, 8]), torch.Size([2, 13, 8]))]
    for side, edge in [(0, 0), (1, 10)]:
        changed = noise.clone()
        changed[:, :, side] += torch.randn(2, 8) * 3
        actual = block(x, boundary_latent=changed)
        assert (actual[:, :, edge] - baseline[:, :, edge]).abs().max() > 1e-5
        untouched = torch.arange(11) != edge
        torch.testing.assert_close(actual[:, :, untouched], baseline[:, :, untouched], rtol=0, atol=1e-7)
    # A true point immediately outside query 0's old window must stay invisible.
    changed = x.clone()
    changed[:, :, 2] += torch.randn(2, 8) * 3
    actual = block(changed, boundary_latent=noise)
    torch.testing.assert_close(actual[:, :, 0], baseline[:, :, 0], rtol=0, atol=1e-7)
    # Chunking must also respect rectangular Q/KV geometry.
    block.attention_batch_limit = 1
    torch.testing.assert_close(block(x, boundary_latent=noise), baseline, rtol=1e-6, atol=1e-7)



@pytest.mark.parametrize("layers", [1, 2, 3])
def test_multiple_blocks_keep_only_real_queries_and_preserve_the_structural_rf(layers):
    torch.manual_seed(13)
    encoder = build_eeg_encoder(encoder_config(num_blocks=layers), time_point_count=11).eval()
    activate(encoder)
    eeg, noise = torch.randn(2, 3, 11), torch.randn(2, 3, 2)
    observed = []
    handles = [block.attention.register_forward_pre_hook(
        lambda _, args: observed.append((args[0].shape[1], args[1].shape[1])))
        for block in encoder.temporal_blocks]
    baseline = encoder.forward_features(eeg, boundary_noise_eeg=noise)
    for handle in handles: handle.remove()
    assert observed == [(11, 13)] * layers
    assert all(value.shape == (2, 11, 8) for value in baseline['block_latents'])
    changed_noise = noise.clone()
    changed_noise[:, :, 0] += torch.randn(2, 3) * 3
    changed = encoder(eeg, boundary_noise_eeg=changed_noise)
    assert (changed[:, 0] - baseline['latent'][:, 0]).abs().max() > 1e-5
    torch.testing.assert_close(changed[:, layers:], baseline['latent'][:, layers:], rtol=0, atol=2e-7)
    assert encoder.receptive_field_samples == 2 * layers + 1



def test_sampling_is_independent_and_training_rng_can_be_restored():
    encoder = build_eeg_encoder(encoder_config(in_channels=64), time_point_count=11).train()
    eeg = torch.zeros(512, 64, 11)
    torch.manual_seed(19)
    state = torch.get_rng_state()
    a = encoder.sample_boundary_noise(eeg)
    b = encoder.sample_boundary_noise(eeg)
    assert not torch.equal(a, b) and not torch.equal(a[:, :, 0], a[:, :, 1])
    assert abs(a.mean().item()) < .02 and abs(a.std().item() - 1) < .02
    torch.set_rng_state(state)
    assert torch.equal(a, encoder.sample_boundary_noise(eeg))



def test_evaluation_pad_is_trial_stable_without_using_eeg_values_or_global_rng():
    encoder = build_eeg_encoder(encoder_config(), time_point_count=11).eval()
    eeg = torch.randn(4, 3, 11)
    identities = torch.tensor([[1, 2, i] for i in range(4)])
    rng = torch.get_rng_state()
    a = encoder.sample_boundary_noise(eeg, trial_ids=identities)
    assert torch.equal(torch.get_rng_state(), rng)
    order = torch.tensor([3, 0, 2, 1])
    b = encoder.sample_boundary_noise(eeg[order] + 100, trial_ids=identities[order])
    assert torch.equal(b, a[order])
    c = torch.cat([encoder.sample_boundary_noise(eeg[i:i+1], trial_ids=identities[i:i+1]) for i in range(4)])
    assert torch.equal(a, c)
    assert not torch.equal(a[0], a[1])
    encoder.boundary_noise_seed += 1
    assert not torch.equal(a, encoder.sample_boundary_noise(eeg, trial_ids=identities))
    for bad in [None, torch.ones(4), identities.float(), identities[:3]]:
        with pytest.raises(ValueError, match='trial_ids'):
            encoder.sample_boundary_noise(eeg, trial_ids=bad)



def test_disabled_mode_has_no_rng_cost_and_config_and_checkpoint_modes_are_explicit():
    old = build_eeg_encoder(encoder_config(boundary_mode='truncate'), time_point_count=11)
    new = build_eeg_encoder(encoder_config(), time_point_count=11)
    x = torch.randn(2, 3, 11)
    rng = torch.get_rng_state()
    assert old.sample_boundary_noise(x) is None
    assert boundary_noise_forward_kwargs(nn.Identity(), dict(eeg=x)) == {}
    assert torch.equal(rng, torch.get_rng_state())
    assert sum(p.numel() for p in old.parameters()) == sum(p.numel() for p in new.parameters())
    for a, b in [(old, new), (new, old)]:
        with pytest.raises(RuntimeError, match='architecture_signature'):
            a.load_state_dict(b.state_dict(), strict=True)
    rebuilt = build_eeg_encoder(new.resolved_config())
    rebuilt.load_state_dict(new.state_dict(), strict=True)
    assert rebuilt.resolved_config() == new.resolved_config()
    before = dict(model=dict(encoder=encoder_config(boundary_mode='truncate')))
    with pytest.raises(ValueError, match='boundary_mode'):
        validate_resume_config(before, dict(model=dict(encoder=encoder_config())))
    with pytest.raises(ValueError, match='boundary_noise_seed'):
        validate_resume_config(dict(model=dict(encoder=encoder_config())),
                               dict(model=dict(encoder=encoder_config(boundary_noise_seed=43))))



@pytest.mark.parametrize('key', ['encoder', 'resolved_encoder'])
@pytest.mark.parametrize('mode', ['truncate', 'noise_pad', 'reflect'])
def test_resume_accepts_equivalent_noise_seed_defaults_without_mutating_inputs(key, mode):
    before = {'model': {key: {'type': 'window_transformer', 'boundary_mode': mode}}}
    after = deepcopy(before)
    after['model'][key]['boundary_noise_seed'] = 42
    original_before, original_after = deepcopy(before), deepcopy(after)
    validate_resume_config(before, after)
    validate_resume_config(after, before)
    assert before == original_before and after == original_after
    if mode != 'noise_pad':
        after['model'][key]['boundary_noise_seed'] = 43
        validate_resume_config(before, after)
    else:
        after['model'][key]['boundary_noise_seed'] = 43
        with pytest.raises(ValueError, match='boundary_noise_seed'):
            validate_resume_config(before, after)



def test_legacy_missing_boundary_mode_resumes_with_explicit_truncate():
    before = {'model': {'encoder': {'type': 'window_transformer'}}}
    after = deepcopy(before)
    after['model']['encoder'].update(boundary_mode='truncate', boundary_noise_seed=42)
    validate_resume_config(before, after)
    validate_resume_config(after, before)
    after['model']['encoder']['boundary_mode'] = 'noise_pad'
    with pytest.raises(ValueError, match='boundary_mode'):
        validate_resume_config(before, after)



@pytest.mark.parametrize('overrides', [dict(type='cnn'), dict(type='patch_mlp', hidden_dim=8),
    dict(attention_pattern='right_local'), dict(boundary_noise_seed=-1),
    dict(boundary_noise_seed=True), dict(boundary_noise_seed=1.5), dict(boundary_noise_seed=2**63)])
def test_incompatible_configs_are_rejected(overrides):
    with pytest.raises(ValueError):
        build_eeg_encoder(encoder_config(**overrides), time_point_count=11)



def test_training_and_validation_use_noise_padding_without_extending_supervision():
    model = build_target(model_config(), time_point_count=11)
    activate(model.encoder)
    data = batch()
    arguments = dict(model=model, loader=[data], task='choice', attention_indices=torch.arange(11),
        loss_weights=dict(time=.5, pooled=1., attention_mass=.1), top_k=None, top_ratio=.8,
        label_smoothing=0., device=torch.device('cpu'), amp=False, progress_bar_enabled=False,
        loss_type='bce_one_vs_rest')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    train = run_target_epoch(**arguments, optimizer=optimizer, scaler=torch.amp.GradScaler('cpu', enabled=False))
    assert torch.isfinite(torch.tensor(train['loss']))
    rng = torch.get_rng_state()
    first = run_target_epoch(**arguments)
    second = run_target_epoch(**arguments)
    assert first == second and torch.equal(rng, torch.get_rng_state())



def test_freezing_encoder_does_not_freeze_training_noise_sampling():
    model = build_target(model_config(), time_point_count=11)
    model.configure_frozen_components(encoder=True)
    model.train()
    assert model.training and not model.encoder.training
    data = batch()
    first = boundary_noise_forward_kwargs(model, data)['boundary_noise_eeg']
    second = boundary_noise_forward_kwargs(model, data)['boundary_noise_eeg']
    assert not torch.equal(first, second)
    assert model(data['eeg'], 'choice')['time_logits'].shape == (4, 11, 3)



@pytest.mark.parametrize('seed', [0, 42, 2**63 - 1])
def test_full_supported_seed_range_round_trips(seed):
    encoder = build_eeg_encoder(encoder_config(boundary_noise_seed=seed), time_point_count=11).eval()
    assert build_eeg_encoder(encoder.resolved_config()).boundary_noise_seed == seed
    assert torch.isfinite(encoder.sample_boundary_noise(torch.zeros(1, 3, 11), trial_ids=torch.tensor([[1, 0, 0]]))).all()



@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bfloat16, torch.float16])
def test_noise_padding_supports_model_dtypes_and_autocast(dtype, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    if device == 'cpu' and dtype == torch.float16: pytest.skip('CPU AMP uses bfloat16')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('GPU has no bfloat16 support')
    model = build_target(model_config(), time_point_count=11).to(device).train()
    activate(model.encoder)
    data = {key: value.to(device) for key, value in batch().items()}
    if dtype == torch.float64:
        model.double()
        data['eeg'] = data['eeg'].double()
    with torch.autocast(device, dtype=dtype if dtype in (torch.bfloat16, torch.float16) else torch.bfloat16,
                        enabled=dtype in (torch.bfloat16, torch.float16)):
        outputs = model(data['eeg'], 'choice', **boundary_noise_forward_kwargs(model, data))
        loss = outputs['time_logits'].float().square().mean() + outputs['pooled_logits'].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())



@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_noise_padding_and_augmentation_resume_exactly(tmp_path, monkeypatch, device):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    config = dict(model=model_config(attention_dropout=.1), train=dict(
        iterations=4, validation_interval_iterations=1, log_interval_iterations=1,
        learning_rate=1e-4, lr_schedule='constant', weight_decay=.1, amp=False, progress_bar=False,
        time_loss_type='bce_one_vs_rest', pooled_loss_type='focal_bce', pooled_focal_gamma=1.,
        time_loss_weight=.5, pooled_loss_weight=1., attention_mask_penalty_weight=.7,
        attention_topk_ratio=.8, attention_mass_penalty_weight=.1,
        checkpoint_metric='val_pooled_accuracy', checkpoint_mode='max',
        temporal_crop_noise=dict(enabled=True, geometry='random_blocks', sample_interval_s=.01,
            probability=.8, min_blocks=1, max_blocks=1, min_block_duration_s=.02,
            min_total_duration_s=.02, max_total_duration_s=.08, supervision_mode='all_positions'),
        near_full_noise=dict(enabled=True, probability=.1, coverage_ratio=.95, time_loss_scale=.25),
        augmentation_pipeline=dict(enabled=True, roll_max_shift_s=.05)))
    torch.manual_seed(37)
    batches = [{key: value.to(device) for key, value in batch().items()} for _ in range(3)]
    def new_model(): return build_target(deepcopy(config['model']), time_point_count=11).to(device)
    initial = deepcopy(new_model().state_dict())
    def fit(model, path, resume=None):
        return training.fit_target_decoder(model, batches, batches, 'choice', torch.arange(11, device=device),
            config, path, torch.device(device), resume_from=resume)
    complete, interrupted = tmp_path/'complete', tmp_path/'interrupted'
    complete.mkdir(); interrupted.mkdir()
    for path in [complete, interrupted]:
        model = new_model()
        model.load_state_dict(initial)
        torch.manual_seed(41)
        if path == complete:
            fit(model, path)
        else:
            append = training.append_metrics_csv
            def stop(metrics_path, row):
                if metrics_path.name == 'train_metrics.csv' and int(row['iteration']) == 3:
                    raise InterruptedError('simulated stop')
                append(metrics_path, row)
            with monkeypatch.context() as patched:
                patched.setattr(training, 'append_metrics_csv', stop)
                with pytest.raises(InterruptedError): fit(model, path)
    checkpoint = interrupted/'last.ckpt'
    assert load_checkpoint(checkpoint)['iteration'] == 2
    fit(new_model(), interrupted, checkpoint)
    expected, actual = load_checkpoint(complete/'last.ckpt'), load_checkpoint(checkpoint)
    for key, value in expected['model_state'].items():
        torch.testing.assert_close(actual['model_state'][key], value, rtol=0, atol=0)
    assert torch.equal(expected['rng_states']['torch_cpu'], actual['rng_states']['torch_cpu'])
    if device == 'cuda':
        for before, after in zip(expected['rng_states']['torch_cuda'], actual['rng_states']['torch_cuda']):
            assert torch.equal(before, after)
    for name in ['metrics.csv', 'train_metrics.csv']:
        assert (complete/name).read_bytes() == (interrupted/name).read_bytes()

