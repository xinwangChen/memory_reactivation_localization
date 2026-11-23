from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from pb2023_eeg.models.encoder import EEGEncoder
from pb2023_eeg.models.target_decoder import GatedTimeAttentionPooling, TargetAttentionDecoder
from pb2023_eeg.training.common import validate_resume_config
from pb2023_eeg.workflows.reexport import resolve_model_config_for_checkpoint
from pb2023_eeg.workflows.train import build_model


def _model(**kwargs):
    encoder = EEGEncoder(in_channels=3, latent_dim=6, num_blocks=1,
                         kernel_size=3, dropout=0., expansion_ratio=1)
    model = TargetAttentionDecoder(encoder, attention_mode='gated_scorer',
        classifier_num_layers=1, **kwargs).eval()
    with torch.no_grad():
        model.attention.score[-1].weight.copy_(torch.linspace(-.8, 1.1, 6)[None])
        model.attention.gate_proj.weight.copy_(torch.linspace(-.5, .7, 6)[None])
    return model


def _forward(model, latent, indices=None):
    with patch.object(model.encoder, 'forward_features',
            return_value={'latent': latent, 'block_latents': [latent]}):
        return model(torch.empty(latent.shape[0], 3, latent.shape[1], device=latent.device),
                     'choice', attention_indices=indices)


class AttentionGateCenteringTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(221)

    def test_independent_inputs_use_full_trial_mean_and_raw_pool_values(self):
        latent = torch.randn(2, 9, 6) + torch.arange(6)[None, None]
        indices = torch.tensor([1, 3, 6])
        raw = latent.index_select(1, indices)
        centered = raw - latent.mean(1, keepdim=True)
        for score_centering in (False, True):
            for gate_centering in (False, True):
                with self.subTest(score=score_centering, gate=gate_centering):
                    model = _model(attention_score_centering=score_centering,
                        attention_gate_centering=gate_centering, reconstruction_enabled=True,
                        time_no_class_enabled=True)
                    seen = {}

                    def capture_score(_module, args):
                        seen['score'] = args[0].detach().clone()

                    def capture_gate(_module, args):
                        seen['gate'] = args[0].detach().clone()

                    handles = [model.attention.score.register_forward_pre_hook(capture_score),
                               model.attention.gate_proj.register_forward_pre_hook(capture_gate)]
                    try:
                        out = _forward(model, latent, indices)
                    finally:
                        for handle in handles:
                            handle.remove()
                    torch.testing.assert_close(seen['score'], centered if score_centering else raw)
                    torch.testing.assert_close(seen['gate'], centered if gate_centering else raw)
                    expected_gate = torch.sigmoid(model.attention.gate_proj(seen['gate']).squeeze(-1))
                    torch.testing.assert_close(out['attention_gate'], expected_gate)
                    weights = out['attention_weights'] * expected_gate
                    torch.testing.assert_close(out['effective_attention_weights'], weights)
                    torch.testing.assert_close(out['pooled_time_latent'], (weights[..., None] * raw).sum(1))
                    torch.testing.assert_close(out['time_logits'], model.class_heads['choice'](latent))
                    torch.testing.assert_close(out['reconstructed_eeg'], model.reconstruction_head(latent).transpose(1, 2))
                    model.attention_gate_centering = not gate_centering
                    other = _forward(model, latent, indices)
                    for key in ('attention_scores', 'time_logits', 'reconstructed_eeg'):
                        torch.testing.assert_close(out[key], other[key], rtol=0, atol=0)

    def test_gate_is_invariant_to_trial_feature_offsets_and_preserves_bias(self):
        model = _model(attention_score_centering=True, attention_gate_centering=True)
        latent = torch.randn(2, 9, 6)
        offset = torch.randn(2, 1, 6) * 3
        before, after = _forward(model, latent), _forward(model, latent + offset)
        torch.testing.assert_close(before['attention_gate'], after['attention_gate'])
        torch.testing.assert_close(before['attention_weights'], after['attention_weights'])
        torch.testing.assert_close(after['pooled_time_latent'], before['pooled_time_latent'] +
            before['effective_attention_weights'].sum(1, keepdim=True) * offset.squeeze(1))
        expected_logits = model.attention.gate_proj(latent - latent.mean(1, keepdim=True)).squeeze(-1)
        torch.testing.assert_close(expected_logits.mean(1), model.attention.gate_proj.bias.expand(2))
        for trial in range(2):
            alone = _forward(model, latent[trial:trial + 1])
            torch.testing.assert_close(alone['attention_gate'], before['attention_gate'][trial:trial + 1])

    def test_gate_background_gradients_reach_outside_pool_support(self):
        model = _model(attention_gate_centering=True, pool_valid_context_only=True).double()
        latent = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
        out = _forward(model, latent, torch.tensor([0, 2, 4, 8]))
        self.assertEqual(int(torch.count_nonzero(out['effective_attention_weights'][:, [0, -1]])), 0)
        out['attention_gate'][:, 1].sum().backward()
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(float(latent.grad[:, 7].abs().sum()), 0.)
        x = torch.randn(1, 5, 6, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: _forward(model, value, torch.tensor([1, 3]))['attention_gate'], (x,)))

    def test_low_precision_both_centered_branches_have_finite_gradients(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = _model(attention_score_centering=True, attention_gate_centering=True).to(dtype)
                latent = torch.randn(2, 9, 6).to(dtype).requires_grad_()
                out = _forward(model, latent)
                self.assertEqual(out['attention_gate'].dtype, dtype)
                self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
                out['pooled_logits'].float().square().mean().backward()
                self.assertTrue(torch.isfinite(latent.grad).all())

    def test_false_default_preserves_weights_and_outputs(self):
        legacy = _model(attention_score_centering=True)
        disabled = _model(attention_score_centering=True, attention_gate_centering=False)
        enabled = _model(attention_score_centering=True, attention_gate_centering=True)
        for other in (disabled, enabled):
            self.assertEqual(set(legacy.state_dict()), set(other.state_dict()))
            other.load_state_dict(legacy.state_dict(), strict=True)
        x = torch.randn(2, 3, 9, requires_grad=True)
        y = x.detach().clone().requires_grad_()
        a, b = legacy(x, 'choice'), disabled(y, 'choice')
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        a['pooled_logits'].sum().backward()
        b['pooled_logits'].sum().backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)

    def test_builder_and_reexport_preserve_gate_setting(self):
        config = {'encoder': {'in_channels': 3, 'latent_dim': 6, 'num_blocks': 1},
                  'attention_mode': 'gated_scorer', 'attention_score_centering': True,
                  'attention_gate_centering': True}
        model = build_model(config).eval()
        resolved = resolve_model_config_for_checkpoint(config, model.state_dict())
        restored = build_model(resolved).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(restored.attention_gate_centering)
        self.assertTrue(restored.attention_score_centering)
        eeg = torch.randn(2, 3, 9)
        a, b = model(eeg, 'choice'), restored(eeg, 'choice')
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        for value in (None, 0, 1, 'true', 'false'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'attention_gate_centering.*boolean'):
                build_model({**config, 'attention_gate_centering': value})
        for mode in ('scorer', 'query', 'gated_query'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'requires gated_scorer'):
                build_model({**config, 'attention_score_centering': False, 'attention_mode': mode})

    def test_resume_gate_flag_is_part_of_training_identity(self):
        old = {'model': {'attention_mode': 'gated_scorer', 'attention_score_centering': True}}
        disabled = deepcopy(old)
        disabled['model']['attention_gate_centering'] = False
        enabled = deepcopy(old)
        enabled['model']['attention_gate_centering'] = True
        validate_resume_config(old, disabled)
        validate_resume_config(disabled, old)
        validate_resume_config(enabled, deepcopy(enabled))
        for a, b in ((old, enabled), (enabled, old), (disabled, enabled)):
            with self.assertRaisesRegex(ValueError, 'config.model.attention_gate_centering'):
                validate_resume_config(a, b)

    def test_separate_gate_input_rejects_shape_dtype_and_device_mismatches(self):
        pool = GatedTimeAttentionPooling(6)
        latent = torch.randn(2, 9, 6)
        for value, message in ((latent[:, :2], 'same shape'), (latent.double(), 'same device and dtype'),
                               (torch.empty(2, 9, 6, device='meta'), 'same device and dtype')):
            with self.assertRaisesRegex(ValueError, 'gate_latent.*' + message):
                pool(latent, gate_latent=value)


if __name__ == '__main__':
    unittest.main()
