from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

import torch

from pb2023_eeg.models.encoder import EEGEncoder
from pb2023_eeg.models.target_decoder import (
    GatedTimeAttentionPooling,
    TargetAttentionDecoder,
    TimeAttentionPooling,
)
from pb2023_eeg.training.common import validate_resume_config
from pb2023_eeg.workflows.reexport import resolve_model_config_for_checkpoint
from pb2023_eeg.workflows.train import build_model


def _encoder() -> EEGEncoder:
    return EEGEncoder(
        in_channels=3, latent_dim=6, num_blocks=1, kernel_size=3,
        dropout=0.0, expansion_ratio=1,
    )


def _model(mode: str = "gated_scorer", **kwargs) -> TargetAttentionDecoder:
    model = TargetAttentionDecoder(
        encoder=_encoder(), attention_mode=mode,
        classifier_num_layers=1, **kwargs,
    ).eval()
    # Initial zero scorer/gate weights would hide accidental changes to either path.
    with torch.no_grad():
        model.attention.score[-1].weight.copy_(torch.linspace(-0.8, 1.1, 6)[None])
        if mode == "gated_scorer":
            model.attention.gate_proj.weight.copy_(torch.linspace(-0.5, 0.7, 6)[None])
    return model


def _forward_latent(model, latent, indices=None):
    with patch.object(
        model.encoder, "forward_features",
        return_value={"latent": latent, "block_latents": [latent]},
    ):
        return model(
            torch.empty(latent.shape[0], 3, latent.shape[1], device=latent.device),
            task="choice", attention_indices=indices,
        )


class AttentionScoreCenteringTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(127)

    def test_centering_changes_scores_but_preserves_raw_feature_paths(self) -> None:
        latent = torch.randn(2, 9, 6) + torch.arange(6)[None, None, :]
        indices = torch.tensor([1, 3, 6])
        for mode in ("scorer", "gated_scorer"):
            with self.subTest(mode=mode):
                model = _model(mode, reconstruction_enabled=True)
                before = _forward_latent(model, latent, indices)
                seen = []

                def capture(_module, args):
                    seen.append(args[0].detach().clone())

                handle = model.attention.score.register_forward_pre_hook(capture)
                model.attention_score_centering = True
                try:
                    after = _forward_latent(model, latent, indices)
                finally:
                    handle.remove()

                selected = latent.index_select(1, indices)
                expected_score_input = (
                    latent - latent.mean(1, keepdim=True)
                ).index_select(1, indices)
                torch.testing.assert_close(seen[0], expected_score_input)
                self.assertFalse(torch.allclose(before["attention_scores"], after["attention_scores"]))
                for key in ("time_logits", "reconstructed_eeg"):
                    torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
                weights = after["attention_weights"]
                if mode == "gated_scorer":
                    torch.testing.assert_close(before["attention_gate"], after["attention_gate"], rtol=0, atol=0)
                    weights = after["effective_attention_weights"]
                    torch.testing.assert_close(weights, after["attention_weights"] * after["attention_gate"])
                torch.testing.assert_close(
                    after["pooled_time_latent"], (weights[..., None] * selected).sum(1),
                )
                # Pooling centered values instead would subtract this nonzero vector.
                removed_value = weights.sum(1, keepdim=True) * latent.mean(1)
                self.assertGreater(float(removed_value.abs().max()), 0.1)

    def test_full_trial_reference_is_independent_of_pool_window_and_mask(self) -> None:
        model = _model(attention_score_centering=True, pool_valid_context_only=True)
        latent = torch.randn(2, 9, 6)
        indices = torch.tensor([0, 2, 4, 8])
        changed = latent.clone()
        changed[:, 7] += torch.arange(6) * 8.0  # Outside the attention window.
        before = _forward_latent(model, latent, indices)
        after = _forward_latent(model, changed, indices)
        self.assertFalse(torch.allclose(before["attention_scores"], after["attention_scores"]))
        torch.testing.assert_close(before["attention_gate"], after["attention_gate"], rtol=0, atol=0)
        self.assertEqual(int(torch.count_nonzero(after["effective_attention_weights"][:, [0, -1]])), 0)
        torch.testing.assert_close(
            after["time_selection_attention_weights"], after["effective_attention_weights"],
        )
        for trial in range(2):
            alone = _forward_latent(model, changed[trial:trial + 1], indices)
            for key in ("attention_scores", "attention_gate", "pooled_logits"):
                torch.testing.assert_close(alone[key], after[key][trial:trial + 1])

    def test_background_reference_retains_gradients_outside_pool_window(self) -> None:
        model = _model(attention_score_centering=True).double()
        latent = torch.randn(2, 9, 6, dtype=torch.float64, requires_grad=True)
        outputs = _forward_latent(model, latent, torch.tensor([2, 4, 6]))
        outputs["attention_scores"][:, 0].sum().backward()
        self.assertIsNotNone(latent.grad)
        self.assertTrue(torch.isfinite(latent.grad).all())
        self.assertGreater(float(latent.grad[:, 8].abs().sum()), 0.0)
        self.assertTrue(any(p.grad is not None for p in model.attention.score.parameters()))

    def test_low_precision_centering_preserves_dtype_and_finite_gradients(self) -> None:
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = _model(attention_score_centering=True).to(dtype=dtype)
                latent = torch.randn(2, 9, 6).to(dtype).requires_grad_()
                outputs = _forward_latent(model, latent)
                self.assertEqual(outputs["attention_scores"].dtype, dtype)
                self.assertTrue(all(torch.isfinite(value).all() for value in outputs.values()))
                outputs["pooled_logits"].float().square().sum().backward()
                self.assertTrue(torch.isfinite(latent.grad).all())

    def test_disabled_flag_preserves_state_and_forward_backward(self) -> None:
        for mode in ("scorer", "gated_scorer"):
            with self.subTest(mode=mode):
                default = _model(mode)
                disabled = _model(mode, attention_score_centering=False)
                enabled = _model(mode, attention_score_centering=True)
                for other in (disabled, enabled):
                    self.assertEqual(set(default.state_dict()), set(other.state_dict()))
                    other.load_state_dict(default.state_dict(), strict=True)
                eeg = torch.randn(2, 3, 9, requires_grad=True)
                copied = eeg.detach().clone().requires_grad_()
                left, right = default(eeg, "choice"), disabled(copied, "choice")
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                left["pooled_logits"].sum().backward()
                right["pooled_logits"].sum().backward()
                torch.testing.assert_close(eeg.grad, copied.grad, rtol=0, atol=0)

    def test_build_and_checkpoint_reexport_retain_flag(self) -> None:
        config = {
            "encoder": {"in_channels": 3, "latent_dim": 6, "num_blocks": 1,
                        "kernel_size": 3, "dropout": 0.0, "expansion_ratio": 1},
            "attention_mode": "gated_scorer", "attention_score_centering": True,
            "classifier_num_layers": 1,
        }
        source = build_model(config).eval()
        with torch.no_grad():
            source.attention.score[-1].weight.fill_(0.3)
        resolved = resolve_model_config_for_checkpoint(config, source.state_dict())
        restored = build_model(resolved).eval()
        restored.load_state_dict(source.state_dict(), strict=True)
        self.assertTrue(restored.attention_score_centering)
        eeg = torch.randn(2, 3, 9)
        left, right = source(eeg, "choice"), restored(eeg, "choice")
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_invalid_flag_and_unsupported_modes_fail_explicitly(self) -> None:
        encoder_config = {"in_channels": 3, "latent_dim": 6, "num_blocks": 1}
        for value in (None, "false", "true", 0, 1):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "attention_score_centering.*boolean"):
                    build_model({"encoder": encoder_config, "attention_score_centering": value})
        for mode in ("query", "gated_query"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(ValueError, "requires scorer or gated_scorer"):
                    TargetAttentionDecoder(_encoder(), attention_mode=mode, attention_score_centering=True)
                legacy = TargetAttentionDecoder(_encoder(), attention_mode=mode, n_subjects=3).eval()
                self.assertFalse(legacy.attention_score_centering)
                outputs = legacy(
                    torch.randn(1, 3, 9), "choice", subject_id=torch.tensor([0]),
                )
                self.assertTrue(torch.isfinite(outputs["pooled_logits"]).all())

    def test_resume_allows_legacy_false_but_rejects_switching_behavior(self) -> None:
        legacy = {"model": {"attention_mode": "gated_scorer"}}
        disabled = deepcopy(legacy)
        disabled["model"]["attention_score_centering"] = False
        enabled = deepcopy(legacy)
        enabled["model"]["attention_score_centering"] = True
        validate_resume_config(legacy, disabled)
        validate_resume_config(disabled, legacy)
        validate_resume_config(enabled, deepcopy(enabled))
        for left, right in ((legacy, enabled), (enabled, legacy), (disabled, enabled)):
            with self.subTest(saved=left, current=right):
                with self.assertRaisesRegex(ValueError, "config.model.attention_score_centering"):
                    validate_resume_config(left, right)

    def test_separate_score_input_rejects_mismatched_shapes_and_dtypes(self) -> None:
        latent = torch.randn(2, 9, 6)
        for cls in (TimeAttentionPooling, GatedTimeAttentionPooling):
            pool = cls(6)
            with self.subTest(cls=cls.__name__):
                with self.assertRaisesRegex(ValueError, "same shape"):
                    pool(latent, score_latent=latent[:, :3])
                with self.assertRaisesRegex(ValueError, "same device and dtype"):
                    pool(latent, score_latent=latent.double())


if __name__ == "__main__":
    unittest.main()
