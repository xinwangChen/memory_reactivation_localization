from __future__ import annotations

import unittest

import torch

from memory_reactivation.training.target import (
    attention_mask_penalty,
    attention_smoothness_penalty,
)


class AttentionMaskPenaltyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.attention_indices = torch.tensor([0, 2, 4])
        self.covered_time_mask = torch.tensor(
            [
                [True, False, False, False, True],
                [False, False, True, False, False],
            ]
        )
        self.weights = torch.tensor(
            [
                [0.2, 0.3, 0.5],
                [0.2, 0.4, 1.4],
            ],
            dtype=torch.float32,
        )

    def test_default_preserves_absolute_effective_mass(self) -> None:
        loss = attention_mask_penalty(
            self.weights,
            self.attention_indices,
            self.covered_time_mask,
        )
        self.assertAlmostEqual(float(loss), 0.55, places=6)

    def test_relative_effective_mass_matches_trialwise_ratio(self) -> None:
        loss = attention_mask_penalty(
            self.weights,
            self.attention_indices,
            self.covered_time_mask,
            mode="relative_effective_mass",
        )
        self.assertAlmostEqual(float(loss), 0.45, places=6)

    def test_relative_mode_is_invariant_to_global_scale(self) -> None:
        relative = attention_mask_penalty(
            self.weights,
            self.attention_indices,
            self.covered_time_mask,
            mode="relative_effective_mass",
        )
        scaled_relative = attention_mask_penalty(
            self.weights * 0.017,
            self.attention_indices,
            self.covered_time_mask,
            mode="relative_effective_mass",
        )
        absolute = attention_mask_penalty(
            self.weights,
            self.attention_indices,
            self.covered_time_mask,
        )
        scaled_absolute = attention_mask_penalty(
            self.weights * 0.017,
            self.attention_indices,
            self.covered_time_mask,
        )
        self.assertTrue(torch.allclose(relative, scaled_relative, atol=1e-7, rtol=1e-6))
        self.assertFalse(torch.allclose(absolute, scaled_absolute))

    def test_relative_mode_penalizes_mass_redistribution_not_global_scale(self) -> None:
        base = torch.tensor([[1.0, 1.0]], requires_grad=True)
        scale = torch.tensor(0.3, requires_grad=True)
        covered = torch.tensor([[True, False]])
        loss = attention_mask_penalty(
            base * scale,
            torch.tensor([0, 1]),
            covered,
            mode="relative_effective_mass",
        )
        loss.backward()
        self.assertAlmostEqual(float(scale.grad), 0.0, places=6)
        self.assertGreater(float(base.grad[0, 0]), 0.0)
        self.assertLess(float(base.grad[0, 1]), 0.0)

    def test_empty_mask_returns_zero_and_invalid_mode_fails(self) -> None:
        empty = torch.zeros(2, 5, dtype=torch.bool)
        for covered in (None, empty):
            loss = attention_mask_penalty(
                self.weights,
                self.attention_indices,
                covered,
                mode="relative_effective_mass",
            )
            self.assertEqual(float(loss), 0.0)
            self.assertEqual(loss.dtype, self.weights.dtype)
        differentiable_weights = self.weights.clone().requires_grad_()
        zero_loss = attention_mask_penalty(
            differentiable_weights,
            self.attention_indices,
            empty,
        )
        zero_loss.backward()
        self.assertTrue(torch.equal(differentiable_weights.grad, torch.zeros_like(differentiable_weights)))
        with self.assertRaisesRegex(ValueError, "attention_mask_penalty_mode"):
            attention_mask_penalty(
                self.weights,
                self.attention_indices,
                empty,
                mode="unknown",
            )

    def test_empty_mask_still_validates_shape_and_indices(self) -> None:
        # 早返回不能掩盖 batch/time 轴错误，否则配置问题会静默变成零正则项。
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            attention_mask_penalty(
                self.weights,
                self.attention_indices,
                torch.zeros(1, 5, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, r"shape \[batch, time\]"):
            attention_mask_penalty(
                self.weights,
                self.attention_indices,
                torch.zeros(2, 5, 1, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "within covered_time_mask"):
            attention_mask_penalty(
                self.weights,
                self.attention_indices,
                torch.zeros(2, 4, dtype=torch.bool),
            )
        with self.assertRaisesRegex(ValueError, "int32 or int64"):
            attention_mask_penalty(
                self.weights,
                self.attention_indices.to(dtype=torch.int16),
                torch.zeros(2, 5, dtype=torch.bool),
            )

    def test_empty_mask_avoids_fp16_overflow_in_excluded_weights(self) -> None:
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for mode in ['absolute_effective_mass', 'relative_effective_mass']:
                for missing in [False, True]:
                    with self.subTest(device=device, mode=mode, missing=missing):
                        weights = torch.full((256, 275), 1.9, dtype=torch.float16,
                                             device=device, requires_grad=True)
                        self.assertTrue(bool(torch.isfinite(weights).all()))
                        self.assertFalse(bool(torch.isfinite(weights.sum())))
                        mask = None if missing else torch.zeros_like(weights, dtype=torch.bool)
                        loss = attention_mask_penalty(weights, torch.arange(275, device=device), mask, mode=mode)
                        self.assertEqual(loss.item(), 0.)
                        self.assertEqual(loss.dtype, weights.dtype)
                        loss.backward()
                        self.assertTrue(torch.equal(weights.grad, torch.zeros_like(weights)))


class AttentionSmoothnessPenaltyTest(unittest.TestCase):
    def test_second_difference_preserves_linear_curve_and_penalizes_kink(self) -> None:
        linear = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        kink = torch.tensor([[0.0, 1.0, 0.0]])
        self.assertAlmostEqual(
            float(attention_smoothness_penalty(linear)), 0.0, places=7
        )
        self.assertAlmostEqual(
            # Mean normalization turns [0, 1, 0] into [0, 3, 0].
            float(attention_smoothness_penalty(kink)), 36.0, places=7
        )

    def test_first_difference_mode_is_available(self) -> None:
        curve = torch.tensor([[0.0, 1.0, 0.0]])
        self.assertAlmostEqual(
            float(
                attention_smoothness_penalty(
                    curve,
                    mode="first_difference",
                )
            ),
            9.0,
            places=7,
        )

    def test_valid_mask_excludes_invalid_boundary_pairs(self) -> None:
        curve = torch.tensor([[99.0, 0.0, 1.0, 3.0]])
        valid = torch.tensor([False, True, True, True])
        # Valid values [0, 1, 3] have mean 4/3; their second difference is 3/4.
        self.assertAlmostEqual(
            float(attention_smoothness_penalty(curve, valid)), 0.5625, places=7
        )

    def test_zero_when_no_valid_difference_and_keeps_gradient_path(self) -> None:
        curve = torch.tensor([[1.0, 2.0]], requires_grad=True)
        valid = torch.tensor([True, False])
        loss = attention_smoothness_penalty(curve, valid)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(curve.grad, torch.zeros_like(curve)))

    def test_trialwise_scale_invariance_and_invalid_values_do_not_change_loss(self) -> None:
        curves = torch.tensor([[99.0, 1.0, 4.0, 1.0], [99.0, 2.0, 1.0, 3.0]])
        valid = torch.tensor([False, True, True, True])
        scaled = curves * torch.tensor([[0.1], [3.0]])
        scaled[:, 0] = float("nan")
        for mode in ("first_difference", "second_difference"):
            with self.subTest(mode=mode):
                torch.testing.assert_close(
                    attention_smoothness_penalty(curves, valid, mode=mode),
                    attention_smoothness_penalty(scaled, valid, mode=mode),
                )

    def test_gradient_updates_shape_without_penalizing_overall_gate_scale(self) -> None:
        curve = torch.tensor([[1.0, 4.0, 1.0]], requires_grad=True)
        scale = torch.tensor(0.5, requires_grad=True)
        loss = attention_smoothness_penalty(curve * scale)
        loss.backward()
        self.assertTrue(torch.isfinite(curve.grad).all())
        self.assertGreater(float(curve.grad.abs().sum()), 0.0)
        self.assertAlmostEqual(float(scale.grad), 0.0, places=6)

    def test_invalid_mode_and_mask_shape_fail(self) -> None:
        curve = torch.ones(2, 4)
        with self.assertRaisesRegex(ValueError, "attention_smoothness_mode"):
            attention_smoothness_penalty(curve, mode="unknown")
        with self.assertRaisesRegex(ValueError, "aligned with attention"):
            attention_smoothness_penalty(curve, torch.ones(3, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()
