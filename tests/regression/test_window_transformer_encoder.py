from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from memory_reactivation.models.encoder import (
    EEGEncoder,
    RightLocalTransformerBlock,
    TemporalPatchMLPBlock,
    build_eeg_encoder,
)
from memory_reactivation.models.target_decoder import TargetAttentionDecoder


def _window_encoder(
    *,
    window_size: int,
    time_blocks: int = 2,
    latent_dim: int = 8,
    attention_pattern: str = "right_local",
    position_encoding_mode: str = "sinusoidal_qk",
) -> EEGEncoder:
    return build_eeg_encoder(
        {
            "type": "window_transformer",
            "in_channels": 3,
            "latent_dim": latent_dim,
            "num_blocks": time_blocks,
            "window_size": window_size,
            "num_heads": 2,
            "expansion_ratio": 2,
            "dropout": 0.0,
            "attention_pattern": attention_pattern,
            "position_encoding_mode": position_encoding_mode,
        },
        time_point_count=7,
    )


def _activate_transformer_residual_paths(encoder: EEGEncoder) -> None:
    """Make locality tests exercise attention instead of identity initialization."""
    with torch.no_grad():
        for block in encoder.temporal_blocks:
            if not isinstance(block, RightLocalTransformerBlock):
                raise TypeError("Expected a RightLocalTransformerBlock")
            nn.init.normal_(block.attention.out_proj.weight, std=0.15)
            if block.attention.out_proj.bias is not None:
                nn.init.normal_(block.attention.out_proj.bias, std=0.05)
            output_linear = block.ffn[-2]
            if not isinstance(output_linear, nn.Linear):
                raise TypeError("Expected the Transformer FFN output projection")
            nn.init.normal_(output_linear.weight, std=0.15)
            nn.init.normal_(output_linear.bias, std=0.05)


def _configure_uniform_value_attention(block: RightLocalTransformerBlock) -> None:
    """让 attention 输出成为允许 key 的归一化 value 均值，便于精确测局部性。"""
    with torch.no_grad():
        channels = block.channels
        block.attention.in_proj_weight.zero_()
        if block.attention.in_proj_bias is not None:
            block.attention.in_proj_bias.zero_()
        block.attention.in_proj_weight[2 * channels :].copy_(
            torch.eye(channels)
        )
        block.attention.out_proj.weight.copy_(torch.eye(channels))
        if block.attention.out_proj.bias is not None:
            block.attention.out_proj.bias.zero_()


def _activate_single_temporal_block(encoder: EEGEncoder) -> None:
    """激活零初始化的残差输出，以便结构 RF 测试真正经过时间路径。"""
    block = encoder.temporal_blocks[0]
    with torch.no_grad():
        if isinstance(block, RightLocalTransformerBlock):
            nn.init.normal_(block.attention.out_proj.weight, std=0.2)
            if block.attention.out_proj.bias is not None:
                block.attention.out_proj.bias.zero_()
        elif isinstance(block, TemporalPatchMLPBlock):
            output = block.mlp[-2]
            assert isinstance(output, nn.Linear)
            nn.init.normal_(output.weight, std=0.2)
            output.bias.zero_()
        else:
            pointwise = getattr(block, "pointwise")
            output = pointwise[-2]
            assert isinstance(output, nn.Conv1d)
            nn.init.normal_(output.weight, std=0.2)
            output.bias.zero_()


class WindowTransformerEncoderTest(unittest.TestCase):
    def test_small_windows_preserve_every_time_point_without_padding(self) -> None:
        torch.manual_seed(101)
        eeg = torch.randn(3, 3, 7)

        for window_size in (2, 3, 4):
            with self.subTest(window_size=window_size):
                encoder = _window_encoder(window_size=window_size)
                features = encoder.forward_features(eeg)
                latent = features["latent"]
                block_latents = features["block_latents"]

                self.assertIsInstance(latent, torch.Tensor)
                assert isinstance(latent, torch.Tensor)
                self.assertEqual(latent.shape, (3, 7, 8))
                self.assertTrue(torch.isfinite(latent).all())
                self.assertEqual(len(block_latents), 2)
                for block_latent in block_latents:
                    self.assertEqual(block_latent.shape, (3, 7, 8))
                    self.assertTrue(torch.isfinite(block_latent).all())

    def test_right_local_mask_buffer_has_exact_forbidden_pattern(self) -> None:
        block = RightLocalTransformerBlock(
            channels=4,
            window_size=3,
            time_point_count=5,
            num_heads=1,
            dropout=0.0,
        )
        mask = block.attention_mask
        expected_allowed = torch.tensor(
            [
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
                [False, False, False, False, True],
            ]
        )

        self.assertEqual(mask.dtype, torch.bool)
        torch.testing.assert_close(mask, ~expected_allowed)
        self.assertNotIn("attention_mask", block.state_dict())

    def test_centered_local_mask_and_boundaries_are_exact(self) -> None:
        block = RightLocalTransformerBlock(
            channels=4,
            window_size=3,
            time_point_count=5,
            num_heads=1,
            dropout=0.0,
            attention_pattern="centered_local",
            position_encoding_mode="none",
        )
        expected_allowed = torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [False, True, True, True, False],
                [False, False, True, True, True],
                [False, False, False, True, True],
            ]
        )

        torch.testing.assert_close(block.attention_mask, ~expected_allowed)
        self.assertEqual(block.attention_pattern, "centered_local")
        self.assertEqual(block.position_encoding_mode, "none")
        self.assertEqual(int(torch.count_nonzero(block.position_encoding)), 0)
        with self.assertRaisesRegex(ValueError, "centered_local.*odd"):
            RightLocalTransformerBlock(
                channels=4,
                window_size=4,
                time_point_count=5,
                num_heads=1,
                dropout=0.0,
                attention_pattern="centered_local",
            )

    def test_centered_reflect_boundary_has_full_context_at_every_query(self) -> None:
        block = RightLocalTransformerBlock(
            channels=4,
            window_size=5,
            time_point_count=7,
            num_heads=1,
            dropout=0.0,
            attention_pattern="centered_local",
            position_encoding_mode="none",
            boundary_mode="reflect",
        )
        self.assertEqual(block.attention_mask.shape, (7, 11))
        torch.testing.assert_close(
            (~block.attention_mask).sum(dim=1),
            torch.full((7,), 5),
        )
        output = block(torch.randn(2, 4, 7))
        self.assertEqual(output.shape, (2, 4, 7))
        self.assertTrue(torch.isfinite(output).all())

        truncate = EEGEncoder(
            in_channels=3,
            latent_dim=4,
            num_blocks=1,
            encoder_type="window_transformer",
            window_size=5,
            time_point_count=7,
            num_heads=1,
            attention_pattern="centered_local",
            position_encoding_mode="none",
        )
        reflect = EEGEncoder(
            in_channels=3,
            latent_dim=4,
            num_blocks=1,
            encoder_type="window_transformer",
            window_size=5,
            time_point_count=7,
            num_heads=1,
            attention_pattern="centered_local",
            position_encoding_mode="none",
            boundary_mode="reflect",
        )
        self.assertEqual(reflect.resolved_config()["boundary_mode"], "reflect")
        with self.assertRaises(RuntimeError):
            reflect.load_state_dict(truncate.state_dict(), strict=True)

    def test_centered_boundary_reads_only_real_neighbours(self) -> None:
        torch.manual_seed(108)
        block = RightLocalTransformerBlock(
            channels=4,
            window_size=3,
            time_point_count=6,
            num_heads=1,
            dropout=0.0,
            attention_pattern="centered_local",
            position_encoding_mode="none",
        )
        _configure_uniform_value_attention(block)
        block.eval()
        baseline = torch.randn(1, 4, 6)
        perturbation = torch.tensor([2.0, -1.0, 0.5, 3.0])
        changed_neighbour = baseline.clone()
        changed_neighbour[:, :, 1] += perturbation
        changed_outside = baseline.clone()
        changed_outside[:, :, 2] += perturbation

        with torch.no_grad():
            output = block(
                torch.cat((baseline, changed_neighbour, changed_outside), dim=0)
            )

        self.assertGreater(
            float((output[0, :, 0] - output[1, :, 0]).abs().max()),
            1e-5,
        )
        torch.testing.assert_close(
            output[0, :, 0],
            output[2, :, 0],
            rtol=0.0,
            atol=1e-7,
        )

    def test_position_buffer_supports_odd_channels_and_follows_module_dtype(self) -> None:
        block = RightLocalTransformerBlock(
            channels=5,
            window_size=3,
            time_point_count=7,
            num_heads=1,
            dropout=0.0,
        )
        encoding = block.position_encoding

        self.assertEqual(encoding.shape, (1, 7, 5))
        self.assertEqual(encoding.dtype, torch.float32)
        self.assertTrue(torch.isfinite(encoding).all())
        self.assertGreater(
            float((encoding[:, 0] - encoding[:, 1]).abs().max()),
            0.0,
        )
        self.assertNotIn("position_encoding", block.state_dict())
        block.to(dtype=torch.float64)
        self.assertEqual(block.position_encoding.dtype, torch.float64)

    def test_none_position_mode_is_exact_zero_and_preserved_by_builder(self) -> None:
        encoder = _window_encoder(
            window_size=5,
            time_blocks=1,
            attention_pattern="centered_local",
            position_encoding_mode="none",
        )
        block = encoder.temporal_blocks[0]
        assert isinstance(block, RightLocalTransformerBlock)

        self.assertEqual(int(torch.count_nonzero(block.position_encoding)), 0)
        self.assertEqual(encoder.left_context_samples, 2)
        self.assertEqual(encoder.right_context_samples, 2)
        self.assertEqual(encoder.receptive_field_samples, 5)
        resolved = encoder.resolved_config()
        self.assertEqual(resolved["attention_pattern"], "centered_local")
        self.assertEqual(resolved["position_encoding_mode"], "none")
        rebuilt = build_eeg_encoder(resolved)
        self.assertEqual(rebuilt.resolved_config(), resolved)

    def test_single_layer_uses_sliding_right_window_across_old_boundary(self) -> None:
        torch.manual_seed(102)
        block = RightLocalTransformerBlock(
            channels=4,
            window_size=3,
            time_point_count=7,
            num_heads=1,
            dropout=0.0,
        )
        _configure_uniform_value_attention(block)
        block.eval()

        baseline = torch.randn(1, 4, 7)
        perturbation = torch.tensor([3.0, -2.0, 1.0, 0.5])
        changed_left = baseline.clone()
        changed_left[:, :, 1] += perturbation
        changed_allowed = baseline.clone()
        changed_allowed[:, :, 3] += perturbation
        changed_too_far_right = baseline.clone()
        changed_too_far_right[:, :, 5] += perturbation
        with torch.no_grad():
            output = block(
                torch.cat(
                    (
                        baseline,
                        changed_left,
                        changed_allowed,
                        changed_too_far_right,
                    ),
                    dim=0,
                )
            )

        # q=2 只允许读取 key=2,3,4。key=3 跨越旧固定分块的 2/3 边界，
        # 必须产生影响；左侧 key=1 和过远的 key=5 都必须完全不可见。
        torch.testing.assert_close(
            output[0, :, 2],
            output[1, :, 2],
            rtol=0.0,
            atol=1e-7,
        )
        torch.testing.assert_close(
            output[0, :, 2],
            output[3, :, 2],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertGreater(
            float((output[0, :, 2] - output[2, :, 2]).abs().max()),
            1e-5,
        )

    def test_tail_query_only_reads_itself(self) -> None:
        torch.manual_seed(103)
        block = RightLocalTransformerBlock(
            channels=4,
            window_size=4,
            time_point_count=6,
            num_heads=1,
            dropout=0.0,
        )
        _configure_uniform_value_attention(block)
        block.eval()
        baseline = torch.randn(1, 4, 6)
        perturbed = baseline.clone()
        perturbed[:, :, :5] += torch.tensor([2.0, -1.0, 0.5, 3.0])[:, None]

        with torch.no_grad():
            output = block(torch.cat((baseline, perturbed), dim=0))

        torch.testing.assert_close(
            output[0, :, -1],
            output[1, :, -1],
            rtol=0.0,
            atol=1e-7,
        )

    def test_trial_batch_chunking_matches_unchunked_attention_and_backward(self) -> None:
        torch.manual_seed(107)
        unchunked = _window_encoder(window_size=3, time_blocks=1)
        _activate_transformer_residual_paths(unchunked)
        chunked = copy.deepcopy(unchunked)
        unchunked_block = unchunked.temporal_blocks[0]
        chunked_block = chunked.temporal_blocks[0]
        assert isinstance(unchunked_block, RightLocalTransformerBlock)
        assert isinstance(chunked_block, RightLocalTransformerBlock)
        unchunked_block.attention_batch_limit = 10_000
        # 四个 trial 以 batch limit=2 分成两次 MHA；每次仍处理完整时间轴。
        chunked_block.attention_batch_limit = 2
        unchunked.eval()
        chunked.eval()

        eeg = torch.randn(4, 3, 7)
        with torch.no_grad():
            expected = unchunked(eeg)
            actual = chunked(eeg)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

        differentiable_eeg = eeg.clone().requires_grad_(True)
        chunked(differentiable_eeg).square().mean().backward()
        self.assertIsNotNone(differentiable_eeg.grad)
        assert differentiable_eeg.grad is not None
        self.assertTrue(torch.isfinite(differentiable_eeg.grad).all())
        self.assertGreater(float(differentiable_eeg.grad.abs().sum()), 0.0)

    def test_precomputed_buffers_are_reused_and_input_length_is_enforced(self) -> None:
        encoder = _window_encoder(window_size=3, time_blocks=1)
        block = encoder.temporal_blocks[0]
        assert isinstance(block, RightLocalTransformerBlock)
        mask_pointer = block.attention_mask.data_ptr()
        position_pointer = block.position_encoding.data_ptr()

        with torch.no_grad():
            encoder(torch.randn(2, 3, 7))

        self.assertEqual(block.attention_mask.data_ptr(), mask_pointer)
        self.assertEqual(block.position_encoding.data_ptr(), position_pointer)
        with self.assertRaisesRegex(ValueError, "time length.*7"):
            encoder(torch.randn(2, 3, 6))

    def test_patch_mlp_schema_shape_metadata_and_roundtrip(self) -> None:
        for window_size, hidden_dim in ((3, 16), (5, 12), (7, 8)):
            with self.subTest(window_size=window_size):
                config = {
                    "type": "patch_mlp",
                    "in_channels": 3,
                    "latent_dim": 8,
                    "num_blocks": 1,
                    "window_size": window_size,
                    "hidden_dim": hidden_dim,
                    "dropout": 0.0,
                }
                encoder = build_eeg_encoder(config, time_point_count=9)
                self.assertIsInstance(
                    encoder.temporal_blocks[0], TemporalPatchMLPBlock
                )
                self.assertEqual(encoder.left_context_samples, window_size // 2)
                self.assertEqual(encoder.right_context_samples, window_size // 2)
                self.assertEqual(encoder.receptive_field_samples, window_size)
                self.assertEqual(encoder(torch.randn(2, 3, 9)).shape, (2, 9, 8))
                resolved = encoder.resolved_config()
                self.assertEqual(resolved["hidden_dim"], hidden_dim)
                self.assertEqual(build_eeg_encoder(resolved).resolved_config(), resolved)

        with self.assertRaisesRegex(ValueError, "num_blocks=1"):
            build_eeg_encoder(
                {
                    "type": "patch_mlp",
                    "num_blocks": 2,
                    "window_size": 3,
                    "hidden_dim": 8,
                }
            )
        with self.assertRaisesRegex(ValueError, "window_size.*3, 5, 7"):
            build_eeg_encoder(
                {
                    "type": "patch_mlp",
                    "num_blocks": 1,
                    "window_size": 9,
                    "hidden_dim": 8,
                }
            )

    def test_cnn_centered_transformer_and_patch_have_exact_rf_ceiling(self) -> None:
        torch.manual_seed(109)
        center = 5
        delta = torch.tensor([2.0, -1.0, 0.5])
        for family in ("cnn", "window_transformer", "patch_mlp"):
            for window_size in (3, 5, 7):
                with self.subTest(family=family, window_size=window_size):
                    if family == "cnn":
                        config = {
                            "type": "cnn",
                            "in_channels": 3,
                            "latent_dim": 8,
                            "num_blocks": 1,
                            "kernel_size": window_size,
                            "dilations": [1],
                            "expansion_ratio": 2,
                            "dropout": 0.0,
                        }
                    elif family == "window_transformer":
                        config = {
                            "type": "window_transformer",
                            "in_channels": 3,
                            "latent_dim": 8,
                            "num_blocks": 1,
                            "window_size": window_size,
                            "num_heads": 2,
                            "expansion_ratio": 2,
                            "dropout": 0.0,
                            "attention_pattern": "centered_local",
                            "position_encoding_mode": "none",
                        }
                    else:
                        config = {
                            "type": "patch_mlp",
                            "in_channels": 3,
                            "latent_dim": 8,
                            "num_blocks": 1,
                            "window_size": window_size,
                            "hidden_dim": 8,
                            "dropout": 0.0,
                        }
                    encoder = build_eeg_encoder(config, time_point_count=11)
                    _activate_single_temporal_block(encoder)
                    encoder.eval()
                    radius = window_size // 2
                    baseline = torch.randn(1, 3, 11)
                    changed_inside = baseline.clone()
                    changed_inside[:, :, center - radius] += delta
                    changed_outside = baseline.clone()
                    changed_outside[:, :, center - radius - 1] += delta

                    with torch.no_grad():
                        output = encoder(
                            torch.cat(
                                (baseline, changed_inside, changed_outside), dim=0
                            )
                        )

                    self.assertEqual(encoder.left_context_samples, radius)
                    self.assertEqual(encoder.right_context_samples, radius)
                    self.assertEqual(encoder.receptive_field_samples, window_size)
                    self.assertGreater(
                        float(
                            (output[0, center] - output[1, center]).abs().max()
                        ),
                        1e-6,
                    )
                    torch.testing.assert_close(
                        output[0, center],
                        output[2, center],
                        rtol=0.0,
                        atol=1e-7,
                    )

    def test_architecture_signatures_reject_pattern_position_and_patch_mismatch(self) -> None:
        right = _window_encoder(window_size=5, time_blocks=1)
        centered_none = _window_encoder(
            window_size=5,
            time_blocks=1,
            attention_pattern="centered_local",
            position_encoding_mode="none",
        )
        centered_sine = _window_encoder(
            window_size=5,
            time_blocks=1,
            attention_pattern="centered_local",
            position_encoding_mode="sinusoidal_qk",
        )
        with self.assertRaisesRegex(RuntimeError, "architecture_signature"):
            centered_none.load_state_dict(right.state_dict(), strict=True)
        with self.assertRaisesRegex(RuntimeError, "architecture_signature"):
            centered_none.load_state_dict(centered_sine.state_dict(), strict=True)

        patch3 = build_eeg_encoder(
            {
                "type": "patch_mlp",
                "latent_dim": 8,
                "num_blocks": 1,
                "window_size": 3,
                "hidden_dim": 8,
            }
        )
        patch5 = build_eeg_encoder(
            {
                "type": "patch_mlp",
                "latent_dim": 8,
                "num_blocks": 1,
                "window_size": 5,
                "hidden_dim": 8,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "patch_mlp_architecture_signature"):
            patch5.load_state_dict(patch3.state_dict(), strict=True)

    def test_resolved_rf_metadata_is_validated_not_silently_overridden(self) -> None:
        encoder = _window_encoder(
            window_size=3,
            time_blocks=1,
            attention_pattern="centered_local",
            position_encoding_mode="none",
        )
        resolved = encoder.resolved_config()
        resolved["left_context_samples"] = 99
        with self.assertRaisesRegex(ValueError, "left_context_samples.*does not match"):
            build_eeg_encoder(resolved)

    def test_builder_aliases_conflicts_and_invalid_heads(self) -> None:
        encoder = build_eeg_encoder(
            {
                "type": "transformer",
                "in_channels": 3,
                "latent_dim": 8,
                "num_layers": 2,
                "window_size": 3,
                "num_heads": 2,
                "ffn_ratio": 3,
                "dropout": 0.0,
            },
            time_point_count=7,
        )
        self.assertEqual(encoder.encoder_type, "window_transformer")
        self.assertEqual(encoder.num_blocks, 2)
        self.assertEqual(encoder.window_size, 3)
        self.assertEqual(encoder.time_point_count, 7)
        self.assertEqual(encoder.num_heads, 2)
        self.assertEqual(encoder.attention_pattern, "right_local")
        self.assertEqual(encoder.receptive_field_samples, 5)
        first_block = encoder.temporal_blocks[0]
        self.assertIsInstance(first_block, RightLocalTransformerBlock)
        assert isinstance(first_block, RightLocalTransformerBlock)
        first_linear = first_block.ffn[0]
        self.assertIsInstance(first_linear, nn.Linear)
        assert isinstance(first_linear, nn.Linear)
        self.assertEqual(first_linear.out_features, 24)

        with self.assertRaisesRegex(ValueError, "type.*disagree"):
            build_eeg_encoder(
                {
                    "type": "cnn",
                    "encoder_type": "window_transformer",
                }
            )
        with self.assertRaisesRegex(ValueError, "num_layers.*num_blocks.*disagree"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "num_layers": 2,
                    "num_blocks": 3,
                },
                time_point_count=7,
            )
        with self.assertRaisesRegex(ValueError, "num_heads.*divide latent_dim"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "latent_dim": 10,
                    "window_size": 3,
                    "num_heads": 3,
                },
                time_point_count=7,
            )
        with self.assertRaisesRegex(ValueError, "expansion_ratio.*positive integer"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "latent_dim": 8,
                    "window_size": 3,
                    "num_heads": 2,
                    "ffn_ratio": 2.5,
                },
                time_point_count=7,
            )
        with self.assertRaisesRegex(ValueError, "shift_size must be 0"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "latent_dim": 8,
                    "window_size": 3,
                    "num_heads": 2,
                    "shift_size": 1,
                },
                time_point_count=7,
            )
        with self.assertRaisesRegex(ValueError, "attention_pattern.*right_local"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "latent_dim": 8,
                    "window_size": 3,
                    "num_heads": 2,
                    "attention_pattern": "fixed_from_zero",
                },
                time_point_count=7,
            )
        with self.assertRaisesRegex(ValueError, "time_point_count is required"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "latent_dim": 8,
                    "window_size": 3,
                    "num_heads": 2,
                }
            )
        with self.assertRaisesRegex(ValueError, "does not match data"):
            build_eeg_encoder(
                {
                    "type": "window_transformer",
                    "latent_dim": 8,
                    "window_size": 3,
                    "time_point_count": 6,
                    "num_heads": 2,
                },
                time_point_count=7,
            )

        alias_encoder = build_eeg_encoder(
            {
                "type": "window_transformer",
                "latent_dim": 8,
                "window_size": 3,
                "num_heads": 2,
                "window_partition": "right_local",
            },
            time_point_count=7,
        )
        self.assertEqual(alias_encoder.attention_pattern, "right_local")
        rebuilt = build_eeg_encoder(encoder.resolved_config())
        self.assertEqual(rebuilt.resolved_config(), encoder.resolved_config())


    def test_strict_state_dict_rejects_different_attention_head_count(self) -> None:
        source = build_eeg_encoder(
            {
                "type": "window_transformer",
                "in_channels": 3,
                "latent_dim": 8,
                "num_layers": 2,
                "window_size": 3,
                "num_heads": 2,
                "ffn_ratio": 2,
                "dropout": 0.0,
            },
            time_point_count=7,
        )
        different_heads = build_eeg_encoder(
            {
                "type": "window_transformer",
                "in_channels": 3,
                "latent_dim": 8,
                "num_layers": 2,
                "window_size": 3,
                "num_heads": 4,
                "ffn_ratio": 2,
                "dropout": 0.0,
            },
            time_point_count=7,
        )

        with self.assertRaisesRegex(RuntimeError, "right_local_architecture_signature"):
            different_heads.load_state_dict(source.state_dict(), strict=True)

    def test_strict_state_dict_rejects_legacy_fixed_window_signature(self) -> None:
        encoder = _window_encoder(window_size=3, time_blocks=1)
        legacy_state = encoder.state_dict()
        signature = legacy_state.pop("right_local_architecture_signature")
        legacy_state["window_architecture_signature"] = signature[..., 0]

        with self.assertRaisesRegex(
            RuntimeError,
            "right_local_architecture_signature|window_architecture_signature",
        ):
            encoder.load_state_dict(legacy_state, strict=True)

    def test_default_cnn_strict_state_dict_roundtrip(self) -> None:
        torch.manual_seed(104)
        config = {
            "in_channels": 3,
            "latent_dim": 8,
            "num_blocks": 2,
            "kernel_size": 3,
            "dilations": [1, 2],
            "expansion_ratio": 2,
            "dropout": 0.0,
        }
        source = EEGEncoder(**config)
        restored = build_eeg_encoder(config)
        incompatible = restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(source.encoder_type, "cnn")
        self.assertEqual(restored.encoder_type, "cnn")
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertFalse(
            any("attention" in key for key in source.state_dict())
        )
        source.eval()
        restored.eval()
        eeg = torch.randn(2, 3, 7)
        with torch.no_grad():
            torch.testing.assert_close(source(eeg), restored(eeg), rtol=0.0, atol=0.0)

    def test_target_decoder_subject_conditioning_independent_heads_backward(self) -> None:
        torch.manual_seed(105)
        encoder = _window_encoder(window_size=3, time_blocks=2)
        model = TargetAttentionDecoder(
            encoder=encoder,
            attention_mode="gated_scorer",
            n_subjects=3,
            subject_conditioning_enabled=True,
            subject_embedding_dim=6,
            subject_condition_dropout=0.0,
            subject_adaln_dropout=0.0,
            time_head_mode="independent_linear",
            time_point_count=7,
            classifier_num_layers=1,
        )
        outputs = model(
            torch.randn(2, 3, 7),
            task="pitch",
            subject_id=torch.tensor([0, 2]),
        )

        self.assertEqual(outputs["time_logits"].shape, (2, 7, 5))
        self.assertEqual(outputs["pooled_logits"].shape, (2, 5))
        self.assertEqual(outputs["attention_weights"].shape, (2, 7))
        self.assertEqual(outputs["effective_attention_weights"].shape, (2, 7))
        self.assertIsNotNone(model.subject_adapters)
        self.assertIsNotNone(model.pooled_heads)

        loss = outputs["time_logits"].square().mean() + outputs["pooled_logits"].square().mean()
        loss.backward()

        self.assertIsNotNone(model.class_heads["pitch"].weight.grad)
        assert model.pooled_heads is not None
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.pooled_heads["pitch"].parameters())
        )
        assert model.subject_adapters is not None
        adapter_projection = model.subject_adapters[0].modulation[-1]
        self.assertIsInstance(adapter_projection, nn.Linear)
        assert isinstance(adapter_projection, nn.Linear)
        self.assertIsNotNone(adapter_projection.weight.grad)
        first_block = encoder.temporal_blocks[0]
        assert isinstance(first_block, RightLocalTransformerBlock)
        self.assertIsNotNone(first_block.attention.out_proj.weight.grad)



if __name__ == "__main__":
    unittest.main()
