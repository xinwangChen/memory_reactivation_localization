from __future__ import annotations

import unittest

import torch

from memory_reactivation.models.classification_heads import SharedTaskClassifier
from memory_reactivation.models.encoder import EEGEncoder
from memory_reactivation.models.target_decoder import TargetAttentionDecoder
from regression.helpers_smoke import target_test_config
from memory_reactivation.workflows.train import (
    build_model,
    resolve_dataset_time_axis,
    resolved_time_axis_metadata,
)
from memory_reactivation.workflows.reexport import (
    resolve_checkpoint_n_subjects,
    resolve_model_config_for_checkpoint,
    resolve_reexport_splits,
)


def _encoder() -> EEGEncoder:
    return EEGEncoder(
        in_channels=3,
        latent_dim=6,
        num_blocks=1,
        kernel_size=3,
        dropout=0.0,
        expansion_ratio=1,
    )


class _TimeAxisDataset:
    def __init__(self, time_s: list[float]) -> None:
        self.time_s = time_s


class TargetMultiheadModelTest(unittest.TestCase):
    def test_shared_classifier_explicit_initialization(self) -> None:
        for num_layers in (1, 2):
            with self.subTest(num_layers=num_layers):
                classifier = SharedTaskClassifier(
                    latent_dim=6,
                    n_classes=5,
                    num_layers=num_layers,
                    hidden_dim=9,
                )
                linear_layers = [
                    module
                    for module in classifier.net.modules()
                    if isinstance(module, torch.nn.Linear)
                ]
                self.assertEqual(len(linear_layers), num_layers)
                for layer in linear_layers:
                    bound = layer.in_features**-0.5
                    self.assertLessEqual(float(layer.weight.abs().max()), bound)
                    self.assertGreater(int(torch.count_nonzero(layer.weight)), 0)
                    assert layer.bias is not None
                    self.assertEqual(int(torch.count_nonzero(layer.bias)), 0)

                classifier.reset_parameters()
                for layer in linear_layers:
                    assert layer.bias is not None
                    self.assertEqual(int(torch.count_nonzero(layer.bias)), 0)

    def test_shared_mode_preserves_legacy_state_dict_layout(self) -> None:
        source = TargetAttentionDecoder(
            encoder=_encoder(),
            classifier_num_layers=1,
        )
        restored = TargetAttentionDecoder(
            encoder=_encoder(),
            classifier_num_layers=1,
        )

        restored.load_state_dict(source.state_dict(), strict=True)

        self.assertEqual(restored.time_head_mode, "shared")
        self.assertIsNone(restored.time_point_count)
        self.assertFalse(restored.time_no_class_enabled)
        self.assertEqual(restored.time_class_count("pitch"), 5)
        self.assertIsNone(restored.time_no_class_index("pitch"))
        self.assertIsNone(restored.pooled_heads)
        self.assertFalse(
            any(key.startswith("pooled_heads.") for key in restored.state_dict())
        )

    def test_constructor_rejects_non_boolean_time_no_class_flag(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_no_class_enabled must be a boolean"):
            TargetAttentionDecoder(
                encoder=_encoder(),
                time_no_class_enabled="false",
            )

    def test_time_no_class_expands_only_time_heads_for_every_head_mode(self) -> None:
        for mode in ("shared", "independent_linear", "independent_mlp"):
            with self.subTest(time_head_mode=mode):
                model = TargetAttentionDecoder(
                    encoder=_encoder(),
                    time_head_mode=mode,
                    time_point_count=(None if mode == "shared" else 7),
                    time_no_class_enabled=True,
                    classifier_num_layers=2,
                    classifier_hidden_dim=9,
                    fusion_enabled=True,
                )
                outputs = model(torch.randn(2, 3, 7), task="pitch")

                self.assertEqual(outputs["time_logits"].shape, (2, 7, 6))
                self.assertEqual(outputs["pooled_logits"].shape, (2, 5))
                self.assertEqual(outputs["fusion_logits"].shape, (2, 5))
                self.assertEqual(model.time_class_count("pitch"), 6)
                self.assertEqual(model.time_no_class_index("pitch"), 5)
                self.assertEqual(model.time_class_count("choice"), 3)
                self.assertEqual(model.time_no_class_index("choice"), 2)
                self.assertIsNotNone(model.pooled_heads)
                assert model.pooled_heads is not None
                time_parameter_ids = {
                    id(parameter) for parameter in model.class_heads["pitch"].parameters()
                }
                pooled_parameter_ids = {
                    id(parameter) for parameter in model.pooled_heads["pitch"].parameters()
                }
                self.assertTrue(time_parameter_ids.isdisjoint(pooled_parameter_ids))

        with self.assertRaisesRegex(ValueError, "Unknown task"):
            model.time_class_count("unknown")
        with self.assertRaisesRegex(ValueError, "Unknown task"):
            model.time_no_class_index("unknown")

    def test_shared_time_no_class_head_and_pooled_head_have_disjoint_gradients(self) -> None:
        model = TargetAttentionDecoder(
            encoder=_encoder(),
            time_head_mode="shared",
            time_no_class_enabled=True,
            classifier_num_layers=1,
        )
        eeg = torch.randn(2, 3, 7)
        assert model.pooled_heads is not None

        model.zero_grad(set_to_none=True)
        model(eeg, task="pitch")["time_logits"].sum().backward()
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.class_heads["pitch"].parameters())
        )
        self.assertTrue(
            all(parameter.grad is None for parameter in model.pooled_heads["pitch"].parameters())
        )

        model.zero_grad(set_to_none=True)
        model(eeg, task="pitch")["pooled_logits"].sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.class_heads["pitch"].parameters())
        )
        self.assertTrue(
            any(parameter.grad is not None for parameter in model.pooled_heads["pitch"].parameters())
        )

    def test_shared_real_class_pooling_reuses_time_head_exactly(self) -> None:
        model = TargetAttentionDecoder(
            encoder=_encoder(),
            time_head_mode="shared",
            time_no_class_enabled=True,
            pooled_head_mode="shared_real_class",
            classifier_num_layers=1,
        )
        model.eval()
        outputs = model(torch.randn(3, 3, 7), task="pitch")

        expected = (
            outputs["attention_weights"][:, :, None]
            * outputs["time_logits"][..., :5]
        ).sum(dim=1)
        torch.testing.assert_close(outputs["pooled_logits"], expected)
        self.assertIsNone(model.pooled_heads)
        self.assertFalse(
            any(key.startswith("pooled_heads.") for key in model.state_dict())
        )

    def test_pool_valid_context_mask_only_changes_attention_support(self) -> None:
        model = TargetAttentionDecoder(
            encoder=_encoder(),
            pool_valid_context_only=True,
            classifier_num_layers=1,
        )
        outputs = model(torch.randn(2, 3, 7), task="pitch")

        self.assertEqual(outputs["time_logits"].shape, (2, 7, 5))
        torch.testing.assert_close(
            outputs["attention_valid_mask"],
            torch.tensor([False, True, True, True, True, True, False]),
        )
        self.assertTrue(torch.isfinite(outputs["attention_scores"]).all())
        self.assertEqual(
            int(torch.count_nonzero(outputs["attention_weights"][:, [0, -1]])),
            0,
        )
        # Top-q 排序权重必须与 pooling 支持域一致：invalid 边界位置严格为 0，
        # 否则 Top-q 会监督 pooling 明确排除的位置（历史行为，已修复）。
        self.assertEqual(
            int(torch.count_nonzero(
                outputs["time_selection_attention_weights"][:, [0, -1]]
            )),
            0,
        )
        torch.testing.assert_close(
            outputs["attention_weights"].sum(dim=1),
            torch.ones(2),
        )
        torch.testing.assert_close(
            outputs["time_selection_attention_weights"],
            outputs["attention_weights"],
        )

    def test_shared_real_class_rejects_confounding_head_contracts(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_no_class_enabled"):
            TargetAttentionDecoder(
                encoder=_encoder(),
                pooled_head_mode="shared_real_class",
                classifier_num_layers=1,
            )
        with self.assertRaisesRegex(ValueError, "classifier_num_layers=1"):
            TargetAttentionDecoder(
                encoder=_encoder(),
                time_no_class_enabled=True,
                pooled_head_mode="shared_real_class",
                classifier_num_layers=2,
            )
        with self.assertRaisesRegex(ValueError, "gated attention"):
            TargetAttentionDecoder(
                encoder=_encoder(),
                attention_mode="gated_scorer",
                time_no_class_enabled=True,
                pooled_head_mode="shared_real_class",
                classifier_num_layers=1,
            )

    def test_independent_linear_time_and_pooled_heads_are_separate(self) -> None:
        model = TargetAttentionDecoder(
            encoder=_encoder(),
            attention_mode="gated_scorer",
            time_head_mode="independent_linear",
            time_point_count=7,
            classifier_num_layers=2,
            classifier_hidden_dim=9,
        )
        eeg = torch.randn(2, 3, 7)
        outputs = model(eeg, task="pitch", attention_indices=torch.arange(1, 6))

        self.assertEqual(model.time_point_count, 7)
        self.assertEqual(outputs["time_logits"].shape, (2, 7, 5))
        self.assertEqual(outputs["pooled_logits"].shape, (2, 5))
        self.assertIsNotNone(model.pooled_heads)
        time_head = model.class_heads["pitch"]
        self.assertEqual(int(torch.count_nonzero(time_head.bias).item()), 0)
        assert model.pooled_heads is not None
        time_parameter_ids = {
            id(parameter) for parameter in model.class_heads["pitch"].parameters()
        }
        pooled_parameter_ids = {
            id(parameter) for parameter in model.pooled_heads["pitch"].parameters()
        }
        self.assertTrue(time_parameter_ids.isdisjoint(pooled_parameter_ids))

        model.zero_grad(set_to_none=True)
        outputs["time_logits"][:, 3].sum().backward()
        self.assertGreater(float(time_head.weight.grad[3].abs().sum()), 0.0)
        self.assertEqual(float(time_head.weight.grad[:3].abs().sum()), 0.0)
        self.assertEqual(float(time_head.weight.grad[4:].abs().sum()), 0.0)
        self.assertTrue(
            all(parameter.grad is None for parameter in model.pooled_heads.parameters())
        )

        model.zero_grad(set_to_none=True)
        model(eeg, task="pitch")["pooled_logits"].sum().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in model.class_heads.parameters())
        )
        self.assertTrue(
            any(
                parameter.grad is not None
                for parameter in model.pooled_heads["pitch"].parameters()
            )
        )

    def test_independent_mlp_and_time_length_validation(self) -> None:
        model = TargetAttentionDecoder(
            encoder=_encoder(),
            time_head_mode="independent_mlp",
            time_point_count=5,
            classifier_num_layers=2,
            classifier_hidden_dim=4,
        )
        outputs = model(torch.randn(2, 3, 5), task="choice")
        self.assertEqual(outputs["time_logits"].shape, (2, 5, 2))
        self.assertEqual(outputs["pooled_logits"].shape, (2, 2))

        with self.assertRaisesRegex(ValueError, "Independent time head expected"):
            model(torch.randn(2, 3, 6), task="choice")
        with self.assertRaisesRegex(ValueError, "time_point_count"):
            TargetAttentionDecoder(
                encoder=_encoder(),
                time_head_mode="independent_linear",
            )

    def test_builder_passes_data_derived_time_point_count(self) -> None:
        config = {
            "encoder": {
                "in_channels": 3,
                "latent_dim": 6,
                "num_blocks": 1,
                "kernel_size": 3,
                "dropout": 0.0,
                "expansion_ratio": 1,
            },
            "time_head_mode": "independent_linear",
            "time_no_class_enabled": True,
            "classifier_num_layers": 1,
        }
        model = build_model(config, time_point_count=11)
        self.assertEqual(model.time_head_mode, "independent_linear")
        self.assertEqual(model.time_point_count, 11)
        self.assertTrue(model.time_no_class_enabled)
        self.assertEqual(model.time_class_count("pitch"), 6)
        self.assertIsNotNone(model.pooled_heads)

        with self.assertRaisesRegex(ValueError, "time_no_class_enabled must be a boolean"):
            build_model(
                {**config, "time_no_class_enabled": "false"},
                time_point_count=11,
            )

    def test_encoder_variants_keep_no_class_multihead_contract(self) -> None:
        for encoder_type in ("cnn", "window_transformer"):
            with self.subTest(encoder=encoder_type):
                config = target_test_config("pitch", "block", encoder_type)
                model_config = config["model"]
                model_config["time_head_mode"] = "independent_linear"
                model = build_model(model_config, n_subjects=30, time_point_count=275,
                                    input_channels=64)

                self.assertEqual(model.time_head_mode, "independent_linear")
                self.assertTrue(model.time_no_class_enabled)
                self.assertEqual(model.time_class_count("pitch"), 6)
                self.assertEqual(model.time_no_class_index("pitch"), 5)
                self.assertEqual(model.class_heads["pitch"].weight.shape[-1], 6)
                time_logits = model.class_heads["pitch"](
                    torch.zeros(1, 275, model.encoder.latent_dim)
                )
                self.assertEqual(time_logits.shape, (1, 275, 6))
                self.assertIsNotNone(model.pooled_heads)
                assert model.pooled_heads is not None
                pooled_logits = model.pooled_heads["pitch"](
                    torch.zeros(1, model.encoder.latent_dim)
                )
                self.assertEqual(pooled_logits.shape, (1, 5))
                self.assertFalse(model.fusion_enabled)
                self.assertIsNone(model.fusion)
                self.assertIsNone(model.fusion_heads)


    def test_builder_falls_back_to_saved_time_point_count_and_rejects_conflict(self) -> None:
        config = {
            "encoder": {
                "in_channels": 3,
                "latent_dim": 6,
                "num_blocks": 1,
                "kernel_size": 3,
                "dropout": 0.0,
                "expansion_ratio": 1,
            },
            "time_head_mode": "independent_linear",
            "resolved_time_point_count": 9,
            "classifier_num_layers": 1,
        }
        model = build_model(config)
        self.assertEqual(model.time_point_count, 9)
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_model(config, time_point_count=8)

    def test_historical_direct_linear_head_checkpoint_loads_strictly(self) -> None:
        source = TargetAttentionDecoder(
            encoder=_encoder(),
            classifier_num_layers=1,
        )
        historical_state = {}
        for key, value in source.state_dict().items():
            historical_key = key
            for task in ("pitch", "choice", "motor"):
                historical_key = historical_key.replace(
                    f"class_heads.{task}.net.", f"class_heads.{task}."
                )
            historical_state[historical_key] = value.clone()
        self.assertIn("class_heads.pitch.weight", historical_state)
        self.assertNotIn("class_heads.pitch.net.weight", historical_state)

        base_config = {
            "encoder": {
                "in_channels": 3,
                "latent_dim": 6,
                "num_blocks": 1,
                "kernel_size": 3,
                "dropout": 0.0,
                "expansion_ratio": 1,
            }
        }
        resolved = resolve_model_config_for_checkpoint(base_config, historical_state)
        self.assertEqual(resolved["classifier_num_layers"], 1)
        self.assertFalse(resolved["time_no_class_enabled"])
        restored = build_model(resolved)
        restored.load_state_dict(historical_state, strict=True)
        for key, expected in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], expected)

    def test_recent_single_layer_checkpoint_schema_still_loads(self) -> None:
        source = TargetAttentionDecoder(encoder=_encoder(), classifier_num_layers=1)
        resolved = resolve_model_config_for_checkpoint({}, source.state_dict())
        self.assertEqual(resolved["classifier_num_layers"], 1)
        self.assertFalse(resolved["time_no_class_enabled"])
        restored = TargetAttentionDecoder(encoder=_encoder(), classifier_num_layers=1)
        restored.load_state_dict(source.state_dict(), strict=True)

    def test_recent_two_layer_checkpoint_config_is_inferred(self) -> None:
        source = TargetAttentionDecoder(
            encoder=_encoder(),
            classifier_num_layers=2,
            classifier_hidden_dim=9,
        )
        resolved = resolve_model_config_for_checkpoint({}, source.state_dict())
        self.assertEqual(resolved["classifier_num_layers"], 2)
        self.assertEqual(resolved["classifier_hidden_dim"], 9)

    def test_reexport_infers_and_strictly_loads_time_no_class_checkpoints(self) -> None:
        base_encoder_config = {
            "in_channels": 3,
            "latent_dim": 6,
            "num_blocks": 1,
            "kernel_size": 3,
            "dropout": 0.0,
            "expansion_ratio": 1,
        }
        cases = (
            ("shared", None),
            ("independent_linear", 7),
            ("independent_mlp", 7),
        )
        for mode, time_point_count in cases:
            with self.subTest(time_head_mode=mode):
                source = TargetAttentionDecoder(
                    encoder=_encoder(),
                    time_head_mode=mode,
                    time_point_count=time_point_count,
                    time_no_class_enabled=True,
                    classifier_num_layers=2,
                    classifier_hidden_dim=9,
                )
                config = {
                    "encoder": base_encoder_config,
                    "time_head_mode": mode,
                }
                resolved = resolve_model_config_for_checkpoint(config, source.state_dict())

                self.assertTrue(resolved["time_no_class_enabled"])
                restored = build_model(resolved, time_point_count=time_point_count)
                restored.load_state_dict(source.state_dict(), strict=True)
                self.assertEqual(restored.time_class_count("pitch"), 6)

    def test_reexport_rejects_time_no_class_flag_and_schema_conflicts(self) -> None:
        ordinary = TargetAttentionDecoder(
            encoder=_encoder(),
            classifier_num_layers=1,
        )
        no_class = TargetAttentionDecoder(
            encoder=_encoder(),
            time_no_class_enabled=True,
            classifier_num_layers=1,
        )
        with self.assertRaisesRegex(ValueError, "time_no_class_enabled conflicts"):
            resolve_model_config_for_checkpoint(
                {"time_no_class_enabled": True},
                ordinary.state_dict(),
            )
        with self.assertRaisesRegex(ValueError, "time_no_class_enabled conflicts"):
            resolve_model_config_for_checkpoint(
                {"time_no_class_enabled": False},
                no_class.state_dict(),
            )
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            resolve_model_config_for_checkpoint(
                {"time_no_class_enabled": "true"},
                no_class.state_dict(),
            )

        mixed_state = dict(no_class.state_dict())
        for suffix in ("weight", "bias"):
            key = f"class_heads.choice.net.{suffix}"
            mixed_state[key] = mixed_state[key][:-1].clone()
        with self.assertRaisesRegex(ValueError, "disagree"):
            resolve_model_config_for_checkpoint({}, mixed_state)

    def test_reexport_preserves_checkpoint_subject_embedding_size(self) -> None:
        state = {"subject_embedding.weight": torch.zeros(12, 4)}
        self.assertEqual(resolve_checkpoint_n_subjects(7, state), 12)
        with self.assertRaisesRegex(ValueError, "outside the checkpoint"):
            resolve_checkpoint_n_subjects(13, state)

    def test_reexport_defaults_to_saved_splits_and_allows_explicit_override(self) -> None:
        self.assertEqual(resolve_reexport_splits(None, {"splits": ["val"]}), ("val",))
        self.assertEqual(
            resolve_reexport_splits(["test"], {"splits": ["val"]}),
            ("test",),
        )

    def test_dataset_time_axes_must_match_exactly(self) -> None:
        datasets = {
            "train": _TimeAxisDataset([-0.1, 0.0, 0.1]),
            "val": _TimeAxisDataset([-0.1, 0.0, 0.1]),
            "test": _TimeAxisDataset([-0.1, 0.0, 0.1]),
        }
        times = resolve_dataset_time_axis(datasets, reference_split="train")
        metadata = resolved_time_axis_metadata(times)

        self.assertTrue(torch.equal(times, torch.tensor([-0.1, 0.0, 0.1])))
        self.assertEqual(metadata["resolved_time_point_count"], 3)
        self.assertEqual(len(str(metadata["resolved_time_axis_sha256"])), 64)

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            resolve_dataset_time_axis(
                {
                    "train": datasets["train"],
                    "val": _TimeAxisDataset([-0.1, 0.0]),
                },
                reference_split="train",
            )
        with self.assertRaisesRegex(ValueError, "value mismatch"):
            resolve_dataset_time_axis(
                {
                    "train": datasets["train"],
                    "val": _TimeAxisDataset([-0.1, 0.0, 0.100001]),
                },
                reference_split="train",
            )


if __name__ == "__main__":
    unittest.main()
