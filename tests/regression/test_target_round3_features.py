from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from memory_reactivation.models.encoder import EEGEncoder
from memory_reactivation.models.target_decoder import TargetAttentionDecoder
from memory_reactivation.training.common import trainable_parameters
from memory_reactivation.training.target import fit_target_decoder, run_target_epoch
from memory_reactivation.workflows.train import (
    configure_target_freezing,
    load_pretrained_dpc_components,
    resolve_export_splits,
    resolve_target_dataset_splits,
    resolve_pretrained_dpc_config,
)


def _encoder(latent_dim: int = 4) -> EEGEncoder:
    return EEGEncoder(
        in_channels=3,
        latent_dim=latent_dim,
        num_blocks=2,
        kernel_size=3,
        dilations=[1, 2],
        expansion_ratio=1,
        dropout=0.2,
    )


def _target(*, latent_dim: int = 4, subject: bool = True) -> TargetAttentionDecoder:
    return TargetAttentionDecoder(
        encoder=_encoder(latent_dim),
        attention_mode="gated_scorer",
        n_subjects=3,
        subject_conditioning_enabled=subject,
        subject_embedding_dim=latent_dim,
        subject_condition_dropout=0.5,
        subject_adaln_dropout=0.2,
        classifier_num_layers=1,
        fusion_enabled=True,
    )




class _FixedPredictionTarget(torch.nn.Module):
    fusion_enabled = True

    def forward(self, eeg, task, attention_indices, subject_id=None):
        del task, attention_indices, subject_id
        prediction = eeg[:, 0, 0].to(dtype=torch.long)
        logits = torch.nn.functional.one_hot(prediction, num_classes=2).float() * 8.0
        weights = torch.ones(eeg.shape[0], 1, device=eeg.device)
        return {
            "time_logits": logits[:, None, :],
            "pooled_logits": logits,
            "fusion_logits": logits,
            "attention_weights": weights,
        }


class TargetRound3FeatureTest(unittest.TestCase):
    def test_target_epoch_applies_minimum_cell_class_count(self) -> None:
        target = torch.tensor([0, 0, 1, 1, 0, 1])
        prediction = torch.tensor([0, 0, 1, 1, 1, 0])
        batch = {
            "eeg": prediction.float().reshape(-1, 1, 1),
            "target": target,
            "subject_id": torch.zeros(6, dtype=torch.long),
            "choice_cell_id": torch.tensor([0, 0, 0, 0, 1, 1]),
        }
        common = {
            "model": _FixedPredictionTarget(),
            "loader": [batch],
            "task": "choice",
            "attention_indices": torch.tensor([0]),
            "loss_weights": {"time": 1.0, "pooled": 1.0, "fusion": 1.0},
            "top_k": 1,
            "top_ratio": 1.0,
            "label_smoothing": 0.0,
            "device": torch.device("cpu"),
            "amp": False,
            "progress_bar_enabled": False,
        }
        default_metrics = run_target_epoch(**common)
        filtered_metrics = run_target_epoch(**common, min_cell_class_count=2)

        self.assertEqual(default_metrics["fusion_cell_count"], 2.0)
        self.assertAlmostEqual(default_metrics["fusion_subject_cell_balanced_accuracy"], 0.5)
        self.assertEqual(filtered_metrics["fusion_cell_count"], 1.0)
        self.assertEqual(filtered_metrics["fusion_subject_cell_count"], 1.0)
        self.assertAlmostEqual(filtered_metrics["fusion_cell_balanced_accuracy"], 1.0)
        self.assertAlmostEqual(filtered_metrics["fusion_subject_cell_balanced_accuracy"], 1.0)

    def test_fit_target_uses_eval_cell_threshold_for_checkpoint(self) -> None:
        target = torch.tensor([0, 0, 1, 1, 0, 1])
        batch = {
            "eeg": torch.randn(6, 3, 16),
            "target": target,
            "subject_id": torch.zeros(6, dtype=torch.long),
            "choice_cell_id": torch.tensor([0, 0, 0, 0, 1, 1]),
        }
        config = {
            "data": {"eval_min_cell_class_count": 2},
            "train": {
                "iterations": 1,
                "validation_interval_iterations": 1,
                "log_interval_iterations": 1,
                "learning_rate": 1e-3,
                "weight_decay": 0.0,
                "time_loss_weight": 0.2,
                "pooled_loss_weight": 1.0,
                "fusion_loss_weight": 1.0,
                "checkpoint_metric": "val_fusion_subject_cell_balanced_accuracy",
                "checkpoint_mode": "max",
                "amp": False,
                "progress_bar": False,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            history = fit_target_decoder(
                model=_target(subject=False),
                train_loader=[batch],
                val_loader=[batch],
                task="choice",
                attention_indices=torch.arange(16),
                config=config,
                output_dir=output_dir,
                device=torch.device("cpu"),
            )
            self.assertTrue((output_dir / "best_val.ckpt").exists())

        row = history[0]
        self.assertEqual(row["train_fusion_cell_count"], 2.0)
        self.assertEqual(row["val_fusion_cell_count"], 1.0)
        self.assertIn("val_fusion_subject_cell_balanced_accuracy", row)

    def test_export_split_resolution(self) -> None:
        self.assertEqual(resolve_export_splits({}), ("val", "test"))
        self.assertEqual(resolve_export_splits({"splits": ["val"]}), ("val",))
        self.assertEqual(
            resolve_export_splits({"splits": ["test", "val", "test"]}),
            ("test", "val"),
        )
        with self.assertRaisesRegex(ValueError, "val and/or test"):
            resolve_export_splits({"splits": []})
        with self.assertRaisesRegex(ValueError, "val and/or test"):
            resolve_export_splits({"splits": ["train"]})

        self.assertEqual(
            resolve_target_dataset_splits({"splits": ["val"]}),
            ("train", "val"),
        )
        self.assertEqual(
            resolve_target_dataset_splits({}),
            ("train", "val", "test"),
        )

    def test_pretrained_config_requires_path_and_preserves_provenance(self) -> None:
        with self.assertRaisesRegex(ValueError, "no checkpoint"):
            resolve_pretrained_dpc_config(
                {"pretrained_dpc": {"enabled": True}},
                None,
            )
        self.assertIsNone(
            resolve_pretrained_dpc_config(
                {"pretrained_dpc": {"enabled": False}},
                None,
            )
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            configured = Path(temp_dir) / "configured.ckpt"
            override = Path(temp_dir) / "override.ckpt"
            configured.write_bytes(b"configured")
            override.write_bytes(b"override")
            resolved = resolve_pretrained_dpc_config(
                {
                    "pretrained_dpc": {
                        "enabled": True,
                        "checkpoint": str(configured),
                        "source_experiment_id": "D31",
                        "load_subject_conditioning": True,
                    }
                },
                override,
            )
        assert resolved is not None
        self.assertEqual(resolved["checkpoint"], str(override.resolve()))
        self.assertEqual(resolved["source_experiment_id"], "D31")
        self.assertTrue(resolved["load_subject_conditioning"])




    def test_freezing_preserves_trainable_heads_and_eval_mode(self) -> None:
        model = _target()
        summary = configure_target_freezing(
            model,
            {"freeze": {"encoder": True, "subject_conditioning": True}},
        )
        self.assertTrue(summary["encoder"])
        self.assertTrue(summary["subject_conditioning"])
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        assert model.subject_embedding is not None
        assert model.subject_adapters is not None
        self.assertTrue(
            all(
                not parameter.requires_grad
                for module in (model.subject_embedding, model.subject_adapters)
                for parameter in module.parameters()
            )
        )
        self.assertTrue(any(parameter.requires_grad for parameter in model.class_heads.parameters()))

        model.train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.subject_embedding.training)
        self.assertFalse(model.subject_adapters.training)
        self.assertTrue(model.attention.training)

        expected_ids = {
            id(parameter) for parameter in model.parameters() if parameter.requires_grad
        }
        optimizer = torch.optim.AdamW(trainable_parameters(model), lr=1e-3)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_ids, expected_ids)


if __name__ == "__main__":
    unittest.main()
