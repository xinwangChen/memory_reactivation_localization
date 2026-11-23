from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from pb2023_eeg.models.encoder import EEGEncoder
from pb2023_eeg.models.target_decoder import TargetAttentionDecoder
from pb2023_eeg.training.metrics import BalancedAccuracyAccumulator
from pb2023_eeg.training.target import run_target_epoch


class _MaskedTargetModel(nn.Module):
    """Channel 0 defines valid positions; channel 1 defines binary logits."""

    def forward(self, eeg, task, attention_indices, subject_id=None):
        valid = eeg[:, 0].bool()
        logits = torch.stack((eeg[:, 1], -eeg[:, 1]), dim=-1)
        return {
            "time_logits": logits,
            "pooled_logits": logits[:, 0],
            "attention_weights": valid.float() / valid.sum(dim=1, keepdim=True),
            "attention_valid_mask": valid,
        }


def _batch(masks: list[list[int]], scores: list[list[float]], targets: list[int]) -> dict:
    return {
        "eeg": torch.stack((torch.tensor(masks), torch.tensor(scores)), dim=1).float(),
        "target": torch.tensor(targets),
        "subject_id": torch.zeros(len(targets), dtype=torch.long),
        "choice_cell_id": torch.zeros(len(targets), dtype=torch.long),
    }


def _evaluate(model, batches, min_count=1):
    return run_target_epoch(
        model=model, loader=batches, task="choice", attention_indices=torch.arange(5),
        loss_weights={"time": 1.0, "pooled": 1.0}, top_k=None, top_ratio=1.0,
        label_smoothing=0.0, device=torch.device("cpu"), amp=False,
        progress_bar_enabled=False, min_cell_class_count=min_count,
    )


class TargetCellSupportTest(unittest.TestCase):
    def test_cnn_boundary_mask_preserves_complete_cells(self) -> None:
        model = TargetAttentionDecoder(
            EEGEncoder(in_channels=2, latent_dim=4, num_blocks=2, kernel_size=3,
                       dropout=0.0, expansion_ratio=1),
            classifier_num_layers=1, pool_valid_context_only=True,
        )
        batch = _batch([[1] * 5] * 2, [[1.0] * 5, [-1.0] * 5], [0, 1])
        metrics = _evaluate(model, [batch])
        self.assertEqual(metrics["selected_time_count_per_sample"], 1.0)
        self.assertEqual(metrics["time_cell_count"], 1.0)
        self.assertTrue(math.isfinite(metrics["time_cell_balanced_accuracy"]))
        self.assertEqual(_evaluate(model, [batch], min_count=2)["time_cell_count"], 0.0)

    def test_variable_valid_counts_preserve_time_prediction_accuracy(self) -> None:
        batch = _batch(
            [[1, 1, 1, 1, 1], [1, 0, 0, 0, 0]],
            [[1, -1, 1, -1, 1], [-1, -1, -1, -1, -1]],
            [0, 1],
        )
        metrics = _evaluate(_MaskedTargetModel(), [batch])
        self.assertEqual(metrics["time_cell_count"], 1.0)
        self.assertAlmostEqual(metrics["time_cell_balanced_accuracy"], 0.8)
        self.assertAlmostEqual(metrics["time_accuracy"], 4.0 / 6.0)

    def test_minimum_support_counts_trials_across_batches_and_excludes_empty_trials(self) -> None:
        first = _batch([[1] * 5] * 2, [[1] * 5, [-1] * 5], [0, 1])
        second = _batch([[0, 1, 0, 0, 0]], [[1] * 5], [0])
        third = _batch([[0, 0, 1, 0, 0]], [[-1] * 5], [1])
        model = _MaskedTargetModel()
        incomplete = _evaluate(model, [first, second], min_count=2)
        self.assertEqual(incomplete["time_cell_count"], 0.0)
        complete = _evaluate(model, [first, second, third], min_count=2)
        self.assertEqual(complete["time_cell_count"], 1.0)
        self.assertEqual(complete["time_cell_balanced_accuracy"], 1.0)

    def test_support_mask_does_not_change_confusion_and_survives_finalization(self) -> None:
        accumulator = BalancedAccuracyAccumulator(2, min_cell_class_count=2)
        for _ in range(2):
            accumulator.update(
                torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]),
                torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long),
                cell_support_mask=torch.tensor([True, False, True]),
            )
            metrics = accumulator.metrics()
        self.assertEqual(accumulator.cell_class_support[(0, 0)].tolist(), [2, 2])
        self.assertEqual(accumulator.confusion.tolist(), [[2, 2], [0, 2]])
        self.assertEqual(metrics["cell_count"], 1.0)
        self.assertEqual(metrics["cell_balanced_accuracy"], 0.75)
        self.assertEqual(accumulator.metrics(), metrics)
        with self.assertRaisesRegex(ValueError, "cell_support_mask must align"):
            accumulator.update(torch.tensor([0]), torch.tensor([0]),
                               cell_support_mask=torch.tensor([True, False]))
        self.assertEqual(accumulator.metrics(), metrics)


if __name__ == "__main__":
    unittest.main()
