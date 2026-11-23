from __future__ import annotations

import csv
import os
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pb2023_eeg.data.dataset import LagDecodingDataset, make_batch_sampler
from pb2023_eeg.data.manifest import build_manifest, write_block_splits
from pb2023_eeg.data.normalization import compute_normalization_stats
from pb2023_eeg.models.encoder import EEGEncoder, TemporalResidualBlock
from pb2023_eeg.models.target_decoder import TargetAttentionDecoder, TimeAttentionPooling
from pb2023_eeg.training.augmentation import EEGInputAugmenter, build_eeg_augmenter
from pb2023_eeg.training.common import scheduled_iteration_learning_rate
from pb2023_eeg.training.swanlab_logger import SwanLabLogger, target_swanlab_metrics
from pb2023_eeg.training.target import (
    attention_shape_diagnostics,
    attention_topk_time_loss,
    export_target_outputs,
    finalize_target_augmentation_diagnostics,
    fit_target_decoder,
    resolve_checkpoint_metric,
    target_augmentation_diagnostic_totals,
    update_checkpoint_state,
)
from pb2023_eeg.workflows.report import (
    TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S,
    _activation_summary,
    _activation_ylabel,
    _attention_curve_diagnostics,
    _gaussian_display_smooth,
    _subject_time_summary,
    _time_accuracy_summary,
    plot_target_run,
)
from pb2023_eeg.workflows.reexport import (
    _gaussian_baseline_explicitly_disabled,
    _remove_stale_gaussian_exports,
)
from pb2023_eeg.workflows.train import resolve_attention_window, resolve_single_lag


class TargetPipelineSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        exp_dir = self.root / "preprocessed" / "Exp1"
        exp_dir.mkdir(parents=True)
        rng = np.random.default_rng(42)
        n_blocks = 8
        trials_per_block = 30
        n_trials = n_blocks * trials_per_block
        trial0 = np.arange(n_trials, dtype=np.int32)
        x = rng.normal(size=(n_trials, 64, 40)).astype(np.float32)
        choice = rng.integers(0, 2, size=n_trials, dtype=np.int16)
        motor = rng.integers(1, 3, size=n_trials, dtype=np.int16)
        np.savez_compressed(
            exp_dir / "Sub01_eeg_clean.npz",
            X=x,
            trial0=trial0,
            eeg_trial0=trial0,
            y_pitch=(trial0 % 5 + 1).astype(np.int16),
            y_choice=choice,
            y_motor=motor,
            valid_for_history=np.ones(n_trials, dtype=np.uint8),
            fsample=np.asarray([100], dtype=np.uint8),
            time_s=-0.1 + np.arange(40, dtype=np.float32) / 100.0,
        )
        with (exp_dir / "Sub01_meta.csv").open("w", newline="", encoding="utf-8") as f:
            fields = ["trial0", "block", "valid_eeg", "pitch", "choice", "motor"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for trial in trial0:
                writer.writerow(
                    {
                        "trial0": int(trial),
                        "block": int(trial // trials_per_block + 1),
                        "valid_eeg": 1,
                        "pitch": int(trial % 5 + 1),
                        "choice": int(choice[int(trial)]),
                        "motor": int(motor[int(trial)]),
                    }
                )
        artifact_dir = self.root / "artifacts"
        self.manifest = artifact_dir / "manifest.csv"
        self.splits = artifact_dir / "block_splits.csv"
        self.normalization = artifact_dir / "normalization.json"
        self.records = build_manifest(self.root / "preprocessed", self.manifest, experiments=[1])
        write_block_splits(self.records, self.splits, val_blocks=1, test_blocks=1, seed=11)
        compute_normalization_stats(self.records, self.splits, self.normalization)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()
























if __name__ == "__main__":
    unittest.main()
