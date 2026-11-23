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

from memory_reactivation.data.dataset import LagDecodingDataset, make_batch_sampler
from memory_reactivation.data.manifest import build_manifest, write_block_splits
from memory_reactivation.data.normalization import compute_normalization_stats
from memory_reactivation.models.encoder import EEGEncoder, TemporalResidualBlock
from memory_reactivation.models.target_decoder import TargetAttentionDecoder, TimeAttentionPooling
from memory_reactivation.training.augmentation import EEGInputAugmenter, build_eeg_augmenter
from memory_reactivation.training.common import scheduled_iteration_learning_rate
from memory_reactivation.training.swanlab_logger import SwanLabLogger, target_swanlab_metrics
from memory_reactivation.training.target import (
    attention_shape_diagnostics,
    attention_topk_time_loss,
    export_target_outputs,
    finalize_target_augmentation_diagnostics,
    fit_target_decoder,
    resolve_checkpoint_metric,
    target_augmentation_diagnostic_totals,
    update_checkpoint_state,
)
from memory_reactivation.workflows.report import (
    TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S,
    _activation_summary,
    _activation_ylabel,
    _attention_curve_diagnostics,
    _gaussian_display_smooth,
    _subject_time_summary,
    _time_accuracy_summary,
    plot_target_run,
)
from memory_reactivation.workflows.reexport import (
    _gaussian_baseline_explicitly_disabled,
    _remove_stale_gaussian_exports,
)
from memory_reactivation.workflows.train import resolve_attention_window, resolve_single_lag


def target_test_config(task="choice", protocol="block", encoder_type="window_transformer"):
    """Small synthetic configuration covering Target and subject-CV behavior."""
    encoder = dict(type=encoder_type, in_channels="auto", latent_dim=8, dropout=0.)
    if encoder_type == "cnn":
        encoder.update(num_blocks=1, kernel_size=3, expansion_ratio=2)
    else:
        encoder.update(num_layers=1, window_size=3, num_heads=2, ffn_ratio=2,
                       attention_pattern="centered_local", position_encoding_mode="sinusoidal_qk",
                       boundary_mode="noise_pad", boundary_noise_seed=42)
    return dict(
        seed=42, task=task, data=dict(cv_strategy=protocol),
        model=dict(encoder=encoder, attention_mode="gated_scorer", attention_dim=8,
                   attention_score_centering=True, attention_gate_activation="tanh_plus_one",
                   time_no_class_enabled=True, classifier_num_layers=1, classifier_dropout=0.,
                   subject_conditioning=dict(enabled=True, embedding_dim=4,
                       mode="shared_plus_residual" if protocol == "subject" else "individual"),
                   fusion=dict(enabled=False)),
        train=dict(iterations=2, validation_interval_iterations=1, log_interval_iterations=1,
                   learning_rate=.001, lr_schedule="constant", amp=False, progress_bar=False,
                   loss_type="bce_one_vs_rest", pooled_loss_type="focal_bce",
                   time_loss_weight=.5, pooled_loss_weight=1., fusion_loss_weight=0.,
                   invalid_time_loss_weight=0., attention_topk_ratio=.8,
                   checkpoint_metric="val_pooled_accuracy", checkpoint_mode="max",
                   temporal_crop_noise=dict(enabled=False, sample_interval_s=.01)))


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
