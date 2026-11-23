from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pb2023_eeg.training.target import (
    export_target_outputs,
    time_roll_eeg,
    time_shuffle_eeg,
)
from pb2023_eeg.workflows.report import (
    _add_event_markers,
    load_plot_config,
    normalize_plot_input_kinds,
    normalize_plot_splits,
    plot_target_conditions,
    plot_target_run,
    resolve_plot_config,
    resolve_plot_time_axis_limits,
)
from pb2023_eeg.workflows.reexport import resolve_time_roll_shift_samples


class _FixedPitchTargetModel(torch.nn.Module):
    fusion_enabled = False

    def __init__(self, predictions: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("predictions", predictions.to(dtype=torch.long))

    def class_count(self, task: str) -> int:
        return 5

    def class_indices(self, task: str) -> tuple[int, ...]:
        return tuple(range(5))

    def time_class_count(self, task: str) -> int:
        return 5

    def time_no_class_index(self, task: str) -> None:
        return None

    def forward(
        self,
        eeg: torch.Tensor,
        task: str,
        attention_indices: torch.Tensor,
        subject_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        predictions = self.predictions.to(eeg.device)
        time_logits = torch.full((*predictions.shape, 5), -4.0, device=eeg.device)
        time_logits.scatter_(dim=-1, index=predictions[..., None], value=4.0)
        pooled_prediction = predictions[:, 0]
        pooled_logits = torch.full((len(predictions), 5), -4.0, device=eeg.device)
        pooled_logits.scatter_(
            dim=-1, index=pooled_prediction[:, None], value=4.0
        )
        attention_count = int(attention_indices.numel())
        weights = torch.full(
            (len(predictions), attention_count),
            1.0 / attention_count,
            device=eeg.device,
        )
        return {
            "pooled_logits": pooled_logits,
            "time_logits": time_logits,
            "attention_weights": weights,
            "attention_scores": torch.zeros_like(weights),
        }


class TargetTimeCellExportTest(unittest.TestCase):
    def test_time_figures_accept_an_exact_display_range(self) -> None:
        base_config = {"activation": {"enabled": False}}
        resolved_config = resolve_plot_config(
            base_config,
            time_axis_start_s=0.0,
            time_axis_end_s=2.25,
        )
        self.assertNotIn("time_axis", base_config)
        self.assertEqual(
            resolve_plot_time_axis_limits(resolved_config),
            (0.0, 2.25),
        )
        with self.assertRaises(ValueError):
            resolve_plot_config(base_config, time_axis_start_s=0.0)
        self.assertEqual(
            normalize_plot_input_kinds(("real", "real", "time_roll")),
            ("real", "time_roll"),
        )
        self.assertEqual(
            normalize_plot_splits(split for split in ("val", "test", "val")),
            ("val", "test"),
        )

        figure, axis = plt.subplots()
        try:
            axis.plot([0.004, 2.244], [0.0, 1.0])
            _add_event_markers(
                axis,
                {
                    "event_markers": {
                        "enabled": True,
                        "markers": [{"time_s": 0.0, "label": "start"}],
                    },
                    "time_axis": {"start_s": 0.0, "end_s": 2.25},
                },
            )
            self.assertEqual(axis.get_xlim(), (0.0, 2.25))
            self.assertEqual(axis.texts[0].get_ha(), "left")
        finally:
            plt.close(figure)

        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "config_snapshot.yaml"
            snapshot.write_text(
                "plots:\n"
                "  activation:\n"
                "    enabled: false\n"
                "  time_axis:\n"
                "    start_s: 0.0\n"
                "    end_s: 2.25\n",
                encoding="utf-8",
            )
            loaded = load_plot_config(snapshot)
            self.assertEqual(
                resolve_plot_time_axis_limits(loaded),
                (0.0, 2.25),
            )
            snapshot.write_text(
                "task: pitch\nmodel: {}\ntrain: {}\ndata: {}\n",
                encoding="utf-8",
            )
            self.assertEqual(load_plot_config(snapshot), {})
            with self.assertRaises(FileNotFoundError):
                plot_target_conditions(
                    Path(temp_dir),
                    input_kinds=("real",),
                    splits=("val",),
                )

    def test_evaluation_time_controls_are_deterministic_and_channel_aligned(self) -> None:
        base = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
        eeg = torch.cat((base, base + 100.0), dim=1).repeat(2, 1, 1)
        identity = {
            "record_id": torch.tensor([3, 3]),
            "trial0": torch.tensor([10, 11]),
            "lag": torch.tensor([1, 1]),
        }
        shuffled_a = time_shuffle_eeg(eeg, seed=17, **identity)
        shuffled_b = time_shuffle_eeg(eeg, seed=17, **identity)
        torch.testing.assert_close(shuffled_a, shuffled_b)
        torch.testing.assert_close(
            shuffled_a[:, 1] - shuffled_a[:, 0],
            torch.full((2, 12), 100.0),
        )
        self.assertEqual(
            sorted(shuffled_a[0, 0].tolist()),
            sorted(eeg[0, 0].tolist()),
        )
        self.assertFalse(torch.equal(shuffled_a[0], shuffled_a[1]))

        rolled = time_roll_eeg(eeg, shift_samples=3)
        torch.testing.assert_close(rolled, torch.roll(eeg, shifts=3, dims=-1))
        shift_samples, resolved_s = resolve_time_roll_shift_samples(
            torch.arange(225, dtype=torch.float64) / 100.0,
            0.5,
        )
        self.assertEqual(shift_samples, 50)
        self.assertAlmostEqual(resolved_s, 0.5)

    def test_all_target_input_exports_feed_standalone_plots(self) -> None:
        targets = torch.arange(5).repeat(2)
        predictions = torch.stack((targets, torch.zeros_like(targets)), dim=1)
        batch = {
            "eeg": torch.zeros(len(targets), 1, 2),
            "target": targets,
            "subject_id": torch.zeros(len(targets), dtype=torch.long),
            "record_id": torch.arange(len(targets)),
            "exp": torch.ones(len(targets), dtype=torch.long),
            "trial0": torch.arange(len(targets)),
            "lag": torch.ones(len(targets), dtype=torch.long),
            "current_stimulus": torch.tensor([1] * 5 + [3] * 5),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            for input_kind in ("real", "gaussian", "time_shuffle", "time_roll"):
                export_target_outputs(
                    model=_FixedPitchTargetModel(predictions),
                    loader=[batch],
                    task="pitch",
                    device=torch.device("cpu"),
                    amp=False,
                    output_dir=output_dir,
                    split="val",
                    attention_indices=torch.arange(2),
                    time_s=[0.0, 0.1],
                    progress_bar_enabled=False,
                    input_kind=input_kind,
                    gaussian_seed=7,
                    time_shuffle_seed=(19 if input_kind == "time_shuffle" else None),
                    time_roll_shift_samples=(1 if input_kind == "time_roll" else None),
                )

            with (output_dir / "val_time_cell_metrics.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                metric_rows = list(csv.DictReader(file))
            with (output_dir / "val_time_cell_summary.csv").open(
                newline="", encoding="utf-8"
            ) as file:
                summary_rows = list(csv.DictReader(file))
            gaussian_path = output_dir / "val_gaussian_time_cell_summary.csv"
            self.assertTrue(gaussian_path.is_file())

            self.assertEqual(len(metric_rows), 4)
            self.assertEqual(
                {int(row["current_stimulus"]) for row in metric_rows}, {1, 3}
            )
            self.assertTrue(
                all(row["current_stimulus"] == row["cell_id"] for row in metric_rows)
            )
            by_time = {int(row["time_index"]): row for row in summary_rows}
            self.assertAlmostEqual(
                float(by_time[0]["subject_cell_balanced_accuracy"]), 1.0
            )
            self.assertAlmostEqual(
                float(by_time[1]["subject_cell_balanced_accuracy"]), 0.2
            )
            self.assertEqual(int(by_time[0]["cell_count"]), 2)
            self.assertEqual(int(by_time[0]["subject_cell_count"]), 1)
            self.assertEqual(by_time[0]["time_output_class_count"], "5")

            plotted = plot_target_run(
                output_dir, splits=("val",), output_dir=output_dir / "figures_test"
            )["time_accuracy"]
            self.assertEqual(
                {row["display_metric"] for row in plotted},
                {"subject_cell_balanced_accuracy"},
            )
            self.assertEqual({row["input_kind"] for row in plotted}, {"real", "gaussian"})

            input_kinds = ("real", "gaussian", "time_shuffle", "time_roll")
            condition_summaries = plot_target_conditions(
                output_dir,
                input_kinds=input_kinds,
                splits=(split for split in ("val",)),
                output_root=output_dir / "figures_by_input",
            )
            self.assertEqual(tuple(condition_summaries), input_kinds)
            for input_kind, condition_summary in condition_summaries.items():
                figure_dir = output_dir / "figures_by_input" / input_kind
                self.assertEqual(
                    {row["input_kind"] for row in condition_summary["time_accuracy"]},
                    {input_kind},
                )
                self.assertTrue((figure_dir / "04_effective_attention_normalized_val.png").is_file())
                self.assertTrue((figure_dir / "time_accuracy_val.png").is_file())
                self.assertTrue(
                    (figure_dir / "attention_per_subject" / "0_val.png").is_file()
                )
                self.assertTrue(
                    (
                        figure_dir
                        / "time_by_stimulus"
                        / "attention_by_stimulus_1_val.png"
                    ).is_file()
                )
                self.assertTrue(
                    (
                        figure_dir
                        / "time_by_stimulus"
                        / "time_accuracy_by_stimulus_1_val.png"
                    ).is_file()
                )


if __name__ == "__main__":
    unittest.main()
