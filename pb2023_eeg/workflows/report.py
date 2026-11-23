"""workflows.report: shared Target report operations."""
from __future__ import annotations
import argparse
import csv
from collections.abc import Iterable, Mapping
from pathlib import Path
from pb2023_eeg.training.common import write_rows_csv
from pb2023_eeg.analysis.decoding import STANDALONE_INPUT_KINDS, _attention_curve_diagnostics, _attention_curve_label, _default_plot_config, _gaussian_baseline_explicitly_disabled, _read_csv, _require_standalone_export, _rows_with_input_kind, load_plot_config, normalize_plot_input_kinds, normalize_plot_splits, resolve_plot_config
from pb2023_eeg.visualization.layouts import _plot_activation, _plot_attention, _plot_attention_by_current_stimulus, _plot_attention_per_subject, _plot_per_subject_time_accuracy, _plot_time_accuracy, _plot_time_by_current_stimulus, _plot_trials
from pb2023_eeg.analysis.decoding import ATTENTION_FIGURE_STEMS, DEFAULT_EVENT_MARKERS, KIND_COLORS, KIND_LABELS, KIND_ORDER, LEGACY_ACTIVATION_DEFINITION, LEGACY_TIME_SCORE_MODE, PLOT_INPUT_KINDS, STANDALONE_INPUT_KINDS, STANDALONE_PLOT_EXPORT_STEMS, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S, _STIMULUS_COLORS, _activation_metadata_by_kind, _activation_summary, _activation_ylabel, _attention_by_current_stimulus_summary, _attention_component_summary, _attention_curve_diagnostics, _attention_curve_label, _attention_plot_value, _attention_pool_valid, _attention_subject_curves, _branch_subject_values, _build_attention_subject_matrix, _build_subject_matrix_from_rows, _chance, _cluster_from_t, _default_plot_config, _event_markers, _filter_grouped_for_plot, _filter_rows_for_plot, _find_peak_windows, _float, _gaussian_baseline_explicitly_disabled, _gaussian_display_smooth, _input_kind, _input_kinds, _int, _is_plottable_input_kind, _kind_color, _kind_label, _kind_sort_key, _read_csv, _require_standalone_export, _rows_with_input_kind, _sem, _sign_flip_cluster_perm, _standalone_export_prefix, _stimulus_attention_trial_key, _subject_time_summary, _summary_by_kind, _time_accuracy_summary, _time_accuracy_summary_for_plot, _trial_summary, attention_figure_filename, load_plot_config, normalize_plot_input_kinds, normalize_plot_splits, resolve_plot_config, resolve_plot_time_axis_limits
from pb2023_eeg.visualization.layouts import _add_event_markers, _annotate_bars, _draw_peak_windows, _pad_ylim_for_bar_labels, _plot_activation, _plot_attention, _plot_attention_by_current_stimulus, _plot_attention_mass_time, _plot_attention_per_subject, _plot_attention_score_time, _plot_attention_stimulus_summary, _plot_per_subject_time_accuracy, _plot_time_accuracy, _plot_time_by_current_stimulus, _plot_trials, _save_time_accuracy_figure, _save_time_evidence_figure


def parse_args(argv=None) -> argparse.Namespace:
    """解析导出目录、条件列表、绘图配置和输出位置。"""
    parser = argparse.ArgumentParser(description="Plot target-decoder outputs from exported CSV files.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Figure directory for legacy/single-condition mode, or the root "
            "containing one subdirectory per condition with --input-kinds."
        ),
    )
    parser.add_argument("--splits", nargs="+", default=None, help="Override snapshot plots.splits.")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-kind",
        choices=STANDALONE_INPUT_KINDS,
        help="Plot one exported input condition as a complete standalone figure set.",
    )
    input_group.add_argument(
        "--input-kinds",
        nargs="+",
        choices=STANDALONE_INPUT_KINDS,
        help=(
            "Plot several exported conditions in one command, writing each complete "
            "figure set under OUTPUT_DIR/<input-kind>."
        ),
    )
    parser.add_argument(
        "--config-snapshot",
        type=Path,
        help=(
            "YAML containing either a full run snapshot or a plots mapping. "
            "Defaults to RUN_DIR/config_snapshot.yaml when present."
        ),
    )
    parser.add_argument(
        "--time-axis-start-s",
        type=float,
        help="Override plots.time_axis.start_s; requires --time-axis-end-s.",
    )
    parser.add_argument(
        "--time-axis-end-s",
        type=float,
        help="Override plots.time_axis.end_s; requires --time-axis-start-s.",
    )
    return parser.parse_args(argv)


def _validate_report_inputs(run_dir, splits, plot_config, input_kind):
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Report run directory does not exist: {run_dir}")
    for split in splits:
        kinds = [input_kind or "real"]
        if input_kind is None and not _gaussian_baseline_explicitly_disabled(plot_config):
            gaussian = (plot_config or {}).get("gaussian_baseline") or {}
            if gaussian.get("enabled", False) or any(
                (run_dir / f"{split}_gaussian_{stem}.csv").is_file() for stem in STANDALONE_PLOT_EXPORT_STEMS
            ):
                kinds.append("gaussian")
        for kind in kinds:
            prefix = _require_standalone_export(run_dir, split=split, input_kind=kind)
            if ((plot_config or {}).get("activation") or {}).get("enabled", False):
                path = run_dir / f"{prefix}time_activation.csv"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing enabled activation export: {path}")
                with path.open(newline="", encoding="utf-8") as stream:
                    if next(csv.DictReader(stream), None) is None:
                        raise ValueError(f"Empty enabled activation export: {path}")


def plot_target_run(
    run_dir: Path,
    *,
    splits: Iterable[str] = ("val", "test"),
    output_dir: Path | None = None,
    plot_config: Mapping[str, object] | None = None,
    input_kind: str | None = None,
) -> dict[str, list[dict[str, object]]]:
    """读取一个运行目录的导出 CSV，生成全部汇总表与图。

    参数：
        run_dir: 包含 ``{split}_*.csv`` 和可选配置快照的运行目录。
        splits: 需要处理的 split 名称；字符串会视为单个 split。
        output_dir: 图表目录，缺省为 ``run_dir/figures``。
        plot_config: 绘图配置；缺省从运行快照读取。

    返回：
        含 ``attention``、``attention_curve``、``trial``、``time_accuracy``、
        ``activation`` 五组汇总行的字典；同样内容会写成 CSV。汇总可含
        控制条件；显式 input_kind 时只含该条件并生成完整独立图树。
    """
    run_dir = Path(run_dir)
    if input_kind is not None and input_kind not in STANDALONE_INPUT_KINDS:
        raise ValueError(f"Unsupported input_kind: {input_kind!r}")
    if plot_config is None:
        plot_config = _default_plot_config(run_dir)
    split_names = normalize_plot_splits(splits)
    _validate_report_inputs(run_dir, split_names, plot_config, input_kind)
    figure_dir = Path(output_dir) if output_dir is not None else run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    attention_summary: list[dict[str, object]] = []
    attention_curve_summary: list[dict[str, object]] = []
    trial_summary: list[dict[str, object]] = []
    time_accuracy_summary: list[dict[str, object]] = []
    activation_summary: list[dict[str, object]] = []
    primary_input_kind = input_kind or "real"
    plot_input_kinds = {input_kind} if input_kind is not None else None
    # 只有显式关闭时才忽略 Gaussian 文件；旧快照缺少开关时仍可重绘历史基线。
    load_gaussian = not _gaussian_baseline_explicitly_disabled(plot_config)
    for split in split_names:
        if input_kind is not None:
            prefix = _standalone_export_prefix(split, input_kind)
            attention_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{prefix}attention_weights.csv"), input_kind
            )
            trial_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{prefix}trial_metrics.csv"), input_kind
            )
            time_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{prefix}time_accuracy.csv"), input_kind
            )
            time_cell_summary_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{prefix}time_cell_summary.csv"), input_kind
            )
            time_cell_metric_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{prefix}time_cell_metrics.csv"), input_kind
            )
            activation_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{prefix}time_activation.csv"), input_kind
            )
        else:
            # 给旧 CSV 补 input_kind，再按相同 schema 追加高斯行，供后续统一分组绘制。
            attention_rows = _rows_with_input_kind(_read_csv(run_dir / f"{split}_attention_weights.csv"), "real")
            trial_rows = _rows_with_input_kind(_read_csv(run_dir / f"{split}_trial_metrics.csv"), "real")
            time_rows = _rows_with_input_kind(_read_csv(run_dir / f"{split}_time_accuracy.csv"), "real")
            time_cell_summary_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{split}_time_cell_summary.csv"), "real"
            )
            time_cell_metric_rows = _rows_with_input_kind(
                _read_csv(run_dir / f"{split}_time_cell_metrics.csv"), "real"
            )
            activation_rows = _rows_with_input_kind(_read_csv(run_dir / f"{split}_time_activation.csv"), "real")
        if input_kind is None and load_gaussian:
            attention_rows.extend(
                _rows_with_input_kind(
                    _read_csv(run_dir / f"{split}_gaussian_attention_weights.csv"),
                    "gaussian",
                )
            )
            trial_rows.extend(
                _rows_with_input_kind(
                    _read_csv(run_dir / f"{split}_gaussian_trial_metrics.csv"),
                    "gaussian",
                )
            )
            time_rows.extend(
                _rows_with_input_kind(
                    _read_csv(run_dir / f"{split}_gaussian_time_accuracy.csv"),
                    "gaussian",
                )
            )
            time_cell_summary_rows.extend(
                _rows_with_input_kind(
                    _read_csv(run_dir / f"{split}_gaussian_time_cell_summary.csv"),
                    "gaussian",
                )
            )
            activation_rows.extend(
                _rows_with_input_kind(
                    _read_csv(run_dir / f"{split}_gaussian_time_activation.csv"),
                    "gaussian",
                )
            )
        # 时间打乱控制：若存在 {split}_shuffled_attention_weights.csv 则自动加载
        _shuf_attn = (
            _read_csv(run_dir / f"{split}_shuffled_attention_weights.csv")
            if input_kind is None
            else []
        )
        if _shuf_attn:
            attention_rows.extend(_rows_with_input_kind(_shuf_attn, "shuffled"))

        split_attention_summary = _plot_attention(
            split,
            attention_rows,
            figure_dir,
            plot_config,
            plot_input_kinds=plot_input_kinds,
            primary_input_kind=primary_input_kind,
        )
        attention_summary.extend(split_attention_summary)
        attention_curve_summary.extend(
            _attention_curve_diagnostics(split, split_attention_summary)
        )
        trial_summary.extend(_plot_trials(split, trial_rows, figure_dir))
        time_accuracy_summary.extend(
            _plot_time_accuracy(
                split,
                time_rows,
                figure_dir,
                plot_config,
                time_cell_summary_rows,
                plot_input_kinds=plot_input_kinds,
                primary_input_kind=primary_input_kind,
            )
        )
        activation_summary.extend(
            _plot_activation(
                split,
                activation_rows,
                figure_dir,
                plot_config,
                plot_input_kinds=plot_input_kinds,
                primary_input_kind=primary_input_kind,
            )
        )
        _plot_per_subject_time_accuracy(
            split, time_rows, figure_dir, plot_config, input_kind=primary_input_kind
        )
        _plot_attention_per_subject(
            split, attention_rows, figure_dir, plot_config, input_kind=primary_input_kind
        )
        _plot_attention_by_current_stimulus(
            split,
            attention_rows,
            trial_rows,
            figure_dir,
            plot_config,
            input_kind=primary_input_kind,
            curve_label=_attention_curve_label(attention_rows),
        )
        _plot_time_by_current_stimulus(
            split,
            time_cell_metric_rows,
            figure_dir,
            plot_config,
            input_kind=primary_input_kind,
        )

    # 即使某一类为空也写出统一命名的汇总产物，方便批量收集脚本发现运行状态。
    write_rows_csv(figure_dir / "attention_time_summary.csv", attention_summary)
    write_rows_csv(figure_dir / "attention_curve_summary.csv", attention_curve_summary)
    write_rows_csv(figure_dir / "trial_metric_summary.csv", trial_summary)
    write_rows_csv(figure_dir / "time_accuracy_summary.csv", time_accuracy_summary)
    write_rows_csv(figure_dir / "time_activation_summary.csv", activation_summary)
    return {
        "attention": attention_summary,
        "attention_curve": attention_curve_summary,
        "trial": trial_summary,
        "time_accuracy": time_accuracy_summary,
        "activation": activation_summary,
    }


def plot_target_conditions(
    run_dir: Path,
    *,
    input_kinds: Iterable[str],
    splits: Iterable[str] = ("val", "test"),
    output_root: Path | None = None,
    plot_config: Mapping[str, object] | None = None,
) -> dict[str, dict[str, list[dict[str, object]]]]:
    """Generate one complete, isolated figure tree per exported input condition."""
    run_dir = Path(run_dir)
    normalized_kinds = normalize_plot_input_kinds(input_kinds)
    split_names = normalize_plot_splits(splits)
    resolved_config = (
        _default_plot_config(run_dir)
        if plot_config is None
        else resolve_plot_config(plot_config)
    )
    figure_root = (
        Path(output_root) if output_root is not None else run_dir / "figures"
    )
    for kind in normalized_kinds:
        _validate_report_inputs(run_dir, split_names, resolved_config, kind)
    return {
        input_kind: plot_target_run(
            run_dir,
            splits=split_names,
            output_dir=figure_root / input_kind,
            plot_config=resolved_config,
            input_kind=input_kind,
        )
        for input_kind in normalized_kinds
    }


def main(argv=None) -> None:
    """命令行入口：支持历史合并、单条件和多条件批量绘图。"""
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    base_plot_config = (
        load_plot_config(args.config_snapshot.expanduser().resolve())
        if args.config_snapshot is not None
        else _default_plot_config(run_dir)
    )
    plot_config = resolve_plot_config(
        base_plot_config,
        time_axis_start_s=args.time_axis_start_s,
        time_axis_end_s=args.time_axis_end_s,
    )
    splits = args.splits if args.splits is not None else plot_config.get("splits", ("val", "test"))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else None
    )
    if args.input_kinds is not None:
        plot_target_conditions(
            run_dir,
            input_kinds=args.input_kinds,
            splits=splits,
            output_root=output_dir,
            plot_config=plot_config,
        )
    else:
        plot_target_run(
            run_dir,
            splits=splits,
            output_dir=output_dir,
            plot_config=plot_config,
            input_kind=args.input_kind,
        )
