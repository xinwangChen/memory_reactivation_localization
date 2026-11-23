"""visualization.layouts: shared Target report operations."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from memory_reactivation.training.common import write_rows_csv
from memory_reactivation.analysis.decoding import KIND_COLORS, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S, _STIMULUS_COLORS, _activation_summary, _activation_ylabel, _attention_by_current_stimulus_summary, _attention_curve_label, _attention_plot_value, _attention_pool_valid, _attention_subject_curves, _chance, _event_markers, _filter_grouped_for_plot, _find_peak_windows, _float, _gaussian_display_smooth, _input_kind, _int, _is_plottable_input_kind, _kind_color, _kind_label, _kind_sort_key, _sem, _subject_time_summary, _summary_by_kind, _time_accuracy_summary_for_plot, _trial_summary, attention_figure_filename, resolve_plot_time_axis_limits


def _draw_peak_windows(
    ax,
    windows: list[tuple[float, float, float]],
    *,
    color: str = "#1f77b4",
    label: str = "peak window (FWHM)",
) -> float | None:
    """Draw colored FWHM spans, centers, and peak height/prominence labels.

    Existing plot functions pass the same compact ``(start, end, relative_prominence)``
    records. The plotted primary curve is recovered from the axis so all legacy
    attention, performance, evidence, per-subject, and stimulus panels receive the
    same annotation without changing their aggregation logic. The return value remains
    ``None`` for compatibility with older callers that used it to reserve bottom bars.
    """
    drawable = [(float(s), float(e), float(p)) for s, e, p in windows if float(s) < float(e)]
    if not drawable:
        return None

    # Pick the thickest monotone curve: in the existing figures this is the primary
    # real/current-condition or group-mean line, while control and subject traces are thinner.
    candidates = []
    for line in ax.lines:
        try:
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
        except (TypeError, ValueError):
            continue
        if x.ndim != 1 or y.ndim != 1 or len(x) < 3 or len(x) != len(y):
            continue
        finite = np.isfinite(x) & np.isfinite(y)
        if finite.sum() < 3:
            continue
        x = x[finite]
        y = y[finite]
        if np.any(np.diff(x) < 0.0):
            continue
        candidates.append((float(line.get_linewidth()), len(x), x, y))
    if candidates:
        _, _, curve_times, curve_values = max(candidates, key=lambda item: (item[0], item[1]))
    else:
        curve_times = np.array([s for s, _, _ in drawable], dtype=float)
        curve_values = np.zeros_like(curve_times)

    curve_range = float(np.nanmax(curve_values) - np.nanmin(curve_values)) if len(curve_values) else 0.0
    if not np.isfinite(curve_range) or curve_range <= 0.0:
        curve_range = 1.0
    y_min, y_max = ax.get_ylim()
    y_span = y_max - y_min if y_max > y_min else 1.0
    # Reserve room for the two-line labels while keeping the data baseline unchanged.
    ax.set_ylim(y_min, y_max + 0.24 * y_span)

    peak_colors = ("#D55E00", "#009E73", "#A21CAF", "#0072B2", "#CC79A7", "#E69F00")
    for peak_number, (start_s, end_s, relative_prominence) in enumerate(drawable, start=1):
        mask = (curve_times >= start_s) & (curve_times <= end_s)
        if np.any(mask):
            local_indices = np.flatnonzero(mask)
            peak_index = local_indices[int(np.nanargmax(curve_values[local_indices]))]
            center_s = float(curve_times[peak_index])
            height = float(curve_values[peak_index])
        else:
            center_s = 0.5 * (start_s + end_s)
            height = float(np.interp(center_s, curve_times, curve_values)) if len(curve_times) else 0.0
        prominence = float(relative_prominence) * curve_range
        peak_color = peak_colors[(peak_number - 1) % len(peak_colors)]
        ax.axvspan(
            start_s,
            end_s,
            color=peak_color,
            alpha=0.14,
            linewidth=0,
            label=label if peak_number == 1 else None,
            zorder=0,
        )
        ax.axvline(center_s, color=peak_color, linewidth=1.25, alpha=0.9, zorder=3)
        ax.scatter(
            [center_s],
            [height],
            s=28,
            color=peak_color,
            edgecolor="white",
            linewidth=0.8,
            zorder=7,
        )
        ax.annotate(
            f"P{peak_number} {center_s:.3f} s\n"
            f"height={height:.3f}, prom={prominence:.3f}\n"
            f"FWHM {start_s:.3f}-{end_s:.3f} s",
            xy=(center_s, height),
            xytext=(0, 18 + 24 * ((peak_number - 1) % 2)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=6.8,
            color=peak_color,
            arrowprops={"arrowstyle": "-", "color": peak_color, "lw": 0.65},
            annotation_clip=False,
            zorder=8,
        )
    return None


def _annotate_bars(
    ax,
    bars,
    values: Sequence[float],
    *,
    errors: Sequence[float] | None = None,
    fmt: str = "{:.3f}",
) -> None:
    """在柱顶（或负柱底）标注数值，并把误差棒高度计入文本位置。

    参数：
        ax: Matplotlib 坐标轴。
        bars: ``ax.bar`` 返回的柱对象序列。
        values: 每根柱的中心值。
        errors: 可选误差棒；缺省按零处理。
        fmt: 数值格式模板。
    """
    y_min, y_max = ax.get_ylim()
    span = max(abs(y_max - y_min), 1e-9)
    text_offset = 0.018 * span
    for index, (bar, value) in enumerate(zip(bars, values)):
        error = 0.0 if errors is None else float(errors[index])
        x = bar.get_x() + bar.get_width() / 2
        if value >= 0.0:
            y = value + error + text_offset
            va = "bottom"
        else:
            y = value - error - text_offset
            va = "top"
        ax.text(
            x,
            y,
            fmt.format(value),
            ha="center",
            va=va,
            fontsize=8,
            color="#333333",
        )


def _pad_ylim_for_bar_labels(
    ax,
    values: Sequence[float],
    errors: Sequence[float],
    *,
    min_zero: bool = False,
    min_span: float = 1e-3,
) -> None:
    """扩展 y 轴范围，为柱状图数值标签预留空间。

    ``min_zero`` 要求范围至少覆盖零，``min_span`` 防止数值几乎相同时坐标轴塌缩。
    """
    if not values:
        return
    upper = max(value + error for value, error in zip(values, errors))
    lower = min(value - error for value, error in zip(values, errors))
    if min_zero:
        lower = min(0.0, lower)
        upper = max(0.0, upper)
    span = max(upper - lower, 1e-9)
    if span < min_span:
        center = (upper + lower) / 2
        lower = center - min_span / 2
        upper = center + min_span / 2
        span = min_span
    pad = 0.16 * span
    ax.set_ylim(lower - pad, upper + pad)


def _add_event_markers(ax, plot_config: Mapping[str, object] | None = None) -> None:
    """Apply the configured range, then draw events without expanding that range."""
    configured_limits = resolve_plot_time_axis_limits(plot_config)
    if configured_limits is not None:
        ax.set_xlim(*configured_limits)
    # 调试时间窗可能裁掉部分事件；跳过窗外事件，避免 axvline 反向拉宽数据时间轴。
    x_min, x_max = ax.get_xlim()
    x_span = max(float(x_max - x_min), 1e-12)
    for marker in _event_markers(plot_config):
        time_s = float(marker["time_s"])
        if time_s < x_min or time_s > x_max:
            continue
        label = str(marker["label"])
        color = str(marker["color"])
        alpha = float(marker.get("alpha", 0.8))
        ax.axvline(time_s, color=color, linewidth=1.0, linestyle="--", alpha=alpha)
        ax.text(
            time_s,
            0.98,
            label,
            transform=ax.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha=("left" if time_s <= x_min + x_span * 1e-9 else "right"),
            color=color,
            fontsize=8,
            alpha=min(1.0, alpha + 0.1),
        )
    ax.set_xlim(x_min, x_max)


def _plot_attention_score_time(
    split: str,
    summary: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    plot_input_kinds: Collection[str] | None = None,
) -> None:
    """单独画 scorer 的 softmax 输出；不组合 gate，不再归一化。"""
    score_rows = [
        row for row in summary
        if row.get("mean_raw_attention") not in {None, ""}
    ]
    grouped = _filter_grouped_for_plot(_summary_by_kind(score_rows), plot_input_kinds)
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for input_kind, kind_rows in grouped.items():
        times = np.array([float(row["time_s"]) for row in kind_rows])
        means = np.array([float(row["mean_raw_attention"]) for row in kind_rows])
        sems = np.array([float(row["sem_raw_attention"]) for row in kind_rows])
        color = _kind_color(input_kind)
        ax.plot(times, means, color=color, linewidth=2.0,
                label=f"{_kind_label(input_kind)} softmax(score)")
        ax.fill_between(times, means - sems, means + sems,
                        color=color, alpha=0.16, linewidth=0)
    _add_event_markers(ax, plot_config)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel("Score branch weight: softmax(score), before gate")
    ax.set_title(f"{split} score branch over time (before gate)")
    ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / attention_figure_filename("attention_score", split), dpi=180)
    plt.close(fig)


def _plot_attention_mass_time(
    split: str,
    summary: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    plot_input_kinds: Collection[str] | None = None,
) -> None:
    """画未归一化贡献；signed 模式同时展示 w 与 |w|，不隐藏负贡献。"""
    mass_time_rows = [
        row for row in summary
        if row.get("mean_effective_attention") not in {None, ""}
    ]
    grouped = _filter_grouped_for_plot(_summary_by_kind(mass_time_rows), plot_input_kinds)
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    has_signed_weights = False
    for input_kind, kind_rows in grouped.items():
        times = np.array([float(row["time_s"]) for row in kind_rows])
        means = np.array([float(row["mean_effective_attention"]) for row in kind_rows])
        sems = np.array([float(row["sem_effective_attention"]) for row in kind_rows])
        color = _kind_color(input_kind)
        ax.plot(times, means, color=color, linewidth=2.0,
                label=f"{_kind_label(input_kind)} mean(w)")
        ax.fill_between(times, means - sems, means + sems,
                        color=color, alpha=0.16, linewidth=0)
        absolute_rows = [row for row in kind_rows
                         if row.get("mean_absolute_effective_attention") not in {None, ""}]
        if absolute_rows:
            has_signed_weights = True
            ax.plot(
                [float(row["time_s"]) for row in absolute_rows],
                [float(row["mean_absolute_effective_attention"]) for row in absolute_rows],
                color=color, linewidth=1.5, linestyle="--",
                label=f"{_kind_label(input_kind)} mean(|w|)",
            )
    _add_event_markers(ax, plot_config)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel("Effective attention weight")
    ax.set_title("Effective attention")
    if has_signed_weights:
        ax.axhline(0.0, color="#999999", linestyle=":", linewidth=1.0)
    else:
        ax.set_ylim(bottom=0.0)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / attention_figure_filename("attention_mass_time", split), dpi=180)
    plt.close(fig)


def _plot_attention(
    split: str,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    plot_input_kinds: Collection[str] | None = None,
    primary_input_kind: str = "real",
) -> list[dict[str, object]]:
    """绘制一个 split 的注意力图组并返回时间汇总行。

    分别生成 softmax(score)、激活后的 gate、未归一化组合权重图，以及归一化
    权重、相对均匀基线、个体曲线和整段 effective mass 柱状图；
    没有注意力行（例如 time-only 模型）时直接返回空列表。
    attention_time 图在底部标注组均值曲线的 FWHM 峰区间。
    """
    summary = _subject_time_summary(rows)
    if not summary:
        return []
    curve_label = _attention_curve_label(rows)
    grouped = _summary_by_kind(summary)          # real + gaussian 保留在汇总
    grouped_plot_only = _filter_grouped_for_plot(grouped, plot_input_kinds)
    if not grouped_plot_only:
        return [{"split": split, **row} for row in summary]

    # 重激活时间曲线只绘制 real；Gaussian 仍保留在返回的汇总 CSV 和柱状图中。
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    att_means_for_fwhm: np.ndarray | None = None
    att_times_for_fwhm: np.ndarray | None = None
    uniform_plotted = False
    for input_kind, kind_rows in grouped_plot_only.items():
        times = [float(row["time_s"]) for row in kind_rows]
        means = [float(row["mean_attention"]) for row in kind_rows]
        sems = [float(row["sem_attention"]) for row in kind_rows]
        color = _kind_color(input_kind)
        ax.plot(times, means, color=color,
                linewidth=2.0, linestyle="-", alpha=1.0,
                label=f"{_kind_label(input_kind)} attention")
        ax.fill_between(
            times,
            [value - error for value, error in zip(means, sems)],
            [value + error for value, error in zip(means, sems)],
            color=color, alpha=0.16, linewidth=0,
        )
        if input_kind == primary_input_kind:
            att_means_for_fwhm = np.array(means)
            att_times_for_fwhm = np.array(times)
        if not uniform_plotted:
            uniform = [float(row["uniform_attention"]) for row in kind_rows]
            ax.plot(times, uniform, color="#999999", linestyle="--", linewidth=1.2, label="uniform")
            uniform_plotted = True
    _add_event_markers(ax, plot_config)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel(curve_label)
    ax.set_title(f"{split} combined attention strength over time")
    if att_means_for_fwhm is not None and att_times_for_fwhm is not None:
        pws = _find_peak_windows(att_means_for_fwhm, att_times_for_fwhm)
        _bar_ylim = _draw_peak_windows(ax, pws, color=_kind_color(primary_input_kind))
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / attention_figure_filename("attention_time", split), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for input_kind, kind_rows in grouped_plot_only.items():
        times = [float(row["time_s"]) for row in kind_rows]
        relative = [float(row["relative_attention"]) for row in kind_rows]
        sems = [float(row["sem_attention"]) for row in kind_rows]
        color = _kind_color(input_kind)
        ax.plot(times, relative, color=color, linewidth=2.0,
                linestyle="-", alpha=1.0, label=_kind_label(input_kind))
        ax.fill_between(
            times,
            [value - error for value, error in zip(relative, sems)],
            [value + error for value, error in zip(relative, sems)],
            color=color, alpha=0.16, linewidth=0,
        )
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.2)
    _add_event_markers(ax, plot_config)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel(f"{curve_label} minus uniform")
    ax.set_title(f"{split} combined attention strength relative to uniform")
    # FWHM 用实际绘图的 relative_attention 曲线，不用 mean_attention
    primary_rel_rows = grouped.get(primary_input_kind, [])
    if primary_rel_rows:
        _rel_t = np.array([float(r["time_s"]) for r in primary_rel_rows])
        _rel_v = np.array([float(r["relative_attention"]) for r in primary_rel_rows])
        _bar_ylim = _draw_peak_windows(
            ax,
            _find_peak_windows(_rel_v, _rel_t),
            color=_kind_color(primary_input_kind),
        )
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / attention_figure_filename("attention_relative", split), dpi=180)
    plt.close(fig)

    # 个体灰线只画真实 EEG；图中组均值同样仅 real。
    curves = _attention_subject_curves(rows, input_kind=primary_input_kind)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for curve in curves.values():
        if curve:
            ax.plot(
                [point[0] for point in curve],
                [point[1] for point in curve],
                color="#888888",
                alpha=0.20,
                linewidth=0.8,
            )
    for input_kind, kind_rows in grouped_plot_only.items():
        times = [float(row["time_s"]) for row in kind_rows]
        means = [float(row["mean_attention"]) for row in kind_rows]
        ax.plot(times, means, color=_kind_color(input_kind),
                linewidth=2.2, linestyle="-", alpha=1.0,
                label=_kind_label(input_kind))
    first_rows = grouped_plot_only.get(primary_input_kind) or next(iter(grouped.values()))
    ax.plot(
        [float(row["time_s"]) for row in first_rows],
        [float(row["uniform_attention"]) for row in first_rows],
        color="#999999",
        linestyle="--",
        linewidth=1.2,
        label="uniform",
    )
    _add_event_markers(ax, plot_config)
    if att_means_for_fwhm is not None and att_times_for_fwhm is not None:
        _bar_ylim = _draw_peak_windows(
            ax,
            _find_peak_windows(att_means_for_fwhm, att_times_for_fwhm),
            color=_kind_color(primary_input_kind),
        )
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel(curve_label)
    ax.set_title(f"{split} subject combined attention strength")
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / attention_figure_filename("attention_subject_curves", split), dpi=180)
    plt.close(fig)

    # Scorer softmax is an independent branch: do not derive it from effective
    # weights, since gates may be zero, negative, or combined by addition.
    _plot_attention_score_time(split, summary, output_dir, plot_config,
                               plot_input_kinds=plot_input_kinds)

    # Total mass is a scalar; its temporal contributions are the original
    # effective weights, not the weights renormalized to sum to one. These
    # means/SEMs already average trials within each subject, then subjects.
    _plot_attention_mass_time(split, summary, output_dir, plot_config,
                              plot_input_kinds=plot_input_kinds)

    gate_rows = [
        row
        for row in summary
        if row.get("mean_attention_gate") not in {None, ""}
    ]
    if gate_rows:
        gate_grouped = _filter_grouped_for_plot(
            _summary_by_kind(gate_rows), plot_input_kinds
        )
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for input_kind, kind_rows in gate_grouped.items():
            ax.plot(
                [float(row["time_s"]) for row in kind_rows],
                [float(row["mean_attention_gate"]) for row in kind_rows],
                linestyle="-", alpha=1.0,
                color=_kind_color(input_kind),
                linewidth=2.0,
                label=f"{_kind_label(input_kind)} gate (after activation)",
            )
        _add_event_markers(ax, plot_config)
        ax.set_xlabel("Time from white-noise onset (s)")
        ax.set_ylabel("Gate branch output (after activation)")
        ax.set_title(f"{split} gate branch over time (after activation)")
        ax.legend(frameon=False)
        fig.tight_layout()
        if locals().get("_bar_ylim") is not None and all(
            float(row["mean_attention_gate"]) >= 0.0 for row in gate_rows
        ):
            ax.set_ylim(bottom=_bar_ylim)
        fig.savefig(output_dir / attention_figure_filename("attention_gate", split), dpi=180)
        plt.close(fig)

    # schema v2 直接提供每 trial 的有效注意力质量；历史 CSV 则由逐时间权重求和重建。
    trial_masses: dict[tuple[str, int, int, int, int], float] = {}
    fallback_masses: dict[tuple[str, int, int, int, int], float] = defaultdict(float)
    for row in rows:
        key = (
            _input_kind(row),
            _int(row, "subject_id"),
            _int(row, "record_id"),
            _int(row, "trial0"),
            _int(row, "lag"),
        )
        mass = row.get("effective_attention_mass")
        if mass not in {None, ""}:
            trial_masses[key] = float(mass)
        else:
            fallback_masses[key] += _float(
                row,
                "effective_attention_weight",
                _float(row, "attention_weight"),
            )
    for key, mass in fallback_masses.items():
        trial_masses.setdefault(key, mass)
    subject_masses: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (input_kind, subject_id, _record_id, _trial0, _lag), mass in trial_masses.items():
        subject_masses[(input_kind, subject_id)].append(mass)
    # attention_mass 柱状图与 trial 柱状图一致：real 与 gaussian 噪声基线并排绘制。
    kind_masses: dict[str, list[float]] = defaultdict(list)
    for (input_kind, _subject_id), masses in subject_masses.items():
        kind_masses[input_kind].append(mean(masses))
    if kind_masses:
        kinds = sorted(kind_masses, key=_kind_sort_key)
        values = [mean(kind_masses[kind]) for kind in kinds]
        errors = [_sem(kind_masses[kind]) for kind in kinds]
        fig, ax = plt.subplots(figsize=(max(5.0, 1.5 * len(kinds)), 4.5))
        bars = ax.bar(
            range(len(kinds)),
            values,
            yerr=errors,
            color=[_kind_color(kind) for kind in kinds],
            capsize=4,
        )
        ax.set_xticks(range(len(kinds)), [_kind_label(kind) for kind in kinds])
        absolute_mass = any(row.get("absolute_effective_attention_weight") not in {None, ""} for row in rows)
        ax.set_ylabel("Total absolute combined weight" if absolute_mass else "Total combined weight")
        ax.set_title(f"{split} " + ("absolute combined attention mass" if absolute_mass else "combined attention mass"))
        _pad_ylim_for_bar_labels(ax, values, errors, min_zero=True)
        _annotate_bars(ax, bars, values, errors=errors, fmt="{:.3f}")
        fig.tight_layout()
        if locals().get("_bar_ylim") is not None:
            ax.set_ylim(bottom=_bar_ylim)
        fig.savefig(output_dir / attention_figure_filename("attention_mass", split), dpi=180)
        plt.close(fig)

    return [{"split": split, **row} for row in summary]


def _plot_trials(split: str, rows: Sequence[Mapping[str, str]], output_dir: Path) -> list[dict[str, object]]:
    """绘制 trial-level 准确率和 margin 柱状图，并返回汇总行。

    准确率图包含机会水平；margin 图并列展示 evidence 与 probability 两种定义。
    汇总与 PNG 均含 real/gaussian：噪声基线在柱状图中保留，便于对照 trial 级证据
    与准确率；时间曲线类图不画 gaussian（见 :data:`PLOT_INPUT_KINDS`）。同类柱状
    图还有 ``attention_mass_*``。
    """
    summary = _trial_summary(split, rows)
    if not summary:
        return []
    # trial 柱状图刻意不走 _filter_rows_for_plot：gaussian 噪声证据/准确率要画出来。
    plot_summary = list(summary)
    # 仅一种 input_kind 时标签只需分支名，避免多余的 "real EEG" 后缀。
    labels = [
        str(row["branch"])
        if len({str(item["input_kind"]) for item in plot_summary}) == 1
        else f"{row['branch']}\n{_kind_label(str(row['input_kind']))}"
        for row in plot_summary
    ]
    accuracy = [float(row["accuracy_mean"]) for row in plot_summary]
    accuracy_sem = [float(row["accuracy_sem"]) for row in plot_summary]
    colors = [_kind_color(str(row["input_kind"])) for row in plot_summary]
    chance = float(plot_summary[0]["chance_accuracy"])

    fig, ax = plt.subplots(figsize=(max(5.8, 1.3 * len(labels)), 4.5))
    accuracy_bars = ax.bar(labels, accuracy, yerr=accuracy_sem, color=colors, capsize=4)
    ax.axhline(chance, color="#999999", linestyle="--", linewidth=1.2, label=f"chance {chance:.3f}")
    ax.set_ylim(0.0, max(1.0, max(value + error for value, error in zip(accuracy, accuracy_sem)) + 0.05))
    _annotate_bars(ax, accuracy_bars, accuracy, errors=accuracy_sem, fmt="{:.3f}")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{split} trial accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / f"trial_accuracy_{split}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(6.8, 1.3 * len(labels)), 4.5))
    x = list(range(len(labels)))
    width = 0.34
    evidence = [float(row["evidence_margin_mean"]) for row in plot_summary]
    evidence_sem = [float(row["evidence_margin_sem"]) for row in plot_summary]
    probability = [float(row["probability_margin_mean"]) for row in plot_summary]
    probability_sem = [float(row["probability_margin_sem"]) for row in plot_summary]
    evidence_bars = ax.bar(
        [value - width / 2 for value in x],
        evidence,
        width=width,
        yerr=evidence_sem,
        color="#2ca02c",
        capsize=4,
        label="evidence",
    )
    probability_bars = ax.bar(
        [value + width / 2 for value in x],
        probability,
        width=width,
        yerr=probability_sem,
        color="#9467bd",
        capsize=4,
        label="probability",
    )
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xticks(x, labels)
    margin_values = evidence + probability
    margin_errors = evidence_sem + probability_sem
    _pad_ylim_for_bar_labels(ax, margin_values, margin_errors, min_zero=True)
    _annotate_bars(ax, evidence_bars, evidence, errors=evidence_sem, fmt="{:.4f}")
    _annotate_bars(ax, probability_bars, probability, errors=probability_sem, fmt="{:.4f}")
    ax.set_ylabel("Target margin")
    ax.set_title(f"{split} trial margins")
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / f"trial_margins_{split}.png", dpi=180)
    plt.close(fig)

    return summary


def _save_time_accuracy_figure(
    split: str,
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    chance: float,
    output_dir: Path,
    plot_config: Mapping[str, object] | None,
    *,
    smoothing_window_s: float | None,
    plot_input_kinds: Collection[str] | None = None,
    primary_input_kind: str = "real",
) -> None:
    """保存一张原始或显示平滑的逐时间准确率图（仅 real）。

    自动在图底部标注组均值曲线的 FWHM 峰区间（半峰全宽）。
    """
    plot_grouped = _filter_grouped_for_plot(grouped, plot_input_kinds)
    if not plot_grouped:
        return
    smoothed = smoothing_window_s is not None
    display_metrics = {
        str(row.get("display_metric", "raw_accuracy"))
        for kind_rows in plot_grouped.values()
        for row in kind_rows
    }
    if len(display_metrics) != 1:
        raise ValueError("time accuracy plot cannot mix display metrics")
    conditional = next(iter(display_metrics)) == "subject_cell_balanced_accuracy"
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for input_kind, kind_rows in plot_grouped.items():
        times = [float(row["time_s"]) for row in kind_rows]
        accuracy = [float(row["accuracy_mean"]) for row in kind_rows]
        sems = [float(row["accuracy_sem"]) for row in kind_rows]
        display_accuracy = (
            _gaussian_display_smooth(accuracy, times, smoothing_window_s)
            if smoothed
            else np.asarray(accuracy, dtype=float)
        )
        color = _kind_color(input_kind)
        ax.plot(
            times,
            display_accuracy,
            color=color,
            linewidth=2.0,
            label=_kind_label(input_kind),
        )
        ax.fill_between(
            times,
            [value - error for value, error in zip(display_accuracy, sems)],
            [value + error for value, error in zip(display_accuracy, sems)],
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    # Gaussian 基线曲线（灰虚线，供目视对比且用于峰过滤）
    _gauss_rows = list(grouped.get("gaussian", []))
    _gauss_times_arr: np.ndarray | None = None
    _gauss_acc_disp: np.ndarray | None = None
    if _gauss_rows and "gaussian" not in plot_grouped:
        _gauss_times = [float(r["time_s"]) for r in _gauss_rows]
        _gauss_acc = [float(r["accuracy_mean"]) for r in _gauss_rows]
        _gauss_acc_disp = (
            _gaussian_display_smooth(_gauss_acc, _gauss_times, smoothing_window_s)
            if smoothed else np.asarray(_gauss_acc, dtype=float)
        )
        _gauss_times_arr = np.asarray(_gauss_times, dtype=float)
        ax.plot(
            _gauss_times_arr,
            _gauss_acc_disp,
            color="#888888",
            linewidth=1.4,
            linestyle="--",
            label="Gaussian EEG",
            zorder=1,
        )
    ax.axhline(
        chance,
        color="#999999",
        linestyle="--",
        linewidth=1.2,
        label=f"chance {chance:.3f}",
    )
    _add_event_markers(ax, plot_config)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel("Subject×cell balanced accuracy" if conditional else "Accuracy")
    metric_title = (
        "conditional time-wise balanced accuracy"
        if conditional
        else "time-wise accuracy"
    )
    if smoothed:
        window_ms = int(round(float(smoothing_window_s) * 1000.0))
        ax.set_title(f"{split} {metric_title} ({window_ms} ms Gaussian-smoothed)")
        filename = f"time_accuracy_smoothed_{split}.png"
    else:
        ax.set_title(f"{split} {metric_title}")
        filename = f"time_accuracy_{split}.png"
    # FWHM 峰区间：baseline 优先用 Gaussian 时间曲线，回退到 chance 标量
    primary_rows = plot_grouped.get(primary_input_kind, [])
    if primary_rows:
        _times = np.array([float(r["time_s"]) for r in primary_rows])
        _acc = np.array([float(r["accuracy_mean"]) for r in primary_rows])
        _disp = (
            _gaussian_display_smooth(_acc, _times, smoothing_window_s)
            if smoothed else _acc
        )
        _peak_baseline = (
            _gauss_acc_disp
            if (_gauss_acc_disp is not None and len(_gauss_acc_disp) == len(_times))
            else chance
        )
        pws = _find_peak_windows(_disp, _times, baseline=_peak_baseline, height_reference=chance,
                                  merge_gap_s=0.10, min_window_width_s=0.15)
        _bar_ylim = _draw_peak_windows(ax, pws, color=_kind_color(primary_input_kind))
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def _save_time_evidence_figure(
    split: str,
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None,
    *,
    smoothing_window_s: float | None,
    metric_key: str = "evidence_margin_mean",
    sem_key: str = "evidence_margin_sem",
    ylabel: str = "True-class evidence margin",
    title_metric: str = "time-wise true-class evidence",
    filename_stem: str = "time_evidence",
    plot_input_kinds: Collection[str] | None = None,
    primary_input_kind: str = "real",
) -> None:
    """保存逐时间点真实类证据强度曲线（仅 real；raw 或显示平滑）。

    ``metric_key`` 缺省为 logit evidence margin；SEM 不平滑，与 accuracy 图一致。
    中性参考线为 0（真实类相对其他类无偏好）。自动在图底部标注 FWHM 峰区间。
    """
    plot_grouped = _filter_grouped_for_plot(grouped, plot_input_kinds)
    usable: dict[str, list[Mapping[str, object]]] = {}
    for input_kind, kind_rows in plot_grouped.items():
        kept = [
            row
            for row in kind_rows
            if row.get(metric_key) not in {None, ""}
            and row.get(sem_key) not in {None, ""}
        ]
        if kept:
            usable[input_kind] = kept
    if not usable:
        return
    smoothed = smoothing_window_s is not None
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for input_kind, kind_rows in usable.items():
        times = [float(row["time_s"]) for row in kind_rows]
        values = [float(row[metric_key]) for row in kind_rows]
        sems = [float(row[sem_key]) for row in kind_rows]
        display_values = (
            _gaussian_display_smooth(values, times, smoothing_window_s)
            if smoothed
            else np.asarray(values, dtype=float)
        )
        color = _kind_color(input_kind)
        ax.plot(
            times,
            display_values,
            color=color,
            linewidth=2.0,
            label=_kind_label(input_kind),
        )
        ax.fill_between(
            times,
            [value - error for value, error in zip(display_values, sems)],
            [value + error for value, error in zip(display_values, sems)],
            color=color,
            alpha=0.16,
            linewidth=0,
        )

    # Gaussian 基线曲线（灰虚线）
    _gauss_ev_rows = [
        r for r in grouped.get("gaussian", [])
        if r.get(metric_key) not in {None, ""}
    ]
    _gauss_ev_times_arr: np.ndarray | None = None
    _gauss_ev_disp: np.ndarray | None = None
    if _gauss_ev_rows and "gaussian" not in plot_grouped:
        _gauss_ev_times = [float(r["time_s"]) for r in _gauss_ev_rows]
        _gauss_ev_vals = [float(r[metric_key]) for r in _gauss_ev_rows]
        _gauss_ev_disp = (
            _gaussian_display_smooth(_gauss_ev_vals, _gauss_ev_times, smoothing_window_s)
            if smoothed else np.asarray(_gauss_ev_vals, dtype=float)
        )
        _gauss_ev_times_arr = np.asarray(_gauss_ev_times, dtype=float)
        ax.plot(
            _gauss_ev_times_arr,
            _gauss_ev_disp,
            color="#888888",
            linewidth=1.4,
            linestyle="--",
            label="Gaussian EEG",
            zorder=1,
        )
    _add_event_markers(ax, plot_config)
    ax.axhline(0.0, color="#999999", linestyle=":", linewidth=1.0, label="neutral 0")
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel(ylabel)
    if smoothed:
        window_ms = int(round(float(smoothing_window_s) * 1000.0))
        ax.set_title(f"{split} {title_metric} ({window_ms} ms Gaussian-smoothed)")
        filename = f"{filename_stem}_smoothed_{split}.png"
    else:
        ax.set_title(f"{split} {title_metric}")
        filename = f"{filename_stem}_{split}.png"
    # FWHM 峰区间：baseline 优先用 Gaussian 曲线，回退到 0
    primary_rows = usable.get(primary_input_kind, [])
    if primary_rows:
        _times = np.array([float(r["time_s"]) for r in primary_rows])
        _vals = np.array([float(r[metric_key]) for r in primary_rows])
        _disp = (
            _gaussian_display_smooth(_vals, _times, smoothing_window_s)
            if smoothed else _vals
        )
        _peak_baseline_ev = (
            _gauss_ev_disp
            if (_gauss_ev_disp is not None and len(_gauss_ev_disp) == len(_times))
            else 0.0
        )
        pws = _find_peak_windows(_disp, _times, baseline=_peak_baseline_ev, height_reference=0.0,
                                  merge_gap_s=0.10, min_window_width_s=0.15)
        _bar_ylim = _draw_peak_windows(ax, pws, color=_kind_color(primary_input_kind))
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / filename, dpi=180)
    plt.close(fig)


def _plot_time_accuracy(
    split: str,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    cell_summary_rows: Sequence[Mapping[str, str]] = (),
    *,
    plot_input_kinds: Collection[str] | None = None,
    primary_input_kind: str = "real",
) -> list[dict[str, object]]:
    """绘制逐时间点准确率与证据强度曲线，并返回汇总行。

    汇总含 real/gaussian；曲线 PNG 只画 real（噪声不叠到时间轴上）。准确率与
    evidence 各输出原始图和 150 ms 显示平滑图；SEM 与汇总 CSV 不平滑。
    每张图在图底部自动标注组均值曲线的 FWHM 峰区间横线。
    """
    summary = _time_accuracy_summary_for_plot(split, rows, cell_summary_rows)
    if not summary:
        return []
    grouped = _summary_by_kind(summary)
    chance = float(summary[0]["chance_accuracy"])

    _save_time_accuracy_figure(
        split, grouped, chance, output_dir, plot_config,
        smoothing_window_s=None,
        plot_input_kinds=plot_input_kinds,
        primary_input_kind=primary_input_kind,
    )
    _save_time_accuracy_figure(
        split, grouped, chance, output_dir, plot_config,
        smoothing_window_s=TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S,
        plot_input_kinds=plot_input_kinds,
        primary_input_kind=primary_input_kind,
    )
    # 证据强度：logit evidence margin（与 trial-level evidence 定义一致）。
    _save_time_evidence_figure(
        split, grouped, output_dir, plot_config,
        smoothing_window_s=None,
        plot_input_kinds=plot_input_kinds,
        primary_input_kind=primary_input_kind,
    )
    _save_time_evidence_figure(
        split, grouped, output_dir, plot_config,
        smoothing_window_s=TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S,
        plot_input_kinds=plot_input_kinds,
        primary_input_kind=primary_input_kind,
    )
    return summary


def _plot_activation(
    split: str,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    plot_input_kinds: Collection[str] | None = None,
    primary_input_kind: str = "real",
) -> list[dict[str, object]]:
    """绘制逐时间激活分数与激活率曲线，并返回汇总行。

    可由 ``plots.activation.enabled`` 关闭；分数图显示配置阈值，激活率图固定在 0 到 1。
    """
    activation_config: Mapping[str, object] = {}
    if isinstance(plot_config, Mapping):
        configured = plot_config.get("activation", {})
        if configured is None:
            configured = {}
        if isinstance(configured, Mapping):
            activation_config = configured
    if not bool(activation_config.get("enabled", True)):
        return []
    summary = _activation_summary(split, rows)
    if not summary:
        return []
    grouped = _filter_grouped_for_plot(_summary_by_kind(summary), plot_input_kinds)
    if not grouped:
        return summary
    threshold = float(next(iter(grouped.values()))[0]["activation_threshold"])
    # 提前计算 real 曲线行，score 和 rate 两张图共用，避免 rate 图块依赖 score 图块的局部变量
    primary_score_rows = [
        row
        for row in summary
        if str(row.get("input_kind", "real")) == primary_input_kind
        and _is_plottable_input_kind(primary_input_kind, plot_input_kinds)
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for input_kind, kind_rows in grouped.items():
        times = [float(row["time_s"]) for row in kind_rows]
        scores = [float(row["activation_score_mean"]) for row in kind_rows]
        sems = [float(row["activation_score_sem"]) for row in kind_rows]
        color = _kind_color(input_kind)
        ax.plot(times, scores, color=color, linewidth=2.0, label=_kind_label(input_kind))
        ax.fill_between(
            times,
            [value - error for value, error in zip(scores, sems)],
            [value + error for value, error in zip(scores, sems)],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    ax.axhline(threshold, color="#999999", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.2f}")
    _add_event_markers(ax, plot_config)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel(_activation_ylabel(rows))
    ax.set_title(f"{split} time activation score")
    # FWHM 峰区间（primary_score_rows 已在函数顶部定义）
    if primary_score_rows:
        _ts = np.array([float(r["time_s"]) for r in primary_score_rows])
        _vs = np.array([float(r["activation_score_mean"]) for r in primary_score_rows])
        _bar_ylim = _draw_peak_windows(
            ax,
            _find_peak_windows(_vs, _ts),
            color=_kind_color(primary_input_kind),
        )
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / f"time_activation_score_{split}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for input_kind, kind_rows in grouped.items():
        times = [float(row["time_s"]) for row in kind_rows]
        rates = [float(row["activation_rate_mean"]) for row in kind_rows]
        sems = [float(row["activation_rate_sem"]) for row in kind_rows]
        color = _kind_color(input_kind)
        ax.plot(times, rates, color=color, linewidth=2.0, label=_kind_label(input_kind))
        ax.fill_between(
            times,
            [value - error for value, error in zip(rates, sems)],
            [value + error for value, error in zip(rates, sems)],
            color=color,
            alpha=0.16,
            linewidth=0,
        )
    _add_event_markers(ax, plot_config)
    # FWHM 峰区间
    if primary_score_rows:
        _ts_r = np.array([float(r["time_s"]) for r in primary_score_rows])
        _vr = np.array([float(r["activation_rate_mean"]) for r in primary_score_rows])
        _bar_ylim = _draw_peak_windows(
            ax,
            _find_peak_windows(_vr, _ts_r),
            color=_kind_color(primary_input_kind),
        )
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Time from white-noise onset (s)")
    ax.set_ylabel("Activation rate")
    ax.set_title(f"{split} time activation rate")
    ax.legend(frameon=False)
    fig.tight_layout()
    if locals().get("_bar_ylim") is not None:
        ax.set_ylim(bottom=_bar_ylim)
    fig.savefig(output_dir / f"time_activation_rate_{split}.png", dpi=180)
    plt.close(fig)
    return summary


def _plot_per_subject_time_accuracy(
    split: str,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    input_kind: str = "real",
) -> None:
    """画每个被试独立的逐时点准确率与证据曲线（灰细线）加粗组均值。

    输出两张图（原始 + 150 ms 平滑），文件名：
    ``time_accuracy_per_subject_{split}.png``
    ``time_evidence_per_subject_{split}.png``
    """
    kind_rows = [r for r in rows if str(r.get("input_kind", "real")) == input_kind]
    if not kind_rows:
        return
    # 按 subject_id → 按 time_index 收集
    subj_acc: dict[str, dict[int, tuple[float, float]]] = {}  # subj → ti → (time_s, accuracy)
    subj_ev: dict[str, dict[int, tuple[float, float]]] = {}   # subj → ti → (time_s, evidence)
    for r in kind_rows:
        subj = str(r["subject_id"])
        ti = int(float(r["time_index"]))
        ts = float(r["time_s"])
        try:
            acc = float(r["accuracy"])
            subj_acc.setdefault(subj, {})[ti] = (ts, acc)
        except (ValueError, TypeError, KeyError):
            pass
        try:
            ev_str = r.get("evidence_margin_mean", "")
            if ev_str not in ("", None):
                ev = float(ev_str)
                subj_ev.setdefault(subj, {})[ti] = (ts, ev)
        except (ValueError, TypeError):
            pass

    if not subj_acc:
        return

    subjects = sorted(subj_acc)
    # Bug fix: 从所有行建全局时间轴，避免依赖特定被试的 key 顺序
    global_ti_map: dict[int, float] = {}
    for r in kind_rows:
        try:
            global_ti_map[int(float(r["time_index"]))] = float(r["time_s"])
        except (KeyError, ValueError, TypeError):
            pass
    ti_sorted = sorted(global_ti_map)
    times_arr = np.array([global_ti_map[ti] for ti in ti_sorted])
    chance = _chance(kind_rows)

    for smoothed in (False, True):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        group_vals: list[np.ndarray] = []
        for subj in subjects:
            acc_vals = np.array([subj_acc[subj].get(ti, (0, float("nan")))[1] for ti in ti_sorted])
            display = (
                _gaussian_display_smooth(acc_vals, times_arr, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else acc_vals
            )
            ax.plot(times_arr, display, color="#888888", linewidth=0.7, alpha=0.35)
            group_vals.append(acc_vals)
        # 组均值粗线
        group_mean = np.nanmean(np.stack(group_vals), axis=0)
        display_mean = (
            _gaussian_display_smooth(group_mean, times_arr, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
            if smoothed else group_mean
        )
        ax.plot(times_arr, display_mean, color=_kind_color(input_kind), linewidth=2.2, label="group mean", zorder=5)
        ax.axhline(chance, color="#999999", linestyle="--", linewidth=1.2, label=f"chance {chance:.3f}")
        _add_event_markers(ax, plot_config)
        _bar_ylim = _draw_peak_windows(ax, _find_peak_windows(display_mean, times_arr, baseline=chance, height_reference=chance,
                                                                merge_gap_s=0.10, min_window_width_s=0.15), color=_kind_color(input_kind))
        ax.legend(frameon=False)
        fig.tight_layout()
        if locals().get("_bar_ylim") is not None:
            ax.set_ylim(bottom=_bar_ylim)
        fname = f"time_accuracy_per_subject_smoothed_{split}.png" if smoothed else f"time_accuracy_per_subject_{split}.png"
        fig.savefig(output_dir / fname, dpi=180)
        plt.close(fig)

    # 证据曲线（同样画 per-subject）
    if not subj_ev:
        return
    for smoothed in (False, True):
        fig, ax = plt.subplots(figsize=(9.6, 5.0))
        group_ev: list[np.ndarray] = []
        for subj in subjects:
            ev_vals = np.array([subj_ev.get(subj, {}).get(ti, (0, float("nan")))[1] for ti in ti_sorted])
            display = (
                _gaussian_display_smooth(ev_vals, times_arr, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else ev_vals
            )
            ax.plot(times_arr, display, color="#888888", linewidth=0.7, alpha=0.35)
            group_ev.append(ev_vals)
        group_mean_ev = np.nanmean(np.stack(group_ev), axis=0)
        display_mean_ev = (
            _gaussian_display_smooth(group_mean_ev, times_arr, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
            if smoothed else group_mean_ev
        )
        ax.plot(times_arr, display_mean_ev, color=_kind_color(input_kind), linewidth=2.2, label="group mean", zorder=5)
        ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.2, label="neutral 0")
        _add_event_markers(ax, plot_config)
        _bar_ylim = _draw_peak_windows(ax, _find_peak_windows(display_mean_ev, times_arr, baseline=0.0, height_reference=0.0,
                                                                merge_gap_s=0.10, min_window_width_s=0.15), color=_kind_color(input_kind))
        suffix = " (150 ms smoothed)" if smoothed else ""
        ax.set_title(f"{split} per-subject time evidence{suffix}")
        ax.legend(frameon=False)
        fig.tight_layout()
        if locals().get("_bar_ylim") is not None:
            ax.set_ylim(bottom=_bar_ylim)
        fname = f"time_evidence_per_subject_smoothed_{split}.png" if smoothed else f"time_evidence_per_subject_{split}.png"
        fig.savefig(output_dir / fname, dpi=180)
        plt.close(fig)


def _plot_attention_per_subject(
    split: str,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    input_kind: str = "real",
) -> None:
    """每个被试单独一张注意力曲线图，存入 ``output_dir/attention_per_subject/``。

    每张图同时画该被试的曲线（蓝实线）和组均值（灰虚线）作为参照，
    并标注均匀基线（黑虚线）。文件名：``{subject_id}_{split}.png``。
    """
    kind_rows = [r for r in rows if str(r.get("input_kind", "real")) == input_kind]
    if not kind_rows:
        return

    subdir = output_dir / "attention_per_subject"
    subdir.mkdir(parents=True, exist_ok=True)
    curve_label = _attention_curve_label(kind_rows)

    # 聚合：每被试每时间点的均值注意力
    subj_time: dict[str, dict[int, list[float]]] = {}
    time_s_map: dict[int, float] = {}
    pool_valid_by_time: dict[int, bool] = {}
    for r in kind_rows:
        if all(r.get(key) in {"", None} for key in (
            "normalized_effective_attention_weight", "effective_attention_weight", "attention_weight"
        )):
            continue
        try:
            val = _attention_plot_value(r)
            subj = str(r["subject_id"])
            ti = int(float(r["time_index"]))
            time_s_map[ti] = float(r["time_s"])
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(val):
            valid = _attention_pool_valid(r)
            if pool_valid_by_time.setdefault(ti, valid) != valid:
                raise ValueError("attention_pool_valid must be constant across trials for each time")
            subj_time.setdefault(subj, {}).setdefault(ti, []).append(val)

    if not subj_time:
        return

    ti_sorted = sorted(time_s_map)
    times = np.array([time_s_map[ti] for ti in ti_sorted])

    # 高斯噪声输入的逐被试注意力（噪声基线，点虚线叠加；对齐到 real 的时间轴）
    gauss_subj_time: dict[str, dict[int, list[float]]] = {}
    for r in (
        row
        for row in rows
        if input_kind != "gaussian"
        and str(row.get("input_kind", "real")) == "gaussian"
    ):
        if all(r.get(key) in {"", None} for key in (
            "normalized_effective_attention_weight", "effective_attention_weight", "attention_weight"
        )):
            continue
        try:
            val = _attention_plot_value(r)
            subj = str(r["subject_id"])
            ti = int(float(r["time_index"]))
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(val) and ti in time_s_map:
            gauss_subj_time.setdefault(subj, {}).setdefault(ti, []).append(val)
    gauss_subj_vals: dict[str, np.ndarray] = {
        subj: np.array([
            float(np.mean(t_vals[ti])) if ti in t_vals else np.nan
            for ti in ti_sorted
        ])
        for subj, t_vals in gauss_subj_time.items()
    }

    # 组均值（作为参照线）
    all_means = []
    for subj in sorted(subj_time):
        vals = np.array([
            float(np.mean(subj_time[subj][ti])) if ti in subj_time[subj] else np.nan
            for ti in ti_sorted
        ])
        all_means.append(vals)
    group_mean = np.nanmean(np.stack(all_means), axis=0) if all_means else np.full(len(ti_sorted), np.nan)

    # Uniform uses the same valid support as pooling and the group summary.
    valid_support = np.array([pool_valid_by_time.get(ti, True) for ti in ti_sorted])
    n_valid_times = int(valid_support.sum())
    subj_uniform = {
        subj: sum(float(np.mean(vals)) for ti, vals in t_vals.items()
                  if pool_valid_by_time.get(ti, True)) / max(1, n_valid_times)
        for subj, t_vals in subj_time.items()
    }
    group_uniform = float(np.mean(list(subj_uniform.values()))) if subj_uniform else 0.0

    subjects = sorted(subj_time)
    for subj in subjects:
        subj_vals = np.array([
            float(np.mean(subj_time[subj][ti])) if ti in subj_time[subj] else np.nan
            for ti in ti_sorted
        ])
        uniform_val = subj_uniform.get(subj, group_uniform)

        for smoothed in (False, True):
            display_subj = (
                _gaussian_display_smooth(subj_vals, times, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else subj_vals
            )
            display_group = (
                _gaussian_display_smooth(group_mean, times, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else group_mean
            )
            fig, ax = plt.subplots(figsize=(8.6, 4.2))
            # 组均值参照（灰虚）
            ax.plot(times, display_group, color="#aaaaaa", linewidth=1.2,
                    linestyle="--", label="group mean", zorder=1)
            # 均匀基线（黑点虚）
            if bool(valid_support.all()):
                ax.axhline(uniform_val, color="#444444", linewidth=1.0,
                           linestyle=":", label=f"uniform ({uniform_val:.4f})", zorder=2)
            else:
                ax.plot(times, np.where(valid_support, uniform_val, 0.0), color="#444444",
                        linewidth=1.0, linestyle=":", label=f"uniform ({uniform_val:.4f})", zorder=2)
            # 该被试曲线（蓝实）
            ax.plot(times, display_subj, color=_kind_color(input_kind), linewidth=1.8,
                    label=f"subject {subj}", zorder=3)
            # 高斯噪声输入的同被试曲线（噪声基线，点虚线）
            gauss_vals = gauss_subj_vals.get(subj)
            if gauss_vals is not None and np.any(np.isfinite(gauss_vals)):
                display_gauss = (
                    _gaussian_display_smooth(gauss_vals, times, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                    if smoothed else gauss_vals
                )
                ax.plot(times, display_gauss, color=KIND_COLORS["gaussian"], linewidth=1.3,
                        linestyle=":", alpha=0.85, label="Gaussian noise", zorder=2)
            _add_event_markers(ax, plot_config)
            _bar_ylim = _draw_peak_windows(
                ax, _find_peak_windows(display_subj, times), color=_kind_color(input_kind)
            )
            ax.set_xlabel("Time from white-noise onset (s)")
            ax.set_ylabel(curve_label)
            suffix = " (150 ms smoothed)" if smoothed else ""
            ax.set_title(f"{split} combined attention strength — subject {subj}{suffix}")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            if _bar_ylim is not None:
                ax.set_ylim(bottom=_bar_ylim)
            fname = f"{subj}_{split}_smoothed.png" if smoothed else f"{subj}_{split}.png"
            fig.savefig(subdir / fname, dpi=150)
            plt.close(fig)


def _plot_attention_by_current_stimulus(
    split: str,
    attention_rows: Iterable[Mapping[str, str]],
    trial_rows: Iterable[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    input_kind: str = "real",
    curve_label: str = "Combined attention strength",
) -> None:
    """按每条trial的当前刺激分组，绘制组内trial等权平均注意力。

    原始图的阴影为trial间SEM；150ms平滑图仅展示均值曲线。逐时点
    均值、原始SEM、trial数和被试数另存CSV，便于核对每个刺激组。
    """
    summary, reference = _attention_by_current_stimulus_summary(
        split, attention_rows, trial_rows, input_kind=input_kind,
    )
    _plot_attention_stimulus_summary(split, summary, reference, output_dir, plot_config,
                                    curve_label=curve_label)


def _plot_attention_stimulus_summary(
    split: str,
    summary: Sequence[Mapping[str, object]],
    reference: Sequence[Mapping[str, object]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    curve_label: str = "Combined attention strength",
) -> None:
    """Render the original trial-weighted stimulus summaries without reaggregation."""
    if not summary:
        return
    subdir = output_dir / "time_by_stimulus"
    subdir.mkdir(parents=True, exist_ok=True)
    write_rows_csv(subdir / f"attention_by_stimulus_summary_{split}.csv", summary)
    write_rows_csv(subdir / f"attention_by_stimulus_reference_{split}.csv", reference)
    ti_sorted = [int(row["time_index"]) for row in reference]
    times = np.array([float(row["time_s"]) for row in reference])
    stims = sorted({str(row["current_stimulus"]) for row in summary})
    by_stim_time = {(str(row["current_stimulus"]), int(row["time_index"])): row for row in summary}
    stim_curves = {
        stim: (
            np.array([float(by_stim_time[stim, ti]["attention_mean"]) if (stim, ti) in by_stim_time else np.nan for ti in ti_sorted]),
            np.array([float(by_stim_time[stim, ti]["attention_sem"]) if (stim, ti) in by_stim_time else np.nan for ti in ti_sorted]),
        ) for stim in stims
    }
    # References follow the same trial weighting, including unequal stimulus sizes.
    group_ref = np.array([float(row["attention_mean"]) for row in reference])
    valid_support = np.array([_attention_pool_valid(row) for row in reference])
    valid_rows = [row for row, valid in zip(reference, valid_support) if valid]
    valid_count = sum(int(row["trial_count"]) for row in valid_rows)
    uniform_val = (sum(float(row["attention_mean"]) * int(row["trial_count"]) for row in valid_rows)
                   / valid_count) if valid_count else 0.0
    gauss_ref = np.array([float(row["gaussian_attention_mean"]) if row["gaussian_attention_mean"] not in {None, ""} else np.nan for row in reference])
    if not np.any(np.isfinite(gauss_ref)):
        gauss_ref = None

    # 每个刺激单独一张图（原始 + 平滑各一张）
    for stim in stims:
        if stim not in stim_curves:
            continue
        means, sems = stim_curves[stim]
        color = _STIMULUS_COLORS.get(stim, "#333333")
        for smoothed in (False, True):
            display = (
                _gaussian_display_smooth(means, times, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else means
            )
            display_ref = (
                _gaussian_display_smooth(group_ref, times, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else group_ref
            )
            fig, ax = plt.subplots(figsize=(8.6, 4.2))
            # 全刺激参照均值（灰虚线）
            ax.plot(times, display_ref, color="#aaaaaa", linewidth=1.2,
                    linestyle="--", label="all-stim mean", zorder=1)
            if gauss_ref is not None:
                display_gauss = (
                    _gaussian_display_smooth(gauss_ref, times, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                    if smoothed else gauss_ref
                )
                ax.plot(times, display_gauss, color=KIND_COLORS["gaussian"], linewidth=1.2,
                        linestyle=":", alpha=0.85, label=_kind_label("gaussian"), zorder=2)
            if bool(valid_support.all()):
                ax.axhline(uniform_val, color="#999999", linestyle=":", linewidth=1.0,
                           label=f"uniform ({uniform_val:.4f})", zorder=1)
            else:
                ax.plot(times, np.where(valid_support, uniform_val, 0.0), color="#999999",
                        linestyle=":", linewidth=1.0, label=f"uniform ({uniform_val:.4f})", zorder=1)
            ax.plot(times, display, color=color, linewidth=1.8,
                    label=f"current stimulus {stim} (trial mean)", zorder=3)
            if not smoothed:
                ax.fill_between(times, display - sems, display + sems,
                                color=color, alpha=0.15, linewidth=0, label="trial SEM")
            _add_event_markers(ax, plot_config)
            _bar_ylim = _draw_peak_windows(
                ax, _find_peak_windows(display, times), color=color
            )
            ax.set_xlabel("Time from white-noise onset (s)")
            ax.set_ylabel(curve_label)
            suffix = " (150 ms smoothed)" if smoothed else ""
            ax.set_title(f"{split} combined attention strength — stimulus {stim}, trial mean{suffix}")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            if _bar_ylim is not None:
                ax.set_ylim(bottom=_bar_ylim)
            fname = (
                f"attention_by_stimulus_{stim}_smoothed_{split}.png"
                if smoothed
                else f"attention_by_stimulus_{stim}_{split}.png"
            )
            fig.savefig(subdir / fname, dpi=180)
            plt.close(fig)


def _plot_time_by_current_stimulus(
    split: str,
    cell_metric_rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    plot_config: Mapping[str, object] | None = None,
    *,
    input_kind: str = "real",
) -> None:
    """按当前刺激条件分色画逐时点均衡准确率曲线（仅 real）。

    每条曲线是该 current_stimulus 条件下跨被试的均值 ± SEM。
    输出存入 ``output_dir/time_by_stimulus/``：
    ``time_accuracy_by_stimulus_{split}.png`` 和平滑版本。
    """
    kind_rows = [r for r in cell_metric_rows if str(r.get("input_kind", "real")) == input_kind]
    if not kind_rows:
        return

    subdir = output_dir / "time_by_stimulus"
    subdir.mkdir(parents=True, exist_ok=True)

    # 可用准确率字段：优先 complete_cell_balanced_accuracy，回退 balanced_accuracy
    def _pick_acc(r: Mapping[str, str]) -> float | None:
        for key in ("complete_cell_balanced_accuracy", "balanced_accuracy", "accuracy"):
            v = r.get(key, "")
            if v not in ("", None):
                try:
                    return float(v)
                except (ValueError, TypeError):
                    pass
        return None

    # stim → ti → list[per-subject accuracy]
    stim_ti_vals: dict[str, dict[int, list[float]]] = {}
    stim_ti_ts: dict[str, dict[int, float]] = {}
    for r in kind_rows:
        stim = str(r.get("current_stimulus", ""))
        if not stim:
            continue
        ti = int(float(r["time_index"]))
        ts = float(r["time_s"])
        acc = _pick_acc(r)
        if acc is not None and np.isfinite(acc):
            stim_ti_vals.setdefault(stim, {}).setdefault(ti, []).append(acc)
            stim_ti_ts.setdefault(stim, {})[ti] = ts

    if not stim_ti_vals:
        return

    stims = sorted(stim_ti_vals)
    chance = _chance(kind_rows)

    # 预计算每个 stim 的 mean/SEM 及全局均值参照
    stim_curves_acc: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for stim in stims:
        ti_sorted_s = sorted(stim_ti_vals[stim])
        times_s = np.array([stim_ti_ts[stim][ti] for ti in ti_sorted_s])
        means_s = np.array([float(np.mean(stim_ti_vals[stim][ti])) for ti in ti_sorted_s])
        sems_s = np.array([
            float(np.std(stim_ti_vals[stim][ti], ddof=1) / math.sqrt(len(stim_ti_vals[stim][ti])))
            if len(stim_ti_vals[stim][ti]) > 1 else 0.0
            for ti in ti_sorted_s
        ])
        stim_curves_acc[stim] = (times_s, means_s, sems_s)

    # 全刺激参照：使用时间点最多的 stim 的时间轴
    ref_stim = max(stim_curves_acc, key=lambda s: len(stim_curves_acc[s][0]))
    ref_times = stim_curves_acc[ref_stim][0]
    # 对齐到 ref_times 做均值参照（只用时间轴完全一致的 stim）
    aligned = [m for _, (t, m, _) in stim_curves_acc.items() if len(t) == len(ref_times) and np.allclose(t, ref_times)]
    group_ref_acc = np.nanmean(np.stack(aligned), axis=0) if aligned else stim_curves_acc[ref_stim][1]

    # 每个刺激单独一张图（原始 + 平滑）
    for stim in stims:
        times_s, means_s, sems_s = stim_curves_acc[stim]
        color = _STIMULUS_COLORS.get(stim, "#333333")
        # 全局参照对齐
        if len(ref_times) == len(times_s) and np.allclose(ref_times, times_s):
            ref_curve = group_ref_acc
        else:
            ref_curve = None
        for smoothed in (False, True):
            display = (
                _gaussian_display_smooth(means_s, times_s, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                if smoothed else means_s
            )
            fig, ax = plt.subplots(figsize=(8.6, 4.2))
            # 全刺激均值参照（灰虚）
            if ref_curve is not None:
                display_ref = (
                    _gaussian_display_smooth(ref_curve, times_s, TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S)
                    if smoothed else ref_curve
                )
                ax.plot(times_s, display_ref, color="#aaaaaa", linewidth=1.2,
                        linestyle="--", label="all-stim mean", zorder=1)
            ax.axhline(chance, color="#999999", linestyle=":", linewidth=1.0,
                       label=f"chance {chance:.3f}", zorder=1)
            ax.plot(times_s, display, color=color, linewidth=1.8,
                    label=f"current pitch {stim}", zorder=3)
            ax.fill_between(times_s, display - sems_s, display + sems_s,
                            color=color, alpha=0.15, linewidth=0)
            _add_event_markers(ax, plot_config)
            _bar_ylim = _draw_peak_windows(
                ax, _find_peak_windows(display, times_s, height_reference=chance), color=color
            )
            ax.set_xlabel("Time from white-noise onset (s)")
            ax.set_ylabel("Cell balanced accuracy")
            suffix = " (150 ms smoothed)" if smoothed else ""
            ax.set_title(f"{split} time accuracy — current pitch {stim}{suffix}")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            if _bar_ylim is not None:
                ax.set_ylim(bottom=_bar_ylim)
            fname = (
                f"time_accuracy_by_stimulus_{stim}_smoothed_{split}.png"
                if smoothed
                else f"time_accuracy_by_stimulus_{stim}_{split}.png"
            )
            fig.savefig(subdir / fname, dpi=180)
            plt.close(fig)
