"""analysis.decoding: shared Target report operations."""
from __future__ import annotations
import csv
import math
from collections import defaultdict
from collections.abc import Collection, Iterable, Mapping
from pathlib import Path
from statistics import mean
from typing import Sequence
from scipy.signal import find_peaks
import numpy as np
from pb2023_eeg.models.target_decoder import TASK_CLASS_COUNTS
from pb2023_eeg.utils.config import load_yaml


DEFAULT_EVENT_MARKERS = (
    {"time_s": 0.0, "label": "white-noise onset", "color": "#666666", "alpha": 0.70},
    {"time_s": 1.0, "label": "tone onset", "color": "#4C78A8", "alpha": 0.90},
    {"time_s": 1.25, "label": "tone offset / response cue", "color": "#F58518", "alpha": 0.90},
)


KIND_ORDER = {
    "real": 0,
    "gaussian": 1,
    "time_shuffle": 2,
    "shuffled": 2,
    "time_roll": 3,
}


KIND_LABELS = {
    "real": "real EEG",
    "gaussian": "Gaussian EEG",
    "time_shuffle": "time-shuffled EEG",
    "shuffled": "time-shuffled EEG",
    "time_roll": "time-rolled EEG",
}


KIND_COLORS = {
    "real": "#1f77b4",
    "gaussian": "#555555",
    "time_shuffle": "#d62728",
    "shuffled": "#d62728",
    "time_roll": "#9467bd",
}


STANDALONE_INPUT_KINDS = ("real", "gaussian", "time_shuffle", "time_roll")


STANDALONE_PLOT_EXPORT_STEMS = (
    "attention_weights",
    "trial_metrics",
    "time_accuracy",
    "time_cell_summary",
    "time_cell_metrics",
    "time_activation",
)


PLOT_INPUT_KINDS = frozenset({"real", "shuffled"})


LEGACY_ACTIVATION_DEFINITION = "max_all_class_score"


LEGACY_TIME_SCORE_MODE = "sigmoid"


TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S = 0.150


ATTENTION_FIGURE_STEMS = {
    "attention_score": "01_attention_score",
    "attention_gate": "02_attention_gate",
    "attention_mass_time": "03_effective_attention",
    "attention_time": "04_effective_attention_normalized",
    "attention_relative": "05_effective_attention_relative",
    "attention_subject_curves": "06_effective_attention_subject_curves",
    "attention_mass": "07_effective_attention_mass",
}


def attention_figure_filename(kind: str, split: str) -> str:
    """Return the ordered name of a base attention figure."""
    return f"{ATTENTION_FIGURE_STEMS[kind]}_{split}.png"


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字符串字典列表；文件不存在时返回空列表。

    这里只允许可选表缺失；report workflow 在读取前检查 Target 必需输入。
    """
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_plot_config(config_path: Path) -> dict[str, object]:
    """Load either a full run snapshot or a standalone plots mapping."""
    config = load_yaml(Path(config_path))
    if not isinstance(config, Mapping):
        raise ValueError("Plot configuration YAML must contain a mapping")
    if "plots" in config:
        configured = config["plots"]
    elif any(key in config for key in ("task", "data", "model", "train")):
        configured = {}
    else:
        configured = config
    if configured is None:
        return {}
    if not isinstance(configured, Mapping):
        raise ValueError("plots must be a mapping when provided")
    return dict(configured)


def _default_plot_config(run_dir: Path) -> dict[str, object]:
    """从运行快照读取 plots 配置；快照不存在时返回空配置。"""
    snapshot = run_dir / "config_snapshot.yaml"
    if not snapshot.exists():
        return {}
    return load_plot_config(snapshot)


def resolve_plot_config(
    plot_config: Mapping[str, object] | None = None,
    *,
    time_axis_start_s: float | None = None,
    time_axis_end_s: float | None = None,
) -> dict[str, object]:
    """Copy plot options and apply a validated optional time-axis override."""
    if plot_config is not None and not isinstance(plot_config, Mapping):
        raise ValueError("plot_config must be a mapping when provided")
    resolved = dict(plot_config or {})
    has_start = time_axis_start_s is not None
    has_end = time_axis_end_s is not None
    if has_start != has_end:
        raise ValueError(
            "time_axis_start_s and time_axis_end_s must be provided together"
        )
    if has_start:
        resolved["time_axis"] = {
            "start_s": float(time_axis_start_s),
            "end_s": float(time_axis_end_s),
        }
    resolve_plot_time_axis_limits(resolved)
    return resolved


def normalize_plot_input_kinds(
    requested: Iterable[str],
) -> tuple[str, ...]:
    """Validate and de-duplicate standalone plot conditions in request order."""
    values = (requested,) if isinstance(requested, str) else tuple(requested)
    normalized = tuple(dict.fromkeys(str(kind) for kind in values))
    if not normalized:
        raise ValueError("input_kinds must contain at least one condition")
    unsupported = [
        kind for kind in normalized if kind not in STANDALONE_INPUT_KINDS
    ]
    if unsupported:
        raise ValueError(f"Unsupported standalone input kinds: {unsupported}")
    return normalized


def normalize_plot_splits(splits: Iterable[str]) -> tuple[str, ...]:
    """Materialize and de-duplicate split names so batch iterations can reuse them."""
    values = (splits,) if isinstance(splits, str) else tuple(splits)
    normalized = tuple(dict.fromkeys(str(split) for split in values))
    if not normalized:
        raise ValueError("splits must contain at least one name")
    return normalized


def _standalone_export_prefix(split: str, input_kind: str) -> str:
    return f"{split}_" if input_kind == "real" else f"{split}_{input_kind}_"


def _require_standalone_export(
    run_dir: Path,
    *,
    split: str,
    input_kind: str,
) -> str:
    """Require nonempty Target core tables before creating any report outputs."""
    prefix = _standalone_export_prefix(split, input_kind)
    for stem in ("attention_weights", "trial_metrics", "time_accuracy"):
        path = run_dir / f"{prefix}{stem}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required {input_kind} Target export for split {split!r}: {path}")
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.DictReader(stream), None) is None:
                raise ValueError(f"Empty required Target export: {path}")
    return prefix


def _gaussian_baseline_explicitly_disabled(
    plot_config: Mapping[str, object] | None,
) -> bool:
    """判断高斯基线是否在配置中被明确关闭。

    显式 ``enabled: false`` 时绘图器不会读取目录中可能残留的历史 Gaussian CSV。
    """
    if not isinstance(plot_config, Mapping):
        return False
    configured = plot_config.get("gaussian_baseline")
    return (
        isinstance(configured, Mapping)
        and configured.get("enabled") is False
    )


def _float(row: Mapping[str, str], key: str, default: float = 0.0) -> float:
    """读取 CSV 行中的浮点字段，空值回退到 ``default``。"""
    value = row.get(key)
    if value in {None, ""}:
        return default
    return float(value)


def _int(row: Mapping[str, str], key: str, default: int = 0) -> int:
    """读取 CSV 行中的整数字段，并兼容 ``"1.0"`` 形式；空值使用默认值。"""
    value = row.get(key)
    if value in {None, ""}:
        return default
    return int(float(value))


def _input_kind(row: Mapping[str, str], default: str = "real") -> str:
    """读取输入类型（如 real/gaussian），兼容没有该列的历史真实 EEG 行。"""
    value = str(row.get("input_kind", "") or default).strip()
    return value or default


def _rows_with_input_kind(rows: Sequence[Mapping[str, str]], input_kind: str) -> list[dict[str, str]]:
    """复制 CSV 行，并仅为缺失的 ``input_kind`` 填入调用方给定类型。"""
    normalized: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        if not item.get("input_kind"):
            item["input_kind"] = input_kind
        normalized.append(item)
    return normalized


def _attention_plot_value(row: Mapping[str, str]) -> float:
    """按新到旧 schema 选择可比较的注意力绘图值。

    优先使用归一化 effective attention，其次是未归一化 effective attention，最后回退
    到历史 ``attention_weight``。
    """
    normalized = row.get("normalized_effective_attention_weight")
    if normalized not in {None, ""}:
        return float(normalized)
    effective = row.get("effective_attention_weight")
    if effective not in {None, ""}:
        return float(effective)
    return _float(row, "attention_weight")


def _attention_pool_valid(row: Mapping[str, object]) -> bool:
    """读取 pooling support 标记；历史 CSV 缺列时视为全轴有效。"""
    value = row.get("attention_pool_valid")
    if value in {None, ""}:
        return True
    return bool(int(float(value)))


def _attention_curve_label(rows: Sequence[Mapping[str, object]]) -> str:
    """Label normalized strength separately from scorer and signed pooling weights."""
    if any(row.get("normalized_effective_attention_weight") not in {None, ""} for row in rows):
        return "Normalized |combined weight|"
    if any(row.get("effective_attention_weight") not in {None, ""} for row in rows):
        return "Combined pooling weight"
    return "Score branch weight"


def _kind_sort_key(kind: str) -> tuple[int, str]:
    """返回输入类型的稳定排序键：真实 EEG、高斯基线、其他扩展类型。"""
    return (KIND_ORDER.get(kind, 100), kind)


def _kind_label(kind: str) -> str:
    """将内部输入类型映射为图例标签，未知类型原样显示。"""
    return KIND_LABELS.get(kind, kind)


def _kind_color(kind: str) -> str:
    """返回输入类型的固定绘图颜色，未知类型使用备用紫色。"""
    return KIND_COLORS.get(kind, "#9467bd")


def _is_plottable_input_kind(
    kind: str,
    plot_input_kinds: Collection[str] | None = None,
) -> bool:
    """判断输入类型是否进入时间曲线类 PNG；gaussian 不画曲线，仍进汇总表。

    柱状图（trial accuracy / margins / attention mass）不走此过滤。
    """
    allowed = PLOT_INPUT_KINDS if plot_input_kinds is None else plot_input_kinds
    return str(kind) in allowed


def _filter_rows_for_plot(
    rows: Sequence[Mapping[str, object]],
    plot_input_kinds: Collection[str] | None = None,
) -> list[Mapping[str, object]]:
    """过滤出应绘制到时间曲线类图的汇总行（默认仅 real）。"""
    return [
        row
        for row in rows
        if _is_plottable_input_kind(
            str(row.get("input_kind", "real")),
            plot_input_kinds,
        )
    ]


def _filter_grouped_for_plot(
    grouped: Mapping[str, Sequence[Mapping[str, object]]],
    plot_input_kinds: Collection[str] | None = None,
) -> dict[str, list[Mapping[str, object]]]:
    """从按 input_kind 分组的曲线数据中去掉不画的类型（默认去掉 gaussian）。"""
    return {
        kind: list(kind_rows)
        for kind, kind_rows in grouped.items()
        if _is_plottable_input_kind(kind, plot_input_kinds) and kind_rows
    }


def _sem(values: Sequence[float]) -> float:
    """计算样本均值标准误；不足两个观测时按零返回。"""
    if len(values) <= 1:
        return 0.0
    center = mean(values)
    variance = sum((value - center) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance) / math.sqrt(len(values))


def _gaussian_display_smooth(
    values: Sequence[float],
    times: Sequence[float],
    window_s: float = TIME_ACCURACY_DISPLAY_SMOOTHING_WINDOW_S,
) -> np.ndarray:
    """按论文的居中 Gaussian 移动平均平滑组均值，仅供绘图显示。

    ``window_s`` 是 Gaussian 核的完整时间窗，而不是 sigma。核标准差取窗口采样数的
    1/6，与 ``pb2023_eeg_analysis`` 的实现一致；边界和 NaN 位置按有效权重重新归一化。
    """
    array = np.asarray(values, dtype=float)
    time_array = np.asarray(times, dtype=float)
    if array.ndim != 1 or time_array.shape != array.shape:
        raise ValueError("values and times must be aligned 1D arrays")
    if array.size < 2 or window_s <= 0.0:
        return array.copy()

    time_steps = np.diff(time_array)
    valid_steps = time_steps[np.isfinite(time_steps) & (time_steps > 0.0)]
    if valid_steps.size == 0:
        return array.copy()
    sample_interval_s = float(np.median(valid_steps))
    window_samples = max(
        1,
        min(array.size, int(round(float(window_s) / sample_interval_s))),
    )
    if window_samples <= 1:
        return array.copy()

    offsets = np.arange(window_samples, dtype=float) - (window_samples - 1) / 2.0
    sigma_samples = max(window_samples / 6.0, np.finfo(float).eps)
    kernel = np.exp(-0.5 * np.square(offsets / sigma_samples))
    finite = np.isfinite(array)
    numerator = np.convolve(np.where(finite, array, 0.0), kernel, mode="same")
    denominator = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(array, np.nan),
        where=denominator > 0.0,
    )


def _cluster_from_t(
    t_values: np.ndarray,
    point_p: float = 0.05,
) -> list[tuple[int, int, float]]:
    """找连续正向显著时间点 cluster，返回 (start_idx, end_idx, cluster_mass)。

    ``point_p`` 为双侧逐点入门阈值（正态近似）；只收集 t > 0 且满足阈值的时间点，
    cluster_mass = sum(t_values[start:end+1])。
    """
    p_values = np.array(
        [
            math.erfc(abs(float(t)) / math.sqrt(2.0)) if np.isfinite(t) else 1.0
            for t in t_values
        ]
    )
    in_cluster = (t_values > 0) & (p_values < point_p)
    clusters: list[tuple[int, int, float]] = []
    start: int | None = None
    for i, flag in enumerate(in_cluster):
        if flag and start is None:
            start = i
        if (not flag or i == len(in_cluster) - 1) and start is not None:
            end = i if (flag and i == len(in_cluster) - 1) else i - 1
            clusters.append((start, end, float(np.sum(t_values[start : end + 1]))))
            start = None
    return clusters


def _sign_flip_cluster_perm(
    subject_matrix: np.ndarray,
    n_permutations: int = 1000,
    point_p: float = 0.05,
    seed: int = 20260530,
) -> list[tuple[int, int, float]]:
    """Sign-flip cluster permutation 检验，返回通过校正的 cluster。

    ``subject_matrix`` 形状 ``[n_subjects, n_times]``，值为每个被试每时间点相对
    零假设的偏差（准确率已减去 chance，evidence 直接用，注意力已减去均匀基线）。

    每次置换随机翻转全体被试整条曲线的符号，保留时间相关结构。返回
    ``(start_idx, end_idx, corrected_p)`` 列表，按 start_idx 排序，包含所有观察
    到的 cluster（调用方再按 p 阈值筛选）。被试数 < 2 时返回空列表。
    """
    n_subj, n_times = subject_matrix.shape
    if n_subj < 2:
        return []

    # 计算观察到的组水平 t 值时间序列
    col_mean = np.mean(subject_matrix, axis=0)
    col_std = np.std(subject_matrix, axis=0, ddof=1)
    safe_std = np.where(col_std > 0, col_std, np.nan)
    t_obs = col_mean / (safe_std / math.sqrt(n_subj))
    t_obs = np.where(np.isfinite(t_obs), t_obs, 0.0)

    observed = _cluster_from_t(t_obs, point_p)
    if not observed:
        return []

    rng = np.random.default_rng(seed)
    null_max = np.zeros(n_permutations)
    for i in range(n_permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_subj, 1))
        perm = subject_matrix * signs
        perm_mean = np.mean(perm, axis=0)
        perm_std = np.std(perm, axis=0, ddof=1)
        safe_perm_std = np.where(perm_std > 0, perm_std, np.nan)
        perm_t = perm_mean / (safe_perm_std / math.sqrt(n_subj))
        perm_t = np.where(np.isfinite(perm_t), perm_t, 0.0)
        perm_clusters = _cluster_from_t(perm_t, point_p)
        null_max[i] = max((c[2] for c in perm_clusters), default=0.0)

    result = []
    for start, end, mass in observed:
        corrected_p = float((1 + np.sum(null_max >= mass)) / (n_permutations + 1))
        result.append((start, end, corrected_p))
    return sorted(result, key=lambda x: x[0])


def _build_subject_matrix_from_rows(
    rows: Sequence[Mapping[str, str]],
    value_key: str,
    input_kind: str = "real",
) -> tuple[np.ndarray, np.ndarray]:
    """从 time_accuracy CSV 原始行提取 [n_subjects, n_times] 矩阵和 times 数组。

    ``rows`` 中每行已是某被试某时间点的汇总值；按 time_index 排序确定列序。
    缺失位置填 NaN。返回 ``(matrix, times_array)``；没有有效数据时返回空数组。
    """
    filtered = [r for r in rows if str(r.get("input_kind", "real")) == input_kind]
    if not filtered:
        return np.empty((0, 0)), np.empty(0)

    time_map: dict[int, float] = {}
    for r in filtered:
        try:
            ti = int(float(r["time_index"]))
            time_map[ti] = float(r["time_s"])
        except (KeyError, ValueError, TypeError):
            pass
    if not time_map:
        return np.empty((0, 0)), np.empty(0)
    time_indices = sorted(time_map)
    times = np.array([time_map[ti] for ti in time_indices])
    ti_col = {ti: col for col, ti in enumerate(time_indices)}

    subj_time: dict[str, dict[int, float]] = {}
    for r in filtered:
        val_str = r.get(value_key, "")
        if val_str in {"", None}:
            continue
        try:
            val = float(val_str)
            subj = str(r["subject_id"])
            ti = int(float(r["time_index"]))
        except (KeyError, ValueError, TypeError):
            continue
        if np.isfinite(val):
            subj_time.setdefault(subj, {})[ti] = val

    subjects = sorted(subj_time)
    if not subjects:
        return np.empty((0, 0)), times
    matrix = np.full((len(subjects), len(time_indices)), np.nan)
    for si, subj in enumerate(subjects):
        for ti, val in subj_time[subj].items():
            if ti in ti_col:
                matrix[si, ti_col[ti]] = val
    return matrix, times


def _build_attention_subject_matrix(
    rows: Sequence[Mapping[str, str]],
    input_kind: str = "real",
) -> tuple[np.ndarray, np.ndarray]:
    """从 attention_weights trial 级行聚合为 [n_subjects, n_times] 均值矩阵。

    优先使用 ``normalized_effective_attention_weight``，回退 ``effective_attention_weight``
    再回退 ``attention_weight``。每个被试每时间点取所有 trial 的均值。
    没有有效数据时返回空数组。
    """
    filtered = [r for r in rows if str(r.get("input_kind", "real")) == input_kind]
    if not filtered:
        return np.empty((0, 0)), np.empty(0)

    time_map: dict[int, float] = {}
    for r in filtered:
        try:
            ti = int(float(r["time_index"]))
            time_map[ti] = float(r["time_s"])
        except (KeyError, ValueError, TypeError):
            pass
    if not time_map:
        return np.empty((0, 0)), np.empty(0)
    time_indices = sorted(time_map)
    times = np.array([time_map[ti] for ti in time_indices])
    ti_col = {ti: col for col, ti in enumerate(time_indices)}

    # 每个被试每时间点累积 trial 级 attention 值
    accum: dict[str, dict[int, list[float]]] = {}
    for r in filtered:
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
        if np.isfinite(val):
            accum.setdefault(subj, {}).setdefault(ti, []).append(val)

    subjects = sorted(accum)
    if not subjects:
        return np.empty((0, 0)), times
    matrix = np.full((len(subjects), len(time_indices)), np.nan)
    for si, subj in enumerate(subjects):
        for ti, vals in accum[subj].items():
            if ti in ti_col:
                matrix[si, ti_col[ti]] = float(np.mean(vals))
    return matrix, times


def _find_peak_windows(
    values: np.ndarray,
    times: np.ndarray,
    *,
    min_prominence_fraction: float = 0.10,
    baseline: float | np.ndarray | None = None,
    height_reference: float | str = "mean",
    min_relative_height_fraction: float = 0.45,
    min_peak_over_reference_fraction: float = 0.0,
    max_fwhm_width: float = 0.35,
    merge_gap_s: float = 0.0,
    min_window_width_s: float = 0.0,
    max_windows: int = 3,
) -> list[tuple[float, float, float]]:
    """找曲线上所有显著局部峰的 FWHM 时间窗。

    返回 ``[(start_s, end_s, rel_prominence), ...]``，按时间顺序排列。
    ``rel_prominence`` 是原始 prominence 除以整条曲线的振幅范围（0~1），
    可跨曲线比较，也供绘图层决定阴影深浅和标注优先级。

    ``min_peak_over_reference_fraction``（默认 0.0，关闭）：绝对平坦门。要求最高峰
    相对 height_reference 的抬升 >= reference × 此比例，否则整条曲线判为无显著峰、
    返回 []。其他门都是自相对的（按各曲线自身量程缩放），无法区分「真实峰曲线」和
    「纯噪声平坦曲线」——平坦曲线里最高的噪声点也能通过自相对门。attention 曲线取 3%
    可干净区分真实峰（抬升 >10%）与时间打乱对照（抬升 <1%）。

    两个过滤门用**不同**的参考点，缺一不可：

    - ``baseline``（语义门）：逐时间点的零假设水平，如 accuracy 传入 Gaussian
      曲线、evidence 传入 Gaussian 曲线或 0。只保留峰值 > 同时刻 baseline 的峰，
      排除"解码还不如随机"的伪峰。可以是标量或时间轴对齐的数组。
    - ``height_reference``（相对高度门的量尺原点）：固定标量，或 ``'mean'``：
      - ``'mean'``（默认）：自动用曲线均值作为原点。对归一化注意力权重（值
        聚集在 1/T 附近），这能让相对高度过滤正确区分真实峰与平台伪峰。若传
        ``0.0``，则所有峰的"高度"都约等于均值，过滤器实际无效（已观察到错标）。
      - accuracy 传 chance（显式数值），evidence 传 0.0（显式数值）。

    流程：
    1. 用 scipy.signal.find_peaks 找局部最大值，按 prominence 过滤（默认至少为
       全局振幅的 10%），作为噪声下限。attention 曲线的高对比峰有时因曲线边界导致
       scipy 双侧 prominence 被低估；因此主要依靠第 3 步的相对高度门做区分。
    2. 语义门（baseline）：仅保留峰值 > 同时刻 baseline 的峰。
    3. 相对高度门（min_relative_height_fraction）：以 height_reference 为原点计算
       每个峰的高度，只保留高度 >= 最强峰高度 × 阈值（默认 30%）的峰，排除相对
       主峰过弱的次级峰。设为 0 可关闭本门。在 30 个被试的测试中，真实峰的相对
       高度集中在 31–100%，噪声/平台伪峰在 ≤25%，存在清晰间隙。
    4. 对每个峰，用其两侧相邻谷值的较高者作为局部基线，计算半峰阈值，向两侧扩展
       得到 FWHM 区间。
    5. 按时间顺序排列，合并重叠区间；用曲线振幅归一化 prominence。

    曲线全为 NaN 或找不到任何通过过滤的峰时返回空列表。
    """
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return []
    v = np.where(finite_mask, values, np.nanmin(values))
    global_range = float(np.nanmax(v) - np.nanmin(v))
    if global_range <= 0:
        return []

    # 'mean' 哨兵：用曲线均值作为高度参考原点（适合归一化注意力权重，均值 = 1/T）
    resolved_height_reference: float
    if isinstance(height_reference, str):
        if height_reference == "mean":
            resolved_height_reference = float(np.nanmean(v))
        else:
            raise ValueError(f"height_reference must be a float or 'mean', got {height_reference!r}")
    else:
        resolved_height_reference = float(height_reference)

    # baseline 数组：统一转成与 v 等长的 numpy 数组方便逐点比较
    if baseline is None:
        baseline_arr: np.ndarray | None = None
    elif np.ndim(baseline) == 0:
        baseline_arr = np.full(len(v), float(baseline))
    else:
        baseline_arr = np.asarray(baseline, dtype=float)
        if len(baseline_arr) != len(v):
            baseline_arr = None  # 长度不对齐时回退到无过滤

    min_prominence = min_prominence_fraction * global_range
    peak_indices, props = find_peaks(v, prominence=min_prominence)
    if len(peak_indices) == 0:
        return []

    # 语义过滤：峰值必须高于对应时间点的基线
    if baseline_arr is not None:
        keep = np.array([
            float(v[idx]) > float(baseline_arr[idx])
            for idx in peak_indices
        ])
        peak_indices = peak_indices[keep]
        props = {k: arr[keep] for k, arr in props.items()}
    if len(peak_indices) == 0:
        return []

    # 相对高度过滤：排除相比最强峰过弱的次级峰。
    # 高度以 resolved_height_reference 为原点。
    heights = np.array([
        float(v[idx]) - resolved_height_reference
        for idx in peak_indices
    ])
    max_height = float(np.max(heights))
    # 没有任何峰高于 height_reference（如 accuracy 全在 chance 以下）→ 无有效峰。
    # 与 window_comparison.find_peak_windows 保持一致：直接返回空列表，
    # 避免对低于参考水平的峰错误标注 FWHM 窗口。
    if max_height <= 0:
        return []
    # 绝对平坦门：最高峰相对参考原点的抬升不足 → 整条曲线近似均匀（噪声），无真实峰。
    # 与 window_comparison.find_peak_windows 保持一致。
    if min_peak_over_reference_fraction > 0.0:
        ref_scale = abs(resolved_height_reference)
        if ref_scale > 0.0 and max_height < min_peak_over_reference_fraction * ref_scale:
            return []
    if min_relative_height_fraction > 0:
        height_keep = heights >= min_relative_height_fraction * max_height
        peak_indices = peak_indices[height_keep]
        props = {k: arr[height_keep] for k, arr in props.items()}
    if len(peak_indices) == 0:
        return []

    # (start_s, end_s, raw_prominence)
    windows: list[tuple[float, float, float]] = []
    for i, peak_idx in enumerate(peak_indices):
        left_base = int(props["left_bases"][i])
        right_base = int(props["right_bases"][i])
        # 局部基线：取两侧 prominence base 的较高者，并限制不低于 height_reference。
        # 若 base 落在远端深谷（如负值区）而不截断，半峰阈值会被拉得过低，
        # 导致 FWHM 向两侧过度扩展，超过 max_fwhm_width 而被误判为平台伪峰。
        # 用 height_reference 截断后，evidence 曲线以 0 为底、accuracy 以 chance 为底量宽。
        local_baseline = max(
            float(v[left_base]),
            float(v[right_base]),
            resolved_height_reference,
        )
        peak_val = float(v[peak_idx])
        if peak_val <= local_baseline:
            continue
        threshold = local_baseline + 0.5 * (peak_val - local_baseline)
        # 向左扩展
        start = peak_idx
        while start > 0 and v[start - 1] >= threshold:
            start -= 1
        # 向右扩展
        end = peak_idx
        while end < len(v) - 1 and v[end + 1] >= threshold:
            end += 1
        # FWHM 宽度上限：宽于 max_fwhm_width（默认 0.35s）的峰为平台伪峰，丢弃
        fwhm_width = float(times[end]) - float(times[start])
        if fwhm_width > max_fwhm_width:
            continue
        windows.append((float(times[start]), float(times[end]), float(props["prominences"][i])))

    # 合并重叠或相近区间：gap ≤ merge_gap_s 的相邻窗口视为同一激活簇
    if not windows:
        return []
    windows.sort()
    merged: list[tuple[float, float, float]] = [windows[0]]
    for s, e, prom in windows[1:]:
        if s <= merged[-1][1] + merge_gap_s:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e), max(merged[-1][2], prom))
        else:
            merged.append((s, e, prom))

    # 最小宽度：对过窄的窗口向两侧对称扩展（不超出曲线时间范围）
    if min_window_width_s > 0.0:
        t_min, t_max = float(times[0]), float(times[-1])
        padded = []
        for s, e, p in merged:
            w = e - s
            if w < min_window_width_s:
                pad = (min_window_width_s - w) / 2.0
                s = max(t_min, s - pad)
                e = min(t_max, e + pad)
            padded.append((s, e, p))
        merged = padded

    # 只保留最突出的少数窗口，避免组曲线/控制曲线被弱噪声峰淹没。
    # 先按 prominence 选 top-K，再恢复时间顺序，保证 P1/P2/P3 表示时间顺序。
    if max_windows > 0 and len(merged) > max_windows:
        merged = sorted(
            sorted(merged, key=lambda item: item[2], reverse=True)[:max_windows],
            key=lambda item: item[0],
        )

    # 相对显著度按曲线自身振幅归一化，保留跨曲线/试次的可比性。
    return [(s, e, p / global_range) for s, e, p in merged]


def _event_markers(plot_config: Mapping[str, object] | None = None) -> list[dict[str, object]]:
    """解析时间图上的实验事件标记。

    支持关闭标记或通过配置提供 ``time_s/label/color/alpha``；未提供有效自定义列表时
    返回白噪声、音调开始和音调结束/反应提示三个默认事件的副本。
    """
    marker_config: Mapping[str, object] = {}
    if isinstance(plot_config, Mapping):
        configured = plot_config.get("event_markers", {})
        if configured is None:
            configured = {}
        if isinstance(configured, Mapping):
            marker_config = configured
    if not bool(marker_config.get("enabled", True)):
        return []
    configured_markers = marker_config.get("markers")
    if isinstance(configured_markers, Sequence) and not isinstance(configured_markers, (str, bytes)):
        markers: list[dict[str, object]] = []
        for marker in configured_markers:
            if isinstance(marker, Mapping):
                markers.append(
                    {
                        "time_s": float(marker.get("time_s", marker.get("time", 0.0))),
                        "label": str(marker.get("label", "event")),
                        "color": str(marker.get("color", "#666666")),
                        "alpha": float(marker.get("alpha", 0.8)),
                    }
                )
        if markers:
            return markers
    return [dict(marker) for marker in DEFAULT_EVENT_MARKERS]


def resolve_plot_time_axis_limits(
    plot_config: Mapping[str, object] | None = None,
) -> tuple[float, float] | None:
    """Resolve an optional fixed display range for every time-based figure."""
    if not isinstance(plot_config, Mapping):
        return None
    configured = plot_config.get("time_axis")
    if configured is None:
        return None
    if not isinstance(configured, Mapping):
        raise ValueError("plots.time_axis must be a mapping")
    start = configured.get("start_s")
    end = configured.get("end_s")
    if start is None and end is None:
        return None
    if start is None or end is None:
        raise ValueError("plots.time_axis requires both start_s and end_s")
    start_s = float(start)
    end_s = float(end)
    if not math.isfinite(start_s) or not math.isfinite(end_s) or start_s >= end_s:
        raise ValueError(
            "plots.time_axis start_s/end_s must be finite with start_s < end_s"
        )
    return start_s, end_s


def _attention_component_summary(
    rows: Sequence[Mapping[str, str]],
    key: str,
) -> dict[tuple[str, int, float], tuple[float, float]]:
    """按输入类型和时间点汇总一个注意力组件的跨被试均值与 SEM。

    聚合顺序是先对同一被试/时间点的 trial 求均值，再跨被试统计，防止 trial 较多的
    被试获得更大权重。返回键为 ``(input_kind, time_index, time_s)``。
    """
    subject_values: dict[tuple[str, int, int, float], list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value in {None, ""}:
            continue
        subject_values[
            (
                _input_kind(row),
                _int(row, "subject_id"),
                _int(row, "time_index"),
                _float(row, "time_s"),
            )
        ].append(float(value))
    time_values: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (input_kind, _subject_id, time_index, time_s), values in subject_values.items():
        time_values[(input_kind, time_index, time_s)].append(mean(values))
    return {key_: (mean(values), _sem(values)) for key_, values in time_values.items()}


def _subject_time_summary(rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """生成注意力时间曲线的被试等权汇总行。

    输出每行对应一个输入类型和时间点，包含归一化 effective attention 的均值/SEM、
    raw/effective/gate 分量、经验均匀基线及相对基线差值。该列表可直接写入汇总 CSV。
    """
    subject_time_values: dict[tuple[str, int, int, float], list[float]] = defaultdict(list)
    validity_by_time: dict[tuple[str, int, float], set[bool]] = defaultdict(set)
    for row in rows:
        time_key = (
            _input_kind(row),
            _int(row, "time_index"),
            _float(row, "time_s"),
        )
        validity_by_time[time_key].add(_attention_pool_valid(row))
        subject_time_values[
            (
                time_key[0],
                _int(row, "subject_id"),
                time_key[1],
                time_key[2],
            )
        ].append(_attention_plot_value(row))

    inconsistent = [key for key, values in validity_by_time.items() if len(values) != 1]
    if inconsistent:
        raise ValueError(
            "attention_pool_valid must be constant across trials for each input/time"
        )
    pool_valid_by_time = {
        key: next(iter(values)) for key, values in validity_by_time.items()
    }

    time_subject_values: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    subject_values_by_kind: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (input_kind, subject_id, time_index, time_s), values in subject_time_values.items():
        value_mean = mean(values)
        time_subject_values[(input_kind, time_index, time_s)].append(value_mean)
        if pool_valid_by_time[(input_kind, time_index, time_s)]:
            subject_values_by_kind[(input_kind, subject_id)].append(value_mean)

    # 均匀基线只分配到 pooling 的有效 support；invalid 点在完整图轴上保持 0。
    n_valid_times_by_kind: dict[str, int] = defaultdict(int)
    for time_key in time_subject_values:
        input_kind, _time_index, _time_s = time_key
        if pool_valid_by_time[time_key]:
            n_valid_times_by_kind[input_kind] += 1
    subject_mass_by_kind: dict[str, list[float]] = defaultdict(list)
    for (input_kind, _subject_id), values in subject_values_by_kind.items():
        subject_mass_by_kind[input_kind].append(sum(values))
    uniform_by_kind = {
        input_kind: mean(masses) / max(1, n_valid_times_by_kind[input_kind])
        for input_kind, masses in subject_mass_by_kind.items()
        if masses
    }

    # 同时保留注意力分解项，便于区分 softmax 权重变化与 gate 抑制。
    component_summaries = {
        "raw": _attention_component_summary(rows, "attention_weight"),
        "effective": _attention_component_summary(rows, "effective_attention_weight"),
        "absolute_effective": _attention_component_summary(rows, "absolute_effective_attention_weight"),
        "gate": _attention_component_summary(rows, "attention_gate"),
    }
    summary: list[dict[str, object]] = []
    for (input_kind, time_index, time_s), values in sorted(
        time_subject_values.items(), key=lambda item: (_kind_sort_key(item[0][0]), item[0][2])
    ):
        pool_valid = pool_valid_by_time[(input_kind, time_index, time_s)]
        uniform = (
            uniform_by_kind.get(
                input_kind,
                1.0 / max(1, n_valid_times_by_kind[input_kind]),
            )
            if pool_valid
            else 0.0
        )
        value_mean = mean(values)
        component_key = (input_kind, time_index, time_s)
        raw_mean, raw_sem = component_summaries["raw"].get(component_key, ("", ""))
        effective_mean, effective_sem = component_summaries["effective"].get(
            component_key, ("", "")
        )
        gate_mean, gate_sem = component_summaries["gate"].get(component_key, ("", ""))
        summary.append(
            {
                "input_kind": input_kind,
                "time_index": time_index,
                "time_s": time_s,
                "attention_pool_valid": int(pool_valid),
                "n_subjects": len(values),
                "mean_attention": value_mean,
                "sem_attention": _sem(values),
                "mean_normalized_effective_attention": value_mean,
                "sem_normalized_effective_attention": _sem(values),
                "mean_raw_attention": raw_mean,
                "sem_raw_attention": raw_sem,
                "mean_effective_attention": effective_mean,
                "sem_effective_attention": effective_sem,
                "mean_attention_gate": gate_mean,
                "sem_attention_gate": gate_sem,
                **(
                    {
                        "mean_absolute_effective_attention": component_summaries["absolute_effective"][component_key][0],
                        "sem_absolute_effective_attention": component_summaries["absolute_effective"][component_key][1],
                    }
                    if component_key in component_summaries["absolute_effective"] else {}
                ),
                "uniform_attention": uniform,
                "relative_attention": value_mean - uniform,
            }
        )
    return summary


def _summary_by_kind(summary: Sequence[Mapping[str, object]]) -> dict[str, list[Mapping[str, object]]]:
    """将汇总行按输入类型分组，并在组内按 ``time_s`` 升序排列。"""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in summary:
        grouped[str(row.get("input_kind", "real"))].append(row)
    return {
        kind: sorted(rows, key=lambda row: float(row["time_s"]))
        for kind, rows in sorted(grouped.items(), key=lambda item: _kind_sort_key(item[0]))
    }


def _attention_curve_diagnostics(
    split: str,
    summary: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """计算每种输入的注意力曲线集中度与峰形诊断。

    曲线先按质量归一化为概率分布，再计算 CV、熵、对均匀分布 KL、峰值位置、边界距离
    和半高全宽（FWHM）。返回每个 ``split/input_kind`` 一行的诊断表。
    """
    diagnostics: list[dict[str, object]] = []
    for input_kind, all_rows in _summary_by_kind(summary).items():
        rows = [row for row in all_rows if _attention_pool_valid(row)]
        times = [float(row["time_s"]) for row in rows]
        values = [max(0.0, float(row["mean_attention"])) for row in rows]
        total = sum(values)
        if not values or total <= 0.0:
            continue
        # 诊断只关心曲线形状；总质量另存，形状指标在归一化概率上计算。
        probabilities = [value / total for value in values]
        n_times = len(probabilities)
        curve_mean = 1.0 / n_times
        variance = sum((value - curve_mean) ** 2 for value in probabilities) / n_times
        entropy = -sum(value * math.log(max(value, 1e-12)) for value in probabilities)
        support_peak_index = max(range(n_times), key=probabilities.__getitem__)
        peak_index = int(rows[support_peak_index]["time_index"])
        half_maximum = probabilities[support_peak_index] * 0.5
        left_support_index = support_peak_index
        right_support_index = support_peak_index
        while (
            left_support_index > 0
            and probabilities[left_support_index - 1] >= half_maximum
        ):
            left_support_index -= 1
        while (
            right_support_index + 1 < n_times
            and probabilities[right_support_index + 1] >= half_maximum
        ):
            right_support_index += 1
        sample_interval_s = abs(times[-1] - times[0]) / (n_times - 1) if n_times > 1 else 0.0
        all_time_indices = [int(row["time_index"]) for row in all_rows]
        physical_left = min(all_time_indices)
        physical_right = max(all_time_indices)
        physical_distance = min(
            peak_index - physical_left,
            physical_right - peak_index,
        )
        support_distance = min(
            support_peak_index,
            n_times - 1 - support_peak_index,
        )
        raw_mass_values = [
            float(row["mean_effective_attention"])
            for row in rows
            if row.get("mean_effective_attention") not in {None, ""}
        ]
        absolute_mass_values = [
            float(row["mean_absolute_effective_attention"])
            for row in rows
            if row.get("mean_absolute_effective_attention") not in {None, ""}
        ]
        # signed 模式必须先对各 trial 取绝对值，再做被试等权平均；
        # 对均值曲线取绝对值仍会丢失不同 trial 间相互抵消的贡献。
        mass_values = absolute_mass_values if absolute_mass_values else raw_mass_values
        diagnostics.append(
            {
                "split": split,
                "input_kind": input_kind,
                "n_times": n_times,
                "n_total_times": len(all_rows),
                "curve_mass_before_normalization": total,
                "mean_effective_attention_mass": sum(mass_values) if mass_values else "",
                **(
                    {
                        "mean_signed_effective_attention_mass": sum(raw_mass_values),
                        "attention_mass_basis": "absolute",
                    }
                    if absolute_mass_values else {}
                ),
                "attention_cv": math.sqrt(variance) / curve_mean,
                "attention_entropy": entropy,
                "attention_entropy_normalized": entropy / math.log(n_times) if n_times > 1 else 0.0,
                "attention_kl_to_uniform": max(0.0, math.log(n_times) - entropy) if n_times > 1 else 0.0,
                "attention_peak_index": peak_index,
                "attention_peak_time_s": times[support_peak_index],
                "attention_peak_weight": probabilities[support_peak_index],
                "attention_peak_at_boundary": int(
                    peak_index in {physical_left, physical_right}
                ),
                "attention_peak_distance_to_boundary_samples": physical_distance,
                "attention_peak_distance_to_boundary_s": (
                    physical_distance * sample_interval_s
                ),
                "attention_peak_at_pool_support_boundary": int(
                    support_peak_index in {0, n_times - 1}
                ),
                "attention_peak_distance_to_pool_support_samples": support_distance,
                "attention_peak_distance_to_pool_support_s": (
                    support_distance * sample_interval_s
                ),
                "attention_fwhm_s": (
                    times[right_support_index]
                    - times[left_support_index]
                    if n_times > 1
                    else 0.0
                ),
                "attention_fwhm_left_time_s": times[left_support_index],
                "attention_fwhm_right_time_s": times[right_support_index],
                "attention_fwhm_touches_boundary": int(
                    int(rows[left_support_index]["time_index"]) == physical_left
                    or int(rows[right_support_index]["time_index"]) == physical_right
                ),
                "attention_fwhm_touches_pool_support_boundary": int(
                    left_support_index == 0
                    or right_support_index == n_times - 1
                ),
            }
        )
    return diagnostics


def _attention_subject_curves(
    rows: Sequence[Mapping[str, str]], input_kind: str = "real"
) -> dict[int, list[tuple[float, float]]]:
    """按被试生成指定输入类型的平均注意力时间曲线。

    返回映射的值为按时间排序的 ``(time_s, mean_weight)`` 列表。
    """
    subject_time_values: dict[tuple[int, float], list[float]] = defaultdict(list)
    for row in rows:
        if _input_kind(row) != input_kind:
            continue
        subject_time_values[(_int(row, "subject_id"), _float(row, "time_s"))].append(
            _attention_plot_value(row)
        )
    curves: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for (subject_id, time_s), values in subject_time_values.items():
        curves[subject_id].append((time_s, mean(values)))
    return {subject_id: sorted(values) for subject_id, values in curves.items()}


def _branch_subject_values(
    rows: Sequence[Mapping[str, str]], branch: str, input_kind: str
) -> dict[int, dict[str, list[float]]]:
    """收集指定 trial-level 分支和输入类型的逐被试指标。

    返回每位被试的 accuracy、logit/evidence margin 和 probability margin 原始列表；
    预测值为负数表示该分支未启用，会从统计中跳过。
    """
    correct_key = f"{branch}_correct"
    evidence_key = f"{branch}_evidence_margin"
    probability_key = f"{branch}_probability_margin"
    prediction_key = f"{branch}_prediction"
    values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"accuracy": [], "evidence_margin": [], "probability_margin": []}
    )
    for row in rows:
        if _input_kind(row) != input_kind:
            continue
        if prediction_key in row and _int(row, prediction_key, -1) < 0:
            continue
        subject_id = _int(row, "subject_id")
        values[subject_id]["accuracy"].append(float(_int(row, correct_key)))
        values[subject_id]["evidence_margin"].append(_float(row, evidence_key))
        values[subject_id]["probability_margin"].append(_float(row, probability_key))
    return values


def _chance(rows: Sequence[Mapping[str, str]]) -> float:
    """推断分类随机水平，优先级为显式类别数、任务定义、观测标签数。

    ``no_class`` 是 time-level 额外输出，不改变真实任务类别数和机会水平。
    """
    for row in rows:
        class_count = _int(row, "class_count", 0)
        if class_count > 0:
            return 1.0 / class_count
    tasks = {str(row.get("task", "")) for row in rows if row.get("task")}
    if len(tasks) == 1:
        task = next(iter(tasks))
        if task in TASK_CLASS_COUNTS:
            return 1.0 / TASK_CLASS_COUNTS[task]
    labels = {_int(row, "target") for row in rows if "target" in row}
    return 1.0 / max(1, len(labels))


def _input_kinds(rows: Sequence[Mapping[str, str]]) -> list[str]:
    """提取并按固定顺序返回 CSV 中出现的输入类型。"""
    kinds = {_input_kind(row) for row in rows}
    return sorted(kinds, key=_kind_sort_key)


def _trial_summary(split: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """生成各 trial-level 分支的被试等权汇总统计。

    自动识别 pooled、fusion 和历史 mean_fusion 列；每个输入类型/分支输出准确率、
    evidence margin、probability margin 的跨被试均值与 SEM，以及真实类别机会水平。
    accuracy_mean 是被试等权（macro）平均；accuracy_micro 是全部 trial 混合计算
    的 correct/total，两者仅在被试 trial 数完全一致时才相等。
    """
    chance = _chance(rows)
    summary: list[dict[str, object]] = []
    # 仅统计 CSV 中真实存在的分支，兼容默认关闭 fusion 的最新配置与历史导出。
    branches = [
        branch
        for branch in ("pooled", "fusion", "mean_fusion")
        if any(f"{branch}_prediction" in row for row in rows)
    ]
    for input_kind in _input_kinds(rows):
        for branch in branches:
            subject_values = _branch_subject_values(rows, branch, input_kind)
            if not subject_values:
                continue
            accuracy_pool = [
                value["accuracy"] for value in subject_values.values() if value["accuracy"]
            ]
            subject_accuracy = [mean(values) for values in accuracy_pool]
            subject_evidence = [
                mean(value["evidence_margin"])
                for value in subject_values.values()
                if value["evidence_margin"]
            ]
            subject_probability = [
                mean(value["probability_margin"])
                for value in subject_values.values()
                if value["probability_margin"]
            ]
            if not subject_accuracy:
                continue
            # micro：所有 trial 混合的 correct/total，trial 等权而非被试等权。
            micro_correct = sum(sum(values) for values in accuracy_pool)
            micro_total = sum(len(values) for values in accuracy_pool)
            micro_accuracy = (
                micro_correct / micro_total if micro_total else float("nan")
            )
            summary.append(
                {
                    "split": split,
                    "input_kind": input_kind,
                    "branch": branch,
                    "n_subjects": len(subject_accuracy),
                    "chance_accuracy": chance,
                    "accuracy_mean": mean(subject_accuracy),
                    "accuracy_sem": _sem(subject_accuracy),
                    "accuracy_micro": micro_accuracy,
                    "evidence_margin_mean": mean(subject_evidence) if subject_evidence else 0.0,
                    "evidence_margin_sem": _sem(subject_evidence),
                    "probability_margin_mean": mean(subject_probability) if subject_probability else 0.0,
                    "probability_margin_sem": _sem(subject_probability),
                }
            )
    return summary


def _time_accuracy_summary(split: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """生成逐时间点准确率、证据强度和 ``no_class`` 预测率的被试等权汇总。

    新 schema 使用整数 ``correct_count/total_count`` 做精确聚合，历史仅有 accuracy 的行
    通过固定比例伪计数兼容。输出同时保留被试宏均值、SEM 和总体 micro 指标；预测为
    ``no_class`` 已在上游记作错误，这里只额外统计其出现率。若导出行含
    ``evidence_margin_*`` / ``probability_margin_*``，同步汇总真实类证据强度。
    """
    subject_time_counts: dict[tuple[str, int, int, float], list[int]] = defaultdict(lambda: [0, 0])
    subject_time_no_class_counts: dict[
        tuple[str, int, int, float], list[int]
    ] = defaultdict(lambda: [0, 0])
    # [evidence_sum, probability_sum, evidence_weight, probability_weight]。
    # weight 与 accuracy 的 total 对齐；两类 margin 可独立缺省。
    subject_time_margin_sums: dict[tuple[str, int, int, float], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0]
    )
    margin_keys: set[tuple[str, int, int, float]] = set()
    no_class_keys: set[tuple[str, int, int, float]] = set()
    for row in rows:
        key = (_input_kind(row), _int(row, "subject_id"), _int(row, "time_index"), _float(row, "time_s"))
        if "correct_count" in row and "total_count" in row:
            subject_time_counts[key][0] += _int(row, "correct_count")
            subject_time_counts[key][1] += _int(row, "total_count")
        else:
            # 历史 CSV 没有计数时用等比例大伪计数，保持不同导出行仍可加权汇总。
            subject_time_counts[key][0] += int(round(_float(row, "accuracy") * 1000000.0))
            subject_time_counts[key][1] += 1000000
        total = _int(row, "total_count", 0)
        has_evidence_sum = row.get("evidence_margin_sum") not in {None, ""}
        has_evidence_mean = row.get("evidence_margin_mean") not in {None, ""}
        has_probability_sum = row.get("probability_margin_sum") not in {None, ""}
        has_probability_mean = row.get("probability_margin_mean") not in {None, ""}
        has_any_margin = (
            has_evidence_sum
            or has_evidence_mean
            or has_probability_sum
            or has_probability_mean
        )
        if has_any_margin:
            # 缺哪一类 margin 就只累加哪一类；禁止用 0 填补缺失字段，否则
            # “只有 probability 的行”会把 evidence 均值错误地拉向 0。
            weight = float(total) if total > 0 else 1.0
            evidence_sum = 0.0
            probability_sum = 0.0
            evidence_weight = 0.0
            probability_weight = 0.0
            if has_evidence_sum:
                evidence_sum = _float(row, "evidence_margin_sum")
                evidence_weight = weight
            elif has_evidence_mean:
                evidence_sum = _float(row, "evidence_margin_mean") * weight
                evidence_weight = weight
            if has_probability_sum:
                probability_sum = _float(row, "probability_margin_sum")
                probability_weight = weight
            elif has_probability_mean:
                probability_sum = _float(row, "probability_margin_mean") * weight
                probability_weight = weight
            if evidence_weight > 0.0 or probability_weight > 0.0:
                margin_keys.add(key)
                subject_time_margin_sums[key][0] += evidence_sum
                subject_time_margin_sums[key][1] += probability_sum
                # 索引 2/3 分别是 evidence/probability 的有效权重（可不同）。
                # 为兼容后续 micro 汇总，同时累计两者；分字段归一时各自使用。
                subject_time_margin_sums[key][2] += evidence_weight
                subject_time_margin_sums[key][3] += probability_weight
        if row.get("no_class_prediction_count") not in {None, ""}:
            no_class_keys.add(key)
            subject_time_no_class_counts[key][0] += _int(
                row, "no_class_prediction_count"
            )
            subject_time_no_class_counts[key][1] += total
        elif row.get("no_class_prediction_rate") not in {None, ""} and total > 0:
            # schema v2 优先使用整数计数；该分支兼容只有 rate 的手写或中间 CSV。
            no_class_keys.add(key)
            subject_time_no_class_counts[key][0] += int(
                round(_float(row, "no_class_prediction_rate") * total)
            )
            subject_time_no_class_counts[key][1] += total

    time_subject_values: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    time_totals: dict[tuple[str, int, float], list[int]] = defaultdict(lambda: [0, 0])
    time_no_class_values: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    time_no_class_totals: dict[tuple[str, int, float], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    time_subject_evidence: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    time_subject_probability: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    # [evidence_sum, probability_sum, evidence_weight, probability_weight]
    time_margin_totals: dict[tuple[str, int, float], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0]
    )
    for (input_kind, _subject_id, time_index, time_s), (correct, total) in subject_time_counts.items():
        if total <= 0:
            continue
        key = (input_kind, time_index, time_s)
        time_subject_values[key].append(correct / total)
        time_totals[key][0] += correct
        time_totals[key][1] += total
        subject_key = (input_kind, _subject_id, time_index, time_s)
        if subject_key in no_class_keys:
            no_class_count, no_class_total = subject_time_no_class_counts[subject_key]
            if no_class_total > 0:
                time_no_class_values[key].append(no_class_count / no_class_total)
                time_no_class_totals[key][0] += no_class_count
                time_no_class_totals[key][1] += no_class_total
        if subject_key in margin_keys:
            (
                evidence_sum,
                probability_sum,
                evidence_weight,
                probability_weight,
            ) = subject_time_margin_sums[subject_key]
            if evidence_weight > 0.0:
                time_subject_evidence[key].append(evidence_sum / evidence_weight)
                time_margin_totals[key][0] += evidence_sum
                time_margin_totals[key][2] += evidence_weight
            if probability_weight > 0.0:
                time_subject_probability[key].append(probability_sum / probability_weight)
                time_margin_totals[key][1] += probability_sum
                time_margin_totals[key][3] += probability_weight

    chance = _chance(rows)
    summary: list[dict[str, object]] = []
    for (input_kind, time_index, time_s), values in sorted(
        time_subject_values.items(), key=lambda item: (_kind_sort_key(item[0][0]), item[0][2])
    ):
        correct, total = time_totals[(input_kind, time_index, time_s)]
        key = (input_kind, time_index, time_s)
        no_class_values = time_no_class_values.get(key, [])
        no_class_count, no_class_total = time_no_class_totals.get(key, [0, 0])
        no_class_rate = (
            no_class_count / no_class_total if no_class_total > 0 else ""
        )
        evidence_values = time_subject_evidence.get(key, [])
        probability_values = time_subject_probability.get(key, [])
        (
            evidence_sum,
            probability_sum,
            evidence_weight,
            probability_weight,
        ) = time_margin_totals.get(key, [0.0, 0.0, 0.0, 0.0])
        summary.append(
            {
                "split": split,
                "input_kind": input_kind,
                "time_index": time_index,
                "time_s": time_s,
                "n_subjects": len(values),
                "correct_count": correct,
                "total_count": total,
                "chance_accuracy": chance,
                "accuracy_mean": mean(values),
                "accuracy_sem": _sem(values),
                "accuracy_micro": correct / max(1, total),
                "evidence_margin_mean": (
                    mean(evidence_values) if evidence_values else ""
                ),
                "evidence_margin_sem": (
                    _sem(evidence_values) if evidence_values else ""
                ),
                "evidence_margin_micro": (
                    evidence_sum / evidence_weight if evidence_weight > 0.0 else ""
                ),
                "probability_margin_mean": (
                    mean(probability_values) if probability_values else ""
                ),
                "probability_margin_sem": (
                    _sem(probability_values) if probability_values else ""
                ),
                "probability_margin_micro": (
                    probability_sum / probability_weight
                    if probability_weight > 0.0
                    else ""
                ),
                "no_class_prediction_count": (
                    no_class_count if no_class_total > 0 else ""
                ),
                "no_class_prediction_rate": no_class_rate,
                "no_class_prediction_rate_mean": (
                    mean(no_class_values) if no_class_values else ""
                ),
                "no_class_prediction_rate_sem": (
                    _sem(no_class_values) if no_class_values else ""
                ),
                "no_class_prediction_rate_micro": no_class_rate,
            }
        )
    return summary


def _time_accuracy_summary_for_plot(
    split: str,
    rows: Sequence[Mapping[str, str]],
    cell_summary_rows: Sequence[Mapping[str, str]] = (),
) -> list[dict[str, object]]:
    """优先生成 subject×cell 条件平衡准确率曲线，并保留原始准确率列。

    ``time_accuracy.csv`` 的原始准确率仍作为兼容输入和诊断量。只有每个待绘制的
    input-kind/time 点都有 ``time_cell_summary.csv`` 行时，主曲线才改用被试等权的
    ``subject_cell_balanced_accuracy``；否则整张图回退到 raw accuracy，避免混用指标。
    条件指标没有样本级 micro accuracy，兼容字段 ``accuracy_micro`` 使用 complete-cell
    等权均值，并由 ``display_metric`` 和 ``aggregation_unit`` 明示统计语义。
    """
    raw_summary = _time_accuracy_summary(split, rows)
    if not raw_summary:
        return []

    conditional_by_key: dict[tuple[str, int, float], Mapping[str, str]] = {}
    for row in cell_summary_rows:
        if row.get("subject_cell_balanced_accuracy") in {None, ""}:
            continue
        if _int(row, "subject_cell_count", 0) <= 0:
            continue
        key = (
            _input_kind(row),
            _int(row, "time_index"),
            _float(row, "time_s"),
        )
        if key in conditional_by_key:
            raise ValueError(f"duplicate time cell summary row for {key}")
        conditional_by_key[key] = row

    raw_keys = {
        (
            str(row["input_kind"]),
            int(row["time_index"]),
            float(row["time_s"]),
        )
        for row in raw_summary
    }
    use_conditional = bool(raw_keys) and raw_keys.issubset(conditional_by_key)
    merged: list[dict[str, object]] = []
    for row in raw_summary:
        key = (
            str(row["input_kind"]),
            int(row["time_index"]),
            float(row["time_s"]),
        )
        raw_fields = {
            "raw_n_subjects": row["n_subjects"],
            "raw_correct_count": row["correct_count"],
            "raw_total_count": row["total_count"],
            "raw_accuracy_mean": row["accuracy_mean"],
            "raw_accuracy_sem": row["accuracy_sem"],
            "raw_accuracy_micro": row["accuracy_micro"],
        }
        if not use_conditional:
            merged.append(
                {
                    **row,
                    "display_metric": "raw_accuracy",
                    "aggregation_unit": "subject_equal_raw_accuracy",
                    "conditional_cell_balanced_accuracy": "",
                    "conditional_cell_balanced_accuracy_sem": "",
                    "conditional_cell_count": "",
                    "conditional_subject_cell_balanced_accuracy": "",
                    "conditional_subject_cell_balanced_accuracy_sem": "",
                    "conditional_subject_cell_count": "",
                    "conditional_incomplete_cell_count": "",
                    **raw_fields,
                }
            )
            continue

        conditional = conditional_by_key[key]
        subject_score = _float(conditional, "subject_cell_balanced_accuracy")
        subject_sem = _float(conditional, "subject_cell_balanced_accuracy_sem")
        cell_score = _float(conditional, "cell_balanced_accuracy")
        merged.append(
            {
                **row,
                "n_subjects": _int(conditional, "subject_cell_count"),
                "accuracy_mean": subject_score,
                "accuracy_sem": subject_sem,
                "accuracy_micro": cell_score,
                "display_metric": "subject_cell_balanced_accuracy",
                "aggregation_unit": "subject_equal_mean_of_complete_cells",
                "conditional_cell_balanced_accuracy": cell_score,
                "conditional_cell_balanced_accuracy_sem": _float(
                    conditional, "cell_balanced_accuracy_sem"
                ),
                "conditional_cell_count": _int(conditional, "cell_count"),
                "conditional_subject_cell_balanced_accuracy": subject_score,
                "conditional_subject_cell_balanced_accuracy_sem": subject_sem,
                "conditional_subject_cell_count": _int(
                    conditional, "subject_cell_count"
                ),
                "conditional_incomplete_cell_count": _int(
                    conditional, "incomplete_cell_count"
                ),
                **raw_fields,
            }
        )
    return merged


def _activation_metadata_by_kind(
    rows: Sequence[Mapping[str, str]],
) -> dict[str, tuple[str, str, str]]:
    """解析并严格校验各输入类型的激活分数语义。

    返回值为 ``input_kind -> (activation_definition, time_score_mode, schema_kind)``。
    历史行缺少元数据时按 ``max_all_class_score + sigmoid`` 解释；同一输入类型混用 schema，
    或同一图轴混用不同定义/softmax/sigmoid 时抛出异常，避免生成语义错误的曲线。
    """
    specifications: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for row in rows:
        definition_value = row.get("activation_definition")
        score_mode_value = row.get("time_score_mode")
        has_definition = definition_value not in {None, ""}
        has_score_mode = score_mode_value not in {None, ""}
        if has_definition != has_score_mode:
            raise ValueError(
                "activation rows must provide activation_definition and "
                "time_score_mode together"
            )
        # 新 schema 必须同时声明“对哪些类别取最大值”和“logit 如何转成概率”。
        if has_definition:
            definition = (
                str(definition_value).strip().lower().replace("-", "_")
            )
            score_mode = str(score_mode_value).strip().lower()
            schema_kind = "declared"
        else:
            definition = LEGACY_ACTIVATION_DEFINITION
            score_mode = LEGACY_TIME_SCORE_MODE
            schema_kind = "legacy"
        if score_mode not in {"sigmoid", "softmax"}:
            raise ValueError(f"Unsupported activation time_score_mode: {score_mode!r}")
        specifications[_input_kind(row)].add(
            (definition, score_mode, schema_kind)
        )

    resolved: dict[str, tuple[str, str, str]] = {}
    for input_kind, kind_specifications in specifications.items():
        if len(kind_specifications) != 1:
            raise ValueError(
                "Activation rows for one input kind contain mixed schema, "
                "activation_definition, or time_score_mode values: "
                f"{input_kind}={sorted(kind_specifications)!r}"
            )
        resolved[input_kind] = next(iter(kind_specifications))
    # 不同输入类型会画在同一轴上，因此其数值定义也必须完全相同。
    axis_specifications = set(resolved.values())
    if len(axis_specifications) > 1:
        raise ValueError(
            "Activation curves cannot share an axis with mixed legacy/schema, "
            "activation_definition, or time_score_mode values: "
            f"{resolved!r}"
        )
    return resolved


def _activation_summary(split: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    """按被试等权汇总逐时间激活分数与超过阈值的比例。

    兼容新 schema 的 sum/count 字段和历史 mean/rate 字段；输出包含均值、SEM、阈值，
    以及用于审计坐标含义的激活定义和 score mode。
    """
    subject_scores: dict[tuple[str, int, int, float], list[float]] = defaultdict(list)
    subject_rates: dict[tuple[str, int, int, float], list[float]] = defaultdict(list)
    thresholds: list[float] = []
    metadata_by_kind = _activation_metadata_by_kind(rows)
    for row in rows:
        key = (_input_kind(row), _int(row, "subject_id"), _int(row, "time_index"), _float(row, "time_s"))
        total = max(1, _int(row, "total_count", 1))
        # 新 schema 保存可精确合并的总和与计数；旧 schema 只能使用已计算均值。
        score = (
            _float(row, "activation_score_sum") / total
            if "activation_score_sum" in row
            else _float(row, "activation_score_mean")
        )
        rate = (
            _int(row, "active_count") / total
            if "active_count" in row
            else _float(row, "activation_rate")
        )
        subject_scores[key].append(score)
        subject_rates[key].append(rate)
        if row.get("activation_threshold") not in {None, ""}:
            thresholds.append(_float(row, "activation_threshold"))

    time_scores: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    time_rates: dict[tuple[str, int, float], list[float]] = defaultdict(list)
    for (input_kind, _subject_id, time_index, time_s), values in subject_scores.items():
        key = (input_kind, time_index, time_s)
        time_scores[key].append(mean(values))
        time_rates[key].append(mean(subject_rates[(input_kind, _subject_id, time_index, time_s)]))

    threshold = thresholds[0] if thresholds else 0.5
    summary: list[dict[str, object]] = []
    for (input_kind, time_index, time_s), scores in sorted(
        time_scores.items(), key=lambda item: (_kind_sort_key(item[0][0]), item[0][2])
    ):
        rates = time_rates[(input_kind, time_index, time_s)]
        activation_definition, time_score_mode, _schema_kind = metadata_by_kind[
            input_kind
        ]
        summary.append(
            {
                "split": split,
                "input_kind": input_kind,
                "time_index": time_index,
                "time_s": time_s,
                "n_subjects": len(scores),
                "activation_threshold": threshold,
                "activation_score_mean": mean(scores),
                "activation_score_sem": _sem(scores),
                "activation_rate_mean": mean(rates),
                "activation_rate_sem": _sem(rates),
                "activation_definition": activation_definition,
                "time_score_mode": time_score_mode,
            }
        )
    return summary


def _activation_ylabel(rows: Sequence[Mapping[str, str]]) -> str:
    """根据激活定义与 score mode 生成不含歧义的 y 轴标签。

    当前支持只在真实类别上取最大值的新定义，以及历史对全部输出取最大值的定义。
    """
    metadata_by_kind = _activation_metadata_by_kind(rows)
    if not metadata_by_kind:
        return "Activation probability"
    definition, score_mode, _schema_kind = next(iter(metadata_by_kind.values()))
    if definition == "max_real_class_score":
        return "Max real-class probability"
    if definition != LEGACY_ACTIVATION_DEFINITION:
        raise ValueError(
            f"Unsupported activation_definition: {definition!r}"
        )
    if score_mode == "softmax":
        return "Max class probability"
    if score_mode == "sigmoid":
        return "Max sigmoid probability"
    raise ValueError(f"Unsupported activation time_score_mode: {score_mode!r}")


_STIMULUS_COLORS = {
    "1": "#4575b4",
    "2": "#74add1",
    "3": "#f7c948",
    "4": "#f46d43",
    "5": "#d73027",
}


def _stimulus_attention_trial_key(row: Mapping[str, str]) -> tuple[str, str, str, str]:
    """Identify one trial; a record can contain every stimulus condition."""
    fields = ("record_id", "subject_id", "trial0", "lag")
    values = tuple(str(row.get(field, "")).strip() for field in fields)
    if any(value in {"", "None"} for value in values):
        raise ValueError(f"Stimulus attention requires a complete trial identity {fields}: {values}")
    return values


def _attention_by_current_stimulus_summary(
    split: str,
    attention_rows: Iterable[Mapping[str, str]],
    trial_rows: Iterable[Mapping[str, str]],
    *,
    input_kind: str = "real",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Join individual trials, then calculate trial-weighted means at each time.

    The SEM describes trials, rather than subject means. Attention rows may be
    streamed; online moments avoid retaining the multi-million-row CSV in memory.
    """
    trial_to_stim: dict[tuple[str, str, str, str], str] = {}
    for row in trial_rows:
        if _input_kind(row) != input_kind or str(row.get("split", split)) != split:
            continue
        stim = str(row.get("current_stimulus", "") or "").strip()
        if not stim:
            continue
        key = _stimulus_attention_trial_key(row)
        if key in trial_to_stim:
            detail = "Conflicting current_stimulus" if trial_to_stim[key] != stim else "Duplicate trial metadata"
            raise ValueError(f"{detail} for stimulus attention trial {key}")
        trial_to_stim[key] = stim
    if not trial_to_stim:
        return [], []

    moments: dict[tuple[str, int], list[float]] = {}
    totals: dict[int, list[float]] = {}
    gaussian: dict[int, list[float]] = {}
    times: dict[int, float] = {}
    gaussian_times: dict[int, float] = {}
    subjects: dict[tuple[str, int], set[str]] = {}
    # One compact bitset per trial detects duplicate rows without storing every row.
    seen: dict[tuple[str, tuple[str, str, str, str]], int] = {}

    def update(table: dict, key: object, value: float) -> None:
        state = table.setdefault(key, [0, 0.0, 0.0])
        state[0] += 1
        delta = value - state[1]
        state[1] += delta / state[0]
        state[2] += delta * (value - state[1])

    validity_by_kind_time: dict[tuple[str, int], bool] = {}
    for row in attention_rows:
        kind = _input_kind(row)
        primary = kind == input_kind
        if (not primary and not (kind == "gaussian" and input_kind != "gaussian")) or str(row.get("split", split)) != split:
            continue
        key = _stimulus_attention_trial_key(row)
        if primary and key not in trial_to_stim:
            raise ValueError(f"No current_stimulus metadata for attention trial {key}")
        try:
            value = _attention_plot_value(row)
            index_value = float(row["time_index"])
            time_s = float(row["time_s"])
        except (KeyError, ValueError, TypeError):
            continue
        if not all(math.isfinite(value) for value in (value, index_value, time_s)):
            continue
        ti = int(index_value)
        if ti < 0 or ti != index_value:
            raise ValueError(f"Invalid attention time_index: {index_value}")
        # Exported time axes are short; reject unreasonable indices before bit shifting.
        if ti > 1_000_000:
            raise ValueError(f"Unreasonable attention time_index: {ti}")
        identity = kind, key
        bit = 1 << ti
        previous = seen.get(identity, 0)
        if previous & bit:
            raise ValueError(f"Duplicate attention row for trial {key}, time_index={ti}")
        seen[identity] = previous | bit
        time_map = times if primary else gaussian_times
        if ti in time_map and not math.isclose(time_map[ti], time_s, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Inconsistent attention time_s for time_index={ti}")
        time_map[ti] = time_s
        valid = _attention_pool_valid(row)
        if validity_by_kind_time.setdefault((kind, ti), valid) != valid:
            raise ValueError("attention_pool_valid must be constant across trials for each input/time")
        if primary:
            stim = trial_to_stim[key]
            update(moments, (stim, ti), value)
            update(totals, ti, value)
            subjects.setdefault((stim, ti), set()).add(key[1])
        else:
            update(gaussian, ti, value)

    summary: list[dict[str, object]] = []
    for (stim, ti), (count, average, squared_deviation) in sorted(moments.items()):
        sem = math.sqrt(max(0.0, squared_deviation) / (count * (count - 1))) if count > 1 else 0.0
        summary.append(dict(split=split, input_kind=input_kind, current_stimulus=stim,
            time_index=ti, time_s=times[ti], attention_mean=average, attention_sem=sem,
            trial_count=int(count), subject_count=len(subjects[stim, ti]),
            aggregation="trial_mean", sem_unit="trial",
            attention_pool_valid=int(validity_by_kind_time[input_kind, ti])))
    reference: list[dict[str, object]] = []
    for ti, (count, average, _) in sorted(totals.items()):
        if ti in gaussian_times and not math.isclose(times[ti], gaussian_times[ti], rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Real/control attention time axes differ at time_index={ti}")
        control = gaussian.get(ti)
        reference.append(dict(split=split, input_kind=input_kind, time_index=ti, time_s=times[ti],
            attention_mean=average, trial_count=int(count),
            gaussian_attention_mean=control[1] if control else None,
            gaussian_trial_count=int(control[0]) if control else 0, aggregation="trial_mean",
            attention_pool_valid=int(validity_by_kind_time[input_kind, ti])))
    return summary, reference
