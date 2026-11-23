"""EEG 时间片段噪声增强，以及噪声覆盖范围到潜变量感受野的映射。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
from torch import nn


NOISE_SUPERVISION_MODES = {"rf_aware", "all_positions"}
_SINGLE_NOISE_REPLACEMENT_MODES = {
    "sample_channel", "batch_channel", "standard_normal",
    "time_shuffle", "time_shuffle_independent",
}
NOISE_REPLACEMENT_MODES = _SINGLE_NOISE_REPLACEMENT_MODES | {"mixture"}


def normalize_noise_mode_probabilities(
    noise_mode: str, probabilities: object = None,
) -> dict[str, float] | None:
    """混合概率是增强后选用处理方式的条件概率，与增强命中率分开。"""
    if noise_mode not in NOISE_REPLACEMENT_MODES:
        raise ValueError(f"noise_mode must be one of: {', '.join(sorted(NOISE_REPLACEMENT_MODES))}")
    if noise_mode != "mixture":
        if probabilities is not None:
            raise ValueError("noise_mode_probabilities requires noise_mode=mixture")
        return None
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("noise_mode_probabilities must be a non-empty mapping for noise_mode=mixture")
    unknown = set(probabilities) - _SINGLE_NOISE_REPLACEMENT_MODES
    if unknown:
        raise ValueError(f"Unknown noise_mode_probabilities modes: {sorted(map(str, unknown))}")
    result = {}
    # 固定顺序，YAML mapping 的键顺序不影响随机抽样或可复现续训。
    for mode in sorted(probabilities):
        raw = probabilities[mode]
        try:
            probability = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("noise_mode_probabilities values must be finite probabilities") from exc
        if isinstance(raw, bool) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("noise_mode_probabilities values must be finite probabilities in [0, 1]")
        result[mode] = probability
    if not math.isclose(math.fsum(result.values()), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("noise_mode_probabilities must sum to 1")
    return result


def normalize_noise_supervision_mode(value: object) -> str:
    """规范化噪声监督策略名称，并兼容旧配置别名。

    rf_aware 会由训练循环排除受噪声感受野影响的位置；all_positions 则把
    噪声视为一种合法输入，所有位置仍可参与监督。增强器本身不执行屏蔽。
    """
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "masked": "rf_aware",
        "compatible": "all_positions",
        "robust": "all_positions",
        "unmasked": "all_positions",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in NOISE_SUPERVISION_MODES:
        valid = ", ".join(sorted(NOISE_SUPERVISION_MODES))
        raise ValueError(f"temporal_crop_noise.supervision_mode must be one of: {valid}")
    return normalized


@dataclass
class TemporalCropNoiseResult:
    """增强后的 EEG、时间覆盖掩码及可选的近全噪声 trial 标记。"""

    eeg: torch.Tensor
    # 与输入时间点逐点对齐；这里只表示原始输入未被直接替换。
    valid_time_mask: torch.Tensor
    # True 表示该原始 EEG 采样点位于替换/时间打乱区域。
    covered_time_mask: torch.Tensor
    # [B]；仅这些 trial 跳过 pooled/fusion 分类；主 time 可独立缩放，mask 保留。
    near_full_noise_mask: torch.Tensor | None = None
    # Hierarchical-only metadata; roll preserves real labels and has no covered positions.
    roll_mask: torch.Tensor | None = None
    # Signed shifts: negative=left, positive=right; zero for every non-roll trial.
    roll_shifts: torch.Tensor | None = None


def normalize_augmentation_pipeline_config(config: object) -> dict[str, object]:
    """Optional hierarchical routing; missing/disabled retains the legacy path."""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("train.augmentation_pipeline must be a mapping")
    allowed = {"enabled", "roll_probability", "roll_max_shift_s", "roll_left_probability"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown train.augmentation_pipeline options: {sorted(map(str, unknown))}")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("train.augmentation_pipeline.enabled must be a boolean")
    result: dict[str, object] = {"enabled": enabled}
    for name, default in (("roll_probability", .2), ("roll_max_shift_s", 1.), ("roll_left_probability", .5)):
        raw = config.get(name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"train.augmentation_pipeline.{name} must be finite") from exc
        if isinstance(raw, bool) or not math.isfinite(value):
            raise ValueError(f"train.augmentation_pipeline.{name} must be finite")
        if name == "roll_max_shift_s":
            if value <= 0.:
                raise ValueError("train.augmentation_pipeline.roll_max_shift_s must be positive")
        elif not 0. <= value <= 1.:
            raise ValueError(f"train.augmentation_pipeline.{name} must be in [0, 1]")
        result[name] = value
    return result


def normalize_near_full_noise_config(config: object) -> dict[str, object]:
    """解析独立开关；旧配置未提供此字段时关闭，不改变随机数序列。"""
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("train.near_full_noise must be a mapping")
    unknown = set(config) - {"enabled", "probability", "coverage_ratio", "time_loss_scale"}
    if unknown:
        raise ValueError(f"Unknown train.near_full_noise options: {sorted(map(str, unknown))}")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("train.near_full_noise.enabled must be a boolean")
    result: dict[str, object] = {"enabled": enabled}
    for name, default in (
        ("probability", 0.1), ("coverage_ratio", 0.95), ("time_loss_scale", 1.0),
    ):
        raw = config.get(name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"train.near_full_noise.{name} must be a finite number") from exc
        if isinstance(raw, bool) or not math.isfinite(value):
            raise ValueError(f"train.near_full_noise.{name} must be a finite number")
        if name == "probability" and not 0.0 <= value <= 1.0:
            raise ValueError("train.near_full_noise.probability must be in [0, 1]")
        if name == "coverage_ratio" and not 0.0 < value <= 1.0:
            raise ValueError("train.near_full_noise.coverage_ratio must be in (0, 1]")
        if name == "time_loss_scale" and value < 0.0:
            raise ValueError("train.near_full_noise.time_loss_scale must be non-negative")
        result[name] = value
    return result


def _replace_with_noise(
    eeg: torch.Tensor,
    covered: torch.Tensor,
    *,
    noise_mode: str,
    min_noise_std: float,
    noise_mode_probabilities: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """仅改动 covered 位置，支持原高斯采样及共享/独立通道的时间置换。"""
    if noise_mode == "mixture":
        probabilities = normalize_noise_mode_probabilities(noise_mode, noise_mode_probabilities)
        assert probabilities is not None
        modes = [mode for mode, probability in probabilities.items() if probability > 0.0]
        # 单一非零候选退化为原路径，不额外消耗随机数。
        if len(modes) == 1:
            return _replace_with_noise(eeg, covered, noise_mode=modes[0], min_noise_std=min_noise_std)
        active = covered.any(dim=1)
        count = int(active.sum().item())
        if not count:
            return eeg.clone()
        weights = torch.tensor([probabilities[mode] for mode in modes], device=eeg.device, dtype=torch.float64)
        selected = torch.full((eeg.shape[0],), -1, device=eeg.device, dtype=torch.long)
        selected[active] = torch.multinomial(weights, count, replacement=True)
        augmented = eeg.clone()
        for index, mode in enumerate(modes):
            mode_covered = covered & (selected == index)[:, None]
            if not bool(mode_covered.any().item()):
                continue
            # 各路读取原始整批 EEG；batch_channel 的统计范围不能缩成该路的子批。
            candidate = _replace_with_noise(eeg, mode_covered, noise_mode=mode, min_noise_std=min_noise_std)
            augmented = torch.where(mode_covered[:, None, :], candidate, augmented)
        return augmented
    if noise_mode in {"time_shuffle", "time_shuffle_independent"}:
        # 每个 trial 单独排列；time_shuffle 完整搬运 EEG 空间向量，
        # time_shuffle_independent 则为每个通道单独采样时间排列。
        # 源和目标均限制在 covered 集合，绝不抽取未覆盖的保留片段或其他 trial。
        # 多个不连续 covered 区间合并为同一置换集合。
        batch_size, channels, n_times = eeg.shape
        permutation_channels = channels if noise_mode == "time_shuffle_independent" else 1
        random_order = torch.rand(
            (batch_size, permutation_channels, n_times), device=eeg.device, dtype=torch.float32,
        ).masked_fill(~covered[:, None, :], float("inf")).argsort(dim=2)
        covered_rank = (covered.long().cumsum(dim=1) - 1).clamp_min(0)
        source_time = random_order.gather(2, covered_rank[:, None, :].expand_as(random_order))
        original_time = torch.arange(n_times, device=eeg.device)[None, None, :]
        source_time = torch.where(covered[:, None, :], source_time, original_time)
        return eeg.gather(2, source_time.expand_as(eeg))

    time_major = eeg.transpose(1, 2)
    augmented = time_major.clone()
    selected = time_major[covered]
    if noise_mode == "standard_normal":
        noise = torch.randn_like(selected)
    else:
        covered_batch_indices = covered.nonzero(as_tuple=True)[0]
        if noise_mode == "batch_channel":
            mean = eeg.mean(dim=(0, 2))
            std = eeg.std(dim=(0, 2), unbiased=False)
            selected_mean = mean.unsqueeze(0).expand_as(selected)
            selected_std = std.unsqueeze(0).expand_as(selected)
        else:
            mean = eeg.mean(dim=2)
            std = eeg.std(dim=2, unbiased=False)
            selected_mean = mean.index_select(0, covered_batch_indices)
            selected_std = std.index_select(0, covered_batch_indices)
        noise = selected_mean + torch.randn_like(selected) * selected_std.clamp_min(min_noise_std)
    augmented[covered] = noise
    return augmented.transpose(1, 2)


@dataclass
class TemporalCoverageMasks:
    """CNN 感受野下三类互斥的潜变量位置掩码，形状均为 [B,T]。"""

    clean_time_mask: torch.Tensor
    invalid_core_time_mask: torch.Tensor
    halo_time_mask: torch.Tensor


def _same_padding(kernel_size: int, dilation: int) -> tuple[int, int]:
    """计算 stride=1 时保持时间长度不变所需的左右 padding。"""
    total = int(dilation) * (int(kernel_size) - 1)
    return total // 2, total - total // 2


def _dilate_time_mask(
    mask: torch.Tensor,
    *,
    kernel_size: int,
    dilation: int,
    padding_value: float,
) -> torch.Tensor:
    """用 max-pooling 实现布尔时间掩码的空洞膨胀。"""
    left, right = _same_padding(kernel_size, dilation)
    values = torch.nn.functional.pad(
        mask[:, None, :].to(dtype=torch.float32),
        (left, right),
        value=float(padding_value),
    )
    return torch.nn.functional.max_pool1d(
        values,
        kernel_size=int(kernel_size),
        stride=1,
        dilation=int(dilation),
    ).squeeze(1) > 0.0


def classify_temporal_coverage(
    covered_time_mask: torch.Tensor,
    receptive_field_layers: Sequence[tuple[int, int]],
) -> TemporalCoverageMasks:
    """把输入噪声覆盖映射为 clean、invalid core 与 halo 潜变量位置。

    clean 的完整感受野都来自正常 EEG；invalid core 的完整感受野都位于
    噪声覆盖区；halo 同时看到正常 EEG 与噪声。该映射只适用于显式暴露
    卷积核和 dilation 的 CNN 路径。
    """
    # covered=输入被噪声替换；向下经感受野膨胀得到 clean / invalid_core / halo
    if covered_time_mask.ndim != 2:
        raise ValueError(
            "covered_time_mask must have shape [batch, time], "
            f"got {covered_time_mask.shape}"
        )
    contaminated = covered_time_mask.to(dtype=torch.bool)  # 逐层膨胀：任一路径见噪声
    invalid_core = contaminated.clone()  # 收缩：完整感受野都在噪声里
    for kernel_size, dilation in receptive_field_layers:
        kernel_size = int(kernel_size)
        dilation = int(dilation)
        if kernel_size < 1 or dilation < 1:
            raise ValueError("receptive-field kernel sizes and dilations must be positive")
        conv_contaminated = _dilate_time_mask(
            contaminated,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_value=0.0,
        )
        contaminated = contaminated | conv_contaminated  # residual 或 conv 见噪 → 受影响

        conv_has_noncovered = _dilate_time_mask(
            ~invalid_core,
            kernel_size=kernel_size,
            dilation=dilation,
            padding_value=1.0,  # 边界 pad 视为「干净」，不把边沿误判成 invalid_core
        )
        invalid_core = invalid_core & ~conv_has_noncovered

    clean = ~contaminated  # 主 time loss（rf_aware）只信这里
    halo = contaminated & ~invalid_core  # 半脏：见噪又见净
    return TemporalCoverageMasks(
        clean_time_mask=clean,
        invalid_core_time_mask=invalid_core,
        halo_time_mask=halo,
    )


def temporal_receptive_field_layers(
    model: torch.nn.Module,
) -> tuple[tuple[int, int], ...]:
    """读取通向 time latent 的 CNN 层，返回每层 (kernel_size, dilation)。

    time_source_block 可让时间分类头从中间块读取表征，因此这里只返回它之前
    真正影响 time logits 的卷积层；窗口 Transformer 没有该结构时返回空元组。
    """
    encoder = getattr(model, "encoder", None)
    temporal_blocks = getattr(encoder, "temporal_blocks", None)
    if temporal_blocks is None:
        return ()
    blocks = list(temporal_blocks)
    source_block = getattr(model, "time_source_block", None)
    # 只累计 time 头实际读到的 block 之前的卷积（中间截断时感受野更短）
    block_count = len(blocks) if source_block is None else int(source_block)
    layers: list[tuple[int, int]] = []
    for block in blocks[:block_count]:
        convolution = getattr(block, "depthwise", None)
        if not isinstance(convolution, torch.nn.Conv1d):
            # 非 CNN 后端（如 window_transformer）无 depthwise 层；返回空元组，
            # 调用方的 `if receptive_field_layers:` 分支会退化为「输入 covered」语义。
            return ()
        kernel_size = tuple(int(value) for value in convolution.kernel_size)
        dilation = tuple(int(value) for value in convolution.dilation)
        if len(kernel_size) != 1 or len(dilation) != 1:
            raise ValueError("Time receptive fields require one-dimensional convolutions")
        layers.append((kernel_size[0], dilation[0]))
    return tuple(layers)  # window_transformer 无 depthwise → 上游会要求 all_positions


class EEGInputAugmenter(nn.Module):
    """用噪声替换或时间打乱 EEG 的一个或多个片段。

    输入/输出保持 [B,C,T] 及原时间轴不变。geometry=edge 从一侧覆盖，
    circular_block 允许单块跨越首尾，random_blocks 生成多个不重叠块；块间
    是否必须保留正常 EEG 由 min_clean_gap_s 控制。probability 按 trial
    独立采样，而不是按时间点采样。
    noise_mode=mixture 时每条被覆盖 trial 按 noise_mode_probabilities 选一种处理。

    时长字段按 geometry 独立使用：random_blocks 使用 min/max_total_duration_s，
    circular_block 使用 min/max_block_duration_s，edge 使用 min_duration_s 和
    max_left/right_duration_s；切换 geometry 时需同步设置对应字段。
    """

    def __init__(
        self,
        *,
        sample_interval_s: float,
        geometry: str = "edge",
        mode: str | None = None,
        probability: float = 1.0,
        side: str = "random",
        min_duration_s: float = 0.05,
        max_left_duration_s: float = 1.0,
        max_right_duration_s: float = 0.75,
        block_duration_s: float | None = None,
        min_block_duration_s: float | None = None,
        max_block_duration_s: float | None = None,
        min_blocks: int = 1,
        max_blocks: int = 3,
        min_total_duration_s: float | None = None,
        max_total_duration_s: float | None = None,
        min_clean_gap_s: float = 0.0,
        allow_edge_blocks: bool = True,
        noise_mode: str = "sample_channel",
        noise_mode_probabilities: Mapping[str, float] | None = None,
        min_noise_std: float = 1e-4,
    ) -> None:
        """校验秒级增强配置并保存其规范化形式。"""
        super().__init__()
        sample_interval_s = float(sample_interval_s)
        if not math.isfinite(sample_interval_s) or sample_interval_s <= 0.0:
            raise ValueError("sample_interval_s must be finite and positive")
        probability = float(probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and between 0 and 1")
        if mode is not None:
            if geometry != "edge" and geometry != mode:
                raise ValueError("geometry and mode specify different temporal cover modes")
            geometry = mode
        geometry = str(geometry).lower().replace("-", "_")
        geometry = {
            "edge_crop": "edge",
            "endpoint": "edge",
            "circular": "circular_block",
            "random_block": "random_blocks",
            "multi_block": "random_blocks",
            "multi_blocks": "random_blocks",
        }.get(geometry, geometry)
        if geometry not in {"edge", "circular_block", "random_blocks"}:
            raise ValueError("geometry must be edge, circular_block, or random_blocks")
        if side not in {"left", "right", "random"}:
            raise ValueError("side must be left, right, or random")
        edge_durations = {
            "min_duration_s": float(min_duration_s),
            "max_left_duration_s": float(max_left_duration_s),
            "max_right_duration_s": float(max_right_duration_s),
        }
        for name, value in edge_durations.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        min_duration_s = edge_durations["min_duration_s"]
        max_left_duration_s = edge_durations["max_left_duration_s"]
        max_right_duration_s = edge_durations["max_right_duration_s"]
        if geometry == "edge":
            for edge_side in ("left", "right"):
                maximum_name = f"max_{edge_side}_duration_s"
                if side in (edge_side, "random") and min_duration_s > edge_durations[maximum_name]:
                    raise ValueError(f"min_duration_s must not exceed {maximum_name}")
        if block_duration_s is not None and float(block_duration_s) < 0.0:
            raise ValueError("block_duration_s must be non-negative")
        if min_block_duration_s is not None and float(min_block_duration_s) < 0.0:
            raise ValueError("min_block_duration_s must be non-negative")
        if max_block_duration_s is not None and float(max_block_duration_s) < 0.0:
            raise ValueError("max_block_duration_s must be non-negative")
        if block_duration_s is not None and (
            min_block_duration_s is not None or max_block_duration_s is not None
        ):
            raise ValueError(
                "block_duration_s cannot be combined with min_block_duration_s or max_block_duration_s"
            )
        if block_duration_s is not None:
            min_block_duration_s = float(block_duration_s)
            max_block_duration_s = float(block_duration_s)
        if min_block_duration_s is None:
            min_block_duration_s = float(min_duration_s)
        if max_block_duration_s is None:
            max_block_duration_s = max(float(max_left_duration_s), float(max_right_duration_s))
        if float(min_block_duration_s) > float(max_block_duration_s):
            raise ValueError("min_block_duration_s must not exceed max_block_duration_s")
        min_blocks = self._positive_integer("min_blocks", min_blocks)
        max_blocks = self._positive_integer("max_blocks", max_blocks)
        if min_blocks > max_blocks:
            raise ValueError("min_blocks must not exceed max_blocks")
        if min_total_duration_s is None:
            min_total_duration_s = float(min_duration_s)
        if max_total_duration_s is None:
            max_total_duration_s = max(
                float(max_left_duration_s),
                float(max_right_duration_s),
            )
        random_block_durations = {
            "min_block_duration_s": float(min_block_duration_s),
            "max_block_duration_s": float(max_block_duration_s),
            "min_total_duration_s": float(min_total_duration_s),
            "max_total_duration_s": float(max_total_duration_s),
            "min_clean_gap_s": float(min_clean_gap_s),
        }
        for name, value in random_block_durations.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if float(min_total_duration_s) > float(max_total_duration_s):
            raise ValueError("min_total_duration_s must not exceed max_total_duration_s")
        if not isinstance(allow_edge_blocks, bool):
            raise ValueError("allow_edge_blocks must be a boolean")
        noise_mode_probabilities = normalize_noise_mode_probabilities(noise_mode, noise_mode_probabilities)
        min_noise_std = float(min_noise_std)
        if not math.isfinite(min_noise_std) or min_noise_std < 0.0:
            raise ValueError("min_noise_std must be finite and non-negative")
        self.sample_interval_s = sample_interval_s
        self.geometry = geometry
        self.probability = probability
        self.side = side
        self.min_duration_s = float(min_duration_s)
        self.max_left_duration_s = float(max_left_duration_s)
        self.max_right_duration_s = float(max_right_duration_s)
        self.min_block_duration_s = float(min_block_duration_s)
        self.max_block_duration_s = float(max_block_duration_s)
        self.min_blocks = min_blocks
        self.max_blocks = max_blocks
        self.min_total_duration_s = float(min_total_duration_s)
        self.max_total_duration_s = float(max_total_duration_s)
        self.min_clean_gap_s = float(min_clean_gap_s)
        self.allow_edge_blocks = allow_edge_blocks
        self.noise_mode = noise_mode
        self.noise_mode_probabilities = noise_mode_probabilities
        self.min_noise_std = min_noise_std

    @staticmethod
    def _positive_integer(name: str, value: object) -> int:
        """严格解析正整数，显式拒绝会被 Python 当作整数的 bool。"""
        if isinstance(value, bool):
            raise ValueError(f"{name} must be a positive integer")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 1.0:
            raise ValueError(f"{name} must be a positive integer")
        return int(numeric)

    def _duration_samples(self, duration_s: float, n_times: int) -> int:
        """把秒数四舍五入为采样点数，并裁剪到当前记录长度。"""
        return max(0, min(n_times, int(round(float(duration_s) / self.sample_interval_s))))

    def _minimum_duration_samples(self, duration_s: float) -> int:
        """向上取整秒数，确保实际长度不短于配置下界。"""
        ratio = float(duration_s) / self.sample_interval_s
        # float32 时间轴相邻差分的相对误差可达约 1e-6；吸收整数点附近的误差。
        tolerance = 1e-5 * max(1.0, abs(ratio))
        return max(0, int(math.ceil(ratio - tolerance)))

    def _maximum_duration_samples(self, duration_s: float) -> int:
        """向下取整秒数，确保实际长度不超过配置上界。"""
        ratio = float(duration_s) / self.sample_interval_s
        tolerance = 1e-5 * max(1.0, abs(ratio))
        return max(0, int(math.floor(ratio + tolerance)))

    @staticmethod
    def _sample_bounded_partition(
        totals: torch.Tensor,
        part_counts: torch.Tensor,
        *,
        max_parts: int,
        minimum: int,
        maximum: int,
    ) -> torch.Tensor:
        """把每个总长度精确拆成数量可变、上下界受限的若干块。

        返回 [B,max_parts]；超过每个 trial 实际块数的槽位为 0。块数通常很小，
        因而仅循环槽位，batch 维始终保持向量化。
        """
        batch_size = totals.shape[0]
        device = totals.device
        slots = torch.arange(max_parts, device=device).unsqueeze(0)
        active = slots < part_counts[:, None]
        order = torch.rand(batch_size, max_parts, device=device).masked_fill(
            ~active,
            -1.0,
        ).argsort(dim=1, descending=True)
        extras = torch.zeros(batch_size, max_parts, dtype=torch.long, device=device)
        remaining = totals - part_counts * int(minimum)
        capacity = int(maximum) - int(minimum)

        # 循环只覆盖很小的配置块数，所有 trial 仍在同一批张量中并行采样。
        for rank in range(max_parts):
            rank_active = rank < part_counts
            remaining_parts = (part_counts - rank - 1).clamp_min(0)
            low = (remaining - remaining_parts * capacity).clamp_min(0)
            high = torch.minimum(remaining, torch.full_like(remaining, capacity))
            interval_size = (high - low + 1).clamp_min(1)
            sampled = low + torch.floor(
                torch.rand(batch_size, device=device) * interval_size
            ).to(dtype=torch.long)
            sampled = torch.where(rank_active, sampled, torch.zeros_like(sampled))
            extras.scatter_(1, order[:, rank : rank + 1], sampled[:, None])
            remaining = remaining - sampled

        return (extras + int(minimum)) * active.to(dtype=torch.long)

    @staticmethod
    def _sample_unbounded_partition(
        totals: torch.Tensor,
        active_parts: torch.Tensor,
    ) -> torch.Tensor:
        """把非负总量精确拆给活跃槽位，每个槽位允许分到 0。"""
        batch_size, max_parts = active_parts.shape
        device = totals.device
        part_counts = active_parts.sum(dim=1)
        order = torch.rand(batch_size, max_parts, device=device).masked_fill(
            ~active_parts,
            -1.0,
        ).argsort(dim=1, descending=True)
        values = torch.zeros(batch_size, max_parts, dtype=torch.long, device=device)
        remaining = totals.clone()

        for rank in range(max_parts):
            rank_active = rank < part_counts
            remaining_parts = (part_counts - rank - 1).clamp_min(0)
            low = torch.where(remaining_parts == 0, remaining, torch.zeros_like(remaining))
            interval_size = (remaining - low + 1).clamp_min(1)
            sampled = low + torch.floor(
                torch.rand(batch_size, device=device) * interval_size
            ).to(dtype=torch.long)
            sampled = torch.where(rank_active, sampled, torch.zeros_like(sampled))
            values.scatter_(1, order[:, rank : rank + 1], sampled[:, None])
            remaining = remaining - sampled

        return values

    def _random_blocks_covered_mask(
        self,
        *,
        batch_size: int,
        n_times: int,
        device: torch.device,
        apply_augmentation: torch.Tensor,
    ) -> torch.Tensor:
        """为每个 trial 生成满足块数、总长度和干净间隔约束的 [B,T] 掩码。

        先联合检查离散采样点约束是否可行，再依次采样总覆盖长度、块数、各块
        长度与边界拓扑。这样不会因某个不可行组合而在运行中无限重采样。
        """
        if n_times < 1:
            raise ValueError("random_blocks requires at least one time point")

        min_block_samples = max(
            1,
            self._minimum_duration_samples(self.min_block_duration_s),
        )
        max_block_samples = min(
            n_times,
            self._maximum_duration_samples(self.max_block_duration_s),
        )
        min_total_samples = max(
            1,
            self._minimum_duration_samples(self.min_total_duration_s),
        )
        max_total_samples = min(
            n_times,
            self._maximum_duration_samples(self.max_total_duration_s),
        )
        # 配置为 0 时允许块直接相邻；此时布尔覆盖掩码会把它们显示为一个连通区。
        min_gap_samples = self._minimum_duration_samples(self.min_clean_gap_s)

        block_counts = torch.arange(
            self.min_blocks,
            self.max_blocks + 1,
            dtype=torch.long,
            device=device,
        )
        if min_total_samples <= max_total_samples:
            total_candidates = torch.arange(
                min_total_samples,
                max_total_samples + 1,
                dtype=torch.long,
                device=device,
            )
        else:
            total_candidates = torch.empty(0, dtype=torch.long, device=device)

        # 不允许接触边界时，记录首尾各需预留一个干净采样点。
        minimum_edge_clean = 0 if self.allow_edge_blocks else 2
        feasible = (
            (total_candidates[:, None] >= block_counts[None, :] * min_block_samples)
            & (total_candidates[:, None] <= block_counts[None, :] * max_block_samples)
            & (
                total_candidates[:, None]
                + (block_counts[None, :] - 1) * min_gap_samples
                + minimum_edge_clean
                <= n_times
            )
        )
        feasible_totals = total_candidates[feasible.any(dim=1)]
        if feasible_totals.numel() == 0:
            raise ValueError(
                "random_blocks has no feasible total length/block count for "
                f"n_times={n_times}, total_samples=[{min_total_samples}, {max_total_samples}], "
                f"block_samples=[{min_block_samples}, {max_block_samples}], "
                f"blocks=[{self.min_blocks}, {self.max_blocks}], "
                f"min_gap_samples={min_gap_samples}, allow_edge_blocks={self.allow_edge_blocks}"
            )

        # 先采样总覆盖长度，再在该长度的可行集合中采样块数 K；这样调整块数
        # 范围不会间接改变总长度的边缘分布。
        total_indices = torch.randint(
            feasible_totals.numel(),
            (batch_size,),
            device=device,
        )
        total_lengths = feasible_totals.index_select(0, total_indices)
        feasible_counts = (
            (total_lengths[:, None] >= block_counts[None, :] * min_block_samples)
            & (total_lengths[:, None] <= block_counts[None, :] * max_block_samples)
            & (
                total_lengths[:, None]
                + (block_counts[None, :] - 1) * min_gap_samples
                + minimum_edge_clean
                <= n_times
            )
        )
        count_scores = torch.rand(batch_size, block_counts.numel(), device=device).masked_fill(
            ~feasible_counts,
            -1.0,
        )
        selected_count_indices = count_scores.argmax(dim=1)
        selected_counts = block_counts.index_select(0, selected_count_indices)

        lengths = self._sample_bounded_partition(
            total_lengths,
            selected_counts,
            max_parts=self.max_blocks,
            minimum=min_block_samples,
            maximum=max_block_samples,
        )

        # 四种 topology 分别表示首/尾噪声块是否贴住左边界、右边界或两者；
        # 未贴边的一侧至少保留一个干净点。
        left_touches = torch.tensor([False, True, False, True], device=device)
        right_touches = torch.tensor([False, False, True, True], device=device)
        clean_samples = n_times - total_lengths[:, None]
        topology_base_clean = (
            (selected_counts[:, None] - 1) * min_gap_samples
            + (~left_touches)[None, :].to(dtype=torch.long)
            + (~right_touches)[None, :].to(dtype=torch.long)
        )
        topology_slack = clean_samples - topology_base_clean
        topology_eligible_gaps = (
            (selected_counts[:, None] - 1)
            + (~left_touches)[None, :].to(dtype=torch.long)
            + (~right_touches)[None, :].to(dtype=torch.long)
        )
        feasible_topologies = (topology_slack >= 0) & (
            (topology_slack == 0) | (topology_eligible_gaps > 0)
        )
        if not self.allow_edge_blocks:
            feasible_topologies[:, 1:] = False
        topology_scores = torch.rand(batch_size, 4, device=device).masked_fill(
            ~feasible_topologies,
            -1.0,
        )
        topology = topology_scores.argmax(dim=1)
        left_touch = left_touches.index_select(0, topology)
        right_touch = right_touches.index_select(0, topology)

        gap_slots = torch.arange(self.max_blocks + 1, device=device).unsqueeze(0)
        internal_gaps = (gap_slots >= 1) & (gap_slots < selected_counts[:, None])
        base_gaps = internal_gaps.to(dtype=torch.long) * min_gap_samples
        eligible_gaps = internal_gaps.clone()
        base_gaps[:, 0] = (~left_touch).to(dtype=torch.long)
        eligible_gaps[:, 0] = ~left_touch
        right_gap_indices = selected_counts[:, None]
        base_gaps.scatter_(
            1,
            right_gap_indices,
            (~right_touch).to(dtype=torch.long)[:, None],
        )
        eligible_gaps.scatter_(1, right_gap_indices, (~right_touch)[:, None])
        gap_slack = n_times - total_lengths - base_gaps.sum(dim=1)
        gaps = base_gaps + self._sample_unbounded_partition(gap_slack, eligible_gaps)

        increments = lengths + gaps[:, 1:]
        zero_prefix = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        starts = gaps[:, :1] + torch.cat(
            (zero_prefix, increments.cumsum(dim=1)[:, :-1]),
            dim=1,
        )
        ends = starts + lengths
        positions = torch.arange(n_times, device=device)[None, None, :]
        active_blocks = (
            torch.arange(self.max_blocks, device=device)[None, :]
            < selected_counts[:, None]
        )
        covered = (
            active_blocks[:, :, None]
            & (positions >= starts[:, :, None])
            & (positions < ends[:, :, None])
        ).any(dim=1)
        return covered & apply_augmentation[:, None]

    def _sample_noise_at_covered_positions(
        self,
        eeg: torch.Tensor,
        covered: torch.Tensor,
    ) -> torch.Tensor:
        """只为 covered 位置分配噪声，避免创建完整 [B,C,T] 噪声张量。

        sample_channel 从每个 trial、每个通道的原始整段 EEG 估计均值方差；
        batch_channel 在整个 batch 内按通道估计；standard_normal 固定 N(0,1)。
        高斯统计量均在任何位置被替换之前计算。time_shuffle 则在原始 covered
        集合内置换时间，所有通道共用排列，不生成新的幅值。
        time_shuffle_independent 在同一 covered 集合内为各通道分别采样排列。
        mixture 为每条被覆盖 trial 独立选一种方式，各路始终读取原始 EEG。
        """
        return _replace_with_noise(
            eeg, covered, noise_mode=self.noise_mode, min_noise_std=self.min_noise_std,
            noise_mode_probabilities=self.noise_mode_probabilities,
        )

    def sample_covered_time_mask(
        self, eeg: torch.Tensor, *, selected_trials: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample coverage only; explicit trials bypass the ordinary probability gate.

        The default path keeps the legacy random draws in their original order.
        Hierarchical routing supplies its already-selected local trials here and
        generates replacement values only after both coverage branches are known.
        """
        if eeg.ndim != 3:
            raise ValueError(f"Expected [batch, channel, time] EEG, got {eeg.shape}")
        batch_size, _channels, n_times = eeg.shape
        covered = torch.zeros(batch_size, n_times, dtype=torch.bool, device=eeg.device)
        if selected_trials is not None:
            if selected_trials.dtype != torch.bool or selected_trials.shape != (batch_size,) or selected_trials.device != eeg.device:
                raise ValueError("selected_trials must be one on-device boolean per trial")
        elif self.probability == 0.0:
            return covered
        if self.geometry == "circular_block":
            min_samples = self._duration_samples(self.min_block_duration_s, n_times)
            max_samples = self._duration_samples(self.max_block_duration_s, n_times)
        elif self.geometry == "edge":
            min_samples = self._duration_samples(self.min_duration_s, n_times)
            left_max_samples = self._duration_samples(self.max_left_duration_s, n_times)
            right_max_samples = self._duration_samples(self.max_right_duration_s, n_times)
        # probability 按 trial 抽样，不是按时间点
        if selected_trials is not None:
            apply_augmentation = selected_trials
        elif self.probability == 1.0:
            apply_augmentation = torch.ones(batch_size, dtype=torch.bool, device=eeg.device)
        elif self.probability == 0.0:
            apply_augmentation = torch.zeros(batch_size, dtype=torch.bool, device=eeg.device)
        else:
            apply_augmentation = torch.rand(batch_size, device=eeg.device) < self.probability

        positions = torch.arange(n_times, device=eeg.device).unsqueeze(0)
        if self.geometry == "circular_block":
            if max_samples >= 1:
                low = min(max(1, min_samples), max_samples)
                lengths = torch.randint(
                    low,
                    max_samples + 1,
                    (batch_size,),
                    device=eeg.device,
                )
                starts = torch.randint(0, n_times, (batch_size,), device=eeg.device)
                offsets = (positions - starts[:, None]).remainder(n_times)  # 可跨首尾
                covered = apply_augmentation[:, None] & (offsets < lengths[:, None])
        elif self.geometry == "random_blocks":
            covered = self._random_blocks_covered_mask(
                batch_size=batch_size,
                n_times=n_times,
                device=eeg.device,
                apply_augmentation=apply_augmentation,
            )
        else:
            if self.side == "random":
                crop_left = torch.rand(batch_size, device=eeg.device) < 0.5
            else:
                crop_left = torch.full(
                    (batch_size,),
                    self.side == "left",
                    dtype=torch.bool,
                    device=eeg.device,
                )
            edge_max_samples = torch.where(
                crop_left,
                torch.full_like(crop_left, left_max_samples, dtype=torch.long),
                torch.full_like(crop_left, right_max_samples, dtype=torch.long),
            )
            active = apply_augmentation & (edge_max_samples >= 1)
            low = torch.minimum(
                edge_max_samples.clamp_min(1),
                torch.full_like(edge_max_samples, max(1, min_samples)),
            )
            interval_size = (edge_max_samples - low + 1).clamp_min(1)
            lengths = low + torch.floor(
                torch.rand(batch_size, device=eeg.device) * interval_size
            ).to(dtype=torch.long)
            left_covered = positions < lengths[:, None]
            right_covered = positions >= (n_times - lengths)[:, None]
            covered = active[:, None] & torch.where(
                crop_left[:, None], left_covered, right_covered
            )
        return covered

    def forward(self, eeg: torch.Tensor, *, return_mask: bool = False) -> torch.Tensor | TemporalCropNoiseResult:
        """Apply ordinary augmentation, preserving its original probability and RNG."""
        covered = self.sample_covered_time_mask(eeg)
        augmented = eeg.clone() if self.probability == 0.0 else self._sample_noise_at_covered_positions(eeg, covered)
        if not return_mask:
            return augmented
        return TemporalCropNoiseResult(eeg=augmented, valid_time_mask=~covered, covered_time_mask=covered)


class NearFullNoiseAugmenter(nn.Module):
    """在原增强上混入近全噪声 trial，保留随机连续的少量原始 EEG。

    覆盖量按当前输入 T 的比例决定，所有通道共享时间掩码。被选中的 trial
    从替换前 EEG 重新构造，保留区间不会叠加普通 crop-noise。未选中 trial
    仍使用普通增强。训练循环负责解释 near_full_noise_mask 的损失语义。
    """

    def __init__(
        self,
        ordinary_augmenter: EEGInputAugmenter | None = None,
        *,
        probability: float = 0.1,
        coverage_ratio: float = 0.95,
        noise_mode: str = "sample_channel",
        noise_mode_probabilities: Mapping[str, float] | None = None,
        min_noise_std: float = 1e-4,
    ) -> None:
        super().__init__()
        options = normalize_near_full_noise_config({
            "enabled": True, "probability": probability, "coverage_ratio": coverage_ratio,
        })
        noise_mode_probabilities = normalize_noise_mode_probabilities(noise_mode, noise_mode_probabilities)
        min_noise_std = float(min_noise_std)
        if not math.isfinite(min_noise_std) or min_noise_std < 0.0:
            raise ValueError("min_noise_std must be finite and non-negative")
        self.ordinary_augmenter = ordinary_augmenter
        self.probability = float(options["probability"])
        self.coverage_ratio = float(options["coverage_ratio"])
        self.noise_mode = noise_mode
        self.noise_mode_probabilities = noise_mode_probabilities
        self.min_noise_std = min_noise_std

    def forward(self, eeg: torch.Tensor, *, return_mask: bool = False) -> torch.Tensor | TemporalCropNoiseResult:
        if eeg.ndim != 3 or eeg.shape[0] < 1 or eeg.shape[-1] < 1:
            raise ValueError("eeg must have non-empty [batch, channel, time] shape")
        if self.ordinary_augmenter is None:
            covered = torch.zeros(eeg.shape[0], eeg.shape[-1], dtype=torch.bool, device=eeg.device)
            result = TemporalCropNoiseResult(eeg.clone(), ~covered, covered)
        else:
            result = self.ordinary_augmenter(eeg, return_mask=True)
        if self.probability == 0.0:
            return result if return_mask else result.eeg
        batch_size, _, n_times = eeg.shape
        near_full = (
            torch.ones(batch_size, dtype=torch.bool, device=eeg.device)
            if self.probability == 1.0
            else torch.rand(batch_size, device=eeg.device) < self.probability
        )
        selected_count = int(near_full.sum().item())
        if selected_count:
            covered_count = min(n_times, max(1, math.floor(n_times * self.coverage_ratio + 0.5)))
            clean_count = n_times - covered_count
            positions = torch.arange(n_times, device=eeg.device)[None, :]
            if clean_count:
                starts = torch.randint(n_times - clean_count + 1, (selected_count,), device=eeg.device)
                clean = (positions >= starts[:, None]) & (positions < starts[:, None] + clean_count)
                selected_covered = ~clean
            else:
                selected_covered = torch.ones(selected_count, n_times, dtype=torch.bool, device=eeg.device)
            # 统计量从原始整批估计，batch_channel 与普通增强使用同一统计范围。
            replacement_mask = torch.zeros_like(result.covered_time_mask)
            replacement_mask[near_full] = selected_covered
            replacement = _replace_with_noise(
                eeg, replacement_mask, noise_mode=self.noise_mode, min_noise_std=self.min_noise_std,
                noise_mode_probabilities=self.noise_mode_probabilities,
            )
            # 普通增强的掩码可能已被 autograd 保存，合并时不能原地修改。
            combined_covered = torch.where(
                near_full[:, None], replacement_mask, result.covered_time_mask,
            )
            result = TemporalCropNoiseResult(
                eeg=torch.where(near_full[:, None, None], replacement, result.eeg),
                valid_time_mask=~combined_covered,
                covered_time_mask=combined_covered,
            )
        result.near_full_noise_mask = near_full
        return result if return_mask else result.eeg


class HierarchicalEEGAugmenter(nn.Module):
    """Global gate -> roll or replacement -> near-full or local -> one noise mode.

    All branches read original EEG. Rolling preserves channel alignment and real
    labels; its coverage is false, so noise-suppression losses receive no roll
    positions. The ordinary augmenter supplies geometry without a second gate.
    """

    def __init__(
        self,
        ordinary_augmenter: EEGInputAugmenter,
        *,
        near_full_noise: object = None,
        roll_probability: float = .2,
        roll_max_shift_s: float = 1.,
        roll_left_probability: float = .5,
    ) -> None:
        super().__init__()
        options = normalize_augmentation_pipeline_config(dict(enabled=True,
            roll_probability=roll_probability, roll_max_shift_s=roll_max_shift_s,
            roll_left_probability=roll_left_probability))
        near = normalize_near_full_noise_config(near_full_noise)
        self.ordinary_augmenter = ordinary_augmenter
        self.probability = ordinary_augmenter.probability
        self.sample_interval_s = ordinary_augmenter.sample_interval_s
        self.roll_probability = float(options["roll_probability"])
        self.roll_max_shift_s = float(options["roll_max_shift_s"])
        self.roll_left_probability = float(options["roll_left_probability"])
        maximum_shift_samples = self.roll_max_shift_s / self.sample_interval_s
        if not math.isfinite(maximum_shift_samples):
            raise ValueError(
                "augmentation_pipeline.roll_max_shift_s is too large relative to sample_interval_s"
            )
        self.roll_max_shift_samples = math.floor(maximum_shift_samples + 1e-9)
        if self.probability > 0. and self.roll_probability > 0. and self.roll_max_shift_samples < 1:
            raise ValueError("augmentation_pipeline.roll_max_shift_s is shorter than one EEG sample")
        self.near_full_noise_probability = float(near["probability"]) if near["enabled"] else 0.
        self.coverage_ratio = float(near["coverage_ratio"])
        self.noise_mode = ordinary_augmenter.noise_mode
        self.noise_mode_probabilities = ordinary_augmenter.noise_mode_probabilities
        self.min_noise_std = ordinary_augmenter.min_noise_std

    @staticmethod
    def _select(candidates: torch.Tensor, probability: float) -> torch.Tensor:
        selected = torch.zeros_like(candidates)
        if probability == 1.:
            return candidates.clone()
        if probability > 0.:
            count = int(candidates.sum().item())
            if count:
                selected[candidates] = torch.rand(count, device=candidates.device) < probability
        return selected

    def forward(self, eeg: torch.Tensor, *, return_mask: bool = False) -> torch.Tensor | TemporalCropNoiseResult:
        if eeg.ndim != 3 or min(eeg.shape) < 1:
            raise ValueError("eeg must have non-empty [batch, channel, time] shape")
        batch_size, _, n_times = eeg.shape
        if n_times < 2 and self.probability > 0. and self.roll_probability > 0.:
            raise ValueError("Hierarchical roll requires at least two time points")
        candidates = torch.ones(batch_size, dtype=torch.bool, device=eeg.device)
        active = self._select(candidates, self.probability)
        rolled = self._select(active, self.roll_probability)
        replacement_trials = active & ~rolled
        near = self._select(replacement_trials, self.near_full_noise_probability)
        local = replacement_trials & ~near
        covered = torch.zeros(batch_size, n_times, dtype=torch.bool, device=eeg.device)
        if bool(local.any().item()):
            covered = self.ordinary_augmenter.sample_covered_time_mask(eeg, selected_trials=local)
            if not bool(covered[local].any(dim=1).all().item()):
                raise ValueError("Hierarchical local augmentation requires positive coverage for every selected trial")
        selected_count = int(near.sum().item())
        if selected_count:
            covered_count = min(n_times, max(1, math.floor(n_times * self.coverage_ratio + .5)))
            clean_count = n_times - covered_count
            if clean_count:
                starts = torch.randint(n_times - clean_count + 1, (selected_count,), device=eeg.device)
                positions = torch.arange(n_times, device=eeg.device)[None, :]
                clean = (positions >= starts[:, None]) & (positions < starts[:, None] + clean_count)
                covered[near] = ~clean
            else:
                covered[near] = True
        # Select a replacement mode once per covered trial, after deciding coverage.
        # Full-batch source data preserve batch_channel statistics and clean remnants.
        result = (
            _replace_with_noise(eeg, covered, noise_mode=self.noise_mode,
                noise_mode_probabilities=self.noise_mode_probabilities, min_noise_std=self.min_noise_std)
            if bool(covered.any().item()) else eeg.clone()
        )
        shifts = torch.zeros(batch_size, dtype=torch.long, device=eeg.device)
        roll_count = int(rolled.sum().item())
        if roll_count:
            maximum = min(self.roll_max_shift_samples, n_times - 1)
            if maximum < 1:
                raise ValueError("Hierarchical roll requires at least two time points")
            magnitude = torch.randint(1, maximum + 1, (roll_count,), device=eeg.device)
            left = self._select(torch.ones(roll_count, dtype=torch.bool, device=eeg.device), self.roll_left_probability)
            shifts[rolled] = torch.where(left, -magnitude, magnitude)
            source = (torch.arange(n_times, device=eeg.device)[None, :] - shifts[:, None]).remainder(n_times)
            shifted = eeg.gather(2, source[:, None, :].expand_as(eeg))
            result = torch.where(rolled[:, None, None], shifted, result)
        augmented = TemporalCropNoiseResult(result, ~covered, covered, near, rolled, shifts)
        return augmented if return_mask else augmented.eeg


def _sample_interval_from_time_s(time_s: Sequence[float] | torch.Tensor | None) -> float | None:
    """由时间轴相邻差值的中位数估计采样间隔（秒）。"""
    if time_s is None:
        return None
    values = torch.as_tensor(time_s, dtype=torch.float64)
    if values.numel() < 2:
        return None
    diffs = values[1:] - values[:-1]
    return float(diffs.abs().median().item())


def build_eeg_augmenter(
    config: object,
    *,
    time_s: Sequence[float] | torch.Tensor | None = None,
) -> EEGInputAugmenter | None:
    """从 temporal_crop_noise 配置构建增强器；关闭时返回 None。

    采样间隔优先使用 sample_interval_s，其次由 sample_rate_hz 取倒数，最后
    从数据集 time_s 推断。supervision_mode 由训练循环解析，不传入增强器。
    """
    if not isinstance(config, Mapping) or not bool(config.get("enabled", False)):
        return None  # 关闭增强时训练循环不挂 augmenter
    sample_interval_s = config.get("sample_interval_s")
    if sample_interval_s is None and config.get("sample_rate_hz") is not None:
        sample_rate_hz = float(config["sample_rate_hz"])
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be finite and positive")
        sample_interval_s = 1.0 / sample_rate_hz
    if sample_interval_s is None:
        sample_interval_s = _sample_interval_from_time_s(time_s)  # 默认从数据集 time_s 推断
    if sample_interval_s is None:
        raise ValueError("temporal_crop_noise requires time_s, sample_interval_s, or sample_rate_hz")
    return EEGInputAugmenter(
        sample_interval_s=float(sample_interval_s),
        geometry=str(config.get("geometry", config.get("mode", "edge"))),
        probability=float(config.get("probability", 1.0)),
        side=str(config.get("side", "random")),
        min_duration_s=float(config.get("min_duration_s", 0.05)),
        max_left_duration_s=float(config.get("max_left_duration_s", 1.0)),
        max_right_duration_s=float(config.get("max_right_duration_s", 0.75)),
        block_duration_s=(
            float(config["block_duration_s"]) if config.get("block_duration_s") is not None else None
        ),
        min_block_duration_s=(
            float(config["min_block_duration_s"])
            if config.get("min_block_duration_s") is not None
            else None
        ),
        max_block_duration_s=(
            float(config.get("max_block_duration_s", config.get("max_duration_s")))
            if config.get("max_block_duration_s", config.get("max_duration_s")) is not None
            else None
        ),
        min_blocks=config.get("min_blocks", 1),
        max_blocks=config.get("max_blocks", 3),
        min_total_duration_s=(
            float(config["min_total_duration_s"])
            if config.get("min_total_duration_s") is not None
            else None
        ),
        max_total_duration_s=(
            float(config["max_total_duration_s"])
            if config.get("max_total_duration_s") is not None
            else None
        ),
        min_clean_gap_s=float(config.get("min_clean_gap_s", 0.0)),
        allow_edge_blocks=config.get("allow_edge_blocks", True),
        noise_mode=str(config.get("noise_mode", "sample_channel")),
        noise_mode_probabilities=config.get("noise_mode_probabilities"),
        min_noise_std=float(config.get("min_noise_std", 1e-4)),
    )


def _validate_pipeline_input_config(config: object) -> None:
    """Keep legacy coercions out of the new global gate without changing old runs."""
    if not isinstance(config, Mapping):
        raise ValueError("train.augmentation_pipeline.enabled requires train.temporal_crop_noise.enabled")
    if not isinstance(config.get("enabled", False), bool):
        raise ValueError("train.temporal_crop_noise.enabled must be a boolean")
    clock_field = "sample_interval_s" if config.get("sample_interval_s") is not None else "sample_rate_hz"
    for field, default in (("probability", 1.0), (clock_field, None)):
        value = config.get(field, default)
        if value is None and field != "probability":
            continue
        message = f"train.temporal_crop_noise.{field} must be a finite number"
        if isinstance(value, bool):
            raise ValueError(message)
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(message) from error
        if not math.isfinite(number):
            raise ValueError(message)
        if field == "probability" and not 0.0 <= number <= 1.0:
            raise ValueError("train.temporal_crop_noise.probability must be in [0, 1]")
        if field != "probability" and number <= 0.0:
            raise ValueError(message + " greater than zero")


def _validate_pipeline_time_axis(
    time_s: Sequence[float] | torch.Tensor | None, sample_interval_s: float,
) -> None:
    """A shift bounded in seconds requires the configured clock to match the data."""
    if time_s is None:
        return
    try:
        source = torch.as_tensor(time_s)
        if source.is_complex() or source.dtype == torch.bool:
            raise ValueError("augmentation_pipeline time_s must contain real numeric values")
        values = torch.as_tensor(time_s, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        raise ValueError("augmentation_pipeline time_s must contain real numeric values") from error
    if values.ndim != 1 or values.numel() < 2 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("augmentation_pipeline time_s must be a finite 1D time axis with at least two samples")
    differences = values[1:] - values[:-1]
    interval = float(differences.median().item())
    # Float32 time axes around a few seconds have visible quantization at 1 kHz.
    # Permit that rounding while rejecting irregular or contradictory clocks.
    if not bool((torch.isfinite(differences) & (differences > 0)).all().item()) or not torch.allclose(
        differences, torch.full_like(differences, interval), rtol=1e-3, atol=1e-8,
    ):
        raise ValueError("augmentation_pipeline time_s must be strictly increasing and uniformly sampled")
    if not math.isclose(sample_interval_s, interval, rel_tol=1e-3, abs_tol=1e-8):
        raise ValueError("temporal_crop_noise sample_interval_s/sample_rate_hz disagree with time_s")


def build_target_eeg_augmenter(
    config: object,
    *,
    near_full_noise: object = None,
    augmentation_pipeline: object = None,
    time_s: Sequence[float] | torch.Tensor | None = None,
) -> EEGInputAugmenter | NearFullNoiseAugmenter | HierarchicalEEGAugmenter | None:
    """Target 专用组合；关闭新功能时原样返回原增强器，不消耗额外随机数。"""
    pipeline = normalize_augmentation_pipeline_config(augmentation_pipeline)
    if pipeline["enabled"]:
        _validate_pipeline_input_config(config)
    ordinary = build_eeg_augmenter(config, time_s=time_s)
    options = normalize_near_full_noise_config(near_full_noise)
    if pipeline["enabled"]:
        if ordinary is None:
            raise ValueError("train.augmentation_pipeline.enabled requires train.temporal_crop_noise.enabled")
        _validate_pipeline_time_axis(time_s, ordinary.sample_interval_s)
        return HierarchicalEEGAugmenter(ordinary, near_full_noise=options,
            **{key: value for key, value in pipeline.items() if key != "enabled"})
    if not options["enabled"] or float(options["probability"]) == 0.0:
        return ordinary
    noise_options = config if isinstance(config, Mapping) else {}
    return NearFullNoiseAugmenter(
        ordinary,
        probability=float(options["probability"]),
        coverage_ratio=float(options["coverage_ratio"]),
        noise_mode=str(noise_options.get("noise_mode", "sample_channel")),
        noise_mode_probabilities=noise_options.get("noise_mode_probabilities"),
        min_noise_std=float(noise_options.get("min_noise_std", 1e-4)),
    )
