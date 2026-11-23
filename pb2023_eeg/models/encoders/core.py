"""EEG 时间编码器。

编码器接收 ``[B,C,T]`` EEG，其中 ``B`` 为 trial/batch 数、``C`` 为电极
通道数、``T`` 为时间点数；先通过 1x1 卷积投影到 ``D`` 维，再由 CNN、
滑动局部 Transformer 或 centered patch MLP 编码时间上下文，最终返回
``[B,T,D]``。三种后端都保持时间轴长度不变。

局部 Transformer 始终在完整时间轴上计算 attention，并用局部 mask
限制为 right-local 或 centered-local 窗口，不分块也不 shift。patch MLP 则
显式展开中心对齐的 3/5/7 点窗口。数据增强写入的噪声仍是合法 EEG 输入，
不会被当作 padding 或 attention mask 排除。
"""

from __future__ import annotations

from math import isfinite, log
from typing import Mapping, Sequence

import torch
from torch import nn


ENCODER_TYPES = {"cnn", "patch_mlp", "window_transformer"}
LOCAL_ATTENTION_PATTERNS = {"centered_local", "right_local"}
POSITION_ENCODING_MODES = {"none", "sinusoidal_qk"}
BOUNDARY_MODES = {"truncate", "reflect", "noise_pad"}


def _positive_int(value: object, *, name: str) -> int:
    """把配置值严格解析为正整数，拒绝布尔值、小数及非有限数。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if not isfinite(numeric) or numeric != integer or integer < 1:
        raise ValueError(f"{name} must be a positive integer")
    return integer


def _nonnegative_int(value: object, *, name: str) -> int:
    """把配置值严格解析为非负整数，主要用于验证 ``shift_size``。"""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer")
    try:
        integer = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a non-negative integer") from exc
    if not isfinite(numeric) or numeric != integer or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


def normalize_encoder_type(value: object) -> str:
    """规范化编码器名称及兼容别名。"""
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "conv": "cnn",
        "convolution": "cnn",
        "transformer": "window_transformer",
        "window": "window_transformer",
        "patch": "patch_mlp",
        "temporal_patch_mlp": "patch_mlp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ENCODER_TYPES:
        valid = ", ".join(sorted(ENCODER_TYPES))
        raise ValueError(f"encoder type must be one of: {valid}")
    return normalized


def normalize_attention_pattern(value: object) -> str:
    """规范化局部 attention 的方向，并兼容 centered/symmetric 别名。"""
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "centered": "centered_local",
        "symmetric": "centered_local",
        "symmetric_local": "centered_local",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in LOCAL_ATTENTION_PATTERNS:
        valid = ", ".join(sorted(LOCAL_ATTENTION_PATTERNS))
        raise ValueError(f"attention_pattern must be one of: {valid}")
    return normalized


def normalize_position_encoding_mode(value: object) -> str:
    """规范化 Transformer Q/K 位置编码开关。"""
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "off": "none",
        "disabled": "none",
        "sinusoidal": "sinusoidal_qk",
        "qk_sinusoidal": "sinusoidal_qk",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in POSITION_ENCODING_MODES:
        valid = ", ".join(sorted(POSITION_ENCODING_MODES))
        raise ValueError(f"position_encoding_mode must be one of: {valid}")
    return normalized


def normalize_boundary_mode(value: object) -> str:
    """规范化局部时间编码器的序列边界处理方式。"""
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {"default": "truncate", "truncated": "truncate", "reflection": "reflect"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in BOUNDARY_MODES:
        valid = ", ".join(sorted(BOUNDARY_MODES))
        raise ValueError(f"boundary_mode must be one of: {valid}")
    return normalized


def normalize_boundary_noise_seed(value: object) -> int:
    """Validate a reproducible evaluation seed without truncating floats."""
    seed = value if type(value) is int else _nonnegative_int(value, name="boundary_noise_seed")
    if not 0 <= seed < (1 << 63):
        raise ValueError("boundary_noise_seed must be in [0, 2**63)")
    return seed


def resolve_encoder_input_channels(
    config: Mapping[str, object],
    input_channels: int,
) -> dict[str, object]:
    """Resolve encoder width from the constructed dataset.

    ``in_channels: auto`` (or an omitted field) follows the preprocessing output.
    An explicit value must agree with it, preventing stale configurations from
    silently misreading concatenated inputs.
    """
    resolved = dict(config)
    expected = _positive_int(input_channels, name="dataset input_channels")
    requested = resolved.get("in_channels", "auto")
    if requested is None or (
        isinstance(requested, str) and requested.strip().lower() == "auto"
    ):
        resolved["in_channels"] = expected
        return resolved
    actual = _positive_int(requested, name="encoder.in_channels")
    if actual != expected:
        raise ValueError(
            "encoder.in_channels does not match the constructed dataset: "
            f"{actual} != {expected}"
        )
    resolved["in_channels"] = actual
    return resolved


class ChannelLayerNorm(nn.Module):
    """对每个时间点的特征通道做 LayerNorm。

    ``Conv1d`` 使用 ``[B,D,T]``，而 ``LayerNorm(D)`` 要求被归一化的维度在
    最后，因此前向中临时变为 ``[B,T,D]``，归一化后再恢复原布局。
    """

    def __init__(self, channels: int) -> None:
        """创建作用于 ``D=channels`` 的 LayerNorm。"""
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入和输出均为 ``[B,D,T]``，逐 ``(B,T)`` 位置归一化 D 维。"""
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    """CNN 后端的局部时间残差块。

    深度卷积按特征通道独立地聚合局部时间邻域，感受野由 ``kernel_size`` 与
    ``dilation`` 决定；随后的 1x1 卷积只在同一时间点混合 D 维特征。对称
    padding 保持 ``T`` 不变，输入输出均为 ``[B,D,T]``。增强产生的噪声与
    正常 EEG 一样参与卷积，不存在额外的噪声 mask。
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        expansion_ratio: int = 2,
    ) -> None:
        """构建一个保持时间长度的深度可分离残差块。"""
        super().__init__()
        channels = _positive_int(channels, name="channels")
        kernel_size = _positive_int(kernel_size, name="kernel_size")
        dilation = _positive_int(dilation, name="dilation")
        if kernel_size % 2 != 1:
            raise ValueError("CNN kernel_size must be odd to preserve time alignment")
        expansion_ratio = _positive_int(
            expansion_ratio,
            name="expansion_ratio",
        )
        hidden_channels = channels * expansion_ratio
        self.norm = ChannelLayerNorm(channels)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            groups=channels,
            bias=False,
        )  # 每个 D 通道独立聚合局部时间邻域。
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )  # 只在同一时间点内扩展并混合特征维。
        # 残差分支从零贡献开始，初始块严格保持恒等映射，随后再逐步学习修正。
        nn.init.zeros_(self.pointwise[-2].weight)
        nn.init.zeros_(self.pointwise[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将局部时间修正加回输入，形状始终为 ``[B,D,T]``。"""
        residual = x
        x = self.norm(x)
        x = self.depthwise(x)  # 每通道沿时间局部卷积；padding 保 T
        x = self.pointwise(x)  # 同一时刻混 D 维
        return residual + x


class RightLocalTransformerBlock(nn.Module):
    """在完整时间轴上执行滑动局部注意力的 Transformer block。

    输入保持为 ``[B,T,D]`` 送入 multi-head attention，不再切成互不重叠的
    小块。``right_local`` 下第 ``t`` 个 query 读取 ``t..t+W-1``；
    ``centered_local`` 下奇数窗口 ``W`` 读取 ``t-W//2..t+W//2``；默认边界
    截短，noise_pad 只补越界 K/V，不增加 Q 或扩大真实点的可见范围。
    所有原始时间点（包括噪声增强位置）都是合法 token，mask 只编码时间局部性。
    ``sinusoidal_qk`` 按数据实际长度生成正弦时间编码并只加入 Q/K；``none``
    不注入位置编码。V 始终是原始归一化后的 EEG 内容。
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        time_point_count: int,
        num_heads: int,
        dropout: float,
        expansion_ratio: int = 2,
        attention_pattern: str = "right_local",
        position_encoding_mode: str = "sinusoidal_qk",
        boundary_mode: str = "truncate",
    ) -> None:
        """构建局部 self-attention 与逐 token FFN 残差路径。"""
        super().__init__()
        self.channels = _positive_int(channels, name="channels")
        self.window_size = _positive_int(window_size, name="window_size")
        self.attention_pattern = normalize_attention_pattern(attention_pattern)
        if self.attention_pattern == "centered_local" and self.window_size % 2 != 1:
            raise ValueError("centered_local window_size must be odd")
        self.position_encoding_mode = normalize_position_encoding_mode(
            position_encoding_mode
        )
        self.boundary_mode = normalize_boundary_mode(boundary_mode)
        if self.boundary_mode == "noise_pad" and self.attention_pattern != "centered_local":
            raise ValueError("noise_pad boundary_mode requires centered_local attention")
        if self.boundary_mode == "reflect":
            if self.attention_pattern != "centered_local":
                raise ValueError("reflect boundary_mode requires centered_local attention")
            if self.position_encoding_mode != "none":
                raise ValueError("reflect boundary_mode currently requires position_encoding_mode=none")
        self.time_point_count = _positive_int(
            time_point_count,
            name="time_point_count",
        )
        self.num_heads = _positive_int(num_heads, name="num_heads")
        if self.channels % self.num_heads != 0:
            raise ValueError("num_heads must divide latent_dim")
        expansion_ratio = _positive_int(
            expansion_ratio,
            name="expansion_ratio",
        )
        hidden_dim = self.channels * expansion_ratio
        self.attention_norm = nn.LayerNorm(self.channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.channels,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        # 完整时间轴 attention 的逻辑 score 大小为 B*H*T*Tkv。沿 trial batch
        # 分块不会改变任何 trial 内的注意力语义，同时限制单次 kernel 的网格
        # 和峰值 score 规模；该字段也便于测试强制走分块路径。
        self.attention_batch_limit = max(1, 65_535 // self.num_heads)
        self.attention_score_element_limit = 16 * 1024 * 1024
        self.attention_residual_dropout = nn.Dropout(float(dropout))
        self.ffn_norm = nn.LayerNorm(self.channels)
        self.ffn = nn.Sequential(
            nn.Linear(self.channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, self.channels),
            nn.Dropout(float(dropout)),
        )
        # T 在模型构造前已由数据集解析。非持久化 buffer 只表达运行时
        # 几何关系，随 model.to(...) 迁移设备/浮点类型，但不写入 checkpoint。
        self.register_buffer(
            "attention_mask",
            self._build_attention_mask(),
            persistent=False,
        )
        self.register_buffer(
            "position_encoding",
            self._build_position_encoding(),
            persistent=False,
        )
        if self.boundary_mode == "noise_pad":
            radius = self.window_size // 2
            self.register_buffer(
                "key_position_encoding",
                self._build_position_encoding(
                    start=-radius, count=self.time_point_count + 2 * radius,
                ),
                persistent=False,
            )
        # 与 CNN 残差块一致，从恒等映射开始训练，同时保留可学习路径。
        nn.init.zeros_(self.attention.out_proj.weight)
        if self.attention.out_proj.bias is not None:
            nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.ffn[-2].weight)
        nn.init.zeros_(self.ffn[-2].bias)

    def _build_attention_mask(self) -> torch.Tensor:
        """返回 ``[T,Tkv]`` 的局部 boolean mask。

        行是 query 时间，列是 key 时间。False 表示允许读取，True 表示传入
        MHA 后在相应 score 上屏蔽。pad 只扩展 key 列，不增加 query 行。
        """
        query_index = torch.arange(self.time_point_count)[:, None]
        if self.boundary_mode in {"reflect", "noise_pad"}:
            radius = self.window_size // 2
            key_index = torch.arange(self.time_point_count + 2 * radius)[None, :]
            return (key_index - (query_index + radius)).abs() > radius
        key_index = torch.arange(self.time_point_count)[None, :]
        relative_offset = key_index - query_index  # >0 表示 key 在 query 右侧（更晚）
        if self.attention_pattern == "right_local":
            # True=禁止：挡住左侧(过去) 与超过 W 的右侧；允许 t..t+W-1。
            return (relative_offset < 0) | (relative_offset >= self.window_size)
        radius = self.window_size // 2
        # centered_local：允许 [t-radius, t+radius]，首尾自然截短。
        return relative_offset.abs() > radius

    def _build_position_encoding(
        self, *, start: int = 0, count: int | None = None,
    ) -> torch.Tensor:
        """按构造时 ``T`` 生成 ``[1,T,D]`` 正弦绝对时间编码。

        这种编码没有固定最大长度，也不会像旧窗口内 embedding 那样每隔 W
        个点周期性重置，因此适用于滑动窗口及不同数据长度。
        """
        count = self.time_point_count if count is None else count
        if self.position_encoding_mode == "none":
            return torch.zeros(
                (1, count, self.channels),
                dtype=torch.float32,
            )
        even_channels = torch.arange(
            0,
            self.channels,
            2,
            dtype=torch.float32,
        )
        angular_frequency = torch.exp(
            even_channels * (-log(10_000.0) / self.channels)
        )
        position = torch.arange(
            start, start + count,
            dtype=torch.float32,
        )[:, None]
        angles = position * angular_frequency[None, :]
        encoding = torch.zeros(
            (count, self.channels),
            dtype=torch.float32,
        )
        encoding[:, 0::2] = torch.sin(angles)
        odd_channel_count = encoding[:, 1::2].shape[1]
        if odd_channel_count:
            encoding[:, 1::2] = torch.cos(angles[:, :odd_channel_count])
        return encoding[None, :, :]

    def _attention_chunk_size(self, batch_size: int, time_count: int) -> int:
        """按逻辑 score 元素预算确定一次处理的 trial 数。"""
        score_elements_per_trial = self.num_heads * time_count * self.attention_mask.shape[1]
        score_limited_batch = max(
            1,
            self.attention_score_element_limit // score_elements_per_trial,
        )
        return max(
            1,
            min(batch_size, self.attention_batch_limit, score_limited_batch),
        )

    def forward(
        self, x: torch.Tensor, *, boundary_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在完整时间轴上编码 ``[B,D,T]``，并返回相同形状。"""
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [batch, {self.channels}, time] input, got {x.shape}"
            )
        batch_size, _channels, time_count = x.shape
        if time_count != self.time_point_count:
            raise ValueError(
                "Window Transformer input time length does not match construction: "
                f"{time_count} != {self.time_point_count}"
            )
        if self.boundary_mode != "noise_pad" and boundary_latent is not None:
            raise ValueError("boundary_latent requires noise_pad boundary_mode")
        latent = x.transpose(1, 2)
        normalized = self.attention_norm(latent)
        # 位置只影响 Q/K 路由；value 保持为归一化后的 EEG（噪声 token 同样保留）。
        # 数值和设备均复用构造期 buffer；仅在 AMP 令 activation dtype 与模型
        # buffer 不同时转换小型 [1,T,D] 张量，避免把整批 [B,T,D] Q/K 提升到 FP32。
        position_encoding = self.position_encoding
        if position_encoding.dtype != normalized.dtype:
            position_encoding = position_encoding.to(dtype=normalized.dtype)
        positioned = (
            normalized
            if self.position_encoding_mode == "none"
            else normalized + position_encoding
        )
        if self.boundary_mode == "noise_pad":
            radius = self.window_size // 2
            if radius:
                expected = (batch_size, self.channels, 2 * radius)
                if boundary_latent is None or tuple(boundary_latent.shape) != expected:
                    raise ValueError(f"noise_pad boundary_latent must have shape {expected}")
                if boundary_latent.device != x.device or boundary_latent.dtype != x.dtype:
                    raise ValueError("boundary_latent must match input device and dtype")
                boundary_normalized = self.attention_norm(boundary_latent.transpose(1, 2))
                value = torch.cat((
                    boundary_normalized[:, :radius], normalized, boundary_normalized[:, radius:],
                ), dim=1)
            else:
                value = normalized
            query = positioned
            key_position = self.key_position_encoding.to(dtype=normalized.dtype)
            key = value if self.position_encoding_mode == "none" else value + key_position
        elif self.boundary_mode == "reflect":
            radius = self.window_size // 2
            reflected = torch.nn.functional.pad(
                normalized.transpose(1, 2),
                (radius, radius),
                mode="reflect",
            ).transpose(1, 2)
            query = normalized
            key = reflected
            value = reflected
        else:
            query = positioned
            key = positioned
            value = normalized
        chunk_size = self._attention_chunk_size(batch_size, time_count)
        if batch_size <= chunk_size:
            attended, _weights = self.attention(
                query,
                key,
                value,
                attn_mask=self.attention_mask,
                need_weights=False,
            )
        else:
            # 这里只切相互独立的 trial 维；每个 chunk 仍接收完整的 T 个时间点
            # 和完全相同的 [T,Tkv] mask，不会形成新的时间边界。
            attended_chunks: list[torch.Tensor] = []
            for start in range(0, batch_size, chunk_size):
                end = min(start + chunk_size, batch_size)
                attended_chunk, _weights = self.attention(
                    query[start:end],
                    key[start:end],
                    value[start:end],
                    attn_mask=self.attention_mask,
                    need_weights=False,
                )
                attended_chunks.append(attended_chunk)
            attended = torch.cat(attended_chunks, dim=0)
        latent = latent + self.attention_residual_dropout(attended)
        latent = latent + self.ffn(self.ffn_norm(latent))
        return latent.transpose(1, 2)


# 名称保留旧 checkpoint/import 兼容；pattern 由构造参数明确记录。
CenteredLocalTransformerBlock = RightLocalTransformerBlock
FixedWindowTransformerBlock = RightLocalTransformerBlock


class TemporalPatchMLPBlock(nn.Module):
    """用一个中心对齐的短时间 patch 生成逐 token 残差修正。

    输入 ``[B,D,T]`` 在时间轴两侧补零后展开为 ``[B,T,D*W]``。每个输出
    token 只读取以自身为中心的 W 个输入 token；MLP 不在不同输出时间点间
    交换信息。V6 只允许 W=3/5/7，并只堆叠一个 block，使最大结构感受野
    与配置的窗口严格一致。
    """

    def __init__(
        self,
        channels: int,
        window_size: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        """构建 ``D*W -> H -> D`` 的中心 patch MLP。"""
        super().__init__()
        self.channels = _positive_int(channels, name="channels")
        self.window_size = _positive_int(window_size, name="window_size")
        if self.window_size not in {3, 5, 7}:
            raise ValueError("patch_mlp window_size must be one of: 3, 5, 7")
        self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        patch_dim = self.channels * self.window_size
        self.norm = nn.LayerNorm(patch_dim)
        self.mlp = nn.Sequential(
            nn.Linear(patch_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.channels),
            nn.Dropout(self.dropout),
        )
        # 与另外两种 temporal block 一致，从严格恒等映射开始。
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将 ``[B,D,T]`` 映射为同形状输出，且每点只读取 centered W 点。"""
        if x.ndim != 3 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected [batch, {self.channels}, time] input, got {x.shape}"
            )
        radius = self.window_size // 2
        padded = torch.nn.functional.pad(x, (radius, radius))
        # unfold: [B,D,T,W]；同一输出 t 的 D*W patch 再进入独立 MLP。
        patches = padded.unfold(2, self.window_size, 1)
        patches = patches.permute(0, 2, 1, 3).reshape(
            x.shape[0], x.shape[2], self.channels * self.window_size
        )
        update = self.mlp(self.norm(patches))
        return x + update.transpose(1, 2)


class EEGEncoder(nn.Module):
    """可在 CNN、局部 Transformer 与 patch MLP 间切换的 EEG 主干。

    公共的 spatial projection 先在每个时间点把 ``C`` 个电极通道映射到
    ``D=latent_dim``。三个后端都保留 ``T``，并显式记录每个 token 在输入
    EEG 上的最大左右上下文与总结构感受野。
    """

    def __init__(
        self,
        in_channels: int = 64,
        latent_dim: int = 48,
        num_blocks: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
        dilations: Sequence[int] | None = None,
        expansion_ratio: int = 2,
        encoder_type: str = "cnn",
        window_size: int = 4,
        time_point_count: int | None = None,
        num_heads: int = 4,
        shift_size: int = 0,
        attention_pattern: str = "right_local",
        position_encoding_mode: str = "sinusoidal_qk",
        boundary_mode: str = "truncate",
        hidden_dim: int | None = None,
        boundary_noise_seed: int = 42,
    ) -> None:
        """按配置构建 EEG 编码器。

        Args:
            in_channels: 输入电极数 ``C``。
            latent_dim: 编码后的特征维度 ``D``。
            num_blocks: 时间编码 block 数。
            kernel_size: CNN 深度卷积核大小，必须为奇数。
            dropout: block 内 dropout 概率。
            dilations: 每个 CNN block 的 dilation，长度须等于 block 数。
            expansion_ratio: CNN pointwise/Transformer FFN 的扩展倍数。
            encoder_type: ``cnn``、``window_transformer`` 或 ``patch_mlp``。
            window_size: Transformer/patch MLP 的局部连续时间点总数 ``W``。
            time_point_count: 数据实际时间点数 ``T``；窗口后端必须提供。
            num_heads: multi-head attention 的 head 数。
            shift_size: 必须为 0；滑动局部窗口不使用 shifted window。
            attention_pattern: Transformer 的 ``right_local`` 或 ``centered_local``。
            position_encoding_mode: Transformer 的 ``sinusoidal_qk`` 或 ``none``。
            boundary_mode: ``truncate``、``reflect`` 或仅补 K/V 的 ``noise_pad``。
            boundary_noise_seed: noise_pad 验证/导出的稳定 trial 采样种子。
            hidden_dim: patch MLP 的中间维度；该后端必须显式提供。
        """
        super().__init__()
        self.encoder_type = normalize_encoder_type(encoder_type)
        normalized_attention_pattern = normalize_attention_pattern(attention_pattern)
        normalized_position_encoding_mode = normalize_position_encoding_mode(
            position_encoding_mode
        )
        normalized_boundary_mode = normalize_boundary_mode(boundary_mode)
        self.boundary_mode = normalized_boundary_mode
        self.boundary_noise_seed = normalize_boundary_noise_seed(boundary_noise_seed)
        self.shift_size = _nonnegative_int(shift_size, name="shift_size")
        if self.shift_size != 0:
            raise ValueError(
                "shift_size must be 0; local temporal encoders do not use shifts"
            )
        self.in_channels = _positive_int(in_channels, name="in_channels")
        self.latent_dim = _positive_int(latent_dim, name="latent_dim")
        self.num_blocks = _positive_int(num_blocks, name="num_blocks")
        self.expansion_ratio = _positive_int(
            expansion_ratio,
            name="expansion_ratio",
        )
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1]")
        # 1x1 卷积逐时间点执行 C -> D 投影，不在此阶段混合相邻时间点。
        self.spatial_projection = nn.Sequential(
            nn.Conv1d(
                self.in_channels,
                self.latent_dim,
                kernel_size=1,
                bias=False,
            ),
            ChannelLayerNorm(self.latent_dim),
            nn.GELU(),
        )
        if self.encoder_type == "cnn":
            if normalized_boundary_mode != "truncate":
                raise ValueError("non-default boundary_mode is only supported for window_transformer")
            if hidden_dim is not None:
                raise ValueError("hidden_dim is only valid for patch_mlp")
            if dilations is None:
                dilations = [1] * self.num_blocks
            if len(dilations) != self.num_blocks:
                raise ValueError("dilations length must match num_blocks")
            self.kernel_size = _positive_int(kernel_size, name="kernel_size")
            self.dilations = tuple(
                _positive_int(dilation, name="dilation")
                for dilation in dilations
            )
            self.window_size = None
            self.time_point_count = None
            self.num_heads = None
            self.hidden_dim = None
            self.attention_pattern = None
            self.position_encoding_mode = None
            temporal_blocks: list[nn.Module] = [
                TemporalResidualBlock(
                    channels=self.latent_dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout,
                    expansion_ratio=self.expansion_ratio,
                )
                for dilation in self.dilations
            ]
            context_radius = (self.kernel_size - 1) * sum(self.dilations) // 2
            self.left_context_samples = context_radius
            self.right_context_samples = context_radius
        elif self.encoder_type == "patch_mlp":
            if normalized_boundary_mode != "truncate":
                raise ValueError("non-default boundary_mode is only supported for window_transformer")
            if self.num_blocks != 1:
                raise ValueError("patch_mlp requires num_blocks=1")
            if hidden_dim is None:
                raise ValueError("patch_mlp requires hidden_dim")
            self.window_size = _positive_int(window_size, name="window_size")
            if self.window_size not in {3, 5, 7}:
                raise ValueError("patch_mlp window_size must be one of: 3, 5, 7")
            self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
            self.time_point_count = None
            self.num_heads = None
            self.kernel_size = None
            self.dilations = None
            self.attention_pattern = None
            self.position_encoding_mode = None
            self.register_buffer(
                "patch_mlp_architecture_signature",
                torch.zeros(
                    self.window_size,
                    self.hidden_dim,
                    1,
                    dtype=torch.uint8,
                ),
                persistent=True,
            )
            temporal_blocks = [
                TemporalPatchMLPBlock(
                    channels=self.latent_dim,
                    window_size=self.window_size,
                    hidden_dim=self.hidden_dim,
                    dropout=self.dropout,
                )
            ]
            context_radius = self.window_size // 2
            self.left_context_samples = context_radius
            self.right_context_samples = context_radius
        else:
            if hidden_dim is not None:
                raise ValueError("hidden_dim is only valid for patch_mlp")
            self.window_size = _positive_int(
                window_size,
                name="window_size",
            )
            self.hidden_dim = None
            self.attention_pattern = normalized_attention_pattern
            self.position_encoding_mode = normalized_position_encoding_mode
            self.boundary_mode = normalized_boundary_mode
            if self.boundary_mode == "noise_pad" and self.attention_pattern != "centered_local":
                raise ValueError("noise_pad boundary_mode requires centered_local attention")
            if self.boundary_mode == "reflect":
                if self.attention_pattern != "centered_local":
                    raise ValueError("reflect boundary_mode requires centered_local attention")
                if self.position_encoding_mode != "none":
                    raise ValueError(
                        "reflect boundary_mode currently requires position_encoding_mode=none"
                    )
            if self.attention_pattern == "centered_local" and self.window_size % 2 != 1:
                raise ValueError("centered_local window_size must be odd")
            if time_point_count is None:
                raise ValueError(
                    "time_point_count is required for window_transformer"
                )
            self.time_point_count = _positive_int(
                time_point_count,
                name="time_point_count",
            )
            self.num_heads = _positive_int(num_heads, name="num_heads")
            if self.latent_dim % self.num_heads != 0:
                raise ValueError("num_heads must divide latent_dim")
            self.kernel_size = None
            self.dilations = None
            # MultiheadAttention 的权重形状本身不编码 window、head 或 mask
            # 语义。旧 right+sine 保持历史 buffer 名；其余组合用 pattern 与
            # position mode 专属键，使 strict load 无法混接不同几何结构。
            signature_name = (
                "right_local_architecture_signature"
                if self.attention_pattern == "right_local"
                and self.position_encoding_mode == "sinusoidal_qk"
                else (
                    f"{self.attention_pattern}_{self.position_encoding_mode}"
                    "_architecture_signature"
                )
            )
            if self.boundary_mode != "truncate":
                signature_name = (
                    f"{self.attention_pattern}_{self.position_encoding_mode}_"
                    f"{self.boundary_mode}_architecture_signature"
                )
            self.register_buffer(
                signature_name,
                torch.zeros(
                    self.window_size,
                    self.num_heads,
                    1,
                    dtype=torch.uint8,
                ),
                persistent=True,
            )
            temporal_blocks = [
                RightLocalTransformerBlock(
                    channels=self.latent_dim,
                    window_size=self.window_size,
                    time_point_count=self.time_point_count,
                    num_heads=self.num_heads,
                    dropout=self.dropout,
                    expansion_ratio=self.expansion_ratio,
                    attention_pattern=self.attention_pattern,
                    position_encoding_mode=self.position_encoding_mode,
                    boundary_mode=self.boundary_mode,
                )
                for _ in range(self.num_blocks)
            ]
            if self.attention_pattern == "right_local":
                self.left_context_samples = 0
                self.right_context_samples = self.num_blocks * (
                    self.window_size - 1
                )
            else:
                context_radius = self.num_blocks * (self.window_size // 2)
                self.left_context_samples = context_radius
                self.right_context_samples = context_radius
        self.receptive_field_samples = (
            self.left_context_samples + 1 + self.right_context_samples
        )
        # 三种后端都保持输入时间长度，并可按 block 输出中间 latent。
        self.temporal_blocks = nn.Sequential(*temporal_blocks)
        self.output_norm = ChannelLayerNorm(self.latent_dim)

    def sample_boundary_noise(
        self, eeg: torch.Tensor, *, trial_ids: torch.Tensor | None = None,
        training: bool | None = None,
    ) -> torch.Tensor | None:
        """Sample independent N(0,1) context in normalized EEG coordinates.

        Train draws use the checkpointed PyTorch RNG. Eval draws are keyed by
        stable trial IDs, independent of batch order, size, device and global RNG.
        No current-trial moments or labels are used to construct the noise.
        Returned [B,C,2r] holds left r samples followed by right r samples.
        """
        if self.boundary_mode != "noise_pad":
            return None
        if eeg.ndim != 3 or eeg.shape[1] != self.in_channels or not eeg.is_floating_point():
            raise ValueError("noise_pad expects floating EEG [batch, in_channels, time]")
        batch_size = eeg.shape[0]
        radius = self.window_size // 2
        shape = (batch_size, self.in_channels, 2 * radius)
        if not radius or not batch_size:
            return eeg.new_empty(shape)
        resample = self.training if training is None else training
        if resample:
            return torch.randn(shape, device=eeg.device, dtype=torch.float32).to(dtype=eeg.dtype)
        if (
            not isinstance(trial_ids, torch.Tensor) or trial_ids.ndim != 2
            or trial_ids.shape[0] != batch_size or trial_ids.shape[1] < 1
            or trial_ids.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        ):
            raise ValueError("noise_pad evaluation requires integer trial_ids [batch, identity_fields] or explicit boundary_noise_eeg")
        samples = []
        modulus = (1 << 63) - 1
        for identity in trial_ids.detach().cpu().tolist():
            seed = self.boundary_noise_seed
            for component in identity:
                seed = (seed * 6364136223846793005 + int(component) + 1442695040888963407) % modulus
            generator = torch.Generator(device="cpu").manual_seed(seed)
            samples.append(torch.randn((self.in_channels, 2 * radius), generator=generator, dtype=torch.float32))
        return torch.stack(samples).to(device=eeg.device, dtype=eeg.dtype)

    def resolved_config(self) -> dict[str, object]:
        """返回规范化后的实际结构配置，供日志和 checkpoint 元数据记录。

        只输出当前后端有意义的字段，并固化解析后的左右上下文与总 RF。
        :func:`build_eeg_encoder` 会验证这些派生字段后再重建模型。
        """
        config: dict[str, object] = {
            "type": self.encoder_type,
            "in_channels": self.in_channels,
            "latent_dim": self.latent_dim,
            "num_blocks": self.num_blocks,
            "dropout": self.dropout,
            "left_context_samples": self.left_context_samples,
            "right_context_samples": self.right_context_samples,
            "receptive_field_samples": self.receptive_field_samples,
        }
        if self.encoder_type == "cnn":
            config.update(
                {
                    "kernel_size": self.kernel_size,
                    "dilations": list(self.dilations or ()),
                    "expansion_ratio": self.expansion_ratio,
                }
            )
        elif self.encoder_type == "patch_mlp":
            config.update(
                {
                    "window_size": self.window_size,
                    "hidden_dim": self.hidden_dim,
                }
            )
        else:
            config.update(
                {
                    "window_size": self.window_size,
                    "time_point_count": self.time_point_count,
                    "num_heads": self.num_heads,
                    "expansion_ratio": self.expansion_ratio,
                    "attention_pattern": self.attention_pattern,
                    "position_encoding_mode": self.position_encoding_mode,
                    "boundary_mode": self.boundary_mode,
                    "shift_size": self.shift_size,
                }
            )
            if self.boundary_mode == "noise_pad":
                config["boundary_noise_seed"] = self.boundary_noise_seed
        return config

    def forward_features(
        self,
        eeg: torch.Tensor,
        subject_id: torch.Tensor | None = None,
        subject_adapters: Sequence[nn.Module] | None = None,
        subject_condition: torch.Tensor | None = None,
        boundary_noise_eeg: torch.Tensor | None = None,
        boundary_trial_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """编码 EEG，并返回最终及逐 block 的时间表征。

        Args:
            eeg: ``[B,C,T]`` 输入；正常 EEG 与增强噪声使用同一数值路径。
            subject_id: 为调用接口保留；当前 block 适配直接使用已生成的
                ``subject_condition``，本参数不参与本方法计算。
            subject_adapters: 可选的逐 block subject AdaLN 模块，数量必须与
                temporal block 相同。
            subject_condition: ``[B,S]`` 被试条件，传给每个 adapter。
            boundary_noise_eeg: 可选的标准化 EEG 空间噪声 ``[B,C,2r]``；
                只作局部 K/V 上下文，不进入任何输出时间轴。
            boundary_trial_ids: eval 自动采样时使用的稳定整数身份 ``[B,N]``。

        Returns:
            字典 ``latent`` 为 output norm 后的 ``[B,T,D]``；
            ``block_latents`` 是长度为 block 数的列表，每项也是 ``[B,T,D]``，
            且若启用 subject adapter，保存的是适配后的表示。
        """
        if eeg.ndim != 3:
            raise ValueError(f"Expected [batch, channel, time] EEG, got {eeg.shape}")
        if subject_adapters is not None:
            if len(subject_adapters) != len(self.temporal_blocks):
                raise ValueError(
                    "subject_adapters length must match encoder temporal block count"
                )
            if subject_condition is None:
                raise ValueError(
                    "subject_condition is required when blockwise subject adapters are enabled"
                )

        # eeg:[B,C,T] C=电极 T=时间点；subject_condition:[B,S] 可选
        x = self.spatial_projection(eeg)  # [B,D,T] D=latent_dim；只混通道
        boundary_latent = None
        if self.boundary_mode == "noise_pad":
            if boundary_noise_eeg is None:
                boundary_noise_eeg = self.sample_boundary_noise(eeg, trial_ids=boundary_trial_ids)
            radius = self.window_size // 2
            expected = (eeg.shape[0], self.in_channels, 2 * radius)
            if not isinstance(boundary_noise_eeg, torch.Tensor) or tuple(boundary_noise_eeg.shape) != expected:
                raise ValueError(f"boundary_noise_eeg must have shape {expected}")
            if boundary_noise_eeg.device != eeg.device or boundary_noise_eeg.dtype != eeg.dtype:
                raise ValueError("boundary_noise_eeg must match EEG device and dtype")
            if radius:
                # 同一次前向的各层复用同一份投影噪声；pad 没有 Q、残差输出或
                # FFN 输出，不会从真实 token 读取内容再传播到下一层。
                boundary_latent = self.spatial_projection(boundary_noise_eeg)
        elif boundary_noise_eeg is not None or boundary_trial_ids is not None:
            raise ValueError("boundary noise inputs require noise_pad boundary_mode")
        block_latents: list[torch.Tensor] = []  # 每 block 后 [B,T,D]，给 time_source_block
        for block_index, block in enumerate(self.temporal_blocks):
            if self.boundary_mode == "noise_pad":
                x = block(x, boundary_latent=boundary_latent)
            else:
                x = block(x)  # 仍 [B,D,T]（CNN）或内部转成 [B,T,D] 再转回
            latent = x.transpose(1, 2)  # [B,T,D]
            if subject_adapters is not None:
                # subject_condition 调制 latent；下一 block 吃调制后的 x
                latent = subject_adapters[block_index](latent, subject_condition)  # [B,T,D]
                x = latent.transpose(1, 2)  # 回 [B,D,T] 给下一卷积块
            block_latents.append(latent)
        x = self.output_norm(x)  # 仍 [B,D,T]
        return {
            "latent": x.transpose(1, 2),  # [B,T,D] 默认 decoder 读这个
            "block_latents": block_latents,  # 未过 output_norm（但已 AdaLN）
        }

    def forward(
        self, eeg: torch.Tensor, *, boundary_noise_eeg: torch.Tensor | None = None,
        boundary_trial_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将 ``[B,C,T]`` EEG 编码为最终 ``[B,T,D]``。"""
        features = self.forward_features(
            eeg, boundary_noise_eeg=boundary_noise_eeg, boundary_trial_ids=boundary_trial_ids,
        )
        return features["latent"]


def build_eeg_encoder(
    config: Mapping[str, object],
    *,
    time_point_count: int | None = None,
) -> EEGEncoder:
    """从 YAML 风格映射构建 :class:`EEGEncoder`。

    同时兼容 ``type``/``encoder_type``、``num_layers``/``num_blocks`` 和
    ``ffn_ratio``/``expansion_ratio``。同义字段若同时出现但值冲突会立即报错，
    避免日志配置与实际结构不一致。调用方可用关键字参数注入数据解析出的
    ``time_point_count``；若 YAML 同时提供不同值则立即报错。resolved config
    中的 RF 派生字段只用于一致性校验，不会作为可修改的结构输入。
    """
    values = dict(config)
    yaml_type = values.pop("type", None)
    explicit_type = values.pop("encoder_type", None)
    if yaml_type is not None and explicit_type is not None:
        if normalize_encoder_type(yaml_type) != normalize_encoder_type(explicit_type):
            raise ValueError("encoder.type and encoder.encoder_type disagree")
    values["encoder_type"] = (
        explicit_type if explicit_type is not None else yaml_type or "cnn"
    )
    if time_point_count is not None:
        resolved_time_point_count = _positive_int(
            time_point_count,
            name="time_point_count",
        )
        configured_time_point_count = values.get("time_point_count")
        if (
            configured_time_point_count is not None
            and _positive_int(
                configured_time_point_count,
                name="time_point_count",
            )
            != resolved_time_point_count
        ):
            raise ValueError(
                "encoder.time_point_count does not match data time_point_count"
            )
        values["time_point_count"] = resolved_time_point_count
    aliases = {
        "num_layers": "num_blocks",
        "ffn_ratio": "expansion_ratio",
        "window_partition": "attention_pattern",
    }
    for alias, canonical in aliases.items():
        if alias not in values:
            continue
        if canonical in values and values[canonical] != values[alias]:
            raise ValueError(f"encoder.{alias} and encoder.{canonical} disagree")
        values[canonical] = values.pop(alias)
    declared_metadata = {
        name: values.pop(name)
        for name in (
            "left_context_samples",
            "right_context_samples",
            "receptive_field_samples",
        )
        if name in values
    }
    encoder = EEGEncoder(**values)
    for name, raw_value in declared_metadata.items():
        expected = (
            _positive_int(raw_value, name=name)
            if name == "receptive_field_samples"
            else _nonnegative_int(raw_value, name=name)
        )
        actual = int(getattr(encoder, name))
        if expected != actual:
            raise ValueError(
                f"encoder.{name} does not match resolved architecture: "
                f"{expected} != {actual}"
            )
    return encoder
