"""Attention scoring, gating and pooling. State names match historical models."""
from __future__ import annotations
import torch
from torch import nn

ATTENTION_MODES = {"scorer", "gated_scorer", "query", "gated_query"}

ATTENTION_QUERY_SOURCES = {"subject"}

POOLED_HEAD_MODES = {"auto", "shared_real_class"}

GATE_ACTIVATIONS = {"sigmoid", "tanh", "tanh_plus_one"}

GATE_COMBINATIONS = {"multiply", "add"}

EFFECTIVE_NORMALIZATIONS = {"none", "one_plus_mass", "softmax"}

def _validate_gate_options(activation: str, combination: str) -> None:
    if not isinstance(activation, str) or activation not in GATE_ACTIVATIONS:
        raise ValueError("attention_gate_activation must be sigmoid, tanh or tanh_plus_one")
    if not isinstance(combination, str) or combination not in GATE_COMBINATIONS:
        raise ValueError("attention_gate_combination must be multiply or add")

def _validate_effective_normalization(mode: str, activation: str) -> None:
    if not isinstance(mode, str) or mode not in EFFECTIVE_NORMALIZATIONS:
        raise ValueError("attention_effective_normalization must be none, one_plus_mass or softmax")
    if mode == "one_plus_mass" and activation == "tanh":
        raise ValueError("one_plus_mass requires nonnegative attention gates (sigmoid or tanh_plus_one)")

def _combine_gate_weights(
    weights: torch.Tensor, gate: torch.Tensor, combination: str,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    effective = weights * gate if combination == "multiply" else weights + gate
    if combination == "add" and valid_mask is not None:
        effective = effective.masked_fill(~valid_mask.to(device=weights.device, dtype=torch.bool), 0.0)
    return effective

def _masked_attention_softmax(
    scores: torch.Tensor,
    valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """在最后一维执行 softmax，并把 pool-invalid 时间点严格置零。"""
    if valid_mask is None:
        return torch.softmax(scores, dim=-1)
    if valid_mask.ndim == 1:
        if valid_mask.shape[0] != scores.shape[-1]:
            raise ValueError(
                "attention valid_mask must align with the time dimension: "
                f"{valid_mask.shape} vs {scores.shape}"
            )
        valid = valid_mask[None, :].expand_as(scores)
    elif valid_mask.shape == scores.shape:
        valid = valid_mask
    else:
        raise ValueError(
            "attention valid_mask must have shape [time] or [batch, time], "
            f"got {valid_mask.shape} for scores {scores.shape}"
        )
    valid = valid.to(device=scores.device, dtype=torch.bool)
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("attention valid_mask must retain at least one time point per trial")
    masked_scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    return weights.masked_fill(~valid, 0.0)

class TimeAttentionPooling(nn.Module):
    """用内容打分器把时间序列汇聚为每个 trial 一个向量。

    对 ``[B,T,D]`` 的每个时间点生成标量 score ``[B,T]``，在 T 维 softmax
    得到和为 1 的 attention weight，再加权求和得到 ``[B,D]``。最后一层
    零初始化使训练开始时 attention 为均匀分布。
    """

    def __init__(
        self,
        latent_dim: int,
        attention_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        """构建 ``D -> attention_dim -> 1`` 的逐时间点打分器。"""
        super().__init__()
        self.latent_dim = int(latent_dim)
        hidden_dim = int(attention_dim) if attention_dim is not None else self.latent_dim
        if hidden_dim < 1:
            raise ValueError("attention_dim must be positive")
        self.score = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def _attention_input(
        self,
        latent: torch.Tensor,
        branch_latent: torch.Tensor | None,
        *,
        name: str,
    ) -> torch.Tensor:
        """允许 attention 分支读取单独特征，汇聚值始终使用原始 latent。"""
        if branch_latent is None:
            return latent
        if not isinstance(branch_latent, torch.Tensor) or branch_latent.shape != latent.shape:
            raise ValueError(f"{name} must have the same shape as latent")
        if branch_latent.device != latent.device or branch_latent.dtype != latent.dtype:
            raise ValueError(f"{name} must have the same device and dtype as latent")
        return branch_latent

    def forward(
        self,
        latent: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        score_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 pooled、weights、scores；score_latent 仅影响打分路径。"""
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected [batch, time, {self.latent_dim}] latent, got {latent.shape}")
        if latent.shape[1] == 0:
            raise ValueError("Cannot apply attention over an empty time axis")
        scores = self.score(self._attention_input(latent, score_latent, name="score_latent")).squeeze(-1)
        weights = _masked_attention_softmax(scores, valid_mask)
        pooled = torch.bmm(weights[:, None, :], latent).squeeze(1)
        return pooled, weights, scores

class GatedTimeAttentionPooling(TimeAttentionPooling):
    """在 softmax 后组合逐时间点 gate，默认保留历史 sigmoid 乘法。

    softmax weight 表示时间点间的相对分配，``sigmoid(gate_proj)`` 表示该点
    的绝对保留强度；两者乘积为 ``effective_weights``。该乘积不再强制和为
    1。探索模式可使用 tanh / tanh+1，并在 softmax 后相加或相乘；组合后可
    独立选择 one_plus_mass 缩放或 softmax。未再次softmax的tanh可产生带符号汇聚权重；其 Top-q 和质量约束由 decoder 提供
    的绝对值权重驱动。attention dropout 只用于实际 pooling。
    """

    def __init__(
        self,
        latent_dim: int,
        attention_dim: int | None = None,
        dropout: float = 0.0,
        gate_init_bias: float = 2.0,
        *,
        gate_activation: str = "sigmoid",
        gate_combination: str = "multiply",
        effective_normalization: str = "none",
    ) -> None:
        """构建 scorer 与只经过一次激活的 ``D -> 1`` gate。"""
        super().__init__(latent_dim=latent_dim, attention_dim=attention_dim, dropout=0.0)
        _validate_gate_options(gate_activation, gate_combination)
        _validate_effective_normalization(effective_normalization, gate_activation)
        self.gate_activation = gate_activation
        self.gate_combination = gate_combination
        self.effective_normalization = effective_normalization
        self.attention_dropout = nn.Dropout(float(dropout))
        self.gate_proj = nn.Linear(self.latent_dim, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_init_bias))

    def forward(
        self,
        latent: torch.Tensor,
        *,
        valid_mask: torch.Tensor | None = None,
        score_latent: torch.Tensor | None = None,
        gate_latent: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 pooled、softmax weights、scores、gate 与 effective weights。

        除 pooled 为 ``[B,D]`` 外，其余四个时间量均为 ``[B,T]``。
        score_latent 与 gate_latent 分别只供 scorer、gate 使用；省略时各自
        读取原始 latent。实际汇聚值始终使用原始 latent。
        """
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected [batch, time, {self.latent_dim}] latent, got {latent.shape}")
        if latent.shape[1] == 0:
            raise ValueError("Cannot apply attention over an empty time axis")
        # latent: [B,Ta,D]（Ta=attention 窗长，可能 < 全 T）
        scores = self.score(self._attention_input(latent, score_latent, name="score_latent")).squeeze(-1)
        weights = _masked_attention_softmax(scores, valid_mask)  # [B,Ta] 相对份额，行和=1
        gate_input = self._attention_input(latent, gate_latent, name="gate_latent")
        gate_logits = self.gate_proj(gate_input).squeeze(-1)
        if self.gate_activation == "sigmoid":
            gate = torch.sigmoid(gate_logits)
        else:
            gate = torch.tanh(gate_logits)
            if self.gate_activation == "tanh_plus_one":
                gate = gate + 1.0
        effective_weights = _combine_gate_weights(weights, gate, self.gate_combination, valid_mask)
        if self.effective_normalization != "none":
            # 后置处理均沿每条 trial 的有效时间轴计算并保留梯度。
            # 低精度先在 float32 中执行缩放/softmax。
            reference = (
                effective_weights.float()
                if effective_weights.dtype in (torch.float16, torch.bfloat16)
                else effective_weights
            )
            if self.effective_normalization == "one_plus_mass":
                # 此分支保留整段权重趋近零的能力。
                normalized = reference / (1.0 + reference.sum(dim=-1, keepdim=True))
            else:
                # 对合成后的w执行softmax；pool-invalid位置仍严格排除。
                normalized = _masked_attention_softmax(reference, valid_mask)
            effective_weights = normalized.to(effective_weights.dtype)
        pooled_weights = self.attention_dropout(effective_weights)  # [B,Ta] 只影响下面 pooling
        pooled = torch.bmm(pooled_weights[:, None, :], latent).squeeze(1)  # [B,D] trial 表征
        return pooled, weights, scores, gate, effective_weights

class QueryTimeAttentionPooling(nn.Module):
    """以被试 embedding 为 query 的 QK 时间汇聚。

    ``Q=Linear(subject_condition)`` 得到 ``[B,A]``，
    ``K=Linear(latent)`` 得到 ``[B,T,A]``，``V`` 直接使用 ``[B,T,D]`` latent。
    缩放点积产生 ``[B,T]`` score，经 softmax 后汇聚为 ``[B,D]``。Q 投影
    零初始化，因而初始 attention 同样是均匀分布。
    """

    def __init__(
        self,
        latent_dim: int,
        subject_dim: int,
        attention_dim: int | None = None,
        query_source: str = "subject",
    ) -> None:
        """构建被试 query 与时间 latent key 的投影。"""
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.subject_dim = int(subject_dim)
        self.attention_dim = int(attention_dim) if attention_dim is not None else self.latent_dim
        if self.subject_dim < 1:
            raise ValueError("subject_dim must be positive")
        if self.attention_dim < 1:
            raise ValueError("attention_dim must be positive")
        if query_source not in ATTENTION_QUERY_SOURCES:
            valid = ", ".join(sorted(ATTENTION_QUERY_SOURCES))
            raise ValueError(f"Unknown attention query_source: {query_source}; expected one of {valid}")
        self.query_source = query_source
        self.q_proj = nn.Linear(self.subject_dim, self.attention_dim)
        self.k_proj = nn.Linear(self.latent_dim, self.attention_dim)
        nn.init.zeros_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)

    def _validate_inputs(
        self,
        latent: torch.Tensor,
        subject_condition: torch.Tensor | None,
        *,
        mode_name: str,
    ) -> None:
        """验证 latent ``[B,T,D]`` 与被试条件 ``[B,S]`` 的形状和存在性。"""
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected [batch, time, {self.latent_dim}] latent, got {latent.shape}")
        if latent.shape[1] == 0:
            raise ValueError("Cannot apply attention over an empty time axis")
        if subject_condition is None:
            raise ValueError(f"subject_condition is required when attention_mode is {mode_name}")
        if subject_condition.ndim != 2 or subject_condition.shape != (latent.shape[0], self.subject_dim):
            raise ValueError(
                f"Expected [batch, {self.subject_dim}] subject_condition, got {subject_condition.shape}"
            )

    def _query_scores(self, latent: torch.Tensor, subject_condition: torch.Tensor) -> torch.Tensor:
        """计算缩放 QK 点积，返回每个 trial 的 ``[B,T]`` 时间分数。"""
        query = self.q_proj(subject_condition)
        keys = self.k_proj(latent)
        return torch.bmm(keys, query[:, :, None]).squeeze(-1) * (self.attention_dim ** -0.5)

    def forward(
        self,
        latent: torch.Tensor,
        *,
        subject_condition: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 pooled ``[B,D]``、weights ``[B,T]`` 与 scores ``[B,T]``。"""
        self._validate_inputs(latent, subject_condition, mode_name="query")
        assert subject_condition is not None
        scores = self._query_scores(latent, subject_condition)
        weights = _masked_attention_softmax(scores, valid_mask)
        pooled = torch.bmm(weights[:, None, :], latent).squeeze(1)
        return pooled, weights, scores

class GatedQueryTimeAttentionPooling(QueryTimeAttentionPooling):
    """带逐时间点 sigmoid gate 的被试 query attention。

    QK softmax weight 与由 latent 生成的 gate 相乘得到 effective weight；后者
    是 Top-q 的排序量。与 gated scorer 一样，dropout 只施加在送入加权求和的
    权重副本上，不改变返回给训练和导出逻辑的 effective weight。
    """

    def __init__(
        self,
        latent_dim: int,
        subject_dim: int,
        attention_dim: int | None = None,
        dropout: float = 0.0,
        gate_init_bias: float = 2.0,
        query_source: str = "subject",
    ) -> None:
        """构建 query/key 投影及 ``D -> 1`` gate。"""
        super().__init__(
            latent_dim=latent_dim,
            subject_dim=subject_dim,
            attention_dim=attention_dim,
            query_source=query_source,
        )
        self.attention_dropout = nn.Dropout(float(dropout))
        self.gate_proj = nn.Linear(self.latent_dim, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, float(gate_init_bias))

    def forward(
        self,
        latent: torch.Tensor,
        *,
        subject_condition: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 pooled ``[B,D]`` 及四个 ``[B,T]`` attention 相关张量。"""
        self._validate_inputs(latent, subject_condition, mode_name="gated_query")
        assert subject_condition is not None
        query = self.q_proj(subject_condition)
        keys = self.k_proj(latent)
        scores = torch.bmm(keys, query[:, :, None]).squeeze(-1) * (self.attention_dim ** -0.5)
        weights = _masked_attention_softmax(scores, valid_mask)
        gate = torch.sigmoid(self.gate_proj(latent).squeeze(-1))
        effective_weights = weights * gate
        pooled_weights = self.attention_dropout(effective_weights)
        pooled = torch.bmm(pooled_weights[:, None, :], latent).squeeze(1)
        return pooled, weights, scores, gate, effective_weights

