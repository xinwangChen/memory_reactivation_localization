"""Optional shared subject condition with train-only individual residuals."""
from collections.abc import Sequence
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


def normalize_subject_conditioning_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in {"individual", "shared_plus_residual"}:
        raise ValueError("subject_conditioning.mode must be individual or shared_plus_residual")
    return mode


def normalize_training_subject_ids(subject_ids: Sequence[int] | None) -> tuple[int, ...]:
    if (not isinstance(subject_ids, Sequence) or isinstance(subject_ids, (str, bytes))
            or not subject_ids or any(isinstance(s, bool) or not isinstance(s, Integral) or s < 0 for s in subject_ids)):
        raise ValueError("shared_plus_residual requires nonempty nonnegative integer training_subject_ids")
    ids = tuple(int(s) for s in subject_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("training_subject_ids must be unique")
    return tuple(sorted(ids))


class SharedResidualSubjectEmbedding(nn.Embedding):
    """Return shared + dropped individual residual; unseen IDs receive shared only.

    Keep ``weight`` at its historical checkpoint path. The additional parameter and
    persistent training-ID buffer distinguish this architecture from old embeddings.
    ``condition_dropout`` drops an entire residual before addition, without rescaling.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, training_subject_ids: Sequence[int]):
        ids = normalize_training_subject_ids(training_subject_ids)
        if num_embeddings <= max(ids):
            raise ValueError("subject embedding capacity must cover training_subject_ids")
        super().__init__(num_embeddings, embedding_dim)
        nn.init.zeros_(self.weight)
        self.shared_condition = nn.Parameter(torch.empty(embedding_dim))
        nn.init.normal_(self.shared_condition, mean=0., std=embedding_dim ** -.5)
        self.register_buffer("training_subject_ids", torch.tensor(ids, dtype=torch.long))

    def forward(self, subject_id: torch.Tensor, *, condition_dropout: float = 0.) -> torch.Tensor:
        if subject_id.ndim != 1 or subject_id.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("subject_id must be a one-dimensional integer tensor")
        if not 0. <= condition_dropout <= 1.:
            raise ValueError("condition_dropout must be between 0 and 1")
        ids = subject_id.to(device=self.weight.device, dtype=torch.long)
        known = (ids[:, None] == self.training_subject_ids[None, :]).any(dim=1)
        # Never index an unknown ID, even if it is negative or outside the table.
        safe_ids = torch.where(known, ids, torch.zeros_like(ids))
        residual = F.embedding(safe_ids, self.weight)
        residual = torch.where(known[:, None], residual, torch.zeros_like(residual))
        if self.training and condition_dropout > 0.:
            keep = torch.rand(ids.shape[0], 1, device=residual.device) >= condition_dropout
            residual = residual * keep.to(dtype=residual.dtype)
        return self.shared_condition[None, :] + residual

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        saved_ids = state_dict.get(prefix + "training_subject_ids")
        if saved_ids is not None and (not isinstance(saved_ids, torch.Tensor)
                or saved_ids.dtype != torch.long
                or not torch.equal(saved_ids.to(self.training_subject_ids.device), self.training_subject_ids)):
            error_msgs.append(prefix + "training_subject_ids differ from the configured training subject mapping")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


def build_subject_embedding(n_subjects: int | None, subject_dim: int, *, mode: str,
                            training_subject_ids: Sequence[int] | None = None) -> nn.Embedding:
    mode = normalize_subject_conditioning_mode(mode)
    if mode == "shared_plus_residual":
        ids = normalize_training_subject_ids(training_subject_ids)
        capacity = max(ids) + 1 if n_subjects is None else int(n_subjects)
        return SharedResidualSubjectEmbedding(capacity, subject_dim, ids)
    if n_subjects is None or int(n_subjects) < 1:
        raise ValueError("n_subjects must be positive when subject conditioning or query attention is enabled")
    # Preserve the old initialization, state keys and RNG consumption exactly.
    embedding = nn.Embedding(int(n_subjects), subject_dim)
    nn.init.normal_(embedding.weight, mean=0., std=subject_dim ** -.5)
    return embedding


class SubjectAdaLNZero(nn.Module):
    """在 encoder block 后注入被试条件的零初始化 AdaLN。

    输入 latent 为 ``[B,T,D]``，被试条件为 ``[B,S]``。条件网络生成逐 trial
    的缩放 ``gamma`` 与平移 ``beta``，并广播到全部时间点。最后一层零初始化
    使模块初始时严格等价于恒等映射，避免刚启用 subject conditioning 就改变
    已有编码器表示；训练后每个被试可学习自己的调制。
    """

    def __init__(self, latent_dim: int, condition_dim: int, dropout: float = 0.0) -> None:
        """构建 ``S -> 2D`` 的条件调制网络。"""
        super().__init__()
        if int(condition_dim) < 1:
            raise ValueError("subject condition_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.condition_dim = int(condition_dim)
        self.norm = nn.LayerNorm(self.latent_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.condition_dim, self.latent_dim * 2),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, latent: torch.Tensor, subject_condition: torch.Tensor) -> torch.Tensor:
        """用 ``[B,S]`` 条件调制 ``[B,T,D]``，输出形状保持不变。"""
        if latent.ndim != 3 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected [batch, time, {self.latent_dim}] latent, got {latent.shape}")
        if subject_condition.ndim != 2 or subject_condition.shape != (latent.shape[0], self.condition_dim):
            raise ValueError(
                f"Expected [batch, {self.condition_dim}] subject_condition, got {subject_condition.shape}"
            )
        gamma, beta = self.modulation(subject_condition).chunk(2, dim=-1)
        # gamma/beta 的 [B,D] 在时间维广播；残差项保留未调制的 latent。
        return latent + gamma[:, None, :] * self.norm(latent) + beta[:, None, :]
