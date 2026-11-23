"""training.diagnostics: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from typing import Mapping, Sequence
import torch
import torch.nn.functional as F
from memory_reactivation.training.objectives.target import TARGET_AUGMENTATION_PROBE_RATIOS, _topk_count


@torch.no_grad()
def target_augmentation_diagnostic_totals(
    time_logits: torch.Tensor,
    attention_weights: torch.Tensor,
    attention_indices: torch.Tensor,
    covered_time_mask: torch.Tensor | None,
    *,
    configured_topk_count: int,
    configured_top_indices: torch.Tensor | None = None,
    attention_gate: torch.Tensor | None = None,
    invalid_core_time_mask: torch.Tensor | None = None,
    no_class_index: int | None = None,
    probe_ratios: Sequence[float] = TARGET_AUGMENTATION_PROBE_RATIOS,
    selection_valid_mask: torch.Tensor | None = None,
    roll_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """累计一批不参与反传的增强/注意力诊断分子与分母。

    同时统计正式 Top-q 和固定 q10/q20/q30 探针：噪声覆盖召回率、covered
    BCE、effective attention 质量及 gate。函数返回设备端标量，epoch 末才统一
    搬到 CPU，避免每个 batch 因 .item() 产生同步。
    """
    selected_logits = time_logits.detach().index_select(dim=1, index=attention_indices).float()
    weights = attention_weights.detach().float()
    if weights.shape != selected_logits.shape[:2]:
        raise ValueError(
            "attention_weights must align with diagnostic time logits: "
            f"got {weights.shape} vs {selected_logits.shape[:2]}"
        )
    if no_class_index is not None:
        no_class_index = int(no_class_index)
        if no_class_index != selected_logits.shape[-1] - 1:
            raise ValueError("no_class_index must be the final diagnostic logit class")
    batch_size, n_times = weights.shape
    if covered_time_mask is None:
        covered = torch.zeros((batch_size, n_times), dtype=torch.bool, device=weights.device)
    else:
        covered = covered_time_mask.detach().to(
            device=weights.device,
            dtype=torch.bool,
        ).index_select(dim=1, index=attention_indices)
    if covered.shape != weights.shape:
        raise ValueError(f"covered mask must align with attention weights: {covered.shape} vs {weights.shape}")
    if invalid_core_time_mask is None:
        # 直接调用时保留历史行为；训练主路径可显式传入 RF-aware invalid core，
        # 从而把原始输入覆盖与完整噪声感受野分别报告。
        invalid_core = covered
    else:
        invalid_core = invalid_core_time_mask.detach().to(
            device=weights.device,
            dtype=torch.bool,
        ).index_select(
            dim=1, index=attention_indices
        )
    if invalid_core.shape != weights.shape:
        raise ValueError(
            "invalid-core mask must align with attention weights: "
            f"{invalid_core.shape} vs {weights.shape}"
        )

    if attention_gate is None:
        gate = torch.ones_like(weights)
    else:
        gate = attention_gate.detach().float()
        if gate.shape != weights.shape:
            raise ValueError(f"attention gate must align with attention weights: {gate.shape} vs {weights.shape}")

    configured_k = max(1, min(n_times, int(configured_topk_count)))
    probe_counts = {
        f"q{int(round(float(ratio) * 100)):02d}": _topk_count(n_times, None, float(ratio))
        for ratio in probe_ratios
    }
    max_probe_k = max(probe_counts.values())

    def all_zero_bce(logits: torch.Tensor) -> torch.Tensor:
        """返回各位置相对全零 BCE 目标的平均损失。"""
        return F.softplus(logits).mean(dim=-1)

    def covered_target_bce(logits: torch.Tensor) -> torch.Tensor:
        """按当前 no-class 开关构造 covered 位置的诊断目标。"""
        if no_class_index is None:
            return all_zero_bce(logits)
        targets = torch.zeros_like(logits)
        targets[..., no_class_index] = 1.0
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        ).mean(dim=-1)

    valid_for_selection: torch.Tensor | None = None
    if selection_valid_mask is not None:
        if selection_valid_mask.ndim == 1 and selection_valid_mask.numel() == n_times:
            valid_for_selection = selection_valid_mask.to(
                device=weights.device, dtype=torch.bool
            )[None, :].expand(batch_size, -1)
        elif selection_valid_mask.shape == weights.shape:
            valid_for_selection = selection_valid_mask.to(
                device=weights.device, dtype=torch.bool
            )
        else:
            raise ValueError(
                "selection_valid_mask must have shape [time] or [batch, time] "
                "aligning with diagnostic attention weights"
            )
        if not bool(valid_for_selection.any(dim=1).all()):
            raise ValueError(
                "selection_valid_mask must retain at least one candidate per trial"
            )
    probe_weights = (
        weights
        if valid_for_selection is None
        else torch.where(
            valid_for_selection, weights, torch.full_like(weights, float("-inf"))
        )
    )
    if valid_for_selection is not None and selection_valid_mask.ndim == 1:
        configured_k = min(configured_k, int(valid_for_selection[0].sum().item()))
    ranked_indices = torch.topk(probe_weights, k=max_probe_k, dim=1).indices
    ranked_logits = selected_logits.gather(
        dim=1,
        index=ranked_indices[:, :, None].expand(-1, -1, selected_logits.shape[-1]),
    )
    ranked_covered_bce = covered_target_bce(ranked_logits)
    ranked_invalid_core_bce = all_zero_bce(ranked_logits)
    if configured_top_indices is None:
        configured_indices = torch.topk(
            probe_weights, k=configured_k, dim=1
        ).indices
    else:
        configured_indices = configured_top_indices.detach()
        if configured_indices.shape != (batch_size, configured_k):
            raise ValueError(
                "configured_top_indices must match the configured Top-q shape: "
                f"got {configured_indices.shape} vs {(batch_size, configured_k)}"
            )
    configured_logits = selected_logits.gather(
        dim=1,
        index=configured_indices[:, :, None].expand(-1, -1, selected_logits.shape[-1]),
    )
    configured_covered_bce = covered_target_bce(configured_logits)
    configured_invalid_core_bce = all_zero_bce(configured_logits)
    covered_logits = selected_logits.masked_select(
        covered[:, :, None].expand_as(selected_logits)
    ).reshape(-1, selected_logits.shape[-1])
    invalid_core_logits = selected_logits.masked_select(
        invalid_core[:, :, None].expand_as(selected_logits)
    ).reshape(-1, selected_logits.shape[-1])

    sample_count = weights.new_tensor(float(batch_size))
    covered_count = covered.sum()
    uncovered = ~covered
    uncovered_count = uncovered.sum()
    augmented_trials = covered.any(dim=1)
    augmented_trial_count = augmented_trials.sum()
    total_mass = weights.sum(dim=1)
    covered_mass_by_trial = (weights * covered).sum(dim=1)
    uncovered_mass_by_trial = (weights * uncovered).sum(dim=1)
    covered_mass_fraction = covered_mass_by_trial / total_mass.clamp_min(1e-12)
    invalid_core_count = invalid_core.sum()
    invalid_core_trials = invalid_core.any(dim=1)
    invalid_core_mass_by_trial = (weights * invalid_core).sum(dim=1)
    invalid_core_mass_fraction = invalid_core_mass_by_trial / total_mass.clamp_min(1e-12)

    totals: dict[str, torch.Tensor] = {
        "sample_count": sample_count,
        "selected_time_count": weights.new_tensor(float(batch_size * n_times)),
        "covered_count": covered_count,
        "uncovered_count": uncovered_count,
        "augmented_trial_count": augmented_trial_count,
        "all_covered_invalid_bce_sum": covered_target_bce(covered_logits).sum(),
        "invalid_core_count": invalid_core_count,
        "invalid_core_trial_count": invalid_core_trials.sum(),
        "all_invalid_core_bce_sum": all_zero_bce(invalid_core_logits).sum(),
        "effective_mass_sum": total_mass.sum(),
        "covered_effective_weight_sum": weights.masked_select(covered).sum(),
        "uncovered_effective_weight_sum": weights.masked_select(uncovered).sum(),
        "covered_effective_mass_sum": covered_mass_by_trial.sum(),
        "uncovered_effective_mass_sum": uncovered_mass_by_trial.sum(),
        "covered_effective_mass_fraction_sum": covered_mass_fraction.masked_select(augmented_trials).sum(),
        "covered_gate_sum": gate.masked_select(covered).sum(),
        "uncovered_gate_sum": gate.masked_select(uncovered).sum(),
        "invalid_core_effective_weight_sum": weights.masked_select(invalid_core).sum(),
        "invalid_core_effective_mass_sum": invalid_core_mass_by_trial.sum(),
        "invalid_core_effective_mass_fraction_sum": invalid_core_mass_fraction.masked_select(
            invalid_core_trials
        ).sum(),
        "invalid_core_gate_sum": gate.masked_select(invalid_core).sum(),
    }
    if roll_mask is not None:
        if not isinstance(roll_mask, torch.Tensor) or roll_mask.dtype != torch.bool or roll_mask.shape != (batch_size,):
            raise ValueError("roll_mask must contain one boolean per diagnostic trial")
        rolled = roll_mask.detach().to(device=weights.device)
        # Routing rates describe the entire input, even if attention uses a crop.
        replacement_trials = (
            covered_time_mask.detach().to(device=weights.device, dtype=torch.bool).any(dim=1)
            if covered_time_mask is not None else torch.zeros_like(rolled)
        )
        if bool((rolled & replacement_trials).any().item()):
            raise ValueError("Roll trials must not have noise-covered positions")
        totals["replacement_trial_count"] = replacement_trials.sum()
        totals["roll_trial_count"] = rolled.sum()
        totals["augmented_trial_count"] = replacement_trials.sum() + rolled.sum()

    def add_topk_totals(
        prefix: str,
        indices: torch.Tensor,
        topk_covered_bce: torch.Tensor,
        topk_invalid_core_bce: torch.Tensor,
    ) -> None:
        """把一组排名索引对应的 covered/invalid-core 统计并入 totals。

        2D 支持域下索引可能含无效 filler（该 trial 有效点不足 k）；分母按
        indices 中实际有效的位置数计算，而不是 batch_size * k，避免诊断口径
        被 filler 拉偏。
        """
        k = int(indices.shape[1])
        if valid_for_selection is None:
            valid_at = None
            selected_count = weights.new_tensor(float(batch_size * k))
        else:
            valid_at = valid_for_selection.gather(dim=1, index=indices)
            selected_count = valid_at.sum().to(dtype=weights.dtype)
        totals[f"{prefix}_selected_count"] = selected_count
        topk_covered = covered.gather(dim=1, index=indices)
        topk_invalid_core = invalid_core.gather(dim=1, index=indices)
        if valid_at is not None:
            topk_covered = topk_covered & valid_at
            topk_invalid_core = topk_invalid_core & valid_at
        totals[f"{prefix}_covered_count"] = topk_covered.sum()
        totals[f"{prefix}_covered_trial_count"] = topk_covered.any(dim=1).sum()
        totals[f"{prefix}_invalid_bce_sum"] = topk_covered_bce.masked_select(
            topk_covered
        ).sum()
        totals[f"{prefix}_invalid_core_count"] = topk_invalid_core.sum()
        totals[f"{prefix}_invalid_core_trial_count"] = topk_invalid_core.any(dim=1).sum()
        totals[f"{prefix}_invalid_core_bce_sum"] = topk_invalid_core_bce.masked_select(
            topk_invalid_core
        ).sum()

    add_topk_totals(
        "configured_topk",
        configured_indices,
        configured_covered_bce,
        configured_invalid_core_bce,
    )
    for label, k in probe_counts.items():
        add_topk_totals(
            f"probe_{label}",
            ranked_indices[:, :k],
            ranked_covered_bce[:, :k],
            ranked_invalid_core_bce[:, :k],
        )
    return totals


def finalize_target_augmentation_diagnostics(
    totals: Mapping[str, torch.Tensor],
    *,
    probe_ratios: Sequence[float] = TARGET_AUGMENTATION_PROBE_RATIOS,
) -> dict[str, float]:
    """把设备端诊断总量转成可记录的 epoch 比例和均值。

    分母为 0 时用 1 防止 NaN，因此没有覆盖位置的相关指标为 0；这与日志和
    CSV 的稳定数值约定一致，不代表模型在该位置取得了真实性能。
    """
    if not totals:
        return {}
    keys = list(totals)
    values = torch.stack([totals[key].detach().to(dtype=torch.float64) for key in keys]).cpu().tolist()
    value = dict(zip(keys, (float(item) for item in values)))

    def ratio(numerator: str, denominator: str) -> float:
        """以安全分母计算 totals 中两个累计量的比值。"""
        return value.get(numerator, 0.0) / max(1.0, value.get(denominator, 0.0))

    metrics = {
        "augmented_trial_fraction": ratio("augmented_trial_count", "sample_count"),
        "all_covered_invalid_bce": ratio("all_covered_invalid_bce_sum", "covered_count"),
        "topk_covered_count_per_sample": ratio("configured_topk_covered_count", "sample_count"),
        "topk_covered_position_fraction": ratio(
            "configured_topk_covered_count", "configured_topk_selected_count"
        ),
        "topk_covered_recall": ratio("configured_topk_covered_count", "covered_count"),
        "topk_covered_trial_fraction": ratio("configured_topk_covered_trial_count", "sample_count"),
        "topk_covered_probe_bce": ratio(
            "configured_topk_invalid_bce_sum", "configured_topk_covered_count"
        ),
        "effective_attention_mass": ratio("effective_mass_sum", "sample_count"),
        "covered_effective_attention_mean": ratio("covered_effective_weight_sum", "covered_count"),
        "uncovered_effective_attention_mean": ratio(
            "uncovered_effective_weight_sum", "uncovered_count"
        ),
        "covered_effective_attention_mass": ratio("covered_effective_mass_sum", "sample_count"),
        "uncovered_effective_attention_mass": ratio("uncovered_effective_mass_sum", "sample_count"),
        "covered_effective_attention_mass_fraction": ratio(
            "covered_effective_mass_fraction_sum",
            "replacement_trial_count" if "replacement_trial_count" in value else "augmented_trial_count",
        ),
        "covered_attention_gate_mean": ratio("covered_gate_sum", "covered_count"),
        "uncovered_attention_gate_mean": ratio("uncovered_gate_sum", "uncovered_count"),
    }
    metrics.update(
        {
            "invalid_core_fraction": ratio("invalid_core_count", "selected_time_count"),
            "invalid_core_trial_fraction": ratio("invalid_core_trial_count", "sample_count"),
            "all_invalid_core_bce": ratio("all_invalid_core_bce_sum", "invalid_core_count"),
            "topq_invalid_core_count_per_sample": ratio(
                "configured_topk_invalid_core_count", "sample_count"
            ),
            "topq_invalid_core_position_fraction": ratio(
                "configured_topk_invalid_core_count", "configured_topk_selected_count"
            ),
            "topq_invalid_core_recall": ratio(
                "configured_topk_invalid_core_count", "invalid_core_count"
            ),
            "topq_invalid_core_trial_fraction": ratio(
                "configured_topk_invalid_core_trial_count", "sample_count"
            ),
            "topq_invalid_core_probe_bce": ratio(
                "configured_topk_invalid_core_bce_sum", "configured_topk_invalid_core_count"
            ),
            "invalid_core_effective_attention_mean": ratio(
                "invalid_core_effective_weight_sum", "invalid_core_count"
            ),
            "invalid_core_effective_attention_mass": ratio(
                "invalid_core_effective_mass_sum", "sample_count"
            ),
            "invalid_core_effective_attention_mass_fraction": ratio(
                "invalid_core_effective_mass_fraction_sum", "invalid_core_trial_count"
            ),
            "invalid_core_attention_gate_mean": ratio(
                "invalid_core_gate_sum", "invalid_core_count"
            ),
        }
    )
    for probe_ratio in probe_ratios:
        label = f"q{int(round(float(probe_ratio) * 100)):02d}"
        prefix = f"probe_{label}"
        metrics[f"{prefix}_covered_count_per_sample"] = ratio(
            f"{prefix}_covered_count", "sample_count"
        )
        metrics[f"{prefix}_covered_position_fraction"] = ratio(
            f"{prefix}_covered_count", f"{prefix}_selected_count"
        )
        metrics[f"{prefix}_covered_recall"] = ratio(f"{prefix}_covered_count", "covered_count")
        metrics[f"{prefix}_covered_trial_fraction"] = ratio(
            f"{prefix}_covered_trial_count", "sample_count"
        )
        metrics[f"{prefix}_invalid_bce"] = ratio(
            f"{prefix}_invalid_bce_sum", f"{prefix}_covered_count"
        )
        metrics[f"{prefix}_invalid_core_count_per_sample"] = ratio(
            f"{prefix}_invalid_core_count", "sample_count"
        )
        metrics[f"{prefix}_invalid_core_position_fraction"] = ratio(
            f"{prefix}_invalid_core_count", f"{prefix}_selected_count"
        )
        metrics[f"{prefix}_invalid_core_recall"] = ratio(
            f"{prefix}_invalid_core_count", "invalid_core_count"
        )
        metrics[f"{prefix}_invalid_core_trial_fraction"] = ratio(
            f"{prefix}_invalid_core_trial_count", "sample_count"
        )
        metrics[f"{prefix}_invalid_core_bce"] = ratio(
            f"{prefix}_invalid_core_bce_sum", f"{prefix}_invalid_core_count"
        )
    return metrics
