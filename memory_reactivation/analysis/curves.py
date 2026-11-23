"""analysis.curves: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from typing import Sequence
import torch


def attention_shape_diagnostics(
    normalized_effective: torch.Tensor,
    time_s: Sequence[float],
    valid_mask: torch.Tensor | None = None,
) -> list[dict[str, float | int]]:
    """从归一化 effective attention 计算每个 trial 的时间曲线形状诊断。

    输出包括熵、相对均匀分布的 KL、变异系数、峰值位置、边界距离及主峰半高
    全宽 FWHM。输入 [B,T] 的每行应能归一化为概率质量。
    """
    if normalized_effective.ndim != 2:
        raise ValueError("normalized_effective must have shape [batch, time]")
    if normalized_effective.shape[1] != len(time_s) or not time_s:
        raise ValueError("time_s must be non-empty and align with normalized_effective")
    curves = normalized_effective.detach().float().cpu()
    times = torch.as_tensor(time_s, dtype=torch.float64)
    if valid_mask is None:
        valid_rows = torch.ones_like(curves, dtype=torch.bool)
    elif valid_mask.ndim == 1 and valid_mask.shape[0] == curves.shape[1]:
        valid_rows = valid_mask.detach().to(dtype=torch.bool, device="cpu")[
            None, :
        ].expand_as(curves)
    elif valid_mask.shape == curves.shape:
        valid_rows = valid_mask.detach().to(dtype=torch.bool, device="cpu")
    else:
        raise ValueError(
            "valid_mask must have shape [time] or [batch, time] aligned with attention"
        )
    if not bool(valid_rows.any(dim=1).all()):
        raise ValueError("valid_mask must retain at least one attention position per trial")
    if len(time_s) > 1:
        sample_interval_s = float((times[1:] - times[:-1]).abs().median().item())
    else:
        sample_interval_s = 0.0
    total_time_count = curves.shape[1]
    diagnostics: list[dict[str, float | int]] = []
    for curve, row_valid in zip(curves, valid_rows):
        valid_indices = torch.nonzero(row_valid, as_tuple=False).flatten()
        support_curve = curve.index_select(0, valid_indices)
        support_curve = support_curve / support_curve.sum().clamp_min(1e-12)
        support_count = int(support_curve.numel())
        log_support_count = (
            float(torch.log(torch.tensor(float(support_count))).item())
            if support_count > 1
            else 0.0
        )
        entropy = float(
            -(support_curve * support_curve.clamp_min(1e-12).log()).sum().item()
        )
        support_peak_index = int(support_curve.argmax().item())
        peak_index = int(valid_indices[support_peak_index].item())
        peak_value = float(support_curve[support_peak_index].item())
        half_maximum = peak_value * 0.5
        left_support_index = support_peak_index
        right_support_index = support_peak_index
        while (
            left_support_index > 0
            and float(support_curve[left_support_index - 1].item()) >= half_maximum
        ):
            left_support_index -= 1
        while (
            right_support_index + 1 < support_count
            and float(support_curve[right_support_index + 1].item()) >= half_maximum
        ):
            right_support_index += 1
        left_index = int(valid_indices[left_support_index].item())
        right_index = int(valid_indices[right_support_index].item())
        fwhm_s = (
            float(times[right_index].item() - times[left_index].item()) + sample_interval_s
            if support_count > 1
            else 0.0
        )
        distance_samples = min(peak_index, total_time_count - 1 - peak_index)
        support_distance_samples = min(
            support_peak_index,
            support_count - 1 - support_peak_index,
        )
        diagnostics.append(
            {
                "attention_entropy": entropy,
                "attention_entropy_normalized": (
                    entropy / log_support_count if log_support_count > 0.0 else 0.0
                ),
                "attention_kl_to_uniform": max(
                    0.0, log_support_count - entropy
                ),
                "attention_cv": float(
                    support_curve.std(unbiased=False).item()
                    / support_curve.mean().clamp_min(1e-12).item()
                ),
                "attention_pool_support_count": support_count,
                "attention_peak_index": peak_index,
                "attention_peak_time_s": float(times[peak_index].item()),
                "attention_peak_weight": peak_value,
                "attention_peak_at_boundary": int(
                    peak_index in {0, total_time_count - 1}
                ),
                "attention_peak_distance_to_boundary_samples": distance_samples,
                "attention_peak_distance_to_boundary_s": distance_samples * sample_interval_s,
                "attention_peak_at_pool_support_boundary": int(
                    support_peak_index in {0, support_count - 1}
                ),
                "attention_peak_distance_to_pool_support_samples": (
                    support_distance_samples
                ),
                "attention_peak_distance_to_pool_support_s": (
                    support_distance_samples * sample_interval_s
                ),
                "attention_fwhm_s": fwhm_s,
                "attention_fwhm_left_time_s": float(times[left_index].item()),
                "attention_fwhm_right_time_s": float(times[right_index].item()),
                "attention_fwhm_touches_boundary": int(
                    left_index == 0 or right_index == total_time_count - 1
                ),
                "attention_fwhm_touches_pool_support_boundary": int(
                    left_support_index == 0
                    or right_support_index == support_count - 1
                ),
            }
        )
    return diagnostics
