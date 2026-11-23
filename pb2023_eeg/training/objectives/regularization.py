"""Optional training-only penalties on final effective attention weights."""

from math import isfinite
from typing import Mapping

import torch


def normalize_attention_mass_penalty_config(
    train_config: Mapping[str, object],
) -> dict[str, float]:
    """Resolve the independent weight for the fixed, one-sided mass ceiling of 1."""
    key = "attention_mass_penalty_weight"
    raw = train_config.get(key, 0.0)
    try:
        if isinstance(raw, bool):
            raise ValueError
        weight = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"train.{key} must be a finite non-negative number") from exc
    if not isfinite(weight) or weight < 0.0:
        raise ValueError(f"train.{key} must be a finite non-negative number")
    return {"weight": weight}


def attention_mass_penalty(
    effective_attention_weights: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean per-trial relu(sum_valid |w| - 1)^2, without sample or Top-q weights.

    All trials participate, including roll and near/full-noise inputs. Current
    E4 weights are non-negative; signed exploratory modes use magnitude, matching
    effective_attention_mass diagnostics and preventing sign cancellation.
    Only final pooling support is masked. This neither rescales forward weights
    nor pushes masses below 1 upward. Reduction stays in FP32 for half precision.
    """
    weights = effective_attention_weights
    if weights.ndim != 2 or not weights.is_floating_point():
        raise ValueError("attention mass weights must be floating [batch, attention_time]")
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape not in (weights.shape, weights.shape[1:]):
            raise ValueError("attention mass valid_mask must be bool [time] or [batch, time]")
    values = weights.float() if weights.dtype in (torch.float16, torch.bfloat16) else weights
    if values.shape[0] == 0 or values.shape[1] == 0:
        return values[:, :0].sum()
    values = values.abs()
    if valid_mask is not None:
        values = values.masked_fill(~valid_mask.to(device=weights.device), 0.0)
    mass = values.sum(dim=1)
    return (mass - 1.0).clamp_min(0.0).square().mean()


def normalize_clean_attention_penalty_config(
    train_config: Mapping[str, object],
) -> dict[str, float]:
    """Resolve flat train.* options; weight=0 or probability=0 disables the term."""
    result = {}
    for name, default in (("weight", 0.0), ("probability", 0.1)):
        key = f"clean_attention_penalty_{name}"
        raw = train_config.get(key, default)
        try:
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"train.{key} must be a finite number") from exc
        if not isfinite(value) or value < 0.0 or (name == "probability" and value > 1.0):
            limit = "in [0, 1]" if name == "probability" else "non-negative"
            raise ValueError(f"train.{key} must be finite and {limit}")
        result[name] = value
    return result


def clean_attention_penalty(
    attention_strength_weights: torch.Tensor,
    attention_indices: torch.Tensor,
    contaminated_time_mask: torch.Tensor | None,
    *,
    probability: float = 0.1,
    valid_mask: torch.Tensor | None = None,
    roll_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (loss, candidates, selected), aligned with [batch, attention_time].

    The caller supplies final effective attention strength (|w| for signed modes).
    Candidates are the complement of the existing noise-penalty mask, intersected
    with valid pooling support and excluding every rolled trial. In all_positions
    mode this means input positions not directly replaced; rf_aware also excludes
    the existing contaminated halo. With no replacement mask, all valid positions
    are candidates. Top-q and trial sample weights do not enter this penalty.

    Independently sample each candidate; sum selected strengths per trial and then
    average over the whole batch, including zero-contribution trials, just as for
    absolute_effective_mass. Do not divide by selected count or probability.
    Reduction uses FP32 for half precision. Empty candidates and probability=0
    return a graph-connected zero without consuming RNG; probability=1 is exact.
    """
    probability = normalize_clean_attention_penalty_config(
        {"clean_attention_penalty_probability": probability}
    )["probability"]
    weights = attention_strength_weights
    if weights.ndim != 2 or not weights.is_floating_point():
        raise ValueError("clean attention strengths must be floating [batch, attention_time]")
    if attention_indices.ndim != 1 or attention_indices.dtype not in (torch.int32, torch.int64):
        raise ValueError("clean attention indices must be a 1D int32/int64 tensor")
    if attention_indices.numel() != weights.shape[1]:
        raise ValueError("clean attention indices must align with attention strengths")
    indices = attention_indices.to(device=weights.device)
    if indices.numel() and bool((indices < 0).any().item()):
        raise ValueError("clean attention indices must be non-negative")
    candidates = torch.ones_like(weights, dtype=torch.bool)
    if contaminated_time_mask is not None:
        if (contaminated_time_mask.ndim != 2
                or contaminated_time_mask.shape[0] != weights.shape[0]
                or contaminated_time_mask.dtype != torch.bool):
            raise ValueError("clean attention contaminated_time_mask must be bool [batch, time]")
        if indices.numel() and bool((indices >= contaminated_time_mask.shape[1]).any().item()):
            raise ValueError("clean attention indices exceed the contaminated mask time axis")
        contaminated = contaminated_time_mask.to(device=weights.device).index_select(1, indices)
        candidates &= ~contaminated
    if valid_mask is not None:
        if valid_mask.dtype != torch.bool or valid_mask.shape not in (weights.shape, weights.shape[1:]):
            raise ValueError("clean attention valid_mask must be bool [time] or [batch, time]")
        candidates &= valid_mask.to(device=weights.device)
    if roll_mask is not None:
        if roll_mask.dtype != torch.bool or roll_mask.shape != weights.shape[:1]:
            raise ValueError("clean attention roll_mask must be bool [batch]")
        candidates &= ~roll_mask.to(device=weights.device)[:, None]
    selected = torch.zeros_like(candidates)
    if probability == 0.0 or not bool(candidates.any().item()):
        # Summing excluded FP16 weights can overflow before multiplication by 0.
        # An empty slice gives an exact zero with zero gradients to every weight.
        return weights[:, :0].sum(), candidates, selected
    if probability == 1.0:
        selected = candidates.clone()
    else:
        selected = candidates & (
            torch.rand(weights.shape, device=weights.device, dtype=torch.float32) < probability
        )
    values = weights.float() if weights.dtype in (torch.float16, torch.bfloat16) else weights
    loss = values.masked_fill(~selected, 0.0).sum(dim=1).mean()
    return loss.to(dtype=weights.dtype), candidates, selected
