"""Masked denoising reconstruction loss for time-level EEG representations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...evaluation.metrics import weighted_loss_mean


def masked_eeg_reconstruction_loss(
    reconstructed_eeg: torch.Tensor,
    clean_eeg: torch.Tensor,
    covered_time_mask: torch.Tensor | None,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average Smooth L1 over covered positions per trial, then over trials.

    Both EEG tensors use ``[B,C,T]``. The mask uses ``[B,T]`` and marks time
    positions replaced by augmentation. Trials without a covered position are
    excluded; an entirely empty mask returns a graph-connected zero.
    """
    if reconstructed_eeg.ndim != 3:
        raise ValueError(
            "reconstructed_eeg must have shape [batch, channel, time]"
        )
    if clean_eeg.shape != reconstructed_eeg.shape:
        raise ValueError(
            "clean_eeg must have the same shape as reconstructed_eeg: "
            f"{clean_eeg.shape} != {reconstructed_eeg.shape}"
        )
    if covered_time_mask is None:
        return reconstructed_eeg.sum() * 0.0
    if covered_time_mask.shape != (
        reconstructed_eeg.shape[0],
        reconstructed_eeg.shape[2],
    ):
        raise ValueError(
            "covered_time_mask must have shape [batch, time] aligned with EEG"
        )

    mask = covered_time_mask.to(
        device=reconstructed_eeg.device,
        dtype=torch.bool,
    )
    valid_trials = mask.any(dim=1)
    if not bool(valid_trials.any().item()):
        return reconstructed_eeg.sum() * 0.0

    # Channel-average first, then normalize by the covered count in each trial.
    clean_target = clean_eeg.detach().to(
        device=reconstructed_eeg.device,
        dtype=torch.float32,
    )
    masked_prediction = torch.where(
        mask[:, None, :],
        reconstructed_eeg.float(),
        clean_target,
    )
    position_loss = F.smooth_l1_loss(
        masked_prediction,
        clean_target,
        reduction="none",
    ).mean(dim=1)
    per_trial = (
        position_loss * mask.to(dtype=position_loss.dtype)
    ).sum(dim=1) / mask.sum(dim=1).clamp_min(1).to(dtype=position_loss.dtype)
    valid_weight = (
        sample_weight.to(device=reconstructed_eeg.device).reshape(-1)[valid_trials]
        if sample_weight is not None
        else None
    )
    return weighted_loss_mean(per_trial[valid_trials], valid_weight)
