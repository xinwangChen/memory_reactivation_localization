"""data.transforms: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from numbers import Integral
import torch


def gaussian_like_eeg(
    eeg: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    noise_mode: str = "sample_channel",
    min_noise_std: float = 1e-4,
) -> torch.Tensor:
    """生成与真实 EEG 统计尺度一致的完整高斯基线输入。

    与局部增强不同，这里替换整个 [B,C,T]，用于导出阶段检验模型在纯噪声上的
    输出。sample_channel、batch_channel 和 standard_normal 语义与增强器一致。
    """
    if noise_mode == "standard_normal":
        mean = torch.zeros(eeg.shape[0], eeg.shape[1], 1, device=eeg.device, dtype=eeg.dtype)
        std = torch.ones_like(mean)
    elif noise_mode == "batch_channel":
        mean = eeg.mean(dim=(0, 2), keepdim=True)
        std = eeg.std(dim=(0, 2), keepdim=True, unbiased=False)
    elif noise_mode == "sample_channel":
        mean = eeg.mean(dim=2, keepdim=True)
        std = eeg.std(dim=2, keepdim=True, unbiased=False)
    else:
        raise ValueError("noise_mode must be sample_channel, batch_channel, or standard_normal")
    std = std.clamp_min(float(min_noise_std))
    noise = torch.randn(eeg.shape, device=eeg.device, dtype=eeg.dtype, generator=generator)
    return mean + noise * std


EVALUATION_INPUT_KINDS = ("real", "gaussian", "time_shuffle", "time_roll")


def _stable_evaluation_seed(base_seed: int, *identity: int) -> int:
    """Mix a base seed with stable trial identifiers without Python's salted hash."""
    modulus = (1 << 63) - 1
    value = int(base_seed) % modulus
    for component in identity:
        value = (
            value * 6364136223846793005
            + int(component)
            + 1442695040888963407
        ) % modulus
    return value


def time_shuffle_eeg(
    eeg: torch.Tensor,
    *,
    seed: int,
    record_id: torch.Tensor,
    trial0: torch.Tensor,
    lag: torch.Tensor,
) -> torch.Tensor:
    """Deterministically permute time within each trial, shared by all channels."""
    if eeg.ndim != 3:
        raise ValueError("eeg must have shape [batch, channel, time]")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("time-shuffle seed must be a non-negative integer")
    batch_size, _, time_count = eeg.shape
    identities: list[list[int]] = []
    for name, values in (("record_id", record_id), ("trial0", trial0), ("lag", lag)):
        if (not isinstance(values, torch.Tensor) or values.shape != (batch_size,)
                or values.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
                or bool((values < 0).any())):
            raise ValueError(f"{name} must contain one value per EEG trial")
        identities.append(
            [int(value) for value in values.detach().cpu().reshape(-1).tolist()]
        )
    if time_count < 2:
        return eeg.clone()

    shuffled: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        trial_seed = _stable_evaluation_seed(
            int(seed),
            identities[0][batch_index],
            identities[1][batch_index],
            identities[2][batch_index],
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(trial_seed)
        permutation = torch.randperm(
            time_count,
            generator=generator,
            device="cpu",
        ).to(eeg.device)
        shuffled.append(eeg[batch_index].index_select(-1, permutation))
    return torch.stack(shuffled, dim=0)


def time_roll_eeg(eeg: torch.Tensor, *, shift_samples: int) -> torch.Tensor:
    """Circularly shift every trial and channel by one fixed sample offset."""
    if eeg.ndim != 3:
        raise ValueError("eeg must have shape [batch, channel, time]")
    if isinstance(shift_samples, bool) or not isinstance(shift_samples, Integral):
        raise ValueError("time-roll shift_samples must be an integer")
    time_count = int(eeg.shape[-1])
    if time_count < 1:
        raise ValueError("eeg time axis must be non-empty")
    shift = int(shift_samples)
    if shift % time_count == 0:
        raise ValueError("time-roll shift must not be zero modulo the time-axis length")
    return torch.roll(eeg, shifts=shift, dims=-1)
