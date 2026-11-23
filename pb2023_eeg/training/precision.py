"""training.precision: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from contextlib import nullcontext
import torch


def autocast_context(device: torch.device, enabled: bool):
    """按设备和 AMP 开关返回 autocast 上下文；不支持的设备退化为空上下文。"""
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    if device.type == "cpu":
        return torch.amp.autocast("cpu")
    return nullcontext()
