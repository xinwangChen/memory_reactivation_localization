"""Shared physical-time support for training, prediction and reexport."""
from collections.abc import Mapping
import math

import torch


def resolve_attention_window(train_config: Mapping[str, object], times: torch.Tensor) -> tuple[float, float]:
    """解析注意力池化和 time-level 损失共用的闭区间时间窗。

    新配置 ``attention_time_*`` 优先；兼容旧 ``train_time_*``。两套配置同时存在时必须
    完全一致，任一套的起止值也必须成对提供；均缺省则使用数据时间轴首尾。
    """
    has_attention_window = (
        train_config.get("attention_time_start_s") is not None
        or train_config.get("attention_time_end_s") is not None
    )
    has_train_window = (
        train_config.get("train_time_start_s") is not None
        or train_config.get("train_time_end_s") is not None
    )
    # 兼容旧键但禁止两个窗口悄然产生不同的监督/池化区域。
    if has_attention_window:
        if train_config.get("attention_time_start_s") is None or train_config.get("attention_time_end_s") is None:
            raise ValueError("attention_time_start_s and attention_time_end_s must be configured together")
        start_s = float(train_config["attention_time_start_s"])
        end_s = float(train_config["attention_time_end_s"])
        if has_train_window:
            if train_config.get("train_time_start_s") is None or train_config.get("train_time_end_s") is None:
                raise ValueError("train_time_start_s and train_time_end_s must be configured together")
            train_start = float(train_config["train_time_start_s"])
            train_end = float(train_config["train_time_end_s"])
            if abs(start_s - train_start) > 1e-6 or abs(end_s - train_end) > 1e-6:
                raise ValueError(
                    "attention_time_* and train_time_* disagree; keep only one window or make them identical"
                )
    elif has_train_window:
        if train_config.get("train_time_start_s") is None or train_config.get("train_time_end_s") is None:
            raise ValueError("train_time_start_s and train_time_end_s must be configured together")
        start_s = float(train_config["train_time_start_s"])
        end_s = float(train_config["train_time_end_s"])
    else:
        start_s = float(times[0])
        end_s = float(times[-1])
    if not math.isfinite(start_s) or not math.isfinite(end_s):
        raise ValueError("Attention window bounds must be finite")
    if start_s > end_s:
        raise ValueError("Configured attention/time-loss window start must not exceed end")
    return start_s, end_s

