"""同步训练涉及的伪随机数生成器。"""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """为 Python、NumPy、PyTorch CPU 及所有可见 GPU 设置同一随机种子。

    该函数建立各随机源的初始状态，但不强制 PyTorch 使用确定性算子；若底层
    CUDA 算子本身非确定，完全逐位复现仍需在训练入口额外配置确定性策略。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
