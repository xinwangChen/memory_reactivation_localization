"""training.state: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from typing import Iterator, Mapping
import torch
from torch.utils.data import DataLoader


class CyclingDataLoader:
    """把有限 DataLoader 包装成按 iteration 连续取 batch 的循环数据流。"""

    def __init__(self, loader: DataLoader) -> None:
        """保存 loader，并把数据 epoch 从 0 开始计数。"""
        self.loader = loader
        self.iterator: Iterator[Mapping[str, torch.Tensor]] | None = None
        self.epoch = 0
        self.batches_in_epoch = 0

    def _set_epoch(self) -> None:
        """把当前 epoch 同步给 sampler 和 dataset，以刷新确定性随机采样。"""
        batch_sampler = getattr(self.loader, "batch_sampler", None)
        if hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(self.epoch)
        dataset = getattr(self.loader, "dataset", None)
        if hasattr(dataset, "set_epoch"):
            dataset.set_epoch(self.epoch)

    def take(self, n_batches: int):
        """连续产出指定数量的 batch；耗尽底层 loader 时自动进入下一 epoch。"""
        for _ in range(n_batches):
            if self.iterator is None:
                self._set_epoch()
                self.iterator = iter(self.loader)
            try:
                yield next(self.iterator)
            except StopIteration:
                self.epoch += 1
                self.batches_in_epoch = 0
                self._set_epoch()
                self.iterator = iter(self.loader)
                yield next(self.iterator)
            self.batches_in_epoch += 1

    def skip(self, n_batches: int) -> None:
        """丢弃接下来的 n_batches 个 batch（可跨 epoch），用于恢复 epoch 内游标。

        只消耗数据不产生前向；与中断前已消费的 batch 数量一致时，后续 take 的
        数据流与未中断训练完全相同。
        """
        remaining = int(n_batches)
        while remaining > 0:
            if self.iterator is None:
                self._set_epoch()
                self.iterator = iter(self.loader)
            try:
                next(self.iterator)
            except StopIteration:
                self.epoch += 1
                self.batches_in_epoch = 0
                self._set_epoch()
                self.iterator = iter(self.loader)
                continue
            self.batches_in_epoch += 1
            remaining -= 1
