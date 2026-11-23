"""分类任务的加权损失、平衡准确率累计器与秩相关统计。"""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Sequence

import numpy as np
import torch


def weighted_loss_mean(
    loss_values: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """把未约简损失先按 trial 求均值，再按可选样本权重求加权均值。

    loss_values 的第 0 维必须是 batch，其余维度（时间、类别等）先在每个
    trial 内平均。sample_weight 不要求预先归一化，函数会除以权重总和。
    """
    if loss_values.ndim < 1 or loss_values.shape[0] < 1:
        raise ValueError("loss_values must have a non-empty sample dimension")
    per_sample = loss_values.reshape(loss_values.shape[0], -1).mean(dim=1)  # 先压掉时间/类维
    if sample_weight is None:
        return per_sample.mean()
    weight = sample_weight.to(device=loss_values.device, dtype=loss_values.dtype).reshape(-1)
    if weight.numel() != per_sample.numel():
        raise ValueError(
            "sample_weight must contain one value per sample: "
            f"got {weight.numel()} for {per_sample.numel()} losses"
        )
    if not bool(torch.isfinite(weight).all().item()):
        raise ValueError("sample_weight must be finite")
    if bool((weight < 0).any().item()):
        raise ValueError("sample_weight must be non-negative")
    weight_sum = weight.sum()
    if float(weight_sum.detach().item()) <= 0.0:
        raise ValueError("sample_weight must have a positive sum")
    return (per_sample * weight).sum() / weight_sum  # 权重无需事先归一


def balanced_accuracy_from_confusion(confusion: np.ndarray) -> float:
    """由“真实类别 × 预测类别”混淆矩阵计算宏平均召回率。

    预测列可以多于真实类别行，例如 time-level 开启 no-class 后矩阵为
    K x (K+1)。额外列会进入真实类的分母但不可能进入对角线，因此
    预测为 no-class 会按错误处理，符合最终性能曲线的统计约定。
    """
    values = np.asarray(confusion, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < values.shape[0]:
        raise ValueError(
            "confusion must have one row per target class and at least as many "
            "prediction columns"
        )
    support = values.sum(axis=1)  # 每真实类样本量（含预测到 no-class 的计数）
    valid = support > 0
    if not bool(valid.any()):
        return float("nan")
    return float(np.mean(np.diag(values)[valid] / support[valid]))  # 宏平均召回；no-class 永不进对角


def classification_cell_metadata(
    batch: Mapping[str, torch.Tensor],
    task: str,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """提取计算条件平衡准确率所需的被试编号与干扰条件 cell 编号。

    pitch lag=0 的 target 就是当前刺激，使用 cell 0；pitch 正 lag 的 target
    来自历史 trial，必须按当前刺激分 cell，排除刺激转移先验。其他任务优先
    使用数据集已经编码好的 choice_cell_id，旧数据缺少该字段时回退到当前
    刺激类别。任一必要字段缺失时返回 (None, None)，仅累计全局混淆矩阵。
    """
    subject_id = batch.get("subject_id", batch.get("record_id"))  # 优先跨实验 participant
    if subject_id is None:
        return None, None
    if task == "pitch":
        lag = batch.get("lag")
        if lag is None:
            return subject_id, torch.zeros_like(subject_id)  # 兼容未导出 lag 的旧 batch
        current_stimulus = batch.get("current_stimulus")
        if current_stimulus is None:
            return None, None
        lag_values = lag.to(device=subject_id.device, dtype=subject_id.dtype)
        current_values = current_stimulus.to(
            device=subject_id.device,
            dtype=subject_id.dtype,
        )
        cell_id = torch.where(
            lag_values > 0,
            current_values,
            torch.zeros_like(subject_id),
        )
        return subject_id, cell_id
    choice_cell_id = batch.get("choice_cell_id")
    if choice_cell_id is not None:
        return subject_id, choice_cell_id  # 通常含 current_pitch ± 历史字段
    current_stimulus = batch.get("current_stimulus")
    if current_stimulus is None:
        return None, None
    return subject_id, current_stimulus


class BalancedAccuracyAccumulator:
    """流式累计全局及 subject/cell 条件混淆矩阵。

    batch 内计数一直保留在输入张量所在设备，直到 :meth: metrics 才一次性
    搬到 CPU，避免训练循环频繁同步。class_count 是真实任务类别数，
    prediction_class_count 可以更大，用来容纳 time-level 的 no-class 列。
    """

    def __init__(
        self,
        class_count: int,
        min_cell_class_count: int = 1,
        prediction_class_count: int | None = None,
    ) -> None:
        """初始化固定形状的混淆矩阵及延迟合并缓冲区。"""
        self.class_count = int(class_count)  # 真实任务 K
        if self.class_count < 2:
            raise ValueError("class_count must be at least 2")
        self.prediction_class_count = (
            self.class_count
            if prediction_class_count is None
            else int(prediction_class_count)  # time 可 K+1（含 no-class 列）
        )
        if self.prediction_class_count < self.class_count:
            raise ValueError("prediction_class_count must be at least class_count")
        self.min_cell_class_count = int(min_cell_class_count)  # 完整 cell 每类最少计数
        if self.min_cell_class_count < 1:
            raise ValueError("min_cell_class_count must be positive")
        confusion_shape = (self.class_count, self.prediction_class_count)
        self.confusion = np.zeros(confusion_shape, dtype=np.int64)
        self.cell_confusions: dict[tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(confusion_shape, dtype=np.int64)
        )
        self.cell_class_support: dict[tuple[int, int], np.ndarray] = defaultdict(
            lambda: np.zeros(self.class_count, dtype=np.int64)
        )
        self._device: torch.device | None = None
        self._device_confusion: torch.Tensor | None = None
        self._cell_rows: list[torch.Tensor] = []
        self._cell_counts: list[torch.Tensor] = []
        self._cell_support_counts: list[torch.Tensor] = []
        self._invalid_class_count: torch.Tensor | None = None
        self._invalid_cell_count: torch.Tensor | None = None

    def _set_or_check_device(self, device: torch.device) -> None:
        """首次更新时创建设备端计数器，之后禁止跨设备混合累计。"""
        if self._device is None:
            self._device = device
            self._device_confusion = torch.zeros(
                self.class_count * self.prediction_class_count,
                dtype=torch.long,
                device=device,
            )
            self._invalid_class_count = torch.zeros((), dtype=torch.long, device=device)
            self._invalid_cell_count = torch.zeros((), dtype=torch.long, device=device)
        elif self._device != device:
            raise ValueError("all accumulator updates must use the same device")

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        subject_id: torch.Tensor | None = None,
        cell_id: torch.Tensor | None = None,
        *,
        cell_support_mask: torch.Tensor | None = None,
    ) -> None:
        """用一批预测更新全局及可选 subject/cell 条件计数。

        所有输入都会展平成一维，因此既可统计 trial-level [B]，也可统计
        选中时间点展开后的 [B*q]。GPU 上不立即读取非法值标志，以免每批
        触发同步；非法计数在最终汇总时统一检查。

        cell_support_mask 只控制完整 cell 的支持数，不改变准确率的混淆计数。
        时间指标可令每条 trial 仅一个有效位置为 True，以 trial 数判定支持度。
        缺省时每条预测都计入支持数，保持 trial-level 调用的原有行为。
        """
        prediction_values = prediction.detach().reshape(-1).to(dtype=torch.long)  # trial 或 [B*q]
        target_values = target.detach().reshape(-1).to(
            device=prediction_values.device,
            dtype=torch.long,
        )
        if prediction_values.numel() != target_values.numel():
            raise ValueError("prediction and target must have the same number of elements")
        support_values = None
        if cell_support_mask is not None:
            support_values = cell_support_mask.detach().reshape(-1).to(
                device=prediction_values.device, dtype=torch.bool
            )
            if support_values.numel() != target_values.numel():
                raise ValueError("cell_support_mask must align with prediction and target")
        if prediction_values.numel() == 0:
            return
        invalid_class = (
            (prediction_values < 0)
            | (target_values < 0)
            | (prediction_values >= self.prediction_class_count)
            | (target_values >= self.class_count)
        )
        if prediction_values.device.type == "cpu" and bool(invalid_class.any()):
            raise ValueError(
                "prediction and target indices must be within their configured class counts"
            )
        # GPU 上先 clamp 再记 invalid 计数，metrics() 时再统一 raise
        valid_prediction = prediction_values.clamp(0, self.prediction_class_count - 1)
        valid_target = target_values.clamp(0, self.class_count - 1)

        subject_values: torch.Tensor | None = None
        cell_values: torch.Tensor | None = None
        invalid_cell: torch.Tensor | None = None
        if subject_id is None or cell_id is None:
            if subject_id is not None or cell_id is not None:
                raise ValueError("subject_id and cell_id must be provided together")
        else:
            subject_values = subject_id.detach().reshape(-1).to(
                device=prediction_values.device,
                dtype=torch.long,
            )
            cell_values = cell_id.detach().reshape(-1).to(
                device=prediction_values.device,
                dtype=torch.long,
            )
            if (
                subject_values.numel() != target_values.numel()
                or cell_values.numel() != target_values.numel()
            ):
                raise ValueError("subject_id and cell_id must align with prediction and target")
            invalid_cell = (subject_values < 0) | (cell_values < 0)
            if prediction_values.device.type == "cpu" and bool(invalid_cell.any()):
                raise ValueError("subject_id and cell_id must be non-negative")

        self._set_or_check_device(prediction_values.device)
        assert self._invalid_class_count is not None
        assert self._device_confusion is not None
        self._invalid_class_count.add_(invalid_class.sum())
        # 把二维 (target, prediction) 坐标编码成一维索引，用一次 bincount
        # 在 GPU 上完成整批混淆矩阵计数。
        encoded = valid_target * self.prediction_class_count + valid_prediction
        self._device_confusion.add_(
            torch.bincount(
                encoded,
                minlength=self.class_count * self.prediction_class_count,
            )
        )
        if subject_values is None or cell_values is None or invalid_cell is None:
            return
        assert self._invalid_cell_count is not None
        self._invalid_cell_count.add_(invalid_cell.sum())
        rows = torch.stack(
            (
                subject_values.clamp_min(0),
                cell_values.clamp_min(0),
                valid_target,
                valid_prediction,
            ),
            dim=1,
        )
        if support_values is None:
            unique_rows, counts = torch.unique(rows, dim=0, return_counts=True)
            support_counts = counts
        else:
            unique_rows, inverse, counts = torch.unique(
                rows, dim=0, return_inverse=True, return_counts=True
            )
            support_counts = torch.zeros_like(counts)
            support_counts.scatter_add_(0, inverse, support_values.to(dtype=torch.long))
        self._cell_rows.append(unique_rows)
        self._cell_counts.append(counts)
        self._cell_support_counts.append(support_counts)

    def _finalize_device_totals(self) -> None:
        """把设备端批次统计合并为 CPU/Numpy 混淆矩阵。"""
        if self._device_confusion is None:
            return
        assert self._invalid_class_count is not None
        assert self._invalid_cell_count is not None
        invalid_counts = torch.stack(
            (self._invalid_class_count, self._invalid_cell_count)
        ).cpu()
        if int(invalid_counts[0]) > 0:
            raise ValueError(
                "prediction and target indices must be within their configured class counts"
            )
        if int(invalid_counts[1]) > 0:
            raise ValueError("subject_id and cell_id must be non-negative")
        self.confusion = self._device_confusion.reshape(
            self.class_count, self.prediction_class_count
        ).cpu().numpy()
        self.cell_confusions.clear()
        self.cell_class_support.clear()
        if not self._cell_rows:
            return
        # 不同 batch 可能出现同一 subject/cell/target/prediction 四元组，
        # 先在设备端二次去重并合并计数，再构建少量 Numpy 子矩阵。
        rows = torch.cat(self._cell_rows, dim=0)
        counts = torch.cat(self._cell_counts, dim=0)
        unique_rows, inverse = torch.unique(rows, dim=0, return_inverse=True)
        combined_counts = torch.zeros(
            unique_rows.shape[0], dtype=torch.long, device=rows.device
        )
        combined_counts.scatter_add_(0, inverse, counts)
        combined_support = torch.zeros_like(combined_counts)
        combined_support.scatter_add_(0, inverse, torch.cat(self._cell_support_counts))
        rows_np = unique_rows.cpu().numpy()
        counts_np = combined_counts.cpu().numpy()
        support_np = combined_support.cpu().numpy()
        for row, count, support in zip(rows_np, counts_np, support_np):
            subject, cell, target_index, prediction_index = (int(value) for value in row)
            self.cell_confusions[(subject, cell)][target_index, prediction_index] += int(
                count
            )
            self.cell_class_support[(subject, cell)][target_index] += int(support)

    def metrics(self) -> dict[str, float]:
        """计算全局、cell 平均及 subject 等权平均的平衡准确率。

        只有每个真实类别支持数均达到 min_cell_class_count 的完整 cell 才会
        纳入条件指标，避免缺类 cell 产生虚高或不可比较的宏平均召回率。
        """
        self._finalize_device_totals()
        complete_cell_scores: list[float] = []
        scores_by_subject: dict[int, list[float]] = defaultdict(list)
        for (subject, _cell), confusion in self.cell_confusions.items():
            if not bool(
                (self.cell_class_support[(subject, _cell)] >= self.min_cell_class_count).all()
            ):
                continue  # 某真实类样本不足 → 整 cell 不进条件指标
            score = balanced_accuracy_from_confusion(confusion)
            complete_cell_scores.append(score)
            scores_by_subject[subject].append(score)
        subject_scores = [
            float(np.mean(scores))
            for scores in scores_by_subject.values()
            if scores
        ]
        return {
            "balanced_accuracy": balanced_accuracy_from_confusion(self.confusion),
            "cell_balanced_accuracy": (
                float(np.mean(complete_cell_scores))
                if complete_cell_scores
                else float("nan")
            ),
            "cell_count": float(len(complete_cell_scores)),
            "subject_cell_balanced_accuracy": (
                float(np.mean(subject_scores))
                if subject_scores
                else float("nan")
            ),
            "subject_cell_count": float(len(subject_scores)),
        }


def build_time_cell_export_rows(
    time_cell_confusion: Mapping[tuple[int, int, int, int, int], int],
    *,
    class_count: int,
    prediction_class_count: int | None = None,
    time_s: Sequence[float],
    row_metadata: Mapping[str, object],
    cell_metadata: Mapping[tuple[int, int], Mapping[str, object]] | None = None,
    cell_metadata_fields: Sequence[str] = (),
    min_cell_class_count: int = 1,
    time_metadata: Callable[[int, float], Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """由逐 cell/time 混淆计数生成统一的明细和时间曲线汇总。

    输入键依次为 ``subject_id, cell_id, time_index, target, prediction``。
    prediction 维可以是 K 或 K+1；额外的 no-class 列进入真实类支持数但不会进入
    对角线，因此自然按错误计入 balanced accuracy。只有每个真实类支持数均达到
    ``min_cell_class_count`` 的 cell 才进入 complete-cell 与 subject-equal 汇总。
    """
    class_count = int(class_count)
    prediction_class_count = (
        class_count
        if prediction_class_count is None
        else int(prediction_class_count)
    )
    min_cell_class_count = int(min_cell_class_count)
    if class_count < 2:
        raise ValueError("class_count must be at least 2")
    if prediction_class_count < class_count:
        raise ValueError("prediction_class_count must be at least class_count")
    if min_cell_class_count < 1:
        raise ValueError("min_cell_class_count must be positive")
    all_times = [float(value) for value in time_s]
    nuisance_by_cell = cell_metadata or {}
    extra_time_metadata = time_metadata or (lambda _index, _time: {})

    cell_keys = sorted(
        {
            (subject_id, cell_id, time_index)
            for (subject_id, cell_id, time_index, _target, _prediction), count
            in time_cell_confusion.items()
            if int(count) != 0
        }
    )
    cell_metric_rows: list[dict[str, object]] = []
    complete_cell_scores: dict[int, list[float]] = defaultdict(list)
    complete_subject_cell_scores: dict[tuple[int, int], list[float]] = defaultdict(list)
    total_cell_counts: dict[int, int] = defaultdict(int)
    for subject_id, cell_id, time_index in cell_keys:
        if time_index < 0 or time_index >= len(all_times):
            raise ValueError("time cell confusion contains an out-of-range time index")
        confusion = np.asarray(
            [
                [
                    int(
                        time_cell_confusion.get(
                            (
                                subject_id,
                                cell_id,
                                time_index,
                                target_index,
                                prediction_index,
                            ),
                            0,
                        )
                    )
                    for prediction_index in range(prediction_class_count)
                ]
                for target_index in range(class_count)
            ],
            dtype=np.int64,
        )
        target_totals = confusion.sum(axis=1)
        total = int(target_totals.sum())
        correct = int(np.trace(confusion[:, :class_count]))
        present_class_count = int((target_totals > 0).sum())
        balanced_accuracy = balanced_accuracy_from_confusion(confusion)
        complete = bool((target_totals >= min_cell_class_count).all())
        total_cell_counts[time_index] += 1
        if complete:
            complete_cell_scores[time_index].append(balanced_accuracy)
            complete_subject_cell_scores[(subject_id, time_index)].append(
                balanced_accuracy
            )
        nuisance = nuisance_by_cell.get((subject_id, cell_id), {})
        time_value = all_times[time_index]
        cell_metric_rows.append(
            {
                **row_metadata,
                "subject_id": subject_id,
                "cell_id": cell_id,
                "current_stimulus": nuisance.get("current_stimulus", ""),
                **{field: nuisance.get(field, "") for field in cell_metadata_fields},
                "time_index": time_index,
                "time_s": time_value,
                **extra_time_metadata(time_index, time_value),
                "correct_count": correct,
                "total_count": total,
                "accuracy": correct / max(1, total),
                "present_class_count": present_class_count,
                "min_cell_class_count": min_cell_class_count,
                "complete_class_cell": int(complete),
                "balanced_accuracy": balanced_accuracy,
                "complete_cell_balanced_accuracy": (
                    balanced_accuracy if complete else ""
                ),
            }
        )

    cell_summary_rows: list[dict[str, object]] = []
    for time_index, time_value in enumerate(all_times):
        scores = complete_cell_scores.get(time_index, [])
        score_count = len(scores)
        mean_score = float(np.mean(scores)) if scores else float("nan")
        score_std = (
            float(np.std(scores, ddof=1)) if score_count > 1 else 0.0
        )
        subject_scores = [
            float(np.mean(subject_cell_scores))
            for (subject_id, subject_time_index), subject_cell_scores in sorted(
                complete_subject_cell_scores.items()
            )
            if subject_time_index == time_index and subject_cell_scores
        ]
        subject_score_count = len(subject_scores)
        subject_mean_score = (
            float(np.mean(subject_scores)) if subject_scores else float("nan")
        )
        subject_std = (
            float(np.std(subject_scores, ddof=1))
            if subject_score_count > 1
            else 0.0
        )
        total_cell_count = total_cell_counts.get(time_index, 0)
        cell_summary_rows.append(
            {
                **row_metadata,
                "time_index": time_index,
                "time_s": time_value,
                **extra_time_metadata(time_index, time_value),
                "cell_balanced_accuracy": mean_score,
                "cell_balanced_accuracy_std": score_std,
                "cell_balanced_accuracy_sem": (
                    score_std / np.sqrt(score_count)
                    if score_count
                    else float("nan")
                ),
                "cell_count": score_count,
                "total_cell_count": total_cell_count,
                "incomplete_cell_count": total_cell_count - score_count,
                "subject_cell_balanced_accuracy": subject_mean_score,
                "subject_cell_balanced_accuracy_std": subject_std,
                "subject_cell_balanced_accuracy_sem": (
                    subject_std / np.sqrt(subject_score_count)
                    if subject_score_count
                    else float("nan")
                ),
                "subject_cell_count": subject_score_count,
                "min_cell_class_count": min_cell_class_count,
            }
        )
    return cell_metric_rows, cell_summary_rows


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """返回从 1 开始的平均秩；并列值共享其占据名次的均值。"""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = np.argsort(array, kind="mergesort")
    sorted_values = array[order]
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman_rho(pred: Sequence[float], target: Sequence[float]) -> float:
    """用平均秩处理并列值后计算 Spearman 相关系数。"""
    if len(pred) != len(target):
        raise ValueError("pred and target must have the same length")
    if len(pred) < 2:
        return float("nan")
    pred_rank = average_ranks(pred)
    target_rank = average_ranks(target)
    if np.ptp(pred_rank) == 0.0 or np.ptp(target_rank) == 0.0:
        return float("nan")
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])
