"""training.objectives.target: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from math import isfinite
from typing import Mapping, Sequence
import torch
import torch.nn.functional as F
from memory_reactivation.training.augmentation import temporal_receptive_field_layers
from memory_reactivation.training.metrics import weighted_loss_mean


def probability_margin(logits: torch.Tensor, target: torch.Tensor, *, mode: str = "softmax") -> torch.Tensor:
    """计算真实类概率减去其他类别平均概率，返回每个 trial 的判别 margin。

    CE 输出用 softmax；one-vs-rest BCE 的各类概率不互斥，因此使用 sigmoid。
    """
    if mode == "sigmoid":
        probabilities = torch.sigmoid(logits.float())
    elif mode == "softmax":
        probabilities = torch.softmax(logits.float(), dim=-1)
    else:
        raise ValueError("mode must be softmax or sigmoid")
    true_probability = probabilities.gather(dim=-1, index=target[:, None]).squeeze(-1)
    other_sum = probabilities.sum(dim=-1) - true_probability
    other_count = max(1, probabilities.shape[-1] - 1)
    return true_probability - other_sum / other_count


def evidence_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """计算真实类 logit 减去其余类别平均 logit，作为未归一化 evidence。"""
    true_logit = logits.float().gather(dim=-1, index=target[:, None]).squeeze(-1)
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(dim=-1, index=target[:, None], value=False)
    other_mean = logits.float().masked_select(mask).reshape(logits.shape[0], -1).mean(dim=-1)
    return true_logit - other_mean


def _task_class_logits(
    logits: torch.Tensor,
    *,
    task_class_count: int | None,
) -> torch.Tensor:
    """截取任务真实类 logits，忽略 time no-class 等额外输出维。"""
    if task_class_count is None:
        return logits
    count = int(task_class_count)
    if count < 2:
        raise ValueError("task_class_count must be at least 2")
    if int(logits.shape[-1]) < count:
        raise ValueError(
            f"logits last dim {logits.shape[-1]} is smaller than task_class_count {count}"
        )
    return logits[..., :count]


def probability_margin_over_time(
    time_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    mode: str = "sigmoid",
    task_class_count: int | None = None,
) -> torch.Tensor:
    """对 ``[B,T,Kt]`` 逐时间点计算真实类 probability margin，返回 ``[B,T]``。

    任务类取前 ``task_class_count`` 维；no-class 不进入 other 均值。

    - ``sigmoid``（OVR BCE）：对任务类各自 sigmoid，与 trial-level 定义一致。
    - ``softmax``（CE）：先对完整 Kt 维做 softmax（保留 no-class 对概率质量的
      占用），再在任务 K 类上算 ``p_true - mean(p_other_task)``，避免“先截断再
      softmax”把 no-class 质量错误地重新归一到任务类上。
    """
    if time_logits.ndim != 3:
        raise ValueError(f"Expected [batch, time, class] time_logits, got {time_logits.shape}")
    if target.ndim != 1 or int(target.shape[0]) != int(time_logits.shape[0]):
        raise ValueError(
            f"target must be [batch] aligned with time_logits batch, got {target.shape}"
        )
    logits = time_logits.float()
    if task_class_count is None:
        task_count = int(logits.shape[-1])
    else:
        task_count = int(task_class_count)
        if task_count < 2:
            raise ValueError("task_class_count must be at least 2")
        if int(logits.shape[-1]) < task_count:
            raise ValueError(
                f"logits last dim {logits.shape[-1]} is smaller than task_class_count {task_count}"
            )
    if mode == "sigmoid":
        # OVR：no-class 与任务类独立，截取任务维即可。
        probabilities = torch.sigmoid(logits[..., :task_count])
    elif mode == "softmax":
        full_probabilities = torch.softmax(logits, dim=-1)
        probabilities = full_probabilities[..., :task_count]
    else:
        raise ValueError("mode must be softmax or sigmoid")
    index = target[:, None, None].expand(-1, probabilities.shape[1], 1)
    true_probability = probabilities.gather(dim=-1, index=index).squeeze(-1)
    other_sum = probabilities.sum(dim=-1) - true_probability
    other_count = max(1, task_count - 1)
    return true_probability - other_sum / other_count


def evidence_margin_over_time(
    time_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    task_class_count: int | None = None,
) -> torch.Tensor:
    """对 ``[B,T,Kt]`` 逐时间点计算真实类 evidence margin，返回 ``[B,T]``。

    定义与 :func:`evidence_margin` 一致：``logit_true - mean(logit_other)``。
    仅使用前 ``task_class_count`` 个任务类；no-class 维不参与。
    """
    if time_logits.ndim != 3:
        raise ValueError(f"Expected [batch, time, class] time_logits, got {time_logits.shape}")
    if target.ndim != 1 or int(target.shape[0]) != int(time_logits.shape[0]):
        raise ValueError(
            f"target must be [batch] aligned with time_logits batch, got {target.shape}"
        )
    logits = _task_class_logits(time_logits, task_class_count=task_class_count).float()
    index = target[:, None, None].expand(-1, logits.shape[1], 1)
    true_logit = logits.gather(dim=-1, index=index).squeeze(-1)
    other_sum = logits.sum(dim=-1) - true_logit
    other_count = max(1, logits.shape[-1] - 1)
    return true_logit - other_sum / other_count


def _topk_count(n_times: int, top_k: int | None, top_ratio: float) -> int:
    """把显式 k 或比例 q 转成 [1,n_times] 内的实际时间点数量。"""
    if top_k is not None:
        k = int(top_k)
        if k < 1:
            raise ValueError("attention_topk_k must be positive")
        return min(k, n_times)
    ratio = float(top_ratio)
    if not 0.0 < ratio <= 1.0:
        raise ValueError("attention_topk_ratio must be in (0, 1]")
    return max(1, min(n_times, int(round(n_times * ratio))))


def normalize_invalid_time_loss_scope(value: object) -> str:
    """规范化 covered/invalid-core 辅助损失的作用范围名称。

    ``*_covered`` 表示增强器直接返回的覆盖掩码；``*_invalid_core`` 是
    旧版感受野核心掩码语义，保留显式别名以便复现实验。没有后缀的旧
    ``topq/topk/all`` 名称继续按历史 invalid-core 语义解释。
    """
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"topk_covered", "topq_covered"}:
        return "topq_covered"
    if normalized in {"all_covered"}:
        return "all_covered"
    if normalized in {"topk", "topq", "topk_invalid_core", "topq_invalid_core"}:
        return "topq_invalid_core"
    if normalized in {"all", "all_invalid_core"}:
        return "all_invalid_core"
    raise ValueError(
        "train.invalid_time_loss_scope must be topq_covered or all_covered "
        "(invalid-core aliases are also accepted)"
    )


def target_time_receptive_field_layers(model: torch.nn.Module) -> tuple[tuple[int, int], ...]:
    """返回 target decoder 的 time logits 所依赖的 CNN 感受野层。"""
    return temporal_receptive_field_layers(model)


def attention_topk_schedule(
    *,
    n_times: int,
    top_k: int | None,
    top_ratio: float,
    iteration: int,
    training: bool,
    all_head_warmup_iterations: int,
) -> tuple[int | None, float, bool]:
    """解析当前 iteration 的 Top-q 策略。

    训练最初的 all_head_warmup_iterations 内返回全部 T 个头，之后恢复配置的
    top_k/top_ratio。验证始终使用正式 Top-q，第三个返回值标记 warmup 是否生效。
    """
    if all_head_warmup_iterations < 0:
        raise ValueError("all_time_head_warmup_iterations must be non-negative")
    # iteration 在此从 1 开始；validation 的 training=False 保证即使训练尚在
    # warmup，也始终按正式 Top-q 策略评估。
    warmup_active = (
        training
        and all_head_warmup_iterations > 0
        and 1 <= int(iteration) <= int(all_head_warmup_iterations)
    )
    if warmup_active:
        return int(n_times), 1.0, True
    return top_k, float(top_ratio), False


def _masked_position_loss_mean(
    loss_values: torch.Tensor,
    position_mask: torch.Tensor,
    sample_weight: torch.Tensor | None,
    *,
    sample_loss_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """只在掩码为 True 的位置求 loss，并保持 trial 之间的加权方式一致。

    先在每个 trial 内对有效位置等权平均，再由 sample_weight 聚合 trial；
    sample_loss_scale 只乘入分子，不改变有效 trial 或 sample_weight 的分母。
    完全没有有效位置时返回仍连接计算图的 0。
    """
    if loss_values.shape != position_mask.shape:
        raise ValueError(
            "position mask must align with per-position losses: "
            f"got {position_mask.shape} vs {loss_values.shape}"
        )
    if sample_loss_scale is not None:
        if sample_loss_scale.shape != loss_values.shape[:1]:
            raise ValueError("sample_loss_scale must contain one value per trial")
        sample_loss_scale = sample_loss_scale.to(
            device=loss_values.device, dtype=loss_values.dtype,
        )
        if not bool((torch.isfinite(sample_loss_scale) & (sample_loss_scale >= 0)).all().item()):
            raise ValueError("sample_loss_scale must be finite and non-negative")
    mask = position_mask.to(device=loss_values.device, dtype=torch.bool)
    valid_sample = mask.any(dim=1)
    if not bool(valid_sample.any().item()):
        return loss_values.sum() * 0.0
    mask_values = mask.to(dtype=loss_values.dtype)
    per_sample = (loss_values * mask_values).sum(dim=1) / mask_values.sum(dim=1).clamp_min(1.0)
    if sample_loss_scale is not None:
        per_sample = per_sample * sample_loss_scale
    selected_weight = (
        sample_weight.to(device=loss_values.device)[valid_sample]
        if sample_weight is not None
        else None
    )
    return weighted_loss_mean(per_sample[valid_sample, None], selected_weight)


def attention_topk_clean_time_loss(
    time_logits: torch.Tensor,
    target: torch.Tensor,
    time_indices: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    clean_time_mask: torch.Tensor | None,
    covered_time_mask: torch.Tensor | None = None,
    no_class_index: int | None = None,
    loss_type: str,
    top_k: int | None = None,
    top_ratio: float = 0.2,
    label_smoothing: float = 0.0,
    ordinal_sigma: float | None = None,
    bce_pos_weight: float | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    sample_weight: torch.Tensor | None = None,
    non_topq_time_loss_probability: float = 0.0,
    selection_valid_mask: torch.Tensor | None = None,
    sample_loss_scale: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """在 Top-q 及随机抽中的非 Top-q 时间点计算主 time-level 损失。

    time_logits 为 [B,T,Kt]，attention_weights 只对应 time_indices 选定的窗口。
    Top-q 是离散路由，不通过排序反传梯度。启用 no-class 时 Kt=K+1：
    被噪声直接覆盖的监督位置使用最后一类，未覆盖位置使用 trial 标签；label
    smoothing/ordinal soft label 仅在原 K 类内部生效。每个非 Top-q 候选位置
    独立按给定概率加入主损失，Top-q 永远加入。
    sample_loss_scale 只缩放各 trial 的损失贡献，归约分母与 head 选择均不变。

    返回：
        loss；Top-q 的 logits/局部索引/clean 掩码（指标用）；
        loss_valid=候选∩clean（主 time 实际监督位）；
        loss_position_mask=Top-q∪non_topq 抽样（未与 clean 求交，供 invalid 压零对齐）。
    """
    # 入参约定（debug 时对照）:
    #   time_logits [B,T,Kt]  全时间轴；Kt=K 或 K+1
    #   target [B]            0-based 真实类（不含 no-class）
    #   time_indices [Ta]     attention 窗在全 T 上的全局下标
    #   attention_weights [B,Ta]  排序用（多为 effective）
    #   clean/covered mask [B,T] 或 None，与全 T 对齐
    if time_logits.ndim != 3:
        raise ValueError(f"Expected [batch, time, class] time_logits, got {time_logits.shape}")
    if time_indices.ndim != 1 or time_indices.numel() == 0:
        raise ValueError("time_indices must be a non-empty 1D tensor")
    selected_logits = time_logits.index_select(dim=1, index=time_indices)  # [B,Ta,Kt]
    if attention_weights.shape != selected_logits.shape[:2]:
        raise ValueError(
            "attention_weights must align with selected time logits: "
            f"got {attention_weights.shape} vs {selected_logits.shape[:2]}"
        )
    if clean_time_mask is None:
        selected_clean = torch.ones(
            selected_logits.shape[:2], dtype=torch.bool, device=time_logits.device
        )  # [B,Ta] 全 True：无增强或 all_positions
    else:
        if clean_time_mask.shape != time_logits.shape[:2]:
            raise ValueError(
                "clean_time_mask must align with time_logits: "
                f"got {clean_time_mask.shape} vs {time_logits.shape[:2]}"
            )
        selected_clean = clean_time_mask.to(
            device=time_logits.device,
            dtype=torch.bool,
        ).index_select(dim=1, index=time_indices)  # [B,Ta]
    if covered_time_mask is None:
        selected_covered = torch.zeros_like(selected_clean)  # [B,Ta] 无噪声覆盖
    else:
        if covered_time_mask.shape != time_logits.shape[:2]:
            raise ValueError(
                "covered_time_mask must align with time_logits: "
                f"got {covered_time_mask.shape} vs {time_logits.shape[:2]}"
            )
        selected_covered = covered_time_mask.to(
            device=time_logits.device,
            dtype=torch.bool,
        ).index_select(dim=1, index=time_indices)  # [B,Ta] True=该窗内点被噪声替换
    non_topq_probability = float(non_topq_time_loss_probability)  # 0→只训 Top-q
    if not isfinite(non_topq_probability) or not 0.0 <= non_topq_probability <= 1.0:
        raise ValueError("non_topq_time_loss_probability must be in [0, 1]")
    k = _topk_count(selected_logits.shape[1], top_k, top_ratio)  # 每个 trial 取几个时间点
    if selection_valid_mask is not None:
        # 与 pooling 支持域对齐：Top-q 与非 Top-q 抽样都只在 pool-valid 位置进行，
        # 避免选中 pooling 明确排除的边界位置。
        if selection_valid_mask.ndim == 1 and selection_valid_mask.numel() == selected_logits.shape[1]:
            valid = selection_valid_mask.to(
                device=time_logits.device, dtype=torch.bool
            )
        elif selection_valid_mask.shape == selected_logits.shape[:2]:
            valid = selection_valid_mask.to(
                device=time_logits.device, dtype=torch.bool
            )
        else:
            raise ValueError(
                "selection_valid_mask must have shape [time] or [batch, time] "
                f"aligning with attention window, got {selection_valid_mask.shape} "
                f"for window length {selected_logits.shape[1]}"
            )
        per_trial_valid = valid.any(dim=-1) if valid.ndim == 2 else valid.any()
        if not bool(per_trial_valid.all()):
            raise ValueError(
                "selection_valid_mask must retain at least one candidate time point "
                "per trial for Top-q selection"
            )
        if valid.ndim == 1:
            # 1D（batch 无关）支持域：收缩全局 k，保证 topk 只在有效位置选择。
            k = min(k, int(valid.sum().item()))
    # top_indices: [B,k] 窗内局部下标 0..Ta-1（不是全局 T）
    selection = attention_weights.detach()
    if selection_valid_mask is not None:
        selection = torch.where(
            valid,
            selection,
            torch.full_like(selection, float("-inf")),
        )
    top_indices = torch.topk(selection, k=k, dim=1).indices
    topq_position_mask = torch.zeros_like(selected_clean)  # [B,Ta]
    topq_position_mask.scatter_(dim=1, index=top_indices, value=True)
    if selection_valid_mask is not None and valid.ndim == 2:
        # 2D 时各 trial 有效数不同，无法用全局 k 收缩；topk 仍可能返回 -inf 位置
        #（该 trial 有效点不足 k）。在 topq_position_mask 上按行清零无效位置，
        # 保证监督名额只落在 pool-valid 位置；同一 batch 其他 trial 有效数不受影响。
        topq_position_mask = topq_position_mask & valid
    loss_position_mask = topq_position_mask  # 默认：只在 Top-q 上算 time loss
    if non_topq_probability > 0.0 and k < selected_logits.shape[1]:
        sampled_non_topq = torch.rand(
            attention_weights.shape,
            device=attention_weights.device,
            dtype=torch.float32,
        ).lt(non_topq_probability)  # [B,Ta]
        if selection_valid_mask is not None:
            sampled_non_topq = sampled_non_topq & valid
        loss_position_mask = topq_position_mask | (
            sampled_non_topq & ~topq_position_mask
        )  # 指标仍只用 Top-q
    if no_class_index is not None:
        # no-class 模式：covered 位置被监督到第 K 列（null 类），因此纳入 loss_valid。
        # 与 time_decode.py 的 sampled_clean | sampled_invalid_core 语义对齐。
        loss_valid = loss_position_mask & (selected_clean | selected_covered)
    else:
        # 标准模式：只有干净位置进分类 loss
        loss_valid = loss_position_mask & selected_clean  # [B,Ta] 真正进 CE/BCE 的位置
    gathered_logits = selected_logits.gather(
        dim=1,
        index=top_indices[:, :, None].expand(-1, -1, selected_logits.shape[-1]),
    )  # [B,k,Kt] 仅 Top-q，给 accuracy
    gathered_clean = selected_clean.gather(dim=1, index=top_indices)  # [B,k]
    gathered_covered = selected_covered.gather(dim=1, index=top_indices)  # [B,k]
    if selection_valid_mask is not None and valid.ndim == 2:
        # 2D 支持域下 topk 可能返回该 trial 无效的 -inf 位置；它们不进
        # topq_position_mask，指标也应一并排除（与 loss 监督集合一致）。
        top_valid = torch.gather(valid, dim=1, index=top_indices)
        gathered_clean = gathered_clean & top_valid
        gathered_covered = gathered_covered & top_valid
    # 零系数 trial 在 CE/BCE/focal 之前屏蔽，避免先计算 NaN/Inf 再乘 0。
    # 保留 loss_valid，归约分母与 time 指标不受这个计算掩码影响。
    loss_compute_mask = loss_valid
    if sample_loss_scale is not None:
        if sample_loss_scale.shape != selected_logits.shape[:1]:
            raise ValueError("sample_loss_scale must contain one value per trial")
        loss_compute_mask = loss_compute_mask & sample_loss_scale.to(
            device=selected_logits.device,
        ).ne(0)[:, None]
    selected_logits_for_loss = torch.where(
        loss_compute_mask[:, :, None],
        selected_logits,  # [B,Ta,Kt]
        torch.zeros_like(selected_logits),
    )
    normalized_loss_type = normalize_loss_type(loss_type)
    output_class_count = int(selected_logits.shape[-1])  # Kt
    if no_class_index is not None:
        no_class_index = int(no_class_index)
        if no_class_index != output_class_count - 1:
            raise ValueError("no_class_index must be the final time-logit class")
        if no_class_index < 2:
            raise ValueError("no_class_index must follow at least two target classes")
        invalid_target = (target < 0) | (target >= no_class_index)
        if bool(invalid_target.any().item()):
            raise ValueError("target indices must be below no_class_index")
    if normalized_loss_type in {"bce_one_vs_rest", "focal_bce"}:
        if no_class_index is None:
            targets = _one_vs_rest_targets(
                target,
                output_class_count,
                label_smoothing=label_smoothing,
                ordinal_sigma=ordinal_sigma,
            ).to(dtype=selected_logits.dtype)
            targets = targets[:, None, :].expand(-1, selected_logits.shape[1], -1)
        else:
            # no-class 不是序数类别的相邻项：先在原 K 类空间构造软标签，
            # 再追加严格为 0 的 no-class 分量。
            clean_targets = _one_vs_rest_targets(
                target,
                no_class_index,
                label_smoothing=label_smoothing,
                ordinal_sigma=ordinal_sigma,
            ).to(dtype=selected_logits.dtype)
            targets = selected_logits_for_loss.new_zeros(
                selected_logits.shape[0],
                selected_logits.shape[1],
                output_class_count,
            )
            targets[..., :no_class_index] = clean_targets[:, None, :].expand(
                -1, selected_logits.shape[1], -1
            )
            targets = targets.masked_fill(selected_covered[:, :, None], 0.0)  # 噪声覆盖位清空 K 类
            targets[..., no_class_index] = selected_covered.to(
                dtype=selected_logits.dtype
            )  # 覆盖位 one-hot 到 no-class
        if normalized_loss_type == "focal_bce":
            loss_values = focal_bce_loss_values(
                selected_logits_for_loss, targets,
                gamma=focal_gamma, alpha=focal_alpha, bce_pos_weight=bce_pos_weight,
            ).mean(dim=-1)
        else:
            loss_values = F.binary_cross_entropy_with_logits(
                selected_logits_for_loss,
                targets,
                pos_weight=_bce_pos_weight(selected_logits_for_loss, bce_pos_weight),
                reduction="none",
            ).mean(dim=-1)
    else:
        if no_class_index is None:
            expanded_target = target[:, None].expand(-1, selected_logits.shape[1])
            loss_values = F.cross_entropy(
                selected_logits_for_loss.reshape(-1, output_class_count),
                expanded_target.reshape(-1),
                label_smoothing=float(label_smoothing),
                reduction="none",
            ).reshape(selected_logits.shape[:2])
        else:
            smoothing = float(label_smoothing)
            if not 0.0 <= smoothing < 1.0:
                raise ValueError("label_smoothing must be in [0, 1)")
            clean_targets = F.one_hot(
                target,
                num_classes=no_class_index,
            ).to(dtype=selected_logits.dtype)
            clean_targets = (
                (1.0 - smoothing) * clean_targets
                + smoothing / float(no_class_index)
            )  # smoothing 只在真实 K 类内
            targets = selected_logits_for_loss.new_zeros(
                selected_logits.shape[0],
                selected_logits.shape[1],
                output_class_count,
            )
            targets[..., :no_class_index] = clean_targets[:, None, :].expand(
                -1, selected_logits.shape[1], -1
            )
            targets = targets.masked_fill(selected_covered[:, :, None], 0.0)
            targets[..., no_class_index] = selected_covered.to(
                dtype=selected_logits.dtype
            )
            loss_values = -(
                targets * F.log_softmax(selected_logits_for_loss, dim=-1)
            ).sum(dim=-1)
    loss = _masked_position_loss_mean(
        loss_values, loss_valid, sample_weight, sample_loss_scale=sample_loss_scale,
    )
    # loss_position_mask：含 non_topq 抽样，未与 clean 求交（压零 scope 应对齐此集合）
    return (
        loss,
        gathered_logits,
        top_indices,
        gathered_clean,
        loss_valid,
        loss_position_mask,
    )


def invalid_core_time_bce_loss(
    time_logits: torch.Tensor,
    invalid_core_time_mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """旧版 BCE 辅助项：让完全落在噪声感受野内的位置趋向全零输出。"""
    if invalid_core_time_mask is None:
        return time_logits.sum() * 0.0
    if invalid_core_time_mask.shape != time_logits.shape[:2]:
        raise ValueError(
            "invalid_core_time_mask must align with time_logits: "
            f"got {invalid_core_time_mask.shape} vs {time_logits.shape[:2]}"
        )
    loss_values = F.softplus(time_logits).mean(dim=-1)
    return _masked_position_loss_mean(loss_values, invalid_core_time_mask, sample_weight)


def attention_topk_time_loss(
    time_logits: torch.Tensor,
    target: torch.Tensor,
    time_indices: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    top_k: int | None = None,
    top_ratio: float = 0.2,
    label_smoothing: float = 0.0,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """兼容旧调用的 CE Top-q 损失；当前统一主路径使用 clean-aware 版本。"""
    if time_logits.ndim != 3:
        raise ValueError(f"Expected [batch, time, class] time_logits, got {time_logits.shape}")
    if time_indices.ndim != 1 or time_indices.numel() == 0:
        raise ValueError("time_indices must be a non-empty 1D tensor")
    selected_logits = time_logits.index_select(dim=1, index=time_indices)
    if attention_weights.shape != selected_logits.shape[:2]:
        raise ValueError(
            "attention_weights must align with selected time logits: "
            f"got {attention_weights.shape} vs {selected_logits.shape[:2]}"
        )
    k = _topk_count(selected_logits.shape[1], top_k, top_ratio)
    top_indices = torch.topk(attention_weights.detach(), k=k, dim=1).indices
    gathered_logits = selected_logits.gather(
        dim=1,
        index=top_indices[:, :, None].expand(-1, -1, selected_logits.shape[-1]),
    )
    expanded_target = target[:, None].expand(-1, k)
    loss_values = F.cross_entropy(
        gathered_logits.reshape(-1, gathered_logits.shape[-1]),
        expanded_target.reshape(-1),
        label_smoothing=float(label_smoothing),
        reduction="none",
    ).reshape(gathered_logits.shape[:2])
    loss = weighted_loss_mean(loss_values, sample_weight)
    return loss, gathered_logits, top_indices


def _one_vs_rest_targets(
    target: torch.Tensor,
    class_count: int,
    *,
    label_smoothing: float = 0.0,
    ordinal_sigma: float | None = None,
) -> torch.Tensor:
    """为 one-vs-rest BCE 构造 [B,K] 目标，可选平滑或序数高斯软标签。

    ordinal_sigma 只对 K>2 的有序任务生效；二分类仍使用 hard/smoothed
    one-vs-rest 目标。该函数只处理真实任务类别，不负责追加 no-class。
    """
    smoothing = float(label_smoothing)
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    # 有序软标签按相邻类别距离构造高斯分布；仅 K>2 时使用，
    # 二分类任务保持普通 one-vs-rest 目标。
    if ordinal_sigma is not None and float(ordinal_sigma) > 0.0 and class_count > 2:
        sigma = float(ordinal_sigma)
        classes = torch.arange(class_count, device=target.device, dtype=torch.float32)
        distances = (classes[None, :] - target[:, None].float()).abs()
        targets = torch.exp(-distances ** 2 / (2.0 * sigma ** 2))
        targets = targets / targets.sum(dim=1, keepdim=True)
        if smoothing > 0.0:
            uniform = torch.ones_like(targets) / class_count
            targets = (1.0 - smoothing) * targets + smoothing * uniform
        return targets
    targets = torch.zeros(target.shape[0], class_count, device=target.device, dtype=torch.float32)
    if smoothing > 0.0 and class_count > 1:
        targets.fill_(smoothing / float(class_count - 1))
    targets.scatter_(dim=-1, index=target[:, None], value=1.0 - smoothing)
    return targets


def focal_bce_loss_values(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: float | None = None,
    bce_pos_weight: float | None = None,
) -> torch.Tensor:
    """逐元素 sigmoid focal BCE；保留类别/时间维，之后再做 trial 加权。

    gamma=0、alpha=None 退化为普通 BCE。alpha 可选，作用于每个 one-vs-rest
    二分类项的正/负目标，不是 Choice 类别 1 的专属权重。支持现有软目标；
    p_t = y*p + (1-y)*(1-p)，聚焦因子保留梯度。启用调制时低精度输入在
    FP32 中计算；gamma=0、alpha=None 直接沿用原 BCE 的精度及 pos_weight 路径。
    """
    gamma = normalize_focal_gamma(gamma)
    alpha = normalize_focal_alpha(alpha)
    if gamma == 0.0 and alpha is None:
        # 不能先提升 logits 精度再构造 pos_weight：其量化值会与旧 AMP/BCE
        # 路径不同，导致 gamma=0 控制组出现额外的损失差异。
        return F.binary_cross_entropy_with_logits(
            logits, targets.to(device=logits.device, dtype=logits.dtype),
            pos_weight=_bce_pos_weight(logits, bce_pos_weight), reduction="none",
        )
    values = logits.float() if logits.dtype in {torch.float16, torch.bfloat16} else logits
    targets = targets.to(device=values.device, dtype=values.dtype)
    bce = F.binary_cross_entropy_with_logits(
        values, targets, pos_weight=_bce_pos_weight(values, bce_pos_weight), reduction="none",
    )
    if gamma != 0.0:
        # 在 log 空间算 1-p_t，避免 sigmoid 饱和后 0**gamma 在 0<gamma<1
        # 时产生非有限梯度；同时支持 label smoothing/ordinal 软目标。
        log_error_probability = torch.logaddexp(
            targets.log() + F.logsigmoid(-values),
            torch.log1p(-targets) + F.logsigmoid(values),
        )
        bce = bce * (gamma * log_error_probability).exp()
    if alpha is not None:
        bce = bce * (targets * alpha + (1.0 - targets) * (1.0 - alpha))
    return bce


def one_vs_rest_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    label_smoothing: float = 0.0,
    ordinal_sigma: float | None = None,
    bce_pos_weight: float | None = None,
    focal_gamma: float | None = None,
    focal_alpha: float | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算一条分类分支的 one-vs-rest BCE，并支持 trial 样本权重。"""
    targets = _one_vs_rest_targets(
        target,
        logits.shape[-1],
        label_smoothing=label_smoothing,
        ordinal_sigma=ordinal_sigma,
    ).to(dtype=logits.dtype)
    if focal_gamma is None:
        loss_values = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=_bce_pos_weight(logits, bce_pos_weight),
            reduction="none",
        )
    else:
        loss_values = focal_bce_loss_values(
            logits, targets, gamma=focal_gamma, alpha=focal_alpha, bce_pos_weight=bce_pos_weight,
        )
    return weighted_loss_mean(loss_values, sample_weight)


def normalize_loss_type(value: object, *, name: str = "loss_type") -> str:
    """把分类损失别名规范为 ce、bce_one_vs_rest 或 focal_bce。"""
    normalized = str(value).lower().replace("-", "_")
    if normalized in {"ce", "cross_entropy"}:
        return "ce"
    if normalized in {"bce", "bce_one_vs_rest", "one_vs_rest"}:
        return "bce_one_vs_rest"
    if normalized in {"focal", "focal_bce", "sigmoid_focal", "focal_bce_one_vs_rest"}:
        return "focal_bce"
    raise ValueError(f"train.{name} must be ce, bce_one_vs_rest, or focal_bce")


def score_mode_for_loss_type(loss_type: str) -> str:
    """返回与损失匹配的导出概率变换：BCE/focal 用 sigmoid，CE 用 softmax。"""
    return "softmax" if normalize_loss_type(loss_type) == "ce" else "sigmoid"


def normalize_focal_gamma(value: object, *, name: str = "focal_gamma") -> float:
    if isinstance(value, bool):
        raise ValueError(f"train.{name} must be finite and non-negative")
    try:
        gamma = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"train.{name} must be finite and non-negative") from exc
    if not isfinite(gamma) or gamma < 0.0:
        raise ValueError(f"train.{name} must be finite and non-negative")
    return gamma


def normalize_focal_alpha(value: object, *, name: str = "focal_alpha") -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"train.{name} must be null or a finite number in [0, 1]")
    try:
        alpha = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"train.{name} must be null or a finite number in [0, 1]") from exc
    if not isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError(f"train.{name} must be null or a finite number in [0, 1]")
    return alpha


def resolve_focal_params(
    train_config: Mapping[str, object],
    branches: Sequence[str] = ("time", "pooled", "fusion"),
) -> dict[str, dict[str, float | None]]:
    """分支 gamma/alpha 覆盖全局值；默认 gamma=2、alpha=None（不做 alpha 加权）。"""
    gamma = normalize_focal_gamma(train_config.get("focal_gamma", 2.0))
    alpha = normalize_focal_alpha(train_config.get("focal_alpha"))
    return {
        branch: {
            "focal_gamma": normalize_focal_gamma(
                train_config.get(f"{branch}_focal_gamma", gamma), name=f"{branch}_focal_gamma",
            ),
            "focal_alpha": normalize_focal_alpha(
                train_config.get(f"{branch}_focal_alpha", alpha), name=f"{branch}_focal_alpha",
            ),
        }
        for branch in branches
    }


def resolve_focal_train_config(train_config: Mapping[str, object]) -> dict[str, object]:
    """把生效的 focal 默认参数显式写入配置，供 snapshot/checkpoint/重导出使用。"""
    resolved = dict(train_config)
    default_loss = normalize_loss_type(resolved.get("loss_type", "ce"))
    for branch, parameters in resolve_focal_params(resolved).items():
        if normalize_loss_type(resolved.get(f"{branch}_loss_type", default_loss)) == "focal_bce":
            resolved[f"{branch}_loss_type"] = "focal_bce"
            for name, value in parameters.items():
                resolved[f"{branch}_{name}"] = value
    return resolved


def _normalize_bce_pos_weight(value: object | None, *, name: str) -> float | None:
    """校验 BCE 正目标项权重；它不是 time/pooled 分支损失权重。"""
    if value is None:
        return None
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"train.{name} must be finite and positive") from exc
    if not isfinite(weight) or weight <= 0.0:
        raise ValueError(f"train.{name} must be finite and positive")
    return weight


def resolve_bce_pos_weights(
    train_config: Mapping[str, object],
    branches: Sequence[str] = ("time", "pooled", "fusion"),
) -> dict[str, float | None]:
    """解析各分类分支的 BCE pos_weight，分支配置优先于全局配置。"""
    global_weight = _normalize_bce_pos_weight(
        train_config.get("bce_pos_weight"),
        name="bce_pos_weight",
    )
    return {
        branch: _normalize_bce_pos_weight(
            train_config.get(f"{branch}_bce_pos_weight", global_weight),
            name=f"{branch}_bce_pos_weight",
        )
        for branch in branches
    }


def _bce_pos_weight(logits: torch.Tensor, value: float | None) -> torch.Tensor | None:
    """在 logits 的设备/精度上创建对所有输出类别相同的 pos_weight。"""
    if value is None:
        return None
    return logits.new_full((logits.shape[-1],), value)


def classification_branch_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    *,
    label_smoothing: float = 0.0,
    ordinal_sigma: float | None = None,
    bce_pos_weight: float | None = None,
    focal_gamma: float = 2.0,
    focal_alpha: float | None = None,
    sample_weight: torch.Tensor | None = None,
    valid_sample: torch.Tensor | None = None,
) -> torch.Tensor:
    """分类损失；可独立排除 trial，不影响其他分支的 sample_weight。

    先筛选再计算，分母只含有效样本；整批被排除时返回可反传的零，且不
    读取被排除的 logits（它们即使非有限也不会污染损失）。
    """
    normalized = normalize_loss_type(loss_type)
    if valid_sample is not None:
        if valid_sample.dtype != torch.bool or valid_sample.shape != (logits.shape[0],):
            raise ValueError("valid_sample must be a boolean mask with one value per trial")
        valid_sample = valid_sample.to(device=logits.device)
        if sample_weight is not None:
            sample_weight = sample_weight.to(device=logits.device).reshape(-1)
            if sample_weight.numel() != logits.shape[0]:
                raise ValueError("sample_weight must contain one value per trial")
            if not bool(torch.isfinite(sample_weight).all()) or bool((sample_weight < 0).any()):
                raise ValueError("sample_weight must be finite and non-negative")
            sample_weight = sample_weight[valid_sample]
        if not bool(valid_sample.any()) or (
            sample_weight is not None and float(sample_weight.sum().detach()) == 0.0
        ):
            return logits[:0].sum()
        logits = logits[valid_sample]
        target = target[valid_sample]
    if normalized in {"bce_one_vs_rest", "focal_bce"}:
        return one_vs_rest_loss(
            logits,
            target,
            label_smoothing=label_smoothing,
            ordinal_sigma=ordinal_sigma,
            bce_pos_weight=bce_pos_weight,
            focal_gamma=normalize_focal_gamma(focal_gamma) if normalized == "focal_bce" else None,
            focal_alpha=focal_alpha if normalized == "focal_bce" else None,
            sample_weight=sample_weight,
        )
    loss_values = F.cross_entropy(
        logits,
        target,
        label_smoothing=float(label_smoothing),
        reduction="none",
    )
    return weighted_loss_mean(loss_values, sample_weight)


def attention_topk_time_bce_loss(
    time_logits: torch.Tensor,
    target: torch.Tensor,
    time_indices: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    covered_time_mask: torch.Tensor | None = None,
    top_k: int | None = None,
    top_ratio: float = 0.2,
    label_smoothing: float = 0.0,
    ordinal_sigma: float | None = None,
    bce_pos_weight: float | None = None,
    sample_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """兼容旧配置的 BCE Top-q 损失：covered 位置使用全零目标。

    新的显式 no-class 主路径由 attention_topk_clean_time_loss 负责；保留此函数
    是为了读取历史配置和 checkpoint 时不改变既有实验语义。
    """
    if time_logits.ndim != 3:
        raise ValueError(f"Expected [batch, time, class] time_logits, got {time_logits.shape}")
    if time_indices.ndim != 1 or time_indices.numel() == 0:
        raise ValueError("time_indices must be a non-empty 1D tensor")
    selected_logits = time_logits.index_select(dim=1, index=time_indices)
    if attention_weights.shape != selected_logits.shape[:2]:
        raise ValueError(
            "attention_weights must align with selected time logits: "
            f"got {attention_weights.shape} vs {selected_logits.shape[:2]}"
        )
    if covered_time_mask is None:
        selected_covered = torch.zeros(selected_logits.shape[:2], dtype=torch.bool, device=time_logits.device)
    else:
        selected_covered = covered_time_mask.index_select(dim=1, index=time_indices).to(dtype=torch.bool)
    k = _topk_count(selected_logits.shape[1], top_k, top_ratio)
    top_indices = torch.topk(attention_weights.detach(), k=k, dim=1).indices
    gathered_logits = selected_logits.gather(
        dim=1,
        index=top_indices[:, :, None].expand(-1, -1, selected_logits.shape[-1]),
    )
    gathered_covered = selected_covered.gather(dim=1, index=top_indices)
    targets = _one_vs_rest_targets(
        target,
        gathered_logits.shape[-1],
        label_smoothing=label_smoothing,
        ordinal_sigma=ordinal_sigma,
    ).to(dtype=gathered_logits.dtype)
    targets = targets[:, None, :].expand(-1, k, -1).clone()
    # 历史语义：噪声覆盖位置无论原软标签如何，都改为全零 BCE 目标。
    targets = targets.masked_fill(gathered_covered[:, :, None], 0.0)
    loss_values = F.binary_cross_entropy_with_logits(
        gathered_logits,
        targets,
        pos_weight=_bce_pos_weight(gathered_logits, bce_pos_weight),
        reduction="none",
    )
    loss = weighted_loss_mean(loss_values, sample_weight)
    return loss, gathered_logits, top_indices, gathered_covered


def topk_covered_time_bce_loss(
    time_logits: torch.Tensor,
    covered_time_mask: torch.Tensor | None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """在指定 covered 位置计算全零目标 BCE；无覆盖时返回 0。"""
    if covered_time_mask is None or not bool(covered_time_mask.any().item()):
        return torch.tensor(0.0, device=time_logits.device, dtype=time_logits.dtype)
    zero_targets = torch.zeros_like(time_logits)
    loss_values = F.binary_cross_entropy_with_logits(
        time_logits,
        zero_targets,
        reduction="none",
    ).mean(dim=-1)
    mask = covered_time_mask.to(dtype=torch.bool)
    selected_loss = loss_values.masked_select(mask)
    selected_weight = (
        sample_weight.reshape(-1, 1).expand_as(mask).masked_select(mask)
        if sample_weight is not None
        else None
    )
    return weighted_loss_mean(selected_loss[:, None], selected_weight)


def attention_mask_penalty(
    attention_weights: torch.Tensor,
    attention_indices: torch.Tensor,
    covered_time_mask: torch.Tensor | None,
    *,
    mode: str = "absolute_effective_mass",
) -> torch.Tensor:
    """惩罚 effective attention 落在噪声覆盖区的质量。

    absolute 模式直接平均低信息区权重和；relative 模式先除以每个 trial 的
    effective attention 总质量。该约束与 time head 是否启用 no-class 相互独立。
    """
    normalized_mode = str(mode).strip().lower()
    valid_modes = {"absolute_effective_mass", "relative_effective_mass"}
    if normalized_mode not in valid_modes:
        raise ValueError(
            "attention_mask_penalty_mode must be absolute_effective_mass "
            "or relative_effective_mass"
        )
    if attention_weights.ndim != 2:
        raise ValueError(
            "attention_weights must have shape [batch, selected_time], "
            f"got {attention_weights.shape}"
        )
    if attention_indices.ndim != 1:
        raise ValueError(
            "attention_indices must be a 1D tensor, "
            f"got {attention_indices.shape}"
        )
    if attention_indices.numel() != attention_weights.shape[1]:
        raise ValueError(
            "attention_indices must align with attention_weights: "
            f"got {attention_indices.numel()} indices vs "
            f"{attention_weights.shape[1]} selected weights"
        )
    # ``index_select`` 只接受 int32/int64 索引；更窄的整数 dtype 即使是
    # 数值上的整数，也会在真正索引时才失败，因此在这里尽早报配置错误。
    if attention_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("attention_indices must use int32 or int64 indices")
    if covered_time_mask is not None:
        if covered_time_mask.ndim != 2:
            raise ValueError(
                "covered_time_mask must have shape [batch, time], "
                f"got {covered_time_mask.shape}"
            )
        if covered_time_mask.shape[0] != attention_weights.shape[0]:
            raise ValueError(
                "covered_time_mask batch dimension must align with "
                "attention_weights: "
                f"got {covered_time_mask.shape[0]} vs {attention_weights.shape[0]}"
            )
        if attention_indices.numel() > 0:
            min_index = int(attention_indices.min().item())
            max_index = int(attention_indices.max().item())
            if min_index < 0 or max_index >= covered_time_mask.shape[1]:
                raise ValueError(
                    "attention_indices must be within covered_time_mask time axis: "
                    f"got range [{min_index}, {max_index}] for "
                    f"time length {covered_time_mask.shape[1]}"
                )
    if covered_time_mask is None or not bool(covered_time_mask.any().item()):
        # 保留与 attention 分支的计算图连接；这样即使某个 batch 没有噪声，
        # 用户只启用该正则项时也能安全执行 backward。
        # 不累加已排除的权重：FP16 大 batch / 加性 gate 的总和可能先溢出。
        return attention_weights[:, :0].sum()
    selected_covered = covered_time_mask.index_select(dim=1, index=attention_indices).to(dtype=attention_weights.dtype)
    weights = attention_weights.float()
    covered_mass = (weights * selected_covered.float()).sum(dim=1)
    if normalized_mode == "relative_effective_mass":
        covered_mass = covered_mass / weights.sum(dim=1).clamp_min(1e-12)
    return covered_mass.mean().to(dtype=attention_weights.dtype)


def attention_smoothness_penalty(
    attention_weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    *,
    mode: str = "second_difference",
) -> torch.Tensor:
    """惩罚 attention 重激活曲线的局部高频起伏。

    ``attention_weights`` 是训练中用于 pooling/Top-q 的曲线：gated attention
    传入 ``effective_attention_weights``，普通 attention 传入 softmax 权重。
    每个 trial 先按有效时间点的均值归一化到平均值 1，再计算差分；这样整体
    gate 质量的缩放不会把正则项投机降到 0，约束对应的是导出时看到的曲线形状。
    默认使用二阶有限差分 ``x[t+1] - 2*x[t] + x[t-1]``，主要抑制锯齿和尖锐
    折点，而不会像一阶差分那样把所有有意义的宽峰都推向均匀分布。
    ``valid_mask`` 可排除 pool-valid-context-only 产生的无效边界；只有差分两端
    （二阶模式下的三个点）都有效时才计入正则项。

    权重设为 0 时调用方可以跳过此项；空时间轴或没有有效差分时返回与输入
    保持计算图连接的零标量，便于训练和验证共用同一条路径。
    """
    if attention_weights.ndim != 2:
        raise ValueError(
            "attention_weights must have shape [batch, time], "
            f"got {attention_weights.shape}"
        )
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"first_difference", "second_difference"}:
        raise ValueError(
            "attention_smoothness_mode must be first_difference or second_difference"
        )
    batch_size, time_count = attention_weights.shape
    if valid_mask is None:
        valid = torch.ones(
            (batch_size, time_count), dtype=torch.bool, device=attention_weights.device
        )
    elif valid_mask.ndim == 1 and valid_mask.shape[0] == time_count:
        valid = valid_mask.to(
            device=attention_weights.device, dtype=torch.bool
        )[None, :].expand(batch_size, -1)
    elif valid_mask.shape == attention_weights.shape:
        valid = valid_mask.to(device=attention_weights.device, dtype=torch.bool)
    else:
        raise ValueError(
            "valid_mask must have shape [time] or [batch, time] aligned with attention"
        )

    values = attention_weights.float().masked_fill(~valid, 0.0)
    valid_count = valid.sum(dim=1, keepdim=True).to(dtype=values.dtype).clamp_min(1.0)
    # 以有效支持上的均值为基准，保持对整体 effective mass 缩放的不变性。
    valid_mean = values.sum(dim=1, keepdim=True) / valid_count
    values = values / valid_mean.clamp_min(1e-12)
    if normalized_mode == "first_difference":
        if time_count < 2:
            return attention_weights.sum() * 0.0
        differences = values[:, 1:] - values[:, :-1]
        pair_valid = valid[:, 1:] & valid[:, :-1]
    else:
        if time_count < 3:
            return attention_weights.sum() * 0.0
        differences = values[:, 2:] - 2.0 * values[:, 1:-1] + values[:, :-2]
        pair_valid = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    if not bool(pair_valid.any().item()):
        return attention_weights.sum() * 0.0
    return differences.square().masked_select(pair_valid).mean().to(
        dtype=attention_weights.dtype
    )


TARGET_AUGMENTATION_PROBE_RATIOS = (0.10, 0.20, 0.30)


def attention_kl_to_uniform(
    attention_weights: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """计算逐 trial attention 相对其有效 support 均匀分布的 KL。"""
    if attention_weights.ndim != 2:
        raise ValueError("attention_weights must have shape [batch, time]")
    probabilities = attention_weights.float().clamp_min(0.0)
    if valid_mask is None:
        valid = torch.ones_like(probabilities, dtype=torch.bool)
    elif valid_mask.ndim == 1 and valid_mask.shape[0] == probabilities.shape[1]:
        valid = valid_mask.to(
            device=probabilities.device, dtype=torch.bool
        )[None, :].expand_as(probabilities)
    elif valid_mask.shape == probabilities.shape:
        valid = valid_mask.to(device=probabilities.device, dtype=torch.bool)
    else:
        raise ValueError(
            "valid_mask must have shape [time] or [batch, time] aligned with attention"
        )
    if not bool(valid.any(dim=1).all()):
        raise ValueError("valid_mask must retain at least one attention position per trial")
    probabilities = probabilities.masked_fill(~valid, 0.0)
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)
    valid_count = valid.sum(dim=1).to(dtype=probabilities.dtype)
    log_probability = probabilities.clamp_min(1e-12).log()
    per_trial = (
        probabilities * (log_probability + valid_count.log()[:, None])
    ).sum(dim=1)
    return per_trial.mean()
