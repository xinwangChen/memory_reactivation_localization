"""training.trainer: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from .step import TargetStepContext, target_step
from collections import defaultdict
from math import isfinite
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pb2023_eeg.models.target_decoder import TargetAttentionDecoder
from pb2023_eeg.training.attention_regularization import attention_mass_penalty, clean_attention_penalty, normalize_attention_mass_penalty_config, normalize_clean_attention_penalty_config
from pb2023_eeg.training.augmentation import EEGInputAugmenter, HierarchicalEEGAugmenter, NearFullNoiseAugmenter, build_target_eeg_augmenter, classify_temporal_coverage, normalize_near_full_noise_config, normalize_noise_supervision_mode
from pb2023_eeg.training.common import append_metrics_csv, build_optimizer, gradient_norm, load_training_checkpoint, move_batch, optimizer_learning_rate_metrics, progress_bar, recover_best_checkpoint, restore_rng_states, save_checkpoint, scheduled_iteration_learning_rate, scheduled_iteration_weight_decay, set_optimizer_learning_rate, set_optimizer_weight_decay, trainable_parameters, truncate_metrics_csv, write_rows_csv
from pb2023_eeg.training.metrics import BalancedAccuracyAccumulator, classification_cell_metadata
from pb2023_eeg.training.reconstruction import masked_eeg_reconstruction_loss
from pb2023_eeg.training.swanlab_logger import SwanLabLogger, target_swanlab_metrics
from pb2023_eeg.data.transforms import gaussian_like_eeg
from pb2023_eeg.models.adapters import _model_task_class_count
from pb2023_eeg.training.diagnostics import finalize_target_augmentation_diagnostics, target_augmentation_diagnostic_totals
from pb2023_eeg.training.objectives.target import _normalize_bce_pos_weight, _task_class_logits, _topk_count, attention_kl_to_uniform, attention_mask_penalty, attention_smoothness_penalty, attention_topk_clean_time_loss, attention_topk_schedule, classification_branch_loss, invalid_core_time_bce_loss, normalize_focal_alpha, normalize_focal_gamma, normalize_invalid_time_loss_scope, normalize_loss_type, resolve_bce_pos_weights, resolve_focal_params, resolve_focal_train_config, target_time_receptive_field_layers
from pb2023_eeg.training.precision import autocast_context
from pb2023_eeg.training.state import CyclingDataLoader


def run_target_epoch(
    model: TargetAttentionDecoder,
    loader,
    task: str,
    attention_indices: torch.Tensor,
    loss_weights: Mapping[str, float],
    top_k: int | None,
    top_ratio: float,
    label_smoothing: float,
    device: torch.device,
    amp: bool,
    loss_type: str = "ce",
    loss_types: Mapping[str, str] | None = None,
    bce_pos_weights: Mapping[str, float | None] | None = None,
    eeg_augmenter: EEGInputAugmenter | NearFullNoiseAugmenter | HierarchicalEEGAugmenter | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    grad_clip_norm: float | None = None,
    iteration_start: int = 0,
    learning_rate_fn: Callable[[int], float] | None = None,
    phase: str = "train",
    progress_bar_enabled: bool = True,
    ordinal_sigma: float | None = None,
    weight_decay_fn: Callable[[int], float | None] | None = None,
    min_cell_class_count: int = 1,
    attention_mask_penalty_mode: str = "absolute_effective_mass",
    attention_smoothness_mode: str = "second_difference",
    all_time_head_warmup_iterations: int = 0,
    invalid_time_loss_scope: str = "topq_covered",
    noise_supervision_mode: str = "rf_aware",
    non_topq_time_loss_probability: float = 0.0,
    matched_null_noise_mode: str = "sample_channel",
    matched_null_min_noise_std: float = 1e-4,
    real_noise_margin_threshold: float = 0.5,
    focal_params: Mapping[str, Mapping[str, float | None]] | None = None,
    near_full_noise_time_loss_scale: float = 1.0,
    clean_attention_penalty_probability: float = 0.1,
) -> Dict[str, float]:
    """执行一个训练或验证片段并汇总 target decoder 指标。

    optimizer 非 None 表示训练模式，否则只前向验证。每批先可选进行噪声增强，
    再由 effective attention 排序选择 Top-q time heads；time、pooled、可选
    fusion 各自计算 CE/BCE/focal 后由 loss_weights 线性组合。all_positions 模式不
    屏蔽噪声；开启 no-class 时 covered 的 Top-q time 位置监督为第 K+1 类，
    关闭 no-class 时可额外对 covered 位置启用 invalid_time 全零 BCE 辅助项。
    训练时还可按概率抽取非 Top-q 候选位置加入主 time loss；验证及正式 time
    指标仍只使用 Top-q。pooled/fusion 使用原 trial 标签，但近全噪声 trial
    会跳过这两项分类损失及其训练分类指标；其主 time 损失额外乘
    near_full_noise_time_loss_scale（默认 1），mask 监督保留。可选的
    attention_smoothness 项只作用于 effective attention 重激活曲线，不作用于
    time head。clean_attention 项仅在训练时按概率抽取未扰动有效位置惩罚
    effective attention 质量，排除 roll trial，不改变分类标签或 Top-q。
    attention_mass 项仅训练时约束每条 trial 的最终有效注意力总质量：
    relu(sum_valid |w| - 1)^2，所有 trial（包括 roll、近全噪声）均参与。
    返回值中的下划线字段只供跨片段聚合。
    """
    if grad_clip_norm is not None and (
        not isfinite(float(grad_clip_norm)) or float(grad_clip_norm) <= 0.0
    ):
        raise ValueError("train.grad_clip_norm must be finite and positive")
    clean_attention_config = normalize_clean_attention_penalty_config({
        "clean_attention_penalty_weight": loss_weights.get("clean_attention", 0.0),
        "clean_attention_penalty_probability": clean_attention_penalty_probability,
    })
    attention_mass_config = normalize_attention_mass_penalty_config({
        "attention_mass_penalty_weight": loss_weights.get("attention_mass", 0.0),
    })
    loss_weights = {str(name): float(value) for name, value in loss_weights.items()}
    if any(not isfinite(value) or value < 0.0 for value in loss_weights.values()):
        raise ValueError("loss weights must be finite and non-negative")
    near_full_noise_time_loss_scale = float(normalize_near_full_noise_config(
        {"time_loss_scale": near_full_noise_time_loss_scale}
    )["time_loss_scale"])
    attention_smoothness_mode = str(attention_smoothness_mode).strip().lower()
    if attention_smoothness_mode not in {"first_difference", "second_difference"}:
        raise ValueError(
            "attention_smoothness_mode must be first_difference or second_difference"
        )
    reconstruction_weight = float(loss_weights.get("reconstruction", 0.0))
    if reconstruction_weight > 0.0 and not bool(
        getattr(model, "reconstruction_enabled", False)
    ):
        raise ValueError(
            "reconstruction loss requires model.reconstruction.enabled=true"
        )
    base_loss_type = normalize_loss_type(loss_type)
    branch_loss_types = {"time": base_loss_type, "pooled": base_loss_type, "fusion": base_loss_type}
    if loss_types is not None:
        for branch in branch_loss_types:
            if branch in loss_types:
                branch_loss_types[branch] = normalize_loss_type(loss_types[branch], name=f"{branch}_loss_type")
    branch_bce_pos_weights = {"time": None, "pooled": None, "fusion": None}
    if bce_pos_weights is not None:
        for branch in branch_bce_pos_weights:
            if branch in bce_pos_weights:
                branch_bce_pos_weights[branch] = _normalize_bce_pos_weight(
                    bce_pos_weights[branch],
                    name=f"{branch}_bce_pos_weight",
                )
    branch_focal_params = resolve_focal_params({})
    if focal_params is not None:
        for branch in branch_focal_params:
            if branch in focal_params:
                configured = focal_params[branch]
                branch_focal_params[branch] = {
                    "focal_gamma": normalize_focal_gamma(configured.get("focal_gamma", 2.0), name=f"{branch}_focal_gamma"),
                    "focal_alpha": normalize_focal_alpha(configured.get("focal_alpha"), name=f"{branch}_focal_alpha"),
                }
    training = optimizer is not None
    attention_mass_enabled = training and attention_mass_config["weight"] > 0.0
    clean_attention_enabled = (
        training and clean_attention_config["weight"] > 0.0
        and clean_attention_config["probability"] > 0.0
    )
    clean_attention_totals = (
        torch.zeros(2, dtype=torch.long, device=device) if clean_attention_enabled else None
    )
    if training and reconstruction_weight > 0.0 and eeg_augmenter is None:
        raise ValueError(
            "reconstruction loss requires train.temporal_crop_noise augmentation"
        )
    non_topq_time_loss_probability = float(non_topq_time_loss_probability)
    if (
        not isfinite(non_topq_time_loss_probability)
        or not 0.0 <= non_topq_time_loss_probability <= 1.0
    ):
        raise ValueError("non_topq_time_loss_probability must be in [0, 1]")
    invalid_time_loss_scope = normalize_invalid_time_loss_scope(invalid_time_loss_scope)
    noise_supervision_mode = normalize_noise_supervision_mode(noise_supervision_mode)
    invalid_time_loss_weight = float(
        loss_weights.get(
            "invalid_time",
            loss_weights.get("topk_covered_time", 0.0),
        )
    )
    base_class_count = _model_task_class_count(model, task)
    time_class_count_fn = getattr(model, "time_class_count", None)
    model_time_class_count = (
        int(time_class_count_fn(task))
        if callable(time_class_count_fn)
        else base_class_count
    )
    no_class_index_fn = getattr(model, "time_no_class_index", None)
    no_class_index = (
        no_class_index_fn(task)
        if callable(no_class_index_fn)
        else None
    )
    if no_class_index is not None:
        no_class_index = int(no_class_index)
        if no_class_index != base_class_count or model_time_class_count != base_class_count + 1:
            raise ValueError(
                "time no-class output must append exactly one class after the task classes"
            )
        if eeg_augmenter is not None and noise_supervision_mode != "all_positions":
            raise ValueError(
                "time no-class supervision with augmentation requires "
                "temporal_crop_noise.supervision_mode=all_positions"
            )
    elif model_time_class_count != base_class_count:
        raise ValueError(
            "time_class_count must equal the task class count when no-class is disabled"
        )
    if noise_supervision_mode == "all_positions":
        if no_class_index is not None and invalid_time_loss_weight > 0.0:
            raise ValueError(
                "all_positions no-class supervision requires invalid_time loss weight 0"
            )
    if int(all_time_head_warmup_iterations) < 0:
        raise ValueError("all_time_head_warmup_iterations must be non-negative")
    # rf_aware：有 CNN 感受野则膨胀；无则退化为「输入 covered」语义（CNN/Transformer 共用损失组合）
    receptive_field_layers: tuple[tuple[int, int], ...] = ()
    if eeg_augmenter is not None and noise_supervision_mode == "rf_aware":
        receptive_field_layers = target_time_receptive_field_layers(model)
    model.train(training)
    context = torch.enable_grad if training else torch.no_grad
    sums = defaultdict(float)
    augmentation_diagnostic_totals: dict[str, torch.Tensor] = {}
    epoch_loss_totals: dict[str, torch.Tensor] = {}
    sample_count = 0
    trial_loss_sample_count = 0
    near_full_noise_covered_count = 0
    near_full_noise_time_count = 0
    roll_metadata_seen = False
    roll_sample_count = roll_left_count = roll_right_count = roll_abs_shift_sum = roll_abs_shift_max = 0
    time_prediction_count = 0
    has_fusion_outputs = False
    grad_norm_sum = 0.0
    finite_grad_count = 0
    nonfinite_grad_batches = 0
    last_learning_rate: float | None = None
    metric_top_k, metric_top_ratio, _metric_warmup_active = attention_topk_schedule(
        n_times=int(attention_indices.numel()),
        top_k=top_k,
        top_ratio=top_ratio,
        iteration=int(iteration_start) + 1,
        training=training,
        all_head_warmup_iterations=int(all_time_head_warmup_iterations),
    )
    selected_time_count = _topk_count(
        int(attention_indices.numel()),
        metric_top_k,
        metric_top_ratio,
    )
    metric_accumulators = {
        "time": BalancedAccuracyAccumulator(
            base_class_count,
            min_cell_class_count=min_cell_class_count,
            prediction_class_count=model_time_class_count,
        ),
        "pooled": BalancedAccuracyAccumulator(
            base_class_count,
            min_cell_class_count=min_cell_class_count,
        ),
        "fusion": BalancedAccuracyAccumulator(
            base_class_count,
            min_cell_class_count=min_cell_class_count,
        ),
    }
    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = None
    if training and total_batches:
        end_top_k, end_top_ratio, _ = attention_topk_schedule(
            n_times=int(attention_indices.numel()),
            top_k=top_k,
            top_ratio=top_ratio,
            iteration=int(iteration_start) + int(total_batches),
            training=True,
            all_head_warmup_iterations=int(all_time_head_warmup_iterations),
        )
        if _topk_count(
            int(attention_indices.numel()),
            end_top_k,
            end_top_ratio,
        ) != selected_time_count:
            raise ValueError(
                "run_target_epoch chunks must not cross the all-head warmup boundary"
            )
    progress = progress_bar(
        loader,
        desc=f"target {task} iteration {iteration_start:06d} {phase}",
        enabled=progress_bar_enabled,
        total=total_batches,
    )
    with context():
        for batch_offset, raw_batch in enumerate(progress, start=1):
            batch = move_batch(raw_batch, device)
            sample_weight = batch.get("sample_weight")  # [B] float 或 None（全 1）
            current_iteration = iteration_start + batch_offset
            batch_top_k, batch_top_ratio, warmup_active = attention_topk_schedule(
                n_times=int(attention_indices.numel()),
                top_k=top_k,
                top_ratio=top_ratio,
                iteration=current_iteration,
                training=training,
                all_head_warmup_iterations=int(all_time_head_warmup_iterations),
            )
            if _topk_count(
                int(attention_indices.numel()),
                batch_top_k,
                batch_top_ratio,
            ) != selected_time_count:
                raise ValueError(
                    "run_target_epoch chunks must not cross the all-head warmup boundary"
                )
            current_grad_norm: float | None = None
            if training:
                assert optimizer is not None
                if learning_rate_fn is not None:
                    last_learning_rate = learning_rate_fn(current_iteration)
                    set_optimizer_learning_rate(optimizer, last_learning_rate)
                if weight_decay_fn is not None:
                    wd = weight_decay_fn(current_iteration)
                    if wd is not None:
                        set_optimizer_weight_decay(optimizer, wd)
                optimizer.zero_grad(set_to_none=True)
            result = target_step(model, batch, TargetStepContext(
                amp=amp,
                attention_indices=attention_indices,
                attention_mask_penalty_mode=attention_mask_penalty_mode,
                attention_mass_config=attention_mass_config,
                attention_mass_enabled=attention_mass_enabled,
                attention_smoothness_mode=attention_smoothness_mode,
                batch_top_k=batch_top_k,
                batch_top_ratio=batch_top_ratio,
                branch_bce_pos_weights=branch_bce_pos_weights,
                branch_focal_params=branch_focal_params,
                branch_loss_types=branch_loss_types,
                clean_attention_config=clean_attention_config,
                clean_attention_enabled=clean_attention_enabled,
                device=device,
                eeg_augmenter=eeg_augmenter,
                invalid_time_loss_scope=invalid_time_loss_scope,
                invalid_time_loss_weight=invalid_time_loss_weight,
                label_smoothing=label_smoothing,
                loss_weights=loss_weights,
                matched_null_min_noise_std=matched_null_min_noise_std,
                matched_null_noise_mode=matched_null_noise_mode,
                near_full_noise_time_loss_scale=near_full_noise_time_loss_scale,
                no_class_index=no_class_index,
                noise_supervision_mode=noise_supervision_mode,
                non_topq_time_loss_probability=non_topq_time_loss_probability,
                ordinal_sigma=ordinal_sigma,
                real_noise_margin_threshold=real_noise_margin_threshold,
                receptive_field_layers=receptive_field_layers,
                reconstruction_weight=reconstruction_weight,
                task=task,
                training=training,
            ))
            loss, outputs = result.loss, result.outputs
            att_time_consistency_loss = result.losses["att_time_consistency_loss"]
            attention_mask_loss = result.losses["attention_mask_loss"]
            attention_mass_loss = result.losses["attention_mass_loss"]
            attention_smoothness_loss = result.losses["attention_smoothness_loss"]
            clean_attention_loss = result.losses["clean_attention_loss"]
            fusion_loss = result.losses["fusion_loss"]
            invalid_time_loss = result.losses["invalid_time_loss"]
            null_attention_loss = result.losses["null_attention_loss"]
            pooled_loss = result.losses["pooled_loss"]
            real_noise_margin_loss = result.losses["real_noise_margin_loss"]
            reconstruction_loss = result.losses["reconstruction_loss"]
            time_loss = result.losses["time_loss"]
            attention_strength_weights = result.diagnostics["attention_strength_weights"]
            contaminated_time_mask = result.diagnostics["contaminated_time_mask"]
            covered_time_mask = result.diagnostics["covered_time_mask"]
            halo_time_mask = result.diagnostics["halo_time_mask"]
            invalid_core_time_mask = result.diagnostics["invalid_core_time_mask"]
            invalid_loss_valid_sample = result.diagnostics["invalid_loss_valid_sample"]
            near_full_noise_covered_count += result.diagnostics["near_full_noise_covered_count"]
            near_full_noise_time_count += result.diagnostics["near_full_noise_time_count"]
            reconstruction_valid_sample = result.diagnostics["reconstruction_valid_sample"]
            roll_abs_shift_max = max(roll_abs_shift_max, result.diagnostics["roll_abs_shift_max"])
            roll_abs_shift_sum += result.diagnostics["roll_abs_shift_sum"]
            roll_left_count += result.diagnostics["roll_left_count"]
            roll_mask = result.diagnostics["roll_mask"]
            roll_metadata_seen = roll_metadata_seen or result.diagnostics["roll_metadata_seen"]
            roll_right_count += result.diagnostics["roll_right_count"]
            roll_sample_count += result.diagnostics["roll_sample_count"]
            selected_time_logits = result.diagnostics["selected_time_logits"]
            selected_valid = result.diagnostics["selected_valid"]
            selection_valid_mask = result.diagnostics["selection_valid_mask"]
            time_loss_valid = result.diagnostics["time_loss_valid"]
            top_indices = result.diagnostics["top_indices"]
            trial_loss_valid_sample = result.diagnostics["trial_loss_valid_sample"]
            if clean_attention_totals is not None:
                clean_attention_totals.add_(result.diagnostics["clean_attention_totals"])
            if training:
                assert optimizer is not None and scaler is not None
                if not scaler.is_enabled() and not torch.isfinite(loss):
                    optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("Non-finite training loss with GradScaler disabled")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                current_grad_norm = gradient_norm(model)
                if isfinite(current_grad_norm):
                    grad_norm_sum += current_grad_norm
                    finite_grad_count += 1
                else:
                    nonfinite_grad_batches += 1
                    if not scaler.is_enabled():
                        optimizer.zero_grad(set_to_none=True)
                        raise FloatingPointError("Non-finite gradients with GradScaler disabled; optimizer was not updated")
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            batch_diagnostic_totals = target_augmentation_diagnostic_totals(
                outputs["time_logits"],
                attention_strength_weights,
                attention_indices,
                covered_time_mask,
                configured_topk_count=int(selected_time_logits.shape[1]),
                configured_top_indices=top_indices,
                attention_gate=outputs.get("attention_gate"),
                invalid_core_time_mask=invalid_core_time_mask,
                no_class_index=no_class_index,
                selection_valid_mask=selection_valid_mask,
                roll_mask=roll_mask,
            )
            for key, diagnostic_value in batch_diagnostic_totals.items():
                if key not in augmentation_diagnostic_totals:
                    augmentation_diagnostic_totals[key] = diagnostic_value.clone()
                else:
                    augmentation_diagnostic_totals[key].add_(diagnostic_value)

            batch_size = int(batch["target"].shape[0])
            sample_count += batch_size

            def add_epoch_loss_total(
                name: str,
                value: torch.Tensor,
                valid_sample: torch.Tensor | None,
                *,
                use_sample_weight: bool = True,
            ) -> None:
                """按有效 trial 与 sample weight 累加可跨 batch 正确约简的损失。"""
                if valid_sample is None:
                    valid_sample = torch.ones(
                        batch_size, dtype=torch.bool, device=value.device
                    )
                else:
                    valid_sample = valid_sample.to(device=value.device, dtype=torch.bool)
                if sample_weight is None or not use_sample_weight:
                    denominator = valid_sample.sum().to(dtype=torch.float64)
                else:
                    denominator = sample_weight.to(
                        device=value.device, dtype=torch.float64
                    ).masked_select(valid_sample).sum()
                numerator = value.detach().to(dtype=torch.float64) * denominator
                for suffix, total in (("numerator", numerator), ("denominator", denominator)):
                    key = f"{name}_{suffix}"
                    if key not in epoch_loss_totals:
                        epoch_loss_totals[key] = total.clone()
                    else:
                        epoch_loss_totals[key].add_(total)

            add_epoch_loss_total("time", time_loss, time_loss_valid.any(dim=1))
            add_epoch_loss_total("pooled", pooled_loss, trial_loss_valid_sample)
            if "fusion_logits" in outputs:
                add_epoch_loss_total("fusion", fusion_loss, trial_loss_valid_sample)
            add_epoch_loss_total(
                "invalid_time", invalid_time_loss, invalid_loss_valid_sample
            )
            add_epoch_loss_total(
                "reconstruction",
                reconstruction_loss,
                reconstruction_valid_sample,
            )
            add_epoch_loss_total(
                "attention_mask",
                attention_mask_loss,
                None,
                use_sample_weight=False,
            )
            add_epoch_loss_total(
                "attention_mass",
                attention_mass_loss,
                None,
                use_sample_weight=False,
            )
            add_epoch_loss_total(
                "clean_attention",
                clean_attention_loss,
                None,
                use_sample_weight=False,
            )
            add_epoch_loss_total(
                "attention_smoothness",
                attention_smoothness_loss,
                None,
                use_sample_weight=False,
            )
            add_epoch_loss_total(
                "null_attention",
                null_attention_loss,
                None,
                use_sample_weight=False,
            )
            add_epoch_loss_total(
                "att_time_consistency",
                att_time_consistency_loss,
                None,
                use_sample_weight=False,
            )
            add_epoch_loss_total(
                "real_noise_margin",
                real_noise_margin_loss,
                trial_loss_valid_sample,
                use_sample_weight=False,
            )
            valid_time_predictions = int(selected_valid.sum().item())
            time_loss_position_count = int(time_loss_valid.sum().item())
            time_prediction_count += valid_time_predictions
            sums["loss"] += float(loss.detach().item()) * batch_size
            # time_loss / pooled_loss 的 epoch 均值由 epoch_loss_totals 累加（GPU 端），
            # 不在此额外同步到 CPU（避免每 batch 多余的 .item() 同步）。
            sums["fusion_loss"] += float(fusion_loss.detach().item()) * batch_size
            sums["invalid_time_loss"] += float(invalid_time_loss.detach().item()) * batch_size
            sums["attention_mask_loss"] += float(attention_mask_loss.detach().item()) * batch_size
            sums["attention_smoothness_loss"] += (
                float(attention_smoothness_loss.detach().item()) * batch_size
            )
            sums["null_attention_loss"] += float(null_attention_loss.detach().item()) * batch_size
            sums["att_time_consistency_loss"] += float(att_time_consistency_loss.detach().item()) * batch_size
            sums["real_noise_margin_loss"] += float(real_noise_margin_loss.detach().item()) * batch_size
            sums["selected_time_count"] += float(selected_time_logits.shape[1]) * batch_size
            sums["time_loss_position_count"] += time_loss_position_count
            sums["non_topq_time_loss_count"] += max(
                0,
                time_loss_position_count - valid_time_predictions,
            )
            if warmup_active:
                sums["all_time_head_warmup"] += batch_size
            if covered_time_mask is not None:
                sums["covered_time_fraction"] += float(covered_time_mask.float().mean().item()) * batch_size
            if contaminated_time_mask is not None:
                sums["contaminated_time_fraction"] += (
                    float(contaminated_time_mask.float().mean().item()) * batch_size
                )
            if invalid_core_time_mask is not None:
                sums["invalid_core_time_fraction"] += (
                    float(invalid_core_time_mask.float().mean().item()) * batch_size
                )
            if halo_time_mask is not None:
                sums["halo_time_fraction"] += float(halo_time_mask.float().mean().item()) * batch_size
            expanded_target = batch["target"][:, None].expand(-1, selected_time_logits.shape[1])
            selected_prediction = selected_time_logits.argmax(dim=-1)
            selected_correct = selected_prediction == expanded_target  # no-class 预测 ≠ 真实类 → 算错
            if no_class_index is not None:
                selected_no_class = selected_prediction == no_class_index
                sums["time_no_class_prediction"] += float(
                    selected_no_class.masked_select(selected_valid).sum().item()
                )
            subject_id, cell_id = classification_cell_metadata(batch, task)
            if valid_time_predictions > 0:
                sums["time_correct"] += float(selected_correct.masked_select(selected_valid).sum().item())
                # Every contributing trial counts once toward complete-cell
                # support, even when boundary/augmentation masks change q.
                # Accuracy still uses all valid selected time predictions.
                trial_support_mask = selected_valid & (
                    selected_valid.to(dtype=torch.long).cumsum(dim=1) == 1
                )
                expanded_subject = (
                    subject_id[:, None].expand_as(selected_valid)
                    if subject_id is not None
                    else None
                )
                expanded_cell = (
                    cell_id[:, None].expand_as(selected_valid)
                    if cell_id is not None
                    else None
                )
                metric_accumulators["time"].update(
                    selected_prediction[selected_valid],
                    expanded_target[selected_valid],
                    (
                        expanded_subject[selected_valid]
                        if expanded_subject is not None
                        else None
                    ),
                    expanded_cell[selected_valid] if expanded_cell is not None else None,
                    cell_support_mask=trial_support_mask[selected_valid],
                )
            pooled_prediction = outputs["pooled_logits"].argmax(dim=-1)
            trial_target = batch["target"]
            trial_subject, trial_cell = subject_id, cell_id
            if trial_loss_valid_sample is not None:
                pooled_prediction = pooled_prediction[trial_loss_valid_sample]
                trial_target = trial_target[trial_loss_valid_sample]
                if trial_subject is not None:
                    trial_subject = trial_subject[trial_loss_valid_sample]
                if trial_cell is not None:
                    trial_cell = trial_cell[trial_loss_valid_sample]
            trial_loss_sample_count += trial_target.numel()
            sums["pooled_correct"] += float(
                (pooled_prediction == trial_target).sum().item()
            )
            metric_accumulators["pooled"].update(
                pooled_prediction,
                trial_target,
                trial_subject,
                trial_cell,
            )
            if "fusion_logits" in outputs:
                has_fusion_outputs = True
                fusion_prediction = outputs["fusion_logits"].argmax(dim=-1)
                if trial_loss_valid_sample is not None:
                    fusion_prediction = fusion_prediction[trial_loss_valid_sample]
                sums["fusion_correct"] += float(
                    (fusion_prediction == trial_target).sum().item()
                )
                metric_accumulators["fusion"].update(
                    fusion_prediction,
                    trial_target,
                    trial_subject,
                    trial_cell,
                )
            postfix = {
                "loss": f"{sums['loss'] / max(1, sample_count):.4f}",
                "time_acc": f"{sums['time_correct'] / max(1, time_prediction_count):.4f}",
                "pooled_acc": (
                    f"{sums['pooled_correct'] / trial_loss_sample_count:.4f}"
                    if trial_loss_sample_count else "n/a"
                ),
            }
            if has_fusion_outputs:
                postfix["fusion_acc"] = (
                    f"{sums['fusion_correct'] / trial_loss_sample_count:.4f}"
                    if trial_loss_sample_count else "n/a"
                )
            if current_grad_norm is not None:
                postfix["grad_norm"] = f"{current_grad_norm:.3f}"
            progress.set_postfix(**postfix)
    epoch_loss_values: dict[str, float] = {}
    if epoch_loss_totals:
        total_keys = list(epoch_loss_totals)
        total_values = torch.stack(
            [epoch_loss_totals[key] for key in total_keys]
        ).cpu().tolist()
        loss_total = dict(zip(total_keys, (float(value) for value in total_values)))
        for name in (
            "time",
            "pooled",
            "fusion",
            "invalid_time",
            "attention_mask",
            "attention_mass",
            "clean_attention",
            "attention_smoothness",
            "null_attention",
            "att_time_consistency",
            "real_noise_margin",
            "reconstruction",
        ):
            denominator = loss_total.get(f"{name}_denominator", 0.0)
            epoch_loss_values[name] = (
                loss_total.get(f"{name}_numerator", 0.0) / denominator
                if denominator > 0.0
                else 0.0
            )
    else:
        epoch_loss_values = {
            name: 0.0
            for name in (
                "time",
                "pooled",
                "fusion",
                "invalid_time",
                "attention_mask",
                "attention_mass",
                "clean_attention",
                "attention_smoothness",
                "null_attention",
                "att_time_consistency",
                "real_noise_margin",
                "reconstruction",
            )
        }
    combined_epoch_loss = (
        loss_weights.get("time", 1.0) * epoch_loss_values["time"]
        + loss_weights.get("pooled", 1.0) * epoch_loss_values["pooled"]
        + loss_weights.get("fusion", 0.0) * epoch_loss_values["fusion"]
        + loss_weights.get(
            "invalid_time", loss_weights.get("topk_covered_time", 0.0)
        )
        * epoch_loss_values["invalid_time"]
        + loss_weights.get("attention_mask", 0.0)
        * epoch_loss_values["attention_mask"]
        + loss_weights.get("attention_smoothness", 0.0)
        * epoch_loss_values["attention_smoothness"]
        + loss_weights.get("null_attention", 0.0)
        * epoch_loss_values["null_attention"]
        + loss_weights.get("att_time_consistency", 0.0)
        * epoch_loss_values["att_time_consistency"]
        + loss_weights.get("real_noise_margin", 0.0)
        * epoch_loss_values["real_noise_margin"]
        + reconstruction_weight * epoch_loss_values["reconstruction"]
    )
    if clean_attention_enabled:
        combined_epoch_loss += clean_attention_config["weight"] * epoch_loss_values["clean_attention"]
    if attention_mass_enabled:
        combined_epoch_loss += attention_mass_config["weight"] * epoch_loss_values["attention_mass"]
    clean_candidate_count, clean_selected_count = (
        clean_attention_totals.cpu().tolist() if clean_attention_totals is not None else (0, 0)
    )
    metrics = {
        "loss": combined_epoch_loss,
        "time_loss": epoch_loss_values["time"],
        "pooled_loss": epoch_loss_values["pooled"],
        "invalid_time_loss": epoch_loss_values["invalid_time"],
        # 保留历史指标键，避免已有 CSV/SwanLab 面板失效。
        "topk_covered_time_loss": epoch_loss_values["invalid_time"],
        "attention_mask_loss": epoch_loss_values["attention_mask"],
        "attention_mass_loss": epoch_loss_values["attention_mass"],
        "clean_attention_loss": epoch_loss_values["clean_attention"],
        "clean_attention_candidate_count_per_sample": clean_candidate_count / max(1, sample_count),
        "clean_attention_selected_count_per_sample": clean_selected_count / max(1, sample_count),
        "clean_attention_selected_fraction": clean_selected_count / max(1, clean_candidate_count),
        "attention_smoothness_loss": epoch_loss_values["attention_smoothness"],
        "null_attention_loss": epoch_loss_values["null_attention"],
        "att_time_consistency_loss": epoch_loss_values["att_time_consistency"],
        "real_noise_margin_loss": epoch_loss_values["real_noise_margin"],
        "reconstruction_loss": epoch_loss_values["reconstruction"],
        "covered_time_fraction": sums["covered_time_fraction"] / max(1, sample_count),
        "contaminated_time_fraction": sums["contaminated_time_fraction"] / max(1, sample_count),
        "invalid_core_time_fraction": sums["invalid_core_time_fraction"] / max(1, sample_count),
        "halo_time_fraction": sums["halo_time_fraction"] / max(1, sample_count),
        "selected_time_count_per_sample": sums["selected_time_count"] / max(1, sample_count),
        "time_loss_position_count_per_sample": (
            sums["time_loss_position_count"] / max(1, sample_count)
        ),
        "non_topq_time_loss_count_per_sample": (
            sums["non_topq_time_loss_count"] / max(1, sample_count)
        ),
        "all_time_head_warmup_fraction": sums["all_time_head_warmup"] / max(1, sample_count),
        "time_accuracy": sums["time_correct"] / max(1, time_prediction_count),
        "time_no_class_prediction_fraction": (
            sums["time_no_class_prediction"] / max(1, time_prediction_count)
        ),
        "pooled_accuracy": (
            sums["pooled_correct"] / trial_loss_sample_count
            if trial_loss_sample_count else float("nan")
        ),
        "trial_loss_sample_count": float(trial_loss_sample_count),
        "near_full_noise_sample_count": float(sample_count - trial_loss_sample_count),
        "near_full_noise_fraction": (sample_count - trial_loss_sample_count) / max(1, sample_count),
        "near_full_noise_coverage_fraction": near_full_noise_covered_count / max(1, near_full_noise_time_count),
        "_sample_count": float(sample_count),
        "_time_prediction_count": float(time_prediction_count),
    }
    metrics.update(finalize_target_augmentation_diagnostics(augmentation_diagnostic_totals))
    if roll_metadata_seen:
        replaced = float(augmentation_diagnostic_totals["replacement_trial_count"].item())
        near = float(sample_count - trial_loss_sample_count)
        augmented_count = replaced + roll_sample_count
        metrics.update(
            roll_sample_count=float(roll_sample_count), roll_fraction=roll_sample_count / max(1, sample_count),
            roll_fraction_among_augmented=roll_sample_count / max(1., augmented_count),
            roll_left_fraction=roll_left_count / max(1, roll_sample_count),
            roll_right_fraction=roll_right_count / max(1, roll_sample_count),
            roll_abs_shift_mean_samples=roll_abs_shift_sum / max(1, roll_sample_count),
            roll_abs_shift_max_samples=float(roll_abs_shift_max),
            replacement_trial_fraction=replaced / max(1, sample_count),
            local_noise_fraction=(replaced - near) / max(1, sample_count),
            near_full_noise_fraction_among_replacements=near / max(1., replaced),
        )
    if has_fusion_outputs:
        metrics["fusion_loss"] = epoch_loss_values["fusion"]
        metrics["fusion_accuracy"] = (
            sums["fusion_correct"] / trial_loss_sample_count
            if trial_loss_sample_count else float("nan")
        )
    for branch in ("time", "pooled"):
        for name, value in metric_accumulators[branch].metrics().items():
            metrics[f"{branch}_{name}"] = value
    if has_fusion_outputs:
        for name, value in metric_accumulators["fusion"].metrics().items():
            metrics[f"fusion_{name}"] = value
    if training:
        metrics["grad_norm"] = grad_norm_sum / max(1, finite_grad_count)
        metrics["nonfinite_grad_batches"] = float(nonfinite_grad_batches)
        metrics["_finite_grad_count"] = float(finite_grad_count)
        assert scaler is not None
        metrics["amp_scale"] = float(scaler.get_scale())
        if last_learning_rate is not None:
            metrics["learning_rate"] = last_learning_rate
    return metrics


def _prefixed(prefix: str, metrics: Mapping[str, float]) -> dict[str, float]:
    """给公开指标批量添加 train_/val_ 前缀，并过滤内部下划线字段。"""
    return {f"{prefix}_{key}": value for key, value in metrics.items() if not key.startswith("_")}


def resolve_checkpoint_metric(
    model: TargetAttentionDecoder,
    train_config: Mapping[str, object],
    loss_weights: Mapping[str, float],
) -> str:
    """解析 best checkpoint 的监控指标，并修正与关闭分支冲突的配置。

    未显式配置时按“已训练 fusion > pooled > time”选择；fusion 已关闭却仍指定
    val_fusion_* 时自动替换为对应 pooled 指标并打印提示。
    """
    # 自动推断优先级为 fusion > pooled > time；显式配置仍可覆盖。
    trained_time = float(loss_weights.get("time", 1.0)) > 0.0
    trained_pooled = float(loss_weights.get("pooled", 1.0)) > 0.0
    trained_fusion = (
        model.fusion_enabled and float(loss_weights.get("fusion", 0.0)) > 0.0
    )
    if not any((trained_time, trained_pooled, trained_fusion)):
        raise ValueError(
            "At least one time, pooled, or learned-fusion classification loss must be positive"
        )
    # 默认使用 cell_balanced_accuracy 以应对类不均衡的验证集。
    if "checkpoint_metric" not in train_config:
        if trained_fusion:
            metric = "val_fusion_cell_balanced_accuracy"
        elif trained_pooled:
            metric = "val_pooled_cell_balanced_accuracy"
        else:
            metric = "val_time_cell_balanced_accuracy"
        print(f"[target] checkpoint_metric auto-selected: {metric}", flush=True)
        return metric
    metric = str(train_config["checkpoint_metric"])
    if not trained_fusion and metric.startswith("val_fusion_"):
        suffix = metric.removeprefix("val_fusion_")
        replacement_branch = "pooled" if trained_pooled else "time"
        replacement = f"val_{replacement_branch}_{suffix}"
        print(
            f"[target] fusion is disabled or untrained; checkpoint_metric={metric} is invalid, "
            f"using {replacement}",
            flush=True,
        )
        metric = replacement
    trained_branches = {
        "val_time_": trained_time,
        "val_pooled_": trained_pooled,
        "val_fusion_": trained_fusion,
    }
    for prefix, trained in trained_branches.items():
        if metric.startswith(prefix) and not trained:
            branch = prefix.removeprefix("val_").removesuffix("_")
            raise ValueError(
                f"checkpoint_metric {metric!r} monitors the untrained {branch} branch"
            )
    if not model.fusion_enabled and float(loss_weights.get("fusion", 0.0)) > 0.0:
        print("[target] fusion is disabled; ignoring fusion_loss_weight", flush=True)
    return metric


def update_checkpoint_state(
    *,
    iteration: int,
    score: float,
    best_score: float,
    validations_without_improvement: int,
    checkpoint_mode: str,
    checkpoint_start_iteration: int,
    early_stopping_start_iteration: int,
    patience_validations: int,
) -> tuple[float, int, bool, bool]:
    """根据一次验证更新最佳分数与 early-stopping patience 状态。

    返回新 best_score、连续未提升次数、是否保存 best、是否停止。checkpoint
    起始点之前既不保存也不累计 patience；early-stopping 起始点可独立后移。
    """
    if iteration < checkpoint_start_iteration:
        return best_score, validations_without_improvement, False, False
    improved = score > best_score if checkpoint_mode == "max" else score < best_score
    if improved:
        best_score = score
        validations_without_improvement = 0
    elif iteration >= early_stopping_start_iteration:
        validations_without_improvement += 1
    should_stop = (
        iteration >= early_stopping_start_iteration
        and validations_without_improvement >= patience_validations
    )
    return best_score, validations_without_improvement, improved, should_stop


def fit_target_decoder(
    model: TargetAttentionDecoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    task: str,
    attention_indices: torch.Tensor,
    config: Mapping[str, object],
    output_dir: Path,
    device: torch.device,
    logger: SwanLabLogger | None = None,
    resume_from: Path | None = None,
) -> list[dict[str, object]]:
    """按 iteration 训练 TargetAttentionDecoder，并定期验证和保存 checkpoint。

    这里负责把 YAML 配置解析为分支损失、Top-q/warmup、增强监督策略、优化器
    调度和早停状态；具体 batch 前向由 run_target_epoch 完成。训练日志写入
    train_metrics.csv，每次验证的合并行写入 metrics.csv，并同步到 SwanLab。
    """
    train_config = config["train"]
    assert isinstance(train_config, Mapping)
    train_config = resolve_focal_train_config(train_config)
    from .common import atomic_torch_save, load_checkpoint, resolve_learning_rate_schedule
    saved_train_config = None
    if resume_from is not None:
        saved_train_config = load_checkpoint(resume_from).get("config", {}).get("train", {})
    train_config = resolve_learning_rate_schedule(train_config, saved_train_config)
    clean_attention_config = normalize_clean_attention_penalty_config(train_config)
    attention_mass_config = normalize_attention_mass_penalty_config(train_config)
    config = {**config, "train": train_config}
    data_config = config.get("data", {})
    if not isinstance(data_config, Mapping):
        raise ValueError("data must be a mapping")
    eval_min_cell_class_count = int(
        data_config.get("eval_min_cell_class_count", 1)
    )
    if eval_min_cell_class_count < 1:
        raise ValueError("data.eval_min_cell_class_count must be positive")
    amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    iterations = int(train_config["iterations"])
    validation_interval = int(train_config.get("validation_interval_iterations", 100))
    log_interval = int(train_config.get("log_interval_iterations", validation_interval))
    if iterations < 1 or validation_interval < 1 or log_interval < 1:
        raise ValueError("iterations and intervals must be positive")
    raw_trajectory_iterations = train_config.get(
        "trajectory_checkpoint_iterations", []
    )
    if raw_trajectory_iterations is None:
        raw_trajectory_iterations = []
    if isinstance(raw_trajectory_iterations, (str, bytes)) or not isinstance(
        raw_trajectory_iterations, Sequence
    ):
        raise ValueError("train.trajectory_checkpoint_iterations must be an integer list")
    trajectory_checkpoint_iterations: set[int] = set()
    for raw_iteration in raw_trajectory_iterations:
        if isinstance(raw_iteration, bool):
            raise ValueError(
                "train.trajectory_checkpoint_iterations must contain integers"
            )
        iteration = int(raw_iteration)
        if float(raw_iteration) != iteration or not 1 <= iteration <= iterations:
            raise ValueError(
                "train.trajectory_checkpoint_iterations values must be integer "
                "iterations within train.iterations"
            )
        if iteration % validation_interval != 0 and iteration != iterations:
            raise ValueError(
                "trajectory checkpoint iterations must coincide with validation "
                "iterations"
            )
        trajectory_checkpoint_iterations.add(iteration)
    default_loss_type = normalize_loss_type(train_config.get("loss_type", "ce"), name="loss_type")
    loss_types = {
        "time": normalize_loss_type(train_config.get("time_loss_type", default_loss_type), name="time_loss_type"),
        "pooled": normalize_loss_type(train_config.get("pooled_loss_type", default_loss_type), name="pooled_loss_type"),
        "fusion": normalize_loss_type(train_config.get("fusion_loss_type", default_loss_type), name="fusion_loss_type"),
    }
    bce_pos_weights = resolve_bce_pos_weights(train_config)
    focal_params = resolve_focal_params(train_config)
    invalid_time_loss_weight = float(
        train_config.get(
            "invalid_time_loss_weight",
            train_config.get("topk_covered_time_loss_weight", 0.0),
        )
    )
    loss_weights = {
        "time": float(train_config.get("time_loss_weight", 1.0)),
        "pooled": float(train_config.get("pooled_loss_weight", 1.0)),
        "fusion": float(train_config.get("fusion_loss_weight", 0.0)),
        "invalid_time": invalid_time_loss_weight,
        "attention_mask": float(train_config.get("attention_mask_penalty_weight", 0.0)),
        "attention_mass": attention_mass_config["weight"],
        "clean_attention": clean_attention_config["weight"],
        "attention_smoothness": float(
            train_config.get("attention_smoothness_weight", 0.0)
        ),
        "null_attention": float(
            train_config.get("matched_null_attention_weight", 0.0)
        ),
        # V9 新损失项：默认关闭（权重=0），通过 YAML train.* 参数按需开启
        "att_time_consistency": float(
            train_config.get("att_time_consistency_weight", 0.0)
        ),
        "real_noise_margin": float(
            train_config.get("real_noise_margin_weight", 0.0)
        ),
        "reconstruction": float(
            train_config.get("reconstruction_loss_weight", 0.0)
        ),
    }
    if any(not isfinite(value) or value < 0.0 for value in loss_weights.values()):
        raise ValueError("loss weights must be finite and non-negative")
    if not model.fusion_enabled and loss_weights["fusion"] > 0.0:
        print("[target] fusion is disabled; ignoring fusion_loss_weight", flush=True)
        loss_weights["fusion"] = 0.0
    if (
        loss_weights["time"] <= 0.0
        and loss_weights["pooled"] <= 0.0
        and loss_weights["fusion"] <= 0.0
    ):
        raise ValueError(
            "At least one time, pooled, or learned-fusion classification loss must be positive"
        )
    attention_mask_penalty_mode = str(
        train_config.get("attention_mask_penalty_mode", "absolute_effective_mass")
    ).strip().lower()
    if attention_mask_penalty_mode not in {
        "absolute_effective_mass",
        "relative_effective_mass",
    }:
        raise ValueError(
            "train.attention_mask_penalty_mode must be absolute_effective_mass "
            "or relative_effective_mass"
        )
    attention_smoothness_mode = str(
        train_config.get("attention_smoothness_mode", "second_difference")
    ).strip().lower()
    if attention_smoothness_mode not in {
        "first_difference",
        "second_difference",
    }:
        raise ValueError(
            "train.attention_smoothness_mode must be first_difference or "
            "second_difference"
        )
    top_k_config = train_config.get("attention_topk_k")
    top_k = int(top_k_config) if top_k_config is not None else None
    top_ratio = float(train_config.get("attention_topk_ratio", 0.2))
    non_topq_time_loss_probability = float(
        train_config.get("non_topq_time_loss_probability", 0.0)
    )
    matched_null_noise_mode = str(
        train_config.get("matched_null_noise_mode", "sample_channel")
    )
    if matched_null_noise_mode not in {
        "sample_channel",
        "batch_channel",
        "standard_normal",
    }:
        raise ValueError(
            "train.matched_null_noise_mode must be sample_channel, "
            "batch_channel, or standard_normal"
        )
    matched_null_min_noise_std = float(
        train_config.get("matched_null_min_noise_std", 1e-4)
    )
    if not isfinite(matched_null_min_noise_std) or matched_null_min_noise_std < 0.0:
        raise ValueError("train.matched_null_min_noise_std must be finite and non-negative")
    if (
        not isfinite(non_topq_time_loss_probability)
        or not 0.0 <= non_topq_time_loss_probability <= 1.0
    ):
        raise ValueError(
            "train.non_topq_time_loss_probability must be in [0, 1]"
        )
    all_time_head_warmup_iterations = int(
        train_config.get("all_time_head_warmup_iterations", 0)
    )
    if all_time_head_warmup_iterations < 0:
        raise ValueError("train.all_time_head_warmup_iterations must be non-negative")
    invalid_time_loss_scope = normalize_invalid_time_loss_scope(
        train_config.get("invalid_time_loss_scope", "topq_covered")
    )
    temporal_crop_noise_config = train_config.get("temporal_crop_noise")
    near_full_noise_config = normalize_near_full_noise_config(train_config.get("near_full_noise"))
    default_noise_supervision = (
        "all_positions"
        if (
            near_full_noise_config["enabled"]
            and float(near_full_noise_config["probability"]) > 0.0
            and not (
                isinstance(temporal_crop_noise_config, Mapping)
                and bool(temporal_crop_noise_config.get("enabled", False))
            )
        )
        else "rf_aware"
    )
    if isinstance(temporal_crop_noise_config, Mapping):
        noise_supervision_mode = normalize_noise_supervision_mode(
            temporal_crop_noise_config.get("supervision_mode", default_noise_supervision)
        )
    else:
        noise_supervision_mode = default_noise_supervision
    label_smoothing = float(train_config.get("label_smoothing", 0.0))
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    grad_clip = train_config.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip) if grad_clip is not None else None
    if grad_clip_norm is not None and (not isfinite(grad_clip_norm) or grad_clip_norm <= 0.0):
        raise ValueError("train.grad_clip_norm must be finite and positive")
    progress_bar_enabled = bool(train_config.get("progress_bar", True))
    ordinal_sigma_config = train_config.get("ordinal_sigma")
    ordinal_sigma = float(ordinal_sigma_config) if ordinal_sigma_config is not None else None
    if ordinal_sigma is not None and ordinal_sigma > 0.0:
        # 用已解析的 loss_types（含各分支 override），不能直接用 default_loss_type，
        # 否则 time_loss_type 的配置 override 会被漏判。
        ordinal_loss_branches = [
            branch
            for branch, normalized in loss_types.items()
            if normalized in {"bce_one_vs_rest", "focal_bce"}
        ]
        if not ordinal_loss_branches:
            raise ValueError(
                "train.ordinal_sigma is set but every classification branch uses "
                f"loss_type 'ce' (time/pooled/fusion = {loss_types}); "
                "ordinal soft labels are only consumed by the BCE/focal paths, "
                "so this configuration would silently do nothing. Set the branch "
                "loss types to bce_one_vs_rest, or remove ordinal_sigma."
            )
        elif len(ordinal_loss_branches) < 3:
            print(
                "[target] WARNING: train.ordinal_sigma is set but only branch(es) "
                f"{ordinal_loss_branches} use BCE/focal; the remaining CE "
                "branches will ignore ordinal soft labels.",
                flush=True,
            )
    # real-noise margin 阈值：real_correct_logit - gauss_correct_logit 的最小目标间距
    real_noise_margin_threshold = float(train_config.get("real_noise_margin", 0.5))
    if not isfinite(real_noise_margin_threshold):
        raise ValueError("train.real_noise_margin must be finite")
    if (
        "real_noise_margin" in train_config
        and loss_weights["real_noise_margin"] == 0.0
    ):
        print(
            "[target] WARNING: train.real_noise_margin is set but "
            "train.real_noise_margin_weight is 0 — the margin loss will not fire. "
            "Did you mean train.real_noise_margin_weight?",
            flush=True,
        )

    def weight_decay_fn(iteration: int) -> float | None:
        """返回当前 iteration 的可选 weight decay 调度值。"""
        return scheduled_iteration_weight_decay(train_config, iteration, iterations)
    eeg_augmenter = None

    train_dataset = getattr(train_loader, "dataset", None)
    train_time_s = getattr(train_dataset, "time_s", None)
    eeg_augmenter = build_target_eeg_augmenter(
        temporal_crop_noise_config, near_full_noise=near_full_noise_config,
        augmentation_pipeline=train_config.get("augmentation_pipeline"), time_s=train_time_s,
    )
    if eeg_augmenter is not None:
        eeg_augmenter.to(device)
    if loss_weights["reconstruction"] > 0.0:
        if not bool(getattr(model, "reconstruction_enabled", False)):
            raise ValueError(
                "train.reconstruction_loss_weight requires "
                "model.reconstruction.enabled=true"
            )
        if eeg_augmenter is None:
            raise ValueError(
                "train.reconstruction_loss_weight requires temporal_crop_noise"
            )
        if float(eeg_augmenter.probability) <= 0.0:
            raise ValueError(
                "train.reconstruction_loss_weight requires positive augmentation probability"
            )

    trainable = list(trainable_parameters(model))
    if not trainable:
        raise ValueError(
            "No trainable parameters remain after freezing; unfreeze the encoder, "
            "subject conditioning, and/or classification heads before training"
        )
    optimizer = build_optimizer(trainable, train_config, model=model)
    if any(group.get("algorithm") == "muon" for group in optimizer.param_groups):
        write_rows_csv(output_dir / "optimizer_parameter_groups.csv", [
            {"group": index, "algorithm": group["algorithm"], "parameter": name,
             "shape": str(tuple(parameter.shape)), "elements": parameter.numel(),
             "peak_learning_rate": group["lr"], "lr_scale": group["lr_scale"],
             "weight_decay": group["weight_decay"]}
            for index, group in enumerate(optimizer.param_groups)
            for name, parameter in zip(group["param_names"], group["params"])
        ])
    checkpoint_metric = resolve_checkpoint_metric(model, train_config, loss_weights)
    checkpoint_mode = str(train_config.get("checkpoint_mode", "max"))
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be min or max")
    best_score = -float("inf") if checkpoint_mode == "max" else float("inf")
    patience_iterations = int(train_config.get("patience_iterations", iterations))
    patience_validations = max(1, patience_iterations // validation_interval)
    checkpoint_start_iteration = int(train_config.get("checkpoint_start_iteration", 0))
    early_stopping_start_iteration = int(
        train_config.get("early_stopping_start_iteration", checkpoint_start_iteration)
    )
    if not 0 <= checkpoint_start_iteration <= iterations:
        raise ValueError("checkpoint_start_iteration must be between 0 and train.iterations")
    if not checkpoint_start_iteration <= early_stopping_start_iteration <= iterations:
        raise ValueError(
            "early_stopping_start_iteration must be between checkpoint_start_iteration "
            "and train.iterations"
        )
    validations_without_improvement = 0
    history: list[dict[str, object]] = []
    metrics_path = output_dir / "metrics.csv"
    train_metrics_path = output_dir / "train_metrics.csv"
    stream = CyclingDataLoader(train_loader)
    current_iteration = 0
    resumed_extra: dict[str, object] = {}
    if resume_from is not None:
        resumed = load_training_checkpoint(
            resume_from,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            restore_rng=False,  # 环境（stream/skip）重建后再恢复 RNG
            expected_config=config,
        )
        current_iteration = int(resumed["iteration"])
        if current_iteration > iterations:
            raise ValueError("Resume checkpoint iteration exceeds train.iterations")
        recover_best_checkpoint(resume_from, output_dir, metric=checkpoint_metric,
                                mode=checkpoint_mode, expected_config=config)
        if resume_from.resolve() != (output_dir / "last.ckpt").resolve():
            # A successful rewind also moves last back, including terminal resumes.
            atomic_torch_save(load_checkpoint(resume_from), output_dir / "last.ckpt")
        truncate_metrics_csv(metrics_path, current_iteration)
        truncate_metrics_csv(train_metrics_path, current_iteration)
        if current_iteration >= iterations:
            restore_rng_states(resumed.get("rng_states"))
            print(
                f"[target:{task}] resume_from={resume_from} already at iteration "
                f"{current_iteration} >= {iterations}; nothing to do",
                flush=True,
            )
            return history
        stream.epoch = int(resumed["epoch"])
        resumed_extra = dict(resumed.get("extra_state", {}))
        best_score = float(resumed_extra.get("best_score", best_score))
        validations_without_improvement = int(
            resumed_extra.get("validations_without_improvement", 0)
        )
        # checkpoint 已保存本次验证更新后的 patience；终止状态也必须恢复，
        # 在重建数据游标之前退出，避免已早停的运行再次读取数据或更新参数。
        if (current_iteration >= early_stopping_start_iteration
                and validations_without_improvement >= patience_validations):
            restore_rng_states(resumed.get("rng_states"))
            print(
                f"[target:{task}] resume checkpoint already early-stopped at "
                f"iteration {current_iteration}; nothing to do",
                flush=True,
            )
            return history
        consumed_batches = int(
            resumed_extra.get("batches_in_epoch", 0)
        )
        if consumed_batches:
            # 跳过当前 epoch 内已消费的 batch，使后续数据流与中断前完全一致。
            stream.skip(consumed_batches)
        # 环境重建（iterator/skip）会消耗 torch RNG，必须最后恢复，否则恢复的
        # 随机序列被环境重建消耗，dropout/噪声增强轨迹会偏移。
        restore_rng_states(resumed.get("rng_states"))
        print(
            f"[target:{task}] resuming from iteration {current_iteration} "
            f"(data epoch {stream.epoch}, batches_consumed={consumed_batches}, "
            f"best_score={best_score})",
            flush=True,
        )
    # resume 时保留已有 CSV 追加新行，而不是清空重建。
    for path in (metrics_path, train_metrics_path):
        if path.exists() and resume_from is None:
            path.unlink()

    def learning_rate_fn(iteration: int) -> float:
        """返回当前 iteration 的基础学习率。"""
        return scheduled_iteration_learning_rate(train_config, iteration, iterations)

    while current_iteration < iterations:
        next_log = min(iterations, ((current_iteration // log_interval) + 1) * log_interval)
        next_validation = min(
            iterations,
            ((current_iteration // validation_interval) + 1) * validation_interval,
        )
        chunk_end = min(next_log, next_validation)
        if (
            current_iteration < all_time_head_warmup_iterations
            and chunk_end > all_time_head_warmup_iterations
        ):
            # 每个日志片段只采用一种 head 选择策略；否则同一训练行会混合
            # 全头 warmup 与正式 Top-q 两种不同尺度的损失。
            chunk_end = all_time_head_warmup_iterations
        train_metrics = run_target_epoch(
            model=model,
            loader=stream.take(chunk_end - current_iteration),
            task=task,
            attention_indices=attention_indices,
            loss_weights=loss_weights,
            top_k=top_k,
            top_ratio=top_ratio,
            label_smoothing=label_smoothing,
            device=device,
            loss_type=default_loss_type,
            loss_types=loss_types,
            bce_pos_weights=bce_pos_weights,
            focal_params=focal_params,
            amp=amp,
            eeg_augmenter=eeg_augmenter,
            optimizer=optimizer,
            scaler=scaler,
            grad_clip_norm=grad_clip_norm,
            iteration_start=current_iteration,
            learning_rate_fn=learning_rate_fn,
            phase="train",
            progress_bar_enabled=progress_bar_enabled,
            ordinal_sigma=ordinal_sigma,
            weight_decay_fn=weight_decay_fn,
            min_cell_class_count=1,
            attention_mask_penalty_mode=attention_mask_penalty_mode,
            attention_smoothness_mode=attention_smoothness_mode,
            all_time_head_warmup_iterations=all_time_head_warmup_iterations,
            invalid_time_loss_scope=invalid_time_loss_scope,
            noise_supervision_mode=noise_supervision_mode,
            non_topq_time_loss_probability=non_topq_time_loss_probability,
            matched_null_noise_mode=matched_null_noise_mode,
            matched_null_min_noise_std=matched_null_min_noise_std,
            real_noise_margin_threshold=real_noise_margin_threshold,
            near_full_noise_time_loss_scale=float(near_full_noise_config["time_loss_scale"]),
            clean_attention_penalty_probability=clean_attention_config["probability"],
        )
        current_iteration = chunk_end
        train_row = {
            "iteration": float(current_iteration),
            "epoch": float(stream.epoch),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            "clean_attention_penalty_weight": clean_attention_config["weight"],
            "attention_mass_penalty_weight": attention_mass_config["weight"],
            "clean_attention_penalty_probability": clean_attention_config["probability"],
            **optimizer_learning_rate_metrics(optimizer),
            "cv_fold": config.get("cv_fold"),
            "cv_folds": config.get("cv_folds"),
            **_prefixed("train", train_metrics),
        }
        append_metrics_csv(train_metrics_path, train_row)
        train_status = (
            f"[target:{task}] iteration={current_iteration:06d} "
            f"train_loss={train_row['train_loss']:.6f} "
            f"train_time_acc={train_row['train_time_accuracy']:.4f} "
            f"train_pooled_acc={train_row['train_pooled_accuracy']:.4f} "
        )
        if "train_fusion_accuracy" in train_row:
            train_status += f"train_fusion_acc={train_row['train_fusion_accuracy']:.4f} "
        train_status += (
            f"grad_norm={train_row.get('train_grad_norm', 0.0):.3f} "
            f"lr={train_row.get('train_learning_rate', float(train_config['learning_rate'])):.2e}"
        )
        print(train_status, flush=True)
        should_validate = current_iteration % validation_interval == 0 or current_iteration == iterations
        if not should_validate:
            if logger is not None:
                logger.log(target_swanlab_metrics(train_row), step=current_iteration)
            continue
        val_metrics = run_target_epoch(
            model=model,
            loader=val_loader,
            task=task,
            attention_indices=attention_indices,
            loss_weights=loss_weights,
            top_k=top_k,
            top_ratio=top_ratio,
            label_smoothing=label_smoothing,
            device=device,
            amp=amp,
            loss_type=default_loss_type,
            loss_types=loss_types,
            bce_pos_weights=bce_pos_weights,
            focal_params=focal_params,
            iteration_start=current_iteration,
            phase="val",
            progress_bar_enabled=progress_bar_enabled,
            ordinal_sigma=ordinal_sigma,
            min_cell_class_count=eval_min_cell_class_count,
            attention_mask_penalty_mode=attention_mask_penalty_mode,
            attention_smoothness_mode=attention_smoothness_mode,
            all_time_head_warmup_iterations=all_time_head_warmup_iterations,
            invalid_time_loss_scope=invalid_time_loss_scope,
            noise_supervision_mode=noise_supervision_mode,
            # 验证只使用确定性的正式 Top-q，避免随机补充点影响 checkpoint 比较。
            non_topq_time_loss_probability=0.0,
            matched_null_noise_mode=matched_null_noise_mode,
            matched_null_min_noise_std=matched_null_min_noise_std,
            real_noise_margin_threshold=real_noise_margin_threshold,
        )
        row = {
            "iteration": float(current_iteration),
            "epoch": float(stream.epoch),
            "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
            **optimizer_learning_rate_metrics(optimizer),
            "cv_fold": config.get("cv_fold"),
            "cv_folds": config.get("cv_folds"),
            **_prefixed("train", train_metrics),
            **_prefixed("val", val_metrics),
            "train_val_gap": val_metrics["loss"] - train_metrics["loss"],
            "attention_topk_k": float(top_k) if top_k is not None else 0.0,
            "attention_topk_ratio": top_ratio,
            "time_loss_type": loss_types["time"],
            "pooled_loss_type": loss_types["pooled"],
            "fusion_loss_type": loss_types["fusion"],
            **{
                f"{branch}_{parameter}": value
                for branch, parameters in focal_params.items()
                if loss_types[branch] == "focal_bce"
                for parameter, value in parameters.items()
            },
            "attention_mask_penalty_mode": attention_mask_penalty_mode,
            "clean_attention_penalty_weight": clean_attention_config["weight"],
            "attention_mass_penalty_weight": attention_mass_config["weight"],
            "clean_attention_penalty_probability": clean_attention_config["probability"],
            "attention_smoothness_mode": attention_smoothness_mode,
            "attention_smoothness_weight": loss_weights["attention_smoothness"],
            "invalid_time_loss_scope": invalid_time_loss_scope,
            "noise_supervision_mode": noise_supervision_mode,
            "invalid_time_loss_weight": invalid_time_loss_weight,
            "non_topq_time_loss_probability": non_topq_time_loss_probability,
            "matched_null_attention_weight": loss_weights["null_attention"],
            "matched_null_noise_mode": matched_null_noise_mode,
            "all_time_head_warmup_iterations": float(all_time_head_warmup_iterations),
            "learning_rate": train_metrics.get("learning_rate", float(train_config["learning_rate"])),
            "grad_norm": train_metrics.get("grad_norm", 0.0),
            "nonfinite_grad_batches": train_metrics.get("nonfinite_grad_batches", 0.0),
            "amp_scale": train_metrics.get("amp_scale", 1.0),
        }
        history.append(row)
        append_metrics_csv(metrics_path, row)
        if checkpoint_metric not in row:
            available = ", ".join(sorted(key for key in row if key.startswith("val_")))
            raise ValueError(f"checkpoint_metric {checkpoint_metric!r} is not available; val metrics: {available}")
        score = float(row[checkpoint_metric])
        if not isfinite(score):
            count_key = checkpoint_metric.replace("_cell_balanced_accuracy", "_cell_count")
            raise ValueError(
                f"checkpoint_metric {checkpoint_metric!r} is not finite "
                f"(cell_count={row.get(count_key)})"
            )
        (
            best_score,
            validations_without_improvement,
            should_save_best,
            should_stop,
        ) = update_checkpoint_state(
            iteration=current_iteration,
            score=score,
            best_score=best_score,
            validations_without_improvement=validations_without_improvement,
            checkpoint_mode=checkpoint_mode,
            checkpoint_start_iteration=checkpoint_start_iteration,
            early_stopping_start_iteration=early_stopping_start_iteration,
            patience_validations=patience_validations,
        )
        # last/trajectory 在 best/patience 状态更新之后保存，保证 resume 恢复的
        # 早停状态与本次验证一致（不落后一次验证）。
        resume_state = {
            "best_score": best_score,
            "validations_without_improvement": validations_without_improvement,
            "batches_in_epoch": int(stream.batches_in_epoch),
        }
        # Publish best before last can advertise its score. Each file remains
        # atomic; a failed best save leaves the previous recoverable last intact.
        checkpoint_paths = ([output_dir / "best_val.ckpt"] if should_save_best else [])
        checkpoint_paths.append(output_dir / "last.ckpt")
        if current_iteration in trajectory_checkpoint_iterations:
            checkpoint_paths.append(output_dir / f"trajectory_iter{current_iteration:06d}.ckpt")
        for checkpoint_path in checkpoint_paths:
            save_checkpoint(
                checkpoint_path, model, optimizer, stream.epoch, config, row,
                iteration=current_iteration, scaler=scaler, extra_state=resume_state,
                selection_metric=checkpoint_metric, selection_mode=checkpoint_mode,
            )
        if logger is not None:
            logger.log(target_swanlab_metrics(row), step=current_iteration)
        status = (
            f"[target:{task}] iteration={current_iteration:06d} "
            f"val_loss={row['val_loss']:.6f} "
            f"val_time_acc={row['val_time_accuracy']:.4f} "
            f"val_pooled_acc={row['val_pooled_accuracy']:.4f} "
        )
        if "val_fusion_accuracy" in row:
            status += f"val_fusion_acc={row['val_fusion_accuracy']:.4f} "
        status += f"grad_norm={row['grad_norm']:.3f} lr={row['learning_rate']:.2e}"
        print(status, flush=True)
        if should_stop:
            print(f"[target:{task}] early stopping at iteration {current_iteration}", flush=True)
            break
    return history
