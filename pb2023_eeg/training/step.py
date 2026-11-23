"""One Target batch: augmentation, forward and loss; optimizer updates live in trainer.

TargetStepContext contains validated values resolved once by run_target_epoch.
Diagnostics retain native numerators, masks and precision for the original reducers.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping
from ..contracts import StepResult
import torch
import torch.nn.functional as F
from pb2023_eeg.training.attention_regularization import attention_mass_penalty, clean_attention_penalty
from pb2023_eeg.training.augmentation import classify_temporal_coverage
from pb2023_eeg.training.reconstruction import masked_eeg_reconstruction_loss
from pb2023_eeg.data.transforms import gaussian_like_eeg
from pb2023_eeg.models.adapters import _model_task_class_count, boundary_noise_forward_kwargs
from pb2023_eeg.training.objectives.target import _task_class_logits, attention_kl_to_uniform, attention_mask_penalty, attention_smoothness_penalty, attention_topk_clean_time_loss, classification_branch_loss, invalid_core_time_bce_loss
from pb2023_eeg.training.precision import autocast_context

@dataclass(frozen=True)
class TargetStepContext:
    amp: bool
    attention_indices: torch.Tensor
    attention_mask_penalty_mode: str
    attention_mass_config: Mapping[str, float]
    attention_mass_enabled: bool
    attention_smoothness_mode: str
    batch_top_k: int | None
    batch_top_ratio: float
    branch_bce_pos_weights: Mapping[str, float | None]
    branch_focal_params: Mapping[str, Mapping[str, float | None]]
    branch_loss_types: Mapping[str, str]
    clean_attention_config: Mapping[str, float]
    clean_attention_enabled: bool
    device: torch.device
    eeg_augmenter: Callable | None
    invalid_time_loss_scope: str
    invalid_time_loss_weight: float
    label_smoothing: float
    loss_weights: Mapping[str, float]
    matched_null_min_noise_std: float
    matched_null_noise_mode: str
    near_full_noise_time_loss_scale: float
    no_class_index: int | None
    noise_supervision_mode: str
    non_topq_time_loss_probability: float
    ordinal_sigma: float | None
    real_noise_margin_threshold: float
    receptive_field_layers: tuple[tuple[int, int], ...]
    reconstruction_weight: float
    task: str
    training: bool


def target_step(model, batch, context: TargetStepContext) -> StepResult:
    amp = context.amp
    attention_indices = context.attention_indices
    attention_mask_penalty_mode = context.attention_mask_penalty_mode
    attention_mass_config = context.attention_mass_config
    attention_mass_enabled = context.attention_mass_enabled
    attention_smoothness_mode = context.attention_smoothness_mode
    batch_top_k = context.batch_top_k
    batch_top_ratio = context.batch_top_ratio
    branch_bce_pos_weights = context.branch_bce_pos_weights
    branch_focal_params = context.branch_focal_params
    branch_loss_types = context.branch_loss_types
    clean_attention_config = context.clean_attention_config
    clean_attention_enabled = context.clean_attention_enabled
    device = context.device
    eeg_augmenter = context.eeg_augmenter
    invalid_time_loss_scope = context.invalid_time_loss_scope
    invalid_time_loss_weight = context.invalid_time_loss_weight
    label_smoothing = context.label_smoothing
    loss_weights = context.loss_weights
    matched_null_min_noise_std = context.matched_null_min_noise_std
    matched_null_noise_mode = context.matched_null_noise_mode
    near_full_noise_time_loss_scale = context.near_full_noise_time_loss_scale
    no_class_index = context.no_class_index
    noise_supervision_mode = context.noise_supervision_mode
    non_topq_time_loss_probability = context.non_topq_time_loss_probability
    ordinal_sigma = context.ordinal_sigma
    real_noise_margin_threshold = context.real_noise_margin_threshold
    receptive_field_layers = context.receptive_field_layers
    reconstruction_weight = context.reconstruction_weight
    task = context.task
    training = context.training
    sample_weight = batch.get("sample_weight")
    near_full_noise_covered_count = 0
    near_full_noise_time_count = 0
    roll_abs_shift_max = 0
    roll_abs_shift_sum = 0
    roll_left_count = 0
    roll_metadata_seen = False
    roll_right_count = 0
    roll_sample_count = 0
    clean_attention_totals = torch.zeros(2, dtype=torch.long, device=device) if clean_attention_enabled else None
    eeg = batch["eeg"]  # [B,C,T]
    clean_eeg = eeg.detach()
    trial_loss_valid_sample: torch.Tensor | None = None
    time_sample_loss_scale: torch.Tensor | None = None
    # 下列 mask 均为 [B,T] bool 或 None（验证/无增强时多为 None）
    covered_time_mask: torch.Tensor | None = None  # True=该 t 被噪声直接替换
    clean_time_mask: torch.Tensor | None = None  # True=感受野干净（rf_aware）
    invalid_core_time_mask: torch.Tensor | None = None  # 感受野全在噪声里
    halo_time_mask: torch.Tensor | None = None  # 半脏
    contaminated_time_mask: torch.Tensor | None = None  # 给 attention mask penalty
    roll_mask: torch.Tensor | None = None
    if training and eeg_augmenter is not None:
        augmented = eeg_augmenter(eeg, return_mask=True)
        eeg = augmented.eeg  # [B,C,T] 已掺噪
        covered_time_mask = augmented.covered_time_mask  # [B,T]
        near_full_noise_mask = getattr(augmented, "near_full_noise_mask", None)
        if near_full_noise_mask is not None:
            if near_full_noise_mask.dtype != torch.bool or near_full_noise_mask.shape != (eeg.shape[0],):
                raise ValueError("near_full_noise_mask must contain one boolean per trial")
            trial_loss_valid_sample = ~near_full_noise_mask
            if near_full_noise_time_loss_scale != 1.0:
                time_sample_loss_scale = torch.where(
                    near_full_noise_mask, near_full_noise_time_loss_scale, 1.0,
                )
            near_covered = covered_time_mask[near_full_noise_mask]
            near_full_noise_covered_count += int(near_covered.sum().item())
            near_full_noise_time_count += near_covered.numel()
        roll_mask = getattr(augmented, "roll_mask", None)
        if roll_mask is not None:
            roll_metadata_seen = True
            shifts = getattr(augmented, "roll_shifts", None)
            if not isinstance(roll_mask, torch.Tensor) or roll_mask.dtype != torch.bool or roll_mask.shape != (eeg.shape[0],) or roll_mask.device != eeg.device:
                raise ValueError("roll_mask must contain one on-device boolean per trial")
            if not isinstance(shifts, torch.Tensor) or shifts.dtype != torch.long or shifts.shape != roll_mask.shape or shifts.device != eeg.device:
                raise ValueError("roll_shifts must contain one on-device signed integer per trial")
            if bool((roll_mask & covered_time_mask.any(dim=1)).any().item()):
                raise ValueError("Roll trials must have no noise-covered positions")
            if near_full_noise_mask is not None and bool((roll_mask & near_full_noise_mask).any().item()):
                raise ValueError("Roll and near-full-noise trials must be disjoint")
            if bool((shifts[~roll_mask] != 0).any().item()) or bool((shifts[roll_mask] == 0).any().item()):
                raise ValueError("Only roll trials may have nonzero roll_shifts")
            selected_shifts = shifts[roll_mask]
            if selected_shifts.numel() and int(selected_shifts.abs().max().item()) >= eeg.shape[-1]:
                raise ValueError("roll_shifts must be smaller than the EEG time-axis length")
            roll_sample_count += int(roll_mask.sum().item())
            roll_left_count += int((selected_shifts < 0).sum().item())
            roll_right_count += int((selected_shifts > 0).sum().item())
            roll_abs_shift_sum += int(selected_shifts.abs().sum().item())
            if selected_shifts.numel():
                roll_abs_shift_max = max(roll_abs_shift_max, int(selected_shifts.abs().max().item()))
        if noise_supervision_mode == "rf_aware":
            if receptive_field_layers:
                coverage = classify_temporal_coverage(
                    covered_time_mask,
                    receptive_field_layers,
                )
                clean_time_mask = coverage.clean_time_mask
                invalid_core_time_mask = coverage.invalid_core_time_mask
                halo_time_mask = coverage.halo_time_mask
                contaminated_time_mask = ~clean_time_mask
            else:
                # Transformer 等无 depthwise RF：与输入 covered 对齐，不膨胀
                clean_time_mask = ~covered_time_mask
                invalid_core_time_mask = covered_time_mask
                halo_time_mask = torch.zeros_like(covered_time_mask)
                contaminated_time_mask = covered_time_mask
        else:
            # all_positions：噪声位仍可进 Top-q / attention
            # - 开 no-class：L_time 用 trial 或 no-class 标签；禁止 invalid 压零
            # - 关 no-class：L_time 仅 trial 标签监督非噪声位；噪声位只靠 invalid 压零 BCE
            contaminated_time_mask = covered_time_mask
            if no_class_index is not None:
                clean_time_mask = None  # 全 True；covered 在 loss 内改为 no-class 目标
                invalid_core_time_mask = torch.zeros_like(covered_time_mask)
            else:
                clean_time_mask = ~covered_time_mask  # 主 time 分类排除噪声位
                invalid_core_time_mask = covered_time_mask  # 压零 BCE 作用在 covered
    boundary_kwargs = boundary_noise_forward_kwargs(model, batch)
    with autocast_context(device, enabled=amp):
        outputs = model(
            eeg=eeg,
            task=task,
            attention_indices=attention_indices,  # [Ta] 全局时间下标
            subject_id=batch.get("subject_id"),  # [B]
            **boundary_kwargs,
        )
        # outputs 键：time_logits[B,T,Kt] pooled_logits[B,K]
        # attention_weights/effective_attention_weights [B,Ta]
        effective_smoothing = label_smoothing if training else 0.0
        pooling_weights = outputs.get(
            "effective_attention_weights", outputs["attention_weights"]
        )  # [B,Ta] 实际 pooling 权重；tanh 探索允许正负贡献
        attention_strength_weights = outputs.get(
            "attention_magnitude_weights", pooling_weights
        )  # 非负强度：旧模式不变，signed 模式使用 |pooling_weights|
        time_selection_weights = outputs.get(
            "time_selection_attention_weights", attention_strength_weights
        )  # [B,Ta] Top-q 排序权重；pool_valid_context_only 下与 pooling 支持域一致
        selection_valid_mask = outputs.get("attention_valid_mask")
        if (
            noise_supervision_mode == "rf_aware"
            and contaminated_time_mask is not None
            and covered_time_mask is not None
            and attention_indices is not None
        ):
            # rf_aware 下"受污染但未被直接覆盖"的 halo 位置不进主 time 分类
            # 监督（clean=False），默认 topq_covered 压零 scope 也不覆盖它们：
            # 若被 Top-q 选中则梯度恒为 0。把 halo 位置从 Top-q 候选集合
            # 硬排除（-inf 排序 + k 收缩），而不是只把权重置零——top_k 超过
            # 有效候选数时置零仍可能被 topk 选中，监督空洞依然存在。covered
            # 位置保留候选资格以接受 invalid 压零 BCE。
            uncovered_contaminated = (
                contaminated_time_mask & ~covered_time_mask
            ).index_select(dim=1, index=attention_indices)  # [B,Ta]
            if selection_valid_mask is None:
                selection_valid_mask = ~uncovered_contaminated
            elif selection_valid_mask.ndim == 1:
                selection_valid_mask = (
                    selection_valid_mask[None, :].expand(
                        uncovered_contaminated.shape[0], -1
                    )
                    & ~uncovered_contaminated
                )
            else:
                selection_valid_mask = (
                    selection_valid_mask & ~uncovered_contaminated
                )
        null_attention_loss = outputs["attention_weights"].sum() * 0.0
        # ------------------------------------------------------------------
        # V9 att-time 一致性损失（att_time_consistency_weight）
        # 强制 attention 分布向 time head 的逐时正确类 evidence 集中。
        # 仅训练时启用；grad 不穿过 time_logits（stop_gradient）。
        # ------------------------------------------------------------------
        att_time_consistency_loss = outputs["attention_weights"].sum() * 0.0
        if training and float(loss_weights.get("att_time_consistency", 0.0)) > 0.0:
            task_k = _model_task_class_count(model, task)
            with torch.no_grad():
                # time_logits: [B, T, Kt]；截取任务类维
                task_time_logits = outputs["time_logits"][..., :task_k].float()
                # 每时间点的正确类 logit → [B, T]
                _tgt_idx = batch["target"][:, None, None].expand(
                    -1, task_time_logits.shape[1], 1
                )
                correct_time_logit = task_time_logits.gather(-1, _tgt_idx).squeeze(-1)
                # softmax over time → 目标分布 [B, T]（全轴）
                time_att_target_full = torch.softmax(correct_time_logit, dim=-1)
                # 截取 attention 窗口内的子分布并重归一化
                time_att_target = time_att_target_full.index_select(
                    dim=1, index=attention_indices
                )
                time_att_target = time_att_target / (
                    time_att_target.sum(dim=-1, keepdim=True).clamp(min=1e-8)
                )
            # 对非负贡献强度归一化；signed 模式以 |w| 对齐时间证据，
            # 不能把抵消后的代数和当作概率分母。
            consistency_weights = attention_strength_weights
            if consistency_weights.dtype in (torch.float16, torch.bfloat16):
                # tanh 可精确输出零，sigmoid 也可能因饱和而下溢。
                # FP16 的 1e-8 下溢为0；归一化及 log 使用 FP32，
                # 避免 0/0、log(0)，实际 gate / pooling 不受影响。
                consistency_weights = consistency_weights.float()
            pw_norm = consistency_weights / consistency_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            att_time_consistency_loss = F.kl_div(
                input=pw_norm.clamp(min=1e-8).log(),
                target=time_att_target,
                reduction="batchmean",
                log_target=False,
            )
        # ------------------------------------------------------------------
        # V9 real-noise margin 损失（real_noise_margin_weight）
        # 对每个 batch 额外做一次全 Gaussian forward（encoder 参与 grad），
        # 要求当前增强后 EEG 的正确类 pooled logit 比 Gaussian 版高出 margin。
        # 近全噪声 trial 同样不参与依赖 trial 标签的 margin 监督。
        # ------------------------------------------------------------------
        real_noise_margin_loss = outputs["attention_weights"].sum() * 0.0
        if (
            training
            and float(loss_weights.get("real_noise_margin", 0.0)) > 0.0
            and (
                trial_loss_valid_sample is None
                or bool(trial_loss_valid_sample.any())
            )
        ):
            noise_eeg_margin = gaussian_like_eeg(
                batch["eeg"],
                noise_mode=matched_null_noise_mode,
                min_noise_std=matched_null_min_noise_std,
            )
            margin_null_outputs = model(
                eeg=noise_eeg_margin,
                task=task,
                attention_indices=attention_indices,
                subject_id=batch.get("subject_id"),
                **boundary_kwargs,
            )
            task_k = _model_task_class_count(model, task)
            margin_target = batch["target"]
            real_margin_logits = _task_class_logits(
                outputs["pooled_logits"], task_class_count=task_k
            )
            gauss_margin_logits = _task_class_logits(
                margin_null_outputs["pooled_logits"], task_class_count=task_k
            )
            if trial_loss_valid_sample is not None:
                margin_target = margin_target[trial_loss_valid_sample]
                real_margin_logits = real_margin_logits[trial_loss_valid_sample]
                gauss_margin_logits = gauss_margin_logits[trial_loss_valid_sample]
            _margin_tgt = margin_target.unsqueeze(-1)  # [有效 B, 1]
            real_correct_logit = real_margin_logits.gather(-1, _margin_tgt).squeeze(-1)
            gauss_correct_logit = gauss_margin_logits.gather(-1, _margin_tgt).squeeze(-1)
            real_noise_margin_loss = F.relu(
                real_noise_margin_threshold - (real_correct_logit - gauss_correct_logit)
            ).mean()

        if training and float(loss_weights.get("null_attention", 0.0)) > 0.0:
            null_eeg = gaussian_like_eeg(
                batch["eeg"],
                noise_mode=matched_null_noise_mode,
                min_noise_std=matched_null_min_noise_std,
            )
            null_outputs = model(
                eeg=null_eeg,
                task=task,
                attention_indices=attention_indices,
                subject_id=batch.get("subject_id"),
                **boundary_kwargs,
            )
            null_weights = null_outputs.get(
                "attention_magnitude_weights",
                null_outputs.get("effective_attention_weights", null_outputs["attention_weights"]),
            )
            null_attention_loss = attention_kl_to_uniform(
                null_weights,
                null_outputs.get("attention_valid_mask"),
            )
        time_loss_type = branch_loss_types["time"]
        pooled_loss_type = branch_loss_types["pooled"]
        fusion_loss_type = branch_loss_types["fusion"]
        (
            time_loss,  # 标量
            selected_time_logits,  # [B,k,Kt] Top-q logits（指标）
            top_indices,  # [B,k] 窗内局部下标
            selected_valid,  # [B,k] Top-q 是否 clean（指标用）
            time_loss_valid,  # [B,Ta] 主 time 实际监督位（候选∩clean）
            time_candidate_mask,  # [B,Ta] Top-q∪non_topq 抽样（未∩clean）
        ) = attention_topk_clean_time_loss(
            outputs["time_logits"],
            batch["target"],  # [B]
            attention_indices,
            time_selection_weights,
            clean_time_mask=clean_time_mask,
            covered_time_mask=covered_time_mask,
            no_class_index=no_class_index,  # None 或 =K
            loss_type=time_loss_type,
            top_k=batch_top_k,
            top_ratio=batch_top_ratio,
            label_smoothing=effective_smoothing,
            ordinal_sigma=ordinal_sigma,
            bce_pos_weight=branch_bce_pos_weights["time"],
            **branch_focal_params["time"],
            sample_weight=sample_weight,
            non_topq_time_loss_probability=(
                non_topq_time_loss_probability if training else 0.0
            ),
            selection_valid_mask=selection_valid_mask,
            sample_loss_scale=time_sample_loss_scale,
        )
        # 窗内全长 logits，供 topq_* 压零覆盖 non_topq 噪声位
        window_time_logits = outputs["time_logits"].index_select(
            dim=1, index=attention_indices
        )
        # 空切片求和保留计算图；未启用的辅助项不读取非有限 time 输出。
        zero_time_loss = outputs["time_logits"][..., :0].sum()
        if invalid_time_loss_weight == 0.0:
            # 权重 0 是该辅助目标的显式关闭开关；不要仅计算后再乘 0，
            # 否则日志会看起来像仍启用了 invalid-time 监督。
            invalid_time_loss = zero_time_loss
            invalid_loss_valid_sample = torch.zeros(
                batch["target"].shape[0],
                dtype=torch.bool,
                device=device,
            )
        elif noise_supervision_mode == "all_positions" and no_class_index is not None:
            # 此模式禁止 invalid/noise-suppression 辅助目标：上面的普通
            # CE/BCE 已经把 covered 的候选头监督为显式 no-class。
            invalid_time_loss = zero_time_loss
            invalid_loss_valid_sample = torch.zeros(
                batch["target"].shape[0],
                dtype=torch.bool,
                device=device,
            )
        elif invalid_time_loss_scope in {
            "topq_covered",
            "topq_invalid_core",
        }:
            # 名称仍叫 topq_*，但压零范围 = 噪声区 ∩ (Top-q ∪ non_topq 抽样)
            # 与 L_time 候选集合对齐，避免 non_topq 噪声位既无 trial 也无压零。
            scope_time_mask = (
                covered_time_mask
                if invalid_time_loss_scope == "topq_covered"
                else invalid_core_time_mask
            )
            if scope_time_mask is None:
                selected_invalid = torch.zeros(
                    batch["target"].shape[0],
                    int(attention_indices.numel()),
                    dtype=torch.bool,
                    device=device,
                )
            else:
                selected_invalid = (
                    scope_time_mask.index_select(
                        dim=1, index=attention_indices
                    )
                    & time_candidate_mask
                )
            invalid_time_loss = invalid_core_time_bce_loss(
                window_time_logits,
                selected_invalid,
                sample_weight=sample_weight,
            )
            invalid_loss_valid_sample = selected_invalid.any(dim=1)
        else:
            scope_time_mask = (
                covered_time_mask
                if invalid_time_loss_scope == "all_covered"
                else invalid_core_time_mask
            )
            invalid_time_loss = invalid_core_time_bce_loss(
                outputs["time_logits"],
                scope_time_mask,
                sample_weight=sample_weight,
            )
            invalid_loss_valid_sample = (
                scope_time_mask.any(dim=1)
                if scope_time_mask is not None
                else torch.zeros(
                    batch["target"].shape[0], dtype=torch.bool, device=device
                )
            )
        attention_mask_loss = attention_mask_penalty(
            attention_strength_weights,
            attention_indices,
            contaminated_time_mask,
            mode=attention_mask_penalty_mode,
        )
        attention_mass_loss = attention_mask_loss.new_zeros(())
        if attention_mass_enabled:
            attention_mass_loss = attention_mass_penalty(
                pooling_weights,
                valid_mask=outputs.get("attention_valid_mask"),
            )
        clean_attention_loss = attention_mask_loss.new_zeros(())
        if clean_attention_enabled:
            clean_attention_loss, clean_candidates, clean_selected = clean_attention_penalty(
                attention_strength_weights,
                attention_indices,
                contaminated_time_mask,
                probability=clean_attention_config["probability"],
                valid_mask=outputs.get("attention_valid_mask"),
                roll_mask=roll_mask,
            )
            assert clean_attention_totals is not None
            clean_attention_totals.add_(torch.stack((
                clean_candidates.sum(), clean_selected.sum(),
            )))
        attention_smoothness_loss = attention_smoothness_penalty(
            attention_strength_weights,
            outputs.get("attention_valid_mask"),
            mode=attention_smoothness_mode,
        )
        pooled_loss = classification_branch_loss(
            outputs["pooled_logits"],  # [B,K]
            batch["target"],
            pooled_loss_type,
            label_smoothing=effective_smoothing,
            ordinal_sigma=ordinal_sigma,
            bce_pos_weight=branch_bce_pos_weights["pooled"],
            **branch_focal_params["pooled"],
            sample_weight=sample_weight,
            valid_sample=trial_loss_valid_sample,
        )
        fusion_loss = torch.tensor(0.0, device=device)
        if "fusion_logits" in outputs and float(loss_weights.get("fusion", 0.0)) > 0.0:
            fusion_loss = classification_branch_loss(
                outputs["fusion_logits"],  # [B,K]
                batch["target"],
                fusion_loss_type,
                label_smoothing=effective_smoothing,
                ordinal_sigma=ordinal_sigma,
                bce_pos_weight=branch_bce_pos_weights["fusion"],
                **branch_focal_params["fusion"],
                sample_weight=sample_weight,
                valid_sample=trial_loss_valid_sample,
            )
        # 标量总 loss；权重来自 YAML train.*_loss_weight
        reconstruction_loss = zero_time_loss
        reconstruction_valid_sample = torch.zeros(
            clean_eeg.shape[0],
            dtype=torch.bool,
            device=clean_eeg.device,
        )
        if training and reconstruction_weight > 0.0:
            reconstructed_eeg = outputs.get("reconstructed_eeg")
            if not isinstance(reconstructed_eeg, torch.Tensor):
                raise ValueError(
                    "reconstruction-enabled model must return reconstructed_eeg"
                )
            reconstruction_loss = masked_eeg_reconstruction_loss(
                reconstructed_eeg,
                clean_eeg,
                covered_time_mask,
                sample_weight=sample_weight,
            )
            if covered_time_mask is not None:
                reconstruction_valid_sample = covered_time_mask.any(dim=1)
        loss = (
            float(loss_weights.get("time", 1.0)) * time_loss
            + float(loss_weights.get("pooled", 1.0)) * pooled_loss
            + float(loss_weights.get("fusion", 0.0)) * fusion_loss
            + float(
                loss_weights.get(
                    "invalid_time",
                    loss_weights.get("topk_covered_time", 0.0),
                )
            )
            * invalid_time_loss
            + float(loss_weights.get("attention_mask", 0.0)) * attention_mask_loss
            + float(loss_weights.get("attention_smoothness", 0.0))
            * attention_smoothness_loss
            + float(loss_weights.get("null_attention", 0.0))
            * null_attention_loss
            + float(loss_weights.get("att_time_consistency", 0.0))
            * att_time_consistency_loss
            + float(loss_weights.get("real_noise_margin", 0.0))
            * real_noise_margin_loss
            + reconstruction_weight * reconstruction_loss
        )
        if clean_attention_enabled:
            loss = loss + clean_attention_config["weight"] * clean_attention_loss
        if attention_mass_enabled:
            loss = loss + attention_mass_config["weight"] * attention_mass_loss
    return StepResult(loss, {
        "att_time_consistency_loss": att_time_consistency_loss,
        "attention_mask_loss": attention_mask_loss,
        "attention_mass_loss": attention_mass_loss,
        "attention_smoothness_loss": attention_smoothness_loss,
        "clean_attention_loss": clean_attention_loss,
        "fusion_loss": fusion_loss,
        "invalid_time_loss": invalid_time_loss,
        "null_attention_loss": null_attention_loss,
        "pooled_loss": pooled_loss,
        "real_noise_margin_loss": real_noise_margin_loss,
        "reconstruction_loss": reconstruction_loss,
        "time_loss": time_loss,
    }, outputs, {
        "attention_strength_weights": attention_strength_weights,
        "contaminated_time_mask": contaminated_time_mask,
        "covered_time_mask": covered_time_mask,
        "halo_time_mask": halo_time_mask,
        "invalid_core_time_mask": invalid_core_time_mask,
        "invalid_loss_valid_sample": invalid_loss_valid_sample,
        "near_full_noise_covered_count": near_full_noise_covered_count,
        "near_full_noise_time_count": near_full_noise_time_count,
        "reconstruction_valid_sample": reconstruction_valid_sample,
        "roll_abs_shift_max": roll_abs_shift_max,
        "roll_abs_shift_sum": roll_abs_shift_sum,
        "roll_left_count": roll_left_count,
        "roll_mask": roll_mask,
        "roll_metadata_seen": roll_metadata_seen,
        "roll_right_count": roll_right_count,
        "roll_sample_count": roll_sample_count,
        "selected_time_logits": selected_time_logits,
        "selected_valid": selected_valid,
        "selection_valid_mask": selection_valid_mask,
        "time_loss_valid": time_loss_valid,
        "top_indices": top_indices,
        "trial_loss_valid_sample": trial_loss_valid_sample,
        "clean_attention_totals": clean_attention_totals,
    })
