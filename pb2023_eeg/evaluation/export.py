"""evaluation.export: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import torch
from torch.utils.data import DataLoader
from pb2023_eeg.models.target_decoder import TargetAttentionDecoder
from pb2023_eeg.training.common import move_batch, progress_bar, write_rows_csv
from pb2023_eeg.training.metrics import build_time_cell_export_rows, classification_cell_metadata, spearman_rho
from pb2023_eeg.training.export_integrity import verified_export
from pb2023_eeg.analysis.curves import attention_shape_diagnostics
from pb2023_eeg.data.transforms import EVALUATION_INPUT_KINDS, gaussian_like_eeg, time_roll_eeg, time_shuffle_eeg
from pb2023_eeg.models.adapters import _model_task_class_count, _model_task_class_indices, boundary_noise_forward_kwargs
from pb2023_eeg.training.objectives.target import evidence_margin, evidence_margin_over_time, probability_margin, probability_margin_over_time
from pb2023_eeg.training.precision import autocast_context


def _write_subject_metrics(
    path: Path,
    trial_rows: list,
    split: str,
    input_kind: str,
    task: str,
) -> None:
    """把 trial_rows 聚合为逐被试的 pooled/fusion 性能表。

    pitch 等有序任务同时给出 MAE 与 Spearman；fusion 未启用时相应字段留空，
    而不是伪造零值。
    """
    buckets: dict = defaultdict(lambda: {"pooled_correct": [], "targets": [], "pooled_preds": [],
                                          "fusion_correct": [], "fusion_preds": [], "maes": [], "fusion_maes": []})
    for row in trial_rows:
        sid = row["subject_id"]
        buckets[sid]["pooled_correct"].append(int(row["pooled_correct"]))
        buckets[sid]["targets"].append(int(row["target"]))
        buckets[sid]["pooled_preds"].append(int(row["pooled_prediction"]))
        fusion_prediction = row.get("fusion_prediction", "")
        if fusion_prediction != "" and fusion_prediction is not None:
            buckets[sid]["fusion_correct"].append(int(row["fusion_correct"]))
            buckets[sid]["fusion_preds"].append(int(fusion_prediction))
        if isinstance(row.get("pooled_mae"), (int, float)):
            buckets[sid]["maes"].append(float(row["pooled_mae"]))
        if isinstance(row.get("fusion_mae"), (int, float)):
            buckets[sid]["fusion_maes"].append(float(row["fusion_mae"]))
    rows = []
    for sid in sorted(buckets.keys()):
        d = buckets[sid]
        n = len(d["pooled_correct"])
        rows.append({
            "split": split,
            "input_kind": input_kind,
            "task": task,
            "subject_id": sid,
            "n_trials": n,
            "pooled_accuracy": sum(d["pooled_correct"]) / max(1, n),
            "pooled_mae_mean": sum(d["maes"]) / max(1, len(d["maes"])) if d["maes"] else "",
            "pooled_spearman": spearman_rho(d["pooled_preds"], d["targets"]) if len(d["targets"]) > 2 else "",
            "fusion_accuracy": sum(d["fusion_correct"]) / max(1, len(d["fusion_correct"])) if d["fusion_correct"] else "",
            "fusion_mae_mean": sum(d["fusion_maes"]) / max(1, len(d["fusion_maes"])) if d["fusion_maes"] else "",
            "fusion_spearman": spearman_rho(d["fusion_preds"], d["targets"]) if len(d["fusion_preds"]) > 2 else "",
        })
    write_rows_csv(path, rows)


@torch.no_grad()
@verified_export("target")
def export_target_outputs(
    model: TargetAttentionDecoder,
    loader: DataLoader,
    task: str,
    device: torch.device,
    amp: bool,
    output_dir: Path,
    split: str,
    attention_indices: torch.Tensor,
    time_s: Sequence[float],
    progress_bar_enabled: bool = True,
    input_kind: str = "real",
    gaussian_seed: int | None = None,
    gaussian_noise_mode: str = "sample_channel",
    time_shuffle_seed: int | None = None,
    time_roll_shift_samples: int | None = None,
    activation_threshold: float = 0.5,
    time_score_mode: str = "sigmoid",
    score_mode: str | None = None,
    pooled_score_mode: str = "softmax",
    fusion_score_mode: str | None = None,
    min_cell_class_count: int = 1,
) -> None:
    """导出真实、噪声、时间置乱或固定 circular-roll 输入的完整性能表。

    time logits 可为 K 类或启用 no-class 后的 K+1 类；pooled/fusion 必须始终
    为原任务 K 类。time 预测为 no-class 时，真实标签仍是 0..K-1，因此性能
    correct_count 不增加、分母不变，直接计为错误。函数还导出 K x Kt 混淆矩阵、
    每类 precision/recall/F1、真实类别 activation 及独立 no-class 分数。
    """
    if input_kind not in EVALUATION_INPUT_KINDS:
        raise ValueError(
            f"input_kind must be one of {EVALUATION_INPUT_KINDS}, got {input_kind!r}"
        )
    if input_kind == "time_shuffle" and time_shuffle_seed is None:
        raise ValueError("time_shuffle input requires time_shuffle_seed")
    if input_kind == "time_roll" and time_roll_shift_samples is None:
        raise ValueError("time_roll input requires time_roll_shift_samples")
    min_cell_class_count = int(min_cell_class_count)
    if min_cell_class_count < 1:
        raise ValueError("min_cell_class_count must be positive")
    if score_mode is not None:
        pooled_score_mode = score_mode
        if fusion_score_mode is None:
            fusion_score_mode = score_mode
    if fusion_score_mode is None:
        fusion_score_mode = pooled_score_mode
    normalized_time_score_mode = str(time_score_mode).strip().lower()
    if normalized_time_score_mode not in {"sigmoid", "softmax"}:
        raise ValueError("time_score_mode must be sigmoid or softmax")
    target_class_count = _model_task_class_count(model, task)
    class_index_to_original = _model_task_class_indices(model, task)
    if len(class_index_to_original) != target_class_count:
        raise ValueError("Model class index mapping length does not match class count")
    time_class_count_fn = getattr(model, "time_class_count", None)
    time_output_class_count = (
        int(time_class_count_fn(task))
        if callable(time_class_count_fn)
        else target_class_count
    )
    no_class_index_fn = getattr(model, "time_no_class_index", None)
    time_no_class_index = (
        no_class_index_fn(task)
        if callable(no_class_index_fn)
        else None
    )
    if time_no_class_index is not None:
        time_no_class_index = int(time_no_class_index)
        if (
            time_no_class_index != target_class_count
            or time_output_class_count != target_class_count + 1
        ):
            raise ValueError(
                "time no-class output must append exactly one class after the task classes"
            )
    elif time_output_class_count != target_class_count:
        raise ValueError(
            "time_class_count must equal the task class count when no-class is disabled"
        )
    fusion_output_class_count = (
        target_class_count
        if bool(getattr(model, "fusion_enabled", False))
        else 0
    )

    gate_metadata: dict[str, object] = {}
    gate_activation = getattr(model, "attention_gate_activation", "sigmoid")
    gate_combination = getattr(model, "attention_gate_combination", "multiply")
    effective_normalization = getattr(model, "attention_effective_normalization", "none")
    has_signed_effective = gate_activation == "tanh" and effective_normalization != "softmax"
    if gate_activation != "sigmoid" or gate_combination != "multiply":
        gate_metadata = {
            "attention_gate_activation": gate_activation,
            "attention_gate_combination": gate_combination,
            "attention_gate_combination_stage": "post_softmax",
            "attention_mass_basis": "absolute" if has_signed_effective else "nonnegative",
            "normalized_attention_basis": "absolute" if has_signed_effective else "nonnegative",
        }
    if effective_normalization != "none":
        gate_metadata["attention_effective_normalization"] = effective_normalization

    def schema_metadata() -> dict[str, object]:
        """返回写入每张 CSV 的类别布局与 schema 版本字段。"""
        return {
            "export_schema_version": 2,
            "target_class_count": target_class_count,
            "time_output_class_count": time_output_class_count,
            "time_no_class_enabled": int(time_no_class_index is not None),
            "time_no_class_index": (
                time_no_class_index if time_no_class_index is not None else ""
            ),
            "pooled_output_class_count": target_class_count,
            "fusion_output_class_count": fusion_output_class_count,
            "class_index_to_original": ",".join(
                str(index) for index in class_index_to_original
            ),
            **gate_metadata,
        }

    model.eval()
    trial_rows: list[dict[str, object]] = []
    attention_rows: list[dict[str, object]] = []
    time_correct: dict[tuple[int, int], int] = defaultdict(int)
    time_total: dict[tuple[int, int], int] = defaultdict(int)
    time_evidence_sum: dict[tuple[int, int], float] = defaultdict(float)
    time_probability_sum: dict[tuple[int, int], float] = defaultdict(float)
    activation_score_sum: dict[tuple[int, int], float] = defaultdict(float)
    activation_active_count: dict[tuple[int, int], int] = defaultdict(int)
    activation_total: dict[tuple[int, int], int] = defaultdict(int)
    no_class_prediction_count: dict[tuple[int, int], int] = defaultdict(int)
    no_class_score_sum: dict[tuple[int, int], float] = defaultdict(float)
    time_confusion_count: dict[tuple[int, int, int, int], int] = defaultdict(int)
    time_cell_confusion_count: dict[
        tuple[int, int, int, int, int], int
    ] = defaultdict(int)
    cell_metadata: dict[tuple[int, int], dict[str, int]] = {}
    choice_history_fields = tuple(
        getattr(getattr(loader, "dataset", None), "choice_cell_history_fields", ())
    )
    all_times = [float(value) for value in time_s]
    selected_times = [float(time_s[int(index)]) for index in attention_indices.detach().cpu().tolist()]
    generator = None
    if input_kind == "gaussian":
        generator = torch.Generator(device=device)
        if gaussian_seed is not None:
            generator.manual_seed(int(gaussian_seed))
    progress = progress_bar(
        loader,
        desc=f"target {task} export {split} {input_kind}",
        enabled=progress_bar_enabled,
        total=len(loader),
    )
    for raw_batch in progress:
        batch = move_batch(raw_batch, device)
        eeg = batch["eeg"]
        if input_kind == "gaussian":
            eeg = gaussian_like_eeg(
                eeg,
                generator=generator,
                noise_mode=gaussian_noise_mode,
            )
        elif input_kind == "time_shuffle":
            assert time_shuffle_seed is not None
            eeg = time_shuffle_eeg(
                eeg,
                seed=time_shuffle_seed,
                record_id=batch["record_id"],
                trial0=batch["trial0"],
                lag=batch["lag"],
            )
        elif input_kind == "time_roll":
            assert time_roll_shift_samples is not None
            eeg = time_roll_eeg(eeg, shift_samples=time_roll_shift_samples)
        with autocast_context(device, enabled=amp):
            outputs = model(
                eeg,
                task=task,
                attention_indices=attention_indices,
                subject_id=batch.get("subject_id"),
                **boundary_noise_forward_kwargs(model, batch),
            )
        target = batch["target"]
        pooled_logits = outputs["pooled_logits"].float()
        if (
            pooled_logits.ndim != 2
            or int(pooled_logits.shape[-1]) != target_class_count
        ):
            raise ValueError(
                "pooled_logits must have the original task class count: "
                f"got {pooled_logits.shape}, expected {target_class_count} classes"
            )
        pooled_probability = probability_margin(pooled_logits, target, mode=pooled_score_mode)
        pooled_evidence = evidence_margin(pooled_logits, target)
        fusion_logits = outputs.get("fusion_logits")
        if fusion_logits is not None:
            fusion_logits = fusion_logits.float()
            if (
                fusion_logits.ndim != 2
                or int(fusion_logits.shape[-1]) != target_class_count
            ):
                raise ValueError(
                    "fusion_logits must have the original task class count: "
                    f"got {fusion_logits.shape}, expected {target_class_count} classes"
                )
            fusion_output_class_count = target_class_count
            fusion_probability = probability_margin(fusion_logits, target, mode=fusion_score_mode)
            fusion_evidence = evidence_margin(fusion_logits, target)
            fusion_pred = fusion_logits.argmax(dim=-1)
            has_fusion_branch = True
        else:
            # Fusion 关闭时不写假的 0 准确率：下游应把空字符串当成“未启用”。
            fusion_probability = None
            fusion_evidence = None
            fusion_pred = None
            has_fusion_branch = False
        time_logits = outputs["time_logits"].float()
        if time_logits.ndim != 3 or int(time_logits.shape[-1]) != time_output_class_count:
            raise ValueError(
                "time_logits class dimension does not match model.time_class_count: "
                f"got {time_logits.shape}, expected {time_output_class_count} classes"
            )
        if time_logits.shape[1] != len(all_times):
            raise ValueError(
                f"time_s length ({len(all_times)}) does not match time_logits ({time_logits.shape[1]})"
            )
        time_prediction = time_logits.argmax(dim=-1)
        time_prediction_cpu = time_prediction.detach().cpu()
        time_correct_batch = (time_prediction == target[:, None]).detach().cpu()
        time_evidence_batch = evidence_margin_over_time(
            time_logits,
            target,
            task_class_count=target_class_count,
        ).detach().cpu()
        time_probability_batch = probability_margin_over_time(
            time_logits,
            target,
            mode=normalized_time_score_mode,
            task_class_count=target_class_count,
        ).detach().cpu()
        if time_no_class_index is None:
            time_no_class_prediction = torch.zeros_like(
                time_prediction,
                dtype=torch.bool,
            )
        else:
            time_no_class_prediction = time_prediction == time_no_class_index
        time_no_class_prediction = time_no_class_prediction.detach().cpu()
        time_probability = (
            torch.sigmoid(time_logits)
            if normalized_time_score_mode == "sigmoid"
            else torch.softmax(time_logits, dim=-1)
        )
        time_activation_score = (
            time_probability[..., :target_class_count].max(dim=-1).values.detach().cpu()
        )
        time_no_class_score = (
            time_probability[..., time_no_class_index].detach().cpu()
            if time_no_class_index is not None
            else torch.zeros_like(time_activation_score)
        )
        time_activation_active = time_activation_score >= float(activation_threshold)
        weights = outputs["attention_weights"].float()
        effective_weights = outputs.get("effective_attention_weights")
        if effective_weights is not None:
            effective_weights = effective_weights.float()
        else:
            effective_weights = weights
        signed_attention = "attention_magnitude_weights" in outputs
        strength_weights = outputs.get("attention_magnitude_weights", effective_weights).float()
        attention_scores = outputs["attention_scores"].float()
        time_selection_weights = outputs.get(
            "time_selection_attention_weights", effective_weights
        ).float()
        attention_valid_mask = outputs.get("attention_valid_mask")
        if attention_valid_mask is None:
            attention_valid = torch.ones_like(weights, dtype=torch.bool)
        elif (
            attention_valid_mask.ndim == 1
            and attention_valid_mask.shape[0] == weights.shape[1]
        ):
            attention_valid = attention_valid_mask.to(
                device=weights.device, dtype=torch.bool
            )[None, :].expand_as(weights)
        elif attention_valid_mask.shape == weights.shape:
            attention_valid = attention_valid_mask.to(
                device=weights.device, dtype=torch.bool
            )
        else:
            raise ValueError(
                "attention_valid_mask must align with exported attention weights"
            )
        attention_gate = outputs.get("attention_gate")
        effective_attention_mass = strength_weights.sum(dim=-1)
        pre_normalization_weights = outputs.get("pre_normalization_effective_attention_weights")
        if pre_normalization_weights is not None:
            pre_normalization_weights = pre_normalization_weights.float()
        if attention_gate is not None:
            attention_gate = attention_gate.float()
            attention_gate_mean = attention_gate.mean(dim=-1)
            attention_gate_min = attention_gate.min(dim=-1).values
            attention_gate_max = attention_gate.max(dim=-1).values
        else:
            attention_gate_mean = torch.zeros(target.shape[0], device=device)
            attention_gate_min = torch.zeros(target.shape[0], device=device)
            attention_gate_max = torch.zeros(target.shape[0], device=device)
        normalized_effective = strength_weights / effective_attention_mass[:, None].clamp_min(1e-12)
        attention_diagnostics = attention_shape_diagnostics(
            normalized_effective,
            selected_times,
            attention_valid,
        )
        subject_ids = batch["subject_id"].detach().cpu().tolist()
        cell_subject_tensor, cell_tensor = classification_cell_metadata(batch, task)
        cell_subject_ids = (
            cell_subject_tensor.detach().cpu().tolist()
            if cell_subject_tensor is not None and cell_tensor is not None
            else None
        )
        cell_ids = (
            cell_tensor.detach().cpu().tolist()
            if cell_subject_tensor is not None and cell_tensor is not None
            else None
        )
        for row_index in range(target.shape[0]):
            subject_id = int(subject_ids[row_index])
            target_index = int(target[row_index].item())
            signed_mass_metadata = (
                {
                    "signed_effective_attention_mass": float(effective_weights[row_index].sum().item()),
                    "absolute_effective_attention_mass": float(effective_attention_mass[row_index].item()),
                }
                if signed_attention else {}
            )
            normalization_metadata = (
                {"pre_normalization_effective_attention_mass": float(pre_normalization_weights[row_index].sum().item())}
                if pre_normalization_weights is not None else {}
            )
            cell_id: int | None = None
            if cell_ids is not None and cell_subject_ids is not None:
                if int(cell_subject_ids[row_index]) != subject_id:
                    raise ValueError(
                        "classification cell subject does not match exported subject_id"
                    )
                cell_id = int(cell_ids[row_index])
                if task == "pitch":
                    lag_tensor = batch.get("lag")
                    lag_value = (
                        int(lag_tensor[row_index].item())
                        if lag_tensor is not None
                        else 0
                    )
                    row_cell_metadata = (
                        {
                            "current_stimulus": int(
                                batch["current_stimulus"][row_index].item()
                            )
                        }
                        if lag_value > 0
                        else {}
                    )
                else:
                    row_cell_metadata = {
                        "current_stimulus": int(
                            batch["current_stimulus"][row_index].item()
                        ),
                        **{
                            field: int(batch[field][row_index].item())
                            for field in choice_history_fields
                        },
                    }
                metadata_key = (subject_id, cell_id)
                previous_metadata = cell_metadata.setdefault(
                    metadata_key, row_cell_metadata
                )
                if previous_metadata != row_cell_metadata:
                    raise ValueError(
                        "classification cell id maps to inconsistent nuisance metadata"
                    )
            has_attention_gate = attention_gate is not None
            trial_no_class_count = int(
                time_no_class_prediction[row_index].sum().item()
            )
            trial_rows.append(
                {
                    "split": split,
                    "input_kind": input_kind,
                    "task": task,
                    "class_count": target_class_count,
                    **schema_metadata(),
                    **normalization_metadata,
                    "no_class_prediction_count": trial_no_class_count,
                    "no_class_prediction_rate": (
                        trial_no_class_count / max(1, len(all_times))
                    ),
                    "record_id": int(batch["record_id"][row_index].item()),
                    "subject_id": subject_id,
                    "exp": int(batch["exp"][row_index].item()),
                    "trial0": int(batch["trial0"][row_index].item()),
                    "lag": int(batch["lag"][row_index].item()),
                    "current_stimulus": int(batch["current_stimulus"][row_index].item()),
                    "target": int(target[row_index].item()),
                    "pooled_prediction": int(pooled_logits[row_index].argmax().item()),
                    "pooled_correct": int(pooled_logits[row_index].argmax().item() == target[row_index].item()),
                    "pooled_evidence_margin": float(pooled_evidence[row_index].item()),
                    "pooled_probability_margin": float(pooled_probability[row_index].item()),
                    "fusion_prediction": (
                        int(fusion_pred[row_index].item()) if has_fusion_branch else ""
                    ),
                    "fusion_correct": (
                        int(
                            fusion_pred[row_index].item() == target[row_index].item()
                        )
                        if has_fusion_branch
                        else ""
                    ),
                    "fusion_evidence_margin": (
                        float(fusion_evidence[row_index].item())
                        if has_fusion_branch
                        else ""
                    ),
                    "fusion_probability_margin": (
                        float(fusion_probability[row_index].item())
                        if has_fusion_branch
                        else ""
                    ),
                    "attention_gate_mean": (
                        float(attention_gate_mean[row_index].item()) if has_attention_gate else ""
                    ),
                    "attention_gate_min": (
                        float(attention_gate_min[row_index].item()) if has_attention_gate else ""
                    ),
                    "attention_gate_max": (
                        float(attention_gate_max[row_index].item()) if has_attention_gate else ""
                    ),
                    "effective_attention_mass": float(effective_attention_mass[row_index].item()),
                    **signed_mass_metadata,
                    "attention_pool_valid_count": int(
                        attention_valid[row_index].sum().item()
                    ),
                    "attention_pool_valid_fraction": float(
                        attention_valid[row_index].float().mean().item()
                    ),
                    **attention_diagnostics[row_index],
                    "pooled_mae": abs(int(pooled_logits[row_index].argmax().item()) - int(target[row_index].item())),
                    "fusion_mae": (
                        abs(
                            int(fusion_pred[row_index].item())
                            - int(target[row_index].item())
                        )
                        if has_fusion_branch
                        else ""
                    ),
                }
            )
            for time_offset, time_value in enumerate(selected_times):
                selected_time_index = int(attention_indices[time_offset].item())
                selected_no_class = int(
                    time_no_class_prediction[row_index, selected_time_index].item()
                )
                attention_rows.append(
                    {
                        "split": split,
                        "input_kind": input_kind,
                        "record_id": int(batch["record_id"][row_index].item()),
                        "subject_id": subject_id,
                        "trial0": int(batch["trial0"][row_index].item()),
                        "lag": int(batch["lag"][row_index].item()),
                        "target": int(target[row_index].item()),
                        **schema_metadata(),
                        "time_index": selected_time_index,
                        "time_s": time_value,
                        "no_class_prediction_count": selected_no_class,
                        "no_class_prediction_rate": float(selected_no_class),
                        "attention_weight": float(weights[row_index, time_offset].item()),
                        "attention_score": float(attention_scores[row_index, time_offset].item()),
                        "attention_pool_valid": int(
                            attention_valid[row_index, time_offset].item()
                        ),
                        "attention_gate": (
                            float(attention_gate[row_index, time_offset].item()) if has_attention_gate else ""
                        ),
                        "effective_attention_weight": float(
                            effective_weights[row_index, time_offset].item()
                        ),
                        **normalization_metadata,
                        **({"pre_normalization_effective_attention_weight": float(
                            pre_normalization_weights[row_index, time_offset].item())}
                            if pre_normalization_weights is not None else {}),
                        **(
                            {
                                "absolute_effective_attention_weight": float(strength_weights[row_index, time_offset].item()),
                                "normalized_signed_effective_attention_weight": float(
                                    (effective_weights[row_index, time_offset]
                                     / effective_attention_mass[row_index].clamp_min(1e-12)).item()
                                ),
                                **signed_mass_metadata,
                            }
                            if signed_attention else {}
                        ),
                        "time_selection_attention_weight": float(
                            time_selection_weights[row_index, time_offset].item()
                        ),
                        "normalized_effective_attention_weight": float(
                            normalized_effective[row_index, time_offset].item()
                        ),
                        "effective_attention_mass": float(effective_attention_mass[row_index].item()),
                    }
                )
            for time_index in range(len(all_times)):
                key = (subject_id, time_index)
                prediction_index = int(
                    time_prediction_cpu[row_index, time_index].item()
                )
                time_correct[key] += int(time_correct_batch[row_index, time_index].item())
                time_total[key] += 1
                time_evidence_sum[key] += float(
                    time_evidence_batch[row_index, time_index].item()
                )
                time_probability_sum[key] += float(
                    time_probability_batch[row_index, time_index].item()
                )
                time_confusion_count[
                    (subject_id, time_index, target_index, prediction_index)
                ] += 1
                if cell_id is not None:
                    time_cell_confusion_count[
                        (
                            subject_id,
                            cell_id,
                            time_index,
                            target_index,
                            prediction_index,
                        )
                    ] += 1
                activation_score_sum[key] += float(time_activation_score[row_index, time_index].item())
                activation_active_count[key] += int(time_activation_active[row_index, time_index].item())
                activation_total[key] += 1
                no_class_prediction_count[key] += int(
                    time_no_class_prediction[row_index, time_index].item()
                )
                no_class_score_sum[key] += float(
                    time_no_class_score[row_index, time_index].item()
                )
    # time_correct 只比较真实任务标签；no-class 不会命中 target，天然计为错误。
    # evidence/probability margin 仅基于任务 K 类 logits，不把 no-class 计入 other。
    time_rows = [
        {
            "split": split,
            "input_kind": input_kind,
            "task": task,
            "class_count": target_class_count,
            **schema_metadata(),
            "subject_id": subject_id,
            "time_index": time_index,
            "time_s": all_times[time_index],
            "correct_count": correct,
            "total_count": time_total[(subject_id, time_index)],
            "accuracy": correct / max(1, time_total[(subject_id, time_index)]),
            "evidence_margin_sum": time_evidence_sum[(subject_id, time_index)],
            "evidence_margin_mean": (
                time_evidence_sum[(subject_id, time_index)]
                / max(1, time_total[(subject_id, time_index)])
            ),
            "probability_margin_sum": time_probability_sum[(subject_id, time_index)],
            "probability_margin_mean": (
                time_probability_sum[(subject_id, time_index)]
                / max(1, time_total[(subject_id, time_index)])
            ),
            "time_score_mode": normalized_time_score_mode,
            "no_class_prediction_count": no_class_prediction_count[
                (subject_id, time_index)
            ],
            "no_class_prediction_rate": (
                no_class_prediction_count[(subject_id, time_index)]
                / max(1, time_total[(subject_id, time_index)])
            ),
        }
        for (subject_id, time_index), correct in sorted(time_correct.items())
    ]
    activation_rows = [
        {
            "split": split,
            "input_kind": input_kind,
            "task": task,
            "class_count": target_class_count,
            **schema_metadata(),
            "subject_id": subject_id,
            "time_index": time_index,
            "time_s": all_times[time_index],
            "time_score_mode": normalized_time_score_mode,
            "activation_definition": "max_real_class_score",
            "activation_threshold": float(activation_threshold),
            "activation_score_sum": activation_score_sum[(subject_id, time_index)],
            "active_count": activation_active_count[(subject_id, time_index)],
            "total_count": total,
            "activation_score_mean": activation_score_sum[(subject_id, time_index)] / max(1, total),
            "activation_rate": activation_active_count[(subject_id, time_index)] / max(1, total),
            "no_class_prediction_count": no_class_prediction_count[
                (subject_id, time_index)
            ],
            "no_class_prediction_rate": (
                no_class_prediction_count[(subject_id, time_index)] / max(1, total)
            ),
            "no_class_score_mean": (
                no_class_score_sum[(subject_id, time_index)] / max(1, total)
            ),
        }
        for (subject_id, time_index), total in sorted(activation_total.items())
    ]
    time_confusion_rows: list[dict[str, object]] = []
    time_class_metric_rows: list[dict[str, object]] = []
    # 为每个被试/时间点构造完整 K x Kt 网格，即使某个 cell 计数为 0 也写出。
    for (subject_id, time_index), total in sorted(time_total.items()):
        cell_count = {
            (target_index, prediction_index): time_confusion_count.get(
                (subject_id, time_index, target_index, prediction_index),
                0,
            )
            for target_index in range(target_class_count)
            for prediction_index in range(time_output_class_count)
        }
        target_support = {
            target_index: sum(
                cell_count[(target_index, prediction_index)]
                for prediction_index in range(time_output_class_count)
            )
            for target_index in range(target_class_count)
        }
        prediction_count = {
            prediction_index: sum(
                cell_count[(target_index, prediction_index)]
                for target_index in range(target_class_count)
            )
            for prediction_index in range(time_output_class_count)
        }
        confusion_total = sum(cell_count.values())
        correct_count = sum(
            cell_count[(class_index, class_index)]
            for class_index in range(target_class_count)
        )
        expected_no_class_count = no_class_prediction_count[
            (subject_id, time_index)
        ]
        actual_no_class_count = (
            prediction_count[time_no_class_index]
            if time_no_class_index is not None
            else 0
        )
        # 写盘前交叉校验三套独立累计量，防止性能曲线与混淆矩阵静默分叉。
        if confusion_total != total or correct_count != time_correct[(subject_id, time_index)]:
            raise RuntimeError(
                "time confusion counts do not match exported performance totals"
            )
        if actual_no_class_count != expected_no_class_count:
            raise RuntimeError(
                "time confusion no-class column does not match no-class predictions"
            )
        common_metrics = {
            "split": split,
            "input_kind": input_kind,
            "task": task,
            "class_count": target_class_count,
            **schema_metadata(),
            "subject_id": subject_id,
            "time_index": time_index,
            "time_s": all_times[time_index],
            "performance_correct_count": correct_count,
            "performance_total_count": total,
            "performance_accuracy": correct_count / max(1, total),
            "no_class_prediction_count": expected_no_class_count,
            "no_class_prediction_rate": expected_no_class_count / max(1, total),
        }
        for target_index in range(target_class_count):
            for prediction_index in range(time_output_class_count):
                time_confusion_rows.append(
                    {
                        **common_metrics,
                        "target_class_index": target_index,
                        "prediction_class_index": prediction_index,
                        "prediction_is_no_class": int(
                            time_no_class_index is not None
                            and prediction_index == time_no_class_index
                        ),
                        "target_support_count": target_support[target_index],
                        "prediction_count": prediction_count[prediction_index],
                        "count": cell_count[(target_index, prediction_index)],
                    }
                )
        for class_index in range(time_output_class_count):
            is_no_class = (
                time_no_class_index is not None
                and class_index == time_no_class_index
            )
            class_prediction_count = prediction_count[class_index]
            if is_no_class:
                support = 0
                true_positive: int | str = ""
                recall: float | str = ""
                precision: float | str = ""
                f1: float | str = ""
                class_role = "no_class_prediction"
            else:
                support = target_support[class_index]
                true_positive = cell_count[(class_index, class_index)]
                recall = (
                    true_positive / support
                    if support > 0
                    else ""
                )
                precision = (
                    true_positive / class_prediction_count
                    if class_prediction_count > 0
                    else ""
                )
                f1 = (
                    2.0 * recall * precision / (recall + precision)
                    if isinstance(recall, float)
                    and isinstance(precision, float)
                    and recall + precision > 0.0
                    else ""
                )
                class_role = "target_class"
            time_class_metric_rows.append(
                {
                    **common_metrics,
                    "class_index": class_index,
                    "class_role": class_role,
                    "is_no_class": int(is_no_class),
                    "target_support_count": support,
                    "prediction_count": class_prediction_count,
                    "prediction_rate": class_prediction_count / max(1, total),
                    "true_positive_count": true_positive,
                    "recall_denominator_count": support,
                    "recall": recall,
                    "precision": precision,
                    "f1": f1,
                }
            )
    supervised_time_indices = {
        int(index) for index in attention_indices.detach().cpu().tolist()
    }
    cell_metric_rows, cell_summary_rows = build_time_cell_export_rows(
        time_cell_confusion_count,
        class_count=target_class_count,
        prediction_class_count=time_output_class_count,
        time_s=all_times,
        row_metadata={
            "split": split,
            "input_kind": input_kind,
            "task": task,
            "class_count": target_class_count,
            **schema_metadata(),
        },
        cell_metadata=cell_metadata,
        cell_metadata_fields=choice_history_fields,
        min_cell_class_count=min_cell_class_count,
        time_metadata=lambda time_index, _time_value: {
            "is_supervised_time": int(time_index in supervised_time_indices)
        },
    )
    prefix = f"{split}_" if input_kind == "real" else f"{split}_{input_kind}_"
    write_rows_csv(output_dir / f"{prefix}trial_metrics.csv", trial_rows)
    write_rows_csv(output_dir / f"{prefix}attention_weights.csv", attention_rows)
    write_rows_csv(output_dir / f"{prefix}time_accuracy.csv", time_rows)
    write_rows_csv(output_dir / f"{prefix}time_activation.csv", activation_rows)
    write_rows_csv(output_dir / f"{prefix}time_confusion.csv", time_confusion_rows)
    write_rows_csv(
        output_dir / f"{prefix}time_class_metrics.csv",
        time_class_metric_rows,
    )
    write_rows_csv(output_dir / f"{prefix}time_cell_metrics.csv", cell_metric_rows)
    write_rows_csv(output_dir / f"{prefix}time_cell_summary.csv", cell_summary_rows)
    # 逐被试汇总 accuracy、MAE，以及有序任务的 Spearman rho。
    if trial_rows:
        _write_subject_metrics(output_dir / f"{prefix}subject_metrics.csv", trial_rows, split, input_kind, task)
