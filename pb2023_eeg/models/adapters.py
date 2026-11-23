"""models.adapters: migrated Target semantics, one implementation per operation."""
from __future__ import annotations
from collections.abc import Mapping
import torch
from pb2023_eeg.models.target_decoder import TASK_CLASS_COUNTS


def _model_task_class_count(model: torch.nn.Module, task: str) -> int:
    """优先读取模型实例的动态 K；旧测试模型回退到默认任务类别数。"""
    resolver = getattr(model, "class_count", None)
    return (
        int(resolver(task))
        if callable(resolver)
        else int(TASK_CLASS_COUNTS[task])
    )


def _model_task_class_indices(
    model: torch.nn.Module,
    task: str,
) -> tuple[int, ...]:
    """返回内部标签到原始标签的映射，兼容未实现该接口的旧模型。"""
    resolver = getattr(model, "class_indices", None)
    if callable(resolver):
        return tuple(int(index) for index in resolver(task))
    return tuple(range(_model_task_class_count(model, task)))


def boundary_noise_forward_kwargs(
    model: torch.nn.Module, batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Sample one pad for matched forwards; eval uses physical trial identity.

    Call once per training batch and reuse the result for real/null comparisons.
    Evaluation uses (exp, subject_id, trial0), so control input transformations
    and batch ordering do not change a physical trial's boundary context.
    """
    encoder = getattr(model, "encoder", None)
    if getattr(encoder, "boundary_mode", None) != "noise_pad":
        return {}
    trial_ids = None
    if not model.training:
        fields = ("exp", "subject_id", "trial0")
        missing = [name for name in fields if name not in batch]
        if missing:
            raise ValueError(f"noise_pad evaluation requires stable trial metadata: {missing}")
        values = [batch[name] for name in fields]
        if any(not isinstance(value, torch.Tensor) or value.ndim != 1
               or value.shape[0] != batch["eeg"].shape[0] for value in values):
            raise ValueError("noise_pad trial metadata must have shape [batch]")
        integer_types = {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        if any(value.dtype not in integer_types or bool((value < 0).any()) for value in values):
            raise ValueError("noise_pad trial metadata must contain nonnegative integers")
        trial_ids = torch.stack(values, dim=1)
    noise = encoder.sample_boundary_noise(batch["eeg"], trial_ids=trial_ids, training=model.training)
    return {"boundary_noise_eeg": noise}


def boundary_noise_metadata(model: torch.nn.Module) -> dict | None:
    """Describe eval sampling for prediction/export provenance."""
    encoder = getattr(model, "encoder", None)
    if getattr(encoder, "boundary_mode", None) != "noise_pad":
        return None
    return {"distribution": "standard_normal", "space": "normalized_eeg",
            "evaluation_seed": encoder.boundary_noise_seed,
            "identity_fields": ["exp", "subject_id", "trial0"]}


def forward_batch(model, batch, *, task: str, attention_indices=None):
    """Adapt native calls without introducing a parameter-name prefix."""
    from .target import TargetAttentionDecoder
    from .baselines.spatial import SpatialTrialDecoder
    eeg = batch["eeg"]
    subjects = batch.get("subject_id")
    if subjects is not None and (not isinstance(subjects, torch.Tensor) or subjects.shape != (len(eeg),)
                                 or subjects.dtype != torch.long or bool((subjects < 0).any())):
        raise ValueError("subject_id must contain one nonnegative int64 global ID per trial")
    if isinstance(model, TargetAttentionDecoder):
        if model.subject_embedding is not None and subjects is None:
            raise ValueError("This model requires the saved global subject_id namespace")
        return model(eeg, task, attention_indices=attention_indices, subject_id=subjects,
                     **boundary_noise_forward_kwargs(model, batch))
    if isinstance(model, SpatialTrialDecoder):
        trained_task = getattr(model, "_pb_trained_task", None)
        if trained_task is None or task != trained_task:
            raise ValueError(f"Baseline prediction task {task!r} differs from its trained task {trained_task!r}")
        if attention_indices is not None:
            raise ValueError("Spatial baseline does not provide attention outputs")
        if model.encoder.subject_specific_spatial and subjects is None:
            raise ValueError("Subject-conditioned baseline requires subject_id")
        return {"pooled_logits": model(eeg, subjects)}
    raise ValueError(f"No native batch adapter registered for {type(model).__name__}")
