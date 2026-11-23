"""Read historical checkpoint schemas without guessing forward semantics."""
from __future__ import annotations
from typing import Mapping
import torch
from ..models.factory import resolve_model_task_class_indices

def _checkpoint_time_class_count(
    model_state: Mapping[str, object],
    *,
    task: str,
    time_head_mode: str,
) -> int:
    """从当前或历史检查点 schema 读取一个 time head 的输出类别数。

    参数：
        model_state: 检查点中的扁平参数映射。
        task: ``pitch``、``choice`` 或 ``motor``。
        time_head_mode: ``shared`` 或独立时间头模式。

    返回：
        分类器输出轴宽度；无法识别或多个候选宽度不一致时抛出异常。
    """
    prefix = f"class_heads.{task}."
    # 历史 shared head 曾直接用 Linear，也曾包在 net 中；独立头将时间轴放在首维。
    if time_head_mode == "shared":
        output_tensors = (
            ("weight", 0),
            ("bias", 0),
            ("net.weight", 0),
            ("net.bias", 0),
            ("net.3.weight", 0),
            ("net.3.bias", 0),
        )
    else:
        output_tensors = (
            ("weight", -1),
            ("bias", -1),
            ("output_weight", -1),
            ("output_bias", -1),
        )
    counts: set[int] = set()
    for suffix, axis in output_tensors:
        value = model_state.get(f"{prefix}{suffix}")
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim == 0:
            raise ValueError(
                f"Checkpoint time head tensor {prefix}{suffix} has an invalid shape"
            )
        counts.add(int(value.shape[axis]))
    if not counts:
        raise ValueError(
            f"Checkpoint is missing a recognizable time classifier for task {task!r}"
        )
    if len(counts) > 1:
        raise ValueError(
            f"Checkpoint time classifier for task {task!r} has inconsistent output dimensions"
        )
    return next(iter(counts))

def resolve_model_config_for_checkpoint(
    model_config: Mapping[str, object],
    model_state: Mapping[str, object],
) -> dict[str, object]:
    """用检查点参数形状补全并校验模型配置。

    该兼容层推断 time-level 是否包含额外 ``no_class``、分类器是一层还是两层，以及
    MLP 隐藏维度。快照已显式配置但与检查点冲突时拒绝加载，避免静默错配。

    返回：
        可直接交给 ``build_model`` 的新配置字典，不修改传入映射。
    """
    resolved = dict(model_config)
    task_class_indices = resolve_model_task_class_indices(resolved)
    resolved["resolved_task_class_indices"] = {
        task: list(indices) for task, indices in task_class_indices.items()
    }
    task_class_counts = {
        task: len(indices) for task, indices in task_class_indices.items()
    }
    mode = str(resolved.get("time_head_mode", "shared")).lower().replace("-", "_")
    inferred_no_class_values: set[bool] = set()
    # 三个任务必须对 no_class 开关给出一致证据，才能确定共享模型的完整 schema。
    for task, ordinary_class_count in task_class_counts.items():
        checkpoint_class_count = _checkpoint_time_class_count(
            model_state,
            task=task,
            time_head_mode=mode,
        )
        if checkpoint_class_count == ordinary_class_count:
            inferred_no_class_values.add(False)
        elif checkpoint_class_count == ordinary_class_count + 1:
            inferred_no_class_values.add(True)
        else:
            raise ValueError(
                f"Checkpoint time classifier for task {task!r} has "
                f"{checkpoint_class_count} classes; expected {ordinary_class_count} "
                f"or {ordinary_class_count + 1}"
            )
    if len(inferred_no_class_values) != 1:
        raise ValueError(
            "Checkpoint time classifiers disagree on whether the no-class output is enabled"
        )
    inferred_no_class = next(iter(inferred_no_class_values))
    if "time_no_class_enabled" in resolved:
        configured_no_class = resolved["time_no_class_enabled"]
        if not isinstance(configured_no_class, bool):
            raise ValueError("model.time_no_class_enabled must be a boolean")
        if configured_no_class != inferred_no_class:
            raise ValueError(
                "model.time_no_class_enabled conflicts with the checkpoint "
                f"time classifier schema ({configured_no_class} != {inferred_no_class})"
            )
    else:
        resolved["time_no_class_enabled"] = inferred_no_class
    reconstruction_keys = {
        str(key)
        for key in model_state
        if str(key).startswith("reconstruction_head.")
    }
    expected_reconstruction_keys = {
        "reconstruction_head.weight",
        "reconstruction_head.bias",
    }
    if reconstruction_keys and reconstruction_keys != expected_reconstruction_keys:
        raise ValueError(
            "Checkpoint has an incomplete or unsupported reconstruction head schema"
        )
    inferred_reconstruction = bool(reconstruction_keys)
    configured_reconstruction = resolved.get("reconstruction")
    if configured_reconstruction is None:
        configured_reconstruction = {}
    if not isinstance(configured_reconstruction, Mapping):
        raise ValueError("model.reconstruction must be a mapping when provided")
    configured_reconstruction = dict(configured_reconstruction)
    if "enabled" in configured_reconstruction:
        configured_enabled = configured_reconstruction["enabled"]
        if not isinstance(configured_enabled, bool):
            raise ValueError("model.reconstruction.enabled must be a boolean")
        if configured_enabled != inferred_reconstruction:
            raise ValueError(
                "model.reconstruction.enabled conflicts with the checkpoint schema "
                f"({configured_enabled} != {inferred_reconstruction})"
            )
    else:
        configured_reconstruction["enabled"] = inferred_reconstruction
    resolved["reconstruction"] = configured_reconstruction
    # 独立时间头模式下，trial-level 分类器位于 pooled_heads，而非 class_heads。
    head_group = "class_heads" if mode == "shared" else "pooled_heads"
    inferred_layers: set[int] = set()
    inferred_hidden_dims: set[int] = set()
    for task in ("pitch", "choice", "motor"):
        direct = model_state.get(f"{head_group}.{task}.weight")
        current_linear = model_state.get(f"{head_group}.{task}.net.weight")
        first_mlp = model_state.get(f"{head_group}.{task}.net.0.weight")
        final_mlp = model_state.get(f"{head_group}.{task}.net.3.weight")
        if isinstance(direct, torch.Tensor) or isinstance(current_linear, torch.Tensor):
            inferred_layers.add(1)
        if isinstance(first_mlp, torch.Tensor) or isinstance(final_mlp, torch.Tensor):
            inferred_layers.add(2)
            if isinstance(first_mlp, torch.Tensor) and first_mlp.ndim == 2:
                inferred_hidden_dims.add(int(first_mlp.shape[0]))
    if len(inferred_layers) > 1:
        raise ValueError(
            f"Checkpoint has inconsistent {head_group} classifier schemas"
        )
    if inferred_layers:
        inferred = next(iter(inferred_layers))
        configured = resolved.get("classifier_num_layers")
        if configured is None:
            resolved["classifier_num_layers"] = inferred
        elif int(configured) != inferred:
            raise ValueError(
                f"model.classifier_num_layers={int(configured)} conflicts with "
                f"the checkpoint {head_group} schema ({inferred} layer)"
            )
    if len(inferred_hidden_dims) > 1:
        raise ValueError(
            f"Checkpoint has inconsistent {head_group} classifier hidden dimensions"
        )
    if inferred_hidden_dims and resolved.get("classifier_hidden_dim") is None:
        resolved["classifier_hidden_dim"] = next(iter(inferred_hidden_dims))
    return resolved

def resolve_checkpoint_n_subjects(
    dataset_n_subjects: int,
    model_state: Mapping[str, object],
) -> int:
    """确定重导出时的被试 embedding 容量。

    只导出部分 split 时数据中最大被试索引可能小于训练期容量，因此优先保留检查点
    ``subject_embedding.weight`` 的行数；若数据索引超出检查点容量则报错。
    """
    if int(dataset_n_subjects) < 1:
        raise ValueError("dataset_n_subjects must be positive")
    weight = model_state.get("subject_embedding.weight")
    if not isinstance(weight, torch.Tensor):
        return int(dataset_n_subjects)
    if weight.ndim != 2 or weight.shape[0] < 1:
        raise ValueError("Checkpoint subject_embedding.weight has an invalid shape")
    checkpoint_n_subjects = int(weight.shape[0])
    shared = model_state.get("subject_embedding.shared_condition")
    training_ids = model_state.get("subject_embedding.training_subject_ids")
    if shared is not None or training_ids is not None:
        if (not isinstance(shared, torch.Tensor) or shared.shape != (weight.shape[1],)
                or not isinstance(training_ids, torch.Tensor) or training_ids.ndim != 1
                or training_ids.dtype != torch.long or training_ids.numel() == 0
                or bool((training_ids < 0).any()) or bool((training_ids >= checkpoint_n_subjects).any())
                or not torch.equal(training_ids, training_ids.unique(sorted=True))):
            raise ValueError("Checkpoint has invalid shared subject conditioning state")
        # Unknown subjects do not require another embedding row in shared mode.
        return checkpoint_n_subjects
    if checkpoint_n_subjects < int(dataset_n_subjects):
        raise ValueError(
            "Selected datasets contain a participant index outside the checkpoint "
            f"subject embedding ({dataset_n_subjects} > {checkpoint_n_subjects})"
        )
    return checkpoint_n_subjects
