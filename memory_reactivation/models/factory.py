"""Single Target construction entry point; preserves historical defaults."""
from __future__ import annotations
from typing import Mapping, Sequence
from copy import deepcopy
from .encoders import build_eeg_encoder, resolve_encoder_input_channels
from .target import TargetAttentionDecoder, normalize_task_class_indices

def optional_int(config: Mapping[str, object], name: str) -> int | None:
    """读取可缺省的整数配置；键不存在或值为 ``None`` 时返回 ``None``。"""
    value = config.get(name)
    return int(value) if value is not None else None

def resolve_model_task_class_indices(
    model_config: Mapping[str, object],
    explicit: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, tuple[int, ...]]:
    """合并数据解析结果与快照类别映射，并拒绝两者不一致。"""
    saved_value = model_config.get("resolved_task_class_indices")
    saved = (
        normalize_task_class_indices(saved_value)
        if saved_value is not None
        else None
    )
    resolved_explicit = (
        normalize_task_class_indices(explicit) if explicit is not None else None
    )
    if saved is not None and resolved_explicit is not None and saved != resolved_explicit:
        raise ValueError(
            "Explicit task class indices do not match "
            "model.resolved_task_class_indices"
        )
    return resolved_explicit or saved or normalize_task_class_indices(None)

def required_int_with_default(
    config: Mapping[str, object],
    name: str,
    default: int,
) -> int:
    """读取整数配置；键缺失或 YAML 显式 null 时回落到 default。

    注意：``dict.get(name, default)`` 在键存在且值为 ``None`` 时仍返回
    ``None``，因此不能用于处理 ``classifier_num_layers:`` 这类空字段。
    """
    value = config.get(name, default)
    if value is None:
        return int(default)
    return int(value)

def build_model(
    model_config: Mapping[str, object],
    n_subjects: int | None = None,
    time_point_count: int | None = None,
    task_class_indices: Mapping[str, Sequence[int]] | None = None,
    input_channels: int | None = None,
) -> TargetAttentionDecoder:
    """根据模型配置组装编码器、注意力、多时间头和 trial-level 分类头。

    参数：
        model_config: YAML 中的 ``model`` 映射。
        n_subjects: 被试 embedding 容量。
        time_point_count: 数据实际时间点数，决定 independent_linear 头的数量。

    返回：
        尚未迁移设备的 ``TargetAttentionDecoder``。``time_no_class_enabled`` 必须是
        真正的布尔值，避免字符串 ``"false"`` 被 Python 误判为真。
    """
    encoder_config = model_config.get("encoder")
    if not isinstance(encoder_config, Mapping):
        raise ValueError("model.encoder must be a mapping")
    if input_channels is not None:
        encoder_config = resolve_encoder_input_channels(encoder_config, input_channels)
    fusion_config = model_config.get("fusion", {})
    if fusion_config is None:
        fusion_config = {}
    if not isinstance(fusion_config, Mapping):
        raise ValueError("model.fusion must be a mapping when provided")
    reconstruction_config = model_config.get("reconstruction", {})
    if reconstruction_config is None:
        reconstruction_config = {}
    if not isinstance(reconstruction_config, Mapping):
        raise ValueError("model.reconstruction must be a mapping when provided")
    reconstruction_enabled = reconstruction_config.get("enabled", False)
    if not isinstance(reconstruction_enabled, bool):
        raise ValueError("model.reconstruction.enabled must be a boolean")
    subject_config = model_config.get("subject_conditioning", {})
    if subject_config is None:
        subject_config = {}
    if not isinstance(subject_config, Mapping):
        raise ValueError("model.subject_conditioning must be a mapping when provided")
    time_no_class_enabled = model_config.get("time_no_class_enabled", False)
    if not isinstance(time_no_class_enabled, bool):
        raise ValueError("model.time_no_class_enabled must be a boolean")
    resolved_task_class_indices = resolve_model_task_class_indices(
        model_config,
        task_class_indices,
    )
    # 新版快照记录解析后的头数量；重跑时必须与当前数据时间轴长度一致。
    saved_time_point_count = optional_int(model_config, "resolved_time_point_count")
    if saved_time_point_count is not None:
        if saved_time_point_count < 1:
            raise ValueError("model.resolved_time_point_count must be positive")
        if time_point_count is None:
            time_point_count = saved_time_point_count
        elif int(time_point_count) != saved_time_point_count:
            raise ValueError(
                "Explicit time_point_count does not match "
                "model.resolved_time_point_count: "
                f"{int(time_point_count)} != {saved_time_point_count}"
            )
    # encoder.type 决定 CNN 或不移位的右向局部 Transformer，其余解码头接口保持一致。
    encoder = build_eeg_encoder(
        encoder_config,
        time_point_count=time_point_count,
    )
    model = TargetAttentionDecoder(
        encoder=encoder,
        attention_dim=optional_int(model_config, "attention_dim"),
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        attention_mode=str(model_config.get("attention_mode", "scorer")),
        attention_score_centering=model_config.get("attention_score_centering", False),
        attention_gate_centering=model_config.get("attention_gate_centering", False),
        attention_gate_activation=model_config.get("attention_gate_activation", "sigmoid"),
        attention_gate_combination=model_config.get("attention_gate_combination", "multiply"),
        attention_effective_normalization=model_config.get("attention_effective_normalization", "none"),
        attention_query_source=str(model_config.get("attention_query_source", "subject")),
        attention_gate_init_bias=float(model_config.get("attention_gate_init_bias", 2.0)),
        n_subjects=n_subjects,
        subject_conditioning_enabled=bool(subject_config.get("enabled", False)),
        subject_conditioning_mode=subject_config.get("mode", "individual"),
        subject_training_ids=subject_config.get("resolved_training_subject_ids"),
        subject_embedding_dim=optional_int(subject_config, "embedding_dim"),
        subject_condition_dropout=float(subject_config.get("condition_dropout", 0.0)),
        subject_adaln_dropout=float(subject_config.get("adaln_dropout", subject_config.get("dropout", 0.0))),
        time_source_block=optional_int(model_config, "time_source_block"),
        time_head_mode=str(model_config.get("time_head_mode", "shared")),
        time_point_count=time_point_count,
        time_no_class_enabled=time_no_class_enabled,
        reconstruction_enabled=reconstruction_enabled,
        pooled_head_mode=str(model_config.get("pooled_head_mode", "auto")),
        pool_valid_context_only=model_config.get("pool_valid_context_only", False),
        task_class_indices=resolved_task_class_indices,
        classifier_num_layers=required_int_with_default(
            model_config, "classifier_num_layers", 2
        ),
        classifier_hidden_dim=optional_int(model_config, "classifier_hidden_dim"),
        classifier_dropout=float(
            0.0
            if model_config.get("classifier_dropout") is None
            else model_config.get("classifier_dropout", 0.0)
        ),
        fusion_enabled=bool(fusion_config.get("enabled", False)),
        fusion_hidden_dim=optional_int(fusion_config, "hidden_dim"),
        fusion_dropout=float(fusion_config.get("dropout", 0.0)),
    )
    from ..contracts import ModelSpec
    resolved = deepcopy(dict(model_config))
    resolved["encoder"] = deepcopy(dict(encoder_config))
    resolved["encoder"]["in_channels"] = encoder.in_channels
    resolved["resolved_time_point_count"] = time_point_count
    resolved["resolved_task_class_indices"] = {k: list(v) for k, v in resolved_task_class_indices.items()}
    for name in ("attention_gate_activation", "attention_gate_combination", "attention_effective_normalization",
                 "attention_score_centering", "attention_gate_centering", "time_no_class_enabled"):
        resolved[name] = getattr(model, name)
    capacity = None if model.subject_embedding is None else model.subject_embedding.num_embeddings
    model._pb_model_spec = ModelSpec("target", resolved, encoder.in_channels, time_point_count, capacity).to_dict()
    return model
