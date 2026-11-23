"""训练带注意力池化、多时间分类头的 PB2023 EEG target 解码器。

入口读取 YAML、任务与单个 lag，构造训练/验证及按需测试数据集，按真实时间轴长度创建
逐时间点分类头，并训练独立 trial-level pooled 头和可选 fusion 头。它还支持严格加载
time-only DPC 预训练组件、冻结编码器/被试条件化模块、导出真实 EEG 与高斯基线 CSV，
以及绘制最终结果。所有可复现信息和最优检查点均写入新运行目录。
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from pb2023_eeg.data.configured import (
    build_lag_dataset, configure_cross_validation, validate_subject_cv_datasets,
)
from pb2023_eeg.data.dataset import (
    DEFAULT_PITCH_CLASS_INDICES,
    LagDecodingDataset,
    TASK_SPECS,
    make_batch_sampler,
    normalize_choice_cell_history_fields,
    normalize_choice_label_shuffle,
    normalize_cell_weight_max,
    normalize_cell_weighting,
    normalize_dynamic_choice_sampling,
    normalize_lags,
    normalize_pitch_class_indices,
    normalize_time_shuffle,
    resolve_balance_seed,
    write_lag_example_manifest,
)
from pb2023_eeg.data.manifest import load_manifest
from pb2023_eeg.data.subject_conditioning import (
    configure_training_subject_conditioning, validate_pretrained_subject_mapping,
)
from pb2023_eeg.models.encoder import (
    build_eeg_encoder, normalize_encoder_type, resolve_encoder_input_channels,
)
from pb2023_eeg.models.target_decoder import (
    TargetAttentionDecoder,
    normalize_task_class_indices,
)
from pb2023_eeg.data.time_axis import resolve_attention_window
from pb2023_eeg.training.common import load_checkpoint, resolve_learning_rate_schedule, validate_checkpoint_cv_identity
from pb2023_eeg.training.swanlab_logger import SwanLabLogger
from pb2023_eeg.training.target import (
    export_target_outputs,
    fit_target_decoder,
    normalize_loss_type,
    resolve_focal_train_config,
    score_mode_for_loss_type,
)
from pb2023_eeg.data.spatial import configure_spatial_preprocessing
from pb2023_eeg.utils.config import load_yaml, save_yaml
from pb2023_eeg.utils.run import (
    configure_runtime_metadata,
    prepare_run_directory,
    snapshot_data_artifacts,
)
from pb2023_eeg.utils.seed import seed_everything
from pb2023_eeg.workflows.report import plot_target_run
from pb2023_eeg.models.factory import build_model, optional_int, required_int_with_default, resolve_model_task_class_indices


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None) -> argparse.Namespace:
    """解析 target 训练配置、任务、lag 和可选 DPC 检查点覆盖值。

    返回：
        经 argparse 完成基本类型与任务枚举校验的参数命名空间。
    """
    parser = argparse.ArgumentParser(description="Train a single-stage PB2023 target decoder.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), required=True)
    parser.add_argument("--lag", type=int, help="Override data.lags with one lag for this run.")
    parser.add_argument(
        "--cv-fold", type=int, default=None,
        help="Select one generated CV fold (0-based); validation is the held-out fold.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=None,
        help="Number of generated CV folds (default: 5 when --cv-fold is used).",
    )
    parser.add_argument(
        "--cv-strategy", choices=["block", "subject"], default=None,
        help="Use block-level or held-out-subject CV artifacts (defaults to config).",
    )
    parser.add_argument(
        "--cv-output-dir", type=Path, default=None,
        help="CV root containing manifest.csv and fold_{i}/ artifacts (e.g. artifacts/cv5).",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help=(
            "Resume training from an existing last.ckpt/best_val.ckpt: restores "
            "model/optimizer/AMP scaler/RNG states and continues from its iteration."
        ),
    )
    parser.add_argument(
        "--pretrained-dpc-checkpoint",
        type=Path,
        help=(
            "Override model.pretrained_dpc.checkpoint. This is intended for staged "
            "launchers that resolve a completed time-only run at launch time."
        ),
    )
    return parser.parse_args(argv)


def make_loader(
    dataset: LagDecodingDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    batch_sampling: str,
) -> DataLoader:
    """按项目统一的批采样策略构造 DataLoader。

    ``batch_sampler`` 同时负责 batch 尺寸、打乱和 cell 采样策略，因此 DataLoader
    本身不再接收 ``batch_size`` 或 ``shuffle``。CUDA 可用时自动启用锁页内存。
    """
    # 分组采样必须由同一个 sampler 统一控制，避免 DataLoader 二次打乱破坏 cell 平衡。
    sampler = make_batch_sampler(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        strategy=batch_sampling,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )






def resolve_dataset_task_class_indices(
    datasets: Mapping[str, object],
    *,
    task: str,
) -> dict[str, tuple[int, ...]]:
    """校验所有 split 使用同一类别映射，并补齐其他任务的默认映射。"""
    if not datasets:
        raise ValueError("At least one dataset is required to resolve task classes")
    first_name = next(iter(datasets))
    first = datasets[first_name]
    if not hasattr(first, "class_indices"):
        raise TypeError(f"Dataset split {first_name!r} has no class_indices")
    indices = tuple(int(index) for index in getattr(first, "class_indices"))
    for split, dataset in datasets.items():
        candidate = tuple(int(index) for index in getattr(dataset, "class_indices"))
        if candidate != indices:
            raise ValueError(
                f"Dataset task class mapping mismatch: {split} has {candidate}, "
                f"expected {indices}"
            )
    return normalize_task_class_indices({task: indices})




def resolve_dataset_input_channels(datasets: Mapping[str, object]) -> int:
    """Require every split to expose the same preprocessed input width."""
    if not datasets:
        raise ValueError("At least one dataset is required to resolve input_channels")
    counts = {split: getattr(dataset, "input_channels", None)
              for split, dataset in datasets.items()}
    if any(value is None for value in counts.values()) or len(set(counts.values())) != 1:
        raise ValueError(f"Dataset input_channels mismatch or missing: {counts}")
    return int(next(iter(counts.values())))


def resolve_dataset_time_axis(
    datasets: Mapping[str, object],
    *,
    reference_split: str | None = None,
) -> torch.Tensor:
    """校验所有数据 split 的时间轴完全一致，并返回统一的一维轴。

    参数：
        datasets: split 名到数据集对象的映射；对象必须提供 ``time_s``。
        reference_split: 参考 split，缺省取映射中的第一个。

    返回：
        位于 CPU、类型为 float32 的一维时间张量。长度、具体值或有限性不一致均报错；
        这是保证独立时间头与样本位置一一对应的硬约束。
    """
    if not datasets:
        raise ValueError("At least one dataset is required to resolve time_s")
    reference_name = reference_split or next(iter(datasets))
    if reference_name not in datasets:
        raise ValueError(f"Reference split {reference_name!r} is not available")

    def axis_for(split: str) -> torch.Tensor:
        """将指定 split 的 ``time_s`` 校验并规范为 CPU float64 张量。"""
        dataset = datasets[split]
        if not hasattr(dataset, "time_s"):
            raise TypeError(f"Dataset split {split!r} has no time_s attribute")
        axis = torch.as_tensor(getattr(dataset, "time_s"), dtype=torch.float64)
        if axis.ndim != 1 or axis.numel() == 0:
            raise ValueError(f"Dataset split {split!r} time_s must be a non-empty 1D axis")
        if not bool(torch.isfinite(axis).all()):
            raise ValueError(f"Dataset split {split!r} time_s contains non-finite values")
        return axis.detach().cpu()

    reference = axis_for(reference_name)
    for split in datasets:
        if split == reference_name:
            continue
        candidate = axis_for(split)
        if candidate.shape != reference.shape:
            raise ValueError(
                "Dataset time_s length mismatch: "
                f"{reference_name} has {reference.numel()} points but "
                f"{split} has {candidate.numel()}"
            )
        if not torch.equal(candidate, reference):
            raise ValueError(
                "Dataset time_s value mismatch: "
                f"{split} does not exactly match {reference_name}"
            )
    return reference.to(dtype=torch.float32)


def resolved_time_axis_metadata(times: torch.Tensor) -> dict[str, object]:
    """生成可写入模型快照的时间轴长度和 SHA-256 指纹。

    哈希基于连续 float64 原始字节，可在重导出时发现长度相同但采样时刻不同的情况。
    """
    axis = torch.as_tensor(times, dtype=torch.float64).detach().cpu().contiguous()
    if axis.ndim != 1 or axis.numel() == 0:
        raise ValueError("Resolved time axis must be a non-empty 1D tensor")
    return {
        "resolved_time_point_count": int(axis.numel()),
        "resolved_time_axis_sha256": hashlib.sha256(
            axis.numpy().tobytes()
        ).hexdigest(),
    }


def resolve_export_splits(plots_config: Mapping[str, object]) -> tuple[str, ...]:
    """规范化允许导出和绘制的评估 split。

    字符串被视作单元素列表，重复项按首次出现去重；仅接受 ``val`` 与 ``test``。
    """
    value = plots_config.get("splits", ("val", "test"))
    if isinstance(value, str):
        values: Sequence[object] = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise ValueError("plots.splits must be a sequence containing val and/or test")
    splits = tuple(dict.fromkeys(str(split) for split in values))
    if not splits or any(split not in {"val", "test"} for split in splits):
        raise ValueError("plots.splits must contain only val and/or test")
    return splits


def resolve_target_dataset_splits(
    plots_config: Mapping[str, object],
) -> tuple[str, ...]:
    """确定训练流程真正需要构造的数据集 split。

    训练和验证始终加载；只有导出策略包含 ``test`` 时才加载测试集，以节约调试开销。
    """
    export_splits = resolve_export_splits(plots_config)
    return ("train", "val") + (("test",) if "test" in export_splits else ())


def _sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，返回十六进制摘要且不一次读入整个检查点。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_pretrained_dpc_config(
    model_config: Mapping[str, object],
    cli_checkpoint: Path | None,
) -> dict[str, object] | None:
    """合并 YAML 与命令行的可选 DPC 初始化配置。

    命令行检查点优先并隐式启用预训练；显式关闭或完全缺省时返回 ``None``。启用后会
    解析绝对路径并要求文件真实存在，返回值同时固化是否加载被试条件化组件。
    """
    raw = model_config.get("pretrained_dpc")
    if raw is None and cli_checkpoint is None:
        return None
    if raw is None:
        pretrained_config: dict[str, object] = {}
    elif isinstance(raw, Mapping):
        pretrained_config = dict(raw)
    else:
        raise ValueError("model.pretrained_dpc must be a mapping when provided")

    enabled = bool(pretrained_config.get("enabled", True))
    if cli_checkpoint is not None:
        enabled = True
        checkpoint_value: object | None = cli_checkpoint
    else:
        checkpoint_value = pretrained_config.get("checkpoint")
    if not enabled:
        return None
    if checkpoint_value is None or not str(checkpoint_value).strip():
        raise ValueError(
            "model.pretrained_dpc is enabled but no checkpoint was provided; set "
            "model.pretrained_dpc.checkpoint or --pretrained-dpc-checkpoint"
        )
    checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Pretrained DPC checkpoint does not exist: {checkpoint}")
    return {
        **pretrained_config,
        "enabled": True,
        "checkpoint": str(checkpoint),
        "load_subject_conditioning": bool(
            pretrained_config.get("load_subject_conditioning", False)
        ),
    }


def _strict_component_state(
    model_state: Mapping[str, torch.Tensor],
    checkpoint_state: Mapping[str, object],
    prefix: str,
    component_name: str,
) -> dict[str, torch.Tensor]:
    """严格提取检查点中某个前缀组件的参数。

    目标模型与检查点必须具有完全相同的键集合、张量类型和形状；函数返回可交给
    ``load_state_dict(strict=False)`` 的子状态字典，但不会在这里修改模型。
    """
    # 同时拒绝缺失键和多余键，避免预训练组件只加载了一部分却继续训练。
    expected = {key for key in model_state if key.startswith(prefix)}
    if not expected:
        raise ValueError(
            f"Cannot load pretrained {component_name}: target model has no {prefix} parameters"
        )
    available = {key for key in checkpoint_state if key.startswith(prefix)}
    missing = sorted(expected - available)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Pretrained DPC checkpoint is missing {len(missing)} {component_name} keys: {preview}"
        )
    unexpected = sorted(available - expected)
    if unexpected:
        preview = ", ".join(unexpected[:5])
        raise ValueError(
            f"Pretrained DPC checkpoint has {len(unexpected)} unexpected "
            f"{component_name} keys: {preview}"
        )
    selected: dict[str, torch.Tensor] = {}
    for key in sorted(expected):
        value = checkpoint_state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Checkpoint state {key} is not a tensor")
        if value.shape != model_state[key].shape:
            raise ValueError(
                f"Pretrained DPC shape mismatch for {key}: checkpoint {tuple(value.shape)} "
                f"!= target {tuple(model_state[key].shape)}"
            )
        selected[key] = value
    return selected


def _validate_pretrained_cnn_dilations(
    model: TargetAttentionDecoder,
    checkpoint: Mapping[str, object],
) -> bool:
    """Validate CNN dilation metadata before loading shape-compatible weights."""
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        return False
    model_config = config.get("model")
    if not isinstance(model_config, Mapping):
        return False
    source = model_config.get("resolved_encoder", model_config.get("encoder"))
    if not isinstance(source, Mapping):
        return False
    source_type = normalize_encoder_type(source.get("type", source.get("name", "cnn")))
    target = model.encoder.resolved_config()
    if source_type != "cnn" or target.get("type") != "cnn":
        return False
    source_dilations = source.get("dilations")
    if source_dilations is None:
        source_blocks = source.get("num_blocks")
        if source_blocks is None:
            return False
        source_dilations = [1] * int(source_blocks)
    checkpoint_signature = tuple(int(value) for value in source_dilations)
    target_signature = tuple(int(value) for value in target["dilations"])
    if checkpoint_signature != target_signature:
        raise ValueError(
            "Pretrained DPC CNN dilation mismatch: checkpoint "
            f"{checkpoint_signature} != target {target_signature}"
        )
    return True


def load_pretrained_dpc_components(
    model: TargetAttentionDecoder,
    checkpoint_path: Path,
    *,
    load_subject_conditioning: bool = False,
    expected_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """严格加载 DPC 编码器及可选被试条件化参数。

    参数：
        model: 待初始化的 target 解码器。
        checkpoint_path: time-only/DPC 检查点路径。
        load_subject_conditioning: 是否同时加载 subject embedding 与 adapter。
        expected_config: shared_plus_residual 被试组件加载时必需，用于核对目标被试身份。

    返回：
        包含解析路径、文件哈希、已加载组件和参数键数量的审计元数据。
    """
    resolved = checkpoint_path.expanduser().resolve()
    checkpoint = load_checkpoint(resolved)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Pretrained DPC checkpoint must contain a mapping")
    if expected_config is not None:
        validate_checkpoint_cv_identity(checkpoint, expected_config)
    if load_subject_conditioning and model.subject_conditioning_mode == "shared_plus_residual":
        validate_pretrained_subject_mapping(checkpoint.get("config"), expected_config)
    checkpoint_state = checkpoint.get("model_state")
    if not isinstance(checkpoint_state, Mapping):
        raise ValueError("Pretrained DPC checkpoint is missing a model_state mapping")
    encoder_config_checked = _validate_pretrained_cnn_dilations(model, checkpoint)
    model_state = model.state_dict()
    # 分类头结构可能不同，因此这里只选择 encoder 和明确要求的 subject 组件。
    selected = _strict_component_state(
        model_state,
        checkpoint_state,
        "encoder.",
        "encoder",
    )
    loaded_components = ["encoder"]
    if load_subject_conditioning:
        subject_prefixes = [
            prefix
            for prefix in ("subject_embedding.", "subject_adapters.")
            if any(key.startswith(prefix) for key in model_state)
        ]
        if not subject_prefixes:
            raise ValueError(
                "load_subject_conditioning=true but the target model has no subject embedding/adapters"
            )
        for prefix in subject_prefixes:
            component_name = prefix.rstrip(".")
            selected.update(
                _strict_component_state(
                    model_state,
                    checkpoint_state,
                    prefix,
                    component_name,
                )
            )
            loaded_components.append(component_name)
    incompatible = model.load_state_dict(selected, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "Unexpected keys while loading pretrained DPC components: "
            + ", ".join(incompatible.unexpected_keys)
        )
    return {
        "resolved_checkpoint": str(resolved),
        "checkpoint_sha256": _sha256_file(resolved),
        "load_subject_conditioning": bool(load_subject_conditioning),
        "loaded_components": loaded_components,
        "loaded_key_count": len(selected),
        "encoder_config_checked": encoder_config_checked,
    }


def configure_target_freezing(
    model: TargetAttentionDecoder,
    model_config: Mapping[str, object],
) -> dict[str, object]:
    """按配置冻结编码器和/或被试条件化模块，并汇总参数数量。

    返回字典会写回配置快照，其中包含冻结开关、可训练参数数和总参数数。
    """
    raw = model_config.get("freeze", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("model.freeze must be a mapping when provided")
    freeze_encoder = bool(raw.get("encoder", False))
    freeze_subject = bool(raw.get("subject_conditioning", False))
    model.configure_frozen_components(
        encoder=freeze_encoder,
        subject_conditioning=freeze_subject,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "encoder": freeze_encoder,
        "subject_conditioning": freeze_subject,
        "trainable_parameter_count": trainable,
        "total_parameter_count": total,
    }


def resolve_single_lag(data_config: Mapping[str, object], cli_lag: int | None) -> int:
    """解析本次 target 运行唯一且非负的 lag，命令行值优先。"""
    if cli_lag is not None:
        return normalize_lags([cli_lag])[0]
    unique = normalize_lags(data_config.get("lags", []))
    if len(unique) != 1:
        raise ValueError("Each target run must use exactly one lag; pass --lag or set data.lags to one value")
    return unique[0]


def validate_independent_time_heads_cover_attention_window(
    model_config: Mapping[str, object],
    *,
    time_point_count: int,
    attention_indices: torch.Tensor,
) -> None:
    """拒绝独立时间头数量大于 attention/time-loss 窗口的配置。

    ``independent_linear`` / ``independent_mlp`` 为每个数据时间点建一套参数，但
    time-level 监督只落在 ``attention_indices`` 内的 Top-q 位置。窗口短于全长
    ``T`` 时，窗外 head 永远收不到 time loss，属于易被忽视的死参数。
    """
    time_head_mode = str(model_config.get("time_head_mode", "shared")).lower().replace("-", "_")
    if not time_head_mode.startswith("independent_"):
        return
    window_count = int(attention_indices.numel())
    total_count = int(time_point_count)
    if window_count < 1:
        raise ValueError("attention_indices must be non-empty for independent time heads")
    if window_count != total_count:
        raise ValueError(
            f"time_head_mode={time_head_mode} builds one classification head per data "
            f"time point (T={total_count}), but the attention/time-loss window only "
            f"covers {window_count} samples. Heads outside the window receive no "
            "time-level loss. Expand attention_time_* / train_time_* to the full data "
            "axis, or switch to time_head_mode=shared."
        )


def resolve_target_export_checkpoint(
    output_dir: Path,
    *,
    prefer_best: bool = True,
) -> Path:
    """解析导出用的 target checkpoint，优先 best_val，必要时回退 last。

    正常训练会在首次有效验证后写出 ``best_val.ckpt``。若进程在写出 best 前
    中断、或仅保留了 ``last.ckpt``，导出仍应尽量完成而不是硬失败。
    """
    output_dir = Path(output_dir)
    best_path = output_dir / "best_val.ckpt"
    last_path = output_dir / "last.ckpt"
    primary = best_path if prefer_best else last_path
    fallback = last_path if prefer_best else best_path
    if primary.is_file():
        return primary
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"No exportable target checkpoint under {output_dir}: "
        "expected best_val.ckpt or last.ckpt"
    )




def main(argv=None) -> None:
    """执行一次完整 target 训练、最优检查点评估、CSV 导出和绘图。

    输入为 YAML、任务、单个 lag 与可选预训练检查点；输出包括运行/数据快照、训练日志、
    ``best_val.ckpt``、真实及可选高斯基线评估 CSV 和图表。本函数不返回模型对象。
    """
    args = parse_args(argv)  # --config YAML, --task, --lag, 可选 DPC ckpt
    source_config = args.config.expanduser().resolve()
    from ..config import parse_request
    overrides = {"task": args.task}
    if args.lag is not None:
        overrides["data"] = {"lags": [args.lag]}
    config = parse_request(source_config, overrides).config
    from ..artifacts.provenance import runtime_implementation_identity
    implementation = runtime_implementation_identity()
    if args.resume_from is not None:
        saved_resume = load_checkpoint(args.resume_from)
        if (saved_resume.get("implementation_sha256") != implementation["sha256"]
                or saved_resume.get("implementation_identity_schema") != implementation["schema"]):
            raise ValueError("Exact resume requires this implementation. Use the run's frozen code snapshot for historical training; load_model supports weight migration separately.")
    config["task"] = args.task  # "pitch"|"choice"|"motor"，一次 run 一个
    seed = int(config.get("seed", 20260530))  # 模型/batch 随机种子
    seed_everything(seed)
    if "data" not in config:
        raise ValueError("Config is missing required top-level key: data")
    if "train" not in config:
        raise ValueError("Config is missing required top-level key: train")
    configure_cross_validation(
        config, cv_fold=args.cv_fold, cv_folds=args.cv_folds,
        cv_output_dir=args.cv_output_dir, resume_from=args.resume_from,
        cv_strategy=getattr(args, "cv_strategy", None),
    )
    configure_spatial_preprocessing(config, resume_from=args.resume_from)
    data_config = config["data"]  # 路径、lags、平衡、采样…
    train_config = config["train"]  # 优化器、loss 权重、Top-q、增强…
    if not isinstance(data_config, Mapping):
        raise ValueError("data must be a mapping")
    if not isinstance(train_config, Mapping):
        raise ValueError("train must be a mapping")
    train_config = resolve_focal_train_config(train_config)
    train_config = resolve_learning_rate_schedule(
        train_config, saved_resume.get("config", {}).get("train", {}) if args.resume_from is not None else None,
    )
    config["train"] = train_config
    plots_config = config.get("plots", {}) or {}  # 导出 split、高斯基线
    if not isinstance(plots_config, Mapping):
        raise ValueError("plots must be a mapping when provided")
    data_config = dict(data_config)  # 可写副本，规范化结果写回 snapshot
    export_splits = resolve_export_splits(plots_config)
    dataset_splits = resolve_target_dataset_splits(plots_config)  # train/val[/test]
    config["data"] = data_config
    lag = resolve_single_lag(data_config, args.lag)  # int≥0；0=当前标签
    data_config["lags"] = [lag]
    config["lag"] = lag
    pitch_class_indices = normalize_pitch_class_indices(
        data_config.get("pitch_class_indices")
    )
    if args.task != "pitch" and "pitch_class_indices" in data_config:
        raise ValueError("data.pitch_class_indices is only supported for pitch")
    if (
        args.task == "pitch"
        and pitch_class_indices != DEFAULT_PITCH_CLASS_INDICES
        and lag != 0
    ):
        raise ValueError("data.pitch_class_indices subsets require pitch lag=0")
    if args.task == "pitch":
        # 快照总是保存升序规范化结果；数组顺序不会改变内部标签语义。
        data_config["pitch_class_indices"] = list(pitch_class_indices)
    balance_seed = resolve_balance_seed(data_config, seed)  # 样本平衡用，可≠seed
    train_cell_weighting = normalize_cell_weighting(
        data_config.get("train_cell_weighting", "none")
    )
    train_cell_weight_max = normalize_cell_weight_max(
        data_config.get("train_cell_weight_max")
    )
    train_drop_incomplete_cells = bool(
        data_config.get("train_drop_incomplete_cells", False)
    )
    (
        train_dynamic_cell_sampling,
        train_min_cell_class_count,
        train_majority_ratio_cap,
    ) = normalize_dynamic_choice_sampling(
        data_config.get("train_dynamic_cell_sampling", False),
        data_config.get("train_min_cell_class_count"),
        data_config.get("train_majority_ratio_cap"),
    )
    choice_cell_history_fields = normalize_choice_cell_history_fields(
        data_config.get("choice_cell_history_fields")
    )
    (
        train_label_shuffle_within_cells,
        train_label_shuffle_seed,
    ) = normalize_choice_label_shuffle(
        data_config.get("train_label_shuffle_within_cells", False),
        data_config.get("train_label_shuffle_seed"),
    )
    (
        train_time_shuffle_within_train,
        train_time_shuffle_seed,
    ) = normalize_time_shuffle(
        data_config.get("train_time_shuffle_within_train", False),
        data_config.get("train_time_shuffle_seed"),
    )
    eval_min_cell_class_count = int(
        data_config.get("eval_min_cell_class_count", 1)
    )
    if eval_min_cell_class_count < 1:
        raise ValueError("data.eval_min_cell_class_count must be positive")
    balance_train_cells = bool(data_config.get("balance_train_cells", True))
    if choice_cell_history_fields and args.task != "choice":
        raise ValueError(
            "data.choice_cell_history_fields is only supported for choice"
        )
    if train_label_shuffle_within_cells and args.task not in ("choice", "pitch"):
        raise ValueError(
            "data.train_label_shuffle_within_cells is only supported for choice and pitch"
        )
    if train_cell_weighting != "none" and args.task not in ("choice", "pitch"):
        raise ValueError("data.train_cell_weighting is only supported for choice/pitch")
    if train_cell_weighting != "none" and balance_train_cells:
        raise ValueError(
            "data.train_cell_weighting=inverse_frequency requires balance_train_cells=false"
        )
    if train_cell_weighting == "none" and train_cell_weight_max is not None:
        raise ValueError(
            "data.train_cell_weight_max requires train_cell_weighting=inverse_frequency"
        )
    if train_drop_incomplete_cells and train_cell_weighting != "inverse_frequency":
        raise ValueError(
            "data.train_drop_incomplete_cells requires "
            "train_cell_weighting=inverse_frequency"
        )
    num_workers = int(data_config.get("num_workers", 0))
    if num_workers < 0:
        raise ValueError("data.num_workers must be non-negative")
    if train_dynamic_cell_sampling:
        if args.task != "choice":
            raise ValueError("dynamic Choice sampling is only supported for choice")
        if balance_train_cells:
            raise ValueError(
                "dynamic Choice sampling requires balance_train_cells=false"
            )
        if train_cell_weighting != "inverse_frequency":
            raise ValueError(
                "dynamic Choice sampling requires train_cell_weighting=inverse_frequency"
            )
        if train_cell_weight_max is not None:
            raise ValueError(
                "dynamic Choice sampling does not allow train_cell_weight_max"
            )
        if num_workers > 0:
            raise ValueError(
                "data.train_dynamic_cell_sampling requires data.num_workers=0 because "
                "set_epoch replaces train examples in the parent process and DataLoader "
                "workers would keep stale sample lists/lengths"
            )
    # 重导出只读取快照，因此这里必须记录所有经规范化的隐式配置值。
    data_config["balance_seed"] = balance_seed
    data_config["train_cell_weighting"] = train_cell_weighting
    data_config["train_cell_weight_max"] = train_cell_weight_max
    data_config["train_drop_incomplete_cells"] = train_drop_incomplete_cells
    data_config["train_dynamic_cell_sampling"] = train_dynamic_cell_sampling
    data_config["train_min_cell_class_count"] = train_min_cell_class_count
    data_config["train_majority_ratio_cap"] = train_majority_ratio_cap
    data_config["choice_cell_history_fields"] = list(
        choice_cell_history_fields
    )
    data_config["train_label_shuffle_within_cells"] = (
        train_label_shuffle_within_cells
    )
    data_config["train_label_shuffle_seed"] = train_label_shuffle_seed
    if train_label_shuffle_within_cells:
        data_config["train_label_shuffle_method"] = "within_condition_permutation_v1"
    data_config["train_time_shuffle_within_train"] = train_time_shuffle_within_train
    data_config["train_time_shuffle_seed"] = train_time_shuffle_seed
    data_config["eval_min_cell_class_count"] = eval_min_cell_class_count

    run_base_dir = configure_runtime_metadata(
        config,
        mode="target",
        task=args.task,
        lag=lag,
        cv_fold=config.get("cv_fold"),
        cv_folds=config.get("cv_folds"),
    )
    if args.resume_from is not None:
        # resume 直接复用原 run 目录：CSV/checkpoint 自然追加，不再新建时间戳目录，
        # 也不重复快照（保留原 run 的 code_snapshot/artifacts/config_snapshot 完整性）。
        output_dir = args.resume_from.resolve().parent
        config["output_dir"] = str(output_dir)
        print(f"[target:{args.task}:lag{lag}] resuming into run directory {output_dir}", flush=True)
    else:
        prepared = prepare_run_directory(  # 新建 run_*/ + code_snapshot + input_config
            base_output_dir=run_base_dir,
            project_root=PROJECT_ROOT,
            source_config=source_config,
            command=sys.argv,
        )
        output_dir = prepared.output_dir
        config["output_dir"] = str(output_dir)
        snapshot_data_artifacts(config, output_dir)  # 拷贝 manifest/split/norm + sha256
    # Resume paths were restored and hash-checked by configure_cross_validation.
    data_config = dict(config["data"])
    records = load_manifest(
        Path(data_config["manifest_csv"]),
        experiments=[int(exp) for exp in data_config.get("experiments", [1])],
        subjects=data_config.get("subjects"),
    )
    if args.resume_from is not None:
        # 不覆盖原 config_snapshot.yaml（它代表本次 run 的解析后配置）；resume
        # 的解析结果另存，便于审计。
        save_yaml(output_dir / "config_snapshot_resume.yaml", config)
    else:
        save_yaml(output_dir / "config_snapshot.yaml", config)

    datasets = {
        split: build_lag_dataset(
            data_config, task=args.task, split=split, seed=seed, records=records,
        )
        for split in dataset_splits
    }
    empty_splits = [split for split, dataset in datasets.items() if len(dataset) == 0]
    if empty_splits:
        raise ValueError(
            "Target dataset splits are empty after construction: "
            + ", ".join(empty_splits)
        )
    train_example_summary = write_lag_example_manifest(
        datasets["train"],
        output_dir / "train_examples.csv",
        output_dir / "train_examples_summary.json",
    )
    print(
        f"[target:{args.task}] run directory: {output_dir}\n"
        f"[target:{args.task}:lag{lag}] records={len(records)} "
        + " ".join(f"{split}={len(datasets[split])}" for split in dataset_splits),
        flush=True,
    )
    print(
        f"[target:{args.task}:lag{lag}] balance_seed={balance_seed} "
        f"train_cell_weighting={train_cell_weighting} "
        f"train_cell_weight_max={train_cell_weight_max} "
        f"train_drop_incomplete_cells={train_drop_incomplete_cells} "
        f"train_dynamic_cell_sampling={train_dynamic_cell_sampling} "
        f"train_min_cell_class_count={train_min_cell_class_count} "
        f"train_majority_ratio_cap={train_majority_ratio_cap} "
        f"choice_cell_history_fields={list(choice_cell_history_fields)} "
        f"train_label_shuffle_within_cells={train_label_shuffle_within_cells} "
        f"train_label_shuffle_seed={train_label_shuffle_seed} "
        f"train_time_shuffle_within_train={train_time_shuffle_within_train} "
        f"train_time_shuffle_seed={train_time_shuffle_seed} "
        f"eval_min_cell_class_count={eval_min_cell_class_count} "
        f"train_examples_sha256={train_example_summary['sha256']}",
        flush=True,
    )
    print(
        f"[target:{args.task}:lag{lag}] dataset sizes before training: "
        + " ".join(f"{split}={len(datasets[split])}" for split in dataset_splits),
        flush=True,
    )
    # 各 split 使用固定种子偏移，评估顺序稳定，训练集才启用 shuffle。
    if train_config.get("batch_size") is None:
        raise ValueError("train.batch_size is required")
    loaders = {
        split: make_loader(
            dataset,
            batch_size=int(train_config["batch_size"]),
            shuffle=(split == "train"),
            num_workers=num_workers,
            seed=seed + {"train": 0, "val": 1, "test": 2}[split],
            batch_sampling=str(data_config.get("batch_sampling", "global")),
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_subjects = (
        max(int(index) for dataset in datasets.values() for index in dataset.participant_indices)
        + 1
    )
    if "model" not in config:
        raise ValueError("Config is missing required top-level key: model")
    model_config = config["model"]
    if not isinstance(model_config, Mapping):
        raise ValueError("model must be a mapping")
    model_config = dict(model_config)
    config["model"] = model_config
    validate_subject_cv_datasets(config, datasets)
    training_subject_capacity = configure_training_subject_conditioning(model_config, datasets)
    if training_subject_capacity is not None:
        n_subjects = training_subject_capacity
    task_class_indices = resolve_dataset_task_class_indices(
        datasets,
        task=args.task,
    )
    model_config["resolved_task_class_indices"] = {
        task: list(indices) for task, indices in task_class_indices.items()
    }
    times = resolve_dataset_time_axis(datasets, reference_split="train")  # [T] 秒，各 split 必须一致
    time_axis_metadata = resolved_time_axis_metadata(times)  # count + sha256
    model_config.update(time_axis_metadata)
    model_config["encoder"] = resolve_encoder_input_channels(
        model_config["encoder"], resolve_dataset_input_channels(datasets),
    )
    from ..config import ConfigRequest, DataContext, resolve_request
    resolved = resolve_request(ConfigRequest(config, source_config, {}), DataContext(
        input_channels=resolve_dataset_input_channels(datasets), time_s=times.tolist(),
        task_class_indices=task_class_indices,
        training_subject_ids=(model_config.get("subject_conditioning") or {}).get("resolved_training_subject_ids", ()),
        embedding_capacity=n_subjects,
    ))
    config = resolved.config
    model_config = config["model"]
    model = build_model(
        model_config,
        n_subjects=n_subjects,  # subject embedding 表大小 = max(id)+1
        time_point_count=int(time_axis_metadata["resolved_time_point_count"]),  # T，如 275
        task_class_indices=task_class_indices,
    )
    model_config["resolved_encoder"] = model.encoder.resolved_config()
    # 可选：只装 encoder（及 subject），attention/分类头仍随机
    pretrained_config = resolve_pretrained_dpc_config(
        model_config,
        args.pretrained_dpc_checkpoint,
    )
    if pretrained_config is not None:
        pretrained_metadata = load_pretrained_dpc_components(
            model,
            Path(str(pretrained_config["checkpoint"])),
            load_subject_conditioning=bool(
                pretrained_config["load_subject_conditioning"]
            ),
            expected_config=config,
        )
        model_config["pretrained_dpc"] = {
            **pretrained_config,
            **pretrained_metadata,
        }
        print(
            f"[target:{args.task}:lag{lag}] loaded pretrained DPC "
            f"{pretrained_metadata['resolved_checkpoint']} "
            f"sha256={pretrained_metadata['checkpoint_sha256']} "
            f"components={','.join(pretrained_metadata['loaded_components'])}",
            flush=True,
        )
    freeze_metadata = configure_target_freezing(model, model_config)
    model_config["freeze"] = freeze_metadata
    print(
        f"[target:{args.task}:lag{lag}] freeze_encoder={freeze_metadata['encoder']} "
        f"freeze_subject_conditioning={freeze_metadata['subject_conditioning']} "
        f"trainable_parameters={freeze_metadata['trainable_parameter_count']}/"
        f"{freeze_metadata['total_parameter_count']}",
        flush=True,
    )
    if args.resume_from is not None:
        save_yaml(output_dir / "config_snapshot_resume.yaml", config)
    else:
        save_yaml(output_dir / "config_snapshot.yaml", config)
    model.to(device)
    start_s, end_s = resolve_attention_window(train_config, times)  # 秒，闭区间
    # attention_indices: long [Ta]，落在 [start_s,end_s] 的全局 t；Top-q 在 0..Ta-1 上排
    attention_indices = torch.nonzero((times >= start_s) & (times <= end_s), as_tuple=False).flatten().to(device)
    if attention_indices.numel() == 0:
        raise ValueError("Configured attention/time-loss window contains no EEG samples")
    validate_independent_time_heads_cover_attention_window(  # independent 时窗必须盖满 T
        model_config,
        time_point_count=int(time_axis_metadata["resolved_time_point_count"]),
        attention_indices=attention_indices.detach().cpu(),
    )

    logger = SwanLabLogger.from_config(
        config,
        output_dir,
        job_type="target-train",
    )
    try:
        fit_target_decoder(  # iteration 训练 → best_val.ckpt / last.ckpt
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            task=args.task,
            attention_indices=attention_indices,
            config=config,
            output_dir=output_dir,
            device=device,
            logger=logger,
            resume_from=args.resume_from,
        )
        checkpoint_path = resolve_target_export_checkpoint(output_dir)  # 优先 best，缺则 last
        if checkpoint_path.name != "best_val.ckpt":
            print(
                f"[target:{args.task}:lag{lag}] best_val.ckpt missing; "
                f"exporting from {checkpoint_path.name}",
                flush=True,
            )
        checkpoint = load_checkpoint(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()  # 切换到推理模式：关闭 dropout，BatchNorm 使用 running stats
        amp = bool(train_config.get("amp", True)) and device.type == "cuda"
        gaussian_config = plots_config.get("gaussian_baseline", {})
        if gaussian_config is None:
            gaussian_config = {}
        if not isinstance(gaussian_config, Mapping):
            raise ValueError("plots.gaussian_baseline must be a mapping when provided")
        gaussian_enabled = bool(gaussian_config.get("enabled", False))
        gaussian_seed_base = int(gaussian_config.get("seed", seed + 100_000))
        gaussian_noise_mode = str(gaussian_config.get("noise_mode", "sample_channel"))
        activation_config = plots_config.get("activation", {})
        if activation_config is None:
            activation_config = {}
        if not isinstance(activation_config, Mapping):
            raise ValueError("plots.activation must be a mapping when provided")
        activation_threshold = float(activation_config.get("threshold", 0.5))
        # time、pooled、fusion 可分别采用 CE/BCE，导出分数需逐分支匹配 softmax/sigmoid。
        default_loss_type = normalize_loss_type(train_config.get("loss_type", "ce"), name="loss_type")
        time_loss_type = normalize_loss_type(
            train_config.get("time_loss_type", default_loss_type),
            name="time_loss_type",
        )
        pooled_loss_type = normalize_loss_type(
            train_config.get("pooled_loss_type", default_loss_type),
            name="pooled_loss_type",
        )
        fusion_loss_type = normalize_loss_type(
            train_config.get("fusion_loss_type", default_loss_type),
            name="fusion_loss_type",
        )
        time_score_mode = score_mode_for_loss_type(time_loss_type)
        pooled_score_mode = score_mode_for_loss_type(pooled_loss_type)
        fusion_score_mode = score_mode_for_loss_type(fusion_loss_type)
        split_seed_offsets = {"val": 1, "test": 2}
        # 先导出真实 EEG；启用时再以相同模型和 split 专属固定种子导出高斯噪声基线。
        for split in export_splits:
            export_target_outputs(
                model=model,
                loader=loaders[split],
                task=args.task,
                device=device,
                amp=amp,
                output_dir=output_dir,
                split=split,
                attention_indices=attention_indices,
                time_s=datasets[split].time_s,
                progress_bar_enabled=bool(train_config.get("progress_bar", True)),
                input_kind="real",
                activation_threshold=activation_threshold,
                time_score_mode=time_score_mode,
                pooled_score_mode=pooled_score_mode,
                fusion_score_mode=fusion_score_mode,
                min_cell_class_count=int(
                    data_config.get("eval_min_cell_class_count", 1)
                ),
            )
            if gaussian_enabled:
                export_target_outputs(
                    model=model,
                    loader=loaders[split],
                    task=args.task,
                    device=device,
                    amp=amp,
                    output_dir=output_dir,
                    split=split,
                    attention_indices=attention_indices,
                    time_s=datasets[split].time_s,
                    progress_bar_enabled=bool(train_config.get("progress_bar", True)),
                    input_kind="gaussian",
                    gaussian_seed=gaussian_seed_base + split_seed_offsets.get(split, 0),
                    gaussian_noise_mode=gaussian_noise_mode,
                    activation_threshold=activation_threshold,
                    time_score_mode=time_score_mode,
                    pooled_score_mode=pooled_score_mode,
                    fusion_score_mode=fusion_score_mode,
                    min_cell_class_count=int(
                        data_config.get("eval_min_cell_class_count", 1)
                    ),
                )
        if bool(plots_config.get("enabled", True)):
            figure_dir_config = plots_config.get("output_dir")
            figure_dir = Path(figure_dir_config).expanduser().resolve() if figure_dir_config else output_dir / "figures"
            plot_target_run(
                output_dir,
                splits=export_splits,
                output_dir=figure_dir,
                plot_config=plots_config,
            )
    finally:
        logger.finish()
    del model, loaders, datasets, attention_indices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
