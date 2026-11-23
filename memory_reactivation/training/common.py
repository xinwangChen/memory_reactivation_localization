"""各训练任务共用的优化器调度、设备搬运、结果落盘与 checkpoint 工具。"""

from __future__ import annotations

import copy
import csv
import os
import pickle
import random
import tempfile
from math import cos, isfinite, pi
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm



T = TypeVar("T")


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict:
    """加载训练 checkpoint，并兼容新旧两种保存格式。

    新 checkpoint 的 numpy RNG 状态已由 :func:`save_checkpoint` 转为
    weights_only 安全结构；旧 checkpoint 仍含 ndarray，weights_only 加载会
    抛 ``UnpicklingError``。这里保持 ``weights_only=True`` 的默认安全行为，
    仅在明确遇到该错误时回退为完整反序列化。
    """
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except pickle.UnpicklingError:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Checkpoint {path} must contain a mapping")
    return payload


def progress_bar(
    iterable: Iterable[T],
    *,
    desc: str,
    enabled: bool,
    total: int | None = None,
):
    """创建可由 YAML 关闭的终端进度条。"""
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        disable=not enabled,
        dynamic_ncols=True,
        leave=True,
    )


def move_batch(
    batch: Mapping[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """将一个字典 batch 中的全部张量移动到训练设备。"""
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def trainable_parameters(model: torch.nn.Module) -> Iterable[torch.nn.Parameter]:
    """仅返回 requires_grad=True 的参数，适配冻结编码器等微调模式。"""
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def discriminative_param_groups(
    model: torch.nn.Module,
    backbone_modules: Iterable[torch.nn.Module],
    backbone_lr_scale: float,
) -> list[dict]:
    """把主干和读出层拆成两个参数组，主干用更小的 lr_scale 实现差异化学习率。

    仅纳入 requires_grad=True 的参数；若主干被冻结（linear_probe/adapter_tune），
    主干组为空时自动省略，等价于原来的单组行为。`set_optimizer_learning_rate`
    每步会用 base_lr * lr_scale 刷新各组 LR，因此两组共享同一条调度曲线。
    """
    if not 0.0 < backbone_lr_scale <= 1.0:
        raise ValueError("backbone_lr_scale must be in (0, 1]")
    backbone_param_ids = set()
    backbone_params: list[torch.nn.Parameter] = []
    for module in backbone_modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in backbone_param_ids:
                backbone_param_ids.add(id(parameter))
                backbone_params.append(parameter)
    head_params = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_param_ids
    ]
    groups: list[dict] = []
    if head_params:
        groups.append({"params": head_params, "lr_scale": 1.0})
    if backbone_params:
        groups.append({"params": backbone_params, "lr_scale": float(backbone_lr_scale)})
    if not groups:
        raise ValueError("No trainable parameters found for the optimizer")
    return groups


def gradient_norm(model: torch.nn.Module) -> float:
    """计算全模型梯度的 L2 范数，用于监控 AMP 溢出和训练稳定性。"""
    squared_sum = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_sum += float(parameter.grad.detach().norm(2).item()) ** 2
    return squared_sum**0.5


def scheduled_learning_rate(
    train_config: Mapping[str, object],
    epoch: int,
    epochs: int,
) -> float:
    """根据配置返回当前 epoch 的学习率；默认保持恒定，配置后支持 warmup + cosine。"""
    base_lr = float(train_config["learning_rate"])
    schedule = str(train_config.get("lr_schedule", "constant"))
    if schedule in {"constant", "none"}:
        return base_lr
    if schedule != "warmup_cosine":
        raise ValueError(f"Unknown lr_schedule: {schedule}")

    warmup_epochs = int(train_config.get("warmup_epochs", 0))
    min_lr = float(train_config.get("min_learning_rate", train_config.get("min_lr", 0.0)))
    if epoch < 1 or epoch > epochs:
        raise ValueError(f"epoch must be in [1, {epochs}], got {epoch}")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")
    if min_lr < 0.0 or min_lr > base_lr:
        raise ValueError("min_learning_rate must be between 0 and learning_rate")

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_lr * epoch / warmup_epochs

    decay_epochs = max(1, epochs - warmup_epochs)
    decay_progress = min(1.0, max(0.0, (epoch - warmup_epochs) / decay_epochs))
    cosine_factor = 0.5 * (1.0 + cos(pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


def resolve_learning_rate_schedule(
    train_config: Mapping[str, object],
    saved_train_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Pin implicit cosine duration once; extending training keeps that duration.

    Old checkpoints encoded the duration through train.iterations. An explicit
    conflicting duration is retained here so resume identity validation rejects it.
    """
    resolved = dict(train_config)
    if resolved.get("lr_schedule", "constant") == "warmup_cosine" and "lr_schedule_iterations" not in resolved:
        source = saved_train_config if saved_train_config and saved_train_config.get("lr_schedule") == "warmup_cosine" else resolved
        resolved["lr_schedule_iterations"] = source.get("lr_schedule_iterations", source["iterations"])
    return resolved


def scheduled_iteration_learning_rate(
    train_config: Mapping[str, object],
    iteration: int,
    iterations: int,
) -> float:
    """按 optimizer iteration 计算学习率。

    warmup_cosine 先从 0 线性升到基础学习率，再余弦衰减到
    min_learning_rate；返回的是基础学习率，参数组自身的 lr_scale
    由 :func: set_optimizer_learning_rate 继续应用。
    """
    base_lr = float(train_config["learning_rate"])
    schedule = str(train_config.get("lr_schedule", "constant"))
    if schedule in {"constant", "none"}:
        return base_lr
    if schedule != "warmup_cosine":
        raise ValueError(f"Unknown lr_schedule: {schedule}")

    warmup_iterations = int(train_config.get("warmup_iterations", 0))
    schedule_iterations = int(
        train_config.get("lr_schedule_iterations", iterations)
    )
    min_lr = float(train_config.get("min_learning_rate", train_config.get("min_lr", 0.0)))
    if iteration < 1 or iteration > iterations:
        raise ValueError(f"iteration must be in [1, {iterations}], got {iteration}")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    if not 1 <= schedule_iterations <= iterations:
        raise ValueError(
            "lr_schedule_iterations must be between 1 and train.iterations"
        )
    if warmup_iterations >= schedule_iterations:
        raise ValueError(
            "warmup_iterations must be smaller than lr_schedule_iterations"
        )
    if min_lr < 0.0 or min_lr > base_lr:
        raise ValueError("min_learning_rate must be between 0 and learning_rate")

    scheduled_iteration = min(iteration, schedule_iterations)
    if warmup_iterations > 0 and scheduled_iteration <= warmup_iterations:
        return base_lr * scheduled_iteration / warmup_iterations

    decay_iterations = schedule_iterations - warmup_iterations
    decay_progress = min(
        1.0,
        max(0.0, (scheduled_iteration - warmup_iterations) / decay_iterations),
    )
    cosine_factor = 0.5 * (1.0 + cos(pi * decay_progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


def set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """按 base_lr * 组内 lr_scale 设置每个参数组的学习率。

    没有显式 lr_scale 的参数组按 1.0 处理，因此单组优化器（如预训练）行为不变；
    微调时可给主干参数组设置 <1 的 lr_scale，实现差异化学习率（discriminative fine-tuning）。
    """
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = float(learning_rate) * scale


def optimizer_learning_rate_metrics(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    """Mixed optimizer logs keep both actual rates; historical rows stay unchanged."""
    if not any(group.get("algorithm") == "muon" for group in optimizer.param_groups):
        return {}
    return {
        f"{algorithm}_learning_rate": max(float(group["lr"]) for group in optimizer.param_groups
            if group.get("algorithm") == algorithm)
        for algorithm in ("muon", "adamw")
        if any(group.get("algorithm") == algorithm for group in optimizer.param_groups)
    }


def scheduled_iteration_weight_decay(
    train_config: Mapping[str, object],
    iteration: int,
    iterations: int,
) -> float | None:
    """返回当前 iteration 的 weight decay；未启用调度时返回 None。

    warmup_linear 在 wd_warmup_iterations 内从 min_weight_decay
    线性增加到 weight_decay，之后保持不变。返回 None 的含义是调用者
    不需要改写优化器中初始化好的固定 weight decay。
    """
    schedule = str(train_config.get("wd_schedule", "constant"))
    if schedule in {"constant", "none"}:
        return None
    if schedule != "warmup_linear":
        raise ValueError(f"Unknown wd_schedule: {schedule}; expected warmup_linear")
    base_wd = float(train_config.get("weight_decay", 0.0))
    min_wd = float(train_config.get("min_weight_decay", 0.0))
    warmup_iters = int(train_config.get("wd_warmup_iterations", 0))
    if warmup_iters <= 0 or iteration >= warmup_iters:
        return base_wd
    return min_wd + (base_wd - min_wd) * (iteration / warmup_iters)


def set_optimizer_weight_decay(optimizer: torch.optim.Optimizer, weight_decay: float) -> None:
    """把同一个 weight decay 写入优化器的全部参数组。"""
    for group in optimizer.param_groups:
        group["weight_decay"] = float(weight_decay)


def build_optimizer(
    params: Iterable[torch.nn.Parameter] | Sequence[dict],
    train_config: Mapping[str, object],
    *,
    model: torch.nn.Module | None = None,
) -> torch.optim.Optimizer:
    """按 train.optimizer 构建优化器；缺省为 adamw，与旧配置行为完全一致。

    - ``adamw`` (默认)：仅读 learning_rate / weight_decay，与之前的
      ``torch.optim.AdamW(params, lr=..., weight_decay=...)`` 逐字等价。
    - ``sgd``：读 learning_rate / weight_decay，另支持 momentum(默认 0.9)、
      nesterov(默认 False)、dampening(默认 0.0)。
    - ``muon``：隐藏二维矩阵用 Muon，其余参数使用 AdamW；需要 model= 和
      显式 train.muon.learning_rate，两类状态统一保存并共享 AMP 溢出跳过。
    - ``signum``：所有可训练参数使用梯度EMA的符号更新；读momentum（默认0.9），
      weight_decay采用与AdamW一致的解耦衰减，兼容现有AMP/调度/恢复接口。

    ``params`` 可以是普通参数序列，也可以是带 ``lr_scale`` 的参数组；逐步的
    lr / weight_decay 调度仍由 :func:`set_optimizer_learning_rate` 与
    :func:`set_optimizer_weight_decay` 负责；Muon 用 lr_scale 保持独立的学习率比例。
    """
    name = str(train_config.get("optimizer", "adamw")).lower()
    learning_rate = float(train_config["learning_rate"])
    weight_decay = float(train_config.get("weight_decay", 0.0))
    if name == "adamw":
        return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if name == "muon":
        from .muon import build_muon_optimizer
        return build_muon_optimizer(params, train_config, model)
    if name == "signum":
        from .signum import Signum
        return Signum(params, lr=learning_rate, weight_decay=weight_decay,
                      momentum=float(train_config.get("momentum", 0.9)))
    if name == "sgd":
        momentum = float(train_config.get("momentum", 0.9))
        dampening = float(train_config.get("dampening", 0.0))
        nesterov = train_config.get("nesterov", False)
        if not isinstance(nesterov, bool):
            raise ValueError("train.nesterov must be a boolean")
        if nesterov and (momentum <= 0.0 or dampening != 0.0):
            raise ValueError(
                "nesterov momentum requires momentum > 0 and dampening == 0"
            )
        return torch.optim.SGD(
            params,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=dampening,
            nesterov=nesterov,
        )
    raise ValueError(f"Unknown optimizer: {name!r}; expected 'adamw', 'sgd', 'muon' or 'signum'")


def append_metrics_csv(path: Path, row: Mapping[str, object]) -> None:
    """按列名追加指标；新增列时原子扩展旧表，历史缺失值留空。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    fieldnames = list(row)
    if not write_header:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fields = reader.fieldnames
            if not existing_fields or len(set(existing_fields)) != len(existing_fields):
                raise ValueError(f"{path} has an invalid metrics header")
            added_fields = [name for name in row if name not in existing_fields]
            fieldnames = [*existing_fields, *added_fields]
            if added_fields:
                # Only schema changes rewrite the log. Normal appends read just
                # the header, so long training runs do not become quadratic.
                fd, temp_name = tempfile.mkstemp(
                    prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
                )
                try:
                    with os.fdopen(fd, "w", newline="", encoding="utf-8") as output:
                        writer = csv.DictWriter(output, fieldnames=fieldnames)
                        writer.writeheader()
                        for previous in reader:
                            if None in previous or any(value is None for value in previous.values()):
                                raise ValueError(
                                    f"{path} contains a metrics row that does not match its header"
                                )
                            writer.writerow(previous)
                        writer.writerow(row)
                    os.replace(temp_name, path)
                except BaseException:
                    try:
                        os.unlink(temp_name)
                    except FileNotFoundError:
                        pass
                    raise
                return
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def truncate_metrics_csv(path: Path, max_iteration: int) -> None:
    """Drop rows written after a checkpoint iteration."""
    path = Path(path)
    if not path.exists():
        return
    cutoff = int(max_iteration)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames or "iteration" not in fieldnames:
            return
        rows: list[dict[str, str]] = []
        for row in reader:
            raw_iteration = row.get("iteration")
            if raw_iteration is None or not raw_iteration.strip():
                raise ValueError(f"{path} contains a row without iteration")
            try:
                iteration = float(raw_iteration)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path} contains a non-numeric iteration: {raw_iteration!r}"
                ) from exc
            if not np.isfinite(iteration):
                raise ValueError(f"{path} contains a non-finite iteration")
            if iteration <= cutoff:
                rows.append(row)
    with path.open("r", newline="", encoding="utf-8") as handle:
        existing_count = max(0, sum(1 for _ in handle) - 1)
    if len(rows) == existing_count:
        return
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def write_rows_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """一次性写出结构化结果表，例如第二阶段 evidence 时间曲线。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_loss_curves(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """保存训练与验证损失曲线，并在指标存在时绘制 baseline。"""
    if not rows:
        return
    x_key = "iteration" if "iteration" in rows[0] else "epoch"
    steps = [int(row[x_key]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(steps, [float(row["train_loss"]) for row in rows], label="train")
    ax.plot(steps, [float(row["val_loss"]) for row in rows], label="validation")
    if "val_baseline_loss" in rows[0]:
        ax.plot(
            steps,
            [float(row["val_baseline_loss"]) for row in rows],
            label="baseline",
            linestyle="--",
        )
    ax.set_xlabel("Iteration" if x_key == "iteration" else "Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def atomic_torch_save(payload: object, path: Path) -> None:
    """Replace a checkpoint only after its same-directory temporary file is complete.

    Serialization, flush or replacement failures leave the previous checkpoint
    untouched. A process killed before replacement may leave a temporary file,
    but readers continue to see the previous complete checkpoint.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Mapping[str, object],
    metrics: Mapping[str, object],
    iteration: int | None = None,
    scaler: torch.amp.GradScaler | None = None,
    extra_state: Mapping[str, object] | None = None,
    selection_metric: str | None = None,
    selection_mode: str | None = None,
) -> None:
    """保存可恢复训练的 checkpoint：配置、指标、模型、优化器、AMP 与 RNG 状态。

    只保存 optimizer 与 model 无法真正 resume（AMP 的 loss scale、各 RNG 与
    数据 epoch 都会从默认值重新开始）；本函数把这些状态一并冻结。
    """
    from ..artifacts.provenance import runtime_implementation_identity
    implementation = runtime_implementation_identity()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "pb2023.checkpoint.v1",
        "implementation_sha256": implementation["sha256"],
        "implementation_identity_schema": implementation["schema"],
        "model_spec": getattr(model, "_pb_model_spec", None),
        "epoch": epoch,
        "config": dict(config),
        "metrics": dict(metrics),
        # Keep CV identity at the checkpoint top level so it remains visible even
        # to lightweight checkpoint inspection tools that do not parse config.
        "cv_fold": config.get("cv_fold"),
        "cv_folds": config.get("cv_folds"),
        "cv_split_dir": config.get("cv_split_dir"),
        "cv_split_sha256": config.get("cv_split_sha256"),
        "cv_val_blocks": config.get("cv_val_blocks"),
        "graph_laplacian_sha256": (config.get("data") or {}).get("graph_laplacian_sha256"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_states": {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda": (
                [
                    torch.cuda.get_rng_state(device_index)
                    for device_index in range(torch.cuda.device_count())
                ]
                if torch.cuda.is_available()
                else []
            ),
            "numpy": _serialize_numpy_rng_state(np.random.get_state()),
            "python": random.getstate(),
        },
    }
    for key in ("cv_strategy", "cv_train_subjects", "cv_val_subjects"):
        if key in config:
            payload[key] = config[key]
    if extra_state is not None:
        payload["extra_state"] = dict(extra_state)
    if iteration is not None:
        payload["iteration"] = iteration
    if selection_metric is not None:
        payload["selection_metric"] = selection_metric
        payload["selection_mode"] = selection_mode or "max"
        if selection_metric in metrics:
            payload["selection_score"] = float(metrics[selection_metric])
    atomic_torch_save(payload, path)


def recover_best_checkpoint(resume_from: Path, output_dir: Path, *, metric: str, mode: str,
                            expected_config: Mapping[str, object] | None = None) -> bool:
    """Restore the recorded best using weights available at the resume step."""
    saved = load_checkpoint(resume_from)
    best = float(saved.get('extra_state', {}).get('best_score', float('nan')))
    resume_iteration = int(saved.get('iteration', 0))
    destination = Path(output_dir) / 'best_val.ckpt'
    if destination.exists():
        existing = load_checkpoint(destination)
        # Both checkpoints may carry different historical paths after repeated
        # directory moves. Compare against the validated current artifact paths.
        validate_resume_config(existing.get('config', {}),
                               saved.get('config', {}) if expected_config is None else expected_config,
                               run_dir=output_dir)
        if existing.get('selection_metric', metric) != metric or existing.get('selection_mode', mode) != mode:
            raise ValueError('Best checkpoint selection identity differs from resume checkpoint')
        score = float(existing.get('metrics', {}).get(metric, float('nan')))
        existing_iteration = int(existing.get('iteration', 0))
        if isfinite(best) and score == best and existing_iteration <= resume_iteration:
            return False
        if not isfinite(best):
            if destination.resolve() == Path(resume_from).resolve():
                raise ValueError('Resume best checkpoint has no recorded finite best score')
            # Selection warmup can be rewound to a step before any eligible best.
            destination.unlink()
            return True
    if not isfinite(best):
        return False
    candidate = float(saved.get('metrics', {}).get(metric, float('nan')))
    if candidate != best:
        raise ValueError('Best checkpoint is missing, stale or newer than the resume step, '
                         'and resume weights are not the recorded best; '
                         'restore the matching best_val.ckpt at or before the resume step')
    atomic_torch_save(saved, destination)
    return True


def _serialize_numpy_rng_state(state: object) -> object:
    """把 numpy RNG 状态转为 torch.load(weights_only=True) 可加载的结构。

    ``np.random.get_state()`` 的元组内含 ndarray，weights_only 模式会拒绝；
    递归地把其中的 ndarray 替换为 torch tensor，其余原样保留。恢复时由
    :func:`_deserialize_numpy_rng_state` 转回 ``np.random.set_state`` 接受的格式。
    """
    if isinstance(state, tuple):
        return tuple(_serialize_numpy_rng_state(item) for item in state)
    if isinstance(state, list):
        return [_serialize_numpy_rng_state(item) for item in state]
    if isinstance(state, np.ndarray):
        return torch.as_tensor(state)
    return state


def _deserialize_numpy_rng_state(state: object) -> object:
    """把序列化后的 numpy RNG 状态还原为原生格式。

    兼容两种输入：新 checkpoint 中的 torch tensor（由
    :func:`_serialize_numpy_rng_state` 写入）以及旧 checkpoint 中的 ndarray。
    """
    if isinstance(state, tuple):
        return tuple(_deserialize_numpy_rng_state(item) for item in state)
    if isinstance(state, list):
        return [_deserialize_numpy_rng_state(item) for item in state]
    if isinstance(state, torch.Tensor):
        return state.detach().cpu().numpy()
    return state


def restore_rng_states(rng_states: Mapping[str, object] | None) -> None:
    """恢复保存的 torch/numpy/python RNG 状态。

    调用方必须确保此时训练环境（DataLoader 迭代器、sampler、skip 游标等）
    已经完全重建——创建迭代器/取数据会消耗 torch RNG，若先恢复再重建环境，
    恢复后的随机序列会被环境重建消耗掉，训练轨迹（dropout、噪声增强）偏移。
    """
    if not rng_states:
        return
    if "torch_cpu" in rng_states:
        torch.random.set_rng_state(rng_states["torch_cpu"])
    cuda_states = rng_states.get("torch_cuda")
    if cuda_states and torch.cuda.is_available():
        for device_index, state in enumerate(cuda_states):
            torch.cuda.set_rng_state(state, device_index)
    if "numpy" in rng_states:
        np.random.set_state(_deserialize_numpy_rng_state(rng_states["numpy"]))
    if "python" in rng_states:
        random.setstate(rng_states["python"])


def _resume_config_identity(config: Mapping[str, object]) -> dict[str, object]:
    """Return the part of a run config that determines the training trajectory."""

    def normalize(value: object) -> object:
        if isinstance(value, Mapping):
            return {str(key): normalize(item) for key, item in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [normalize(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return copy.deepcopy(value)

    identity = normalize(config)
    assert isinstance(identity, dict)
    # An explicitly disabled optional preprocessor is equivalent to old configs.
    from memory_reactivation.data.spatial import graph_options, normalize_graph_input_mode
    options = graph_options(identity.get("preprocessing"))
    if not options["enabled"]:
        identity.pop("preprocessing", None)
    else:
        identity["preprocessing"] = {"graph_laplacian": options}
    data = identity.get("data")
    if isinstance(data, dict) and data.get("graph_laplacian_sha256"):
        # Pre-concat snapshots omitted this field and always used replace mode.
        data["graph_laplacian_mode"] = normalize_graph_input_mode(
            data.get("graph_laplacian_mode")
        )
    for key in ("output_dir", "output_root", "run", "swanlab", "plots"):
        identity.pop(key, None)
    model = identity.get("model")
    if isinstance(model, dict):
        from ..models.encoder import normalize_boundary_mode, normalize_boundary_noise_seed
        # Explicit defaults preserve historical resume identity; an active seed
        # affects validation/early stopping and must match the saved run.
        for key in ("encoder", "resolved_encoder"):
            encoder = model.get(key)
            if isinstance(encoder, dict):
                boundary = normalize_boundary_mode(encoder.get("boundary_mode", "truncate"))
                seed = normalize_boundary_noise_seed(encoder.get("boundary_noise_seed", 42))
                if boundary == "truncate":
                    encoder.pop("boundary_mode", None)
                else:
                    encoder["boundary_mode"] = boundary
                if boundary == "noise_pad":
                    encoder["boundary_noise_seed"] = seed
                else:
                    encoder.pop("boundary_noise_seed", None)
        if model.get("attention_score_centering") is False:
            # Missing and explicitly disabled both preserve the historical model.
            model.pop("attention_score_centering", None)
        if model.get("attention_gate_centering") is False:
            model.pop("attention_gate_centering", None)
        # Explicit historical gate defaults do not change resume semantics.
        for key, default in (("attention_gate_activation", "sigmoid"),
                             ("attention_gate_combination", "multiply"),
                             ("attention_effective_normalization", "none")):
            if model.get(key) == default:
                model.pop(key)
        subject = model.get("subject_conditioning")
        if isinstance(subject, dict) and subject.get("mode", "individual") == "individual":
            # Explicitly naming the historical default preserves old resume identity.
            subject.pop("mode", None)
    train = identity.get("train")
    if isinstance(train, dict):
        train.update(resolve_learning_rate_schedule(train))
        from .attention_regularization import (
            normalize_attention_mass_penalty_config,
            normalize_clean_attention_penalty_config,
        )
        attention_mass = normalize_attention_mass_penalty_config(train)
        train.pop("attention_mass_penalty_weight", None)
        if attention_mass["weight"] > 0.0:
            train["attention_mass_penalty_weight"] = attention_mass["weight"]
        clean_attention = normalize_clean_attention_penalty_config(train)
        train.pop("clean_attention_penalty_weight", None)
        train.pop("clean_attention_penalty_probability", None)
        if clean_attention["weight"] > 0.0 and clean_attention["probability"] > 0.0:
            train["clean_attention_penalty_weight"] = clean_attention["weight"]
            train["clean_attention_penalty_probability"] = clean_attention["probability"]
        from .augmentation import normalize_augmentation_pipeline_config, normalize_near_full_noise_config
        pipeline = normalize_augmentation_pipeline_config(train.get("augmentation_pipeline"))
        if pipeline["enabled"]:
            train["augmentation_pipeline"] = pipeline
        else:
            train.pop("augmentation_pipeline", None)
        near_full_noise = normalize_near_full_noise_config(train.get("near_full_noise"))
        if not near_full_noise["enabled"] or float(near_full_noise["probability"]) == 0.0:
            # Missing/disabled/zero-probability preserve the original trajectory.
            train.pop("near_full_noise", None)
        else:
            train["near_full_noise"] = near_full_noise
        # Extending a stopped run and changing terminal presentation are safe; all
        # optimizer, schedule, validation and augmentation fields remain strict.
        train.pop("iterations", None)
        train.pop("progress_bar", None)
    return identity


def _config_difference_paths(
    saved: object, current: object, prefix: str = "config"
) -> list[str]:
    """Collect a concise list of differing nested config paths."""
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        differences: list[str] = []
        for key in sorted(set(saved) | set(current), key=str):
            path = f"{prefix}.{key}"
            if key not in saved or key not in current:
                differences.append(path)
            else:
                differences.extend(
                    _config_difference_paths(saved[key], current[key], path)
                )
        return differences
    if isinstance(saved, list) and isinstance(current, list):
        if len(saved) != len(current):
            return [prefix]
        differences = []
        for index, (saved_item, current_item) in enumerate(zip(saved, current)):
            differences.extend(
                _config_difference_paths(
                    saved_item, current_item, f"{prefix}[{index}]"
                )
            )
        return differences
    return [] if saved == current else [prefix]


def validate_resume_config(
    saved_config: Mapping[str, object] | None,
    current_config: Mapping[str, object],
    *,
    run_dir: Path | None = None,
) -> None:
    """Reject resume attempts that change model, data or training semantics."""
    if not saved_config:
        return
    saved_identity = _resume_config_identity(saved_config)
    current_identity = _resume_config_identity(current_config)
    saved_data = saved_identity.get("data")
    current_data = current_identity.get("data")
    if run_dir is not None and isinstance(saved_data, dict) and isinstance(current_data, dict):
        from memory_reactivation.utils.run import (
            data_artifact_keys, resolve_snapshot_data_artifacts,
            validate_snapshot_data_artifacts,
        )
        changed_paths = [key for key in data_artifact_keys(saved_data)
                         if saved_data.get(key) != current_data.get(key)]
        if all(saved_data.get(key) for key in data_artifact_keys(saved_data)):
            # Unchanged paths can still point to replaced EEG/behavior files.
            resolved = resolve_snapshot_data_artifacts(saved_data, run_dir)
            validate_snapshot_data_artifacts(
                resolved, run_dir, required=saved_identity.get("cv_fold") is not None,
                expected_split_sha256=saved_identity.get("cv_split_sha256"),
            )
        if changed_paths:
            # A directory move may change paths, but never the frozen inputs.
            relocated = resolve_snapshot_data_artifacts(saved_data, run_dir)
            validate_snapshot_data_artifacts(
                relocated, run_dir, required=True,
                expected_split_sha256=saved_identity.get("cv_split_sha256"),
            )
            for key in changed_paths:
                if relocated[key] == current_data.get(key):
                    saved_data[key] = relocated[key]
    differences = _config_difference_paths(saved_identity, current_identity)
    if differences:
        preview = ", ".join(differences[:12])
        if len(differences) > 12:
            preview += f", ... ({len(differences)} paths total)"
        raise ValueError(
            "resume config differs from checkpoint training identity; only "
            "train.iterations, output/logging metadata and verified artifact paths may change. "
            f"Differing paths: {preview}"
        )


def validate_checkpoint_cv_identity(
    checkpoint: Mapping[str, object], expected_config: Mapping[str, object],
) -> None:
    """Reject resume/pretraining weights that saw a different held-out fold."""
    saved_config = checkpoint.get("config") or {}
    if not isinstance(saved_config, Mapping):
        raise ValueError("checkpoint has invalid config metadata")
    from memory_reactivation.data.manifest import normalize_cv_strategy
    from memory_reactivation.data.normalization import normalize_normalization_mode

    def protocol(config):
        data = config.get("data") or {}
        strategy = normalize_cv_strategy(config.get("cv_strategy", data.get("cv_strategy", "block")))
        if "cv_strategy" in data and normalize_cv_strategy(data["cv_strategy"]) != strategy:
            raise ValueError("Conflicting checkpoint CV strategy metadata")
        return strategy, normalize_normalization_mode(data.get("normalization_mode", "per_subject"))

    saved_strategy, saved_normalization = protocol(saved_config)
    expected_strategy, expected_normalization = protocol(expected_config)
    if (saved_strategy != expected_strategy
            or checkpoint.get("cv_strategy", saved_strategy) != expected_strategy):
        raise ValueError("Checkpoint CV strategy mismatch")
    if saved_normalization != expected_normalization:
        raise ValueError("Checkpoint normalization mode mismatch")
    saved_graph = checkpoint.get(
        "graph_laplacian_sha256",
        (saved_config.get("data") or {}).get("graph_laplacian_sha256"),
    )
    expected_graph = (expected_config.get("data") or {}).get("graph_laplacian_sha256")
    if saved_graph != expected_graph:
        raise ValueError("Checkpoint graph Laplacian preprocessing mismatch")
    if saved_graph is not None:
        from memory_reactivation.data.spatial import normalize_graph_input_mode
        saved_mode = normalize_graph_input_mode(
            (saved_config.get("data") or {}).get("graph_laplacian_mode")
        )
        expected_mode = normalize_graph_input_mode(
            (expected_config.get("data") or {}).get("graph_laplacian_mode")
        )
        if saved_mode != expected_mode:
            raise ValueError("Checkpoint graph Laplacian input mode mismatch")
    if expected_config.get("cv_fold") is None:
        return
    saved_config = checkpoint.get("config") or {}
    if not isinstance(saved_config, Mapping):
        raise ValueError("checkpoint has invalid config metadata")
    for key in ("cv_fold", "cv_folds", "cv_split_sha256", "cv_train_subjects", "cv_val_subjects"):
        expected = expected_config.get(key)
        actual = checkpoint.get(key, saved_config.get(key))
        if expected is not None and actual != expected:
            raise ValueError(f"Checkpoint CV identity mismatch: {key}={actual!r}, expected {expected!r}")


def load_training_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    restore_rng: bool = False,
    expected_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """加载训练 checkpoint 并恢复 model/optimizer/scaler 状态。

    默认不恢复 RNG：调用方应在重建训练环境（DataLoader iterator、epoch 游标、
    skip）之后再显式调用 :func:`restore_rng_states`，避免环境重建消耗已恢复的
    RNG 状态。返回 payload 元数据（epoch、iteration、config、metrics、
    extra_state、rng_states）。
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid training checkpoint: {path}")
    checkpoint_config = payload.get("config", {})
    if expected_config is not None:
        validate_checkpoint_cv_identity(payload, expected_config)
        if checkpoint_config is not None and not isinstance(checkpoint_config, Mapping):
            raise ValueError(f"checkpoint has invalid config metadata: {path}")
        validate_resume_config(checkpoint_config, expected_config, run_dir=path.resolve().parent)
    if model is not None:
        model_state = payload.get("model_state")
        if model_state is None:
            raise ValueError(f"checkpoint has no model_state: {path}")
        missing, unexpected = model.load_state_dict(
            model_state, strict=False
        )
        if unexpected:
            raise ValueError(
                f"checkpoint model_state has unexpected keys: {unexpected[:10]}"
            )
        if missing:
            raise ValueError(
                f"checkpoint model_state is missing keys: {missing[:10]}"
            )
    if optimizer is not None:
        optimizer_state = payload.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
    if scaler is not None:
        scaler_state = payload.get("scaler_state")
        if scaler_state and scaler.is_enabled():
            scaler.load_state_dict(scaler_state)
    rng_states = payload.get("rng_states")
    if restore_rng:
        restore_rng_states(rng_states)
    return {
        "epoch": int(payload.get("epoch", 0)),
        "iteration": int(payload.get("iteration", 0)),
        "config": dict(checkpoint_config or {}),
        "metrics": dict(payload.get("metrics", {})),
        "extra_state": dict(payload.get("extra_state", {})),
        "rng_states": rng_states if isinstance(rng_states, dict) else None,
    }
