"""从已有 target-decoder 运行目录重新导出评估结果并绘图。

入口读取配置快照与最优检查点，重建所选验证/测试数据集，并兼容推断历史检查点中
分类器层数、隐藏维度及 time-level ``no_class`` 输出宽度。随后可导出真实 EEG、
高斯噪声、逐 trial 时间置乱和固定 circular-roll 的 CSV，再由统一绘图入口为每种条件
生成独立完整图树；不会重新训练或改写检查点。
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch
from pb2023_eeg.artifacts.checkpoints import load_model
from pb2023_eeg.artifacts.compatibility import _checkpoint_time_class_count, resolve_checkpoint_n_subjects, resolve_model_config_for_checkpoint

from pb2023_eeg.data.dataset import normalize_lags
from pb2023_eeg.data.manifest import load_manifest
from pb2023_eeg.models.target_decoder import TASK_CLASS_COUNTS
from pb2023_eeg.training.common import load_checkpoint, validate_checkpoint_cv_identity, validate_resume_config
from pb2023_eeg.training.export_integrity import (
    TARGET_EXPORT_STEMS,
    copy_verified_exports,
    describe_export,
)
from pb2023_eeg.training.target import (
    EVALUATION_INPUT_KINDS,
    export_target_outputs,
    normalize_loss_type,
    score_mode_for_loss_type,
)
from pb2023_eeg.utils.config import load_yaml
from pb2023_eeg.utils.run import validate_snapshot_data_artifacts
from pb2023_eeg.utils.seed import seed_everything
from pb2023_eeg.workflows.report import (
    plot_target_conditions,
    plot_target_run,
    resolve_plot_config,
    resolve_plot_time_axis_limits,
)
from pb2023_eeg.workflows.train import (
    build_model,
    make_loader,
    resolve_attention_window,
    resolve_dataset_input_channels,
    resolve_dataset_time_axis,
    resolve_dataset_task_class_indices,
    resolve_export_splits,
    resolve_target_export_checkpoint,
    resolve_model_task_class_indices,
    resolved_time_axis_metadata,
    validate_independent_time_heads_cover_attention_window,
)
from pb2023_eeg.data.subject_conditioning import validate_conditioning_subject_namespace
from pb2023_eeg.data.configured import build_lag_dataset, validate_data_protocol, validate_subject_cv_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _gaussian_baseline_explicitly_disabled(
    plots_config: Mapping[str, object],
) -> bool:
    """判断配置是否明确写出 ``gaussian_baseline.enabled: false``。

    缺省该段配置与显式关闭含义不同：只有显式关闭才允许清理旧高斯导出文件。
    """
    configured = plots_config.get("gaussian_baseline")
    return (
        isinstance(configured, Mapping)
        and configured.get("enabled") is False
    )


def _remove_stale_gaussian_exports(run_dir: Path, split: str) -> tuple[Path, ...]:
    """删除某个 split 以前生成的全部高斯基线 CSV。

    返回：
        实际删除的路径元组，便于调用方或测试核对清理范围。
    """
    removed: list[Path] = []
    for path in sorted(Path(run_dir).glob(f"{split}_gaussian_*.csv")):
        if path.is_file():
            path.unlink()
            removed.append(path)
    return tuple(removed)


def parse_args(argv=None) -> argparse.Namespace:
    """解析已有运行目录、可选 split 覆盖值和图表输出目录。"""
    parser = argparse.ArgumentParser(
        description="Re-run export and visualization for an existing target-decode run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Explicitly override config plots.splits (allowed: val, test).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override figure output directory.")
    parser.add_argument(
        "--input-kinds",
        nargs="+",
        choices=EVALUATION_INPUT_KINDS,
        help=(
            "Export the requested evaluation inputs and write one complete figure "
            "tree per condition. Omit to preserve the historical real/Gaussian flow."
        ),
    )
    parser.add_argument(
        "--controls-output-dir",
        type=Path,
        help="Root for isolated control CSVs, figures, and provenance metadata.",
    )
    parser.add_argument(
        "--reuse-existing-kinds",
        nargs="+",
        choices=EVALUATION_INPUT_KINDS,
        default=(),
        help="Reuse complete exports with matching parameters and verified provenance; old CSV-only exports must be regenerated.",
    )
    parser.add_argument(
        "--time-shuffle-seed",
        type=int,
        help="Base seed for deterministic per-trial time permutations.",
    )
    parser.add_argument(
        "--time-roll-shift-s",
        type=float,
        default=0.5,
        help="Fixed positive circular time shift in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--time-axis-start-s",
        type=float,
        help="Override plots.time_axis.start_s; requires --time-axis-end-s.",
    )
    parser.add_argument(
        "--time-axis-end-s",
        type=float,
        help="Override plots.time_axis.end_s; requires --time-axis-start-s.",
    )
    return parser.parse_args(argv)


def resolve_reexport_splits(
    requested_splits: Sequence[str] | None,
    plots_config: Mapping[str, object],
    *,
    cv_fold: int | None = None,
) -> tuple[str, ...]:
    """解析重导出 split；命令行未指定时沿用快照中的绘图策略。

    返回值只可能包含去重后的 ``val`` 和/或 ``test``。
    """
    if requested_splits is None:
        return ("val",) if cv_fold is not None else resolve_export_splits(plots_config)
    splits = resolve_export_splits({"splits": requested_splits})
    if cv_fold is not None and splits != ("val",):
        raise ValueError("CV runs have no test split; re-export val only")
    return splits


def normalize_control_input_kinds(requested: Sequence[str]) -> tuple[str, ...]:
    """Validate and de-duplicate explicit evaluation input conditions."""
    normalized = tuple(dict.fromkeys(str(kind) for kind in requested))
    if not normalized:
        raise ValueError("--input-kinds must contain at least one condition")
    unsupported = [kind for kind in normalized if kind not in EVALUATION_INPUT_KINDS]
    if unsupported:
        raise ValueError(f"Unsupported input kinds: {unsupported}")
    return normalized


def resolve_time_roll_shift_samples(
    time_s: Sequence[float] | torch.Tensor,
    shift_s: float,
) -> tuple[int, float]:
    """Convert a fixed circular shift in seconds to an exact sample offset."""
    times = torch.as_tensor(time_s, dtype=torch.float64).reshape(-1)
    if times.numel() < 2:
        raise ValueError("time axis must contain at least two samples")
    if not torch.isfinite(times).all():
        raise ValueError("time axis must be finite")
    shift = float(shift_s)
    if not torch.isfinite(torch.tensor(shift)) or shift <= 0.0:
        raise ValueError("time-roll shift must be a positive finite duration")
    steps = torch.diff(times)
    if torch.any(steps <= 0.0):
        raise ValueError("time axis must be strictly increasing")
    step_s = float(torch.median(steps).item())
    if not torch.allclose(
        steps,
        torch.full_like(steps, step_s),
        rtol=5e-5,
        atol=5e-8,
    ):
        raise ValueError("time-roll export requires a uniformly sampled time axis")
    shift_samples = int(round(shift / step_s))
    if shift_samples <= 0 or shift_samples % int(times.numel()) == 0:
        raise ValueError("time-roll shift becomes a zero circular offset")
    resolved_shift_s = shift_samples * step_s
    if abs(resolved_shift_s - shift) > max(1e-8, step_s * 1e-4):
        raise ValueError(
            f"time-roll shift {shift:g}s is not aligned to the {step_s:g}s sample interval"
        )
    return shift_samples, resolved_shift_s


def resolve_snapshot_artifact_path(
    run_dir: Path, configured: str | Path, snapshot_name: str | None = None,
) -> Path:
    """Resolve a copied run's artifact path without changing the saved snapshot."""
    configured_path = Path(configured).expanduser()
    if configured_path.exists():
        return configured_path
    candidates = (
        *((run_dir / "artifacts" / snapshot_name,) if snapshot_name else ()),
        run_dir / configured_path.name,
        run_dir / "artifacts" / configured_path.name,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Snapshot artifact not found at {configured_path} or copied-run fallbacks {candidates}"
    )


def copy_existing_target_exports(
    run_dir: Path,
    output_dir: Path,
    *,
    splits: Sequence[str],
    input_kinds: Sequence[str],
    expected_parameters: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
) -> tuple[Path, ...]:
    """Validate the entire reuse request before replacing any destination files."""
    return copy_verified_exports(
        run_dir, output_dir, decoder="target", splits=splits, input_kinds=input_kinds,
        expected_parameters=expected_parameters,
    )








def main(argv=None) -> None:
    """恢复 target 模型，对所选 split 重做推理、CSV 导出和绘图。

    输入来自运行目录的配置快照、数据产物路径和 ``best_val.ckpt``；输出覆盖/补充该
    运行目录中的评估 CSV，图表默认写入 ``figures``。模型与数据对象在结束时显式释放。
    """
    args = parse_args(argv)
    control_mode = args.input_kinds is not None
    reuse_existing_kinds = tuple(dict.fromkeys(args.reuse_existing_kinds))
    if control_mode and args.output_dir is not None:
        raise ValueError("Use --controls-output-dir with --input-kinds, not --output-dir")
    if not control_mode and args.controls_output_dir is not None:
        raise ValueError("--controls-output-dir requires --input-kinds")
    if not control_mode and reuse_existing_kinds:
        raise ValueError("--reuse-existing-kinds requires --input-kinds")
    run_dir = args.run_dir.expanduser().resolve()
    config = load_yaml(run_dir / "config_snapshot.yaml")
    checkpoint_path = resolve_target_export_checkpoint(run_dir)
    if checkpoint_path.name != "best_val.ckpt":
        print(f"[reexport-target] best_val.ckpt missing; loading {checkpoint_path.name}", flush=True)
    checkpoint = load_checkpoint(checkpoint_path)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    validate_checkpoint_cv_identity(checkpoint, config)
    validate_resume_config(checkpoint.get("config"), config, run_dir=run_dir)
    task = str(config["task"])
    seed = int(config.get("seed", 20260530))
    seed_everything(seed)
    data_config = dict(config["data"])
    train_config = config["train"]
    assert isinstance(train_config, Mapping)
    plots_config = config.get("plots", {}) or {}
    if not isinstance(plots_config, Mapping):
        raise ValueError("plots must be a mapping when provided")
    effective_plot_config = resolve_plot_config(
        plots_config,
        time_axis_start_s=args.time_axis_start_s,
        time_axis_end_s=args.time_axis_end_s,
    )
    display_time_limits = resolve_plot_time_axis_limits(effective_plot_config)
    splits = resolve_reexport_splits(args.splits, plots_config, cv_fold=config.get("cv_fold"))

    # 复用训练快照中的实验与被试筛选，防止重导出样本集合漂移。
    manifest_csv = resolve_snapshot_artifact_path(run_dir, data_config["manifest_csv"])
    split_csv = resolve_snapshot_artifact_path(run_dir, data_config["split_csv"])
    normalization_json = resolve_snapshot_artifact_path(
        run_dir,
        data_config["normalization_json"],
    )
    if data_config.get("graph_laplacian_json"):
        data_config["graph_laplacian_json"] = str(resolve_snapshot_artifact_path(
            run_dir, data_config["graph_laplacian_json"], "graph_laplacian_json.json"
        ))
    validate_snapshot_data_artifacts(
        {**data_config, "manifest_csv": manifest_csv, "split_csv": split_csv, "normalization_json": normalization_json},
        run_dir, required=config.get("cv_fold") is not None,
        expected_split_sha256=config.get("cv_split_sha256"),
    )
    data_config.update(manifest_csv=str(manifest_csv), split_csv=str(split_csv),
                       normalization_json=str(normalization_json))
    validate_data_protocol(data_config)
    records = load_manifest(
        manifest_csv,
        experiments=[int(e) for e in data_config.get("experiments", [1])],
        subjects=data_config.get("subjects"),
    )
    lags = normalize_lags(data_config.get("lags", [0]))
    data_config["lags"] = lags
    datasets = {
        split: build_lag_dataset(data_config, task=task, split=split, seed=seed, records=records)
        for split in splits
    }

    validate_subject_cv_datasets({**config, "data": data_config}, datasets)
    loaders = {
        split: make_loader(
            dataset,
            batch_size=int(train_config["batch_size"]),
            shuffle=False,
            num_workers=int(data_config.get("num_workers", 0)),
            seed=seed + {"val": 1, "test": 2}.get(split, 0),
            batch_sampling=str(data_config.get("batch_sampling", "global")),
        )
        for split, dataset in datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_n_subjects = (
        max(int(idx) for dataset in datasets.values() for idx in dataset.participant_indices) + 1
    )
    # 独立时间头的数量由真实时间轴长度决定，且所有所选 split 必须逐值一致。
    times = resolve_dataset_time_axis(datasets)
    time_axis_metadata = resolved_time_axis_metadata(times)
    control_input_kinds = (
        normalize_control_input_kinds(args.input_kinds)
        if control_mode
        else ()
    )
    invalid_reuse_kinds = [
        kind for kind in reuse_existing_kinds if kind not in control_input_kinds
    ]
    if invalid_reuse_kinds:
        raise ValueError(
            "--reuse-existing-kinds must be a subset of --input-kinds: "
            f"{invalid_reuse_kinds}"
        )
    time_shuffle_seed_base = int(
        args.time_shuffle_seed
        if args.time_shuffle_seed is not None
        else seed + 200_000
    )
    if time_shuffle_seed_base < 0:
        raise ValueError("--time-shuffle-seed must be non-negative")
    time_roll_shift_samples: int | None = None
    resolved_time_roll_shift_s: float | None = None
    if "time_roll" in control_input_kinds:
        (
            time_roll_shift_samples,
            resolved_time_roll_shift_s,
        ) = resolve_time_roll_shift_samples(times, args.time_roll_shift_s)
    control_root = (
        args.controls_output_dir.expanduser().resolve()
        if args.controls_output_dir is not None
        else run_dir / "eeg_control_replots"
    )
    export_dir = control_root / "exports" if control_mode else run_dir
    figures_root = control_root / "figures" if control_mode else None
    # Use the checkpoint's ModelSpec and validate snapshot-only forward options.
    loaded = load_model(checkpoint_path, config=config, trusted_legacy=True)
    if loaded.spec.family != "target":
        raise ValueError("Target reexport requires a Target checkpoint")
    model = loaded.model
    model_config = loaded.config["model"]
    resolve_checkpoint_n_subjects(dataset_n_subjects, checkpoint["model_state"])
    validate_conditioning_subject_namespace(model_config, datasets)
    task_class_indices = resolve_dataset_task_class_indices(datasets, task=task)
    resolve_model_task_class_indices(model_config, task_class_indices)
    if loaded.spec.input_channels != resolve_dataset_input_channels(datasets):
        raise ValueError("Checkpoint input channels do not match the selected datasets")
    # Legacy checkpoints can omit time metadata. Recorded values remain binding.
    for key, resolved_value in time_axis_metadata.items():
        saved_value = model_config.get(key)
        if saved_value is not None and saved_value != resolved_value:
            raise ValueError(
                f"Saved model.{key}={saved_value!r} does not match "
                f"the selected datasets ({resolved_value!r})"
            )
    model.to(device)
    model.eval()

    start_s, end_s = resolve_attention_window(train_config, times)
    # attention_indices 是完整时间轴上的全局索引，既用于注意力池化也用于导出对齐。
    attention_indices = torch.nonzero(
        (times >= start_s) & (times <= end_s), as_tuple=False
    ).flatten().to(device)
    if attention_indices.numel() == 0:
        raise ValueError("Configured attention/time-loss window contains no EEG samples")
    validate_independent_time_heads_cover_attention_window(
        model_config,
        time_point_count=int(time_axis_metadata["resolved_time_point_count"]),
        attention_indices=attention_indices.detach().cpu(),
    )

    amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    progress_bar_enabled = bool(train_config.get("progress_bar", True))

    gaussian_config = plots_config.get("gaussian_baseline", {}) or {}
    assert isinstance(gaussian_config, Mapping)
    gaussian_enabled = bool(gaussian_config.get("enabled", False))
    gaussian_explicitly_disabled = _gaussian_baseline_explicitly_disabled(
        plots_config
    )
    gaussian_seed_base = int(gaussian_config.get("seed", seed + 100_000))
    gaussian_noise_mode = str(gaussian_config.get("noise_mode", "sample_channel"))
    activation_config = plots_config.get("activation", {}) or {}
    assert isinstance(activation_config, Mapping)
    activation_threshold = float(activation_config.get("threshold", 0.5))

    # 各分类分支可能分别采用 CE/BCE，导出激活必须匹配对应 softmax/sigmoid 语义。
    default_loss_type = normalize_loss_type(train_config.get("loss_type", "ce"), name="loss_type")
    time_score_mode = score_mode_for_loss_type(
        normalize_loss_type(train_config.get("time_loss_type", default_loss_type), name="time_loss_type")
    )
    pooled_score_mode = score_mode_for_loss_type(
        normalize_loss_type(train_config.get("pooled_loss_type", default_loss_type), name="pooled_loss_type")
    )
    fusion_score_mode = score_mode_for_loss_type(
        normalize_loss_type(train_config.get("fusion_loss_type", default_loss_type), name="fusion_loss_type")
    )

    split_seed_offsets = {"val": 1, "test": 2}
    export_input_kinds = (
        control_input_kinds
        if control_mode
        else (("real", "gaussian") if gaussian_enabled else ("real",))
    )
    export_arguments = {}
    # 显式关闭基线时先清掉旧 Gaussian CSV，避免绘图器误把历史结果混入当前图表。
    for split in splits:
        if not control_mode and gaussian_explicitly_disabled:
            _remove_stale_gaussian_exports(run_dir, split)
        for input_kind in export_input_kinds:
            export_arguments[(split, input_kind)] = dict(
                model=model,
                loader=loaders[split],
                task=task,
                device=device,
                amp=amp,
                output_dir=export_dir,
                split=split,
                attention_indices=attention_indices,
                time_s=datasets[split].time_s,
                progress_bar_enabled=progress_bar_enabled,
                input_kind=input_kind,
                gaussian_seed=gaussian_seed_base + split_seed_offsets.get(split, 0),
                gaussian_noise_mode=gaussian_noise_mode,
                time_shuffle_seed=(
                    time_shuffle_seed_base + split_seed_offsets.get(split, 0)
                    if input_kind == "time_shuffle"
                    else None
                ),
                time_roll_shift_samples=(
                    time_roll_shift_samples if input_kind == "time_roll" else None
                ),
                activation_threshold=activation_threshold,
                time_score_mode=time_score_mode,
                pooled_score_mode=pooled_score_mode,
                fusion_score_mode=fusion_score_mode,
                min_cell_class_count=int(
                    data_config.get("eval_min_cell_class_count", 1)
                ),
            )

    if control_mode:
        copy_existing_target_exports(
            run_dir, export_dir, splits=splits, input_kinds=reuse_existing_kinds,
            expected_parameters={key: describe_export(export_target_outputs, **arguments)
                                 for key, arguments in export_arguments.items()
                                 if key[1] in reuse_existing_kinds},
        )
    for (_, input_kind), arguments in export_arguments.items():
        if input_kind not in reuse_existing_kinds:
            export_target_outputs(**arguments)

    if control_mode:
        assert figures_root is not None
        plot_target_conditions(
            export_dir,
            input_kinds=control_input_kinds,
            splits=splits,
            output_root=figures_root,
            plot_config=effective_plot_config,
        )
        manifest = {
            "source_run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "task": task,
            "lags": lags,
            "splits": list(splits),
            "input_kinds": list(control_input_kinds),
            "reused_existing_input_kinds": list(reuse_existing_kinds),
            "exports_dir": str(export_dir),
            "figures_root": str(figures_root),
            "gaussian_noise_mode": gaussian_noise_mode,
            "gaussian_seed_by_split": {
                split: gaussian_seed_base + split_seed_offsets.get(split, 0)
                for split in splits
            },
            "time_shuffle_seed_by_split": {
                split: time_shuffle_seed_base + split_seed_offsets.get(split, 0)
                for split in splits
            },
            "time_roll_shift_samples": time_roll_shift_samples,
            "time_roll_shift_s": resolved_time_roll_shift_s,
            "time_roll_direction": "positive torch.roll along time; later with circular wrap",
            "time_axis_start_s": float(times[0]),
            "time_axis_end_s": float(times[-1]),
            "time_axis_sample_count": int(times.numel()),
            "plot_time_axis_start_s": (
                display_time_limits[0] if display_time_limits is not None else None
            ),
            "plot_time_axis_end_s": (
                display_time_limits[1] if display_time_limits is not None else None
            ),
        }
        manifest_path = control_root / "control_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[reexport-controls] done - figures at {figures_root}; "
            f"manifest at {manifest_path}",
            flush=True,
        )
    else:
        figure_dir = args.output_dir.expanduser().resolve() if args.output_dir else run_dir / "figures"
        plot_target_run(
            run_dir,
            splits=splits,
            output_dir=figure_dir,
            plot_config=effective_plot_config,
        )
        print(f"[reexport] done - figures at {figure_dir}", flush=True)

    del model, loaders, datasets, attention_indices
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
