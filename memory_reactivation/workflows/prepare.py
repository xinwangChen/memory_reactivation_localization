"""准备 PB2023 EEG 深度学习所需的可复用数据产物。

入口读取命令行中的实验、被试与划分参数，可选调用外部预处理脚本生成 clean EEG，
随后写出样本清单 ``manifest.csv``、block 划分 ``block_splits.csv``，以及仅由
训练 EEG 估计的 ``normalization.json``。``--cv-folds 5`` 在独立目录生成五折；
默认按 block 划分，可选 ``--cv-strategy subject`` 整体留出被试，并只用训练被试
计算全局通道统计。不指定五折参数时保留旧固定划分流程。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from memory_reactivation.data.manifest import (
    build_manifest,
    write_block_cv_splits,
    write_block_splits,
    write_subject_cv_splits,
    normalize_cv_strategy,
    subject_split_membership,
    load_split_map,
    read_csv_rows,
)
from memory_reactivation.data.normalization import compute_normalization_stats, normalize_normalization_mode


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None) -> argparse.Namespace:
    """解析数据准备命令行参数。

    返回：
        包含实验编号、可选被试列表、预处理路径、产物目录、划分规模和随机种子的
        ``argparse.Namespace``。
    """
    parser = argparse.ArgumentParser(
        description="Prepare all-clean PB2023 EEG files and reusable deep-learning artifacts."
    )
    parser.add_argument("--experiments", type=int, nargs="+", default=[1], choices=[1, 2])
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=Path("/nfs1/active_construction/PB2023_preprocessed"),
    )
    parser.add_argument(
        "--preprocess-script",
        type=Path,
        default=Path("/nfs1/active_construction/1_preprocess_pb2023.py"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts",
    )
    parser.add_argument("--run-preprocessor", action="store_true")
    parser.add_argument("--alignment-source", choices=["auto", "events", "pitch"], default="auto")
    parser.add_argument("--ambiguous-policy", choices=["error", "exclude"], default="error")
    parser.add_argument("--val-blocks", type=int, default=2)
    parser.add_argument("--test-blocks", type=int, default=2)
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="Generate cross-validation artifacts instead of fixed train/val/test splits.",
    )
    parser.add_argument(
        "--cv-output-dir",
        type=Path,
        default=None,
        help="Directory for CV artifacts (default: artifact-dir/cv{n}[-subject]).",
    )
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--cv-strategy", choices=["block", "subject"], default="block",
                        help="Hold out blocks within each participant, or entire participants.")
    parser.add_argument("--normalization-mode", choices=["per_subject", "train_global"], default=None,
                        help="Defaults to train_global for subject CV, otherwise per_subject.")
    return parser.parse_args(argv)


def run_preprocessor(args: argparse.Namespace) -> None:
    """调用已有预处理脚本，额外保存全部非 outlier EEG。

    参数：
        args: ``parse_args`` 返回的参数；其中指定外部脚本、输出根目录、实验和被试。

    外部进程非零退出时由 ``subprocess.run(..., check=True)`` 直接抛出异常，避免在
    原始数据不完整的情况下继续生成清单。
    """
    command = [
        sys.executable,  # 与当前环境同一解释器
        str(args.preprocess_script),
        "--output-root",
        str(args.preprocessed_root),
        "--experiments",
        *[str(exp) for exp in args.experiments],
        "--save-all-clean-eeg",  # 写出 Sub*_eeg_clean.npz
        "--alignment-source", getattr(args, "alignment_source", "auto"),
        "--ambiguous-policy", getattr(args, "ambiguous_policy", "error"),
    ]
    if args.subjects:
        command.extend(["--subjects", *args.subjects])
    print("[prepare] running raw-data conversion", flush=True)
    subprocess.run(command, check=True)


def prepare_cv_artifacts(args: argparse.Namespace, output_dir: Path) -> None:
    """Build a complete CV bundle before replacing any usable artifacts.

    In particular, a normalization failure must not leave new splits paired with
    the previous fold statistics. Staging and publication use the same filesystem.
    """
    output_dir = output_dir.expanduser().resolve()
    strategy = normalize_cv_strategy(getattr(args, "cv_strategy", "block"))
    mode = normalize_normalization_mode(getattr(args, "normalization_mode", None) or
                                        ("train_global" if strategy == "subject" else "per_subject"))
    if strategy == "subject" and mode != "train_global":
        raise ValueError("Subject CV requires normalization_mode=train_global")
    previous_split = output_dir / "fold_0/block_splits.csv"
    if previous_split.exists() and {row.get("cv_strategy", "block") for row in read_csv_rows(previous_split)} != {strategy}:
        raise ValueError("Use a separate output directory for a different CV strategy")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-prepare-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "ready"
        records = build_manifest(
            preprocessed_root=args.preprocessed_root,
            output_csv=staging / "manifest.csv",
            experiments=args.experiments,
            subjects=args.subjects,
        )
        writer = write_subject_cv_splits if strategy == "subject" else write_block_cv_splits
        split_paths = writer(
            records, staging, n_folds=args.cv_folds, seed=args.seed,
        )
        moments_cache = {}
        memberships = []
        for split_path in split_paths:
            print(f"[prepare] computing train-only normalization for {split_path.parent.name}", flush=True)
            compute_normalization_stats(
                records=records, split_csv=split_path,
                output_json=split_path.parent / "normalization.json",
                **({"mode": mode, "moments_cache": moments_cache} if mode == "train_global" else {}),
            )
            if strategy == "subject":
                memberships.append(subject_split_membership(records, load_split_map(split_path)))
        if strategy == "subject":
            (staging / "subject_cv.json").write_text(json.dumps({"cv_strategy": strategy, "seed": args.seed,
                "normalization_mode": mode, "folds": memberships}, indent=2) + "\n")
        previous = Path(temporary) / "previous"
        try:
            if output_dir.exists():
                output_dir.rename(previous)
            staging.rename(output_dir)
        except BaseException:
            if previous.exists() and not output_dir.exists():
                previous.rename(output_dir)
            raise
    print(f"[prepare] wrote {args.cv_folds} CV folds and {len(records)} records under {output_dir}", flush=True)


def main(argv=None) -> None:
    """依次生成清单、固定/CV block 划分和各训练集的标准化统计。

    输入来自命令行和预处理目录；结果写入 ``artifact_dir``，函数不返回业务数据。
    标准化统计放在最后计算，确保它只依据刚生成的训练 block，避免验证/测试泄漏。
    """
    args = parse_args(argv)
    strategy = normalize_cv_strategy(getattr(args, "cv_strategy", "block"))
    mode = normalize_normalization_mode(getattr(args, "normalization_mode", None) or
                                        ("train_global" if strategy == "subject" else "per_subject"))
    if strategy == "subject" and args.cv_folds is None:
        raise ValueError("--cv-strategy subject requires --cv-folds")
    if strategy == "subject" and mode != "train_global":
        raise ValueError("Subject CV requires normalization_mode=train_global")
    if args.cv_folds is not None and args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if args.cv_output_dir is not None and args.cv_folds is None:
        raise ValueError("--cv-output-dir requires --cv-folds")
    if args.run_preprocessor:
        run_preprocessor(args)  # 可选：.mat → preprocessed NPZ
    if args.cv_folds is not None:
        suffix = "-subject" if strategy == "subject" else ""
        prepare_cv_artifacts(args, args.cv_output_dir or args.artifact_dir / f"cv{args.cv_folds}{suffix}")
        return
    manifest_csv = args.artifact_dir / "manifest.csv"
    split_csv = args.artifact_dir / "block_splits.csv"
    normalization_json = args.artifact_dir / "normalization.json"
    # 顺序固定：先清单 → 再划分 → 最后仅用 train block 算归一化
    records = build_manifest(
        preprocessed_root=args.preprocessed_root,
        output_csv=manifest_csv,
        experiments=args.experiments,
        subjects=args.subjects,
    )
    write_block_splits(
        records=records,
        output_csv=split_csv,
        val_blocks=args.val_blocks,
        test_blocks=args.test_blocks,
        seed=args.seed,
    )
    compute_normalization_stats(
        records=records,
        split_csv=split_csv,  # 依赖刚写好的划分，避免 val/test 泄漏
        output_json=normalization_json,
        **({"mode": mode} if mode == "train_global" else {}),
    )
    print(
        f"[prepare] wrote {len(records)} experiment-subject records for {len({record.subject for record in records})} participants to {args.artifact_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
