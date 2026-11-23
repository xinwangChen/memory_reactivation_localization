"""构建和读取 PB2023 数据清单、元数据索引及 block 划分。

manifest 的一行对应一条“实验-被试记录”，而不是一个自然人或一个 trial。
subject_uid（如 Exp1_Sub01）唯一标识文件与归一化统计，subject（如 Sub01）
则允许上层把同一参与者跨实验的数据映射到同一 participant。划分以 block
为最小单位，避免相邻 trial 被拆到不同数据集造成时序泄漏。
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SubjectRecord:
    """manifest 中一条实验-被试记录的固定 schema。

    npz_path 指向清洗 EEG，约定至少包含 X 与 trial0；meta_path 指向逐 trial
    行为元数据。n_trials 是 NPZ 中保留下来的清洗后 trial 数，不保证等于
    元数据总行数。Exp1_Sub01 与 Exp2_Sub01 是同一参与者的两条记录。
    """

    exp: int
    subject: str  # 如 Sub01；跨实验可共享 subject_id
    subject_uid: str  # 如 Exp1_Sub01；归一化统计与文件主键
    npz_path: str
    meta_path: str
    n_trials: int


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """按表头读取 CSV，返回保持文件顺序的字符串字典列表。"""
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required CSV file not found: {path}") from None


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """按首行键顺序写出结构化 CSV。

    首行定义固定列 schema，其余行交由 csv.DictWriter 校验。空表通常表示
    上游扫描或筛选失败，不能推断表头，因此作为错误处理而不是生成空文件。
    """
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_manifest(
    preprocessed_root: Path,
    output_csv: Path,
    experiments: Sequence[int],
    subjects: Sequence[str] | None = None,
) -> List[SubjectRecord]:
    """扫描指定实验/被试的清洗 EEG，并生成可复用 manifest。

    文件命名协议是 Exp{n}/Sub*_eeg_clean.npz 与同目录下配对的
    Sub*_meta.csv。函数验证每个请求的实验-被试组合都存在，从 NPZ 的 trial0
    长度记录清洗后 trial 数，最后把完整记录写入 output_csv。
    """
    normalized_experiments: list[int] = []
    for exp in experiments:
        if isinstance(exp, (bool, np.bool_)) or not isinstance(exp, (int, np.integer)):
            raise ValueError("experiments must contain positive integers")
        normalized_exp = int(exp)
        if normalized_exp < 1:
            raise ValueError("experiments must contain positive integers")
        normalized_experiments.append(normalized_exp)
    if not normalized_experiments:
        raise ValueError("experiments must not be empty")
    if len(set(normalized_experiments)) != len(normalized_experiments):
        raise ValueError("experiments must not contain duplicates")

    experiments = tuple(normalized_experiments)
    records: List[SubjectRecord] = []
    allowed_subjects = set(subjects) if subjects is not None else None
    for exp in experiments:
        exp_dir = preprocessed_root / f"Exp{exp}"
        n_before = len(records)
        for npz_path in sorted(exp_dir.glob("Sub*_eeg_clean.npz")):  # 约定文件名
            subject = npz_path.name.split("_", 1)[0]
            if allowed_subjects is not None and subject not in allowed_subjects:
                continue
            meta_path = exp_dir / f"{subject}_meta.csv"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing metadata for {npz_path}: {meta_path}")
            with np.load(npz_path) as data:
                if "trial0" not in data:
                    raise KeyError(
                        f"NPZ file is missing required 'trial0' array: {npz_path}"
                    )
                n_trials = int(data["trial0"].shape[0])  # 清洗后行数，可少于元数据 2000
            records.append(
                SubjectRecord(
                    exp=exp,
                    subject=subject,
                    subject_uid=f"Exp{exp}_{subject}",  # 文件/归一化主键，跨实验不合并
                    npz_path=str(npz_path),
                    meta_path=str(meta_path),
                    n_trials=n_trials,
                )
            )
        if len(records) == n_before:
            raise FileNotFoundError(
                f"No Sub*_eeg_clean.npz files found for Exp{exp} under {exp_dir}. "
                "Run the raw-data preprocessor with --save-all-clean-eeg first."
            )
    selected_subjects = (
        allowed_subjects if allowed_subjects is not None else {record.subject for record in records}
    )
    present_pairs = {(record.exp, record.subject) for record in records}
    missing_pairs = sorted(set(product(experiments, selected_subjects)) - present_pairs)
    if missing_pairs:
        missing = ", ".join(f"Exp{exp}_{subject}" for exp, subject in missing_pairs)
        raise FileNotFoundError(f"Missing requested EEG records under {preprocessed_root}: {missing}")
    if not records:
        raise FileNotFoundError(
            f"No Sub*_eeg_clean.npz files found under {preprocessed_root}. "
            "Run the raw-data preprocessor with --save-all-clean-eeg first."
        )
    write_csv_rows(output_csv, [asdict(record) for record in records])
    return records


def load_manifest(
    path: Path,
    experiments: Sequence[int] | None = None,
    subjects: Sequence[str] | None = None,
) -> List[SubjectRecord]:
    """加载 manifest，按需筛选，并验证请求组合的笛卡尔积完整。

    当同时给出实验和被试时，每个实验都必须包含每个请求被试；只给被试时，
    至少要在清单中找到该被试。返回顺序继承 CSV 顺序，供后续 record_id
    使用，因此这里不会重新排序。
    """
    allowed = set(experiments) if experiments is not None else None
    allowed_subjects = set(subjects) if subjects is not None else None
    all_rows = read_csv_rows(path)
    all_records = []
    for row_index, row in enumerate(all_rows):
        try:
            all_records.append(SubjectRecord(
                exp=int(row["exp"]),
                subject=row["subject"],
                subject_uid=row["subject_uid"],
                npz_path=row["npz_path"],
                meta_path=row["meta_path"],
                n_trials=int(row["n_trials"]),
            ))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse manifest row {row_index} in {path}: {exc}"
            ) from exc
    subject_uids = [record.subject_uid for record in all_records]
    duplicate_uids = sorted(
        {subject_uid for subject_uid in subject_uids if subject_uids.count(subject_uid) > 1}
    )
    if duplicate_uids:
        raise ValueError(
            f"Duplicate subject_uid values in {path}: {duplicate_uids}"
        )
    record_keys = [(record.exp, record.subject) for record in all_records]
    duplicate_record_keys = sorted(
        {key for key in record_keys if record_keys.count(key) > 1}
    )
    if duplicate_record_keys:
        raise ValueError(
            f"Duplicate experiment/subject records in {path}: {duplicate_record_keys}"
        )

    records = all_records  # 保持 CSV 顺序 → 下游 record_id 稳定
    if allowed is not None:
        records = [record for record in records if record.exp in allowed]
    if allowed_subjects is not None:
        records = [record for record in records if record.subject in allowed_subjects]

    if allowed is not None:
        missing_experiments = sorted(allowed - {record.exp for record in records})
        if missing_experiments:
            missing = ", ".join(f"Exp{exp}" for exp in missing_experiments)
            raise ValueError(f"Missing requested experiments in {path}: {missing}")
        selected_subjects = (
            allowed_subjects if allowed_subjects is not None else {record.subject for record in records}
        )
        present_pairs = {(record.exp, record.subject) for record in records}
        missing_pairs = sorted(set(product(allowed, selected_subjects)) - present_pairs)
        if missing_pairs:
            missing = ", ".join(f"Exp{exp}_{subject}" for exp, subject in missing_pairs)
            raise ValueError(f"Missing requested manifest records in {path}: {missing}")
    elif allowed_subjects is not None:
        missing_subjects = sorted(allowed_subjects - {record.subject for record in records})
        if missing_subjects:
            missing = ", ".join(missing_subjects)
            raise ValueError(f"Missing requested subjects in {path}: {missing}")
    if not records:
        raise ValueError(f"No manifest rows selected from {path}")
    return records


def metadata_by_trial0(meta_path: Path) -> Dict[int, Dict[str, str]]:
    """Build a unique trial0 lookup table from behavioral metadata."""
    result: Dict[int, Dict[str, str]] = {}
    for row_index, row in enumerate(read_csv_rows(meta_path)):
        try:
            key = int(row["trial0"])
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse trial0 in metadata row {row_index} of {meta_path}: {exc}"
            ) from exc
        if key in result:
            raise ValueError(f"Duplicate trial0={key} in metadata CSV {meta_path}")
        result[key] = row
    if not result:
        raise ValueError(f"Metadata CSV is empty: {meta_path}")
    return result


def write_block_splits(
    records: Sequence[SubjectRecord],
    output_csv: Path,
    val_blocks: int = 2,
    test_blocks: int = 2,
    seed: int = 20260530,
) -> List[Dict[str, object]]:
    """为每条实验-被试记录独立生成可复现的 block 级数据划分。

    每条输出行的 schema 为 exp、subject、subject_uid、block、split。随机种子
    由全局 seed 与 subject_uid 哈希得到，因此过滤 cohort 或改变 manifest
    行顺序不会改变某条记录自身的划分。每条记录固定抽取 val_blocks 和
    test_blocks，其余 block 均归训练集。
    """
    normalized_counts: dict[str, int] = {}
    for name, value in (("val_blocks", val_blocks), ("test_blocks", test_blocks)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a non-negative integer")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        normalized_counts[name] = normalized
    val_blocks = normalized_counts["val_blocks"]
    test_blocks = normalized_counts["test_blocks"]
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)

    subject_uids = [record.subject_uid for record in records]
    if len(set(subject_uids)) != len(subject_uids):
        raise ValueError("records must contain unique subject_uid values")
    rows: List[Dict[str, object]] = []
    for record in records:
        metadata = read_csv_rows(Path(record.meta_path))
        blocks = sorted({int(row["block"]) for row in metadata})  # 划分单位=block，非 trial
        if len(blocks) <= val_blocks + test_blocks:
            raise ValueError(f"Not enough blocks for fixed splits: {record.subject_uid}")
        # 种子只依赖 global seed + subject_uid → 换 cohort 不改该人划分
        seed_bytes = f"{int(seed)}:{record.subject_uid}".encode("utf-8")
        subject_seed = int.from_bytes(
            hashlib.blake2b(seed_bytes, digest_size=8).digest(),
            "little",
        )
        rng = np.random.default_rng(subject_seed)
        shuffled = np.asarray(blocks, dtype=int)
        rng.shuffle(shuffled)
        val_set = set(int(block) for block in shuffled[:val_blocks])
        test_set = set(int(block) for block in shuffled[val_blocks : val_blocks + test_blocks])
        for block in blocks:
            split = "val" if block in val_set else "test" if block in test_set else "train"
            rows.append(
                {
                    "exp": record.exp,
                    "subject": record.subject,
                    "subject_uid": record.subject_uid,
                    "block": block,
                    "split": split,
                }
            )
    write_csv_rows(output_csv, rows)
    return rows


def write_block_cv_splits(
    records: Sequence[SubjectRecord],
    output_dir: Path,
    n_folds: int = 5,
    seed: int = 20260530,
) -> List[Path]:
    """写出按被试 block 划分的 ``n_folds`` 份交叉验证 split 文件。

    每条实验-被试记录独立打乱自己的 block，并按轮询方式分配到 fold；因此同一
    被试在每一折都有约 ``1/n_folds`` 的 block 做验证，其余 block 做训练。不同
    被试的训练 block 会由数据集层统一混合。每份 CSV 只包含 ``train``/``val``，
    不生成 test，且额外写入 ``cv_fold``/``cv_folds`` 便于审计。
    """
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, (int, np.integer)):
        raise ValueError("n_folds must be an integer")
    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    subject_uids = [record.subject_uid for record in records]
    if len(set(subject_uids)) != len(subject_uids):
        raise ValueError("records must contain unique subject_uid values")

    assignments: dict[str, dict[int, int]] = {}
    for record in records:
        metadata = read_csv_rows(Path(record.meta_path))
        try:
            blocks = sorted({int(row["block"]) for row in metadata})
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse block in metadata CSV {record.meta_path}: {exc}"
            ) from exc
        if len(blocks) < n_folds:
            raise ValueError(
                f"Not enough blocks for {n_folds}-fold CV: {record.subject_uid} "
                f"has {len(blocks)} blocks"
            )
        seed_bytes = f"{seed}:{record.subject_uid}:cv{n_folds}".encode("utf-8")
        subject_seed = int.from_bytes(
            hashlib.blake2b(seed_bytes, digest_size=8).digest(),
            "little",
        )
        rng = np.random.default_rng(subject_seed)
        shuffled = np.asarray(blocks, dtype=int)
        rng.shuffle(shuffled)
        assignments[record.subject_uid] = {
            int(block): int(index % n_folds)
            for index, block in enumerate(shuffled)
        }

    output_dir = Path(output_dir)
    paths: List[Path] = []
    for fold in range(n_folds):
        rows: List[Dict[str, object]] = []
        for record in records:
            record_assignment = assignments[record.subject_uid]
            for block in sorted(record_assignment):
                rows.append(
                    {
                        "exp": record.exp,
                        "subject": record.subject,
                        "subject_uid": record.subject_uid,
                        "block": block,
                        "split": "val" if record_assignment[block] == fold else "train",
                        "cv_fold": fold,
                        "cv_folds": n_folds,
                    }
                )
        path = output_dir / f"fold_{fold}" / "block_splits.csv"
        write_csv_rows(path, rows)
        paths.append(path)
    return paths


def normalize_cv_strategy(value: object) -> str:
    if not isinstance(value, str) or value not in {"block", "subject"}:
        raise ValueError("cv_strategy must be block or subject")
    return value


def subject_split_membership(
    records: Sequence[SubjectRecord], split_map: Mapping[Tuple[str, int], str],
) -> dict[str, list[str]]:
    """Validate whole-participant train/val separation, including across experiments."""
    by_uid = {record.subject_uid: record.subject for record in records}
    if not records or len(by_uid) != len(records):
        raise ValueError("Subject CV requires unique, nonempty manifest records")
    if {uid for uid, _ in split_map} != set(by_uid):
        raise ValueError("Subject CV split records do not match the manifest")
    for record in records:
        expected = {int(row["block"]) for row in read_csv_rows(Path(record.meta_path))}
        actual = {block for uid, block in split_map if uid == record.subject_uid}
        if not expected or actual != expected:
            raise ValueError(f"Subject CV must assign all blocks for {record.subject_uid}")
    membership: dict[str, set[str]] = {}
    for (uid, _), split in split_map.items():
        if split not in {"train", "val"}:
            raise ValueError("Subject CV supports train and val only")
        membership.setdefault(by_uid[uid], set()).add(split)
    if any(len(splits) != 1 for splits in membership.values()):
        raise ValueError("Subject CV train/val participants overlap; all blocks/experiments must stay together")
    result = {split: sorted(name for name, splits in membership.items() if split in splits)
              for split in ("train", "val")}
    if not result["train"] or not result["val"]:
        raise ValueError("Subject CV requires nonempty train and val participant groups")
    return result


def write_subject_cv_splits(
    records: Sequence[SubjectRecord], output_dir: Path, n_folds: int = 5, seed: int = 20260530,
) -> List[Path]:
    """Hold out entire participants; 30 people / 5 folds gives 24 train and 6 val."""
    if isinstance(n_folds, (bool, np.bool_)) or not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer >= 2")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Subject CV seed must be a nonnegative integer")
    uids = [record.subject_uid for record in records]
    subjects = sorted({record.subject for record in records})
    if len(set(uids)) != len(uids) or len(subjects) < n_folds:
        raise ValueError("Subject CV requires unique records and at least n_folds participants")
    shuffled = np.random.default_rng(int(seed)).permutation(subjects)
    assignments = {str(subject): int(index % n_folds) for index, subject in enumerate(shuffled)}
    blocks = {}
    for record in records:
        blocks[record.subject_uid] = sorted({int(row["block"]) for row in read_csv_rows(Path(record.meta_path))})
        if not blocks[record.subject_uid]:
            raise ValueError(f"No blocks for {record.subject_uid}")
    paths = []
    for fold in range(int(n_folds)):
        rows = [{"exp": record.exp, "subject": record.subject, "subject_uid": record.subject_uid,
                 "block": block, "split": "val" if assignments[record.subject] == fold else "train",
                 "cv_fold": fold, "cv_folds": int(n_folds), "cv_strategy": "subject", "cv_seed": int(seed)}
                for record in sorted(records, key=lambda r: r.subject_uid) for block in blocks[record.subject_uid]]
        path = Path(output_dir) / f"fold_{fold}" / "block_splits.csv"
        write_csv_rows(path, rows)
        paths.append(path)
    return paths


def load_split_map(path: Path) -> Dict[Tuple[str, int], str]:
    """Load a unique ``(subject_uid, block) -> split`` mapping."""
    result: Dict[Tuple[str, int], str] = {}
    valid_splits = {"train", "val", "test"}
    for row_index, row in enumerate(read_csv_rows(path)):
        try:
            key = (row["subject_uid"], int(row["block"]))
            split = row["split"]
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Failed to parse split map row {row_index} in {path}: {exc}"
            ) from exc
        if key in result:
            raise ValueError(
                f"Duplicate (subject_uid, block) key {key!r} in split CSV {path}"
            )
        if split not in valid_splits:
            raise ValueError(
                f"Invalid split {split!r} for key {key!r} in {path}"
            )
        result[key] = split
    if not result:
        raise ValueError(f"Split CSV is empty: {path}")
    return result
