"""计算并加载仅由训练 EEG 拟合的通道归一化统计。

NPZ 中 X 的约定形状为 [trial, channel, time]。每条实验-被试记录单独估计
每个通道的均值与标准差，统计维度覆盖训练 trial 和时间轴，但不跨通道、
不跨被试，也不会使用验证或测试 block。可选 train_global 模式跨训练记录
汇总通道均值与方差，未见被试使用相同统计，不参与统计拟合。
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

from .manifest import SubjectRecord, load_split_map, metadata_by_trial0


def normalize_normalization_mode(value: object) -> str:
    if not isinstance(value, str) or value not in {"per_subject", "train_global"}:
        raise ValueError("normalization_mode must be per_subject or train_global")
    return value


def validate_normalization_identity(
    records: Sequence[SubjectRecord], split_csv: Path, path: Path, *, mode: str,
) -> None:
    """Reject incompatible fit identities; warn when legacy provenance is unavailable."""
    mode = normalize_normalization_mode(mode)
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError('Normalization JSON must be a mapping')
    actual = raw.get("normalization_mode", "per_subject") if isinstance(raw, Mapping) else "per_subject"
    if actual != mode:
        raise ValueError("Normalization mode differs from the configured data protocol")
    if mode == "per_subject":
        stats = raw.get("stats", raw)
        if not isinstance(stats, Mapping):
            raise ValueError('Per-subject normalization statistics must be a mapping')
        values = [stats.get(record.subject_uid, {}) for record in records]
        if not values or any(not isinstance(value, Mapping) or 'mean' not in value for value in values):
            raise ValueError("Per-subject normalization does not cover the selected records")
        fit_fields = {'fit_split_sha256', 'fit_train_blocks', 'fit_trial_count'}
        if not any(isinstance(value, Mapping) and fit_fields.intersection(value) for value in stats.values()):
            warnings.warn("Legacy per-subject normalization has no fit-split identity; "
                          "its training-fold provenance cannot be verified. Regenerate statistics "
                          "in a new artifact directory for verified fitting provenance.",
                          RuntimeWarning, stacklevel=2)
            return
        split_map = load_split_map(Path(split_csv))
        split_hash = hashlib.sha256(Path(split_csv).read_bytes()).hexdigest()
        for record, value in zip(records, values):
            blocks = sorted(block for (uid, block), split in split_map.items()
                            if uid == record.subject_uid and split == 'train')
            if (not blocks or value.get('fit_split_sha256') != split_hash
                    or value.get('fit_train_blocks') != blocks
                    or not isinstance(value.get('fit_trial_count'), int) or value['fit_trial_count'] < 1):
                raise ValueError("Per-subject normalization fit population/split identity differs")
        return
    split_map = load_split_map(Path(split_csv))
    record_uids = sorted(record.subject_uid for record in records)
    train_uids = sorted(uid for uid in record_uids if any(key[0] == uid and split == "train" for key, split in split_map.items()))
    train_subjects = sorted({record.subject for record in records if record.subject_uid in train_uids})
    if (raw.get("schema_version") != 3 or raw.get("record_uids") != record_uids
            or raw.get("fit_record_uids") != train_uids or raw.get("fit_subjects") != train_subjects
            or raw.get("split_sha256") != hashlib.sha256(Path(split_csv).read_bytes()).hexdigest()):
        raise ValueError("Train-global normalization fit population/split identity differs")
    if not train_uids or not isinstance(raw.get("fit_trial_count"), int) or raw["fit_trial_count"] < 1:
        raise ValueError("Train-global normalization has no training observations")


def _merge_moments(first, second):
    if first is None:
        return second
    count, mean, m2 = first
    other_count, other_mean, other_m2 = second
    if mean.shape != other_mean.shape:
        raise ValueError("Train-global normalization requires a common channel count")
    total = count + other_count
    delta = other_mean - mean
    return total, mean + delta * (other_count / total), m2 + other_m2 + delta ** 2 * (count * other_count / total)


def _compute_train_global_stats(records, split_csv, output_json, eps, *, graph, graph_mode, moments_cache):
    from .spatial import apply_graph_input, normalize_graph_input_mode
    graph_mode = normalize_graph_input_mode(graph_mode)
    split_map = load_split_map(Path(split_csv))
    combined = None
    fit_uids, fit_subjects, trial_count = [], set(), 0
    for record in sorted(records, key=lambda r: r.subject_uid):
        train_blocks = tuple(sorted(block for (uid, block), split in split_map.items() if uid == record.subject_uid and split == "train"))
        if not train_blocks:
            # In particular, fitting never reads the held-out subjects' EEG.
            continue
        if graph is not None and record.exp != graph.experiment:
            raise ValueError("EEG experiment does not match the graph montage")
        stat = Path(record.npz_path).stat()
        key = (record.npz_path, stat.st_size, stat.st_mtime_ns,
               hashlib.sha256(Path(record.meta_path).read_bytes()).hexdigest(), train_blocks,
               None if graph is None else graph.sha256, graph_mode)
        cached = None if moments_cache is None else moments_cache.get(key)
        if cached is None:
            metadata = metadata_by_trial0(Path(record.meta_path))
            with np.load(record.npz_path) as archive:
                x = np.asarray(archive["X"], dtype=np.float32)
                raw_trial0 = np.asarray(archive["trial0"])
            if (x.ndim != 3 or not all(x.shape) or raw_trial0.shape != (x.shape[0],)
                    or not np.isfinite(raw_trial0).all() or not (raw_trial0 == np.floor(raw_trial0)).all()):
                raise ValueError(f"Invalid EEG/trial0 arrays in {record.npz_path}")
            trial0 = raw_trial0.astype(np.int64)
            if len(set(trial0.tolist())) != len(trial0):
                raise ValueError(f"Duplicate trial0 in {record.npz_path}")
            indices = []
            for index, trial in enumerate(trial0):
                if int(trial) not in metadata:
                    raise ValueError(f"trial0={trial} missing from {record.meta_path}")
                split_key = record.subject_uid, int(metadata[int(trial)]["block"])
                if split_key not in split_map:
                    raise ValueError(f"Split CSV missing key {split_key!r}")
                if split_map[split_key] == "train":
                    indices.append(index)
            if not indices:
                raise ValueError(f"No training trials for {record.subject_uid}")
            moments = None
            for start in range(0, len(indices), 64):
                chunk = apply_graph_input(x[indices[start:start + 64]], graph, graph_mode)
                if not np.isfinite(chunk).all():
                    raise ValueError(f"Non-finite training EEG in {record.npz_path}")
                count = chunk.shape[0] * chunk.shape[2]
                mean = chunk.mean(axis=(0, 2), dtype=np.float64)
                m2 = chunk.var(axis=(0, 2), dtype=np.float64) * count
                moments = _merge_moments(moments, (count, mean, m2))
            cached = moments, len(indices)
            if moments_cache is not None:
                moments_cache[key] = cached
        moments, n_trials = cached
        combined = _merge_moments(combined, moments)
        fit_uids.append(record.subject_uid)
        fit_subjects.add(record.subject)
        trial_count += n_trials
    if combined is None:
        raise ValueError("Train-global normalization requires training trials")
    count, mean, m2 = combined
    std = np.maximum(np.sqrt(np.maximum(m2 / count, 0)), eps)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Non-finite train-global normalization statistics")
    values = {"mean": mean.tolist(), "std": std.tolist()}
    payload = {"schema_version": 3, "normalization_mode": "train_global",
               "graph_laplacian_sha256": None if graph is None else graph.sha256,
               "graph_laplacian_mode": graph_mode, "global_stats": values,
               "record_uids": sorted(record.subject_uid for record in records),
               "fit_record_uids": fit_uids, "fit_subjects": sorted(fit_subjects),
               "fit_trial_count": trial_count, "fit_observation_count_per_channel": count,
               "split_sha256": hashlib.sha256(Path(split_csv).read_bytes()).hexdigest()}
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(output_json).write_text(json.dumps(payload, indent=2, allow_nan=False))
    return {record.subject_uid: dict(values) for record in records}


def compute_normalization_stats(
    records: Sequence[SubjectRecord],
    split_csv: Path,
    output_json: Path,
    eps: float = 1e-6,
    *,
    graph=None,
    graph_mode: str = "replace",
    mode: str = "per_subject",
    moments_cache: dict | None = None,
) -> Dict[str, Dict[str, object]]:
    """Compute finite per-channel statistics from training trials only."""
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    subject_uids = [record.subject_uid for record in records]
    if len(set(subject_uids)) != len(subject_uids):
        raise ValueError("records must contain unique subject_uid values")

    mode = normalize_normalization_mode(mode)
    if mode == "train_global":
        return _compute_train_global_stats(records, split_csv, output_json, eps,
            graph=graph, graph_mode=graph_mode, moments_cache=moments_cache)

    split_map = load_split_map(split_csv)
    stats: Dict[str, Dict[str, object]] = {}
    for record in records:
        metadata = metadata_by_trial0(Path(record.meta_path))
        with np.load(record.npz_path) as data:
            missing_arrays = {"X", "trial0"} - set(data.files)
            if missing_arrays:
                raise ValueError(
                    f"{record.npz_path} missing arrays: {sorted(missing_arrays)}"
                )
            x = np.asarray(data["X"], dtype=np.float32)
            raw_trial0 = np.asarray(data["trial0"])
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0:
            raise ValueError(
                f"X in {record.npz_path} must be non-empty [trial, channel, time]"
            )
        if not np.all(np.isfinite(x)):
            raise ValueError(f"X in {record.npz_path} contains NaN or Inf")
        if raw_trial0.ndim != 1 or raw_trial0.shape[0] != x.shape[0]:
            raise ValueError(
                f"trial0 in {record.npz_path} must be one-dimensional with one value per trial"
            )
        if (
            not np.all(np.isfinite(raw_trial0))
            or not np.all(raw_trial0 == np.floor(raw_trial0))
        ):
            raise ValueError(f"trial0 in {record.npz_path} must contain finite integers")
        trial0 = raw_trial0.astype(np.int64)

        train_indices: list[int] = []
        for index, trial in enumerate(trial0):
            trial_key = int(trial)
            if trial_key not in metadata:
                raise ValueError(
                    f"trial0={trial_key} from {record.npz_path} is missing in {record.meta_path}"
                )
            block = int(metadata[trial_key]["block"])
            split_key = (record.subject_uid, block)
            if split_key not in split_map:
                raise ValueError(f"Split CSV missing key {split_key!r}")
            if split_map[split_key] == "train":
                train_indices.append(index)
        if not train_indices:
            raise ValueError(f"No training trials for {record.subject_uid}")

        train_x = x[np.asarray(train_indices, dtype=int)]
        if graph is not None:
            if record.exp != graph.experiment:
                raise ValueError("EEG experiment does not match the graph montage")
            from .spatial import apply_graph_input, normalize_graph_input_mode

            graph_mode = normalize_graph_input_mode(graph_mode)
            train_x = apply_graph_input(train_x, graph, graph_mode)
        elif str(graph_mode).strip().lower() != "replace":
            raise ValueError("graph_mode concat requires a graph Laplacian")
        mean = np.mean(train_x, axis=(0, 2), dtype=np.float64)
        std = np.std(train_x, axis=(0, 2), dtype=np.float64)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(
                f"Non-finite normalization statistics for {record.subject_uid}"
            )
        std = np.maximum(std, eps)
        stats[record.subject_uid] = {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "fit_split_sha256": hashlib.sha256(Path(split_csv).read_bytes()).hexdigest(),
            "fit_train_blocks": sorted(block for (uid, block), split in split_map.items()
                                       if uid == record.subject_uid and split == 'train'),
            "fit_trial_count": len(train_indices),
        }
    if not stats:
        raise ValueError("records must not be empty")
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        payload = stats if graph is None else {
            "schema_version": 2,
            "graph_laplacian_sha256": graph.sha256,
            "graph_laplacian_mode": graph_mode,
            "stats": stats,
        }
        json.dump(payload, f, indent=2, allow_nan=False)
    return stats


def load_normalization_stats(
    path: Path,
    *,
    graph_sha256: str | None = None,
    graph_mode: str = "replace",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Load and validate finite per-channel normalization statistics."""
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    actual_graph = raw.get("graph_laplacian_sha256") if isinstance(raw, Mapping) else None
    if actual_graph != graph_sha256:
        raise ValueError("Normalization preprocessing mismatch: graph Laplacian identity differs")
    if isinstance(raw, Mapping) and raw.get("schema_version") == 3:
        from .spatial import normalize_graph_input_mode
        if (raw.get("normalization_mode") != "train_global"
                or normalize_graph_input_mode(raw.get("graph_laplacian_mode")) != normalize_graph_input_mode(graph_mode)):
            raise ValueError("Invalid train-global normalization mode")
        uids, values = raw.get("record_uids"), raw.get("global_stats")
        if (not isinstance(uids, list) or not uids or not all(isinstance(uid, str) and uid for uid in uids)
                or len(set(uids)) != len(uids) or not isinstance(values, Mapping)):
            raise ValueError("Invalid train-global normalization records/statistics")
        raw = {uid: values for uid in uids}
    elif actual_graph is not None:
        if raw.get("schema_version") != 2:
            raise ValueError("Unsupported graph normalization schema")
        from .spatial import normalize_graph_input_mode

        expected_mode = normalize_graph_input_mode(graph_mode)
        actual_mode = normalize_graph_input_mode(raw.get("graph_laplacian_mode"))
        if actual_mode != expected_mode:
            raise ValueError("Normalization preprocessing mismatch: graph input mode differs")
        raw = raw["stats"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"Normalization JSON must contain at least one record: {path}")

    result: Dict[str, Dict[str, np.ndarray]] = {}
    for subject_uid, values in raw.items():
        if not isinstance(values, Mapping) or "mean" not in values or "std" not in values:
            raise ValueError(f"Invalid normalization entry for {subject_uid!r} in {path}")
        mean = np.asarray(values["mean"], dtype=np.float32)
        std = np.asarray(values["std"], dtype=np.float32)
        if (
            mean.ndim != 1
            or std.ndim != 1
            or mean.size == 0
            or mean.shape != std.shape
        ):
            raise ValueError(
                f"Normalization mean/std for {subject_uid!r} must be equal non-empty vectors"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError(
                f"Normalization mean/std for {subject_uid!r} must be finite"
            )
        if np.any(std <= 0.0):
            raise ValueError(
                f"Normalization std for {subject_uid!r} must be positive"
            )
        result[str(subject_uid)] = {"mean": mean, "std": std}
    return result
