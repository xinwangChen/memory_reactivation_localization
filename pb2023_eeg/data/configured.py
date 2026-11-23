"""Construct the same decoding dataset for training and snapshot consumers."""

import hashlib
from numbers import Integral
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from .dataset import LagDecodingDataset, normalize_lags, resolve_balance_seed
from .manifest import (SubjectRecord, load_manifest, load_split_map, read_csv_rows,
                       normalize_cv_strategy, subject_split_membership)
from .normalization import normalize_normalization_mode, validate_normalization_identity


def validate_data_protocol(data: Mapping[str, object], records=None):
    """Validate declared split/normalization modes, also for snapshot consumers."""
    strategy = normalize_cv_strategy(data.get("cv_strategy", "block"))
    mode = normalize_normalization_mode(data.get("normalization_mode", "per_subject"))
    rows = read_csv_rows(Path(data["split_csv"]))
    if {row.get("cv_strategy", "block") for row in rows} != {strategy}:
        raise ValueError("CV strategy differs from the split artifact")
    if records is None:
        records = load_manifest(Path(data["manifest_csv"]),
                                experiments=[int(exp) for exp in data.get("experiments", [1])],
                                subjects=data.get("subjects"))
    membership = None
    if strategy == "subject":
        if mode != "train_global":
            raise ValueError("Subject CV requires normalization_mode=train_global")
        membership = subject_split_membership(records, load_split_map(Path(data["split_csv"])))
        for split in ("train", "val"):
            if f"cv_{split}_subjects" in data and data[f"cv_{split}_subjects"] != membership[split]:
                raise ValueError("Subject CV participant lists differ from the split artifact")
    validate_normalization_identity(records, Path(data["split_csv"]), Path(data["normalization_json"]), mode=mode)
    return membership


def validate_subject_cv_datasets(config: Mapping[str, object], datasets: Mapping) -> None:
    """Check actual example membership and reject untrained individual embeddings."""
    data = config["data"]
    if data.get("cv_strategy", "block") != "subject":
        return
    membership = validate_data_protocol(data)
    actual = {}
    for split, dataset in datasets.items():
        if split not in membership:
            raise ValueError("Subject CV supports train and val only")
        examples = getattr(dataset, "dynamic_choice_pool", ()) or dataset.examples
        names = dataset.store.participant_names
        actual[split] = {str(names[int(dataset.participant_indices[e.subject_index])]) for e in examples}
        if actual[split] != set(membership[split]):
            raise ValueError(f"Actual {split} examples do not match the subject-CV participant list")
    if actual.get("train", set()) & actual.get("val", set()):
        raise ValueError("Actual subject-CV train and validation participants overlap")
    model = config.get("model", {})
    subject = model.get("subject_conditioning") or {}
    active = bool(subject.get("enabled", False)) or model.get("attention_mode") in {"query", "gated_query"}
    if active and subject.get("mode", "individual") != "shared_plus_residual":
        raise ValueError("Subject CV with subject conditioning requires mode=shared_plus_residual")


def resolve_cv_data_config(
    data: Mapping[str, object],
    *,
    cv_fold: int | None = None,
    cv_folds: int | None = None,
    cv_output_dir: str | Path | None = None,
    cv_strategy: str | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Resolve one generated block- or subject-CV fold into a dataset config.

    The default artifact layout is next to the configured fixed split file::

        artifacts/cv5/fold_0/{block_splits.csv,normalization.json}

    ``None`` for ``cv_fold`` leaves the input unchanged, preserving fixed train/val/test
    runs. A fold is deliberately selected explicitly so a run can never silently train
    on a different validation fold.
    """
    strategy = normalize_cv_strategy(data.get("cv_strategy", "block") if cv_strategy is None else cv_strategy)
    if cv_fold is None:
        if strategy == "subject":
            raise ValueError("Subject CV requires --cv-fold or config.cv_fold")
        if cv_folds is not None or cv_output_dir is not None:
            raise ValueError("--cv-fold is required when CV options are provided")
        return dict(data), None
    if isinstance(cv_fold, bool) or not isinstance(cv_fold, Integral):
        raise ValueError("cv_fold must be an integer")
    if cv_folds is not None and (isinstance(cv_folds, bool) or not isinstance(cv_folds, Integral)):
        raise ValueError("cv_folds must be an integer")
    fold = int(cv_fold)
    n_folds = 5 if cv_folds is None else int(cv_folds)
    if n_folds < 2:
        raise ValueError("--cv-folds must be at least 2")
    if fold < 0 or fold >= n_folds:
        raise ValueError(f"--cv-fold must be in [0, {n_folds - 1}]")
    cv_output_dir = cv_output_dir or data.get("cv_split_dir")
    if cv_output_dir is None and "split_csv" not in data:
        raise ValueError("Provide cv_split_dir/--cv-output-dir or data.split_csv to locate CV artifacts")
    base_dir = (
        Path(cv_output_dir).expanduser()
        if cv_output_dir is not None
        else Path(str(data["split_csv"])).expanduser().resolve().parent / (f"cv{n_folds}" + ("-subject" if strategy == "subject" else ""))
    )
    fold_dir = base_dir / f"fold_{fold}"
    split_csv = fold_dir / "block_splits.csv"
    normalization_json = fold_dir / "normalization.json"
    manifest_csv = base_dir / "manifest.csv"
    # Resolved configs (including run snapshots) already point at their own files.
    uses_resolved_paths = (
        (data.get("cv_fold"), data.get("cv_folds")) == (fold, n_folds)
        and Path(str(data.get("cv_split_dir", base_dir))).expanduser().resolve() == base_dir.resolve()
    )
    if uses_resolved_paths:
        split_csv = Path(str(data["split_csv"]))
        normalization_json = Path(str(data["normalization_json"]))
        manifest_csv = Path(str(data["manifest_csv"]))
    missing = [str(path) for path in (manifest_csv, split_csv, normalization_json) if not path.is_file()]
    if missing:
        strategy_hint = " --cv-strategy subject" if strategy == "subject" else ""
        raise FileNotFoundError(
            "Missing CV artifacts: " + ", ".join(missing) + ". "
            f"Run scripts/prepare_data.py --experiments 1 --cv-folds {n_folds}{strategy_hint} "
            f"--cv-output-dir {base_dir} first."
        )
    split_map = load_split_map(split_csv)
    if set(split_map.values()) != {"train", "val"}:
        raise ValueError("CV splits must contain train and val only (no test)")
    rows = read_csv_rows(split_csv)
    if {row.get("cv_strategy", "block") for row in rows} != {strategy}:
        raise ValueError("CV strategy differs from the selected fold artifacts")
    if any(row.get("cv_fold") != str(fold) or row.get("cv_folds") != str(n_folds) for row in rows):
        raise ValueError("CV split file fold identity does not match the requested fold")
    val_blocks: dict[str, list[int]] = {}
    for (subject_uid, block), split in split_map.items():
        if split == "val":
            val_blocks.setdefault(subject_uid, []).append(block)
    val_blocks = {uid: sorted(blocks) for uid, blocks in sorted(val_blocks.items())}
    resolved = dict(data)
    resolved["manifest_csv"] = str(manifest_csv.resolve())
    resolved["split_csv"] = str(split_csv.resolve())
    resolved["normalization_json"] = str(normalization_json.resolve())
    resolved["cv_fold"] = fold
    resolved["cv_folds"] = n_folds
    resolved["cv_split_dir"] = str(base_dir.resolve())
    metadata = {
        "cv_fold": fold,
        "cv_folds": n_folds,
        "cv_split_dir": str(base_dir.resolve()),
        "cv_split_sha256": hashlib.sha256(split_csv.read_bytes()).hexdigest(),
        "cv_val_blocks": val_blocks,
    }
    if strategy == "subject":
        resolved.update(cv_strategy="subject", normalization_mode=data.get("normalization_mode", "train_global"))
        if not uses_resolved_paths:
            for split in ("train", "val"):
                resolved.pop(f"cv_{split}_subjects", None)
        membership = validate_data_protocol(resolved)
        for split in ("train", "val"):
            resolved[f"cv_{split}_subjects"] = metadata[f"cv_{split}_subjects"] = membership[split]
        metadata["cv_strategy"] = "subject"
    elif data.get("normalization_mode") == "train_global":
        validate_data_protocol(resolved)
    return resolved, metadata


def configure_cross_validation(
    config: MutableMapping[str, object],
    *,
    cv_fold: int | None = None,
    cv_folds: int | None = None,
    cv_output_dir: Path | None = None,
    resume_from: Path | None = None,
    cv_strategy: str | None = None,
) -> bool:
    """Apply CLI/config CV selection, or restore it from an existing run snapshot."""
    metadata_keys = ("cv_fold", "cv_folds", "cv_split_dir", "cv_split_sha256", "cv_val_blocks",
                     "cv_strategy", "cv_train_subjects", "cv_val_subjects")
    raw_data = config.get("data")
    if not isinstance(raw_data, Mapping):
        raise ValueError("data must be a mapping")
    data = dict(raw_data)
    requested_strategies = [value for value in (cv_strategy, config.get("cv_strategy"), data.get("cv_strategy")) if value is not None]
    if len({normalize_cv_strategy(value) for value in requested_strategies}) > 1:
        raise ValueError("Conflicting cv_strategy settings")
    strategy = requested_strategies[0] if requested_strategies else "block"
    if resume_from is not None:
        from pb2023_eeg.utils.config import load_yaml
        from pb2023_eeg.utils.run import (
            resolve_snapshot_data_artifacts, validate_snapshot_data_artifacts,
        )

        saved = load_yaml(resume_from.resolve().parent / "config_snapshot.yaml")
        for key, requested in (("cv_fold", cv_fold), ("cv_folds", cv_folds)):
            if requested is not None and requested != saved.get(key):
                raise ValueError(f"Cannot resume with different {key}: {requested} != {saved.get(key)}")
        if cv_output_dir is not None:
            raise ValueError("Resume uses the saved CV artifacts; omit --cv-output-dir")
        saved_data = saved.get("data")
        if not isinstance(saved_data, Mapping):
            raise ValueError("resume config_snapshot.yaml has no data mapping")
        saved_strategy = saved.get("cv_strategy", saved_data.get("cv_strategy", "block"))
        if requested_strategies and strategy != saved_strategy:
            raise ValueError("Cannot resume with different cv_strategy")
        if "normalization_mode" in data and data["normalization_mode"] != saved_data.get("normalization_mode", "per_subject"):
            raise ValueError("Cannot resume with different normalization_mode")
        strategy = saved_strategy
        for key in metadata_keys:
            config.pop(key, None)
            if key in saved:
                config[key] = saved[key]
        for key in ("manifest_csv", "split_csv", "normalization_json", "cv_fold", "cv_folds", "cv_split_dir", "graph_laplacian_json", "graph_laplacian_sha256", "graph_laplacian_mode", "normalization_mode", "cv_strategy", "cv_train_subjects", "cv_val_subjects"):
            data.pop(key, None)
            if key in saved_data:
                data[key] = saved_data[key]
        cv_fold = saved.get("cv_fold")
        cv_folds = saved.get("cv_folds")
        cv_output_dir = saved.get("cv_split_dir")
        data = resolve_snapshot_data_artifacts(data, resume_from.resolve().parent)
        validate_snapshot_data_artifacts(
            data, resume_from.resolve().parent, required=cv_fold is not None,
            expected_split_sha256=saved.get("cv_split_sha256"),
        )
    else:
        cv_fold = config.get("cv_fold", data.get("cv_fold")) if cv_fold is None else cv_fold
        cv_folds = config.get("cv_folds", data.get("cv_folds")) if cv_folds is None else cv_folds
        cv_output_dir = cv_output_dir or config.get("cv_split_dir")
    resolved, metadata = resolve_cv_data_config(
        data, cv_fold=cv_fold, cv_folds=cv_folds, cv_output_dir=cv_output_dir,
        cv_strategy=strategy,
    )
    config["data"] = resolved
    if metadata is None:
        return False
    if resume_from is not None and metadata.get("cv_split_sha256") != config.get("cv_split_sha256"):
        raise ValueError("Resume CV split snapshot has changed since training")
    config.update(metadata)
    plots = config.get("plots") or {}
    if not isinstance(plots, Mapping):
        raise ValueError("plots must be a mapping")
    config["plots"] = {**plots, "splits": ["val"]}
    return True


def resolve_target_time_window_options(data: Mapping[str, object]) -> tuple[int, ...] | None:
    """Target heads and attention indices require one fixed time-axis length."""
    samples = data.get("dynamic_time_window_samples")
    if samples is not None and (isinstance(samples, bool)
                                or not isinstance(samples, Integral) or samples < 1):
        raise ValueError("data.dynamic_time_window_samples must be a positive integer")
    options = data.get("dynamic_time_window_sample_options")
    if options is None:
        return None
    if samples is not None:
        raise ValueError("Use either dynamic_time_window_samples or dynamic_time_window_sample_options")
    if (isinstance(options, (str, bytes)) or not isinstance(options, Sequence)
            or not options or any(isinstance(value, bool) or not isinstance(value, Integral)
                                  or value < 1 for value in options)):
        raise ValueError("data.dynamic_time_window_sample_options must contain positive integers")
    resolved = tuple(dict.fromkeys(int(value) for value in options))
    if len(resolved) != 1:
        raise ValueError("Target requires a fixed time axis: data.dynamic_time_window_sample_options "
                         "must contain one distinct length")
    return resolved


def build_lag_dataset(
    data: Mapping[str, object],
    *,
    task: str,
    split: str,
    seed: int,
    records: Sequence[SubjectRecord] | None = None,
) -> LagDecodingDataset:
    """Honor snapshot selection and apply training-only options only to train."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported dataset split: {split!r}")
    lags = normalize_lags(data.get("lags", [1]))
    time_window_options = resolve_target_time_window_options(data)
    if task != "pitch" and "pitch_class_indices" in data:
        raise ValueError("data.pitch_class_indices is only supported for pitch")
    if records is None:
        records = load_manifest(
            Path(data["manifest_csv"]),
            experiments=[int(exp) for exp in data.get("experiments", [1])],
            subjects=data.get("subjects"),
        )
    validate_data_protocol(data, records)
    training = split == "train"
    return LagDecodingDataset(
        records=records,
        split_csv=Path(data["split_csv"]),
        normalization_json=Path(data["normalization_json"]),
        graph_laplacian_json=data.get("graph_laplacian_json"),
        graph_laplacian_sha256=data.get("graph_laplacian_sha256"),
        graph_laplacian_mode=str(data.get("graph_laplacian_mode", "replace")),
        task=task,
        lags=lags,
        split=split,
        pitch_class_indices=data.get("pitch_class_indices") if task == "pitch" else None,
        require_clean_history=bool(data.get("require_clean_history", True)),
        balance_cells=bool(data.get("balance_train_cells", True) if training
                           else data.get("balance_eval_cells", False)),
        balance_seed=resolve_balance_seed(data, seed),
        time_window_start_s=data.get("time_window_start_s"),
        time_window_end_s=data.get("time_window_end_s"),
        cache_size=int(data.get("cache_size", 1)),
        offline_cache_dir=data.get("offline_cache_dir"),
        choice_cell_history_fields=data.get("choice_cell_history_fields"),
        cell_weighting=data.get("train_cell_weighting", "none") if training else "none",
        cell_weight_max=data.get("train_cell_weight_max") if training else None,
        drop_incomplete_cells=bool(data.get("train_drop_incomplete_cells", False)) if training else False,
        dynamic_cell_sampling=data.get("train_dynamic_cell_sampling", False) if training else False,
        min_cell_class_count=data.get("train_min_cell_class_count") if training else None,
        majority_ratio_cap=data.get("train_majority_ratio_cap") if training else None,
        label_shuffle_within_cells=data.get("train_label_shuffle_within_cells", False) if training else False,
        label_shuffle_seed=data.get("train_label_shuffle_seed") if training else None,
        time_shuffle_within_train=data.get("train_time_shuffle_within_train", False) if training else False,
        time_shuffle_seed=data.get("train_time_shuffle_seed") if training else None,
        dynamic_time_window_samples=data.get("dynamic_time_window_samples"),
        dynamic_time_window_sample_options=time_window_options,
        dynamic_time_window_seed=int(data.get("dynamic_time_window_seed", seed)),
        dynamic_time_window_random=bool(data.get("dynamic_time_window_random", False)) if training else False,
        dynamic_time_window_deterministic=bool(data.get("dynamic_time_window_deterministic", False)) if training else True,
    )
