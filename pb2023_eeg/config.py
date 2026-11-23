"""Two-stage request parsing and binding to a verified data context."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import math

import yaml

from .contracts import ModelSpec
from .models.target import normalize_task_class_indices
from .config_fields import validate_request_fields
from .data.configured import resolve_target_time_window_options
from .data.dataset import normalize_lags


@dataclass(frozen=True)
class ConfigRequest:
    config: dict
    source: Path
    overrides: dict


@dataclass(frozen=True)
class DataContext:
    input_channels: int
    time_s: Sequence[float]
    task_class_indices: Mapping[str, Sequence[int]]
    training_subject_ids: Sequence[int] = ()
    embedding_capacity: int | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    config: dict
    model_spec: ModelSpec
    source: Path


def _merge(base: dict, update: Mapping) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge(dict(result[key]), value)
        else:
            result[key] = deepcopy(value)
    return result


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def parse_request(path: str | Path, overrides: Mapping | None = None) -> ConfigRequest:
    source = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration must be a mapping")
    config = _merge(dict(raw), overrides or {})
    validate_request_fields(config)
    if config.get("schema_version", "pb2023.config.v1") != "pb2023.config.v1":
        raise ValueError("Unsupported request schema_version")
    for section in ("data", "model", "train"):
        if not isinstance(config.get(section), Mapping):
            raise ValueError(f"{section} must be a mapping")
    if "seed" in config:
        _integer(config["seed"], "seed")
    if "lag" in config:
        _integer(config["lag"], "lag")
    if "lags" in config["data"]:
        normalize_lags(config["data"]["lags"])
    resolve_target_time_window_options(config["data"])
    if "task" in config and config["task"] not in {"choice", "pitch", "motor"}:
        raise ValueError("task must be choice, pitch or motor")
    # Preserve legacy defaults. Relative references belong to the YAML, not cwd.
    path_keys = {"manifest_csv", "split_csv", "normalization_json", "graph_laplacian_json",
                 "offline_cache_dir", "cv_output_dir", "cv_split_dir", "base_output_dir", "output_root", "output_dir", "checkpoint", "electrodes_npz"}
    boolean_keys = {"enabled", "amp", "progress_bar", "time_no_class_enabled", "pool_valid_context_only",
                    "attention_score_centering", "attention_gate_centering", "require_clean_history",
                    "balance_train_cells", "balance_eval_cells", "train_drop_incomplete_cells", "nesterov"}
    def visit(mapping, prefix=""):
        for key, value in list(mapping.items()):
            name = prefix + str(key)
            if key in boolean_keys and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean")
            if isinstance(value, Mapping):
                mapping[key] = dict(value)
                visit(mapping[key], name + ".")
            elif key in path_keys and value is not None:
                if not isinstance(value, (str, Path)):
                    raise ValueError(f"{name} must be a path")
                file = Path(value).expanduser()
                mapping[key] = str((source.parent / file).resolve() if not file.is_absolute() else file.resolve())
    visit(config)
    return ConfigRequest(config, source, deepcopy(dict(overrides or {})))


def resolve_request(request: ConfigRequest, data: DataContext) -> ResolvedConfig:
    channels = _integer(data.input_channels, "data_context.input_channels", minimum=1)
    times = [float(t) for t in data.time_s]
    if not times or not all(math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("data_context.time_s must be finite and strictly increasing")
    config = deepcopy(request.config)
    model = config["model"]
    encoder = model.get("encoder")
    if not isinstance(encoder, Mapping):
        raise ValueError("model.encoder must be a mapping")
    configured = encoder.get("in_channels", "auto")
    if configured != "auto" and _integer(configured, "model.encoder.in_channels", minimum=1) != channels:
        raise ValueError("model.encoder.in_channels conflicts with data")
    saved_time = model.get("resolved_time_point_count")
    if saved_time is not None and _integer(saved_time, "model.resolved_time_point_count", minimum=1) != len(times):
        raise ValueError("model.resolved_time_point_count conflicts with data")
    classes = normalize_task_class_indices(data.task_class_indices)
    if "resolved_task_class_indices" in model and normalize_task_class_indices(model["resolved_task_class_indices"]) != classes:
        raise ValueError("model.resolved_task_class_indices conflicts with data")
    ids = [_integer(i, "training_subject_ids") for i in data.training_subject_ids]
    if len(ids) != len(set(ids)):
        raise ValueError("training_subject_ids must be unique")
    capacity = data.embedding_capacity
    if capacity is not None:
        capacity = _integer(capacity, "embedding_capacity", minimum=1)
        if ids and max(ids) >= capacity:
            raise ValueError("Embedding capacity does not cover training subjects")
    model["encoder"] = {**encoder, "in_channels": channels}
    model["resolved_time_point_count"] = len(times)
    model["resolved_task_class_indices"] = {k: list(v) for k, v in classes.items()}
    subject = dict(model.get("subject_conditioning") or {})
    if subject.get("mode", "individual") == "shared_plus_residual":
        if not ids:
            raise ValueError("shared_plus_residual requires training subject IDs")
        if subject.get("resolved_training_subject_ids") is not None and sorted(subject["resolved_training_subject_ids"]) != sorted(ids):
            raise ValueError("Configured training subject IDs conflict with data context")
        subject["resolved_training_subject_ids"] = sorted(ids)
        capacity = max(ids) + 1 if capacity is None else capacity
    model["subject_conditioning"] = subject
    return ResolvedConfig(config, ModelSpec("target", deepcopy(model), channels, len(times), capacity), request.source)
