"""Strict loading of historical and current Target checkpoints.

No old source directory is added to sys.path. Parameter names and class loading
hooks remain compatible, while configuration determines parameter-free behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import pickle

import torch
import yaml

from ..contracts import LoadedModel, ModelSpec
from ..models.factory import build_model
from .compatibility import resolve_model_config_for_checkpoint
from .provenance import file_sha256, model_semantics, runtime_implementation_identity


def _model_spec(value) -> ModelSpec:
    if not isinstance(value, Mapping) or set(value) != {
        "schema", "family", "model_config", "input_channels", "time_point_count", "embedding_capacity"
    }:
        raise ValueError("Invalid saved ModelSpec fields")
    if value["schema"] != "pb2023.model.v1" or value["family"] != "target":
        raise ValueError("Unsupported Target ModelSpec schema/family")
    if not isinstance(value["model_config"], Mapping):
        raise ValueError("ModelSpec.model_config must be a mapping")
    for key in ("input_channels", "time_point_count", "embedding_capacity"):
        item = value[key]
        if item is None and key != "input_channels":
            continue
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ValueError(f"ModelSpec.{key} must be a positive integer")
    return ModelSpec(**deepcopy(dict(value)))


def _build_spec_model(spec: ModelSpec, config: Mapping | None = None):
    with torch.random.fork_rng(devices=[]):
        return build_model(spec.model_config if config is None else config,
            input_channels=spec.input_channels, time_point_count=spec.time_point_count,
            n_subjects=spec.embedding_capacity,
            task_class_indices=spec.model_config.get("resolved_task_class_indices"))


def _bind_time_axis(spec: ModelSpec, *model_configs: Mapping) -> ModelSpec:
    """Keep data-axis metadata when normalizing construction defaults."""
    key = "resolved_time_axis_sha256"
    values = [cfg.get(key) for cfg in (spec.model_config, *model_configs) if cfg.get(key) is not None]
    if not values:
        return spec
    if any(not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value)
           for value in values):
        raise ValueError("Invalid resolved time axis SHA256")
    if any(value != values[0] for value in values):
        raise ValueError("Checkpoint configuration conflicts with ModelSpec time axis")
    return replace(spec, model_config={**spec.model_config, key: values[0]})


def _validate_model_config(model, spec: ModelSpec, config: Mapping) -> None:
    """Compare resolved behavior, including defaults that state_dict cannot encode."""
    try:
        requested = _build_spec_model(spec, config)
        requested.load_state_dict(model.state_dict(), strict=True)
    except (ValueError, TypeError, RuntimeError, KeyError) as exc:
        raise ValueError("Model configuration conflicts with ModelSpec/model state") from exc
    if model_semantics(requested)["modules"] != model_semantics(model)["modules"]:
        raise ValueError("Model configuration conflicts with ModelSpec/model forward semantics")


def read_checkpoint(path: str | Path, *, trusted_legacy: bool = False) -> dict:
    path = Path(path).expanduser().resolve(strict=True)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if not trusted_legacy:
            raise ValueError("Legacy pickle checkpoint requires trusted_legacy=True: " + str(path)) from exc
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Checkpoint must contain a mapping: {path}")
    return dict(payload)


def _config(value: Mapping | str | Path) -> dict:
    if not isinstance(value, Mapping):
        value = yaml.safe_load(Path(value).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Checkpoint configuration must be a mapping")
    return deepcopy(dict(value))


def load_model(path: str | Path, *, config: Mapping | str | Path | None = None,
               device: str | torch.device = "cpu", trusted_legacy: bool = False,
               expected_sha256: str | None = None) -> LoadedModel:
    path = Path(path).expanduser().resolve(strict=True)
    sha = file_sha256(path)
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    saved = read_checkpoint(path, trusted_legacy=trusted_legacy)
    if file_sha256(path) != sha:
        raise ValueError(f"Checkpoint changed while loading: {path}")
    if saved.get("schema") == "baseline.checkpoint.v1":
        from ..models.baselines.loading import load_baseline
        return load_baseline(path, saved=saved, config=config, device=device, sha=sha)
    embedded = saved.get("config")
    if config is not None:
        resolved = _config(config)
        if saved.get("model_spec") is None and isinstance(embedded, Mapping) and embedded.get("model") != resolved.get("model"):
            raise ValueError("Supplied model configuration conflicts with checkpoint configuration")
    elif isinstance(embedded, Mapping):
        resolved = _config(embedded)
    elif (path.parent / "config_snapshot.yaml").is_file():
        resolved = _config(path.parent / "config_snapshot.yaml")
    else:
        raise ValueError("A checkpoint needs its saved configuration; weights do not encode gate semantics")
    state = saved.get("model_state")
    if not isinstance(state, Mapping) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Checkpoint model_state must be a nonempty mapping of tensors")
    if not isinstance(resolved.get("model"), Mapping):
        raise ValueError("Checkpoint config.model is missing")
    if saved.get("model_spec") is not None:
        spec = _model_spec(saved["model_spec"])
        spec = _bind_time_axis(spec, resolved["model"],
                              embedded.get("model", {}) if isinstance(embedded, Mapping) else {})
        model = _build_spec_model(spec)
        actual_spec = _model_spec(model._pb_model_spec)
        if (actual_spec.input_channels, actual_spec.time_point_count, actual_spec.embedding_capacity) != (
            spec.input_channels, spec.time_point_count, spec.embedding_capacity
        ):
            raise ValueError("ModelSpec dimensions conflict with its model configuration")
        model.load_state_dict(state, strict=True)
        if config is not None and isinstance(embedded, Mapping):
            _validate_model_config(model, spec, embedded["model"])
        _validate_model_config(model, spec, resolved["model"])
        resolved["model"] = deepcopy(dict(actual_spec.model_config))
        return LoadedModel(model.to(device).eval(), actual_spec, resolved, path, sha, resolved.get("task"),
                           {"schema": saved.get("schema", "pb2023.checkpoint.v1"),
                            "iteration": saved.get("iteration"), "strict": True})
    model_config = resolve_model_config_for_checkpoint(resolved["model"], state)
    capacity = None
    weight = state.get("subject_embedding.weight")
    if weight is not None:
        if weight.ndim != 2 or min(weight.shape) < 1:
            raise ValueError("Invalid subject_embedding.weight shape")
        capacity = int(weight.shape[0])
        subject = deepcopy(model_config.get("subject_conditioning") or {})
        ids = state.get("subject_embedding.training_subject_ids")
        if ids is not None:
            if ids.dtype != torch.long or ids.ndim != 1 or not ids.numel() or not torch.equal(ids, ids.unique(sorted=True)) or bool((ids < 0).any()) or bool((ids >= capacity).any()):
                raise ValueError("Invalid saved training subject IDs")
            if subject.get("mode", "shared_plus_residual") != "shared_plus_residual":
                raise ValueError("Subject mode conflicts with saved shared residual embedding")
            subject["mode"] = "shared_plus_residual"
            if subject.get("resolved_training_subject_ids") is not None and list(subject["resolved_training_subject_ids"]) != ids.tolist():
                raise ValueError("Saved training subject IDs conflict with configuration")
            subject["resolved_training_subject_ids"] = ids.tolist()
        model_config["subject_conditioning"] = subject
    channels = model_config.get("encoder", {}).get("in_channels")
    spatial = state.get("encoder.spatial.weight")
    if spatial is None:
        spatial = state.get("encoder.spatial_projection.0.weight")
    if isinstance(spatial, torch.Tensor):
        inferred = int(spatial.shape[1])
        if channels in (None, "auto"):
            channels = inferred
        elif int(channels) != inferred:
            raise ValueError("Input channels conflict with checkpoint spatial projection")
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        raise ValueError("Checkpoint must record the resolved model.encoder.in_channels")
    time_count = model_config.get("resolved_time_point_count")
    mode = str(model_config.get("time_head_mode", "shared")).replace("-", "_")
    if mode != "shared":
        sizes = {int(v.shape[0]) for k, v in state.items()
                 if k.startswith("class_heads.") and k.endswith(("weight", "bias")) and v.ndim >= 2}
        if len(sizes) != 1:
            raise ValueError("Checkpoint time heads disagree on time support")
        inferred = next(iter(sizes))
        if time_count is None:
            time_count = inferred
        elif int(time_count) != inferred:
            raise ValueError("Time support conflicts with checkpoint heads")
    # Construction must not consume the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        model = build_model(model_config, n_subjects=capacity, time_point_count=time_count,
                            input_channels=channels)
    model.load_state_dict(state, strict=True)
    spec = _model_spec(model._pb_model_spec)
    resolved["model"] = deepcopy(dict(spec.model_config))
    return LoadedModel(model.to(device).eval(), spec, resolved, path, sha, resolved.get("task"),
                       {"schema": saved.get("schema", "legacy.target"),
                        "iteration": saved.get("iteration"), "strict": True})


def save_model(path: str | Path, model: torch.nn.Module, *, config: Mapping) -> None:
    """Atomically save an inference checkpoint with its complete construction config."""
    from ..training.common import atomic_torch_save
    runtime_implementation_identity()
    if not isinstance(config.get("model"), Mapping):
        raise ValueError("config.model is required")
    if getattr(model, "_pb_model_spec", None) is None:
        raise ValueError("save_model requires a complete ModelSpec; construct with build_model or load_model")
    spec = _model_spec(model._pb_model_spec)
    spec = _bind_time_axis(spec, config["model"])
    _validate_model_config(model, spec, spec.model_config)
    _validate_model_config(model, spec, config["model"])
    resolved = deepcopy(dict(config))
    resolved["model"] = deepcopy(dict(spec.model_config))
    atomic_torch_save({"schema": "pb2023.checkpoint.v1", "model_state": model.state_dict(),
                       "config": resolved, "model_spec": spec.to_dict()}, Path(path))
