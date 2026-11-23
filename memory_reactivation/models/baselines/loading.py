"""Load native spatial baseline bundles with their recorded fold identity."""
from __future__ import annotations

from copy import deepcopy
import json
from collections.abc import Mapping

import torch

from ...artifacts.provenance import digest
from ...contracts import LoadedModel, ModelSpec
from .spatial import SpatialTrialDecoder


def load_baseline(path, *, saved, config, device, sha):
    identity_path = path.parent / "identity.json"
    resolved_path = path.parent / "resolved_model.json"
    if not identity_path.is_file() or not resolved_path.is_file():
        raise FileNotFoundError("Native baseline requires adjacent identity.json and resolved_model.json")
    identity = json.loads(identity_path.read_text())
    native = identity["identity"]
    if digest(native) != identity.get("identity_sha256") or saved.get("identity_sha256") != digest(native):
        raise ValueError("Baseline checkpoint/config identity mismatch")
    if saved.get("cv_fold") != native.get("fold"):
        raise ValueError("Baseline checkpoint fold identity mismatch")
    recorded_config = native.get("config")
    if not isinstance(recorded_config, Mapping):
        raise ValueError("Baseline identity has no saved config")
    if config is not None:
        from ...artifacts.checkpoints import _config
        if _config(config) != recorded_config:
            raise ValueError("Supplied baseline config conflicts with recorded identity")
    config = deepcopy(dict(recorded_config))
    resolved = json.loads(resolved_path.read_text())
    if resolved.get("type") != "spatial_only":
        raise ValueError("This baseline loader supports spatial_only checkpoints")
    encoder = dict(config["model"]["encoder"])
    for key in ("spatial_filters", "dropout", "activation", "spatial_max_norm", "classifier_max_norm"):
        if key in encoder and key in resolved and encoder[key] != resolved[key]:
            raise ValueError(f"Resolved baseline {key} conflicts with config")
    state = saved.get("model_state")
    if not isinstance(state, Mapping) or "head.weight" not in state:
        raise ValueError("Missing baseline model_state/head.weight")
    classes = int(state["head.weight"].shape[0])
    channels, times = int(resolved["in_channels"]), int(resolved["n_times"])
    subject_spatial = bool(resolved.get("subject_specific_spatial", False))
    subject_count = int(resolved.get("subject_count", 0))
    with torch.random.fork_rng(devices=[]):
        model = SpatialTrialDecoder(channels, times, classes, encoder,
                                    subject_count=subject_count, subject_spatial=subject_spatial)
    model.load_state_dict(state, strict=True)
    task = config.get("task")
    if task not in {"choice", "pitch", "motor"}:
        raise ValueError("Baseline identity must record its trained task")
    model._pb_trained_task = task
    spec = ModelSpec("spatial_baseline", {"model": config["model"], "resolved": resolved,
                                         "class_values": config.get("target_class_values")},
                     channels, times, subject_count or None)
    model._pb_model_spec = spec.to_dict()
    return LoadedModel(model.to(device).eval(), spec, config, path, sha, config.get("task"),
                       {"schema": saved["schema"], "fold": saved["cv_fold"],
                        "iteration": saved.get("iteration"), "strict": True})
