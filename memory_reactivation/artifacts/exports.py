"""Bind reusable exports to their actual inference parameters and complete CSV set."""

from __future__ import annotations

from functools import wraps
import hashlib
import inspect
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Callable, Mapping, Sequence

import torch
from ..models.adapters import boundary_noise_metadata
from .provenance import data_identity, digest, runtime_implementation_identity, model_semantics


TIME_CORE_STEMS = (
    "time_accuracy", "time_activation", "time_confusion", "time_class_metrics",
    "time_cell_metrics", "time_cell_summary",
)
TARGET_EXPORT_STEMS = (
    "trial_metrics", "attention_weights", *TIME_CORE_STEMS, "subject_metrics",
)
TIME_EXPORT_STEMS = (
    "trial_metrics", "subject_metrics", "time_trial_scores", "time_trial_summary",
    *TIME_CORE_STEMS,
)
# Remove obsolete optional tables of this condition as part of a successful replacement.
MANAGED_STEMS = tuple(dict.fromkeys((*TARGET_EXPORT_STEMS, *TIME_EXPORT_STEMS)))


def export_prefix(split: str, input_kind: str) -> str:
    return f"{split}_" if input_kind == "real" else f"{split}_{input_kind}_"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        digest.update(raw.numpy().tobytes())
    return digest.hexdigest()


def _parameters(decoder: str, arguments: Mapping[str, object]) -> dict[str, object]:
    """Normalize aliases/defaults identically for fresh exports and reuse checks."""
    implementation = runtime_implementation_identity()
    model = arguments["model"]
    input_kind = str(arguments["input_kind"])
    pooled_mode = arguments.get("score_mode") or arguments.get("pooled_score_mode")
    fusion_mode = arguments.get("fusion_score_mode")
    if fusion_mode is None:
        fusion_mode = pooled_mode
    indices = arguments.get("attention_indices", arguments.get("fusion_indices"))
    parameters = {
        "identity_schema": "pb2023.export.v2",
        "decoder": decoder,
        "task": arguments["task"],
        "split": arguments["split"],
        "input_kind": input_kind,
        "model_state_sha256": _model_sha256(model),
        "model_semantics_sha256": digest(model_semantics(model)),
        "implementation_sha256": implementation["sha256"],
        "implementation_identity_schema": implementation["schema"],
        "data_sha256": digest(data_identity(arguments["loader"])),
        "fusion_enabled": bool(getattr(model, "fusion_enabled", False)),
        "time_s": torch.as_tensor(arguments["time_s"]).reshape(-1).tolist(),
        "time_indices": None if indices is None else indices.detach().cpu().tolist(),
        "amp": bool(arguments["amp"]),
        "activation_threshold": float(arguments["activation_threshold"]),
        "min_cell_class_count": int(arguments["min_cell_class_count"]),
        "time_score_mode": str(arguments["time_score_mode"]).strip().lower(),
        "fusion_score_mode": fusion_mode,
    }
    boundary = boundary_noise_metadata(model)
    if boundary is not None:
        parameters["boundary_noise"] = boundary
    if decoder == "target":
        parameters["pooled_score_mode"] = pooled_mode
    else:
        parameters["export_trial_time_scores"] = bool(arguments["export_trial_time_scores"])
        parameters["metadata"] = dict(arguments.get("metadata") or {})
    if input_kind == "gaussian":
        parameters.update(gaussian_seed=arguments["gaussian_seed"],
                          gaussian_noise_mode=arguments["gaussian_noise_mode"])
    elif input_kind == "time_shuffle":
        parameters["time_shuffle_seed"] = arguments["time_shuffle_seed"]
    elif input_kind == "time_roll":
        parameters["time_roll_shift_samples"] = arguments["time_roll_shift_samples"]
    # JSON round trip also normalizes tuples in time-only export metadata.
    return json.loads(json.dumps(parameters, allow_nan=False))


def describe_export(exporter: Callable, **arguments: object) -> dict[str, object]:
    bound = inspect.signature(exporter).bind(**arguments)
    bound.apply_defaults()
    return _parameters(exporter.export_decoder, bound.arguments)


def _required_names(parameters: Mapping[str, object]) -> set[str]:
    if parameters["decoder"] == "target":
        stems = TARGET_EXPORT_STEMS
    elif parameters["decoder"] == "time":
        stems = (*TIME_CORE_STEMS,
                 *(("trial_metrics", "subject_metrics") if parameters["fusion_enabled"] else ()),
                 *(("time_trial_scores", "time_trial_summary")
                   if parameters["export_trial_time_scores"] else ()))
    else:
        raise ValueError("Unknown export decoder")
    prefix = export_prefix(parameters["split"], parameters["input_kind"])
    return {f"{prefix}{stem}.csv" for stem in stems}


def _managed_names(split: str, input_kind: str) -> set[str]:
    prefix = export_prefix(split, input_kind)
    return {f"{prefix}{stem}.csv" for stem in MANAGED_STEMS} | {f"{prefix}export_manifest.json"}


def _publish(staging: Path, destination: Path, managed: set[str]) -> None:
    """Validate/stage first; restore the old set if replacement raises an exception."""
    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".export-backup-", dir=destination) as temporary:
        backup = Path(temporary)
        moved: list[str] = []
        installed: list[str] = []
        try:
            for name in sorted(managed):
                previous = destination / name
                if previous.exists():
                    previous.replace(backup / name)
                    moved.append(name)
            # Publish manifests last: a partial export never has a valid completion record.
            for path in sorted(staging.iterdir(), key=lambda p: (p.suffix == ".json", p.name)):
                path.replace(destination / path.name)
                installed.append(path.name)
        except BaseException:
            for name in installed:
                (destination / name).unlink(missing_ok=True)
            for name in moved:
                (backup / name).replace(destination / name)
            raise


def verified_export(decoder: str) -> Callable:
    """Generate each condition in isolation, then publish CSVs and their provenance."""
    def decorate(exporter: Callable) -> Callable:
        signature = inspect.signature(exporter)

        @wraps(exporter)
        def wrapped(*args: object, **kwargs: object):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            parameters = _parameters(decoder, bound.arguments)
            destination = Path(bound.arguments["output_dir"])
            destination.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".export-stage-", dir=destination) as temporary:
                staging = Path(temporary)
                bound.arguments["output_dir"] = staging
                result = exporter(*bound.args, **bound.kwargs)
                runtime_implementation_identity()
                if digest(data_identity(bound.arguments["loader"])) != parameters["data_sha256"]:
                    raise ValueError("Export inputs changed during generation")
                required = _required_names(parameters)
                missing = sorted(name for name in required if not (staging / name).is_file())
                if missing:
                    raise ValueError(f"Incomplete export (possibly an empty loader); missing {missing}")
                manifest = {"schema_version": 2, "parameters": parameters,
                            "files": {name: file_sha256(staging / name) for name in sorted(required)}}
                prefix = export_prefix(parameters["split"], parameters["input_kind"])
                (staging / f"{prefix}export_manifest.json").write_text(
                    json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                _publish(staging, destination,
                         _managed_names(parameters["split"], parameters["input_kind"]))
            return result

        wrapped.export_decoder = decoder
        return wrapped
    return decorate


def copy_verified_exports(
    source: Path, destination: Path, *, decoder: str, splits: Sequence[str],
    input_kinds: Sequence[str],
    expected_parameters: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
) -> tuple[Path, ...]:
    """Check every requested condition before replacing any destination file.

    Old CSV-only exports cannot prove their control parameters. Regenerate them
    with this exporter; never manufacture provenance from the current request.
    """
    if not input_kinds:
        return ()
    groups = []
    managed: set[str] = set()
    for split in splits:
        for input_kind in input_kinds:
            prefix = export_prefix(split, input_kind)
            manifest_name = f"{prefix}export_manifest.json"
            path = source / manifest_name
            if not path.is_file():
                raise ValueError(
                    f"Cannot safely reuse {split} {input_kind}: missing export provenance {path}. "
                    "Re-export without --reuse-existing-kinds to regenerate verified results."
                )
            manifest_bytes = path.read_bytes()
            manifest = json.loads(manifest_bytes)
            if not isinstance(manifest, dict):
                raise ValueError(f"Invalid export provenance: {path}")
            parameters = manifest.get("parameters")
            if (manifest.get("schema_version") != 2 or not isinstance(parameters, dict)
                    or any(parameters.get(key) != value for key, value in
                           (("decoder", decoder), ("split", split), ("input_kind", input_kind)))):
                raise ValueError(f"Invalid export provenance: {path}")
            if expected_parameters is not None:
                expected = expected_parameters[(split, input_kind)]
                differences = sorted(key for key in set(expected) | set(parameters)
                                     if parameters.get(key) != expected.get(key))
                if differences:
                    raise ValueError(
                        f"Cannot reuse {split} {input_kind}: export parameter mismatch {differences}. "
                        "Re-export without --reuse-existing-kinds to apply the requested parameters."
                    )
            files = manifest.get("files")
            if not isinstance(files, dict) or set(files) != _required_names(parameters):
                raise ValueError(f"Incomplete export file inventory: {path}")
            for name, expected_hash in files.items():
                if not (source / name).is_file():
                    raise FileNotFoundError(f"Cannot reuse incomplete export; missing {source / name}")
                if file_sha256(source / name) != expected_hash:
                    raise ValueError(f"Export checksum mismatch: {source / name}")
            groups.append((manifest_name, manifest_bytes, files))
            managed.update(_managed_names(split, input_kind))

    destination.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".export-reuse-", dir=destination) as temporary:
        staging = Path(temporary)
        for manifest_name, manifest_bytes, files in groups:
            for name, expected_hash in files.items():
                shutil.copy2(source / name, staging / name)
                if file_sha256(staging / name) != expected_hash:
                    raise ValueError(f"Export changed while copying: {source / name}")
            (staging / manifest_name).write_bytes(manifest_bytes)
        copied = tuple(destination / name for _, _, files in groups for name in files)
        _publish(staging, destination, managed)
    return copied
