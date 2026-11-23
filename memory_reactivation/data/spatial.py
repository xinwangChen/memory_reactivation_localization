"""Optional, geometry-only graph Laplacian applied before channel normalization.

The operator is a local weighted difference, I - D^-1 W, not spherical-spline
CSD. Graph construction never uses EEG or labels. Frozen JSON contains the full
geometry, weights and matrix so inference does not depend on a template install.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np


DEFAULT_ELECTRODES = Path("/nfs1/active_construction/PB2023_preprocessed/electrodes/Exp1_electrodes.npz")
METHOD = "sphere_knn_gaussian_random_walk_v1"
GRAPH_INPUT_MODES = ("replace", "concat")


def normalize_graph_input_mode(value: object) -> str:
    """Normalize how the graph feature is exposed to the model input."""
    if value is None:
        return "replace"
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "laplacian": "replace",
        "graph_laplacian": "replace",
        "raw_plus_laplacian": "concat",
        "concat_raw": "concat",
    }
    mode = aliases.get(mode, mode)
    if mode not in GRAPH_INPUT_MODES:
        raise ValueError(
            "graph_laplacian.mode must be one of: "
            + ", ".join(GRAPH_INPUT_MODES)
        )
    return mode


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json_bytes(value))
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def graph_options(preprocessing: Mapping | None) -> dict:
    if preprocessing is None:
        preprocessing = {}
    if not isinstance(preprocessing, Mapping):
        raise ValueError("preprocessing must be a mapping")
    unknown = set(preprocessing) - {"graph_laplacian"}
    if unknown:
        raise ValueError(f"Unknown preprocessing options: {sorted(unknown)}")
    raw = preprocessing.get("graph_laplacian") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("preprocessing.graph_laplacian must be a mapping")
    unknown = set(raw) - {"enabled", "n_neighbors", "electrodes_npz", "mode", "sigma_m"}
    if unknown:
        raise ValueError(f"Unknown graph_laplacian options: {sorted(unknown)}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("graph_laplacian.enabled must be boolean")
    neighbors = raw.get("n_neighbors", 6)
    if isinstance(neighbors, bool) or not isinstance(neighbors, Integral) or neighbors < 1:
        raise ValueError("graph_laplacian.n_neighbors must be a positive integer")
    options = {
        "enabled": enabled,
        "n_neighbors": int(neighbors),
        "electrodes_npz": str(
            Path(raw.get("electrodes_npz", DEFAULT_ELECTRODES))
            .expanduser()
            .resolve()
        ),
        "mode": normalize_graph_input_mode(raw.get("mode")),
    }
    # Omit the optional override when absent to preserve legacy config identity.
    if raw.get("sigma_m") is not None:
        options["sigma_m"] = validate_sigma_m(raw["sigma_m"])
    return options


def validate_sigma_m(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError("graph_laplacian.sigma_m must be finite and positive (meters)")
    return float(value)


def apply_graph_input(
    x: np.ndarray,
    graph: "GraphLaplacian | None",
    mode: str = "replace",
) -> np.ndarray:
    """Return the configured model input from raw EEG and a frozen graph."""
    normalized_mode = normalize_graph_input_mode(mode)
    if graph is None:
        if normalized_mode != "replace":
            raise ValueError("graph input mode concat requires an enabled graph Laplacian")
        return np.asarray(x, dtype=np.float32)
    raw = np.asarray(x, dtype=np.float32)
    if normalized_mode == "replace":
        return graph.apply(raw)
    transformed = graph.apply(raw)
    result = np.concatenate((raw, transformed), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite EEG after graph/raw concatenation")
    return result


def build_graph(electrodes_npz: Path, n_neighbors: int, *, sigma_m: float | None = None) -> dict:
    """Symmetric union of nearest-neighbor edges, weighted by spherical distance.

    Stable channel-index ordering breaks exact distance ties reproducibly. Sigma
    defaults to the median distance of undirected edges, counted once per edge.
    An explicit sigma_m fixes the weight scale across neighbor-count conditions.
    n_neighbors=channel_count-1 gives the full graph, excluding self connections.
    """
    if sigma_m is not None:
        sigma_m = validate_sigma_m(sigma_m)
    with np.load(electrodes_npz, allow_pickle=False) as archive:
        names = archive["channel_names"].tolist()
        positions = np.asarray(archive["positions_m"], dtype=np.float64)
        origin = np.asarray(archive["sphere_origin_m"], dtype=np.float64)
        radius = float(archive["sphere_radius_m"])
        if archive["coord_frame"].item() != "head":
            raise ValueError("Electrodes must use head coordinates in meters")
        if archive["channel_index"].tolist() != list(range(len(names))):
            raise ValueError("Electrode channel order is not sequential")
        template = str(archive["template"].item())
    n = len(names)
    if not all(isinstance(name, str) and name for name in names) or len(set(names)) != n:
        raise ValueError("Electrode names must be unique nonempty strings")
    if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, Integral) or not 1 <= n_neighbors < n:
        raise ValueError("n_neighbors must be an integer between 1 and channel_count - 1")
    if positions.shape != (n, 3) or origin.shape != (3,) or not np.isfinite(positions).all() or not np.isfinite(origin).all():
        raise ValueError("Invalid electrode coordinates")
    if not np.isfinite(radius) or radius <= 0 or np.unique(positions, axis=0).shape[0] != n:
        raise ValueError("Invalid radius or duplicate electrode positions")
    centered = positions - origin
    lengths = np.linalg.norm(centered, axis=1)
    if not np.allclose(lengths, radius, rtol=1e-6, atol=1e-10):
        raise ValueError("Electrodes do not lie on the recorded sphere")
    unit = centered / lengths[:, None]
    distances = radius * np.arccos(np.clip(unit @ unit.T, -1, 1))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbors]
    edges = np.zeros((n, n), dtype=bool)
    edges[np.arange(n)[:, None], nearest] = True
    edges |= edges.T
    reached = {0}
    while True:
        expanded = reached | set(np.flatnonzero(edges[list(reached)].any(axis=0)).tolist())
        if expanded == reached:
            break
        reached = expanded
    if len(reached) != n:
        raise ValueError("Electrode graph is disconnected; increase n_neighbors")
    sigma = float(np.median(distances[np.triu(edges, k=1)])) if sigma_m is None else sigma_m
    weights = np.zeros((n, n), dtype=np.float64)
    weights[edges] = np.exp(-0.5 * (distances[edges] / sigma) ** 2)
    degrees = weights.sum(axis=1)
    if not np.isfinite(sigma) or sigma <= 0 or np.any(degrees <= 0):
        raise ValueError("Degenerate graph weights")
    matrix = np.eye(n) - weights / degrees[:, None]
    payload = {
        "schema_version": 1, "method": METHOD, "experiment": 1,
        "coordinate_source": "standard template, not subject-specific digitization",
        "electrodes_sha256": file_sha256(electrodes_npz), "template": template,
        "channel_names": names, "positions_m": positions.tolist(),
        "sphere_origin_m": origin.tolist(), "sphere_radius_m": radius,
        "n_neighbors": int(n_neighbors), "sigma_m": sigma,
        "neighbor_counts": edges.sum(axis=1).tolist(), "weights": weights.tolist(),
        "matrix": matrix.tolist(), "application_dtype": "float32",
    }
    # Keep the legacy artifact byte-for-byte identical without an override.
    if sigma_m is not None:
        payload.update(sigma_source="explicit", connectivity="full" if n_neighbors == n - 1 else "knn_union")
    return payload


@dataclass(frozen=True)
class GraphLaplacian:
    matrix: np.ndarray
    channel_names: tuple[str, ...]
    sha256: str
    experiment: int

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None):
        digest = file_sha256(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("Graph Laplacian checksum mismatch")
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != 1 or payload.get("method") != METHOD:
            raise ValueError("Unsupported graph Laplacian artifact")
        names = tuple(payload["channel_names"])
        matrix = np.asarray(payload["matrix"], dtype=np.float64)
        weights = np.asarray(payload["weights"], dtype=np.float64)
        n = len(names)
        if len(set(names)) != n or n < 2 or matrix.shape != (n, n) or weights.shape != (n, n):
            raise ValueError("Invalid graph dimensions/channel names")
        if not np.isfinite(matrix).all() or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("Invalid graph values")
        if not np.allclose(weights, weights.T, atol=1e-12, rtol=0) or np.any(np.diag(weights) != 0) or np.any(weights.sum(axis=1) <= 0):
            raise ValueError("Invalid graph adjacency")
        expected = np.eye(n) - weights / weights.sum(axis=1)[:, None]
        if not np.allclose(matrix, expected, atol=1e-12, rtol=0):
            raise ValueError("Invalid graph Laplacian matrix")
        return cls(matrix.astype(np.float32), names, digest, int(payload["experiment"]))

    def apply(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 3 or x.shape[1] != len(self.channel_names):
            raise ValueError("EEG channels do not match the graph Laplacian channel order")
        result = np.empty(x.shape, dtype=np.float32)
        for start in range(0, len(x), 64):
            result[start:start + 64] = np.matmul(self.matrix, np.asarray(x[start:start + 64], dtype=np.float32))
        if not np.isfinite(result).all():
            raise ValueError("Non-finite EEG after graph Laplacian")
        return result


def configure_concat_objective(config: dict, options: Mapping) -> None:
    """Concat experiments use classification only; persist the effective objective."""
    if not options["enabled"] or options["mode"] != "concat":
        return
    model = config.get("model", {})
    train = config.get("train", {})
    if not isinstance(model, Mapping) or not isinstance(train, Mapping):
        raise ValueError("model and train must be mappings")
    reconstruction = model.get("reconstruction") or {}
    if not isinstance(reconstruction, Mapping):
        raise ValueError("model.reconstruction must be a mapping")
    config["model"] = {**model, "reconstruction": {**reconstruction, "enabled": False}}
    config["train"] = {**train, "reconstruction_loss_weight": 0.0}
    print("[preprocessing] concat input: reconstruction disabled (weight=0)", flush=True)


def configure_spatial_preprocessing(config: dict, *, resume_from: Path | None = None,
                                    normalization_cache: dict | None = None) -> None:
    """Resolve an optional training config; prepare fold-specific stats once.

    CV selection must run first. On resume we restore the frozen operator and
    statistics, rejecting explicit changes to preprocessing. No EEG is rewritten.
    """
    from .manifest import load_manifest
    from .normalization import (compute_normalization_stats, load_normalization_stats,
                                normalize_normalization_mode, validate_normalization_identity)

    data = dict(config["data"])
    normalization_mode = normalize_normalization_mode(data.get("normalization_mode", "per_subject"))
    top_level = config.get("preprocessing")
    nested = data.pop("preprocessing", None)
    if top_level is not None and nested is not None and graph_options(top_level) != graph_options(nested):
        raise ValueError("Top-level and data.preprocessing graph Laplacian settings differ")
    options = graph_options(top_level if top_level is not None else nested)
    if top_level is None and nested is not None:
        config["preprocessing"] = nested
    if resume_from is not None:
        from memory_reactivation.utils.config import load_yaml
        from memory_reactivation.utils.run import (
            resolve_snapshot_data_artifacts, validate_snapshot_data_artifacts,
        )
        saved = load_yaml(Path(resume_from).resolve().parent / "config_snapshot.yaml")
        saved_options = graph_options(saved.get("preprocessing"))
        if "preprocessing" in config and options != saved_options:
            raise ValueError("Cannot resume with different graph Laplacian preprocessing")
        if "preprocessing" in saved:
            config["preprocessing"] = saved["preprocessing"]
        saved_mode = normalize_normalization_mode(saved["data"].get("normalization_mode", "per_subject"))
        if "normalization_mode" in data and normalization_mode != saved_mode:
            raise ValueError("Cannot resume with different normalization_mode")
        normalization_mode = saved_mode
        if saved_options["enabled"]:
            saved_data = resolve_snapshot_data_artifacts(
                saved["data"], Path(resume_from).resolve().parent,
            )
            validate_snapshot_data_artifacts(
                saved_data, Path(resume_from).resolve().parent, required=True,
                expected_split_sha256=saved.get("cv_split_sha256"),
            )
            for key in (
                "graph_laplacian_json",
                "graph_laplacian_sha256",
                "graph_laplacian_mode",
                "normalization_json",
                "normalization_mode",
            ):
                # Older snapshots predate the explicit input mode and are replace mode.
                if key == "graph_laplacian_mode":
                    data[key] = saved_data.get(key, saved_options["mode"])
                elif key == "normalization_mode":
                    if key in saved_data:
                        data[key] = saved_data[key]
                else:
                    data[key] = saved_data[key]
            graph = GraphLaplacian.load(Path(data["graph_laplacian_json"]), data["graph_laplacian_sha256"])
            load_normalization_stats(
                Path(data["normalization_json"]),
                graph_sha256=graph.sha256,
                graph_mode=saved_options["mode"],
            )
            if normalization_mode == "train_global":
                records = load_manifest(Path(data["manifest_csv"]), experiments=data.get("experiments", [1]), subjects=data.get("subjects"))
                validate_normalization_identity(records, Path(data["split_csv"]), Path(data["normalization_json"]), mode=normalization_mode)
        config["data"] = data
        configure_concat_objective(config, saved_options)
        return
    if not options["enabled"]:
        if data.get("graph_laplacian_json"):
            raise ValueError("Disabled preprocessing cannot use graph Laplacian artifacts; use the original data config")
        config["data"] = data
        return

    records = load_manifest(Path(data["manifest_csv"]), experiments=data.get("experiments", [1]), subjects=data.get("subjects"))
    if any(record.exp != 1 for record in records):
        raise ValueError("Graph Laplacian currently supports the verified Exp1 montage only")
    payload = build_graph(
        Path(options["electrodes_npz"]), options["n_neighbors"], sigma_m=options.get("sigma_m"),
    )
    graph_sha = hashlib.sha256(json_bytes(payload)).hexdigest()
    input_identity = {
        "graph_sha256": graph_sha,
        "graph_input_mode": options["mode"],
        "split_sha256": file_sha256(Path(data["split_csv"])),
        "records": [{**asdict(record), "size": Path(record.npz_path).stat().st_size,
                     "mtime_ns": Path(record.npz_path).stat().st_mtime_ns,
                     "meta_sha256": file_sha256(Path(record.meta_path))} for record in records],
    }
    if normalization_mode != "per_subject":
        input_identity["normalization_mode"] = normalization_mode
    key = hashlib.sha256(json_bytes(input_identity)).hexdigest()
    # Keep derived statistics separate from the original, including across folds.
    directory = Path(data["split_csv"]).resolve().parent / "spatial" / key
    directory.mkdir(parents=True, exist_ok=True)
    graph_path = directory / "graph_laplacian.json"
    norm_path = directory / "normalization.json"
    with (directory / "prepare.lock").open("a") as lock:
        fcntl.lockf(lock.fileno(), fcntl.LOCK_EX)
        if not graph_path.exists():
            atomic_json(graph_path, payload)
        graph = GraphLaplacian.load(graph_path, graph_sha)
        if not norm_path.exists():
            print("[preprocessing] computing graph-Laplacian train-only normalization", flush=True)
            with tempfile.TemporaryDirectory(dir=directory) as temporary:
                candidate = Path(temporary) / "normalization.json"
                compute_normalization_stats(
                    records,
                    Path(data["split_csv"]),
                    candidate,
                    graph=graph,
                    graph_mode=options["mode"],
                    **({"mode": normalization_mode, "moments_cache": normalization_cache}
                       if normalization_mode == "train_global" else {}),
                )
                candidate.replace(norm_path)
            atomic_json(directory / "inputs.json", input_identity)
        load_normalization_stats(norm_path, graph_sha256=graph_sha, graph_mode=options["mode"])
        if normalization_mode == "train_global":
            validate_normalization_identity(records, Path(data["split_csv"]), norm_path, mode=normalization_mode)
    data.update(
        graph_laplacian_json=str(graph_path),
        graph_laplacian_sha256=graph_sha,
        graph_laplacian_mode=options["mode"],
        normalization_json=str(norm_path),
    )
    config["data"] = data
    config["preprocessing"] = {"graph_laplacian": options}
    configure_concat_objective(config, options)
    print(
        "[preprocessing] graph Laplacian enabled: "
        f"mode={options['mode']}, neighbors={options['n_neighbors']}, "
        f"sigma_m={payload['sigma_m']:.10g}, sha256={graph_sha[:12]}",
        flush=True,
    )
