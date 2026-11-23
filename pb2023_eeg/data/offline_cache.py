"""Content-bound, immutable float32 EEG caches with a verified sidecar.

Version 2 does not reuse the older path/mtime/size cache names. Missing or
interrupted entries are rebuilt under the existing per-entry POSIX lock;
complete entries with inconsistent contents are rejected, never re-signed.
"""

import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from ..utils.source_files import content_sha256
from .spatial import apply_graph_input


def cache_identity(record, graph, mode):
    return {
        "schema": "pb2023.eeg_cache.v2",
        "npz_sha256": content_sha256(record.npz_path),
        "graph_sha256": None if graph is None else graph.sha256,
        "graph_mode": mode,
        "dtype": "float32",
    }


def offline_cache_path(record, directory, graph, mode, *, identity=None):
    identity = cache_identity(record, graph, mode) if identity is None else identity
    key = hashlib.sha256(json.dumps(
        {"path": str(Path(record.npz_path).resolve()), **identity},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()[:24]
    return Path(directory) / f"{record.subject_uid}_v2_{key}.npy"


def _read_verified(path, identity):
    sidecar = path.with_suffix(".json")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise ValueError(f"Invalid EEG cache descriptor: {sidecar}") from error
    if (not isinstance(metadata, dict) or metadata.get("identity") != identity
            or metadata.get("cache_sha256") != content_sha256(path)):
        raise ValueError(f"EEG cache content/identity mismatch: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def load_offline_cache(record, directory, graph, mode, *, readonly=False):
    identity = cache_identity(record, graph, mode)
    path = offline_cache_path(record, directory, graph, mode, identity=identity)
    sidecar = path.with_suffix(".json")

    def check_source():
        if content_sha256(record.npz_path) != identity["npz_sha256"]:
            raise ValueError(f"Source EEG changed while loading cache: {record.npz_path}")

    def read_source():
        with np.load(record.npz_path, allow_pickle=False) as archive:
            values = np.asarray(archive["X"], dtype=np.float32)
        values = apply_graph_input(values, graph, mode)
        check_source()
        return values

    if path.is_file() and sidecar.is_file():
        values = _read_verified(path, identity)
    elif readonly:
        # A read-only consumer may use raw EEG in memory, but cannot bless or
        # overwrite an old or incompletely published shared cache.
        values = read_source()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.lockf(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not (path.is_file() and sidecar.is_file()):
                    values = read_source()
                    if values.ndim != 3 or not np.isfinite(values).all():
                        raise ValueError(f"Invalid EEG array for cache: {record.npz_path}")
                    with tempfile.TemporaryDirectory(prefix="." + path.stem, dir=path.parent) as temporary:
                        data_tmp = Path(temporary) / "eeg.npy"
                        meta_tmp = Path(temporary) / "identity.json"
                        np.save(data_tmp, values, allow_pickle=False)
                        metadata = {"identity": identity, "cache_sha256": content_sha256(data_tmp)}
                        meta_tmp.write_text(json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8")
                        check_source()
                        # Publish the descriptor last. Readers seeing just one
                        # file take the lock before repairing an interrupted write.
                        data_tmp.replace(path)
                        meta_tmp.replace(sidecar)
                values = _read_verified(path, identity)
            finally:
                fcntl.lockf(lock.fileno(), fcntl.LOCK_UN)
    check_source()
    return values
