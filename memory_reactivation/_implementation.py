"""Pin package sources before importing any model or training implementation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def implementation_identity() -> dict:
    """Describe package files currently on disk (also used by audit tools)."""
    root = Path(__file__).resolve().parent
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(root.rglob("*.py"))}
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


_IMPORTED_IMPLEMENTATION = implementation_identity()


def runtime_implementation_identity() -> dict:
    """Reject source drift instead of labelling old functions with new file hashes."""
    if implementation_identity() != _IMPORTED_IMPLEMENTATION:
        raise RuntimeError(
            "Package sources changed since this process imported memory_reactivation; "
            "cannot certify the running implementation. Restart using the run's frozen code snapshot."
        )
    return {"schema": "pb2023.runtime.v1", **deepcopy(_IMPORTED_IMPLEMENTATION)}
