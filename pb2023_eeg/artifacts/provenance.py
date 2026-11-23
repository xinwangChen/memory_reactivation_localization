"""Content and forward-semantics identities; paths alone are not identities."""
from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from collections.abc import Mapping

import numpy as np
import torch
from dataclasses import asdict, is_dataclass
from numbers import Integral
from .._implementation import implementation_identity, runtime_implementation_identity


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"Cannot serialize identity value of type {type(value).__name__}")


def tensor_identity(value: torch.Tensor) -> dict:
    raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest()}


def _selection_identity(sampler, dataset, *, batches=False) -> dict:
    """Describe known deterministic selections without advancing any RNG stream."""
    from torch.utils.data import BatchSampler, SequentialSampler
    from ..data.dataset import GlobalBatchSampler, SubjectBatchSampler

    name = f"{type(sampler).__module__}.{type(sampler).__qualname__}"
    if type(sampler) is BatchSampler:
        return {"class": name, "batch_size": sampler.batch_size, "drop_last": sampler.drop_last,
                "sampler": _selection_identity(sampler.sampler, dataset)}
    if type(sampler) is SequentialSampler:
        return {"class": name, "sample_count": len(sampler)}
    if type(sampler) in (GlobalBatchSampler, SubjectBatchSampler):
        if sampler.dataset is not dataset:
            raise ValueError("Export batch sampler must reference the exported dataset")
        # Native samplers use local seed+epoch generators and do not mutate RNG/state.
        return {"class": name, "batches_sha256": digest(list(sampler))}
    if isinstance(sampler, (list, tuple, range, np.ndarray, torch.Tensor)):
        values = sampler.tolist() if isinstance(sampler, (np.ndarray, torch.Tensor)) else list(sampler)
        groups = values if batches else [values]
        for group in groups:
            if not isinstance(group, (list, tuple, range)) or any(
                isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= i < len(dataset) for i in group
            ):
                raise ValueError("Export sampler indices must be integers within the dataset")
        return {"class": name, "indices_sha256": digest(values), "batched": batches}
    explicit = getattr(sampler, "export_identity", None)
    if explicit is not None:
        return {"class": name, "explicit": json_value(explicit)}
    raise ValueError(
        f"Unverifiable export sampler {name}; use deterministic native sampling, explicit indices, "
        "or provide sampler.export_identity describing its selection and order"
    )


def _loader_selection_identity(loader, dataset) -> dict:
    from torch.utils.data import DataLoader, default_collate, default_convert
    if not isinstance(loader, DataLoader):
        raise ValueError("Export requires a DataLoader or explicit in-memory batches")
    if loader.collate_fn not in (default_collate, default_convert) or not getattr(loader, "in_order", True):
        raise ValueError("Verified export requires default collation and ordered loading; use explicit batches otherwise")
    batch_sampler = loader.batch_sampler
    return _selection_identity(batch_sampler if batch_sampler is not None else loader.sampler,
                               dataset, batches=batch_sampler is not None)


def data_identity(loader) -> dict:
    """Bind in-memory batches or a native dataset to input contents and selection."""
    if isinstance(loader, (list, tuple)):
        return {"kind": "batches", "batches": [
            {str(k): tensor_identity(v) if isinstance(v, torch.Tensor) else json_value(v)
             for k, v in batch.items()} for batch in loader]}
    dataset = getattr(loader, "dataset", None)
    selection = _loader_selection_identity(loader, dataset)
    config = getattr(dataset, "_pb_data_config", None)
    if config is None:
        explicit = getattr(dataset, "export_identity", None)
        if explicit is None:
            raise ValueError("Export dataset needs a verified export_identity or native data configuration")
        return {"kind": "explicit_dataset", "dataset": json_value(explicit), "sampler": selection}
    from ..utils.source_files import content_sha256
    paths = {str(v) for k, v in config.items() if v is not None and k in {
        "normalization_json", "split_csv", "graph_laplacian_json"}}
    records = getattr(dataset, "records", ())
    for record in records:
        paths.update((record.npz_path, record.meta_path))
    files = {str(p): content_sha256(p) for p in sorted(paths)}
    # The arrays, statistics and metadata may predate the files now at these
    # paths. Never assign a new disk identity to an old in-memory dataset.
    dataset.source_files.verify(files)
    dataset.store.source_files.verify(files)
    return {"kind": "lag_dataset", "config": json_value(config),
            "files": files,
            "records": json_value(records),
            "examples_sha256": digest(dataset.examples),
            "crop_epoch": getattr(getattr(dataset, "store", None), "_current_epoch", None),
            "time_sha256": hashlib.sha256(np.asarray(dataset.time_s).tobytes()).hexdigest(),
            "sampler": selection}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(json_value(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def model_semantics(model: torch.nn.Module) -> dict:
    """Include parameter-free options (gate activation, dropout, support, etc.).

    Directly constructed models are supported as well as factory-built models.
    PyTorch's transient training flag is not an inference identity.
    """
    modules = {}
    for name, module in model.named_modules():
        options = {}
        for key, value in vars(module).items():
            if key == "training" or key.startswith("_"):
                continue
            try:
                options[key] = json_value(value)
            except ValueError:
                continue
        cls = type(module)
        try:
            source = hashlib.sha256(inspect.getsource(cls).encode()).hexdigest()
        except (OSError, TypeError):
            source = None
        modules[name] = {"class": f"{cls.__module__}.{cls.__qualname__}",
                         "source_sha256": source, "options": options}
    return {"modules": modules, "spec": getattr(model, "_pb_model_spec", None)}

