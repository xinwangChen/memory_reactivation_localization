"""Streaming inference with explicit head scores and unchanged native precision."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Integral
import numpy as np
import torch

from ..contracts import PredictionBatch, PredictionSpec
from ..models.adapters import forward_batch, boundary_noise_metadata
from ..data.transforms import gaussian_like_eeg, time_shuffle_eeg, time_roll_eeg


def iter_predictions(model: torch.nn.Module, batches: Iterable[Mapping],
                     spec: PredictionSpec):
    device = torch.device(spec.device)
    if spec.input_kind not in {"real", "gaussian", "time_shuffle", "time_roll"}:
        raise ValueError("Unknown prediction input_kind")
    if any(mode not in {"softmax", "sigmoid", "logits"} for mode in spec.score_modes.values()):
        raise ValueError("score_modes must explicitly name softmax, sigmoid or logits")
    if spec.amp and device.type != "cuda":
        raise ValueError("AMP prediction is supported on CUDA only")
    generator = torch.Generator(device=device).manual_seed(spec.control_seed)
    if spec.attention_indices is not None and any(isinstance(i, bool) or not isinstance(i, Integral) for i in spec.attention_indices):
        raise ValueError("attention_indices must contain integers")
    indices = None if spec.attention_indices is None else torch.tensor(spec.attention_indices, device=device, dtype=torch.long)
    if indices is not None and (indices.ndim != 1 or not indices.numel() or bool((indices < 0).any()) or not torch.equal(indices, indices.unique(sorted=True))):
        raise ValueError("attention_indices must be nonempty, sorted, unique nonnegative integers")
    flags = [(module, module.training) for module in model.modules()]
    records = getattr(getattr(batches, "dataset", None), "records", None)
    model.eval()
    try:
        for raw in batches:
            with torch.inference_mode():
                if "eeg" not in raw or not isinstance(raw["eeg"], torch.Tensor) or raw["eeg"].ndim != 3:
                    raise ValueError("Prediction batch eeg must be a tensor [B,C,T]")
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in raw.items()}
                eeg = batch["eeg"]
                if not eeg.is_floating_point() or not bool(torch.isfinite(eeg).all()):
                    raise FloatingPointError("Prediction EEG must be finite floating point values")
                if spec.input_kind == "gaussian":
                    batch["eeg"] = gaussian_like_eeg(eeg, generator=generator)
                elif spec.input_kind == "time_shuffle":
                    if any(k not in batch for k in ("record_id", "trial0", "lag")):
                        raise ValueError("Legacy shuffle requires record_id, trial0 and lag identities")
                    batch["eeg"] = time_shuffle_eeg(eeg, record_id=batch["record_id"],
                                                   trial0=batch["trial0"], lag=batch["lag"], seed=spec.control_seed)
                elif spec.input_kind == "time_roll":
                    batch["eeg"] = time_roll_eeg(eeg, shift_samples=spec.roll_shift)
                with torch.autocast(device_type=device.type, enabled=spec.amp):
                    native = forward_batch(model, batch, task=spec.task, attention_indices=indices)
                    native = dict(native)
                    for head, mode in spec.score_modes.items():
                        key = head + "_logits"
                        if key not in native:
                            raise ValueError(f"Model does not provide head {head!r}")
                        value = native[key]
                        native[head + "_scores"] = (value if mode == "logits" else
                                                       value.sigmoid() if mode == "sigmoid" else value.softmax(dim=-1))
                identity = {}
                for key in ("record_uid", "record_id", "exp", "trial0", "lag", "subject_id", "target"):
                    if key in raw:
                        value = raw[key]
                        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
                        if array.shape != (len(eeg),):
                            raise ValueError(f"Prediction identity {key} must have one value per sample")
                        if key != "record_uid" and array.dtype.kind not in "iu":
                            raise ValueError(f"Prediction identity {key} must use integer values")
                        identity[key] = array
                if "record_uid" not in identity and records is not None and "record_id" in identity:
                    identity["record_uid"] = np.asarray([records[int(i)].subject_uid for i in identity["record_id"]])
                support = native.get("attention_valid_mask")
                outputs = {k: v.detach().cpu().numpy() for k, v in native.items()
                           if isinstance(v, torch.Tensor) and k != "attention_valid_mask"}
                if any(v.ndim == 0 or v.shape[0] != len(eeg) for v in outputs.values()):
                    raise ValueError("Prediction outputs must retain the sample axis")
                if any(not np.isfinite(v).all() for v in outputs.values()):
                    raise FloatingPointError("Model produced non-finite prediction outputs")
                boundary = boundary_noise_metadata(model)
                prediction = PredictionBatch(identity, outputs, {
                    "task": spec.task, "input_kind": spec.input_kind, "amp": spec.amp,
                    "score_modes": dict(spec.score_modes),
                    "dtypes": {k: str(v.dtype) for k, v in outputs.items()},
                    "attention_indices": list(range(eeg.shape[-1])) if indices is None else indices.cpu().tolist(),
                    "attention_valid_mask": None if support is None else support.cpu().tolist(),
                    "class_indices": list(model.class_indices(spec.task)) if hasattr(model, "class_indices") else None,
                    **({"boundary_noise": boundary} if boundary is not None else {}),
                })
            # A suspended generator must not change the caller's autograd mode.
            yield prediction
    finally:
        for module, flag in flags:
            module.training = flag

