"""Evaluate a saved model on explicitly prepared EEG tensors from an NPZ file."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import torch

from ..artifacts.checkpoints import load_model
from ..artifacts.provenance import runtime_implementation_identity
from ..contracts import PredictionSpec
from ..data.time_axis import resolve_attention_window
from ..evaluation.predict import iter_predictions
from ..utils.source_files import SourceFileBinding, stream_sha256


def _attention_indices(config, time_s, time_point_count, *, attention_enabled=True):
    train = config.get("train", {})
    has_window = attention_enabled and any(train.get(key) is not None for key in (
        "attention_time_start_s", "attention_time_end_s", "train_time_start_s", "train_time_end_s"))
    if time_s is None:
        if has_window:
            raise ValueError("Checkpoint has an attention window; input NPZ must provide its shared time_s axis [T]")
        return None
    axis = np.asarray(time_s)
    if axis.dtype.kind not in "fiu" or axis.shape != (time_point_count,):
        raise ValueError("Input time_s must be a numeric shared axis [T] matching eeg")
    axis = np.asarray(axis, dtype=np.float64)
    if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise ValueError("Input time_s must be finite and strictly increasing")
    expected = config.get("model", {}).get("resolved_time_axis_sha256")
    if expected is not None and hashlib.sha256(axis.tobytes()).hexdigest() != expected:
        raise ValueError("Input time_s differs from the checkpoint time axis")
    if not attention_enabled:
        return None
    times = torch.from_numpy(axis)
    start, end = resolve_attention_window(train, times)
    indices = torch.nonzero((times >= start) & (times <= end), as_tuple=False).flatten()
    if not indices.numel():
        raise ValueError("Checkpoint attention window contains no input time_s samples")
    return indices.tolist()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--input", required=True, type=Path, help="Preprocessed eeg [B,C,T]; shared time_s [T] is required for saved attention windows; noise_pad requires exp, subject_id, trial0 [B]; optional record_id, record_uid, lag")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=("choice", "pitch", "motor"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trusted-legacy", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.output.resolve() in {args.checkpoint.resolve(), args.input.resolve()}:
        raise ValueError("Output must be separate from input/checkpoint")
    loaded = load_model(args.checkpoint, device=args.device, trusted_legacy=args.trusted_legacy)
    task = args.task or loaded.task
    if task is None:
        raise ValueError("--task is required when the checkpoint has no task")
    runtime_implementation_identity()
    arrays = {}
    prediction_metadata = None
    input_binding = SourceFileBinding([args.input])
    with args.input.open("rb") as input_stream, np.load(input_stream, allow_pickle=False) as data:
        if "eeg" not in data.files:
            raise ValueError("Input NPZ is missing eeg")
        inputs = {key: data[key] for key in data.files
                  if key in {"eeg", "exp", "subject_id", "record_id", "record_uid", "trial0", "lag", "target"}}
        eeg = inputs["eeg"]
        if eeg.ndim != 3 or not len(eeg):
            raise ValueError("Input eeg must be a nonempty [B,C,T] array")
        indices = _attention_indices(loaded.config, data["time_s"] if "time_s" in data.files else None,
                                     eeg.shape[-1], attention_enabled=loaded.spec.family == "target")
        if stream_sha256(input_stream) != input_binding.hashes[str(args.input)]:
            raise ValueError("Prediction input version changed while loading")
        for key, value in inputs.items():
            if key != "eeg" and value.shape != (len(eeg),):
                raise ValueError(f"Input {key} must contain one value per EEG sample")
        def batches():
            for start in range(0, len(eeg), args.batch_size):
                yield {key: (value[start:start + args.batch_size] if key == "record_uid"
                             else torch.from_numpy(value[start:start + args.batch_size]))
                       for key, value in inputs.items()}
        for batch in iter_predictions(loaded.model, batches(), PredictionSpec(task, args.device, attention_indices=indices)):
            if prediction_metadata is not None and prediction_metadata != batch.metadata:
                raise ValueError("Prediction head/support metadata changed across batches")
            prediction_metadata = dict(batch.metadata)
            for key, value in {**batch.identity, **batch.outputs}.items():
                arrays.setdefault(key, []).append(value)
    implementation = runtime_implementation_identity()
    metadata = {"schema": "pb2023.predictions.v1", "checkpoint_sha256": loaded.checkpoint_sha256,
                "input_sha256": input_binding.hashes[str(args.input)], "task": task, "preprocessing": "caller_prepared",
                "implementation_sha256": implementation["sha256"],
                "implementation_identity_schema": implementation["schema"], "spec": loaded.spec.to_dict(),
                "prediction": prediction_metadata}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=".prediction-", suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        np.savez_compressed(temporary, **{k: np.concatenate(v) for k, v in arrays.items()},
                            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
        input_binding.verify()
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(json.dumps({"output": str(args.output.resolve()), "samples": sum(len(v) for v in next(iter(arrays.values())))}))
