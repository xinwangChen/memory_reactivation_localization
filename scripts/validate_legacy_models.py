"""Check real legacy checkpoints against isolated frozen implementations on CPU."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import pb2023_eeg
from pb2023_eeg.artifacts.checkpoints import load_model
from pb2023_eeg.artifacts.provenance import file_sha256, implementation_identity


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=project.parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="External directory containing reference_validation.json and *_forward.npz.")
    args = parser.parse_args()
    evidence = args.reference_dir.expanduser().resolve()
    if not (evidence / "reference_validation.json").is_file():
        parser.error("--reference-dir must contain reference_validation.json")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    assert Path(pb2023_eeg.__file__).resolve().is_relative_to(project / "pb2023_eeg")
    references = []
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
                   "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    environment.pop("PYTHONPATH", None)
    worker = project / "tests/reference_worker.py"
    for task in ("choice", "pitch"):
        out = args.out / ("legacy_" + task)
        subprocess.run([sys.executable, str(worker), "target", str(args.workspace / "main-config-models" / task / "block"), str(out)],
                       env=environment, cwd=args.out, check=True)
        references.extend(json.loads((out / "references.json").read_text()))
    out = args.out / "baseline"
    subprocess.run([sys.executable, str(worker), "baseline", str(args.workspace / "0-baseline/transf/lag1"), str(out)],
                   env=environment, cwd=args.out, check=True)
    references.extend(json.loads((out / "references.json").read_text()))
    for row in json.loads((evidence / "reference_validation.json").read_text()):
        references.append({**row, "fixture": str(evidence / (row["key"] + "_forward.npz"))})
    results = []
    for row in references:
        before = file_sha256(row["checkpoint"])
        model = load_model(row["checkpoint"], expected_sha256=row.get("checkpoint_sha256"))
        with np.load(row["fixture"]) as reference:
            eeg, subjects = torch.from_numpy(reference["input_eeg"]), torch.from_numpy(reference["subject_ids"])
            with torch.inference_mode():
                if model.spec.family == "target":
                    outputs = model.model(eeg, row["task"], attention_indices=torch.from_numpy(reference["attention_indices"]), subject_id=subjects)
                else:
                    outputs = {"pooled_logits": model.model(eeg, subjects)}
            errors = {}
            for key, value in outputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                actual, expected = value.numpy(), reference[key]
                if actual.shape != expected.shape or actual.dtype != expected.dtype:
                    raise AssertionError(f"{row['key']} {key}: shape/dtype mismatch")
                error = float(np.abs(actual - expected).max())
                errors[key] = error
                if not np.array_equal(actual, expected):
                    raise AssertionError(f"{row['key']} {key}: max_abs_error={error}")
        if before != file_sha256(row["checkpoint"]):
            raise AssertionError("Reference checkpoint changed during validation")
        results.append({"key": row["key"], "checkpoint_sha256": before, "fixture_sha256": file_sha256(row["fixture"]),
                        "strict": True, "exact": True, "max_abs_errors": errors,
                        "embedding_capacity": model.spec.embedding_capacity})
        print(row["key"] + ": strict load; exact CPU forward", flush=True)
    report = {"status": "passed", "count": len(results), "package_origin": pb2023_eeg.__file__,
              "implementation": implementation_identity(), "models": results}
    (args.out / "checkpoint_compatibility.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
