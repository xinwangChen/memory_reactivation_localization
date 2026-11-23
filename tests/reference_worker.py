"""Run a frozen implementation in a fresh interpreter, never in the candidate process."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("target", "baseline"))
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    if args.mode == "target":
        sys.path.insert(0, str(args.root / "code_snapshot"))
        from scripts.train_target_decode import build_model
        from pb2023_eeg.training.common import load_checkpoint
        import pb2023_eeg
        assert Path(pb2023_eeg.__file__).resolve().is_relative_to(args.root / "code_snapshot")
        manifest = json.loads((args.root / "manifest.json").read_text())
        for record in manifest["models"]:
            checkpoint = Path(record["checkpoint"])
            saved = load_checkpoint(checkpoint)
            cfg = yaml.safe_load((checkpoint.parent / "config_snapshot.yaml").read_text())
            model = build_model(cfg["model"], n_subjects=saved["model_state"]["subject_embedding.weight"].shape[0],
                                time_point_count=275, input_channels=64)
            model.load_state_dict(saved["model_state"], strict=True)
            model.eval()
            eeg = torch.randn(2, 64, 275, generator=torch.Generator().manual_seed(20261114))
            subjects, indices = torch.tensor([0, 1]), torch.arange(275)
            task = cfg["task"]
            with torch.inference_mode():
                outputs = model(eeg, task, attention_indices=indices, subject_id=subjects)
            name = f"legacy_{task}_fold{record['fold']}"
            filename = args.out / (name + ".npz")
            np.savez_compressed(filename, input_eeg=eeg.numpy(), subject_ids=subjects.numpy(),
                                attention_indices=indices.numpy(),
                                **{k: v.numpy() for k, v in outputs.items() if isinstance(v, torch.Tensor)})
            results.append({"key": name, "checkpoint": str(checkpoint), "task": task,
                            "checkpoint_sha256": record["checkpoint_sha256"], "fixture": str(filename)})
    else:
        source = args.root / "load_model.py"
        spec = importlib.util.spec_from_file_location("frozen_baseline_loader", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        index = json.loads((args.root / "index.json").read_text())
        for candidate in index["experiments"]:
            for fold in range(5):
                model, metadata = module.load_model(candidate, fold, root=args.root)
                eeg = torch.randn(2, 64, 275, generator=torch.Generator().manual_seed(20261114))
                subjects = torch.tensor([0, 1])
                with torch.inference_mode():
                    logits = model(eeg, subjects)
                name = f"baseline_{candidate}_fold{fold}"
                filename = args.out / (name + ".npz")
                np.savez_compressed(filename, input_eeg=eeg.numpy(), subject_ids=subjects.numpy(), pooled_logits=logits.numpy())
                results.append({"key": name, "checkpoint": metadata["checkpoint"], "task": metadata["task"],
                                "fixture": str(filename)})
    (args.out / "references.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
