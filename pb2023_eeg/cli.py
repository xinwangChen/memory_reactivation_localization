"""Installed command dispatcher. Workflows own orchestration, not model code."""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pb-eeg", description="PB2023 EEG unified workflows")
    parser.add_argument("command", choices=("inspect", "predict", "train", "prepare", "reexport", "report"))
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    remaining = argv[1:]
    if args.command == "inspect":
        inspect_parser = argparse.ArgumentParser(prog="pb-eeg inspect")
        inspect_parser.add_argument("checkpoint")
        inspect_parser.add_argument("--config")
        inspect_parser.add_argument("--trusted-legacy", action="store_true")
        options = inspect_parser.parse_args(remaining)
        from .artifacts.checkpoints import load_model
        loaded = load_model(options.checkpoint, config=options.config, trusted_legacy=options.trusted_legacy)
        print(json.dumps({"checkpoint": str(loaded.checkpoint), "sha256": loaded.checkpoint_sha256,
                          "task": loaded.task, "spec": loaded.spec.to_dict(),
                          "parameter_count": sum(p.numel() for p in loaded.model.parameters()),
                          **loaded.metadata}, ensure_ascii=False, indent=2))
        return
    from importlib import import_module
    module = {"predict": "workflows.evaluate", "train": "workflows.train", "prepare": "workflows.prepare",
              "reexport": "workflows.reexport", "report": "workflows.report"}[args.command]
    import_module("pb2023_eeg." + module).main(remaining)


if __name__ == "__main__":
    main()

