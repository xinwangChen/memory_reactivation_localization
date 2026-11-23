"""Compare original and candidate CPU optimizer trajectories in separate processes."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def worker(args):
    if args.source is not None:
        sys.path.insert(0, str(args.source))
        from scripts.train_target_decode import build_model
        from pb2023_eeg.training.target import fit_target_decoder
    else:
        from pb2023_eeg.models.factory import build_model
        from pb2023_eeg.training.trainer import fit_target_decoder
    import random
    import numpy as np
    import torch
    import pb2023_eeg
    expected = args.source or Path(__file__).resolve().parents[1]
    assert Path(pb2023_eeg.__file__).resolve().is_relative_to(expected)
    torch.set_num_threads(1)
    encoder = dict(type=args.encoder, in_channels=2, latent_dim=4, dropout=.1)
    if args.encoder == "cnn":
        encoder.update(num_blocks=1, kernel_size=3, expansion_ratio=1)
    else:
        encoder.update(num_layers=1, window_size=3, num_heads=2, ffn_ratio=2,
                       attention_pattern="centered_local", position_encoding_mode="sinusoidal_qk")
    model_config = dict(encoder=encoder, attention_mode="gated_scorer", attention_dim=4,
                        attention_gate_activation="tanh_plus_one", attention_score_centering=True,
                        time_no_class_enabled=True, attention_dropout=.1, classifier_num_layers=1, classifier_dropout=.1)
    config = dict(model=model_config, train=dict(
        iterations=4, validation_interval_iterations=1, log_interval_iterations=1,
        learning_rate=1e-3, lr_schedule="constant", weight_decay=.1, amp=False, progress_bar=False,
        time_loss_type="bce_one_vs_rest", pooled_loss_type="focal_bce", pooled_focal_gamma=1.,
        time_loss_weight=.5, pooled_loss_weight=1., attention_mask_penalty_weight=.7,
        attention_topk_ratio=.8, checkpoint_metric="val_pooled_accuracy", checkpoint_mode="max",
        attention_mass_penalty_weight=.3, clean_attention_penalty_weight=.2, clean_attention_penalty_probability=.5,
        temporal_crop_noise=dict(enabled=True, geometry="random_blocks", sample_interval_s=.01,
            probability=.8, min_blocks=1, max_blocks=1, min_block_duration_s=.02,
            min_total_duration_s=.02, max_total_duration_s=.1, supervision_mode="all_positions"),
        near_full_noise=dict(enabled=True, probability=.1, coverage_ratio=.95, time_loss_scale=.25),
        augmentation_pipeline={"enabled": True, "roll_max_shift_s": .1}))
    generator = torch.Generator().manual_seed(73)
    batches = [dict(eeg=torch.randn(8, 2, 20, generator=generator), target=torch.arange(8) % 2,
                    subject_id=torch.zeros(8, dtype=torch.long), record_id=torch.zeros(8, dtype=torch.long),
                    current_stimulus=torch.ones(8, dtype=torch.long)) for _ in range(3)]
    torch.manual_seed(71)
    model = build_model(model_config, time_point_count=20)
    torch.manual_seed(89)
    np.random.seed(89)
    random.seed(89)
    args.out.mkdir(parents=True, exist_ok=True)
    fit_target_decoder(model, batches, batches, "choice", torch.arange(20), config, args.out, torch.device("cpu"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--encoder", default="cnn")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    import torch
    from pb2023_eeg.artifacts.provenance import implementation_identity
    if args.source is None:
        parser.error("--source is required and must point to the external frozen implementation")
    source = args.source.expanduser().resolve()
    if not (source / "pb2023_eeg").is_dir() or not (source / "scripts/train_target_decode.py").is_file():
        parser.error("--source must contain pb2023_eeg and scripts/train_target_decode.py")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    args.out.mkdir(parents=True, exist_ok=True)
    def equal(a, b, key):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True, msg=key)
        elif isinstance(a, dict):
            assert a.keys() == b.keys(), key
            for k in a: equal(a[k], b[k], key + "." + str(k))
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), key
            for i, (x, y) in enumerate(zip(a, b)): equal(x, y, key + "." + str(i))
        else:
            if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
                return
            assert a == b, (key, a, b)
    results = []
    for encoder in ("cnn", "window_transformer"):
        directories = []
        for kind in ("reference", "candidate"):
            directory = args.out / encoder / kind
            directories.append(directory)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(directory), "--encoder", encoder]
            if kind == "reference": command += ["--source", str(source)]
            with (args.out / f"{encoder}_{kind}.log").open("w") as stream:
                subprocess.run(command, env=env, cwd=args.out, stdout=stream, stderr=subprocess.STDOUT, check=True)
        for filename in ("last.ckpt", "best_val.ckpt"):
            original, candidate = [torch.load(d / filename, map_location="cpu", weights_only=True) for d in directories]
            for key in ("model_state", "optimizer_state", "scaler_state", "rng_states", "metrics", "iteration", "extra_state"):
                equal(original[key], candidate[key], encoder + "." + filename + "." + key)
        for name in ("metrics.csv", "train_metrics.csv"):
            assert (directories[0] / name).read_bytes() == (directories[1] / name).read_bytes(), name
        results.append({"encoder": encoder, "updates": 4, "parameters_exact": True,
                        "optimizer_exact": True, "rng_exact": True, "metrics_exact": True,
                        "best_selection_exact": True})
    (args.out / "training_parity.json").write_text(json.dumps({"status": "passed", "results": results,
        "reference_source": str(source), "implementation": implementation_identity()}, indent=2) + "\n")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
