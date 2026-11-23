"""Single-optimizer Muon/AdamW integration for the existing AMP/resume path.

Muon follows Keller Jordan's original single-device algorithm and rectangular
matrix scaling (https://github.com/KellerJordan/Muon). No distributed runtime
or newer PyTorch is required. See third_party/Muon-LICENSE for attribution.
Only hidden 2D matrices use Muon; all other parameters retain AdamW updates.
"""
from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, sqrt

import torch
from torch import nn


def _zeropower(matrix: torch.Tensor, steps: int) -> torch.Tensor:
    """Original BF16 quintic Newton--Schulz update, including transpose-back."""
    # Preserve the original arithmetic for ordinary gradients. Extremely large
    # finite gradients need rescaling before BF16 conversion/norm; otherwise the
    # squared norm can overflow, producing a zero or NaN update. This is entirely
    # on-device and also handles FP64 inputs beyond the FP32/BF16 range.
    reference = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    maximum = reference.abs().amax()
    safe_limit = 0.5 * sqrt(torch.finfo(torch.bfloat16).max / matrix.numel())
    scale = torch.where(maximum > safe_limit, maximum, torch.ones_like(maximum))
    x = (reference / scale).to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    norm = x.norm()
    # The literal scalar addition is intentional: on CUDA, converting epsilon
    # to BF16 before addition changes rounding for tiny ordinary gradients.
    ordinary_denominator = norm + 1e-7
    scaled_denominator = norm + (1e-7 / scale).to(torch.bfloat16)
    denominator = torch.where(maximum > safe_limit, scaled_denominator, ordinary_denominator)
    x = x / denominator
    for _ in range(steps):
        a = x @ x.T
        b = -4.7750 * a + 2.0315 * a @ a
        x = 3.4445 * x + b @ x
    return x.T if transposed else x


class MuonWithAdamW(torch.optim.Optimizer):
    """One optimizer/state_dict: AMP skips both algorithms together on overflow."""

    def __init__(self, groups):
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate every gradient before either branch mutates parameters/state.
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is not None and (grad.is_sparse or grad.is_complex()):
                    raise RuntimeError("MuonWithAdamW requires dense real gradients")
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                # Unused task heads stay untouched, including momentum/decay.
                if grad is None:
                    continue
                state = self.state[parameter]
                if group["algorithm"] == "muon":
                    if not state:
                        state["momentum_buffer"] = torch.zeros_like(parameter)
                    momentum = state["momentum_buffer"]
                    beta = group["momentum"]
                    momentum.lerp_(grad, 1.0 - beta)
                    update = grad.lerp(momentum, beta) if group["nesterov"] else momentum
                    update = _zeropower(update, group["ns_steps"])
                    update = update * sqrt(max(1.0, parameter.shape[0] / parameter.shape[1]))
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.add_(update.to(parameter.dtype), alpha=-group["lr"])
                else:
                    if not state:
                        state.update(step=0, exp_avg=torch.zeros_like(parameter),
                                     exp_avg_sq=torch.zeros_like(parameter))
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].lerp_(grad, 1.0 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denominator = state["exp_avg_sq"].sqrt() / sqrt(1.0 - beta2 ** state["step"])
                    denominator.add_(group["eps"])
                    parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(state["exp_avg"], denominator,
                                      value=-group["lr"] / (1.0 - beta1 ** state["step"]))
        return loss


def build_muon_optimizer(params, train_config, model: nn.Module | None):
    """Route by parameter role, preserving any existing per-group LR scales."""
    if model is None:
        raise ValueError("Muon requires model= to distinguish embeddings and output heads")
    options = train_config.get("muon", {})
    if not isinstance(options, Mapping):
        raise ValueError("train.muon must be a mapping")
    unknown = set(options) - {"learning_rate", "momentum", "nesterov", "ns_steps"}
    if unknown:
        raise ValueError(f"Unknown train.muon options: {sorted(unknown)}")
    if "learning_rate" not in options:
        raise ValueError("train.muon.learning_rate must be explicitly specified")
    base_lr = float(train_config["learning_rate"])
    muon_lr = float(options["learning_rate"])
    weight_decay = float(train_config.get("weight_decay", 0.0))
    momentum = float(options.get("momentum", 0.95))
    steps = options.get("ns_steps", 5)
    nesterov = options.get("nesterov", True)
    if not isfinite(base_lr) or base_lr <= 0 or not isfinite(muon_lr) or muon_lr <= 0:
        raise ValueError("AdamW and Muon learning rates must be finite and positive")
    if not isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("train.muon.momentum must be in [0, 1)")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("train.muon.ns_steps must be a positive integer")
    if not isinstance(nesterov, bool):
        raise ValueError("train.muon.nesterov must be a boolean")

    named = {id(p): name for name, p in model.named_parameters()}
    excluded = set()
    output_roots = {"class_heads", "pooled_heads", "fusion_heads", "heads", "reconstruction_head"}
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) or name.split(".")[0] in output_roots:
            excluded.update(id(p) for p in module.parameters())
    supplied = list(params)
    groups = supplied if supplied and isinstance(supplied[0], dict) else [{"params": supplied}]
    routed = []
    seen = set()
    for source in groups:
        source = dict(source)
        source_params = list(source.pop("params"))
        source_scale = float(source.get("lr_scale", 1.0))
        if not isfinite(source_scale) or source_scale <= 0:
            raise ValueError("Muon parameter group lr_scale must be finite and positive")
        source.pop("lr", None)
        source.pop("param_names", None)
        group_decay = float(source.get("weight_decay", weight_decay))
        if not isfinite(group_decay) or group_decay < 0:
            raise ValueError("Muon parameter group weight_decay must be finite and nonnegative")
        partitions = {"muon": [], "adamw": []}
        for parameter in source_params:
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity not in named or identity in seen:
                raise ValueError("Muon parameters must belong to model and occur exactly once")
            seen.add(identity)
            if parameter.is_complex():
                raise ValueError("Muon requires real model parameters")
            eligible = parameter.ndim == 2 and min(parameter.shape) > 1 and identity not in excluded
            partitions["muon" if eligible else "adamw"].append(parameter)
        for algorithm, selected in partitions.items():
            if not selected:
                continue
            scale = source_scale * (muon_lr / base_lr if algorithm == "muon" else 1.0)
            if not isfinite(scale) or not isfinite(base_lr * scale):
                raise ValueError("Muon parameter group learning rate scale overflow")
            routed.append({**source, "params": selected, "algorithm": algorithm,
                "param_names": [named[id(p)] for p in selected],
                "lr_scale": scale, "lr": base_lr * scale,
                "weight_decay": group_decay,
                "momentum": momentum, "nesterov": nesterov, "ns_steps": steps,
                "betas": (0.9, 0.999), "eps": 1e-8})
    if not any(group["algorithm"] == "muon" for group in routed):
        raise ValueError("Muon found no eligible hidden 2D matrices")
    return MuonWithAdamW(routed)
