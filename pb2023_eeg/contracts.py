"""Small data contracts shared by model loading, inference and export."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class ModelSpec:
    family: str
    model_config: Mapping[str, Any]
    input_channels: int
    time_point_count: int | None
    embedding_capacity: int | None = None
    schema: str = "pb2023.model.v1"

    def to_dict(self) -> dict:
        from .artifacts.provenance import json_value
        return json_value(vars(self))


@dataclass
class LoadedModel:
    model: torch.nn.Module
    spec: ModelSpec
    config: dict
    checkpoint: Path
    checkpoint_sha256: str
    task: str | None
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PredictionSpec:
    task: str
    device: str = "cpu"
    attention_indices: tuple[int, ...] | None = None
    score_modes: Mapping[str, str] = field(default_factory=dict)
    amp: bool = False
    input_kind: str = "real"
    control_seed: int = 0
    roll_shift: int = 0


@dataclass(frozen=True)
class PredictionBatch:
    identity: Mapping[str, np.ndarray]
    outputs: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]


@dataclass
class StepResult:
    loss: torch.Tensor
    losses: Mapping[str, torch.Tensor]
    outputs: Mapping[str, torch.Tensor]
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

