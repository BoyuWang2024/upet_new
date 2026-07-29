"""Checkpoint identity, metadata audit, and current metatrain loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import sha256_file
from .config import FileIdentityConfig


@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    sha256: str
    model_class: str
    loss_reduction: str
    energy_loss_weight: float
    force_loss_weight: float
    energy_huber_delta: float
    force_huber_delta: float


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: torch.nn.Module
    identity: CheckpointIdentity


def _read_loss_metadata(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    train_hypers = checkpoint.get("train_hypers")
    if not isinstance(train_hypers, dict):
        raise ValueError("checkpoint train_hypers metadata is missing")
    loss = train_hypers.get("loss")
    if not isinstance(loss, dict):
        raise ValueError("checkpoint loss metadata is missing")
    return loss


def inspect_training_metadata(
    path: Path,
    model: torch.nn.Module,
    sha256: str,
) -> CheckpointIdentity:
    loss = _read_loss_metadata(path)
    weights = loss.get("weights")
    loss_type = loss.get("type")
    if not isinstance(weights, dict) or not isinstance(loss_type, dict):
        raise ValueError("checkpoint loss weights/type metadata is malformed")
    huber = loss_type.get("huber")
    if not isinstance(huber, dict) or not isinstance(huber.get("deltas"), dict):
        raise ValueError("checkpoint Huber delta metadata is missing")
    deltas = huber["deltas"]
    return CheckpointIdentity(
        path=Path(path).resolve(),
        sha256=sha256,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        loss_reduction=str(loss.get("reduction")),
        energy_loss_weight=float(weights["energy"]),
        force_loss_weight=float(weights["non_conservative_forces"]),
        energy_huber_delta=float(deltas["energy"]),
        force_huber_delta=float(deltas["non_conservative_forces"]),
    )


def validate_loss_contract(identity: CheckpointIdentity) -> None:
    expected = {
        "loss_reduction": "mean",
        "energy_loss_weight": 1.0,
        "force_loss_weight": 0.1,
        "energy_huber_delta": 0.015,
        "force_huber_delta": 0.01,
    }
    actual = {name: getattr(identity, name) for name in expected}
    mismatched = {
        name: (actual[name], value)
        for name, value in expected.items()
        if actual[name] != value
    }
    if mismatched:
        raise ValueError(f"checkpoint loss contract mismatch: {mismatched}")


def load_checkpoint(
    config: FileIdentityConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> LoadedCheckpoint:
    """Verify SHA before deserializing and loading the runtime model."""
    actual_sha = sha256_file(config.path)
    if actual_sha != config.expected_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: {actual_sha} != {config.expected_sha256}"
        )
    from metatrain.utils.io import load_model

    model = load_model(str(config.path)).eval().to(device=device, dtype=dtype)
    identity = inspect_training_metadata(config.path, model, actual_sha)
    validate_loss_contract(identity)
    return LoadedCheckpoint(model=model, identity=identity)
