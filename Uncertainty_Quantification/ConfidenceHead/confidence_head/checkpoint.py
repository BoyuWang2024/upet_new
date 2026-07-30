"""Identity-first loading of frozen UPET checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import sha256_file


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: Any
    sha256: str


def load_upet_checkpoint(
    path: Path,
    expected_sha256: str,
    device: torch.device,
    dtype: torch.dtype,
) -> LoadedCheckpoint:
    """Verify checkpoint bytes before importing or invoking the model loader."""
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"checkpoint SHA mismatch: {actual_sha256} != {expected_sha256}"
        )

    from metatrain.utils.io import load_model

    model = load_model(str(path)).eval().to(device=device, dtype=dtype)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return LoadedCheckpoint(model=model, sha256=actual_sha256)
