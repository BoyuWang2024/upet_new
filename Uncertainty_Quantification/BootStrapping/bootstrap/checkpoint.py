"""Auditing and branch loading for publishable bootstrap checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math
import pickle
from pathlib import Path
from typing import Literal, Mapping, cast

import torch
from torch import Tensor

from .artifacts import sha256_file
from .errors import HardFailure


CHECKPOINT_SCHEMA = "upet.bootstrap.checkpoint/v1"


@dataclass(frozen=True)
class CheckpointAudit:
    path: Path
    sha256: str
    epoch: int
    validation_loss: float | None
    raw_keys: tuple[str, ...]
    ema_keys: tuple[str, ...]
    parameter_count: int
    inference_ready: bool
    resume_ready: bool


def _load_document(path: str | Path) -> tuple[Path, dict[str, object]]:
    checkpoint_path = Path(path).expanduser().resolve()
    try:
        document = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        pickle.UnpicklingError,
    ) as error:
        try:
            document = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
        except (
            OSError,
            RuntimeError,
            ValueError,
            TypeError,
            pickle.UnpicklingError,
        ) as fallback_error:
            raise HardFailure(
                f"could not load checkpoint {checkpoint_path}: {fallback_error}"
            ) from error
    if not isinstance(document, dict):
        raise HardFailure("checkpoint root must be a mapping")
    canonical = document.get("schema") == CHECKPOINT_SCHEMA
    legacy = document.get("schema_version") == 1 and "raw_state" in document
    if not canonical and not legacy:
        raise HardFailure(
            f"checkpoint must use {CHECKPOINT_SCHEMA} or the supported v1 PET schema"
        )
    return checkpoint_path, cast(dict[str, object], document)


def _branch_key(document: Mapping[str, object], mode: Literal["raw", "ema"]) -> str:
    suffix = "state_dict" if document.get("schema") == CHECKPOINT_SCHEMA else "state"
    return f"{mode}_{suffix}"


def _state_dict(document: Mapping[str, object], key: str) -> dict[str, Tensor]:
    value = document.get(key)
    if not isinstance(value, dict) or not value:
        raise HardFailure(f"checkpoint {key} must be a non-empty mapping")
    result: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not isinstance(tensor, Tensor):
            raise HardFailure(f"checkpoint {key} must map strings to tensors")
        result[name] = tensor
        if (torch.is_floating_point(tensor) or torch.is_complex(tensor)) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise HardFailure(f"checkpoint {key} contains non-finite tensor {name}")
    return result


def audit_checkpoint(
    path: str | Path, *, expected_parameter_count: int | None = None
) -> CheckpointAudit:
    """Validate checkpoint branches and report inference/resume capabilities."""

    checkpoint_path, document = _load_document(path)
    epoch = document.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise HardFailure("checkpoint epoch must be a non-negative integer")
    loss_value = document.get("validation_loss")
    if loss_value is not None and (
        isinstance(loss_value, bool)
        or not isinstance(loss_value, (int, float))
        or not math.isfinite(float(loss_value))
    ):
        raise HardFailure("checkpoint validation_loss must be finite numeric or null")
    validation_loss = None if loss_value is None else float(loss_value)

    raw = _state_dict(document, _branch_key(document, "raw"))
    ema = _state_dict(document, _branch_key(document, "ema"))
    raw_keys = tuple(sorted(raw))
    ema_keys = tuple(sorted(ema))
    if raw_keys != ema_keys:
        raise HardFailure("checkpoint raw and EMA branches must have matching keys")
    for name in raw_keys:
        if raw[name].shape != ema[name].shape or raw[name].dtype != ema[name].dtype:
            raise HardFailure(
                f"checkpoint raw and EMA tensor metadata differs for {name}"
            )
    parameter_count = sum(raw[name].numel() for name in raw_keys)
    optimizer = document.get("optimizer_state_dict")
    if (
        expected_parameter_count is not None
        and parameter_count != expected_parameter_count
    ):
        raise HardFailure(
            "checkpoint trainable parameter count differs: "
            f"expected {expected_parameter_count}, found {parameter_count}"
        )
    resume_ready = isinstance(optimizer, dict) and all(
        key in document
        for key in ("python_rng_state", "numpy_rng_state", "torch_rng_state")
    )
    return CheckpointAudit(
        path=checkpoint_path,
        sha256=sha256_file(checkpoint_path),
        epoch=epoch,
        validation_loss=validation_loss,
        raw_keys=raw_keys,
        ema_keys=ema_keys,
        parameter_count=parameter_count,
        inference_ready=True,
        resume_ready=resume_ready,
    )


def load_checkpoint_branch(
    path: str | Path, mode: Literal["raw", "ema"]
) -> dict[str, Tensor]:
    """Load one audited model branch onto CPU."""

    if mode not in {"raw", "ema"}:
        raise HardFailure("checkpoint mode must be raw or ema")
    _, document = _load_document(path)
    audit_checkpoint(path)
    state = _state_dict(document, _branch_key(document, mode))
    return {name: tensor.detach().clone() for name, tensor in state.items()}
