"""Deterministic epoch control and resumable confidence-head training state."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from .artifacts import atomic_torch_save


CHECKPOINT_SCHEMA_VERSION = "upet_confidence_trainer_checkpoint_v1"
_EXTERNAL_STOP_REASON = "external_stop_after_epoch"


@dataclass(frozen=True)
class EarlyStoppingState:
    """Cross-epoch state controlled by validation total-loss EMA."""

    ema: float | None = None
    best: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stopped: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class ControlUpdate:
    """Result of committing one complete validation epoch."""

    state: EarlyStoppingState
    improved: bool
    should_save_best: bool
    should_stop: bool


@dataclass(frozen=True)
class TrainingIdentity:
    """All upstream identities that make a trainer checkpoint reusable."""

    config_id: str
    cache_id: str
    binning_id: str
    model_loss_id: str


@dataclass(frozen=True)
class RestoredTrainingState:
    """Scalar state returned after an exact checkpoint restoration."""

    epoch: int
    next_epoch: int
    global_step: int
    learning_rate: float
    control_state: EarlyStoppingState
    best_step: int | None


@dataclass
class LossAccumulator:
    """Accumulate branch losses by their true sample counts."""

    force_sum: float = 0.0
    force_count: int = 0
    energy_sum: float = 0.0
    energy_count: int = 0

    def update(
        self,
        *,
        force_loss_sum: float,
        force_count: int,
        energy_loss_sum: float,
        energy_count: int,
    ) -> None:
        _require_finite("force_loss_sum", force_loss_sum)
        _require_finite("energy_loss_sum", energy_loss_sum)
        if force_count < 0:
            raise ValueError("force_count must be non-negative")
        if energy_count < 0:
            raise ValueError("energy_count must be non-negative")
        self.force_sum += float(force_loss_sum)
        self.force_count += int(force_count)
        self.energy_sum += float(energy_loss_sum)
        self.energy_count += int(energy_count)

    def result(
        self,
        force_coefficient: float,
        energy_coefficient: float,
    ) -> tuple[float, float, float]:
        if self.force_count <= 0:
            raise ValueError("force loss has no samples")
        if self.energy_count <= 0:
            raise ValueError("energy loss has no samples")
        _require_finite("force_coefficient", force_coefficient)
        _require_finite("energy_coefficient", energy_coefficient)
        force = self.force_sum / self.force_count
        energy = self.energy_sum / self.energy_count
        total = force_coefficient * force + energy_coefficient * energy
        _require_finite("total epoch loss", total)
        return force, energy, total


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


def update_control_state(
    state: EarlyStoppingState,
    raw_total_loss: float,
    epoch: int,
    beta: float,
    min_delta: float,
    patience: int,
    min_epochs: int,
) -> ControlUpdate:
    """Advance EMA and early stopping exactly once for a validation epoch."""
    _require_finite("raw_total_loss", raw_total_loss)
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if min_delta < 0.0 or not math.isfinite(min_delta):
        raise ValueError("min_delta must be finite and non-negative")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if min_epochs <= 0:
        raise ValueError("min_epochs must be positive")
    if state.stopped:
        raise ValueError("cannot advance an already stopped control state")

    raw = float(raw_total_loss)
    ema = raw if state.ema is None else beta * state.ema + (1.0 - beta) * raw
    _require_finite("validation total-loss EMA", ema)
    improved = state.best is None or ema < state.best - min_delta

    if improved:
        new_state = EarlyStoppingState(
            ema=ema,
            best=ema,
            best_epoch=epoch,
            bad_epochs=0,
        )
    else:
        bad_epochs = state.bad_epochs + 1
        should_stop = bad_epochs >= patience and epoch + 1 >= min_epochs
        new_state = EarlyStoppingState(
            ema=ema,
            best=state.best,
            best_epoch=state.best_epoch,
            bad_epochs=bad_epochs,
            stopped=should_stop,
            stop_reason="early_stopping" if should_stop else None,
        )

    return ControlUpdate(
        state=new_state,
        improved=improved,
        should_save_best=improved,
        should_stop=new_state.stopped,
    )


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Construct the single scheduler approved by the experiment design."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-6,
    )


def advance_validation_epoch(
    *,
    state: EarlyStoppingState,
    raw_total_loss: float,
    epoch: int,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    beta: float,
    min_delta: float,
    patience: int,
    min_epochs: int,
) -> ControlUpdate:
    """Update control state and step its scheduler with the same single EMA."""
    update = update_control_state(
        state=state,
        raw_total_loss=raw_total_loss,
        epoch=epoch,
        beta=beta,
        min_delta=min_delta,
        patience=patience,
        min_epochs=min_epochs,
    )
    assert update.state.ema is not None
    scheduler.step(update.state.ema)
    return update


def capture_training_snapshot(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    global_step: int,
    control_state: EarlyStoppingState,
    best_step: int | None,
    identity: TrainingIdentity,
    sampler_generator: torch.Generator,
) -> dict[str, Any]:
    """Capture all state needed to continue at the next epoch exactly."""
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if best_step is not None and best_step < 0:
        raise ValueError("best_step must be non-negative")
    if not optimizer.param_groups:
        raise ValueError("optimizer must contain at least one parameter group")
    learning_rate = float(optimizer.param_groups[0]["lr"])
    _require_finite("learning_rate", learning_rate)

    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "epoch": epoch,
        "global_step": global_step,
        "learning_rate": learning_rate,
        "ema": control_state.ema,
        "best": control_state.best,
        "best_epoch": control_state.best_epoch,
        "best_step": best_step,
        "bad_epochs": control_state.bad_epochs,
        "stopped": control_state.stopped,
        "stop_reason": control_state.stop_reason,
        "python_rng_state": copy.deepcopy(random.getstate()),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "torch_cpu_rng_state": torch.get_rng_state().clone(),
        "torch_cuda_rng_state": [
            state.clone() for state in torch.cuda.get_rng_state_all()
        ]
        if torch.cuda.is_available()
        else [],
        "sampler_rng_state": sampler_generator.get_state().clone(),
        "config_id": identity.config_id,
        "cache_id": identity.cache_id,
        "binning_id": identity.binning_id,
        "model_loss_id": identity.model_loss_id,
    }


def _require_snapshot_identity(
    snapshot: Mapping[str, Any],
    expected: TrainingIdentity,
) -> None:
    for field in (
        "config_id",
        "cache_id",
        "binning_id",
        "model_loss_id",
    ):
        actual_value = snapshot.get(field)
        expected_value = getattr(expected, field)
        if actual_value != expected_value:
            raise ValueError(
                f"{field} mismatch: {actual_value!r} != {expected_value!r}"
            )


def _require_snapshot_schema(snapshot: Mapping[str, Any]) -> None:
    actual = snapshot.get("schema_version")
    if actual != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            "checkpoint schema_version mismatch: "
            f"{actual!r} != {CHECKPOINT_SCHEMA_VERSION!r}"
        )


def _as_non_negative_int(snapshot: Mapping[str, Any], field: str) -> int:
    value = snapshot.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"checkpoint {field} must be a non-negative integer")
    return value


def _control_from_snapshot(snapshot: Mapping[str, Any]) -> EarlyStoppingState:
    state = EarlyStoppingState(
        ema=snapshot.get("ema"),
        best=snapshot.get("best"),
        best_epoch=snapshot.get("best_epoch"),
        bad_epochs=_as_non_negative_int(snapshot, "bad_epochs"),
        stopped=snapshot.get("stopped", False),
        stop_reason=snapshot.get("stop_reason"),
    )
    if state.stop_reason == _EXTERNAL_STOP_REASON:
        return replace(state, stopped=False, stop_reason=None)
    return state


def restore_training_snapshot(
    *,
    snapshot: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    expected_identity: TrainingIdentity,
    sampler_generator: torch.Generator,
) -> RestoredTrainingState:
    """Validate and restore a committed snapshot without repeating its epoch."""
    _require_snapshot_schema(snapshot)
    _require_snapshot_identity(snapshot, expected_identity)
    epoch = _as_non_negative_int(snapshot, "epoch")
    global_step = _as_non_negative_int(snapshot, "global_step")
    best_step_value = snapshot.get("best_step")
    if best_step_value is not None and (
        isinstance(best_step_value, bool)
        or not isinstance(best_step_value, int)
        or best_step_value < 0
    ):
        raise ValueError("checkpoint best_step must be null or non-negative")
    learning_rate = float(snapshot["learning_rate"])
    _require_finite("checkpoint learning_rate", learning_rate)
    control_state = _control_from_snapshot(snapshot)

    required_states = {
        "model": snapshot.get("model"),
        "optimizer": snapshot.get("optimizer"),
        "scheduler": snapshot.get("scheduler"),
    }
    for name, value in required_states.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"checkpoint {name} must contain a state mapping")
    cpu_rng = snapshot.get("torch_cpu_rng_state")
    sampler_rng = snapshot.get("sampler_rng_state")
    if not isinstance(cpu_rng, torch.Tensor):
        raise ValueError("checkpoint torch_cpu_rng_state must be a tensor")
    if not isinstance(sampler_rng, torch.Tensor):
        raise ValueError("checkpoint sampler_rng_state must be a tensor")
    cuda_rng = snapshot.get("torch_cuda_rng_state")
    if not isinstance(cuda_rng, list) or not all(
        isinstance(item, torch.Tensor) for item in cuda_rng
    ):
        raise ValueError("checkpoint torch_cuda_rng_state must be a tensor list")
    if cuda_rng and not torch.cuda.is_available():
        raise ValueError("checkpoint has CUDA RNG state but CUDA is unavailable")

    original_model = copy.deepcopy(model.state_dict())
    original_optimizer = copy.deepcopy(optimizer.state_dict())
    original_scheduler = copy.deepcopy(scheduler.state_dict())
    try:
        model.load_state_dict(required_states["model"])
        optimizer.load_state_dict(required_states["optimizer"])
        scheduler.load_state_dict(required_states["scheduler"])
    except Exception:
        model.load_state_dict(original_model)
        optimizer.load_state_dict(original_optimizer)
        scheduler.load_state_dict(original_scheduler)
        raise

    random.setstate(snapshot["python_rng_state"])
    np.random.set_state(snapshot["numpy_rng_state"])
    torch.set_rng_state(cpu_rng)
    if cuda_rng:
        torch.cuda.set_rng_state_all(cuda_rng)
    sampler_generator.set_state(sampler_rng)

    actual_lr = float(optimizer.param_groups[0]["lr"])
    if actual_lr != learning_rate:
        raise ValueError(
            "checkpoint learning_rate disagrees with optimizer state: "
            f"{learning_rate!r} != {actual_lr!r}"
        )
    return RestoredTrainingState(
        epoch=epoch,
        next_epoch=epoch + 1,
        global_step=global_step,
        learning_rate=learning_rate,
        control_state=control_state,
        best_step=best_step_value,
    )


def commit_epoch_checkpoints(
    *,
    checkpoint_dir: Path,
    snapshot: Mapping[str, Any],
    save_best: bool,
    profile: Literal["smoke", "production"],
    stop_after_epoch: int | None,
    max_epochs: int,
) -> bool:
    """Atomically commit best/last, then report a transient smoke stop."""
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if stop_after_epoch is not None:
        if profile != "smoke":
            raise ValueError("stop_after_epoch is smoke-only")
        if stop_after_epoch < 0:
            raise ValueError("stop_after_epoch must be non-negative")
    _require_snapshot_schema(snapshot)
    epoch = _as_non_negative_int(snapshot, "epoch")
    directory = Path(checkpoint_dir)

    if save_best:
        atomic_torch_save(directory / "best.pt", snapshot)

    external_stop = stop_after_epoch is not None and epoch >= stop_after_epoch
    last_snapshot = dict(snapshot)
    if external_stop and not last_snapshot.get("stop_reason"):
        last_snapshot["stopped"] = True
        last_snapshot["stop_reason"] = _EXTERNAL_STOP_REASON
    atomic_torch_save(directory / "last.pt", last_snapshot)
    return external_stop
