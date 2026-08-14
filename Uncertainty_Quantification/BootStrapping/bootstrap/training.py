"""Deterministic training state machine shared by bootstrap runtimes."""

from __future__ import annotations

import copy
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Protocol

import numpy as np
import torch
from torch import Tensor

from .errors import HardFailure


class TrainingRuntime(Protocol):
    model: torch.nn.Module

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        """Train exactly one epoch and return its mean loss."""

    def validation_loss(self, epoch: int) -> float:
        """Return validation loss for the model parameters currently applied."""


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    training_loss: float
    raw_validation_loss: float
    ema_validation_loss: float


@dataclass(frozen=True)
class TrainingResume:
    completed_epoch: int
    raw_state: dict[str, Tensor]
    ema_state: dict[str, Tensor]
    optimizer_state: dict[str, Any]
    best_epoch: int
    best_loss: float
    best_raw: dict[str, Tensor]
    best_ema: dict[str, Tensor]
    history: tuple[EpochRecord, ...]
    python_rng_state: object
    numpy_rng_state: tuple[Any, ...]
    torch_rng_state: Tensor


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_raw: dict[str, Tensor]
    best_ema: dict[str, Tensor]
    final_raw: dict[str, Tensor]
    final_ema: dict[str, Tensor]
    history: tuple[EpochRecord, ...]
    resume: TrainingResume


def _clone_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


class _EMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0 < decay < 1:
            raise HardFailure("ema_decay must be between 0 and 1")
        self.decay = decay
        self.shadow = _clone_state(model.state_dict())

    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if set(current) != set(self.shadow):
            raise HardFailure("model state keys changed during EMA update")
        with torch.no_grad():
            for name, value in current.items():
                if torch.is_floating_point(value):
                    self.shadow[name].mul_(self.decay).add_(
                        value.detach().cpu(), alpha=1 - self.decay
                    )
                else:
                    self.shadow[name].copy_(value.detach().cpu())

    @contextmanager
    def applied(self, model: torch.nn.Module) -> Iterator[None]:
        raw = _clone_state(model.state_dict())
        try:
            model.load_state_dict(self.shadow, strict=True)
            yield
        finally:
            model.load_state_dict(raw, strict=True)


def _finite(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardFailure(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HardFailure(f"{location} must be finite")
    return result


def fit_runtime(
    runtime: TrainingRuntime,
    optimizer: torch.optim.Optimizer,
    *,
    max_epochs: int,
    ema_decay: float,
    resume: TrainingResume | None = None,
) -> TrainingResult:
    """Fit a runtime with paired best raw/EMA state and exact resume semantics."""

    if isinstance(max_epochs, bool) or max_epochs < 1:
        raise HardFailure("max_epochs must be at least 1")
    ema = _EMA(runtime.model, ema_decay)
    history: list[EpochRecord] = []
    start_epoch = 0
    best_epoch = 0
    best_loss = math.inf
    best_raw: dict[str, Tensor] = {}
    best_ema: dict[str, Tensor] = {}
    if resume is not None:
        if resume.completed_epoch > max_epochs:
            raise HardFailure("resume epoch exceeds max_epochs")
        runtime.model.load_state_dict(resume.raw_state, strict=True)
        ema.shadow = _clone_state(resume.ema_state)
        optimizer.load_state_dict(copy.deepcopy(resume.optimizer_state))
        history = list(resume.history)
        start_epoch = resume.completed_epoch
        best_epoch = resume.best_epoch
        best_loss = resume.best_loss
        best_raw = _clone_state(resume.best_raw)
        best_ema = _clone_state(resume.best_ema)
        random.setstate(resume.python_rng_state)  # type: ignore[arg-type]
        np.random.set_state(resume.numpy_rng_state)
        torch.set_rng_state(resume.torch_rng_state.detach().cpu())

    for epoch in range(start_epoch + 1, max_epochs + 1):
        training_loss = _finite(runtime.train_epoch(optimizer, epoch), "training loss")
        ema.update(runtime.model)
        raw_loss = _finite(runtime.validation_loss(epoch), "raw validation loss")
        with ema.applied(runtime.model):
            ema_loss = _finite(runtime.validation_loss(epoch), "EMA validation loss")
        history.append(EpochRecord(epoch, training_loss, raw_loss, ema_loss))
        if raw_loss < best_loss:
            best_loss = raw_loss
            best_epoch = epoch
            best_raw = _clone_state(runtime.model.state_dict())
            best_ema = _clone_state(ema.shadow)

    if best_epoch == 0:
        raise HardFailure("training produced no best checkpoint")
    final_raw = _clone_state(runtime.model.state_dict())
    final_ema = _clone_state(ema.shadow)
    resume_state = TrainingResume(
        completed_epoch=max_epochs,
        raw_state=_clone_state(final_raw),
        ema_state=_clone_state(final_ema),
        optimizer_state=copy.deepcopy(optimizer.state_dict()),
        best_epoch=best_epoch,
        best_loss=best_loss,
        best_raw=_clone_state(best_raw),
        best_ema=_clone_state(best_ema),
        history=tuple(history),
        python_rng_state=random.getstate(),
        numpy_rng_state=np.random.get_state(),
        torch_rng_state=torch.get_rng_state().clone(),
    )
    return TrainingResult(
        best_epoch=best_epoch,
        best_raw=best_raw,
        best_ema=best_ema,
        final_raw=final_raw,
        final_ema=final_ema,
        history=tuple(history),
        resume=resume_state,
    )
