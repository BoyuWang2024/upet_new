"""Strict recovery of the restart state and scientific loss contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from .errors import HardFailure


LOSS_TERM_NAMES = (
    "energy",
    "forces",
    "virial",
    "non_conservative_forces",
    "non_conservative_stress",
)


@dataclass(frozen=True)
class LossTerm:
    """One immutable Huber term recovered from a checkpoint."""

    name: str
    loss_type: str
    delta: float
    weight: float


@dataclass(frozen=True)
class LossContract:
    """The complete checkpoint-defined five-term loss contract."""

    terms: tuple[LossTerm, ...]
    reduction: str
    sliding_factor: None
    per_structure_targets: tuple[str, ...]
    grad_clip_norm: float


@dataclass(frozen=True)
class CheckpointBundle:
    """The restart model state paired with its immutable loss contract."""

    model_state_dict: Mapping[str, torch.Tensor]
    loss_contract: LossContract


def _mapping(value: object, label: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"invalid checkpoint loss contract: {label} is not a mapping")
    return value


def _required(mapping: Mapping[object, object], key: str, label: str) -> object:
    try:
        return mapping[key]
    except KeyError as exc:
        raise HardFailure(
            f"invalid checkpoint loss contract: missing {label}.{key}"
        ) from exc


def _number(value: object, label: str, *, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardFailure(f"invalid checkpoint loss contract: {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HardFailure(f"invalid checkpoint loss contract: {label} is not finite")
    if positive and result <= 0.0:
        raise HardFailure(f"invalid checkpoint loss contract: {label} must be positive")
    if not positive and result < 0.0:
        raise HardFailure(
            f"invalid checkpoint loss contract: {label} must be non-negative"
        )
    return result


def _exact_term_mapping(value: object, label: str) -> Mapping[object, object]:
    mapping = _mapping(value, label)
    actual = set(mapping)
    expected = set(LOSS_TERM_NAMES)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(str(key) for key in actual - expected)
        raise HardFailure(
            "invalid checkpoint loss contract: "
            f"{label} terms differ (missing={missing}, extra={extra})"
        )
    return mapping


def recover_loss_contract(raw_checkpoint: object) -> LossContract:
    """Recover the exact scientific loss settings without defaults."""

    raw = _mapping(raw_checkpoint, "checkpoint")
    train_hypers = _mapping(
        _required(raw, "train_hypers", "checkpoint"), "train_hypers"
    )
    loss = _mapping(_required(train_hypers, "loss", "train_hypers"), "loss")
    loss_type = _mapping(_required(loss, "type", "loss"), "loss.type")
    huber = _mapping(_required(loss_type, "huber", "loss.type"), "loss.type.huber")
    deltas = _exact_term_mapping(
        _required(huber, "deltas", "loss.type.huber"), "loss.type.huber.deltas"
    )
    weights = _exact_term_mapping(_required(loss, "weights", "loss"), "loss.weights")

    reduction = _required(loss, "reduction", "loss")
    if reduction not in {"mean", "sum"}:
        raise HardFailure(
            "invalid checkpoint loss contract: reduction must be 'mean' or 'sum'"
        )
    sliding_factor = _required(loss, "sliding_factor", "loss")
    if sliding_factor is not None:
        raise HardFailure(
            "invalid checkpoint loss contract: sliding_factor must remain None"
        )

    targets_value = _required(train_hypers, "per_structure_targets", "train_hypers")
    if not isinstance(targets_value, (list, tuple)) or any(
        not isinstance(target, str) for target in targets_value
    ):
        raise HardFailure(
            "invalid checkpoint loss contract: "
            "per_structure_targets must be a sequence of strings"
        )
    per_structure_targets = tuple(targets_value)
    if len(set(per_structure_targets)) != len(per_structure_targets):
        raise HardFailure(
            "invalid checkpoint loss contract: "
            "per_structure_targets contains duplicates"
        )

    grad_clip_norm = _number(
        _required(train_hypers, "grad_clip_norm", "train_hypers"),
        "train_hypers.grad_clip_norm",
        positive=True,
    )
    terms = tuple(
        LossTerm(
            name=name,
            loss_type="huber",
            delta=_number(
                deltas[name], f"loss.type.huber.deltas.{name}", positive=True
            ),
            weight=_number(weights[name], f"loss.weights.{name}", positive=False),
        )
        for name in LOSS_TERM_NAMES
    )
    return LossContract(
        terms=terms,
        reduction=reduction,
        sliding_factor=None,
        per_structure_targets=per_structure_targets,
        grad_clip_norm=grad_clip_norm,
    )


def _load_raw_checkpoint(path: Path) -> object:
    """Keep unsafe full-checkpoint deserialization isolated in this module."""

    import metatomic.torch  # noqa: F401

    return torch.load(path, map_location="cpu", weights_only=False)


def load_checkpoint_bundle(path: Path) -> CheckpointBundle:
    """Load the restart state and reject export/best-state substitution."""

    raw = _load_raw_checkpoint(path)
    if not isinstance(raw, Mapping):
        raise HardFailure("checkpoint is not a mapping")
    restart_state = raw.get("model_state_dict")
    if (
        not isinstance(restart_state, Mapping)
        or not restart_state
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in restart_state.items()
        )
    ):
        raise HardFailure("checkpoint is missing a valid restart model_state_dict")
    return CheckpointBundle(
        model_state_dict=restart_state,
        loss_contract=recover_loss_contract(raw),
    )
