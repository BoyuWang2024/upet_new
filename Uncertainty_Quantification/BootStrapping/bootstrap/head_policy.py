"""Audited PET last-layer fine-tuning policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .errors import HardFailure


EXPECTED_TRAINABLE_PARAMETER_COUNT = 13_338
_TRAINABLE_COMPONENTS = frozenset({"node_last_layers", "edge_last_layers"})


@dataclass(frozen=True)
class TrainablePolicyAudit:
    trainable_names: tuple[str, ...]
    frozen_names: tuple[str, ...]
    trainable_parameter_count: int
    frozen_parameter_count: int


def _is_last_layer(name: str) -> bool:
    return bool(_TRAINABLE_COMPONENTS.intersection(name.split(".")))


def apply_pet_last_layer_policy(model: torch.nn.Module) -> TrainablePolicyAudit:
    """Freeze all parameters except audited PET node/edge final layers."""

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_count = 0
    frozen_count = 0
    for name, parameter in model.named_parameters():
        trainable = _is_last_layer(name)
        parameter.requires_grad_(trainable)
        if trainable:
            trainable_names.append(name)
            trainable_count += parameter.numel()
        else:
            frozen_names.append(name)
            frozen_count += parameter.numel()
    if trainable_count != EXPECTED_TRAINABLE_PARAMETER_COUNT:
        raise HardFailure(
            "PET last-layer policy expected 13,338 trainable parameters, "
            f"found {trainable_count}"
        )
    if not trainable_names:
        raise HardFailure("PET last-layer policy found no trainable parameters")
    return TrainablePolicyAudit(
        trainable_names=tuple(trainable_names),
        frozen_names=tuple(frozen_names),
        trainable_parameter_count=trainable_count,
        frozen_parameter_count=frozen_count,
    )
