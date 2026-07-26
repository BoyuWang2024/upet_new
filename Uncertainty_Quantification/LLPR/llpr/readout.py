"""Stable last-layer parameter discovery for energy and force readouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .artifacts import stable_id


@dataclass(frozen=True)
class ParameterEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int
    tensor: torch.nn.Parameter

    def serializable(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "numel": self.numel,
            "offset": self.offset,
        }


@dataclass(frozen=True)
class TargetReadoutLayout:
    target: str
    source_target: str
    entries: tuple[ParameterEntry, ...]
    dimension: int

    @property
    def parameters(self) -> tuple[torch.nn.Parameter, ...]:
        return tuple(entry.tensor for entry in self.entries)


@dataclass(frozen=True)
class ReadoutLayout:
    energy: TargetReadoutLayout
    force: TargetReadoutLayout
    layout_hash: str

    @property
    def total_dimension(self) -> int:
        return self.energy.dimension + self.force.dimension


def _target_layout(
    *,
    canonical_target: str,
    source_target: str,
    weight_names: list[str],
    named_parameters: dict[str, torch.nn.Parameter],
) -> TargetReadoutLayout:
    if not weight_names:
        raise ValueError(f"{source_target} last-layer parameter list is empty")
    entries: list[ParameterEntry] = []
    seen: set[str] = set()
    offset = 0
    for weight_name in weight_names:
        if weight_name in seen:
            raise ValueError(f"duplicate last-layer parameter {weight_name!r}")
        if weight_name not in named_parameters:
            raise ValueError(f"missing last-layer weight {weight_name!r}")
        if not weight_name.endswith(".weight"):
            raise ValueError(
                f"last-layer weight must end with '.weight': {weight_name}"
            )
        bias_name = weight_name.removesuffix(".weight") + ".bias"
        if bias_name not in named_parameters:
            raise ValueError(f"required bias is missing: {bias_name}")
        for name in (weight_name, bias_name):
            if name in seen:
                raise ValueError(f"duplicate last-layer parameter {name!r}")
            parameter = named_parameters[name]
            entries.append(
                ParameterEntry(
                    name=name,
                    shape=tuple(parameter.shape),
                    numel=parameter.numel(),
                    offset=offset,
                    tensor=parameter,
                )
            )
            offset += parameter.numel()
            seen.add(name)
    return TargetReadoutLayout(
        target=canonical_target,
        source_target=source_target,
        entries=tuple(entries),
        dimension=offset,
    )


def discover_readout_layout(model: torch.nn.Module) -> ReadoutLayout:
    """Discover energy/force weights and mandatory biases in model order."""
    mapping = getattr(model, "last_layer_parameter_names", None)
    if not isinstance(mapping, dict):
        raise ValueError("model does not expose last_layer_parameter_names")
    if "energy" not in mapping:
        raise ValueError("model is missing energy last-layer parameters")
    force_source = next(
        (
            name
            for name in ("non_conservative_force", "non_conservative_forces")
            if name in mapping
        ),
        None,
    )
    if force_source is None:
        raise ValueError("model is missing non-conservative-force parameters")

    named_parameters = dict(model.named_parameters())
    energy = _target_layout(
        canonical_target="energy",
        source_target="energy",
        weight_names=list(mapping["energy"]),
        named_parameters=named_parameters,
    )
    force = _target_layout(
        canonical_target="non_conservative_force",
        source_target=force_source,
        weight_names=list(mapping[force_source]),
        named_parameters=named_parameters,
    )
    energy_names = {entry.name for entry in energy.entries}
    force_names = {entry.name for entry in force.entries}
    overlap = sorted(energy_names & force_names)
    if overlap:
        raise ValueError(f"energy and force readouts overlap: {overlap}")
    payload = {
        "energy": [entry.serializable() for entry in energy.entries],
        "force": [entry.serializable() for entry in force.entries],
    }
    return ReadoutLayout(energy=energy, force=force, layout_hash=stable_id(payload))
