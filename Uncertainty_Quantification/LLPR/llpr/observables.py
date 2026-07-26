"""Model outputs and exact last-layer parameter Jacobians."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .readout import ReadoutLayout


@dataclass(frozen=True)
class StructureJacobians:
    energy_pred_total: torch.Tensor
    energy_pred_per_atom: torch.Tensor
    force_pred: torch.Tensor
    energy_jacobian: torch.Tensor
    force_jacobian: torch.Tensor


def huber_curvature(residual: torch.Tensor, delta: float) -> torch.Tensor:
    """Return the exact second derivative of Huber loss away from its kink."""
    return (residual.detach().abs() <= delta).to(dtype=torch.float64, device="cpu")


def _flatten_parameter_grads(
    gradients: tuple[torch.Tensor | None, ...],
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    batch_size: int | None,
) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        if gradient is None:
            shape = (
                (parameter.numel(),)
                if batch_size is None
                else (batch_size, parameter.numel())
            )
            parts.append(torch.zeros(shape, dtype=torch.float64))
            continue
        detached = gradient.detach().to(dtype=torch.float64, device="cpu")
        if batch_size is None:
            parts.append(detached.reshape(-1))
        else:
            parts.append(detached.reshape(batch_size, -1))
    dimension = 0 if batch_size is None else 1
    return torch.cat(parts, dim=dimension)


def parameter_jacobian(
    values: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    backend: Literal["scalar", "batched"],
    chunk_size: int,
) -> torch.Tensor:
    """Differentiate each flattened output with a stable parameter order."""
    flat = values.reshape(-1)
    if not parameters:
        raise ValueError("parameters must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    rows: list[torch.Tensor] = []
    if backend == "scalar":
        for value in flat:
            gradients = torch.autograd.grad(
                value,
                parameters,
                retain_graph=True,
                allow_unused=False,
            )
            rows.append(
                _flatten_parameter_grads(gradients, parameters, batch_size=None)
            )
        return torch.stack(rows)
    if backend != "batched":
        raise ValueError(f"unsupported Jacobian backend: {backend}")
    output_count = flat.numel()
    for start in range(0, output_count, chunk_size):
        stop = min(start + chunk_size, output_count)
        grad_outputs = torch.eye(
            output_count,
            dtype=flat.dtype,
            device=flat.device,
        )[start:stop]
        gradients = torch.autograd.grad(
            flat,
            parameters,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=True,
            allow_unused=False,
        )
        rows.append(
            _flatten_parameter_grads(gradients, parameters, batch_size=stop - start)
        )
    return torch.cat(rows, dim=0)


def compute_structure_jacobians(
    model: torch.nn.Module,
    system,
    layout: ReadoutLayout,
    backend: Literal["scalar", "batched"],
    force_component_chunk_size: int,
) -> StructureJacobians:
    """Run the canonical outputs and compute target-specific Jacobians."""
    from metatomic.torch import ModelOutput

    force_key = layout.force.source_target

    outputs = model(
        [system],
        {
            "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
            force_key: ModelOutput(quantity="force", unit="eV/A", per_atom=True),
        },
    )
    energy_total = outputs["energy"].block().values.reshape(-1)[0]
    atom_count = int(system.positions.shape[0])
    energy_per_atom = energy_total / atom_count
    force_values = outputs[force_key].block().values.reshape(-1)
    energy_jacobian = parameter_jacobian(
        energy_per_atom.reshape(1),
        layout.energy.parameters,
        backend="scalar",
        chunk_size=1,
    )[0]
    force_jacobian = parameter_jacobian(
        force_values,
        layout.force.parameters,
        backend=backend,
        chunk_size=force_component_chunk_size,
    )
    tensors = (
        energy_total,
        energy_per_atom,
        force_values,
        energy_jacobian,
        force_jacobian,
    )
    if any(not bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("non-finite model output or Jacobian")
    return StructureJacobians(
        energy_pred_total=energy_total.detach().to(dtype=torch.float64, device="cpu"),
        energy_pred_per_atom=energy_per_atom.detach().to(
            dtype=torch.float64, device="cpu"
        ),
        force_pred=force_values.detach().to(dtype=torch.float64, device="cpu"),
        energy_jacobian=energy_jacobian,
        force_jacobian=force_jacobian,
    )
