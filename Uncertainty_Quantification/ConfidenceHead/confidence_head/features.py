"""Extraction and validation of separate energy and force UPET readouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .config import ReadoutConfig


@dataclass(frozen=True)
class ExtractedReadouts:
    structure_ids: torch.Tensor
    atom_counts: torch.Tensor
    atom_offsets: torch.Tensor
    energy_prediction: torch.Tensor
    force_prediction: torch.Tensor
    energy_features: torch.Tensor
    force_features: torch.Tensor


def _validate_batch(
    systems: Sequence[Any],
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if structure_ids.ndim != 1:
        raise ValueError("structure_ids must be one-dimensional")
    if atom_counts.ndim != 1:
        raise ValueError("atom_counts must be one-dimensional")
    if len(structure_ids) != len(atom_counts):
        raise ValueError(
            "structure_ids length must equal atom_counts length: "
            f"{len(structure_ids)} != {len(atom_counts)}"
        )
    if len(systems) != len(structure_ids):
        raise ValueError(
            "systems length must equal structure_ids length: "
            f"{len(systems)} != {len(structure_ids)}"
        )
    if structure_ids.dtype != torch.int64:
        raise ValueError("structure_ids must have dtype int64")
    if atom_counts.dtype != torch.int64:
        raise ValueError("atom_counts must have dtype int64")
    if bool(torch.any(atom_counts <= 0).item()):
        raise ValueError("atom_counts must contain only positive values")

    normalized_ids = structure_ids.to(dtype=torch.int64)
    normalized_counts = atom_counts.to(dtype=torch.int64)
    atom_offsets = torch.zeros(
        len(normalized_counts) + 1,
        dtype=torch.int64,
        device=normalized_counts.device,
    )
    atom_offsets[1:] = torch.cumsum(normalized_counts, dim=0)
    return normalized_ids, normalized_counts, atom_offsets


def _require_output_mapping(
    outputs: Any,
    requested_keys: Sequence[str],
) -> Mapping[str, Any]:
    if not isinstance(outputs, Mapping):
        raise ValueError("UPET output must be a mapping")
    missing = [key for key in requested_keys if key not in outputs]
    if missing:
        raise ValueError(f"UPET call did not return requested outputs: {missing}")
    return outputs


def _block(output: Any, key: str) -> Any:
    try:
        block = output.block()
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{key}: output must provide block()") from error
    if not isinstance(getattr(block, "values", None), torch.Tensor):
        raise ValueError(f"{key}: block values must be a Tensor")
    samples = getattr(block, "samples", None)
    if samples is None:
        raise ValueError(f"{key}: block samples are missing")
    if not isinstance(getattr(samples, "values", None), torch.Tensor):
        raise ValueError(f"{key}: sample values must be a Tensor")
    return block


def _original_structure_id(
    structure_ids: torch.Tensor,
    expected_local_system: int,
) -> int:
    if len(structure_ids) == 0:
        raise ValueError("readout validation requires a non-empty batch")
    local_system = min(max(expected_local_system, 0), len(structure_ids) - 1)
    return int(structure_ids[local_system].item())


def _sample_columns(block: Any, key: str, required: tuple[str, ...]) -> list[int]:
    names = tuple(block.samples.names)
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{key}: samples are missing columns {missing}")
    values = block.samples.values
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"{key}: malformed sample values")
    return [names.index(name) for name in required]


def _validate_structure_samples(
    block: Any,
    key: str,
    structure_ids: torch.Tensor,
) -> None:
    (system_column,) = _sample_columns(block, key, ("system",))
    samples = block.samples.values
    expected_count = len(structure_ids)
    if samples.shape[0] != expected_count:
        mismatch = min(samples.shape[0], max(expected_count - 1, 0))
        structure_id = _original_structure_id(structure_ids, mismatch)
        raise ValueError(
            f"{key}: sample count/order mismatch at structure {structure_id}"
        )
    expected = torch.arange(
        expected_count,
        dtype=samples.dtype,
        device=samples.device,
    )
    actual = samples[:, system_column]
    mismatches = torch.nonzero(actual != expected, as_tuple=False)
    if mismatches.numel() != 0:
        mismatch = int(mismatches[0, 0].item())
        structure_id = _original_structure_id(structure_ids, mismatch)
        raise ValueError(f"{key}: sample order mismatch at structure {structure_id}")


def _expected_atom_samples(
    atom_counts: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    counts = atom_counts.to(device=device)
    systems = torch.repeat_interleave(
        torch.arange(len(counts), dtype=dtype, device=device),
        counts,
    )
    total_atoms = int(counts.sum().item())
    positions = torch.arange(total_atoms, dtype=dtype, device=device)
    offsets = torch.zeros(
        len(counts) + 1,
        dtype=torch.int64,
        device=device,
    )
    offsets[1:] = torch.cumsum(counts, dim=0)
    starts = torch.repeat_interleave(offsets[:-1], counts).to(dtype=dtype)
    return systems, positions - starts


def _validate_atom_samples(
    block: Any,
    key: str,
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> None:
    system_column, atom_column = _sample_columns(
        block,
        key,
        ("system", "atom"),
    )
    samples = block.samples.values
    expected_systems, expected_atoms = _expected_atom_samples(
        atom_counts,
        dtype=samples.dtype,
        device=samples.device,
    )
    expected_count = len(expected_systems)
    if samples.shape[0] != expected_count:
        mismatch = min(samples.shape[0], max(expected_count - 1, 0))
        if expected_count:
            local_system = int(expected_systems[mismatch].item())
        else:
            local_system = 0
        structure_id = _original_structure_id(structure_ids, local_system)
        raise ValueError(
            f"{key}: atom sample count/order mismatch at structure {structure_id}"
        )

    actual_systems = samples[:, system_column]
    actual_atoms = samples[:, atom_column]
    mismatches = torch.nonzero(
        (actual_systems != expected_systems) | (actual_atoms != expected_atoms),
        as_tuple=False,
    )
    if mismatches.numel() != 0:
        mismatch = int(mismatches[0, 0].item())
        local_system = int(expected_systems[mismatch].item())
        structure_id = _original_structure_id(structure_ids, local_system)
        raise ValueError(
            f"{key}: atom sample order mismatch at structure {structure_id}"
        )


def _extract_blocks(
    outputs: Mapping[str, Any],
    config: ReadoutConfig,
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    energy_block = _block(outputs[config.energy_prediction], config.energy_prediction)
    force_block = _block(outputs[config.force_prediction], config.force_prediction)
    energy_features_block = _block(
        outputs[config.energy_features],
        config.energy_features,
    )
    force_features_block = _block(
        outputs[config.force_features],
        config.force_features,
    )

    energy_prediction = energy_block.values
    structure_count = len(structure_ids)
    if energy_prediction.shape == (structure_count, 1):
        energy_prediction = energy_prediction[:, 0]
    elif energy_prediction.shape != (structure_count,):
        raise ValueError(
            f"{config.energy_prediction}: expected shape [S] or [S, 1], "
            f"got {tuple(energy_prediction.shape)}"
        )

    atom_count = int(atom_counts.sum().item())
    force_prediction = force_block.values
    if force_prediction.shape != (atom_count, 3):
        raise ValueError(
            f"{config.force_prediction}: expected shape [N, 3], "
            f"got {tuple(force_prediction.shape)}"
        )

    energy_features = energy_features_block.values
    if (
        energy_features.ndim != 2
        or energy_features.shape[0] != atom_count
        or energy_features.shape[1] <= 0
    ):
        raise ValueError(
            f"{config.energy_features}: expected shape [N, D] with positive D, "
            f"got {tuple(energy_features.shape)}"
        )

    force_features = force_features_block.values
    if (
        force_features.ndim != 2
        or force_features.shape[0] != atom_count
        or force_features.shape[1] <= 0
    ):
        raise ValueError(
            f"{config.force_features}: expected shape [N, D] with positive D, "
            f"got {tuple(force_features.shape)}"
        )

    _validate_structure_samples(
        energy_block,
        config.energy_prediction,
        structure_ids,
    )
    _validate_atom_samples(
        force_block,
        config.force_prediction,
        structure_ids,
        atom_counts,
    )
    _validate_atom_samples(
        energy_features_block,
        config.energy_features,
        structure_ids,
        atom_counts,
    )
    _validate_atom_samples(
        force_features_block,
        config.force_features,
        structure_ids,
        atom_counts,
    )

    if energy_features.data_ptr() == force_features.data_ptr():
        raise ValueError(
            "energy_features and force_features must not share storage "
            f"({config.energy_features!r}, {config.force_features!r})"
        )
    return (
        energy_prediction,
        force_prediction,
        energy_features,
        force_features,
    )


def extract_readouts(
    model: Any,
    systems: Sequence[Any],
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
    config: ReadoutConfig,
) -> ExtractedReadouts:
    """Request and validate separate energy and force prediction/features."""
    normalized_ids, normalized_counts, atom_offsets = _validate_batch(
        systems,
        structure_ids,
        atom_counts,
    )
    keys = (
        config.energy_prediction,
        config.force_prediction,
        config.energy_features,
        config.force_features,
    )
    capability_outputs = model.capabilities().outputs
    missing = [key for key in keys if key not in capability_outputs]
    if missing:
        raise ValueError(f"checkpoint is missing required outputs: {missing}")
    requested = {key: capability_outputs[key] for key in keys}

    try:
        raw_outputs = model(systems, requested)
    except (ValueError, RuntimeError):
        energy_keys = (config.energy_prediction, config.energy_features)
        force_keys = (config.force_prediction, config.force_features)
        energy_request = {key: capability_outputs[key] for key in energy_keys}
        force_request = {key: capability_outputs[key] for key in force_keys}
        energy_outputs = _require_output_mapping(
            model(systems, energy_request),
            energy_keys,
        )
        force_outputs = _require_output_mapping(
            model(systems, force_request),
            force_keys,
        )
        outputs: Mapping[str, Any] = {**energy_outputs, **force_outputs}
    else:
        outputs = _require_output_mapping(raw_outputs, keys)

    (
        energy_prediction,
        force_prediction,
        energy_features,
        force_features,
    ) = _extract_blocks(
        outputs,
        config,
        normalized_ids,
        normalized_counts,
    )
    return ExtractedReadouts(
        structure_ids=normalized_ids,
        atom_counts=normalized_counts,
        atom_offsets=atom_offsets,
        energy_prediction=energy_prediction,
        force_prediction=force_prediction,
        energy_features=energy_features,
        force_features=force_features,
    )
