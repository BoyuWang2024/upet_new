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
    if not systems:
        raise ValueError("systems batch must be non-empty")
    if structure_ids.dtype != torch.int64:
        raise ValueError("structure_ids must have dtype int64")
    if atom_counts.dtype != torch.int64:
        raise ValueError("atom_counts must have dtype int64")
    if bool(torch.any(atom_counts <= 0).item()):
        raise ValueError("atom_counts must contain only positive values")

    normalized_ids = structure_ids.clone()
    normalized_counts = atom_counts.clone()
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


def _single_block(output: Any, key: str) -> Any:
    try:
        block_count = len(output)
    except (AttributeError, TypeError) as error:
        raise ValueError(f"{key}: output must expose its block count") from error
    if block_count != 1:
        raise ValueError(f"{key}: expected exactly one block, found {block_count}")
    try:
        block = output.block()
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"{key}: failed to access the single block") from error
    if not isinstance(getattr(block, "values", None), torch.Tensor):
        raise ValueError(f"{key}: block values must be a Tensor")
    samples = getattr(block, "samples", None)
    if samples is None:
        raise ValueError(f"{key}: block samples are missing")
    if not isinstance(getattr(samples, "values", None), torch.Tensor):
        raise ValueError(f"{key}: sample values must be a Tensor")
    return block


def _labels(
    labels: Any,
    key: str,
    axis: str,
) -> tuple[tuple[str, ...], torch.Tensor]:
    names = tuple(getattr(labels, "names", ()))
    values = getattr(labels, "values", None)
    if not isinstance(values, torch.Tensor):
        raise ValueError(f"{key}: malformed {axis} labels")
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"{key}: malformed {axis} labels")
    return names, values


def _components(block: Any, key: str) -> tuple[Any, ...]:
    components = getattr(block, "components", None)
    if components is None:
        raise ValueError(f"{key}: components metadata is missing")
    try:
        return tuple(components)
    except TypeError as error:
        raise ValueError(f"{key}: malformed components metadata") from error


def _property_count(block: Any, key: str) -> int:
    properties = getattr(block, "properties", None)
    if properties is None:
        raise ValueError(f"{key}: properties metadata is missing")
    _, values = _labels(properties, key, "properties")
    return values.shape[0]


def _require_no_components(block: Any, key: str) -> None:
    if _components(block, key):
        raise ValueError(f"{key}: expected no components")


def _require_one_property(block: Any, key: str) -> None:
    count = _property_count(block, key)
    if count != 1:
        raise ValueError(
            f"{key}: properties axis must contain exactly one property, found {count}"
        )


def _require_xyz_component(block: Any, key: str) -> None:
    components = _components(block, key)
    if len(components) != 1:
        raise ValueError(f"{key}: force components schema requires one xyz component")
    names, values = _labels(components[0], key, "components")
    expected = torch.tensor(
        [[0], [1], [2]],
        dtype=values.dtype,
        device=values.device,
    )
    if (
        names != ("xyz",)
        or values.shape != (3, 1)
        or not torch.equal(
            values,
            expected,
        )
    ):
        raise ValueError(f"{key}: force components must be xyz labels 0, 1, 2")


def _sample_columns(block: Any, key: str, required: tuple[str, ...]) -> list[int]:
    names, values = _labels(block.samples, key, "samples")
    missing = [name for name in required if name not in names]
    if missing:
        raise ValueError(f"{key}: samples are missing columns {missing}")
    return [names.index(name) for name in required]


def _mismatch_structure_id(
    structure_ids: torch.Tensor,
    mismatch: int,
    expected_systems: torch.Tensor,
    actual_systems: torch.Tensor,
) -> int | None:
    if mismatch < len(expected_systems):
        local_system = int(expected_systems[mismatch].item())
    elif mismatch < len(actual_systems):
        local_system = int(actual_systems[mismatch].item())
    else:
        return None
    if local_system < 0 or local_system >= len(structure_ids):
        return None
    return int(structure_ids[local_system].item())


def _raise_identity_mismatch(
    key: str,
    kind: str,
    mismatch: int,
    structure_ids: torch.Tensor,
    expected_systems: torch.Tensor,
    actual_systems: torch.Tensor,
) -> None:
    structure_id = _mismatch_structure_id(
        structure_ids,
        mismatch,
        expected_systems,
        actual_systems,
    )
    if structure_id is None:
        raise ValueError(f"{key}: {kind} mismatch at global sample row {mismatch}")
    raise ValueError(f"{key}: {kind} mismatch at structure {structure_id}")


def _validate_values_sample_rows(
    block: Any,
    key: str,
    structure_ids: torch.Tensor,
    expected_systems: torch.Tensor,
    actual_systems: torch.Tensor,
) -> None:
    values = block.values
    sample_count = len(actual_systems)
    if values.ndim == 0:
        mismatch = 0
    elif values.shape[0] != sample_count:
        mismatch = min(values.shape[0], sample_count)
    else:
        return
    _raise_identity_mismatch(
        key,
        "block values/sample row count",
        mismatch,
        structure_ids,
        expected_systems,
        actual_systems,
    )


def _first_sequence_mismatch(
    actual_columns: tuple[torch.Tensor, ...],
    expected_columns: tuple[torch.Tensor, ...],
) -> int | None:
    actual_count = len(actual_columns[0])
    expected_count = len(expected_columns[0])
    common_count = min(actual_count, expected_count)
    if common_count:
        differs = torch.zeros(
            common_count,
            dtype=torch.bool,
            device=actual_columns[0].device,
        )
        for actual, expected in zip(
            actual_columns,
            expected_columns,
            strict=True,
        ):
            differs |= actual[:common_count] != expected[:common_count]
        mismatches = torch.nonzero(differs, as_tuple=False)
        if mismatches.numel() != 0:
            return int(mismatches[0, 0].item())
    if actual_count != expected_count:
        return common_count
    return None


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
) -> torch.Tensor:
    system_column, atom_column = _sample_columns(
        block,
        key,
        ("system", "atom"),
    )
    samples = block.samples.values
    actual_systems = samples[:, system_column]
    actual_atoms = samples[:, atom_column]
    expected_systems, expected_atoms = _expected_atom_samples(
        atom_counts,
        dtype=samples.dtype,
        device=samples.device,
    )
    _validate_values_sample_rows(
        block,
        key,
        structure_ids,
        expected_systems,
        actual_systems,
    )
    mismatch = _first_sequence_mismatch(
        (actual_systems, actual_atoms),
        (expected_systems, expected_atoms),
    )
    if mismatch is not None:
        _raise_identity_mismatch(
            key,
            "atom sample identity/order/count",
            mismatch,
            structure_ids,
            expected_systems,
            actual_systems,
        )
    return actual_systems


def _normalize_energy(
    block: Any,
    key: str,
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> torch.Tensor:
    systems = _validate_atom_samples(
        block,
        key,
        structure_ids,
        atom_counts,
    )
    _require_no_components(block, key)
    _require_one_property(block, key)
    atom_count = int(atom_counts.sum().item())
    values = block.values
    if values.shape != (atom_count, 1):
        raise ValueError(f"{key}: expected shape [N, 1], got {tuple(values.shape)}")
    totals = torch.zeros(
        len(structure_ids),
        dtype=values.dtype,
        device=values.device,
    )
    return totals.index_add(
        0, systems.to(device=values.device, dtype=torch.int64), values[:, 0]
    )


def _normalize_force(
    block: Any,
    key: str,
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> torch.Tensor:
    _validate_atom_samples(
        block,
        key,
        structure_ids,
        atom_counts,
    )
    _require_xyz_component(block, key)
    _require_one_property(block, key)
    atom_count = int(atom_counts.sum().item())
    values = block.values
    if values.shape != (atom_count, 3, 1):
        raise ValueError(f"{key}: expected shape [N, 3, 1], got {tuple(values.shape)}")
    return values.squeeze(-1)


def _normalize_features(
    block: Any,
    key: str,
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> torch.Tensor:
    _validate_atom_samples(
        block,
        key,
        structure_ids,
        atom_counts,
    )
    _require_no_components(block, key)
    atom_count = int(atom_counts.sum().item())
    values = block.values
    if values.ndim != 2 or values.shape[0] != atom_count or values.shape[1] <= 0:
        raise ValueError(
            f"{key}: expected shape [N, D] with positive D, got {tuple(values.shape)}"
        )
    property_count = _property_count(block, key)
    if property_count != values.shape[1]:
        raise ValueError(
            f"{key}: properties count {property_count} "
            f"does not match feature dimension {values.shape[1]}"
        )
    return values


def _shares_storage(first: torch.Tensor, second: torch.Tensor) -> bool:
    return (
        first.device == second.device
        and first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    )


def _extract_blocks(
    outputs: Mapping[str, Any],
    config: ReadoutConfig,
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    energy_block = _single_block(
        outputs[config.energy_prediction],
        config.energy_prediction,
    )
    force_block = _single_block(
        outputs[config.force_prediction],
        config.force_prediction,
    )
    energy_features_block = _single_block(
        outputs[config.energy_features],
        config.energy_features,
    )
    force_features_block = _single_block(
        outputs[config.force_features],
        config.force_features,
    )

    energy_prediction = _normalize_energy(
        energy_block,
        config.energy_prediction,
        structure_ids,
        atom_counts,
    )
    force_prediction = _normalize_force(
        force_block,
        config.force_prediction,
        structure_ids,
        atom_counts,
    )
    energy_features = _normalize_features(
        energy_features_block,
        config.energy_features,
        structure_ids,
        atom_counts,
    )
    force_features = _normalize_features(
        force_features_block,
        config.force_features,
        structure_ids,
        atom_counts,
    )

    if _shares_storage(energy_features, force_features):
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


def _request_outputs(
    model: Any,
    systems: Sequence[Any],
    keys: tuple[str, str, str, str],
    supported_outputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    requested = {key: supported_outputs[key] for key in keys}
    try:
        raw_outputs = model(systems, requested)
    except (ValueError, RuntimeError) as joint_error:
        energy_keys = (keys[0], keys[2])
        force_keys = (keys[1], keys[3])
        energy_request = {key: supported_outputs[key] for key in energy_keys}
        force_request = {key: supported_outputs[key] for key in force_keys}
        try:
            energy_outputs = _require_output_mapping(
                model(systems, energy_request),
                energy_keys,
            )
            force_outputs = _require_output_mapping(
                model(systems, force_request),
                force_keys,
            )
        except Exception as fallback_error:
            raise ExceptionGroup(
                "joint and paired UPET readout requests failed",
                [joint_error, fallback_error],
            ) from None
        return {**energy_outputs, **force_outputs}
    return _require_output_mapping(raw_outputs, keys)


def extract_readouts(
    model: Any,
    systems: Sequence[Any],
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
    config: ReadoutConfig,
) -> ExtractedReadouts:
    """Request and validate separate energy and force prediction/features."""
    keys = (
        config.energy_prediction,
        config.force_prediction,
        config.energy_features,
        config.force_features,
    )
    duplicate_keys = sorted({key for key in keys if keys.count(key) > 1})
    if duplicate_keys:
        raise ValueError(
            f"readout output keys must be distinct; conflicting keys: {duplicate_keys}"
        )
    normalized_ids, normalized_counts, atom_offsets = _validate_batch(
        systems,
        structure_ids,
        atom_counts,
    )
    supported_outputs = model.supported_outputs()
    if not isinstance(supported_outputs, Mapping):
        raise ValueError("model supported_outputs() must return a mapping")
    missing = [key for key in keys if key not in supported_outputs]
    if missing:
        raise ValueError(f"checkpoint is missing required outputs: {missing}")
    outputs = _request_outputs(
        model,
        systems,
        keys,
        supported_outputs,
    )

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
