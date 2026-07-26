"""Strict validation of resumable evaluation shards."""

from pathlib import Path
from typing import Mapping

import numpy as np

from .artifacts import sha256_file


STRUCTURE_FIELDS = {
    "structure_index",
    "num_atoms",
    "energy_pred_total",
    "energy_true_total",
    "energy_pred_per_atom",
    "energy_true_per_atom",
    "energy_residual",
    "energy_raw_var",
    "energy_calibrated_var",
    "energy_calibrated_std",
    "energy_inverse_variance",
    "energy_raw_var_total_derived",
    "energy_calibrated_var_total_derived",
    "force_raw_var_component_mean_structure",
    "force_calibrated_var_component_mean_structure",
    "force_calibrated_std_component_rms_structure",
}
ATOM_FIELDS = {
    "force_raw_var_atom_mean",
    "force_calibrated_var_atom_mean",
    "force_calibrated_std_atom_rms",
}
FORCE_FIELDS = {
    "force_structure_index",
    "force_component_index_within_structure",
    "force_atom_index",
    "force_cartesian_index",
    "force_pred",
    "force_true",
    "force_residual",
    "force_raw_var_component",
    "force_calibrated_var_component",
    "force_calibrated_std_component",
    "force_inverse_variance_component",
}


def validate_evaluation_shard(
    stage_dir: Path,
    record: Mapping[str, object],
    *,
    expected_next_index: int,
) -> int:
    """Validate a progress record and return its derived next structure index."""
    relative_value = record.get("path")
    expected_sha = record.get("sha256")
    if not isinstance(relative_value, str) or not isinstance(expected_sha, str):
        raise ValueError("evaluation shard record has invalid path or SHA")
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"evaluation shard path escapes stage directory: {relative}")
    root = stage_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"evaluation shard path escapes stage directory: {relative}"
        ) from error
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError(f"evaluation shard hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
        required = STRUCTURE_FIELDS | ATOM_FIELDS | FORCE_FIELDS | {"force_offsets"}
        missing = sorted(required - arrays.keys())
        if missing:
            raise ValueError(f"evaluation shard fields are missing: {missing}")
        structure_indices = np.asarray(arrays["structure_index"])
        num_atoms = np.asarray(arrays["num_atoms"])
        offsets = np.asarray(arrays["force_offsets"])
        structure_count = len(structure_indices)
        component_count = len(arrays["force_residual"])
        declared_structures_value = record.get("structure_count")
        declared_components_value = record.get("force_component_count")
        if (
            not isinstance(declared_structures_value, int)
            or isinstance(declared_structures_value, bool)
            or not isinstance(declared_components_value, int)
            or isinstance(declared_components_value, bool)
        ):
            raise ValueError("evaluation shard record has invalid counts")
        declared_structures = declared_structures_value
        declared_components = declared_components_value
        if declared_structures != structure_count:
            raise ValueError("evaluation shard structure count mismatch")
        if declared_components != component_count:
            raise ValueError("evaluation shard force-component count mismatch")
        expected_indices = np.arange(
            expected_next_index,
            expected_next_index + structure_count,
            dtype=structure_indices.dtype,
        )
        if not np.array_equal(structure_indices, expected_indices):
            raise ValueError("evaluation shard structure indices are not continuous")
        if any(len(arrays[name]) != structure_count for name in STRUCTURE_FIELDS):
            raise ValueError("evaluation shard structure field length mismatch")
        atom_count = int(np.sum(num_atoms))
        if any(len(arrays[name]) != atom_count for name in ATOM_FIELDS):
            raise ValueError("evaluation shard atom field length mismatch")
        if any(len(arrays[name]) != component_count for name in FORCE_FIELDS):
            raise ValueError("evaluation shard force field length mismatch")
        if (
            offsets.shape != (structure_count + 1,)
            or int(offsets[0]) != 0
            or int(offsets[-1]) != component_count
            or np.any(np.diff(offsets) != 3 * num_atoms)
        ):
            raise ValueError("evaluation shard force offsets are inconsistent")
        expected_force_structures = np.repeat(structure_indices, np.diff(offsets))
        if not np.array_equal(
            arrays["force_structure_index"], expected_force_structures
        ):
            raise ValueError("evaluation shard force structure indices mismatch")
    return expected_next_index + structure_count
