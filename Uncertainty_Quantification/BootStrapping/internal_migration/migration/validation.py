"""Read-only equivalence validation for normalized result trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ...bootstrap.artifacts import sha256_file
from ...bootstrap.errors import HardFailure
from ...bootstrap.prediction import load_prediction_arrays, load_target_arrays
from .legacy_reader import inspect_legacy_run, load_legacy_payload


@dataclass(frozen=True)
class MigrationValidation:
    member_count: int
    validated_checkpoints: int
    validated_chunks: int


def _numpy(payload: dict[str, object], key: str, path: Path) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise HardFailure(f"existing chunk tensor is missing: {path}:{key}")
    return value.detach().cpu().numpy()


def _equal(expected: np.ndarray, actual: np.ndarray, location: str) -> None:
    if not np.array_equal(expected, actual):
        raise HardFailure(f"normalized prediction differs: {location}")


def validate_migrated_run(
    source: str | Path, destination: str | Path, config: object
) -> MigrationValidation:
    """Compare every normalized slice against its existing source chunk."""

    audit = inspect_legacy_run(source, config)
    destination_path = Path(destination).expanduser().resolve()
    checkpoint_count = 0
    chunk_count = 0
    for zero_index, member in enumerate(audit.members):
        for kind, source_checkpoint in member.checkpoints.items():
            target = (
                destination_path
                / "members"
                / f"member_{zero_index:03d}"
                / "checkpoints"
                / f"{kind}.pt"
            )
            if sha256_file(source_checkpoint) != sha256_file(target):
                raise HardFailure(f"normalized checkpoint SHA-256 differs: {target}")
            checkpoint_count += 1

    for split in audit.splits:
        targets = load_target_arrays(
            destination_path / "predictions" / split / "targets.npz"
        )
        for zero_index, member in enumerate(audit.members):
            values = load_prediction_arrays(
                destination_path
                / "predictions"
                / split
                / "members"
                / f"member_{zero_index:03d}"
                / "raw.npz"
            )
            for chunk in member.chunks[split]:
                payload = load_legacy_payload(chunk.path)
                structure_start = chunk.structure_start
                structure_stop = structure_start + chunk.structure_count
                atom_start = int(targets.atom_offsets[structure_start])
                atom_stop = int(targets.atom_offsets[structure_stop])
                ids = payload.get("structure_ids")
                if list(targets.structure_ids[structure_start:structure_stop]) != [
                    str(value)
                    for value in ids  # type: ignore[union-attr]
                ]:
                    raise HardFailure(f"normalized structure IDs differ: {chunk.path}")
                _equal(
                    targets.num_atoms[structure_start:structure_stop],
                    _numpy(payload, "n_atoms", chunk.path),
                    f"{chunk.path}:n_atoms",
                )
                comparisons = (
                    (
                        targets.energy[structure_start:structure_stop],
                        "E_ref",
                    ),
                    (targets.forces[atom_start:atom_stop], "F_ref"),
                    (
                        targets.stress[structure_start:structure_stop],
                        "S_ref",
                    ),
                    (values.energy[structure_start:structure_stop], "E_member"),
                    (values.forces[atom_start:atom_stop], "F_member"),
                    (values.stress[structure_start:structure_stop], "S_member"),
                )
                for normalized, key in comparisons:
                    _equal(
                        normalized,
                        _numpy(payload, key, chunk.path),
                        f"{chunk.path}:{key}",
                    )
                chunk_count += 1
    return MigrationValidation(len(audit.members), checkpoint_count, chunk_count)
