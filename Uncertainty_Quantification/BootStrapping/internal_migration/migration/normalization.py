"""Normalization of existing prediction chunks into canonical arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...bootstrap.errors import HardFailure
from ...bootstrap.prediction import PredictionArrays, TargetArrays, validate_targets
from .legacy_reader import LegacyChunk, load_legacy_payload


def _tensor(payload: dict[str, object], key: str, chunk: LegacyChunk) -> np.ndarray:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
        raise HardFailure(f"prediction chunk tensor is invalid: {chunk.path}:{key}")
    return value.detach().cpu().numpy()


def concatenate_legacy_chunks(
    chunks: Sequence[LegacyChunk],
) -> tuple[TargetArrays, PredictionArrays]:
    """Concatenate one member/split while preserving source values and dtypes."""

    ids: list[str] = []
    counts: list[np.ndarray] = []
    target_energy: list[np.ndarray] = []
    target_forces: list[np.ndarray] = []
    target_stress: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    stress: list[np.ndarray] = []
    for chunk in chunks:
        payload = load_legacy_payload(chunk.path)
        chunk_ids = payload.get("structure_ids")
        if not isinstance(chunk_ids, list) or len(chunk_ids) != chunk.structure_count:
            raise HardFailure(
                f"prediction chunk structure IDs are invalid: {chunk.path}"
            )
        ids.extend(str(value) for value in chunk_ids)
        n_atoms = _tensor(payload, "n_atoms", chunk)
        offsets = _tensor(payload, "atom_offsets", chunk)
        expected_offsets = np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(n_atoms, dtype=np.int64)]
        )
        if not np.array_equal(offsets, expected_offsets):
            raise HardFailure(
                f"prediction chunk atom offsets are invalid: {chunk.path}"
            )
        counts.append(n_atoms.astype(np.int64, copy=False))
        target_energy.append(_tensor(payload, "E_ref", chunk))
        target_forces.append(_tensor(payload, "F_ref", chunk))
        target_stress.append(_tensor(payload, "S_ref", chunk))
        energy.append(_tensor(payload, "E_member", chunk))
        forces.append(_tensor(payload, "F_member", chunk))
        stress.append(_tensor(payload, "S_member", chunk))
    num_atoms = np.concatenate(counts)
    targets = TargetArrays(
        structure_ids=np.asarray(ids),
        num_atoms=num_atoms,
        atom_offsets=np.concatenate(
            [np.array([0], dtype=np.int64), np.cumsum(num_atoms, dtype=np.int64)]
        ),
        energy=np.concatenate(target_energy),
        forces=np.concatenate(target_forces),
        stress=np.concatenate(target_stress),
    )
    validate_targets(targets)
    return targets, PredictionArrays(
        energy=np.concatenate(energy),
        forces=np.concatenate(forces),
        stress=np.concatenate(stress),
    )


def assert_same_targets(expected: TargetArrays, actual: TargetArrays) -> None:
    for name in (
        "structure_ids",
        "num_atoms",
        "atom_offsets",
        "energy",
        "forces",
        "stress",
    ):
        if not np.array_equal(getattr(expected, name), getattr(actual, name)):
            raise HardFailure(f"prediction targets differ across members: {name}")
