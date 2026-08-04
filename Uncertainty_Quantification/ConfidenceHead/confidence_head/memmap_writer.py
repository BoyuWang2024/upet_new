"""Streaming writer for preallocated confidence-head cache arrays."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


_FIELDS = (
    "structure_ids",
    "atomic_numbers",
    "force_prediction",
    "force_reference",
    "force_features",
    "energy_features",
    "energy_prediction",
    "energy_reference",
)


def _declared_split(
    identity_payload: Mapping[str, Any], split: str
) -> Mapping[str, Any]:
    splits = identity_payload.get("splits")
    declared = splits.get(split) if isinstance(splits, Mapping) else None
    if not isinstance(declared, Mapping):
        raise ValueError(f"split {split}: identity metadata must be a mapping")
    return declared


def _positive_count(declared: Mapping[str, Any], split: str, field: str) -> int:
    value = declared.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"split {split}: identity {field} must be positive")
    return value


def _descriptor(
    root: Path,
    path: Path,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    sha256: Callable[[Path], str],
) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "dtype": dtype.name,
        "shape": list(shape),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_preallocated_split(
    cache_root: Path,
    split: str,
    structures: Iterable[Any],
    identity_payload: Mapping[str, Any],
    *,
    normalize: Callable[..., Any],
    sha256: Callable[[Path], str],
) -> dict[str, Any]:
    """Normalize and stream one split into fixed-size memory maps."""
    declared = _declared_split(identity_payload, split)
    structure_count = _positive_count(declared, split, "structure_count")
    atom_count = _positive_count(declared, split, "atom_count")
    if declared.get("force_component_count") != 3 * atom_count:
        raise ValueError(f"split {split}: identity force_component_count mismatch")

    iterator = iter(structures)
    try:
        first = normalize(next(iterator), split=split)
    except StopIteration as error:
        raise ValueError(
            f"split {split}: must contain at least one structure"
        ) from error
    force_dim = int(first.force_features.shape[1])
    energy_dim = int(first.energy_features.shape[1])
    feature_identity = identity_payload.get("features")
    if isinstance(feature_identity, Mapping) and (
        feature_identity.get("force_dim") != force_dim
        or feature_identity.get("energy_dim") != energy_dim
    ):
        raise ValueError(f"split {split}: feature dimensions disagree with identity")

    split_root = cache_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "structure_offsets": split_root / "structure_offsets.npy",
        **{field: split_root / f"{field}.bin" for field in _FIELDS},
    }
    dtypes = {
        "structure_offsets": np.dtype("int64"),
        "structure_ids": np.dtype("int64"),
        "atomic_numbers": np.dtype("int64"),
        **{field: np.dtype("float32") for field in _FIELDS[2:]},
    }
    shapes = {
        "structure_offsets": (structure_count + 1,),
        "structure_ids": (structure_count,),
        "atomic_numbers": (atom_count,),
        "force_prediction": (atom_count, 3),
        "force_reference": (atom_count, 3),
        "force_features": (atom_count, force_dim),
        "energy_features": (atom_count, energy_dim),
        "energy_prediction": (structure_count,),
        "energy_reference": (structure_count,),
    }
    arrays: dict[str, np.memmap] = {
        "structure_offsets": np.lib.format.open_memmap(
            paths["structure_offsets"],
            mode="w+",
            dtype=dtypes["structure_offsets"],
            shape=shapes["structure_offsets"],
        )
    }
    arrays.update(
        {
            field: np.memmap(
                paths[field], mode="w+", dtype=dtypes[field], shape=shapes[field]
            )
            for field in _FIELDS
        }
    )
    arrays["structure_offsets"][0] = 0
    seen: set[int] = set()
    atom_cursor = 0
    written = 0

    def write_one(raw: Any) -> None:
        nonlocal atom_cursor, written
        structure = first if raw is first else normalize(raw, split=split)
        if written >= structure_count:
            raise ValueError(f"split {split}: structure_count exceeds identity")
        if structure.structure_id in seen:
            raise ValueError(
                f"split {split}: duplicate structure ID {structure.structure_id}"
            )
        seen.add(structure.structure_id)
        atoms = len(structure.atomic_numbers)
        stop = atom_cursor + atoms
        if stop > atom_count:
            raise ValueError(f"split {split}: atom_count exceeds identity")
        dimensions = (
            int(structure.force_features.shape[1]),
            int(structure.energy_features.shape[1]),
        )
        if dimensions != (force_dim, energy_dim):
            raise ValueError(
                f"split {split} structure {structure.structure_id}: "
                f"feature dimensions {dimensions} do not match "
                f"{(force_dim, energy_dim)}"
            )
        arrays["structure_ids"][written] = structure.structure_id
        arrays["atomic_numbers"][atom_cursor:stop] = structure.atomic_numbers.numpy()
        for field in (
            "force_prediction",
            "force_reference",
            "force_features",
            "energy_features",
        ):
            arrays[field][atom_cursor:stop] = getattr(structure, field).numpy()
        arrays["energy_prediction"][written] = float(structure.energy_prediction)
        arrays["energy_reference"][written] = float(structure.energy_reference)
        atom_cursor = stop
        written += 1
        arrays["structure_offsets"][written] = atom_cursor

    write_one(first)
    for raw in iterator:
        write_one(raw)
    if written != structure_count:
        raise ValueError(f"split {split}: structure_count does not match identity")
    if atom_cursor != atom_count:
        raise ValueError(f"split {split}: atom_count does not match identity")
    for array in arrays.values():
        array.flush()
    arrays.clear()

    outputs = identity_payload.get("outputs", identity_payload.get("readouts"))
    if not isinstance(outputs, Mapping):
        raise ValueError("identity_payload must contain output keys")
    force_key = outputs.get("force_features")
    energy_key = outputs.get("energy_features")
    if (
        not isinstance(force_key, str)
        or not isinstance(energy_key, str)
        or force_key == energy_key
    ):
        raise ValueError("force and energy feature output keys must be distinct")
    descriptors = {
        field: _descriptor(
            cache_root, paths[field], dtypes[field], shapes[field], sha256
        )
        for field in ("structure_offsets", *_FIELDS)
    }
    return {
        "structures": structure_count,
        "atoms": atom_count,
        "force_components": 3 * atom_count,
        "structure_count": structure_count,
        "atom_count": atom_count,
        "force_component_count": 3 * atom_count,
        "force_feature_dim": force_dim,
        "energy_feature_dim": energy_dim,
        "feature_dtype": "float32",
        "force_feature_key": force_key,
        "energy_feature_key": energy_key,
        "arrays": descriptors,
    }
