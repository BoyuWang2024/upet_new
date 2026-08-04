"""Immutable continuous memmap cache for confidence-head training."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import atomic_write_json, sha256_file
from .identity import cache_id
from .memmap_writer import write_preallocated_split


SCHEMA_VERSION = "upet_confidence_raw_cache_v2"
_ATOM_FIELDS = (
    "atomic_numbers",
    "force_prediction",
    "force_reference",
    "force_features",
    "energy_features",
)
_ENERGY_FIELDS = ("energy_prediction", "energy_reference")
_ARRAY_FIELDS = ("structure_offsets", "structure_ids", *_ATOM_FIELDS, *_ENERGY_FIELDS)
_DTYPES = {
    "structure_offsets": np.dtype("int64"),
    "structure_ids": np.dtype("int64"),
    "atomic_numbers": np.dtype("int64"),
    "force_prediction": np.dtype("float32"),
    "force_reference": np.dtype("float32"),
    "force_features": np.dtype("float32"),
    "energy_features": np.dtype("float32"),
    "energy_prediction": np.dtype("float32"),
    "energy_reference": np.dtype("float32"),
}


@dataclass(frozen=True)
class RawStructure:
    """Unmodified predictions, references, and readouts for one structure."""

    structure_id: int
    atomic_numbers: torch.Tensor
    force_prediction: torch.Tensor
    force_reference: torch.Tensor
    energy_prediction: torch.Tensor | float
    energy_reference: torch.Tensor | float
    force_features: torch.Tensor
    energy_features: torch.Tensor


@dataclass(frozen=True)
class RawCacheStaging:
    output_root: Path
    root: Path
    manifest_path: Path


def prepare_raw_cache(output_root: Path) -> RawCacheStaging:
    output_root = Path(output_root)
    root = output_root / f".staging-{uuid.uuid4().hex}"
    manifest_path = root / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "status": "incomplete",
            "identity": None,
            "cache_id": None,
            "identity_payload": None,
            "splits": {},
        },
    )
    return RawCacheStaging(output_root, root, manifest_path)


def _validate_split_name(split: Any) -> str:
    if (
        not isinstance(split, str)
        or not split
        or split in {".", ".."}
        or Path(split).name != split
    ):
        raise ValueError(f"unsafe cache split name: {split!r}")
    return split


def _discard_staging(staging: RawCacheStaging, output_root: Path) -> None:
    root = staging.root.resolve()
    parent = Path(output_root).resolve()
    if root.parent != parent or not root.name.startswith(".staging-"):
        raise ValueError(f"refusing to remove unsafe staging path: {root}")
    if root.exists():
        shutil.rmtree(root)


def _context(split: str, structure_id: int, field: str) -> str:
    return f"split {split} structure {structure_id}: field {field}"


def _float_tensor(
    value: Any,
    *,
    split: str,
    structure_id: int,
    field: str,
    shape: tuple[int, ...] | None = None,
    rows: int | None = None,
) -> torch.Tensor:
    context = _context(split, structure_id, field)
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{context} must be a floating Tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{context} must have shape {shape}, got {tuple(value.shape)}")
    if rows is not None and (
        value.ndim != 2 or value.shape[0] != rows or value.shape[1] <= 0
    ):
        raise ValueError(
            f"{context} must have shape ({rows}, D) with positive D, "
            f"got {tuple(value.shape)}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{context} must contain only finite values")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _float_scalar(
    value: Any,
    *,
    split: str,
    structure_id: int,
    field: str,
) -> torch.Tensor:
    context = _context(split, structure_id, field)
    if isinstance(value, torch.Tensor):
        if value.ndim != 0 or not value.is_floating_point():
            raise ValueError(f"{context} must be a floating scalar")
        result = value.detach().to(device="cpu", dtype=torch.float32)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = torch.tensor(float(value), dtype=torch.float32)
    else:
        raise ValueError(f"{context} must be a scalar tensor or float")
    if not bool(torch.isfinite(result).item()):
        raise ValueError(f"{context} must contain only finite values")
    return result.contiguous().clone()


def _normalize_structure(raw: RawStructure, *, split: str) -> RawStructure:
    if not isinstance(raw, RawStructure):
        raise ValueError(f"split {split}: expected RawStructure")
    if isinstance(raw.structure_id, bool) or not isinstance(raw.structure_id, int):
        raise ValueError(f"split {split}: structure_id must be an integer")
    structure_id = raw.structure_id
    atomic_numbers = raw.atomic_numbers
    context = _context(split, structure_id, "atomic_numbers")
    if (
        not isinstance(atomic_numbers, torch.Tensor)
        or atomic_numbers.dtype != torch.int64
    ):
        raise ValueError(f"{context} must have dtype int64")
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() <= 0:
        raise ValueError(f"{context} must have shape (N,) with positive N")
    atomic_numbers = atomic_numbers.detach().to(device="cpu").contiguous().clone()
    atoms = len(atomic_numbers)
    return RawStructure(
        structure_id=structure_id,
        atomic_numbers=atomic_numbers,
        force_prediction=_float_tensor(
            raw.force_prediction,
            split=split,
            structure_id=structure_id,
            field="force_prediction",
            shape=(atoms, 3),
        ),
        force_reference=_float_tensor(
            raw.force_reference,
            split=split,
            structure_id=structure_id,
            field="force_reference",
            shape=(atoms, 3),
        ),
        energy_prediction=_float_scalar(
            raw.energy_prediction,
            split=split,
            structure_id=structure_id,
            field="energy_prediction",
        ),
        energy_reference=_float_scalar(
            raw.energy_reference,
            split=split,
            structure_id=structure_id,
            field="energy_reference",
        ),
        force_features=_float_tensor(
            raw.force_features,
            split=split,
            structure_id=structure_id,
            field="force_features",
            rows=atoms,
        ),
        energy_features=_float_tensor(
            raw.energy_features,
            split=split,
            structure_id=structure_id,
            field="energy_features",
            rows=atoms,
        ),
    )


def _confined_array_path(cache_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("array path must be a relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"array path must be relative: {relative}")
    root = cache_root.resolve()
    path = (root / candidate).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"array path escapes cache root: {relative}")
    return path


def _write_binary(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mapped = np.memmap(path, mode="w+", dtype=array.dtype, shape=array.shape)
    mapped[...] = array
    mapped.flush()
    del mapped


def _descriptor(cache_root: Path, path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": path.relative_to(cache_root).as_posix(),
        "dtype": array.dtype.name,
        "shape": list(array.shape),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_split_materialized(
    cache_root: Path,
    split: str,
    structures: Iterable[RawStructure],
    identity_payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized: list[RawStructure] = []
    seen: set[int] = set()
    dimensions: tuple[int, int] | None = None
    for raw in structures:
        structure = _normalize_structure(raw, split=split)
        if structure.structure_id in seen:
            raise ValueError(
                f"split {split}: duplicate structure ID {structure.structure_id}"
            )
        seen.add(structure.structure_id)
        current = (
            int(structure.force_features.shape[1]),
            int(structure.energy_features.shape[1]),
        )
        if dimensions is None:
            dimensions = current
        elif current != dimensions:
            raise ValueError(
                f"split {split} structure {structure.structure_id}: "
                f"feature dimensions {current} do not match {dimensions}"
            )
        normalized.append(structure)
    if not normalized or dimensions is None:
        raise ValueError(f"split {split}: must contain at least one structure")

    counts = np.asarray(
        [len(structure.atomic_numbers) for structure in normalized], dtype=np.int64
    )
    offsets = np.zeros(len(normalized) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    arrays: dict[str, np.ndarray] = {
        "structure_offsets": offsets,
        "structure_ids": np.asarray(
            [structure.structure_id for structure in normalized], dtype=np.int64
        ),
        "atomic_numbers": torch.cat(
            [structure.atomic_numbers for structure in normalized]
        ).numpy(),
        "force_prediction": torch.cat(
            [structure.force_prediction for structure in normalized]
        ).numpy(),
        "force_reference": torch.cat(
            [structure.force_reference for structure in normalized]
        ).numpy(),
        "force_features": torch.cat(
            [structure.force_features for structure in normalized]
        ).numpy(),
        "energy_features": torch.cat(
            [structure.energy_features for structure in normalized]
        ).numpy(),
        "energy_prediction": torch.stack(
            [torch.as_tensor(structure.energy_prediction) for structure in normalized]
        ).numpy(),
        "energy_reference": torch.stack(
            [torch.as_tensor(structure.energy_reference) for structure in normalized]
        ).numpy(),
    }
    split_root = cache_root / split
    split_root.mkdir(parents=True, exist_ok=True)
    descriptors: dict[str, dict[str, Any]] = {}
    for field, array in arrays.items():
        expected_dtype = _DTYPES[field]
        array = np.asarray(array, dtype=expected_dtype)
        if field == "structure_offsets":
            path = split_root / "structure_offsets.npy"
            np.save(path, array, allow_pickle=False)
        else:
            path = split_root / f"{field}.bin"
            _write_binary(path, array)
        descriptors[field] = _descriptor(cache_root, path, array)

    structures_count = len(normalized)
    atom_count = int(offsets[-1])
    declared_splits = identity_payload.get("splits")
    declared = (
        declared_splits.get(split) if isinstance(declared_splits, Mapping) else None
    )
    if not isinstance(declared, Mapping):
        raise ValueError(f"split {split}: identity metadata must be a mapping")
    expected_counts = {
        "structure_count": structures_count,
        "atom_count": atom_count,
        "force_component_count": 3 * atom_count,
    }
    for field, actual in expected_counts.items():
        if declared.get(field) != actual:
            raise ValueError(f"split {split}: identity {field} mismatch")
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
    return {
        "structures": structures_count,
        "atoms": atom_count,
        "force_components": 3 * atom_count,
        **expected_counts,
        "force_feature_dim": dimensions[0],
        "energy_feature_dim": dimensions[1],
        "feature_dtype": "float32",
        "force_feature_key": force_key,
        "energy_feature_key": energy_key,
        "arrays": descriptors,
    }


def _write_split(
    cache_root: Path,
    split: str,
    structures: Iterable[RawStructure],
    identity_payload: Mapping[str, Any],
) -> dict[str, Any]:
    return write_preallocated_split(
        cache_root,
        split,
        structures,
        identity_payload,
        normalize=_normalize_structure,
        sha256=sha256_file,
    )


def build_raw_cache(
    output_root: Path,
    split_structures: Mapping[str, Iterable[RawStructure]],
    identity_payload: Mapping[str, Any],
    *,
    staging: RawCacheStaging | None = None,
) -> Path:
    """Build, fully verify, and atomically publish an identity-addressed cache."""
    if not split_structures:
        raise ValueError("split_structures must not be empty")
    for split in split_structures:
        _validate_split_name(split)
    output_root = Path(output_root)
    identity = cache_id(
        {"schema_version": SCHEMA_VERSION, "identity_payload": identity_payload}
    )
    cache_root = output_root / identity
    manifest_path = cache_root / "manifest.json"
    if manifest_path.is_file():
        existing = _validate_complete_cache(
            manifest_path, expected_identity=identity, load_shards=True
        )
        if existing.get("identity_payload") != dict(identity_payload):
            raise ValueError("complete cache identity_payload mismatch")
        if staging is not None:
            _discard_staging(staging, output_root)
        return manifest_path
    if cache_root.exists():
        raise ValueError(f"cache target exists but is not complete: {cache_root}")
    transaction = staging or prepare_raw_cache(output_root)
    if transaction.output_root.resolve() != output_root.resolve():
        raise ValueError("staging output_root mismatch")
    incomplete = {
        "schema_version": SCHEMA_VERSION,
        "status": "incomplete",
        "identity": identity,
        "cache_id": identity,
        "identity_payload": dict(identity_payload),
        "splits": {},
    }
    atomic_write_json(transaction.manifest_path, incomplete)
    try:
        splits = {
            split: _write_split(transaction.root, split, structures, identity_payload)
            for split, structures in split_structures.items()
        }
        complete = {**incomplete, "status": "complete", "splits": splits}
        atomic_write_json(transaction.manifest_path, complete)
        _validate_complete_cache(
            transaction.manifest_path,
            expected_identity=identity,
            load_shards=True,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        os.replace(transaction.root, cache_root)
    except BaseException:
        atomic_write_json(transaction.manifest_path, incomplete)
        raise
    return manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cache manifest {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"cache manifest {path} must contain a mapping")
    return loaded


def _validate_descriptor(
    cache_root: Path,
    split: str,
    field: str,
    descriptor: Any,
    *,
    full: bool,
) -> tuple[Path, tuple[int, ...], np.dtype[Any]]:
    context = f"split {split} array {field}"
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{context}: descriptor must be a mapping")
    path = _confined_array_path(cache_root, descriptor.get("path"))
    try:
        dtype = np.dtype(descriptor.get("dtype"))
    except TypeError as error:
        raise ValueError(f"{context}: invalid dtype") from error
    raw_shape = descriptor.get("shape")
    if (
        not isinstance(raw_shape, list)
        or not raw_shape
        or not all(isinstance(value, int) and value > 0 for value in raw_shape)
    ):
        raise ValueError(f"{context}: invalid shape")
    shape = tuple(raw_shape)
    declared_bytes = descriptor.get("bytes")
    if not isinstance(declared_bytes, int) or declared_bytes <= 0:
        raise ValueError(f"{context}: invalid size")
    if not path.is_file() or path.stat().st_size != declared_bytes:
        raise ValueError(f"{context}: file size mismatch")
    if field == "structure_offsets":
        try:
            offsets = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"{context}: invalid npy file") from error
        if offsets.dtype != dtype or tuple(offsets.shape) != shape:
            raise ValueError(f"{context}: dtype/shape mismatch")
    elif declared_bytes != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
        raise ValueError(f"{context}: file size does not match dtype/shape")
    digest = descriptor.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError(f"{context}: invalid sha256")
    if full and sha256_file(path) != digest:
        raise ValueError(f"{field} sha256 mismatch")
    return path, shape, dtype


def _validate_complete_cache(
    manifest_path: Path,
    *,
    expected_identity: str | None,
    load_shards: bool,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    manifest = _load_manifest(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("cache manifest schema mismatch; rebuild the v2 cache")
    if manifest.get("status") != "complete":
        raise ValueError("cache manifest status must be complete")
    payload = manifest.get("identity_payload")
    if not isinstance(payload, dict):
        raise ValueError("cache manifest identity_payload must be a mapping")
    identity = manifest.get("identity")
    derived = cache_id({"schema_version": SCHEMA_VERSION, "identity_payload": payload})
    if (
        identity != manifest.get("cache_id")
        or identity != derived
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise ValueError("cache manifest identity mismatch")
    splits = manifest.get("splits")
    payload_splits = payload.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("cache manifest splits must be non-empty")
    if not isinstance(payload_splits, dict) or set(payload_splits) != set(splits):
        raise ValueError("cache split set does not match identity payload")
    for split, split_manifest in splits.items():
        if not isinstance(split_manifest, Mapping):
            raise ValueError(f"split {split}: manifest must be a mapping")
        arrays = split_manifest.get("arrays")
        if not isinstance(arrays, Mapping) or set(arrays) != set(_ARRAY_FIELDS):
            raise ValueError(f"split {split}: array fields mismatch")
        descriptors = {
            field: _validate_descriptor(
                manifest_path.parent,
                split,
                field,
                arrays[field],
                full=load_shards,
            )
            for field in _ARRAY_FIELDS
        }
        offsets = np.load(descriptors["structure_offsets"][0], mmap_mode="r")
        structure_count = split_manifest.get("structure_count")
        atom_count = split_manifest.get("atom_count")
        if not isinstance(structure_count, int) or structure_count <= 0:
            raise ValueError(f"split {split}: invalid structure_count")
        if not isinstance(atom_count, int) or atom_count <= 0:
            raise ValueError(f"split {split}: invalid atom_count")
        if tuple(offsets.shape) != (structure_count + 1,):
            raise ValueError(f"split {split}: offsets shape mismatch")
        if offsets[0] != 0 or offsets[-1] != atom_count:
            raise ValueError(f"split {split}: offsets do not cover all atoms")
        if np.any(offsets[1:] <= offsets[:-1]):
            raise ValueError(f"split {split}: offsets must be strictly increasing")
        expected_first_dimensions = {
            "structure_ids": structure_count,
            "energy_prediction": structure_count,
            "energy_reference": structure_count,
            "atomic_numbers": atom_count,
            "force_prediction": atom_count,
            "force_reference": atom_count,
            "force_features": atom_count,
            "energy_features": atom_count,
        }
        for field, first_dimension in expected_first_dimensions.items():
            if descriptors[field][1][0] != first_dimension:
                raise ValueError(f"split {split}: {field} first dimension mismatch")
        force_dim = descriptors["force_features"][1][1]
        energy_dim = descriptors["energy_features"][1][1]
        expected_metadata = {
            "structures": structure_count,
            "atoms": atom_count,
            "structure_count": structure_count,
            "atom_count": atom_count,
            "force_components": 3 * atom_count,
            "force_component_count": 3 * atom_count,
            "force_feature_dim": force_dim,
            "energy_feature_dim": energy_dim,
            "feature_dtype": "float32",
        }
        for field, expected in expected_metadata.items():
            if split_manifest.get(field) != expected:
                raise ValueError(f"split {split}: {field} metadata mismatch")
        declared = payload_splits[split]
        if not isinstance(declared, Mapping):
            raise ValueError(f"split {split}: identity metadata must be a mapping")
        for field in ("structure_count", "atom_count", "force_component_count"):
            if declared.get(field) != expected_metadata[field]:
                raise ValueError(f"split {split}: identity {field} mismatch")
    return manifest


class CachedSplitDataset:
    """Random-access dataset backed by persistent continuous memory maps."""

    def __init__(self, manifest_path: Path, split: str, expected_identity: str) -> None:
        self.manifest_path = Path(manifest_path)
        manifest = _validate_complete_cache(
            self.manifest_path,
            expected_identity=expected_identity,
            load_shards=False,
        )
        splits = manifest["splits"]
        if split not in splits:
            raise ValueError(f"cache manifest is missing split {split!r}")
        self.split = split
        self._cache_root = self.manifest_path.parent
        self._descriptors = dict(splits[split]["arrays"])
        self._length = int(splits[split]["structure_count"])
        self._arrays: dict[str, np.ndarray] = {}

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_arrays"] = {}
        return state

    def __len__(self) -> int:
        return self._length

    def _array(self, field: str) -> np.ndarray:
        if field not in self._arrays:
            descriptor = self._descriptors[field]
            path = _confined_array_path(self._cache_root, descriptor["path"])
            shape = tuple(descriptor["shape"])
            dtype = np.dtype(descriptor["dtype"])
            if field == "structure_offsets":
                array = np.load(path, mmap_mode="c", allow_pickle=False)
            else:
                array = np.memmap(path, mode="c", dtype=dtype, shape=shape)
            self._arrays[field] = array
        return self._arrays[field]

    def _normalize_index(self, index: int) -> int:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return index

    def num_atoms(self, index: int) -> int:
        index = self._normalize_index(index)
        offsets = self._array("structure_offsets")
        return int(offsets[index + 1] - offsets[index])

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = self._normalize_index(index)
        offsets = self._array("structure_offsets")
        start = int(offsets[index])
        stop = int(offsets[index + 1])
        item = {
            field: torch.from_numpy(np.asarray(self._array(field)[start:stop]))
            for field in _ATOM_FIELDS
        }
        item.update(
            {
                field: torch.as_tensor(self._array(field)[index], dtype=torch.float32)
                for field in _ENERGY_FIELDS
            }
        )
        item["structure_id"] = int(self._array("structure_ids")[index])
        item["num_atoms"] = stop - start
        return item


def collate_cached_structures(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Copy memmap views into one contiguous batch and rebuild local offsets."""
    if not items:
        raise ValueError("cannot collate an empty cached batch")
    structure_ids = torch.tensor(
        [int(item["structure_id"]) for item in items], dtype=torch.int64
    )
    atom_counts = torch.tensor(
        [int(item["num_atoms"]) for item in items], dtype=torch.int64
    )
    offsets = torch.zeros(len(items) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(atom_counts, dim=0)
    batch = {
        field: torch.cat([torch.as_tensor(item[field]) for item in items]).contiguous()
        for field in _ATOM_FIELDS
    }
    batch.update(
        {
            field: torch.stack(
                [torch.as_tensor(item[field], dtype=torch.float32) for item in items]
            ).contiguous()
            for field in _ENERGY_FIELDS
        }
    )
    batch.update(
        {
            "structure_ids": structure_ids,
            "atom_counts": atom_counts,
            "num_atoms": atom_counts,
            "atom_offsets": offsets,
        }
    )
    return batch
