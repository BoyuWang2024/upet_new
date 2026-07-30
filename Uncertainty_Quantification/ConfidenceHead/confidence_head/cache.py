"""Immutable, sharded raw readout cache for confidence-head training."""

from __future__ import annotations

import json
import os
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .artifacts import atomic_torch_save, atomic_write_json, sha256_file
from .identity import cache_id


SCHEMA_VERSION = "upet_confidence_raw_cache_v1"
_ATOM_FIELDS = (
    "atomic_numbers",
    "force_prediction",
    "force_reference",
    "force_features",
    "energy_features",
)
_ENERGY_FIELDS = ("energy_prediction", "energy_reference")
_SHARD_FIELDS = {
    "schema_version",
    "split",
    "structure_ids",
    "num_atoms",
    "atom_offsets",
    *_ATOM_FIELDS,
    *_ENERGY_FIELDS,
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


def _context(split: str, shard_index: int, structure_id: int, field: str) -> str:
    return f"split {split} shard {shard_index} structure {structure_id}: field {field}"


def _require_tensor(
    value: Any,
    *,
    split: str,
    shard_index: int,
    structure_id: int,
    field: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(
            f"{_context(split, shard_index, structure_id, field)} must be a Tensor"
        )
    return value


def _finite_float_tensor(
    value: Any,
    *,
    split: str,
    shard_index: int,
    structure_id: int,
    field: str,
    shape: tuple[int, ...] | None = None,
    rows: int | None = None,
) -> torch.Tensor:
    tensor = _require_tensor(
        value,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field=field,
    )
    context = _context(split, shard_index, structure_id, field)
    if not tensor.is_floating_point():
        raise ValueError(f"{context} must have a floating dtype")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(
            f"{context} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    if rows is not None and (
        tensor.ndim != 2 or tensor.shape[0] != rows or tensor.shape[1] <= 0
    ):
        raise ValueError(
            f"{context} must have shape ({rows}, D) with positive D, "
            f"got {tuple(tensor.shape)}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{context} must contain only finite values")
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def _finite_scalar(
    value: Any,
    *,
    split: str,
    shard_index: int,
    structure_id: int,
    field: str,
) -> torch.Tensor:
    context = _context(split, shard_index, structure_id, field)
    if isinstance(value, torch.Tensor):
        if value.ndim != 0:
            raise ValueError(
                f"{context} must be a scalar tensor or float, "
                f"got shape {tuple(value.shape)}"
            )
        if not value.is_floating_point():
            raise ValueError(f"{context} must have a floating dtype")
        tensor = value.detach().to(device="cpu", dtype=torch.float32)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        tensor = torch.tensor(float(value), dtype=torch.float32)
    else:
        raise ValueError(f"{context} must be a scalar tensor or float")
    if not bool(torch.isfinite(tensor).item()):
        raise ValueError(f"{context} must contain only finite values")
    return tensor.contiguous().clone()


def _normalize_structure(
    raw: RawStructure,
    *,
    split: str,
    shard_index: int,
) -> RawStructure:
    if not isinstance(raw, RawStructure):
        raise ValueError(f"split {split} shard {shard_index}: expected RawStructure")
    if isinstance(raw.structure_id, bool) or not isinstance(raw.structure_id, int):
        raise ValueError(
            f"split {split} shard {shard_index}: structure_id must be an integer"
        )
    structure_id = raw.structure_id
    atomic_numbers = _require_tensor(
        raw.atomic_numbers,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="atomic_numbers",
    )
    context = _context(split, shard_index, structure_id, "atomic_numbers")
    if atomic_numbers.dtype != torch.int64:
        raise ValueError(f"{context} must have dtype int64")
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() == 0:
        raise ValueError(
            f"{context} must have shape (N,) with positive N, "
            f"got {tuple(atomic_numbers.shape)}"
        )
    atomic_numbers = atomic_numbers.detach().to(device="cpu").contiguous().clone()
    num_atoms = len(atomic_numbers)
    force_prediction = _finite_float_tensor(
        raw.force_prediction,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="force_prediction",
        shape=(num_atoms, 3),
    )
    force_reference = _finite_float_tensor(
        raw.force_reference,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="force_reference",
        shape=(num_atoms, 3),
    )
    force_features = _finite_float_tensor(
        raw.force_features,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="force_features",
        rows=num_atoms,
    )
    energy_features = _finite_float_tensor(
        raw.energy_features,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="energy_features",
        rows=num_atoms,
    )
    energy_prediction = _finite_scalar(
        raw.energy_prediction,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="energy_prediction",
    )
    energy_reference = _finite_scalar(
        raw.energy_reference,
        split=split,
        shard_index=shard_index,
        structure_id=structure_id,
        field="energy_reference",
    )
    return RawStructure(
        structure_id=structure_id,
        atomic_numbers=atomic_numbers,
        force_prediction=force_prediction,
        force_reference=force_reference,
        energy_prediction=energy_prediction,
        energy_reference=energy_reference,
        force_features=force_features,
        energy_features=energy_features,
    )


def _shard_payload(split: str, structures: Sequence[RawStructure]) -> dict[str, Any]:
    counts = torch.tensor(
        [len(structure.atomic_numbers) for structure in structures],
        dtype=torch.int64,
    )
    offsets = torch.zeros(len(counts) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(counts, dim=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "structure_ids": torch.tensor(
            [structure.structure_id for structure in structures],
            dtype=torch.int64,
        ),
        "num_atoms": counts,
        "atom_offsets": offsets,
        "atomic_numbers": torch.cat(
            [structure.atomic_numbers for structure in structures]
        ),
        "force_prediction": torch.cat(
            [structure.force_prediction for structure in structures]
        ),
        "force_reference": torch.cat(
            [structure.force_reference for structure in structures]
        ),
        "energy_prediction": torch.stack(
            [
                torch.as_tensor(structure.energy_prediction, dtype=torch.float32)
                for structure in structures
            ]
        ),
        "energy_reference": torch.stack(
            [
                torch.as_tensor(structure.energy_reference, dtype=torch.float32)
                for structure in structures
            ]
        ),
        "force_features": torch.cat(
            [structure.force_features for structure in structures]
        ).clone(),
        "energy_features": torch.cat(
            [structure.energy_features for structure in structures]
        ).clone(),
    }


def _validate_shard(
    payload: Any,
    split: str,
    shard_index: int,
    entry: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    context = f"split {split} shard {shard_index}"
    if not isinstance(payload, dict) or set(payload) != _SHARD_FIELDS:
        raise ValueError(f"{context}: shard schema fields mismatch")
    if payload["schema_version"] != SCHEMA_VERSION or payload["split"] != split:
        raise ValueError(f"{context}: schema or split mismatch")
    for field in ("structure_ids", "num_atoms", "atom_offsets", "atomic_numbers"):
        tensor = payload[field]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int64:
            raise ValueError(f"{context}: {field} must have dtype int64")
    for field in (*_ATOM_FIELDS[1:], *_ENERGY_FIELDS):
        tensor = payload[field]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32:
            raise ValueError(f"{context}: {field} must have dtype float32")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"{context}: {field} must contain only finite values")
    ids = payload["structure_ids"]
    counts = payload["num_atoms"]
    offsets = payload["atom_offsets"]
    if ids.ndim != 1 or counts.ndim != 1 or len(ids) == 0 or len(counts) != len(ids):
        raise ValueError(f"{context}: structure_ids/num_atoms shape mismatch")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError(f"{context}: duplicate structure IDs")
    if bool((counts <= 0).any().item()):
        raise ValueError(f"{context}: num_atoms must be positive")
    if offsets.ndim != 1 or len(offsets) != len(ids) + 1:
        raise ValueError(f"{context}: atom_offsets shape mismatch")
    if int(offsets[0].item()) != 0 or not torch.equal(
        offsets[1:] - offsets[:-1], counts
    ):
        raise ValueError(f"{context}: atom_offsets are inconsistent with num_atoms")
    atoms = int(counts.sum().item())
    if int(offsets[-1].item()) != atoms:
        raise ValueError(f"{context}: atom_offsets do not cover all atoms")
    expected_shapes = {
        "atomic_numbers": (atoms,),
        "force_prediction": (atoms, 3),
        "force_reference": (atoms, 3),
        "energy_prediction": (len(ids),),
        "energy_reference": (len(ids),),
    }
    for field, shape in expected_shapes.items():
        if tuple(payload[field].shape) != shape:
            raise ValueError(f"{context}: {field} must have shape {shape}")
    for field in ("force_features", "energy_features"):
        tensor = payload[field]
        if tensor.ndim != 2 or tensor.shape[0] != atoms or tensor.shape[1] <= 0:
            raise ValueError(f"{context}: {field} has invalid shape")
    if (
        payload["force_features"].untyped_storage().data_ptr()
        == payload["energy_features"].untyped_storage().data_ptr()
    ):
        raise ValueError(f"{context}: force_features and energy_features share storage")
    metadata = {
        "structures": len(ids),
        "atoms": atoms,
        "force_components": 3 * atoms,
        "force_feature_dim": int(payload["force_features"].shape[1]),
        "energy_feature_dim": int(payload["energy_features"].shape[1]),
    }
    if entry is not None:
        for field, actual in metadata.items():
            if entry.get(field) != actual:
                raise ValueError(f"{context}: manifest {field} mismatch")
    return metadata


def _write_shard(
    cache_root: Path,
    split: str,
    shard_index: int,
    structures: Sequence[RawStructure],
) -> dict[str, Any]:
    payload = _shard_payload(split, structures)
    metadata = _validate_shard(payload, split, shard_index)
    relative_path = Path(split) / f"shard-{shard_index:06d}.pt"
    path = cache_root / relative_path
    atomic_torch_save(path, payload)
    written = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    metadata = _validate_shard(written, split, shard_index)
    return {
        "path": relative_path.as_posix(),
        "sha256": sha256_file(path),
        **metadata,
    }


def _build_split(
    cache_root: Path,
    split: str,
    structures: Iterable[RawStructure],
    shard_max_atoms: int,
) -> dict[str, Any]:
    shard_entries: list[dict[str, Any]] = []
    structure_to_shard: list[int] = []
    structure_to_index: list[int] = []
    pending: list[RawStructure] = []
    pending_atoms = 0
    seen_ids: set[int] = set()
    feature_dims: tuple[int, int] | None = None
    totals = {"structures": 0, "atoms": 0, "force_components": 0}

    def flush() -> None:
        nonlocal pending, pending_atoms
        if not pending:
            return
        shard_entries.append(
            _write_shard(cache_root, split, len(shard_entries), pending)
        )
        pending = []
        pending_atoms = 0

    for raw in structures:
        proposed_atoms = (
            len(raw.atomic_numbers)
            if isinstance(raw.atomic_numbers, torch.Tensor)
            else 0
        )
        destination = len(shard_entries)
        if pending and pending_atoms + proposed_atoms > shard_max_atoms:
            flush()
            destination = len(shard_entries)
        normalized = _normalize_structure(
            raw,
            split=split,
            shard_index=destination,
        )
        if normalized.structure_id in seen_ids:
            raise ValueError(
                f"split {split}: duplicate structure ID {normalized.structure_id}"
            )
        seen_ids.add(normalized.structure_id)
        dimensions = (
            int(normalized.force_features.shape[1]),
            int(normalized.energy_features.shape[1]),
        )
        if feature_dims is None:
            feature_dims = dimensions
        elif dimensions != feature_dims:
            raise ValueError(
                f"split {split} shard {destination} structure "
                f"{normalized.structure_id}: feature dimensions {dimensions} "
                f"do not match {feature_dims}"
            )
        structure_to_shard.append(destination)
        structure_to_index.append(len(pending))
        pending.append(normalized)
        num_atoms = len(normalized.atomic_numbers)
        pending_atoms += num_atoms
        totals["structures"] += 1
        totals["atoms"] += num_atoms
        totals["force_components"] += 3 * num_atoms
    flush()
    if not shard_entries:
        raise ValueError(f"split {split}: must contain at least one structure")
    return {
        **totals,
        "shards": shard_entries,
        "structure_count": totals["structures"],
        "atom_count": totals["atoms"],
        "force_component_count": totals["force_components"],
        "force_feature_dim": feature_dims[0] if feature_dims else 0,
        "energy_feature_dim": feature_dims[1] if feature_dims else 0,
        "feature_dtype": "float32",
        "structure_to_shard": structure_to_shard,
        "structure_to_index": structure_to_index,
    }


def _verify_written_cache(cache_root: Path, splits: Mapping[str, Any]) -> None:
    for split, split_manifest in splits.items():
        for index, shard in enumerate(split_manifest["shards"]):
            path = cache_root / shard["path"]
            if not path.is_file():
                raise ValueError(f"split {split} shard {index} is missing: {path}")
            actual = sha256_file(path)
            if actual != shard["sha256"]:
                raise ValueError(
                    f"split {split} shard {index} sha256 mismatch: "
                    f"{actual} != {shard['sha256']}"
                )


def build_raw_cache(
    output_root: Path,
    split_structures: Mapping[str, Iterable[RawStructure]],
    identity_payload: Mapping[str, Any],
    shard_max_atoms: int,
    *,
    staging: RawCacheStaging | None = None,
) -> Path:
    """Build, fully verify, and atomically publish an identity-addressed cache."""
    if not isinstance(shard_max_atoms, int) or isinstance(shard_max_atoms, bool):
        raise ValueError("shard_max_atoms must be an integer")
    if shard_max_atoms <= 0:
        raise ValueError("shard_max_atoms must be positive")
    if not split_structures:
        raise ValueError("split_structures must not be empty")
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
        return manifest_path
    if cache_root.exists():
        raise ValueError(f"cache target exists but is not complete: {cache_root}")
    transaction = staging or prepare_raw_cache(output_root)
    if transaction.output_root.resolve() != output_root.resolve():
        raise ValueError("staging output_root mismatch")
    incomplete: dict[str, Any] = {
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
            split: _build_split(transaction.root, split, structures, shard_max_atoms)
            for split, structures in split_structures.items()
        }
        complete = {**incomplete, "status": "complete", "splits": splits}
        atomic_write_json(transaction.manifest_path, complete)
        _validate_complete_cache(
            transaction.manifest_path, expected_identity=identity, load_shards=True
        )
        output_root.mkdir(parents=True, exist_ok=True)
        os.replace(transaction.root, cache_root)
    except BaseException:
        atomic_write_json(transaction.manifest_path, incomplete)
        raise
    return manifest_path


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cache manifest {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"cache manifest {path} must contain a mapping")
    return loaded


def _confined_shard_path(cache_root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("shard path must be a relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError(f"shard path must be relative: {relative}")
    root = cache_root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"shard path escapes cache root: {relative}")
    return resolved


def _validate_complete_cache(
    manifest_path: Path, *, expected_identity: str | None, load_shards: bool
) -> dict[str, Any]:
    manifest = _load_manifest(manifest_path)
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise ValueError("cache manifest schema/status must be complete")
    identity = manifest.get("identity")
    payload = manifest.get("identity_payload")
    if not isinstance(payload, dict):
        raise ValueError("cache manifest identity_payload must be a mapping")
    derived = cache_id({"schema_version": SCHEMA_VERSION, "identity_payload": payload})
    if (
        identity != manifest.get("cache_id")
        or identity != derived
        or (expected_identity is not None and identity != expected_identity)
    ):
        raise ValueError("cache manifest identity mismatch")
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("cache manifest splits must be non-empty")
    root = manifest_path.parent
    for split, split_manifest in splits.items():
        if not isinstance(split_manifest, dict):
            raise ValueError(f"split {split}: manifest must be a mapping")
        shards = split_manifest.get("shards")
        if not isinstance(shards, list) or not shards:
            raise ValueError(f"split {split}: shards must be non-empty")
        totals = {"structures": 0, "atoms": 0, "force_components": 0}
        pairs: list[tuple[int, int]] = []
        seen: set[int] = set()
        dimensions: tuple[int, int] | None = None
        for index, entry in enumerate(shards):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"split {split} shard {index}: entry must be a mapping"
                )
            path = _confined_shard_path(root, entry.get("path"))
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"split {split} shard {index}: invalid sha256")
            for field in totals:
                value = entry.get(field)
                if not isinstance(value, int) or value <= 0:
                    raise ValueError(f"split {split} shard {index}: invalid {field}")
                totals[field] += value
            force_dim = entry.get("force_feature_dim")
            energy_dim = entry.get("energy_feature_dim")
            if not isinstance(force_dim, int) or force_dim <= 0:
                raise ValueError(
                    f"split {split} shard {index}: invalid force_feature_dim"
                )
            if not isinstance(energy_dim, int) or energy_dim <= 0:
                raise ValueError(
                    f"split {split} shard {index}: invalid energy_feature_dim"
                )
            shard_dimensions = (force_dim, energy_dim)
            if dimensions is None:
                dimensions = shard_dimensions
            elif dimensions != shard_dimensions:
                raise ValueError(f"split {split}: feature dimensions disagree")
            pairs.extend(
                (index, local_index) for local_index in range(entry["structures"])
            )
            if load_shards:
                if not path.is_file():
                    raise ValueError(f"split {split} shard {index} missing: {path}")
                if sha256_file(path) != digest:
                    raise ValueError(f"split {split} shard {index}: sha256 mismatch")
                loaded = torch.load(
                    path, map_location="cpu", weights_only=True, mmap=True
                )
                _validate_shard(loaded, split, index, entry)
                for structure_id in loaded["structure_ids"].tolist():
                    if structure_id in seen:
                        raise ValueError(
                            f"split {split}: duplicate structure ID {structure_id}"
                        )
                    seen.add(structure_id)
        aliases = {
            "structures": "structure_count",
            "atoms": "atom_count",
            "force_components": "force_component_count",
        }
        for field, total in totals.items():
            if (
                split_manifest.get(field) != total
                or split_manifest.get(aliases[field]) != total
            ):
                raise ValueError(f"split {split}: {field} total mismatch")
        if dimensions is None or (
            split_manifest.get("force_feature_dim") != dimensions[0]
            or split_manifest.get("energy_feature_dim") != dimensions[1]
        ):
            raise ValueError(f"split {split}: feature dimension metadata mismatch")
        shard_map = split_manifest.get("structure_to_shard")
        index_map = split_manifest.get("structure_to_index")
        if (
            not isinstance(shard_map, list)
            or len(shard_map) != totals["structures"]
            or shard_map != [pair[0] for pair in pairs]
        ):
            raise ValueError("structure_to_shard is inconsistent with shard counts")
        if (
            not isinstance(index_map, list)
            or len(index_map) != totals["structures"]
            or index_map != [pair[1] for pair in pairs]
        ):
            raise ValueError("structure_to_index is inconsistent with shard counts")
    return manifest


class CachedSplitDataset:
    """Map-style, mmap-backed access to one split with a bounded shard LRU."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        expected_identity: str,
        max_cached_shards: int = 1,
    ) -> None:
        if max_cached_shards <= 0:
            raise ValueError("max_cached_shards must be positive")
        self.manifest_path = Path(manifest_path)
        manifest = _validate_complete_cache(
            self.manifest_path, expected_identity=expected_identity, load_shards=False
        )
        if manifest.get("status") != "complete":
            raise ValueError("cache manifest status must be complete")
        if manifest.get("identity") != expected_identity:
            raise ValueError(
                "cache manifest identity mismatch: "
                f"{manifest.get('identity')!r} != {expected_identity!r}"
            )
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("cache manifest schema_version mismatch")
        splits = manifest.get("splits")
        if not isinstance(splits, dict) or split not in splits:
            raise ValueError(f"cache manifest is missing split {split!r}")
        split_manifest = splits[split]
        if not isinstance(split_manifest, dict):
            raise ValueError(f"cache split {split!r} must be a mapping")
        self.split = split
        self._cache_root = self.manifest_path.parent
        self._max_cached_shards = max_cached_shards
        self._shards = self._validate_mapping(split_manifest)
        self._structure_to_shard = split_manifest["structure_to_shard"]
        self._structure_to_index = split_manifest["structure_to_index"]
        self._loaded: OrderedDict[int, dict[str, Any]] = OrderedDict()

    def _validate_mapping(self, split_manifest: dict[str, Any]) -> list[dict[str, Any]]:
        structure_count = split_manifest.get("structures")
        shards = split_manifest.get("shards")
        shard_mapping = split_manifest.get("structure_to_shard")
        index_mapping = split_manifest.get("structure_to_index")
        if not isinstance(structure_count, int) or structure_count < 0:
            raise ValueError("split structures must be a non-negative integer")
        if not isinstance(shards, list) or not all(
            isinstance(entry, dict) for entry in shards
        ):
            raise ValueError("split shards must be a list of mappings")
        if not isinstance(shard_mapping, list) or len(shard_mapping) != structure_count:
            raise ValueError("structure_to_shard must map every structure")
        if not isinstance(index_mapping, list) or len(index_mapping) != structure_count:
            raise ValueError("structure_to_index must map every structure")
        expected_pairs = [
            (shard_index, local_index)
            for shard_index, entry in enumerate(shards)
            for local_index in range(entry.get("structures", -1))
        ]
        if list(zip(shard_mapping, index_mapping, strict=True)) != expected_pairs:
            if shard_mapping != [pair[0] for pair in expected_pairs]:
                raise ValueError("structure_to_shard is inconsistent with shard counts")
            raise ValueError("structure_to_index is inconsistent with shard counts")
        for entry in shards:
            _confined_shard_path(self._cache_root, entry.get("path"))
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise ValueError("shard sha256 must be a 64-character string")
        return shards

    def __len__(self) -> int:
        return len(self._structure_to_shard)

    def _load_shard(self, shard_index: int) -> dict[str, Any]:
        if shard_index in self._loaded:
            loaded = self._loaded.pop(shard_index)
            self._loaded[shard_index] = loaded
            return loaded
        entry = self._shards[shard_index]
        path = _confined_shard_path(self._cache_root, entry["path"])
        if not path.is_file():
            raise ValueError(f"shard {shard_index} missing: {path}")
        actual_sha256 = sha256_file(path)
        if actual_sha256 != entry["sha256"]:
            raise ValueError(
                f"shard {shard_index} sha256 mismatch: "
                f"{actual_sha256} != {entry['sha256']}"
            )
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        _validate_shard(loaded, self.split, shard_index, entry)
        self._loaded[shard_index] = loaded
        while len(self._loaded) > self._max_cached_shards:
            self._loaded.popitem(last=False)
        return loaded

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = self._structure_to_shard[index]
        local_index = self._structure_to_index[index]
        shard = self._load_shard(shard_index)
        start = int(shard["atom_offsets"][local_index].item())
        stop = int(shard["atom_offsets"][local_index + 1].item())
        item = {field: shard[field][start:stop] for field in _ATOM_FIELDS}
        item.update({field: shard[field][local_index] for field in _ENERGY_FIELDS})
        item["structure_id"] = int(shard["structure_ids"][local_index].item())
        item["num_atoms"] = stop - start
        return item


def collate_cached_structures(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate cached structure slices and rebuild batch-local atom offsets."""
    if not items:
        raise ValueError("cannot collate an empty cached batch")
    structure_ids = torch.tensor(
        [int(item["structure_id"]) for item in items],
        dtype=torch.int64,
    )
    num_atoms = torch.tensor(
        [int(item["num_atoms"]) for item in items],
        dtype=torch.int64,
    )
    offsets = torch.zeros(len(items) + 1, dtype=torch.int64)
    offsets[1:] = torch.cumsum(num_atoms, dim=0)
    batch = {
        field: torch.cat([item[field] for item in items]) for field in _ATOM_FIELDS
    }
    batch.update(
        {
            field: torch.stack(
                [torch.as_tensor(item[field], dtype=torch.float32) for item in items]
            )
            for field in _ENERGY_FIELDS
        }
    )
    batch.update(
        {
            "structure_ids": structure_ids,
            "num_atoms": num_atoms,
            "atom_offsets": offsets,
        }
    )
    return batch
