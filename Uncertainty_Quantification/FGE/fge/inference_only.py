"""Resumable, inference-only publication for reused FGE ensembles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch

from .artifacts import (
    assert_safe_result_path,
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)
from .errors import HardFailure
from .inference_authority import EnsembleAuthority, open_ensemble_authority
from .inference_config import InferenceConfig
from .inference_data import (
    DatasetChunk,
    DatasetScan,
    iter_dataset_chunks,
    scan_dataset,
)
from .prediction import PETPredictionRuntime


_CHUNK_KEYS = {
    "schema_version",
    "identity",
    "energy_prediction",
    "forces_prediction",
    "stress_prediction",
    "energy_reference",
    "forces_reference",
    "stress_reference",
    "n_atoms",
    "structure_offsets",
    "atomic_numbers",
    "structure_mapping",
    "statistics",
    "content_sha256",
}
_IDENTITY_KEYS = {
    "run_sha256",
    "ensemble",
    "dataset",
    "chunk_id",
    "chunk_index",
    "start_structure",
    "stop_structure",
    "member_ids",
    "structure_ids",
    "atom_counts",
    "reference_availability",
}
_MANIFEST_KEYS = {
    "schema_version",
    "run_identity",
    "ensemble_identity",
    "dataset_identity",
    "member_ids",
    "chunk_count",
    "statistics",
    "chunks",
    "artifact_writer_code_identity",
    "validator_code_identity",
}
_MANIFEST_CHUNK_KEYS = {
    "chunk_id",
    "path",
    "bytes",
    "sha256",
    "start_structure",
    "stop_structure",
    "S",
    "A",
}


@dataclass(frozen=True)
class PredictionChunkShape:
    K: int
    S: int
    A: int


class ChunkInferenceRuntime(Protocol):
    def load_reused_base(
        self,
        base_checkpoint: Path,
        expected_sha256: str,
        members_directory: Path,
    ) -> object: ...

    def prepare_chunk(self, base: object, chunk: DatasetChunk) -> object: ...

    def restore_and_apply(self, base: object, member_id: str) -> None: ...

    def infer_prepared(
        self, base: object, prepared: object
    ) -> Mapping[str, torch.Tensor]: ...


class _PETChunkRuntime:
    def __init__(self) -> None:
        self._runtime = PETPredictionRuntime()

    def load_reused_base(
        self,
        base_checkpoint: Path,
        expected_sha256: str,
        members_directory: Path,
    ) -> object:
        return self._runtime.load_reused_base(
            base_checkpoint,
            expected_sha256,
            members_directory,
        )

    def prepare_chunk(self, base: object, chunk: DatasetChunk) -> object:
        return self._runtime.prepare_ase_chunk(
            base,
            tuple(record.atoms for record in chunk.records),
        )

    def restore_and_apply(self, base: object, member_id: str) -> None:
        self._runtime.restore_and_apply(base, member_id)

    def infer_prepared(
        self, base: object, prepared: object
    ) -> Mapping[str, torch.Tensor]:
        return self._runtime.infer_prepared(base, prepared)  # type: ignore[arg-type]


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HardFailure("inference identity is not canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _run_sha256(config: InferenceConfig) -> str:
    return _canonical_sha256(config.sanitized())


def _dataset_identity(config: InferenceConfig, scan: DatasetScan) -> dict[str, object]:
    return {
        "label": config.dataset.label,
        "sha256": scan.dataset_sha256,
        "split": config.dataset.split,
        "reference_availability": dict(config.dataset.reference_availability),
        "structure_count": scan.structure_count,
        "atom_count": scan.atom_count,
    }


def _chunk_identity(
    config: InferenceConfig,
    authority: EnsembleAuthority,
    chunk: DatasetChunk,
) -> dict[str, object]:
    return {
        "run_sha256": _run_sha256(config),
        "ensemble": authority.canonical_identity(),
        "dataset": {
            "label": config.dataset.label,
            "sha256": config.dataset.expected_sha256,
            "split": config.dataset.split,
        },
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "start_structure": chunk.start_structure,
        "stop_structure": chunk.stop_structure,
        "member_ids": authority.member_ids,
        "structure_ids": chunk.structure_ids,
        "atom_counts": tuple(record.atom_count for record in chunk.records),
        "reference_availability": dict(config.dataset.reference_availability),
    }


def _tensor(
    payload: Mapping[str, object],
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor):
        raise HardFailure(f"prediction chunk {name} must be a tensor")
    if value.device.type != "cpu" or value.dtype != dtype:
        raise HardFailure(f"prediction chunk {name} has an invalid dtype or device")
    if tuple(value.shape) != shape:
        raise HardFailure(f"prediction chunk {name} has an invalid shape")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise HardFailure(f"prediction chunk {name} must be finite")
    return value


def _optional_tensor(
    payload: Mapping[str, object],
    name: str,
    available: bool,
    shape: tuple[int, ...],
) -> torch.Tensor | None:
    if not available:
        if payload.get(name) is not None:
            raise HardFailure(f"prediction chunk {name} must be absent")
        return None
    return _tensor(payload, name, shape, torch.float32)


def _content_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    identity = payload.get("identity")
    statistics = payload.get("statistics")
    digest.update(
        json.dumps(
            {"identity": identity, "statistics": statistics},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for name in (
        "energy_prediction",
        "forces_prediction",
        "stress_prediction",
        "energy_reference",
        "forces_reference",
        "stress_reference",
        "n_atoms",
        "structure_offsets",
        "atomic_numbers",
        "structure_mapping",
    ):
        digest.update(name.encode("ascii"))
        value = payload.get(name)
        if value is None:
            digest.update(b"none")
            continue
        if not isinstance(value, torch.Tensor):
            raise HardFailure(f"prediction chunk {name} is not hashable")
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_prediction_chunk(
    payload: Mapping[str, object],
    expected_identity: Mapping[str, object],
) -> PredictionChunkShape:
    """Validate one exact, self-authenticating chunk against independent identity."""
    if not isinstance(payload, Mapping) or set(payload) != _CHUNK_KEYS:
        raise HardFailure("prediction chunk has missing or unknown keys")
    if payload.get("schema_version") != "upet.fge.inference-prediction-chunk.v1":
        raise HardFailure("prediction chunk schema version is invalid")
    identity = payload.get("identity")
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _IDENTITY_KEYS
        or dict(identity) != dict(expected_identity)
    ):
        raise HardFailure("prediction chunk identity differs from expected identity")

    member_ids = identity["member_ids"]
    structure_ids = identity["structure_ids"]
    atom_counts = identity["atom_counts"]
    availability = identity["reference_availability"]
    if (
        not isinstance(member_ids, tuple)
        or member_ids != tuple(f"member_{index:03d}" for index in range(1, 9))
        or not isinstance(structure_ids, tuple)
        or not isinstance(atom_counts, tuple)
        or len(structure_ids) != len(atom_counts)
        or any(type(count) is not int or count <= 0 for count in atom_counts)
        or not isinstance(availability, Mapping)
        or set(availability) != {"energy", "forces", "stress"}
        or any(type(value) is not bool for value in availability.values())
    ):
        raise HardFailure("prediction chunk identity schema is invalid")
    K = len(member_ids)
    S = len(structure_ids)
    A = sum(cast(tuple[int, ...], atom_counts))
    if S < 1 or A < 1:
        raise HardFailure("prediction chunk shape symbols must be positive")

    _tensor(payload, "energy_prediction", (K, S), torch.float32)
    _tensor(payload, "forces_prediction", (K, A, 3), torch.float32)
    _tensor(payload, "stress_prediction", (K, S, 3, 3), torch.float32)
    _optional_tensor(
        payload,
        "energy_reference",
        cast(bool, availability["energy"]),
        (S,),
    )
    _optional_tensor(
        payload,
        "forces_reference",
        cast(bool, availability["forces"]),
        (A, 3),
    )
    _optional_tensor(
        payload,
        "stress_reference",
        cast(bool, availability["stress"]),
        (S, 3, 3),
    )
    n_atoms = _tensor(payload, "n_atoms", (S,), torch.int64)
    offsets = _tensor(payload, "structure_offsets", (S + 1,), torch.int64)
    atomic_numbers = _tensor(payload, "atomic_numbers", (A,), torch.int64)
    mapping = _tensor(payload, "structure_mapping", (A,), torch.int64)
    expected_n_atoms = torch.tensor(atom_counts, dtype=torch.int64)
    if not torch.equal(n_atoms, expected_n_atoms):
        raise HardFailure("prediction chunk n_atoms differs from identity")
    expected_offsets = torch.cat(
        (torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(dim=0))
    )
    if not torch.equal(offsets, expected_offsets):
        raise HardFailure("prediction chunk structure offsets are invalid")
    expected_mapping = torch.repeat_interleave(
        torch.arange(S, dtype=torch.int64),
        n_atoms,
    )
    if not torch.equal(mapping, expected_mapping):
        raise HardFailure("prediction chunk structure mapping is invalid")
    if bool((atomic_numbers < 1).any()) or bool((atomic_numbers > 118).any()):
        raise HardFailure("prediction chunk atomic numbers are invalid")
    if payload.get("statistics") != {"K": K, "S": S, "A": A}:
        raise HardFailure("prediction chunk statistics differ from shape")
    content_sha256 = payload.get("content_sha256")
    if not isinstance(content_sha256, str) or content_sha256 != _content_sha256(
        payload
    ):
        raise HardFailure("prediction chunk content SHA-256 differs")
    return PredictionChunkShape(K=K, S=S, A=A)


def _as_float_tensor(value: object, shape: tuple[int, ...], name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().clone()
    except (RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure(f"unable to convert prediction chunk {name}") from exc
    if tuple(tensor.shape) != shape or not bool(torch.isfinite(tensor).all()):
        raise HardFailure(f"prediction chunk {name} has an invalid shape or value")
    return tensor


def _references(
    config: InferenceConfig,
    chunk: DatasetChunk,
) -> dict[str, torch.Tensor | None]:
    availability = config.dataset.reference_availability
    energy = (
        torch.tensor(
            [float(cast(float, record.energy)) for record in chunk.records],
            dtype=torch.float32,
        )
        if availability["energy"]
        else None
    )
    forces = (
        torch.cat(
            [
                _as_float_tensor(
                    record.forces,
                    (record.atom_count, 3),
                    "forces reference",
                )
                for record in chunk.records
            ]
        )
        if availability["forces"]
        else None
    )
    stress = (
        torch.stack(
            [
                _as_float_tensor(record.stress, (3, 3), "stress reference")
                for record in chunk.records
            ]
        )
        if availability["stress"]
        else None
    )
    return {
        "energy_reference": energy,
        "forces_reference": forces,
        "stress_reference": stress,
    }


def _topology(chunk: DatasetChunk) -> dict[str, torch.Tensor]:
    n_atoms = torch.tensor(
        [record.atom_count for record in chunk.records],
        dtype=torch.int64,
    )
    atomic_numbers: list[torch.Tensor] = []
    for record in chunk.records:
        numbers = getattr(record.atoms, "numbers", None)
        try:
            tensor = torch.as_tensor(numbers, dtype=torch.int64)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure("prediction chunk lacks atomic numbers") from exc
        if tuple(tensor.shape) != (record.atom_count,):
            raise HardFailure("prediction chunk atomic numbers differ from atom count")
        atomic_numbers.append(tensor)
    return {
        "n_atoms": n_atoms,
        "structure_offsets": torch.cat(
            (torch.zeros(1, dtype=torch.int64), n_atoms.cumsum(dim=0))
        ),
        "atomic_numbers": torch.cat(atomic_numbers),
        "structure_mapping": torch.repeat_interleave(
            torch.arange(len(chunk.records), dtype=torch.int64),
            n_atoms,
        ),
    }


def _member_output(
    value: Mapping[str, torch.Tensor],
    shape: PredictionChunkShape,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if set(value) != {"energy", "forces", "stress"}:
        raise HardFailure("chunk inference output has an invalid schema")
    payload: dict[str, object] = dict(value)
    energy = _tensor(payload, "energy", (shape.S,), torch.float32)
    forces = _tensor(payload, "forces", (shape.A, 3), torch.float32)
    stress = _tensor(payload, "stress", (shape.S, 3, 3), torch.float32)
    return energy, forces, stress


def _build_chunk_payload(
    config: InferenceConfig,
    authority: EnsembleAuthority,
    chunk: DatasetChunk,
    runtime: ChunkInferenceRuntime,
    base: object,
) -> dict[str, object]:
    identity = _chunk_identity(config, authority, chunk)
    shape = PredictionChunkShape(
        K=8,
        S=len(chunk.records),
        A=chunk.atom_count,
    )
    prepared = runtime.prepare_chunk(base, chunk)
    energies: list[torch.Tensor] = []
    forces: list[torch.Tensor] = []
    stresses: list[torch.Tensor] = []
    for member_id in authority.member_ids:
        runtime.restore_and_apply(base, member_id)
        energy, force, stress = _member_output(
            runtime.infer_prepared(base, prepared),
            shape,
        )
        energies.append(energy)
        forces.append(force)
        stresses.append(stress)
    payload: dict[str, object] = {
        "schema_version": "upet.fge.inference-prediction-chunk.v1",
        "identity": identity,
        "energy_prediction": torch.stack(energies),
        "forces_prediction": torch.stack(forces),
        "stress_prediction": torch.stack(stresses),
        **_references(config, chunk),
        **_topology(chunk),
        "statistics": {"K": shape.K, "S": shape.S, "A": shape.A},
    }
    payload["content_sha256"] = _content_sha256(payload)
    validate_prediction_chunk(payload, identity)
    return payload


def _chunk_path(root: Path, chunk: DatasetChunk) -> Path:
    return root / "prediction" / "chunks" / f"{chunk.chunk_id}.pt"


def _open_chunk(path: Path, identity: Mapping[str, object]) -> PredictionChunkShape:
    if path.is_symlink() or not path.is_file():
        raise HardFailure(f"prediction chunk is not a regular file: {path.name}")
    try:
        payload = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure(f"unable to reopen prediction chunk: {path.name}") from exc
    if not isinstance(payload, Mapping):
        raise HardFailure(f"prediction chunk is not a mapping: {path.name}")
    return validate_prediction_chunk(payload, identity)


def _manifest_chunk(
    root: Path,
    path: Path,
    chunk: DatasetChunk,
    shape: PredictionChunkShape,
) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "start_structure": chunk.start_structure,
        "stop_structure": chunk.stop_structure,
        "S": shape.S,
        "A": shape.A,
    }


def _manifest_header(
    config: InferenceConfig,
    authority: EnsembleAuthority,
    scan: DatasetScan,
) -> dict[str, object]:
    return {
        "schema_version": "upet.fge.inference-prediction.v1",
        "run_identity": {"sha256": _run_sha256(config)},
        "ensemble_identity": authority.canonical_identity(),
        "dataset_identity": _dataset_identity(config, scan),
        "member_ids": list(authority.member_ids),
        "artifact_writer_code_identity": dict(authority.artifact_writer_code_identity),
        "validator_code_identity": dict(authority.validator_code_identity),
    }


def _load_existing_manifest(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    if path.is_symlink() or not path.is_file():
        raise HardFailure("existing prediction manifest is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise HardFailure("existing prediction manifest cannot be loaded") from exc
    if not isinstance(value, Mapping):
        raise HardFailure("existing prediction manifest has an invalid schema")
    return cast(Mapping[str, object], value)


def _validate_manifest_header(
    manifest: Mapping[str, object],
    header: Mapping[str, object],
) -> list[Mapping[str, object]]:
    if set(manifest) != _MANIFEST_KEYS:
        raise HardFailure("existing prediction manifest has an invalid schema")
    for key, value in header.items():
        if manifest.get(key) != value:
            raise HardFailure(f"existing prediction manifest {key} differs")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list) or any(
        not isinstance(item, Mapping) for item in chunks
    ):
        raise HardFailure("existing prediction manifest chunks are invalid")
    return cast(list[Mapping[str, object]], chunks)


def _validate_manifest_entry(
    root: Path,
    entry: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    if set(entry) != _MANIFEST_CHUNK_KEYS or dict(entry) != dict(expected):
        raise HardFailure(
            "existing prediction manifest chunk range or identity differs"
        )
    relative = entry.get("path")
    if not isinstance(relative, str):
        raise HardFailure("existing prediction manifest chunk path is invalid")
    path = root / relative
    assert_safe_result_path(root, path)
    if sha256_file(path) != entry.get("sha256") or path.stat().st_size != entry.get(
        "bytes"
    ):
        raise HardFailure("existing prediction manifest chunk hash differs")


def predict_inference_dataset(
    config: InferenceConfig,
    *,
    runtime: object | None = None,
) -> Path:
    """Publish or read-only resume one K=8 chunk-major inference prediction."""
    authority = open_ensemble_authority(config)
    scan = scan_dataset(config)
    root = config.output.root
    manifest_path = root / "prediction" / "manifest.json"
    assert_safe_result_path(root, manifest_path)
    header = _manifest_header(config, authority, scan)
    existing = _load_existing_manifest(manifest_path)
    existing_chunks = (
        _validate_manifest_header(existing, header) if existing is not None else None
    )

    typed_runtime = cast(
        ChunkInferenceRuntime,
        _PETChunkRuntime() if runtime is None else runtime,
    )
    for name in (
        "load_reused_base",
        "prepare_chunk",
        "restore_and_apply",
        "infer_prepared",
    ):
        if not callable(getattr(typed_runtime, name, None)):
            raise HardFailure("chunk inference runtime has an invalid seam")
    base: object | None = None
    entries: list[dict[str, object]] = []
    total_structures = 0
    total_atoms = 0
    for chunk in iter_dataset_chunks(config, scan):
        path = _chunk_path(root, chunk)
        assert_safe_result_path(root, path)
        identity = _chunk_identity(config, authority, chunk)
        if path.exists():
            shape = _open_chunk(path, identity)
        else:
            if existing is not None:
                raise HardFailure("existing prediction manifest is missing a chunk")
            if base is None:
                base = typed_runtime.load_reused_base(
                    config.ensemble.base_checkpoint,
                    config.ensemble.base_checkpoint_sha256,
                    authority.members_directory,
                )
            payload = _build_chunk_payload(
                config,
                authority,
                chunk,
                typed_runtime,
                base,
            )
            atomic_torch_save(path, payload)
            shape = _open_chunk(path, identity)
        entry = _manifest_chunk(root, path, chunk, shape)
        if existing_chunks is not None:
            if chunk.chunk_index >= len(existing_chunks):
                raise HardFailure("existing prediction manifest has too few chunks")
            _validate_manifest_entry(
                root,
                existing_chunks[chunk.chunk_index],
                entry,
            )
        entries.append(entry)
        total_structures += shape.S
        total_atoms += shape.A

    if total_structures != scan.structure_count or total_atoms != scan.atom_count:
        raise HardFailure("prediction chunk inventory does not cover the dataset")
    if existing is not None:
        if existing_chunks is None or len(existing_chunks) != len(entries):
            raise HardFailure("existing prediction manifest chunk count differs")
        if existing.get("chunk_count") != len(entries) or existing.get(
            "statistics"
        ) != {"K": 8, "S": total_structures, "A": total_atoms}:
            raise HardFailure("existing prediction manifest statistics differ")
        return manifest_path

    manifest = {
        **header,
        "chunk_count": len(entries),
        "statistics": {"K": 8, "S": total_structures, "A": total_atoms},
        "chunks": entries,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


__all__ = [
    "ChunkInferenceRuntime",
    "PredictionChunkShape",
    "predict_inference_dataset",
    "validate_prediction_chunk",
]
