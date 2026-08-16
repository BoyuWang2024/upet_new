"""Streaming dataset scans and bounded chunks for inference-only FGE runs."""

from __future__ import annotations

import os
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from .artifacts import sha256_file
from .errors import HardFailure
from .inference_config import (
    ChunkPolicy,
    DatasetSourceConfig,
    InferenceConfig,
)


@dataclass(frozen=True)
class DatasetRecord:
    structure_id: str | None
    atom_count: int
    atoms: object
    energy: object | None
    forces: object | None
    stress: object | None


@dataclass(frozen=True)
class _FileIdentity:
    sha256: str
    size: int
    mtime_ns: int
    device: int
    inode: int


@dataclass(frozen=True)
class DatasetScan:
    dataset_sha256: str
    file_size: int
    file_mtime_ns: int
    file_device: int
    file_inode: int
    structure_ids: tuple[str, ...]
    atom_counts: tuple[int, ...]
    structure_count: int
    atom_count: int

    def _file_identity(self) -> _FileIdentity:
        return _FileIdentity(
            sha256=self.dataset_sha256,
            size=self.file_size,
            mtime_ns=self.file_mtime_ns,
            device=self.file_device,
            inode=self.file_inode,
        )


@dataclass(frozen=True)
class DatasetChunk:
    chunk_index: int
    start_structure: int
    stop_structure: int
    atom_count: int
    structure_ids: tuple[str, ...]
    records: tuple[DatasetRecord, ...]
    oversize_single_structure: bool

    @property
    def chunk_id(self) -> str:
        return f"chunk_{self.chunk_index:06d}"


DatasetReader = Callable[[DatasetSourceConfig], Iterator[DatasetRecord]]


def canonical_structure_id(label: str, index: int, supplied: str | None) -> str:
    """Return a source-order-preserving, path-neutral structure identifier."""
    if supplied is not None:
        value = supplied.strip()
        if not value:
            raise HardFailure("dataset structure ID is empty: structure_id is required")
        return value
    if label != "mad_test":
        raise HardFailure("dataset structure is missing structure_id")
    return f"mad_test:{index:08d}"


def _source(config: InferenceConfig | DatasetSourceConfig) -> DatasetSourceConfig:
    if isinstance(config, InferenceConfig):
        return config.dataset
    return config


def _ordinary_file_identity(path: Path) -> _FileIdentity:
    try:
        before = os.lstat(path)
    except OSError as exc:
        raise HardFailure(f"unable to inspect inference dataset {path}: {exc}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise HardFailure("inference dataset must be an ordinary non-symlink file")
    digest = sha256_file(path)
    try:
        after = os.lstat(path)
    except OSError as exc:
        raise HardFailure(f"unable to reopen inference dataset {path}: {exc}") from exc
    stable = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if not stable or stat.S_ISLNK(after.st_mode) or not stat.S_ISREG(after.st_mode):
        raise HardFailure("inference dataset changed while its identity was read")
    return _FileIdentity(
        sha256=digest,
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _assert_same_file(path: Path, expected: _FileIdentity, phase: str) -> None:
    actual = _ordinary_file_identity(path)
    if actual != expected:
        raise HardFailure(f"inference dataset changed during {phase}")


def _validate_references(
    record: DatasetRecord,
    availability: Mapping[str, bool],
) -> None:
    if set(availability) != {"energy", "forces", "stress"} or any(
        type(availability[key]) is not bool for key in availability
    ):
        raise HardFailure("dataset reference availability contract is invalid")
    for name in ("energy", "forces", "stress"):
        present = getattr(record, name) is not None
        if present != availability[name]:
            raise HardFailure(
                f"dataset {name} reference availability differs from config"
            )


def _read_records(
    source: DatasetSourceConfig,
    reader: DatasetReader | None,
) -> Iterator[DatasetRecord]:
    if reader is not None:
        yield from reader(source)
        return
    yield from _iter_ase_records(
        source.path,
        source.label,
        source.reference_availability,
    )


def scan_dataset(
    config: InferenceConfig | DatasetSourceConfig,
    *,
    reader: DatasetReader | None = None,
) -> DatasetScan:
    """Scan compact metadata without retaining the full atomistic dataset."""
    source = _source(config)
    identity = _ordinary_file_identity(source.path)
    if identity.sha256 != source.expected_sha256:
        raise HardFailure("inference dataset SHA-256 differs from configuration")

    structure_ids: list[str] = []
    atom_counts: list[int] = []
    seen: set[str] = set()
    total_atoms = 0
    for index, record in enumerate(_read_records(source, reader)):
        if type(record.atom_count) is not int or record.atom_count <= 0:
            raise HardFailure("dataset atom count must be a positive integer")
        structure_id = canonical_structure_id(
            source.label,
            index,
            record.structure_id,
        )
        if structure_id in seen:
            raise HardFailure(f"dataset has duplicate structure_id: {structure_id}")
        _validate_references(record, source.reference_availability)
        seen.add(structure_id)
        structure_ids.append(structure_id)
        atom_counts.append(record.atom_count)
        total_atoms += record.atom_count

    if not structure_ids:
        raise HardFailure("inference dataset is empty")
    _assert_same_file(source.path, identity, "dataset scan")
    return DatasetScan(
        dataset_sha256=identity.sha256,
        file_size=identity.size,
        file_mtime_ns=identity.mtime_ns,
        file_device=identity.device,
        file_inode=identity.inode,
        structure_ids=tuple(structure_ids),
        atom_counts=tuple(atom_counts),
        structure_count=len(structure_ids),
        atom_count=total_atoms,
    )


def _chunk(
    index: int,
    start: int,
    records: list[DatasetRecord],
    structure_ids: list[str],
    atom_count: int,
    policy: ChunkPolicy,
) -> DatasetChunk:
    return DatasetChunk(
        chunk_index=index,
        start_structure=start,
        stop_structure=start + len(records),
        atom_count=atom_count,
        structure_ids=tuple(structure_ids),
        records=tuple(records),
        oversize_single_structure=(
            len(records) == 1 and records[0].atom_count > policy.max_atoms
        ),
    )


def iter_dataset_chunks(
    config: InferenceConfig | DatasetSourceConfig,
    scan_or_policy: DatasetScan | ChunkPolicy,
    direct_scan: DatasetScan | None = None,
    *,
    reader: DatasetReader | None = None,
) -> Iterator[DatasetChunk]:
    """Yield a verified second pass bounded by structure and atom counts."""
    if isinstance(config, InferenceConfig):
        if not isinstance(scan_or_policy, DatasetScan) or direct_scan is not None:
            raise TypeError("full inference config requires one DatasetScan")
        source = config.dataset
        policy = config.chunking
        scan = scan_or_policy
    else:
        if not isinstance(scan_or_policy, ChunkPolicy) or direct_scan is None:
            raise TypeError("dataset config requires ChunkPolicy and DatasetScan")
        source = config
        policy = scan_or_policy
        scan = direct_scan

    _assert_same_file(source.path, scan._file_identity(), "chunk pass start")
    buffered: list[DatasetRecord] = []
    buffered_ids: list[str] = []
    buffered_atoms = 0
    chunk_index = 0
    chunk_start = 0
    record_count = 0

    for index, record in enumerate(_read_records(source, reader)):
        if index >= scan.structure_count:
            raise HardFailure("dataset second pass has more records than scan")
        structure_id = canonical_structure_id(
            source.label,
            index,
            record.structure_id,
        )
        _validate_references(record, source.reference_availability)
        if (
            structure_id != scan.structure_ids[index]
            or record.atom_count != scan.atom_counts[index]
        ):
            raise HardFailure("dataset second pass differs from scan")

        would_exceed = buffered and (
            len(buffered) >= policy.max_structures
            or buffered_atoms + record.atom_count > policy.max_atoms
        )
        if would_exceed:
            yield _chunk(
                chunk_index,
                chunk_start,
                buffered,
                buffered_ids,
                buffered_atoms,
                policy,
            )
            chunk_index += 1
            chunk_start = index
            buffered = []
            buffered_ids = []
            buffered_atoms = 0

        buffered.append(record)
        buffered_ids.append(structure_id)
        buffered_atoms += record.atom_count
        record_count += 1

    if record_count != scan.structure_count:
        raise HardFailure("dataset second pass has fewer records than scan")
    if buffered:
        yield _chunk(
            chunk_index,
            chunk_start,
            buffered,
            buffered_ids,
            buffered_atoms,
            policy,
        )
    _assert_same_file(source.path, scan._file_identity(), "chunk pass")


def _iter_ase_records(
    path: Path,
    label: str,
    availability: Mapping[str, bool],
) -> Iterator[DatasetRecord]:
    """Read ASE records lazily; imported only by the production adapter."""
    del label
    try:
        from ase.io import iread
    except ImportError as exc:  # pragma: no cover - production environment guard
        raise HardFailure("ASE is required to read inference datasets") from exc

    try:
        atoms_iterator = iread(path, index=":")
        for atoms in atoms_iterator:
            info = getattr(atoms, "info", {})
            supplied = info.get("structure_id")
            if supplied is not None and not isinstance(supplied, str):
                supplied = str(supplied)
            yield DatasetRecord(
                structure_id=supplied,
                atom_count=len(atoms),
                atoms=atoms,
                energy=(
                    atoms.get_potential_energy() if availability["energy"] else None
                ),
                forces=atoms.get_forces() if availability["forces"] else None,
                stress=(
                    atoms.get_stress(voigt=False) if availability["stress"] else None
                ),
            )
    except HardFailure:
        raise
    except Exception as exc:
        raise HardFailure(f"unable to read inference dataset {path}: {exc}") from exc


__all__ = [
    "DatasetChunk",
    "DatasetRecord",
    "DatasetScan",
    "canonical_structure_id",
    "iter_dataset_chunks",
    "scan_dataset",
]
