"""Path-neutral semantic fingerprints for labelled extxyz datasets."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import atomic_json_dump
from .data import iter_samples


FINGERPRINT_VERSION = "upet-llpr-dataset-semantic-v1"


@dataclass(frozen=True)
class SemanticDatasetFingerprint:
    """Digest and label counts independent of extxyz text formatting and path."""

    sha256: str
    structure_count: int
    atom_count: int
    force_component_count: int
    quantum: float


def _update_field(digest: hashlib._Hash, tag: str, payload: bytes) -> None:
    tag_bytes = tag.encode("utf-8")
    digest.update(struct.pack("<Q", len(tag_bytes)))
    digest.update(tag_bytes)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _identifier(info: dict[str, object], index: int) -> str:
    values = {
        key: str(info[key]).strip() for key in ("id", "structure_id") if key in info
    }
    if not values:
        raise ValueError(f"structure {index}: missing required id or structure_id")
    if len(set(values.values())) != 1:
        raise ValueError(f"structure {index}: conflicting id and structure_id values")
    identifier = next(iter(values.values()))
    if not identifier:
        raise ValueError(f"structure {index}: id must not be empty")
    return identifier


def _quantized(values: object, quantum: float, *, label: str) -> bytes:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains non-finite values")
    scaled = array / quantum
    limit = float(np.iinfo(np.int64).max)
    if np.any(np.abs(scaled) > limit - 1):
        raise ValueError(f"{label} exceeds semantic fingerprint integer range")
    integers = np.ascontiguousarray(np.rint(scaled).astype("<i8", copy=False))
    shape = np.asarray(array.shape, dtype="<i8")
    return struct.pack("<Q", array.ndim) + shape.tobytes() + integers.tobytes()


def semantic_dataset_fingerprint(
    path: Path, quantum: float = 1.0e-6
) -> SemanticDatasetFingerprint:
    """Hash ordered physical data used by LLPR, quantized to ``quantum``."""
    if not np.isfinite(quantum) or quantum <= 0:
        raise ValueError("quantum must be finite and positive")

    digest = hashlib.sha256()
    _update_field(digest, "version", FINGERPRINT_VERSION.encode("ascii"))
    _update_field(digest, "quantum", struct.pack("<d", quantum))
    structure_count = 0
    atom_count = 0
    force_component_count = 0

    for sample in iter_samples(Path(path)):
        atoms = sample.atoms
        index = sample.index
        identifier = _identifier(atoms.info, index)
        numbers = np.ascontiguousarray(np.asarray(atoms.numbers, dtype="<i8"))

        _update_field(digest, "structure_index", struct.pack("<Q", index))
        _update_field(digest, "structure_id", identifier.encode("utf-8"))
        _update_field(digest, "atomic_numbers", numbers.tobytes())
        _update_field(digest, "pbc", np.asarray(atoms.pbc, dtype=np.uint8).tobytes())
        _update_field(
            digest,
            "cell",
            _quantized(atoms.cell.array, quantum, label=f"structure {index} cell"),
        )
        _update_field(
            digest,
            "positions",
            _quantized(atoms.positions, quantum, label=f"structure {index} positions"),
        )
        _update_field(
            digest,
            "energy",
            _quantized(
                sample.energy_reference_total,
                quantum,
                label=f"structure {index} energy",
            ),
        )
        _update_field(
            digest,
            "forces",
            _quantized(
                sample.force_reference,
                quantum,
                label=f"structure {index} forces",
            ),
        )
        structure_count += 1
        atom_count += len(atoms)
        force_component_count += sample.force_reference.size

    _update_field(digest, "structure_count", struct.pack("<Q", structure_count))
    _update_field(digest, "atom_count", struct.pack("<Q", atom_count))
    _update_field(
        digest,
        "force_component_count",
        struct.pack("<Q", force_component_count),
    )
    return SemanticDatasetFingerprint(
        sha256=digest.hexdigest(),
        structure_count=structure_count,
        atom_count=atom_count,
        force_component_count=force_component_count,
        quantum=quantum,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute a path-neutral LLPR extxyz semantic fingerprint."
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    fingerprint = semantic_dataset_fingerprint(args.dataset)
    atomic_json_dump(args.output, dataclasses.asdict(fingerprint))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
