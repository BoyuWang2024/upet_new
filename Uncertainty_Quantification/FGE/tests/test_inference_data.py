from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.inference_config import (
    ChunkPolicy,
    DatasetSourceConfig,
)
from Uncertainty_Quantification.FGE.fge.inference_data import (
    DatasetRecord,
    canonical_structure_id,
    iter_dataset_chunks,
    scan_dataset,
)


def _dataset_config(
    tmp_path: Path,
    *,
    label: str = "matpes_train",
    stress: bool = True,
    max_structures: int = 2,
    max_atoms: int = 6,
) -> tuple[DatasetSourceConfig, ChunkPolicy]:
    path = tmp_path / "dataset.extxyz"
    path.write_bytes(b"literal dataset identity\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return (
        DatasetSourceConfig(
            label=label,
            path=path,
            expected_sha256=digest,
            split="train" if label == "matpes_train" else "test",
            reference_availability={
                "energy": True,
                "forces": True,
                "stress": stress,
            },
        ),
        ChunkPolicy(max_structures=max_structures, max_atoms=max_atoms),
    )


def _records(
    atom_counts: tuple[int, ...],
    structure_ids: tuple[str | None, ...],
    *,
    stresses: tuple[object | None, ...] | None = None,
) -> tuple[DatasetRecord, ...]:
    if stresses is None:
        stresses = tuple(object() for _ in atom_counts)
    return tuple(
        DatasetRecord(
            structure_id=structure_id,
            atom_count=atom_count,
            atoms={"index": index},
            energy=float(index),
            forces=((float(index), 0.0, 0.0),) * atom_count,
            stress=stresses[index],
        )
        for index, (atom_count, structure_id) in enumerate(
            zip(atom_counts, structure_ids, strict=True)
        )
    )


def test_mad_ids_are_path_neutral_and_preserve_file_order(tmp_path: Path) -> None:
    config, _ = _dataset_config(tmp_path, label="mad_test", stress=False)
    records = _records(
        (2, 5, 3),
        (None, None, None),
        stresses=(None, None, None),
    )

    scan = scan_dataset(config, reader=lambda _: iter(records))

    assert scan.structure_ids == (
        "mad_test:00000000",
        "mad_test:00000001",
        "mad_test:00000002",
    )
    assert scan.atom_counts == (2, 5, 3)
    assert scan.structure_count == 3
    assert scan.atom_count == 10


def test_supplied_ids_preserve_source_order(tmp_path: Path) -> None:
    config, _ = _dataset_config(tmp_path)
    records = _records((1, 1, 1), ("z", "a", "m"))

    scan = scan_dataset(config, reader=lambda _: iter(records))

    assert scan.structure_ids == ("z", "a", "m")


def test_chunking_obeys_structure_and_atom_limits(tmp_path: Path) -> None:
    config, policy = _dataset_config(tmp_path)
    records = _records((3, 4, 8, 2), ("a", "b", "c", "d"))
    scan = scan_dataset(config, reader=lambda _: iter(records))

    chunks = tuple(
        iter_dataset_chunks(
            config,
            policy,
            scan,
            reader=lambda _: iter(records),
        )
    )

    assert [
        (chunk.start_structure, chunk.stop_structure, chunk.atom_count)
        for chunk in chunks
    ] == [(0, 1, 3), (1, 2, 4), (2, 3, 8), (3, 4, 2)]
    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk_000000",
        "chunk_000001",
        "chunk_000002",
        "chunk_000003",
    ]
    assert chunks[2].oversize_single_structure is True
    assert chunks[0].oversize_single_structure is False


def test_mixed_stress_availability_is_rejected(tmp_path: Path) -> None:
    config, _ = _dataset_config(tmp_path, stress=True)
    records = _records(
        (2, 2),
        ("a", "b"),
        stresses=(object(), None),
    )

    with pytest.raises(HardFailure, match="stress reference availability"):
        scan_dataset(config, reader=lambda _: iter(records))


@pytest.mark.parametrize("supplied", [None, "", "  "])
def test_non_mad_structure_id_is_required(tmp_path: Path, supplied: str | None) -> None:
    config, _ = _dataset_config(tmp_path)
    records = _records((1,), (supplied,))

    with pytest.raises(HardFailure, match="structure_id"):
        scan_dataset(config, reader=lambda _: iter(records))


def test_duplicate_structure_ids_are_rejected(tmp_path: Path) -> None:
    config, _ = _dataset_config(tmp_path)
    records = _records((1, 1), ("same", "same"))

    with pytest.raises(HardFailure, match="duplicate"):
        scan_dataset(config, reader=lambda _: iter(records))


def test_dataset_sha_must_match_config(tmp_path: Path) -> None:
    config, _ = _dataset_config(tmp_path)
    wrong = DatasetSourceConfig(
        label=config.label,
        path=config.path,
        expected_sha256="0" * 64,
        split=config.split,
        reference_availability=config.reference_availability,
    )

    with pytest.raises(HardFailure, match="SHA-256"):
        scan_dataset(wrong, reader=lambda _: iter(_records((1,), ("a",))))


def test_second_pass_must_match_scanned_ids_and_counts(tmp_path: Path) -> None:
    config, policy = _dataset_config(tmp_path)
    first = _records((2, 3), ("a", "b"))
    changed = _records((2, 4), ("a", "b"))
    scan = scan_dataset(config, reader=lambda _: iter(first))

    with pytest.raises(HardFailure, match="scan"):
        tuple(
            iter_dataset_chunks(
                config,
                policy,
                scan,
                reader=lambda _: iter(changed),
            )
        )


def test_canonical_structure_id_strips_but_never_reorders() -> None:
    assert canonical_structure_id("matpes_train", 7, " 341224 ") == "341224"
    assert canonical_structure_id("mad_test", 7, None) == "mad_test:00000007"
