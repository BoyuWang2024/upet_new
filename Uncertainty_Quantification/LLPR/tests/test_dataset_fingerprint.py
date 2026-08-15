from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write

from Uncertainty_Quantification.LLPR.llpr.dataset_fingerprint import (
    main,
    semantic_dataset_fingerprint,
)


def _structure(
    *,
    identifier_key: str = "id",
    identifier: int | str = 7,
    position_delta: float = 0.0,
    energy_delta: float = 0.0,
    force_delta: float = 0.0,
) -> Atoms:
    atoms = Atoms(
        numbers=[8, 1],
        positions=[[0.0 + position_delta, 0.0, 0.0], [0.75, 0.0, 0.0]],
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=[True, False, True],
    )
    atoms.info[identifier_key] = identifier
    atoms.info["energy"] = -1.25 + energy_delta
    atoms.info["ignored_note"] = "metadata may differ"
    atoms.arrays["forces"] = np.array(
        [[0.1 + force_delta, 0.2, 0.3], [-0.1, -0.2, -0.3]],
        dtype=np.float64,
    )
    return atoms


def _write(path: Path, structures: list[Atoms]) -> None:
    write(path, structures, format="extxyz")


def test_alias_and_subquantum_format_differences_are_semantically_equal(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.extxyz"
    right = tmp_path / "right.extxyz"
    left_atoms = _structure(identifier_key="id", identifier=7)
    right_atoms = _structure(
        identifier_key="structure_id",
        identifier="7",
        position_delta=2.0e-7,
        energy_delta=2.0e-7,
        force_delta=2.0e-7,
    )
    right_atoms.info["ignored_note"] = "different ignored metadata"
    _write(left, [left_atoms])
    _write(right, [right_atoms])

    left_fingerprint = semantic_dataset_fingerprint(left)
    right_fingerprint = semantic_dataset_fingerprint(right)

    assert left_fingerprint == right_fingerprint
    assert left_fingerprint.structure_count == 1
    assert left_fingerprint.atom_count == 2
    assert left_fingerprint.force_component_count == 6
    assert left_fingerprint.quantum == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("position_delta", 2.0e-6),
        ("energy_delta", 2.0e-6),
        ("force_delta", 2.0e-6),
    ],
)
def test_above_quantum_physical_change_changes_fingerprint(
    tmp_path: Path, mutation: str, value: float
) -> None:
    reference = tmp_path / "reference.extxyz"
    changed = tmp_path / "changed.extxyz"
    _write(reference, [_structure()])
    _write(changed, [_structure(**{mutation: value})])

    assert (
        semantic_dataset_fingerprint(reference).sha256
        != semantic_dataset_fingerprint(changed).sha256
    )


def test_structure_and_atomic_order_are_bound(tmp_path: Path) -> None:
    ordered = tmp_path / "ordered.extxyz"
    reversed_structures = tmp_path / "reversed-structures.extxyz"
    reversed_atoms = tmp_path / "reversed-atoms.extxyz"
    first = _structure(identifier=1)
    second = _structure(identifier=2, energy_delta=0.5)
    _write(ordered, [first, second])
    _write(reversed_structures, [second, first])
    atom_reordered = first[[1, 0]]
    atom_reordered.info = dict(first.info)
    _write(reversed_atoms, [atom_reordered, second])

    reference = semantic_dataset_fingerprint(ordered).sha256

    assert semantic_dataset_fingerprint(reversed_structures).sha256 != reference
    assert semantic_dataset_fingerprint(reversed_atoms).sha256 != reference


def test_missing_structure_identifier_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "missing-id.extxyz"
    atoms = _structure()
    atoms.info.pop("id")
    _write(path, [atoms])

    with pytest.raises(ValueError, match="structure 0.*id.*structure_id"):
        semantic_dataset_fingerprint(path)


def test_conflicting_identifier_aliases_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "conflicting-id.extxyz"
    atoms = _structure()
    atoms.info["structure_id"] = 8
    _write(path, [atoms])

    with pytest.raises(ValueError, match="structure 0.*conflicting"):
        semantic_dataset_fingerprint(path)


def test_cli_writes_deterministic_path_neutral_json(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.extxyz"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _write(dataset, [_structure(identifier=1), _structure(identifier=2)])

    assert main(["--dataset", str(dataset), "--output", str(first)]) == 0
    assert main(["--dataset", str(dataset), "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()
    value = json.loads(first.read_text(encoding="utf-8"))
    assert value["structure_count"] == 2
    assert value["atom_count"] == 4
    assert value["force_component_count"] == 12
    assert str(dataset.resolve()) not in first.read_text(encoding="utf-8")
    assert first.read_bytes().endswith(b"\n")
