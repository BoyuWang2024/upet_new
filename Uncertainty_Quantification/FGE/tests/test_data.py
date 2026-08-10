"""Tests for immutable, path-free dataset identities."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from Uncertainty_Quantification.FGE.fge import DatasetIdentity, HardFailure


def test_dataset_identity_records_only_logical_dataset_provenance() -> None:
    """Absolute source locations must not enter a canonical identity."""
    identity = DatasetIdentity(
        split="test",
        structure_ids=("structure_000", "structure_001"),
        structure_count=2,
        atom_count=3,
        content_sha256="a" * 64,
        target_names=(
            ("energy", "energy"),
            ("forces", "non_conservative_forces"),
            ("stress", "non_conservative_stress"),
        ),
        units=(
            ("energy", "eV"),
            ("forces", "eV/angstrom"),
            ("stress", "eV/angstrom^3"),
        ),
    )

    assert identity.split == "test"
    assert identity.structure_ids == ("structure_000", "structure_001")
    assert identity.structure_count == 2
    assert identity.atom_count == 3
    assert identity.content_sha256 == "a" * 64
    assert identity.target_names[1] == ("forces", "non_conservative_forces")
    assert identity.units[2] == ("stress", "eV/angstrom^3")
    with pytest.raises(FrozenInstanceError):
        identity.split = "train"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", ""),
        ("structure_ids", ("structure_000", "structure_000")),
        ("structure_count", 3),
        ("atom_count", 0),
        ("content_sha256", "UPPER"),
        ("target_names", (("energy", "energy"),)),
        ("units", (("energy", "eV"), ("forces", "eV/angstrom"))),
    ],
)
def test_dataset_identity_rejects_invalid_logical_metadata(
    field: str, value: object
) -> None:
    """Each identity component protects against an incomplete dataset contract."""
    values = {
        "split": "test",
        "structure_ids": ("structure_000", "structure_001"),
        "structure_count": 2,
        "atom_count": 3,
        "content_sha256": "a" * 64,
        "target_names": (
            ("energy", "energy"),
            ("forces", "non_conservative_forces"),
            ("stress", "non_conservative_stress"),
        ),
        "units": (
            ("energy", "eV"),
            ("forces", "eV/angstrom"),
            ("stress", "eV/angstrom^3"),
        ),
    }
    values[field] = value

    with pytest.raises(HardFailure):
        DatasetIdentity(**values)  # type: ignore[arg-type]
