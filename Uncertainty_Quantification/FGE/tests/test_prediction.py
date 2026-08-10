"""Tests for the strict, member-major canonical prediction payload."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import cast

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import (
    FGEConfig,
    HardFailure,
    PredictionShape,
    canonical_prediction,
    validate_prediction_payload,
)


def _canonical_payload() -> dict[str, object]:
    """Hand-build two structures/three atoms without production helpers."""
    return {
        "energy_prediction": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
        ),
        "forces_prediction": torch.tensor(
            [
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                [[4.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        "stress_prediction": torch.tensor(
            [
                [
                    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
                ],
                [
                    [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
                    [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
                ],
            ],
            dtype=torch.float32,
        ),
        "energy_reference": torch.tensor([0.5, 1.5], dtype=torch.float32),
        "forces_reference": torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32,
        ),
        "stress_reference": torch.tensor(
            [
                [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
                [[0.2, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.2]],
            ],
            dtype=torch.float32,
        ),
        "n_atoms": torch.tensor([2, 1], dtype=torch.int64),
        "structure_offsets": torch.tensor([0, 2, 3], dtype=torch.int64),
        "member_ids": ("member_001", "member_002"),
        "structure_ids": ("structure_000", "structure_001"),
        "atomic_numbers": torch.tensor([1, 6, 8], dtype=torch.int64),
        "structure_mapping": torch.tensor([0, 0, 1], dtype=torch.int64),
        "target_names": {
            "energy": "energy",
            "forces": "non_conservative_forces",
            "stress": "non_conservative_stress",
        },
        "units": {
            "energy": "eV",
            "forces": "eV/angstrom",
            "stress": "eV/angstrom^3",
        },
        "statistics": {"K": 2, "S": 2, "A": 3},
    }


def _payload_tensor(payload: dict[str, object], name: str) -> torch.Tensor:
    value = payload[name]
    assert isinstance(value, torch.Tensor)
    return value


def test_canonical_prediction_keeps_one_reference_and_exact_member_order() -> None:
    """The schema protects member-major E/F/stress and one shared reference."""
    raw = _canonical_payload()

    payload = canonical_prediction(raw)
    shape = validate_prediction_payload(payload)

    assert shape == PredictionShape(K=2, S=2, A=3)
    assert payload["member_ids"] == ("member_001", "member_002")
    assert payload["structure_ids"] == ("structure_000", "structure_001")
    assert _payload_tensor(payload, "n_atoms").tolist() == [2, 1]
    assert _payload_tensor(payload, "structure_offsets").tolist() == [0, 2, 3]
    assert _payload_tensor(payload, "structure_mapping").tolist() == [0, 0, 1]
    assert _payload_tensor(payload, "atomic_numbers").tolist() == [1, 6, 8]
    assert payload["target_names"] == {
        "energy": "energy",
        "forces": "non_conservative_forces",
        "stress": "non_conservative_stress",
    }
    assert payload["units"] == {
        "energy": "eV",
        "forces": "eV/angstrom",
        "stress": "eV/angstrom^3",
    }
    assert payload["statistics"] == {"K": 2, "S": 2, "A": 3}
    assert _payload_tensor(payload, "energy_reference").shape == (2,)
    assert _payload_tensor(payload, "forces_reference").shape == (3, 3)
    assert _payload_tensor(payload, "stress_reference").shape == (2, 3, 3)
    with pytest.raises(FrozenInstanceError):
        shape.K = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("energy_prediction", torch.zeros((2, 2), dtype=torch.float64)),
        ("forces_prediction", torch.zeros((2, 2, 3), dtype=torch.float32)),
        ("stress_prediction", torch.zeros((2, 2, 9), dtype=torch.float32)),
        ("energy_reference", torch.zeros((2, 1), dtype=torch.float32)),
        ("forces_reference", torch.zeros((3, 3), dtype=torch.float64)),
        ("stress_reference", torch.zeros((2, 3, 3), dtype=torch.int64)),
        ("n_atoms", torch.tensor([2, 1], dtype=torch.int32)),
        ("structure_offsets", torch.tensor([0, 2, 3], dtype=torch.int32)),
        ("atomic_numbers", torch.tensor([1.0, 6.0, 8.0], dtype=torch.float32)),
        ("structure_mapping", torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)),
        ("member_ids", ("member_001", "member_001")),
        ("structure_ids", ("structure_001", "structure_000")),
        ("target_names", {"energy": "energy"}),
        ("units", {"energy": "eV"}),
        ("statistics", {"K": 2, "S": 3, "A": 3}),
    ],
)
def test_prediction_rejects_wrong_field_dtype_shape_order_or_metadata(
    field: str, value: object
) -> None:
    """Every schema field rejects a mutation that would corrupt its contract."""
    payload = _canonical_payload()
    payload[field] = value

    with pytest.raises(HardFailure):
        validate_prediction_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("structure_offsets", torch.tensor([1, 2, 3], dtype=torch.int64)),
        ("structure_offsets", torch.tensor([0, 2, 2], dtype=torch.int64)),
        ("structure_mapping", torch.tensor([0, 1, 1], dtype=torch.int64)),
        ("n_atoms", torch.tensor([1, 2], dtype=torch.int64)),
        ("atomic_numbers", torch.tensor([1, 0, 8], dtype=torch.int64)),
    ],
)
def test_prediction_rejects_inconsistent_atom_mapping(
    field: str, value: object
) -> None:
    """Offsets, counts, mapping, and atomic numbers must describe the same atoms."""
    payload = _canonical_payload()
    payload[field] = value

    with pytest.raises(HardFailure):
        validate_prediction_payload(payload)


def test_prediction_rejects_unknown_keys_and_nonfinite_tensors() -> None:
    """The payload is closed and all numeric tensors remain finite on CPU."""
    unknown = _canonical_payload()
    unknown["unexpected"] = "value"
    with pytest.raises(HardFailure):
        validate_prediction_payload(unknown)

    nonfinite = _canonical_payload()
    energies = nonfinite["energy_prediction"]
    assert isinstance(energies, torch.Tensor)
    energies = energies.clone()
    energies[0, 0] = math.nan
    nonfinite["energy_prediction"] = energies
    with pytest.raises(HardFailure):
        validate_prediction_payload(nonfinite)


def test_canonical_prediction_detaches_the_validated_payload_from_mutable_input() -> (
    None
):
    """Publication cannot be changed by callers retaining the source dictionary."""
    raw = _canonical_payload()
    payload = canonical_prediction(raw)
    energy = raw["energy_prediction"]
    assert isinstance(energy, torch.Tensor)
    energy[0, 0] = 99.0

    stored = payload["energy_prediction"]
    assert isinstance(stored, torch.Tensor)
    assert stored[0, 0].item() == 1.0


def test_predict_members_uses_the_manifest_order_and_publishes_one_payload(
    tmp_path: Path,
) -> None:
    """The injected runtime proves real stacking without checkpoints or extxyz data."""
    from types import SimpleNamespace

    from Uncertainty_Quantification.FGE.fge import atomic_write_json, predict_members

    class FakeRuntime:
        def __init__(self) -> None:
            self.base_loads = 0
            self.restored: list[str] = []

        def load_base(self, config: object) -> object:
            del config
            self.base_loads += 1
            return object()

        def restore_and_apply(self, base: object, member_id: str) -> None:
            assert base is not None
            self.restored.append(member_id)

        def infer_member(
            self, base: object, member_id: str, config: object
        ) -> dict[str, object]:
            del base, config
            offset = 0.0 if member_id == "member_001" else 10.0
            return {
                "energy": torch.tensor(
                    [1.0 + offset, 2.0 + offset], dtype=torch.float32
                ),
                "forces": torch.full((3, 3), offset, dtype=torch.float32),
                "stress": torch.full((2, 3, 3), offset, dtype=torch.float32),
                "energy_reference": torch.tensor([0.5, 1.5], dtype=torch.float32),
                "forces_reference": torch.zeros((3, 3), dtype=torch.float32),
                "stress_reference": torch.zeros((2, 3, 3), dtype=torch.float32),
                "n_atoms": torch.tensor([2, 1], dtype=torch.int64),
                "structure_offsets": torch.tensor([0, 2, 3], dtype=torch.int64),
                "structure_ids": ("structure_000", "structure_001"),
                "atomic_numbers": torch.tensor([1, 6, 8], dtype=torch.int64),
                "structure_mapping": torch.tensor([0, 0, 1], dtype=torch.int64),
                "target_names": {
                    "energy": "energy",
                    "forces": "non_conservative_forces",
                    "stress": "non_conservative_stress",
                },
                "units": {
                    "energy": "eV",
                    "forces": "eV/angstrom",
                    "stress": "eV/angstrom^3",
                },
            }

    root = tmp_path / "upet_fge_full"
    atomic_write_json(
        root / "training" / "manifest.json",
        {"members": [{"member_id": "member_001"}, {"member_id": "member_002"}]},
    )
    config = SimpleNamespace(
        project=SimpleNamespace(name="upet_fge_full"),
        paths=SimpleNamespace(output_root=tmp_path),
        fge=SimpleNamespace(member_count=2),
    )
    runtime = FakeRuntime()

    prediction_path = predict_members(cast(FGEConfig, config), runtime=runtime)
    payload = torch.load(prediction_path, weights_only=True, map_location="cpu")

    assert runtime.base_loads == 1
    assert runtime.restored == ["member_001", "member_002"]
    assert payload["member_ids"] == ("member_001", "member_002")
    assert payload["energy_prediction"].tolist() == [[1.0, 2.0], [11.0, 12.0]]
    assert validate_prediction_payload(payload) == PredictionShape(K=2, S=2, A=3)
