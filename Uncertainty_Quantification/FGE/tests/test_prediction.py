"""Tests for the strict, member-major canonical prediction payload."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import (
    DatasetIdentity,
    ExperimentLayout,
    FGEConfig,
    HardFailure,
    PredictionShape,
    atomic_write_json,
    canonical_prediction,
    load_config,
    predict_members,
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


def _standard_dataset_identity(
    *,
    content_sha256: str = "a" * 64,
    target_names: tuple[tuple[str, str], ...] = (
        ("energy", "energy"),
        ("forces", "non_conservative_forces"),
        ("stress", "non_conservative_stress"),
    ),
    units: tuple[tuple[str, str], ...] = (
        ("energy", "eV"),
        ("forces", "eV/angstrom"),
        ("stress", "eV/angstrom^3"),
    ),
) -> DatasetIdentity:
    return DatasetIdentity(
        split="test",
        structure_ids=("structure_000", "structure_001"),
        structure_count=2,
        atom_count=3,
        content_sha256=content_sha256,
        target_names=target_names,
        units=units,
    )


def _literal_member_output() -> dict[str, object]:
    payload = _canonical_payload()
    return {
        "energy": _payload_tensor(payload, "energy_prediction")[0].clone(),
        "forces": _payload_tensor(payload, "forces_prediction")[0].clone(),
        "stress": _payload_tensor(payload, "stress_prediction")[0].clone(),
        "energy_reference": _payload_tensor(payload, "energy_reference").clone(),
        "forces_reference": _payload_tensor(payload, "forces_reference").clone(),
        "stress_reference": _payload_tensor(payload, "stress_reference").clone(),
        "n_atoms": _payload_tensor(payload, "n_atoms").clone(),
        "structure_offsets": _payload_tensor(payload, "structure_offsets").clone(),
        "structure_ids": ("structure_000", "structure_001"),
        "atomic_numbers": _payload_tensor(payload, "atomic_numbers").clone(),
        "structure_mapping": _payload_tensor(payload, "structure_mapping").clone(),
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


class _LiteralRuntime:
    def __init__(self, identity: DatasetIdentity) -> None:
        self.identity = identity

    def load_base(self, config: object) -> object:
        del config
        return object()

    def restore_and_apply(self, base: object, member_id: str) -> None:
        del base, member_id

    def dataset_identity(self, config: object) -> DatasetIdentity:
        del config
        return self.identity

    def infer_member(
        self, base: object, member_id: str, config: object
    ) -> dict[str, object]:
        del base, member_id, config
        return _literal_member_output()


def _prediction_config(tmp_path: Path, *, test_data_sha256: str) -> FGEConfig:
    root = tmp_path / "upet_fge_full"
    atomic_write_json(
        root / "training" / "manifest.json",
        {"members": [{"member_id": "member_001"}, {"member_id": "member_002"}]},
    )
    return cast(
        FGEConfig,
        SimpleNamespace(
            project=SimpleNamespace(name="upet_fge_full"),
            paths=SimpleNamespace(output_root=tmp_path),
            fge=SimpleNamespace(member_count=2),
            identity=SimpleNamespace(test_data_sha256=test_data_sha256),
        ),
    )


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

        def dataset_identity(self, config: object) -> DatasetIdentity:
            del config
            return DatasetIdentity(
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
        identity=SimpleNamespace(test_data_sha256="a" * 64),
    )
    runtime = FakeRuntime()

    prediction_path = predict_members(cast(FGEConfig, config), runtime=runtime)
    payload = torch.load(prediction_path, weights_only=True, map_location="cpu")

    assert runtime.base_loads == 1
    assert runtime.restored == ["member_001", "member_002"]
    assert payload["member_ids"] == ("member_001", "member_002")
    assert payload["energy_prediction"].tolist() == [[1.0, 2.0], [11.0, 12.0]]
    assert validate_prediction_payload(payload) == PredictionShape(K=2, S=2, A=3)


def test_predict_members_uses_pet_runtime_when_not_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public prediction stage must bind its official PET runtime by default."""
    from Uncertainty_Quantification.FGE.fge import prediction

    runtime = _LiteralRuntime(_standard_dataset_identity())
    monkeypatch.setattr(prediction, "PETPredictionRuntime", lambda: runtime)
    config = _prediction_config(tmp_path, test_data_sha256="a" * 64)
    layout = ExperimentLayout(tmp_path / "upet_fge_full")
    atomic_write_json(
        layout.training_manifest,
        {"members": [{"member_id": "member_001"}, {"member_id": "member_002"}]},
    )

    path = predict_members(config)

    assert path == layout.prediction_tensor


def test_manifest_member_ids_rejects_reordered_members(tmp_path: Path) -> None:
    """Prediction must bind K to the canonical contiguous member order."""
    from Uncertainty_Quantification.FGE.fge.prediction import _manifest_member_ids

    manifest = tmp_path / "manifest.json"
    atomic_write_json(
        manifest,
        {"members": [{"member_id": "member_002"}, {"member_id": "member_001"}]},
    )

    with pytest.raises(HardFailure):
        _manifest_member_ids(manifest, 2)


def test_predict_members_rejects_sorted_but_wrong_dataset_identity(
    tmp_path: Path,
) -> None:
    """Runtime output structure IDs must equal the expected dataset identity."""
    from types import SimpleNamespace

    from Uncertainty_Quantification.FGE.fge import DatasetIdentity, predict_members

    class WrongIdentityRuntime:
        def load_base(self, config: object) -> object:
            del config
            return object()

        def restore_and_apply(self, base: object, member_id: str) -> None:
            del base, member_id

        def dataset_identity(self, config: object) -> DatasetIdentity:
            del config
            return DatasetIdentity(
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

        def infer_member(
            self, base: object, member_id: str, config: object
        ) -> dict[str, object]:
            del base, member_id, config
            return {
                "energy": torch.zeros(2, dtype=torch.float32),
                "forces": torch.zeros((3, 3), dtype=torch.float32),
                "stress": torch.zeros((2, 3, 3), dtype=torch.float32),
                "energy_reference": torch.zeros(2, dtype=torch.float32),
                "forces_reference": torch.zeros((3, 3), dtype=torch.float32),
                "stress_reference": torch.zeros((2, 3, 3), dtype=torch.float32),
                "n_atoms": torch.tensor([2, 1], dtype=torch.int64),
                "structure_offsets": torch.tensor([0, 2, 3], dtype=torch.int64),
                "structure_ids": ("structure_000", "structure_002"),
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
        identity=SimpleNamespace(test_data_sha256="a" * 64),
    )

    with pytest.raises(HardFailure):
        predict_members(cast(FGEConfig, config), runtime=WrongIdentityRuntime())


@pytest.mark.parametrize(
    "member_ids",
    [("member_002", "member_001"), ("member_001", "member_003")],
)
def test_prediction_payload_rejects_noncanonical_member_ids(
    member_ids: tuple[str, str],
) -> None:
    """A standalone payload must use contiguous canonical FGE member IDs."""
    payload = _canonical_payload()
    payload["member_ids"] = member_ids

    with pytest.raises(HardFailure):
        validate_prediction_payload(payload)


@pytest.mark.parametrize("mismatch", ["target_names", "units"])
def test_predict_members_rejects_runtime_identity_metadata_mismatch(
    tmp_path: Path, mismatch: str
) -> None:
    """Runtime output metadata must bind exactly to DatasetIdentity roles."""
    target_names = (
        ("energy", "energy"),
        ("forces", "non_conservative_forces"),
        ("stress", "non_conservative_stress"),
    )
    units = (
        ("energy", "eV"),
        ("forces", "eV/angstrom"),
        ("stress", "eV/angstrom^3"),
    )
    if mismatch == "target_names":
        target_names = (
            ("energy", "different_energy"),
            ("forces", "non_conservative_forces"),
            ("stress", "non_conservative_stress"),
        )
    else:
        units = (
            ("energy", "kcal/mol"),
            ("forces", "eV/angstrom"),
            ("stress", "eV/angstrom^3"),
        )

    with pytest.raises(HardFailure):
        predict_members(
            _prediction_config(tmp_path, test_data_sha256="a" * 64),
            runtime=_LiteralRuntime(
                _standard_dataset_identity(target_names=target_names, units=units)
            ),
        )


def test_predict_members_rejects_runtime_identity_content_sha_mismatch(
    tmp_path: Path,
) -> None:
    """The runtime's expected test identity must match the config test-data SHA."""
    with pytest.raises(HardFailure):
        predict_members(
            _prediction_config(tmp_path, test_data_sha256="a" * 64),
            runtime=_LiteralRuntime(
                _standard_dataset_identity(content_sha256="b" * 64)
            ),
        )


@pytest.mark.fge_n20
def test_default_pet_runtime_reads_the_restart_checkpoint_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default runtime must use metatrain's restart loader, not export state."""
    checkpoint = Path("/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt")
    if not checkpoint.is_file():
        pytest.skip("remote PET checkpoint is not available")
    from Uncertainty_Quantification.FGE.fge.prediction import PETPredictionRuntime

    n20_data = Path("/home/bywang/code/UQ/upet_new/data/dataset/matpes_n20.extxyz")
    for name, value in {
        "UPET_FGE_BASE_CHECKPOINT": checkpoint,
        "UPET_FGE_TRAIN_DATA": n20_data,
        "UPET_FGE_VAL_DATA": n20_data,
        "UPET_FGE_TEST_DATA": n20_data,
        "UPET_FGE_OUTPUT_ROOT": tmp_path,
    }.items():
        monkeypatch.setenv(name, str(value))
    config = load_config(
        Path(__file__).parents[1] / "configs" / "upet_fge_n20_cpu.yaml"
    )
    runtime = PETPredictionRuntime()
    base = runtime.load_base(config)

    assert base.model is not None
    assert set(base.state_dict) == set(base.model.state_dict())
    assert all(tensor.device.type == "cpu" for tensor in base.state_dict.values())
    identity = runtime.dataset_identity(config)
    assert identity.structure_ids[0] == "341224"
    assert identity.structure_count == 20
    assert identity.atom_count > 0

    output = runtime.infer_member(base, "member_001", config)

    assert set(output) == {
        "energy",
        "forces",
        "stress",
        "energy_reference",
        "forces_reference",
        "stress_reference",
        "n_atoms",
        "structure_offsets",
        "structure_ids",
        "atomic_numbers",
        "structure_mapping",
        "target_names",
        "units",
    }
    energy = output["energy"]
    forces = output["forces"]
    stress = output["stress"]
    assert isinstance(energy, torch.Tensor)
    assert isinstance(forces, torch.Tensor)
    assert isinstance(stress, torch.Tensor)
    assert energy.shape == (20,)
    assert forces.shape == (identity.atom_count, 3)
    assert stress.shape == (20, 3, 3)
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.io import write

    missing_forces = Atoms(
        "H", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True
    )
    missing_forces.info["structure_id"] = "missing_forces"
    missing_forces.calc = SinglePointCalculator(
        missing_forces, energy=0.0, stress=torch.zeros((3, 3)).numpy()
    )
    bad_path = tmp_path / "missing_forces.extxyz"
    write(bad_path, missing_forces, format="extxyz")
    bad_config = replace(config, paths=replace(config.paths, test_data=bad_path))

    with pytest.raises(HardFailure, match="extxyz energy/forces/stress"):
        runtime.infer_member(base, "member_001", bad_config)
