from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_member_state_applies_strictly_to_pet_last_layer() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        apply_member_state,
    )

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_last_layers = torch.nn.Parameter(torch.zeros(13_338))
            self.frozen = torch.nn.Parameter(torch.tensor([7.0]))

    model = Model()
    audit = apply_member_state(
        model, {"node_last_layers": torch.ones(13_338, dtype=torch.float32)}
    )

    assert audit.trainable_parameter_count == 13_338
    assert torch.equal(model.node_last_layers, torch.ones(13_338))
    assert model.frozen.item() == 7.0


def test_prediction_schema_signature_ignores_dataset_size(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
        PredictionStore,
        TargetArrays,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.schema import (
        prediction_schema_signature,
    )

    units = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}

    def write(root: Path, structures: int, *, reference_stress: bool) -> None:
        atoms = structures * 2
        store = PredictionStore(root, split="test", units=units)
        store.write_targets(
            TargetArrays(
                structure_ids=np.array([f"s{i}" for i in range(structures)]),
                num_atoms=np.full(structures, 2),
                atom_offsets=np.arange(0, atoms + 1, 2),
                energy=np.zeros(structures),
                forces=np.zeros((atoms, 3)),
                stress=(
                    np.zeros((structures, 3, 3)) if reference_stress else None
                ),
            )
        )
        store.write_member(
            0,
            "raw",
            PredictionArrays(
                energy=np.zeros(structures),
                forces=np.zeros((atoms, 3)),
                stress=np.zeros((structures, 3, 3)),
            ),
        )

    write(tmp_path / "small", 2, reference_stress=True)
    write(tmp_path / "large", 5, reference_stress=True)

    v1_small = prediction_schema_signature(
        tmp_path / "small", split="test", modes=("raw",), member_count=1
    )
    assert v1_small == prediction_schema_signature(
        tmp_path / "large", split="test", modes=("raw",), member_count=1
    )
    assert v1_small["schema"] == "upet.bootstrap.predictions/v1"
    assert "stress" in v1_small["targets"]

    write(tmp_path / "mad_small", 2, reference_stress=False)
    write(tmp_path / "mad_large", 5, reference_stress=False)
    v2_small = prediction_schema_signature(
        tmp_path / "mad_small", split="test", modes=("raw",), member_count=1
    )
    assert v2_small == prediction_schema_signature(
        tmp_path / "mad_large", split="test", modes=("raw",), member_count=1
    )
    assert v2_small["schema"] == "upet.bootstrap.predictions/v2"
    assert "stress" not in v2_small["targets"]


def test_prediction_schema_signature_accepts_safe_dataset_key(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
        PredictionStore,
        TargetArrays,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.schema import (
        prediction_schema_signature,
    )

    units = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}

    def write(root: Path, structures: int) -> None:
        atoms = structures * 2
        store = PredictionStore(root, split="mad_test", units=units)
        store.write_targets(
            TargetArrays(
                structure_ids=np.array([f"s{i}" for i in range(structures)]),
                num_atoms=np.full(structures, 2),
                atom_offsets=np.arange(0, atoms + 1, 2),
                energy=np.zeros(structures),
                forces=np.zeros((atoms, 3)),
                stress=None,
            )
        )
        store.write_member(
            0,
            "raw",
            PredictionArrays(
                energy=np.zeros(structures),
                forces=np.zeros((atoms, 3)),
                stress=np.zeros((structures, 3, 3)),
            ),
        )

    write(tmp_path / "small", 2)
    write(tmp_path / "large", 5)
    assert prediction_schema_signature(
        tmp_path / "small", split="mad_test", modes=("raw",), member_count=1
    ) == prediction_schema_signature(
        tmp_path / "large", split="mad_test", modes=("raw",), member_count=1
    )


def test_extract_targets_does_not_request_missing_mad_stress() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        extract_targets,
    )

    class MadAtoms:
        info = {"structure_id": "mad-0"}

        def __len__(self) -> int:
            return 2

        def get_potential_energy(self) -> float:
            return -1.25

        def get_forces(self) -> np.ndarray:
            return np.zeros((2, 3))

        def get_stress(self, *, voigt: bool) -> np.ndarray:
            raise AssertionError("MAD reference stress must not be requested")

    targets = extract_targets([MadAtoms()], "mad_test", ("energy", "forces"))

    assert targets.stress is None


def test_predict_run_preserves_raw_and_ema_combined_publication(
    tmp_path: Path, monkeypatch
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import native_prediction
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        predict_run,
    )

    class Atoms:
        info = {}

        def __len__(self) -> int:
            return 1

        def get_potential_energy(self) -> float:
            return -1.0

        def get_forces(self) -> np.ndarray:
            return np.zeros((1, 3), dtype=np.float64)

        def get_stress(self, *, voigt: bool) -> np.ndarray:
            return np.zeros((3, 3), dtype=np.float64)

    def load_member(
        base: object,
        checkpoint: Path,
        *,
        mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[int, str]:
        del base, device, dtype
        return (int(checkpoint.parts[-3].split("_")[1]), mode)

    def predict_member(
        model: tuple[int, str],
        atoms: list[Atoms],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        del atoms, batch_size, device, dtype
        return native_prediction.PredictionArrays(
            energy=np.array([float(model[0])]),
            forces=np.zeros((1, 3)),
            stress=np.zeros((1, 3, 3)),
        )

    monkeypatch.setattr(native_prediction, "_read_atoms", lambda path, limit: [Atoms()])
    monkeypatch.setattr(native_prediction, "load_pet_member_model", load_member)
    monkeypatch.setattr(native_prediction, "_predict_dataset", predict_member)
    config = SimpleNamespace(
        prediction=SimpleNamespace(
            splits=("test",),
            parameter_modes=("raw", "ema"),
            device="cpu",
            batch_size=2,
        ),
        training=SimpleNamespace(precision="float64"),
        bootstrap=SimpleNamespace(ensemble_size=2),
        experiment=SimpleNamespace(run_id="legacy"),
        checkpoint=SimpleNamespace(base_path="base.ckpt"),
        data=SimpleNamespace(test=tmp_path / "test.xyz"),
    )

    (manifest,) = predict_run(config, tmp_path)
    document = __import__("json").loads(manifest.read_text())

    assert [(item["member_index"], item["mode"]) for item in document["members"]] == [
        (0, "raw"),
        (0, "ema"),
        (1, "raw"),
        (1, "ema"),
    ]
