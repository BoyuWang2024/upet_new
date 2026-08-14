from __future__ import annotations

from pathlib import Path

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

    def write(root: Path, structures: int) -> None:
        atoms = structures * 2
        store = PredictionStore(root, split="test", units=units)
        store.write_targets(
            TargetArrays(
                structure_ids=np.array([f"s{i}" for i in range(structures)]),
                num_atoms=np.full(structures, 2),
                atom_offsets=np.arange(0, atoms + 1, 2),
                energy=np.zeros(structures),
                forces=np.zeros((atoms, 3)),
                stress=np.zeros((structures, 3, 3)),
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
        tmp_path / "small", split="test", modes=("raw",), member_count=1
    ) == prediction_schema_signature(
        tmp_path / "large", split="test", modes=("raw",), member_count=1
    )
