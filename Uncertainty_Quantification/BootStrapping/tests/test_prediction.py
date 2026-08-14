from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _targets():
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        TargetArrays,
    )

    return TargetArrays(
        structure_ids=np.array(["s0", "s1"]),
        num_atoms=np.array([2, 1], dtype=np.int64),
        atom_offsets=np.array([0, 2, 3], dtype=np.int64),
        energy=np.array([-1.0, -2.0]),
        forces=np.arange(9, dtype=np.float64).reshape(3, 3),
        stress=np.arange(18, dtype=np.float64).reshape(2, 3, 3),
    )


def _predictions():
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
    )

    return PredictionArrays(
        energy=np.array([-0.9, -2.1], dtype=np.float32),
        forces=np.arange(9, dtype=np.float32).reshape(3, 3),
        stress=np.arange(18, dtype=np.float32).reshape(2, 3, 3),
    )


def test_prediction_store_round_trip_and_single_targets(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionStore,
        load_prediction_arrays,
        load_target_arrays,
    )

    store = PredictionStore(
        tmp_path,
        split="test",
        units={"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"},
    )
    targets_path = store.write_targets(_targets())
    store.write_targets(_targets())
    publication = store.write_member(0, "raw", _predictions())

    loaded_targets = load_target_arrays(targets_path)
    loaded = load_prediction_arrays(publication.path)
    assert np.array_equal(loaded_targets.atom_offsets, np.array([0, 2, 3]))
    assert np.array_equal(loaded.energy, _predictions().energy)
    assert np.array_equal(loaded.forces, _predictions().forces)
    assert np.array_equal(loaded.stress, _predictions().stress)
    assert publication.path == tmp_path / "test/members/member_000/raw.npz"
    assert len(list((tmp_path / "test").glob("targets.npz"))) == 1


def test_prediction_store_rejects_layout_and_mode_mismatch(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
        PredictionStore,
    )

    store = PredictionStore(
        tmp_path,
        split="val",
        units={"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"},
    )
    store.write_targets(_targets())
    mismatched = PredictionArrays(
        energy=np.zeros(2),
        forces=np.zeros((4, 3)),
        stress=np.zeros((2, 3, 3)),
    )
    with pytest.raises(HardFailure, match="layout"):
        store.write_member(1, "raw", mismatched)
    with pytest.raises(HardFailure, match="mode"):
        store.write_member(1, "mixed", _predictions())


def test_targets_require_unique_ids_and_exact_offsets() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        TargetArrays,
        validate_targets,
    )

    targets = _targets()
    duplicate_ids = TargetArrays(
        structure_ids=np.array(["same", "same"]),
        num_atoms=targets.num_atoms,
        atom_offsets=targets.atom_offsets,
        energy=targets.energy,
        forces=targets.forces,
        stress=targets.stress,
    )
    with pytest.raises(HardFailure, match="unique"):
        validate_targets(duplicate_ids)
    wrong_offsets = TargetArrays(
        structure_ids=targets.structure_ids,
        num_atoms=targets.num_atoms,
        atom_offsets=np.array([0, 1, 3]),
        energy=targets.energy,
        forces=targets.forces,
        stress=targets.stress,
    )
    with pytest.raises(HardFailure, match="offset"):
        validate_targets(wrong_offsets)
