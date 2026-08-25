from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from confidence_head.artifacts import sha256_file
from confidence_head.e0_calibration import (
    apply_direct_e0,
    apply_model_aware_e0,
    energy_metrics,
    energy_observed_errors,
    extract_model_e0,
    fit_direct_mad_e0,
    fit_model_aware,
    load_e0_dataset,
    solve_full_rank_svd,
)


def test_full_rank_svd_recovers_solution_and_diagnostics() -> None:
    matrix = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float64)
    expected = torch.tensor([-2.0, -5.0], dtype=torch.float64)
    rhs = matrix @ expected

    fitted = solve_full_rank_svd(matrix, rhs)

    assert torch.allclose(fitted.solution, expected, atol=1e-12, rtol=0)
    assert fitted.rank == 2
    assert fitted.condition_number > 1.0
    assert fitted.residual_rmse < 1e-12
    assert fitted.residual_max_abs < 1e-12


def test_svd_rejects_rank_deficient_and_nonfinite_inputs() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        solve_full_rank_svd(
            torch.ones((3, 2), dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        solve_full_rank_svd(
            torch.eye(2, dtype=torch.float64),
            torch.tensor([1.0, float("nan")], dtype=torch.float64),
        )


def test_direct_and_model_aware_use_required_energy_equations() -> None:
    composition = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float64)
    model_e0 = torch.tensor([-1.0, -3.0], dtype=torch.float64)
    mad_e0 = torch.tensor([-2.0, -5.0], dtype=torch.float64)
    atomization = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    target = atomization + composition @ mad_e0
    raw = atomization + composition @ model_e0
    delta = torch.tensor([0.3, -0.2], dtype=torch.float64)
    model_aware_target = raw + composition @ delta

    direct_fit = fit_direct_mad_e0(composition, target, atomization)
    aware_fit = fit_model_aware(composition, model_aware_target, raw)

    assert torch.allclose(direct_fit.solution, mad_e0, atol=1e-12, rtol=0)
    assert torch.allclose(aware_fit.solution, delta, atol=1e-12, rtol=0)
    assert torch.allclose(
        apply_direct_e0(raw, composition, model_e0, mad_e0),
        target,
        atol=1e-12,
        rtol=0,
    )
    assert torch.allclose(
        apply_model_aware_e0(raw, composition, delta),
        model_aware_target,
        atol=1e-12,
        rtol=0,
    )


def test_energy_errors_and_metrics_are_per_atom() -> None:
    prediction = torch.tensor([12.0, 7.0], dtype=torch.float64)
    target = torch.tensor([10.0, 10.0], dtype=torch.float64)
    atom_counts = torch.tensor([2, 3], dtype=torch.int64)

    observed = energy_observed_errors(prediction, target, atom_counts)
    metrics = energy_metrics(prediction, target, atom_counts)

    assert torch.equal(observed, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert metrics == {
        "mae_mev_per_atom": 1000.0,
        "rmse_mev_per_atom": 1000.0,
        "mean_signed_error_mev_per_atom": 0.0,
        "p95_absolute_error_mev_per_atom": 1000.0,
        "structures": 2,
    }


def _write_extxyz(path: Path) -> None:
    first = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    first.info["atomization_energy"] = 0.25
    first.calc = SinglePointCalculator(
        first,
        energy=-13.5,
        forces=np.zeros((1, 3)),
    )
    second = Atoms("C2", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    second.info["atomization_energy"] = -0.75
    second.calc = SinglePointCalculator(
        second,
        energy=-75.0,
        forces=np.zeros((2, 3)),
    )
    write(path, [first, second], format="extxyz")


def test_extxyz_loader_separates_absolute_and_atomization_energy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "data.extxyz"
    _write_extxyz(path)

    dataset = load_e0_dataset(
        path,
        expected_sha256=sha256_file(path),
        atomic_types=(1, 6),
    )

    assert dataset.structure_ids.tolist() == [0, 1]
    assert dataset.atom_counts.tolist() == [1, 2]
    assert dataset.composition.tolist() == [[1.0, 0.0], [0.0, 2.0]]
    assert dataset.target_energy_r2scan.tolist() == [-13.5, -75.0]
    assert dataset.atomization_energy.tolist() == [0.25, -0.75]
    assert dataset.atomic_numbers.tolist() == [1, 6, 6]


def test_extxyz_loader_rejects_unsupported_elements(tmp_path: Path) -> None:
    path = tmp_path / "data.extxyz"
    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    atoms.info["atomization_energy"] = 0.0
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0, forces=np.zeros((1, 3)))
    write(path, atoms, format="extxyz")

    with pytest.raises(ValueError, match="unsupported atomic type 2"):
        load_e0_dataset(
            path,
            expected_sha256=sha256_file(path),
            atomic_types=(1, 6),
        )


class _Samples:
    names = ["center_type"]

    def __init__(self) -> None:
        self.values = torch.tensor([[6], [1]], dtype=torch.int32)


class _Block:
    samples = _Samples()
    values = torch.tensor([[-3.0], [-1.0]], dtype=torch.float64)


class _Weights:
    def __len__(self) -> int:
        return 1

    def block(self, index: int = 0) -> _Block:
        assert index == 0
        return _Block()


class _Composition:
    atomic_types = [1, 6]

    def __init__(self) -> None:
        self.model = SimpleNamespace(weights={"energy": _Weights()})
        self.synced = False

    def sync_tensor_maps(self) -> None:
        self.synced = True


def test_model_e0_extraction_uses_center_type_labels() -> None:
    composition = _Composition()
    model = SimpleNamespace(additive_models=[composition])

    atomic_types, values = extract_model_e0(model)

    assert composition.synced
    assert atomic_types == (1, 6)
    assert torch.equal(values, torch.tensor([-1.0, -3.0], dtype=torch.float64))
