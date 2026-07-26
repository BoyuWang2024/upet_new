import numpy as np
import pytest
import torch
from ase import Atoms

from Uncertainty_Quantification.LLPR.llpr.calibration import CalibrationRecord
from Uncertainty_Quantification.LLPR.llpr.data import LLPRSample
from Uncertainty_Quantification.LLPR.llpr.inference import (
    evaluate_structure,
    merge_structure_results,
    summarize_evaluation,
)
from Uncertainty_Quantification.LLPR.llpr.observables import StructureJacobians


def _calibration(target: str, alpha: float) -> CalibrationRecord:
    return CalibrationRecord(
        target=target,
        eta=1.0e-6,
        alpha=alpha,
        alpha_sq=alpha**2,
        gaussian_nll=1.0,
        condition_number=2.0,
        condition_warning=False,
        count=10,
        coverage_1sigma=0.5,
        coverage_2sigma=0.9,
        coverage_3sigma=1.0,
    )


def _fixture() -> tuple[
    LLPRSample,
    StructureJacobians,
    dict[str, CalibrationRecord],
    dict[str, torch.Tensor],
]:
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    sample = LLPRSample(
        index=7,
        atoms=atoms,
        energy_reference_total=3.0,
        force_reference=np.zeros((2, 3), dtype=np.float64),
    )
    jacobians = StructureJacobians(
        energy_pred_total=torch.tensor(4.0, dtype=torch.float64),
        energy_pred_per_atom=torch.tensor(2.0, dtype=torch.float64),
        force_pred=torch.tensor([1.0, -1.0, 0.5, 2.0, -2.0, 1.5], dtype=torch.float64),
        energy_jacobian=torch.tensor([1.0, 2.0], dtype=torch.float64),
        force_jacobian=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [1.0, -1.0],
            ],
            dtype=torch.float64,
        ),
    )
    calibration = {
        "energy": _calibration("energy", 2.0),
        "force": _calibration("force", 3.0),
    }
    solvers = {
        "energy": torch.eye(2, dtype=torch.float64),
        "force": torch.eye(2, dtype=torch.float64),
    }
    return sample, jacobians, calibration, solvers


def test_evaluate_structure_uses_per_atom_energy_and_component_forces() -> None:
    sample, jacobians, calibration, solvers = _fixture()

    result = evaluate_structure(sample, jacobians, calibration, solvers)

    np.testing.assert_allclose(
        result["energy_residual"],
        result["energy_pred_per_atom"] - result["energy_true_per_atom"],
    )
    np.testing.assert_allclose(
        result["energy_calibrated_var"],
        calibration["energy"].alpha_sq * result["energy_raw_var"],
    )
    np.testing.assert_allclose(
        result["force_calibrated_var_component"],
        calibration["force"].alpha_sq * result["force_raw_var_component"],
    )
    np.testing.assert_allclose(
        result["energy_raw_var_total_derived"],
        4 * result["energy_raw_var"],
    )
    assert result["force_offsets"].tolist() == [0, 6]
    assert result["force_atom_index"].tolist() == [0, 0, 0, 1, 1, 1]
    assert result["force_cartesian_index"].tolist() == [0, 1, 2, 0, 1, 2]
    assert result["force_component_index_within_structure"].tolist() == list(range(6))
    assert np.all(result["energy_calibrated_var"] > 0)
    assert np.all(result["force_calibrated_var_component"] > 0)


def test_atom_and_structure_force_aggregates_are_consistent() -> None:
    sample, jacobians, calibration, solvers = _fixture()

    result = evaluate_structure(sample, jacobians, calibration, solvers)

    expected_atom_raw = result["force_raw_var_component"].reshape(2, 3).mean(axis=1)
    np.testing.assert_allclose(result["force_raw_var_atom_mean"], expected_atom_raw)
    assert result["force_raw_var_component_mean_structure"] == pytest.approx(
        float(result["force_raw_var_component"].mean())
    )
    assert result["force_calibrated_std_component_rms_structure"] == pytest.approx(
        float(np.sqrt(result["force_calibrated_var_component"].mean()))
    )


def test_merge_results_rebuilds_global_force_offsets() -> None:
    sample, jacobians, calibration, solvers = _fixture()
    first = evaluate_structure(sample, jacobians, calibration, solvers)
    second_sample = LLPRSample(
        index=8,
        atoms=sample.atoms,
        energy_reference_total=sample.energy_reference_total,
        force_reference=sample.force_reference,
    )
    second = evaluate_structure(second_sample, jacobians, calibration, solvers)

    merged = merge_structure_results([first, second])

    assert merged["structure_index"].tolist() == [7, 8]
    assert merged["force_offsets"].tolist() == [0, 6, 12]
    assert merged["force_structure_index"].tolist() == [7] * 6 + [8] * 6


def test_summary_counts_and_rmse_use_canonical_rows() -> None:
    sample, jacobians, calibration, solvers = _fixture()
    details = evaluate_structure(sample, jacobians, calibration, solvers)

    summary = summarize_evaluation(details)

    assert summary["structure_count"] == 1
    assert summary["atom_count"] == 2
    assert summary["force_component_count"] == 6
    assert summary["energy_rmse_per_atom"] == pytest.approx(0.5)
    assert summary["force_rmse_component"] == pytest.approx(
        float(np.sqrt(np.mean(jacobians.force_pred.numpy() ** 2)))
    )


def test_zero_quadratic_form_is_rejected() -> None:
    sample, jacobians, calibration, solvers = _fixture()
    zero = StructureJacobians(
        energy_pred_total=jacobians.energy_pred_total,
        energy_pred_per_atom=jacobians.energy_pred_per_atom,
        force_pred=jacobians.force_pred,
        energy_jacobian=torch.zeros_like(jacobians.energy_jacobian),
        force_jacobian=jacobians.force_jacobian,
    )

    with pytest.raises(ValueError, match="structure 7.*energy"):
        evaluate_structure(sample, zero, calibration, solvers)
