"""Regression tests for the historical FGE numerical formula."""

from __future__ import annotations

import torch

from Uncertainty_Quantification.FGE.fge.evaluation import (
    _risk_coverage,
    evaluate_prediction,
)


def _payload() -> dict[str, object]:
    energy = torch.tensor(
        [
            [999.8, 999.8, 999.8],
            [999.8, 999.8, 999.8],
            [999.8, 999.8, 999.8],
            [999.9, 999.9, 999.9],
        ],
        dtype=torch.float32,
    )
    force = torch.zeros((4, 3, 3), dtype=torch.float32)
    stress = torch.zeros((4, 3, 3, 3), dtype=torch.float32)
    return {
        "energy_prediction": energy,
        "forces_prediction": force,
        "stress_prediction": stress,
        "energy_reference": torch.zeros(3, dtype=torch.float32),
        "forces_reference": torch.zeros((3, 3), dtype=torch.float32),
        "stress_reference": torch.zeros((3, 3, 3), dtype=torch.float32),
        "n_atoms": torch.ones(3, dtype=torch.int64),
        "structure_offsets": torch.arange(4, dtype=torch.int64),
        "member_ids": tuple(f"member_{index:03d}" for index in range(1, 5)),
    }


def test_evaluation_preserves_legacy_float32_mean_bits() -> None:
    payload = _payload()
    energy = payload["energy_prediction"]
    assert isinstance(energy, torch.Tensor)
    native_mean = energy.mean(dim=0)
    widened_mean = energy.to(dtype=torch.float64).mean(dim=0).to(torch.float32)
    assert not torch.equal(native_mean, widened_mean)

    result = evaluate_prediction(payload, [1.0], 1e-12)

    assert torch.equal(result.ensemble["energy"], native_mean)


def test_risk_coverage_preserves_legacy_numpy_tie_boundary() -> None:
    uncertainty = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    error = torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

    rows = _risk_coverage(uncertainty, error, (0.2,))

    assert rows == [{"coverage": 0.2, "kept": 1, "risk": 50.0}]
