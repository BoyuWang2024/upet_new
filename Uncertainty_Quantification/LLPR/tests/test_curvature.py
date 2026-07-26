from pathlib import Path

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.curvature import (
    CurvatureAccumulator,
    load_curvature_progress,
    save_curvature_progress,
)
from Uncertainty_Quantification.LLPR.llpr.observables import (
    huber_curvature,
    parameter_jacobian,
)


def test_huber_curvature_and_legacy_scaling() -> None:
    accumulator = CurvatureAccumulator.zeros(energy_dim=2, force_dim=2)
    accumulator.add_structure(
        energy_gradient=torch.tensor([1.0, 2.0]),
        energy_residual=torch.tensor(0.01),
        force_gradients=torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        force_residuals=torch.tensor([0.001, 0.02, -0.001]),
        num_atoms=1,
        energy_weight=1.0,
        force_weight=0.1,
        energy_delta=0.015,
        force_delta=0.01,
    )

    torch.testing.assert_close(
        accumulator.energy,
        torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64),
    )
    expected_force = (0.1 / 3.0) * torch.tensor(
        [[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(accumulator.force, expected_force)
    assert accumulator.structure_count == 1
    assert accumulator.atom_count == 1
    assert accumulator.force_component_count == 3


def test_all_huber_outliers_have_zero_curvature() -> None:
    residuals = torch.tensor([-2.0, -0.011, 0.011, 3.0])

    result = huber_curvature(residuals, delta=0.01)

    torch.testing.assert_close(result, torch.zeros(4, dtype=torch.float64))


def test_huber_boundary_is_inlier() -> None:
    residuals = torch.tensor([-0.01, 0.0, 0.01])

    result = huber_curvature(residuals, delta=0.01)

    torch.testing.assert_close(result, torch.ones(3, dtype=torch.float64))


def test_scalar_and_batched_parameter_jacobians_match() -> None:
    layer = torch.nn.Linear(3, 4, bias=True, dtype=torch.float64)
    values = layer(torch.tensor([[0.2, -0.5, 1.3]], dtype=torch.float64)).reshape(-1)
    parameters = tuple(layer.parameters())

    scalar = parameter_jacobian(values, parameters, backend="scalar", chunk_size=2)
    batched = parameter_jacobian(values, parameters, backend="batched", chunk_size=2)

    assert scalar.shape == (4, 16)
    torch.testing.assert_close(scalar, batched, rtol=1.0e-10, atol=1.0e-12)


def test_progress_round_trip_and_identity_guard(tmp_path: Path) -> None:
    path = tmp_path / "progress.npz"
    accumulator = CurvatureAccumulator.zeros(energy_dim=2, force_dim=3)
    accumulator.structure_count = 2
    accumulator.atom_count = 5
    accumulator.force_component_count = 15
    accumulator.energy[:] = torch.eye(2, dtype=torch.float64)
    accumulator.force[:] = torch.eye(3, dtype=torch.float64)

    save_curvature_progress(path, accumulator, next_structure_index=2, identity="abc")
    restored, next_index = load_curvature_progress(path, expected_identity="abc")

    assert next_index == 2
    assert restored.structure_count == 2
    assert restored.atom_count == 5
    assert restored.force_component_count == 15
    np.testing.assert_array_equal(restored.energy.numpy(), np.eye(2))
    np.testing.assert_array_equal(restored.force.numpy(), np.eye(3))
    with pytest.raises(ValueError, match="curvature identity mismatch"):
        load_curvature_progress(path, expected_identity="different")
