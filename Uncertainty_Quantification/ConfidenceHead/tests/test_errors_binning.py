from __future__ import annotations

import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head.binning import (
    expected_error,
    fixed_linear_binning,
    labels_from_thresholds,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.errors import (
    energy_per_atom_error,
    force_component_error,
)


def test_force_component_error_preserves_components() -> None:
    prediction = torch.tensor([[1.0, -2.0, 3.0]])
    reference = torch.tensor([[0.0, 1.0, 1.0]])

    error = force_component_error(prediction, reference)

    torch.testing.assert_close(error, torch.tensor([[1.0, 3.0, 2.0]]))


def test_energy_error_is_normalized_per_atom() -> None:
    prediction = torch.tensor([10.0, -5.0])
    reference = torch.tensor([6.0, -1.0])
    atom_counts = torch.tensor([2, 4])

    error = energy_per_atom_error(prediction, reference, atom_counts)

    torch.testing.assert_close(error, torch.tensor([2.0, 1.0]))


def test_fixed_linear_binning_uses_upper_bins_and_saturates() -> None:
    spec = fixed_linear_binning(5, 0.5)
    values = torch.tensor([0.0, 0.099, 0.1, 0.5, 0.9])

    labels = labels_from_thresholds(values, spec.thresholds)

    torch.testing.assert_close(labels, torch.tensor([0, 0, 1, 4, 4]))
    torch.testing.assert_close(
        spec.representatives,
        torch.tensor([0.05, 0.15, 0.25, 0.35, 0.45]),
    )


def test_fixed_linear_binning_rejects_nan_max_error() -> None:
    with pytest.raises(ValueError, match="positive"):
        fixed_linear_binning(num_bins=5, max_error=float("nan"))


def test_expected_error_is_probability_weighted() -> None:
    logits = torch.log(torch.tensor([[0.25, 0.75]]))
    representatives = torch.tensor([0.1, 0.3])

    result = expected_error(logits, representatives)

    torch.testing.assert_close(result, torch.tensor([0.25]))


def test_energy_error_rejects_non_positive_atom_count() -> None:
    with pytest.raises(ValueError, match="positive"):
        energy_per_atom_error(
            torch.tensor([1.0]),
            torch.tensor([0.0]),
            torch.tensor([0]),
        )
