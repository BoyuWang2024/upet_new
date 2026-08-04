from __future__ import annotations

import math

import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head.adapters import (
    LocalToGlobalCumulantAdapter,
)


def _reference_cumulants(
    features: torch.Tensor,
    counts: torch.Tensor,
    order: int,
    signed_root: bool,
) -> torch.Tensor:
    structures: list[torch.Tensor] = []
    start = 0
    for count in counts.tolist():
        values = features[start : start + count]
        start += count
        raw = [values.pow(degree).mean(0) for degree in range(1, order + 1)]
        cumulants: list[torch.Tensor] = []
        outputs: list[torch.Tensor] = []
        for degree in range(1, order + 1):
            cumulant = raw[degree - 1]
            for lower in range(1, degree):
                cumulant = cumulant - (
                    math.comb(degree - 1, lower - 1)
                    * cumulants[lower - 1]
                    * raw[degree - lower - 1]
                )
            cumulants.append(cumulant)
            if degree > 1 and signed_root:
                magnitude = torch.abs(cumulant).clamp_min(
                    torch.finfo(cumulant.dtype).tiny
                )
                cumulant = torch.sign(cumulant) * magnitude.pow(1 / degree)
            outputs.append(cumulant)
        structures.append(torch.cat(outputs))
    return torch.stack(structures)


@pytest.mark.parametrize("order", range(1, 9))
@pytest.mark.parametrize("signed_root", [False, True])
def test_vectorized_cumulants_match_reference_and_gradients(
    order: int,
    signed_root: bool,
) -> None:
    torch.manual_seed(11)
    actual_features = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    expected_features = actual_features.detach().clone().requires_grad_(True)
    counts = torch.tensor([2, 4, 3], dtype=torch.int64)

    actual = LocalToGlobalCumulantAdapter(3, order, signed_root)(
        actual_features, counts
    )
    expected = _reference_cumulants(
        expected_features,
        counts,
        order,
        signed_root,
    )

    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(actual.sum(), actual_features)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), expected_features)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)


@pytest.mark.parametrize(
    "counts",
    [
        torch.tensor([[2, 3]]),
        torch.tensor([2.0, 3.0]),
        torch.tensor([2, 0, 3]),
        torch.tensor([2, -1, 4]),
        torch.tensor([], dtype=torch.int64),
        torch.tensor([2, 2]),
    ],
)
def test_cumulant_adapter_rejects_invalid_atom_counts(counts: torch.Tensor) -> None:
    adapter = LocalToGlobalCumulantAdapter(2, 3, signed_root=True)

    with pytest.raises(ValueError, match="atom_counts"):
        adapter(torch.randn(5, 2), counts)
