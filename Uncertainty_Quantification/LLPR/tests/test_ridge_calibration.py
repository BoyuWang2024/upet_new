import math

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.calibration import (
    CalibrationRecord,
    gaussian_nll,
    moment_alpha,
    select_candidate,
)
from Uncertainty_Quantification.LLPR.llpr.ridge import (
    fit_candidates,
    fixed_candidate,
    prepare_spectrum,
    quadratic_forms,
)


def test_fixed_eta_matches_direct_inverse() -> None:
    h = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    gradients = torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64)
    eta = 1.0e-6
    candidate = fixed_candidate(
        "energy", eta, prepare_spectrum(h), max_condition_number=1.0e10
    )

    q = quadratic_forms(candidate.cholesky, gradients)
    expected = torch.einsum(
        "bi,ij,bj->b",
        gradients,
        torch.linalg.inv(h + eta * torch.eye(2, dtype=torch.float64)),
        gradients,
    )

    assert candidate.eta == eta
    torch.testing.assert_close(q, expected)


def test_fixed_eta_warns_but_is_never_changed() -> None:
    spectrum = prepare_spectrum(
        torch.diag(torch.tensor([0.0, 1.0], dtype=torch.float64))
    )

    candidate = fixed_candidate("force", 1.0e-12, spectrum, max_condition_number=1.0e6)

    assert candidate.eta == 1.0e-12
    assert candidate.condition_number > 1.0e6
    assert candidate.condition_warning


def test_fit_candidates_respect_condition_floor_and_multipliers() -> None:
    spectrum = prepare_spectrum(
        torch.diag(torch.tensor([0.0, 10.0], dtype=torch.float64))
    )

    candidates = fit_candidates(
        "energy",
        spectrum,
        multipliers=[1, 3, 10],
        max_condition_number=101.0,
    )

    assert [candidate.eta for candidate in candidates] == sorted(
        {candidate.eta for candidate in candidates}
    )
    assert len(candidates) == 3
    assert candidates[1].eta == pytest.approx(3 * candidates[0].eta)
    assert candidates[2].eta == pytest.approx(10 * candidates[0].eta)
    assert all(candidate.condition_number <= 101.0 for candidate in candidates)
    assert all(not candidate.condition_warning for candidate in candidates)


def test_moment_alpha_and_gaussian_nll_match_hand_calculation() -> None:
    residuals = torch.tensor([2.0, -1.0], dtype=torch.float64)
    q = torch.tensor([4.0, 1.0], dtype=torch.float64)

    alpha = moment_alpha(residuals, q)
    variance = alpha**2 * q
    nll = gaussian_nll(residuals, variance)

    assert alpha == pytest.approx(1.0)
    expected = np.mean(
        0.5 * (np.log(2 * np.pi * np.array([4.0, 1.0])) + np.array([1.0, 1.0]))
    )
    assert nll == pytest.approx(float(expected))


def _record(target: str, eta: float, nll: float) -> CalibrationRecord:
    return CalibrationRecord(
        target=target,
        eta=eta,
        alpha=1.0,
        alpha_sq=1.0,
        gaussian_nll=nll,
        condition_number=2.0,
        condition_warning=False,
        count=3,
        coverage_1sigma=0.5,
        coverage_2sigma=1.0,
        coverage_3sigma=1.0,
    )


def test_minimum_nll_wins_and_exact_tie_selects_smaller_eta() -> None:
    best = select_candidate(
        [
            _record("energy", 1.0e-4, 3.0),
            _record("energy", 1.0e-3, 2.0),
            _record("energy", 1.0e-2, 2.0),
        ]
    )

    assert best.eta == pytest.approx(1.0e-3)


def test_energy_and_force_selection_are_independent() -> None:
    energy = select_candidate(
        [_record("energy", 1.0e-4, 1.0), _record("energy", 1.0e-3, 2.0)]
    )
    force = select_candidate(
        [_record("force", 1.0e-4, 2.0), _record("force", 1.0e-3, 1.0)]
    )

    assert energy.eta == pytest.approx(1.0e-4)
    assert force.eta == pytest.approx(1.0e-3)
    assert math.isfinite(energy.gaussian_nll)
    assert math.isfinite(force.gaussian_nll)


def test_invalid_quadratic_forms_are_rejected() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        moment_alpha(
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
        )
