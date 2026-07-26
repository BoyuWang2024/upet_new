"""Ridge spectra, candidates, Cholesky factors, and quadratic forms."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class Spectrum:
    matrix: torch.Tensor
    eigenvalues: torch.Tensor
    mu_min: float
    mu_max: float


@dataclass(frozen=True)
class RidgeCandidate:
    target: str
    eta: float
    condition_number: float
    condition_warning: bool
    cholesky: torch.Tensor


def prepare_spectrum(matrix: torch.Tensor) -> Spectrum:
    matrix = matrix.detach().to(dtype=torch.float64, device="cpu")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ridge matrix must be square")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError("ridge matrix contains non-finite values")
    symmetry_error = torch.linalg.matrix_norm(matrix - matrix.T)
    tolerance = (
        torch.finfo(torch.float64).eps
        * max(1.0, float(torch.linalg.matrix_norm(matrix)))
        * matrix.shape[0]
    )
    if float(symmetry_error) > tolerance:
        raise ValueError(
            f"ridge matrix is not symmetric: error={float(symmetry_error)}"
        )
    symmetric = (matrix + matrix.T) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    return Spectrum(
        matrix=symmetric,
        eigenvalues=eigenvalues,
        mu_min=float(eigenvalues[0]),
        mu_max=float(eigenvalues[-1]),
    )


def _condition_number(spectrum: Spectrum, eta: float) -> float:
    denominator = spectrum.mu_min + eta
    numerator = spectrum.mu_max + eta
    if denominator <= 0:
        return float("inf")
    return numerator / denominator


def _factor_candidate(
    target: str,
    eta: float,
    spectrum: Spectrum,
    max_condition_number: float,
    *,
    allow_condition_warning: bool,
) -> RidgeCandidate:
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError(f"{target} eta must be finite and positive")
    condition_number = _condition_number(spectrum, eta)
    condition_warning = condition_number > max_condition_number
    if condition_warning and not allow_condition_warning:
        raise ValueError(
            f"{target} eta {eta} exceeds condition limit: {condition_number}"
        )
    regularized = spectrum.matrix + eta * torch.eye(
        spectrum.matrix.shape[0], dtype=torch.float64
    )
    try:
        cholesky = torch.linalg.cholesky(regularized)
    except RuntimeError as error:
        raise ValueError(f"{target} Cholesky failed for eta={eta}") from error
    return RidgeCandidate(
        target=target,
        eta=float(eta),
        condition_number=float(condition_number),
        condition_warning=condition_warning,
        cholesky=cholesky,
    )


def fixed_candidate(
    target: str,
    eta: float,
    spectrum: Spectrum,
    max_condition_number: float,
) -> RidgeCandidate:
    """Factor exactly the requested eta, recording rather than fixing condition."""
    return _factor_candidate(
        target,
        eta,
        spectrum,
        max_condition_number,
        allow_condition_warning=True,
    )


def fit_candidates(
    target: str,
    spectrum: Spectrum,
    multipliers: Sequence[float],
    max_condition_number: float,
) -> tuple[RidgeCandidate, ...]:
    if not multipliers or any(value <= 0 for value in multipliers):
        raise ValueError("candidate multipliers must be non-empty and positive")
    spectral_eps = torch.finfo(torch.float64).eps * max(1.0, abs(spectrum.mu_max))
    eta_pd = max(0.0, -spectrum.mu_min) + spectral_eps
    eta_condition = max(
        0.0,
        (spectrum.mu_max - max_condition_number * spectrum.mu_min)
        / (max_condition_number - 1.0),
    )
    eta_floor = float(np.nextafter(max(eta_pd, eta_condition), np.inf))
    etas = sorted({float(eta_floor * value) for value in multipliers})
    return tuple(
        _factor_candidate(
            target,
            eta,
            spectrum,
            max_condition_number,
            allow_condition_warning=False,
        )
        for eta in etas
    )


def quadratic_forms(cholesky: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    gradients = gradients.detach().to(dtype=torch.float64, device="cpu")
    if gradients.ndim == 1:
        gradients = gradients.reshape(1, -1)
    solved = torch.cholesky_solve(gradients.T, cholesky).T
    q = torch.sum(gradients * solved, dim=1)
    if not bool(torch.isfinite(q).all()) or bool((q <= 0).any()):
        raise ValueError("quadratic forms must be finite and strictly positive")
    return q
