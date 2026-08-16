"""Deterministic single-dataset plot analysis adapted from carnet FGE plots.

Only the raw uncertainty-versus-absolute-residual panels are retained.  The
checkpoint-count sweep from the reference project is deliberately excluded.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import cast

import numpy as np
import torch
import yaml

from .artifacts import sha256_file
from .errors import HardFailure
from .inference_validation import validate_inference_result
from .uncertainty import population_std, tensor_to_voigt_symmetric
from .validation import validate_completed_result


_DOMAINS = ("energy", "force", "stress")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LABEL = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class PlotSettings:
    dpi: int = 300
    figure_size: tuple[float, float] = (7.0, 7.0)
    scatter_max_points: int = 20_000
    scatter_seed: int = 20260714
    scatter_size: float = 2.0
    scatter_alpha: float = 0.035
    grid_size: int = 160
    gaussian_sigma: float = 1.2
    contour_masses: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 0.99)
    log_margin: float = 0.05

    def __post_init__(self) -> None:
        if type(self.dpi) is not int or self.dpi <= 0:
            raise HardFailure("plot dpi must be a positive integer")
        if len(self.figure_size) != 2 or any(
            not math.isfinite(value) or value <= 0 for value in self.figure_size
        ):
            raise HardFailure("plot figure_size must contain two positive values")
        if type(self.scatter_max_points) is not int or self.scatter_max_points <= 0:
            raise HardFailure("scatter_max_points must be a positive integer")
        if type(self.scatter_seed) is not int:
            raise HardFailure("scatter_seed must be an integer")
        if not 0 < self.scatter_alpha <= 1 or self.scatter_size <= 0:
            raise HardFailure("plot scatter style is invalid")
        if type(self.grid_size) is not int or self.grid_size < 8:
            raise HardFailure("plot grid_size must be at least 8")
        if self.gaussian_sigma <= 0 or self.log_margin <= 0:
            raise HardFailure("plot density settings must be positive")
        if (
            len(self.contour_masses) != 5
            or any(not 0 < value < 1 for value in self.contour_masses)
            or any(left >= right for left, right in pairwise(self.contour_masses))
        ):
            raise HardFailure("plot contour masses must be five increasing fractions")


@dataclass(frozen=True)
class PlotPanelInput:
    domain: str
    uncertainty: torch.Tensor
    absolute_residual: torch.Tensor

    def __post_init__(self) -> None:
        if self.domain not in _DOMAINS:
            raise HardFailure(f"unsupported plot domain: {self.domain}")
        if not isinstance(self.uncertainty, torch.Tensor) or not isinstance(
            self.absolute_residual, torch.Tensor
        ):
            raise HardFailure("plot arrays must be torch tensors")
        if self.uncertainty.shape != self.absolute_residual.shape:
            raise HardFailure("plot uncertainty and residual shapes differ")
        if self.uncertainty.numel() == 0:
            raise HardFailure("plot arrays must not be empty")
        object.__setattr__(
            self, "uncertainty", self.uncertainty.detach().cpu().reshape(-1).clone()
        )
        object.__setattr__(
            self,
            "absolute_residual",
            self.absolute_residual.detach().cpu().reshape(-1).clone(),
        )


@dataclass(frozen=True)
class PlotInput:
    dataset_label: str
    source_kind: str
    source_identity: str
    panels: tuple[PlotPanelInput, ...]

    def __post_init__(self) -> None:
        if _LABEL.fullmatch(self.dataset_label) is None:
            raise HardFailure("plot dataset label is invalid")
        if self.source_kind not in {"completed_fge", "inference"}:
            raise HardFailure("plot source kind is invalid")
        if _SHA256.fullmatch(self.source_identity) is None:
            raise HardFailure("plot source identity is invalid")
        domains = tuple(panel.domain for panel in self.panels)
        if not domains or len(domains) != len(set(domains)):
            raise HardFailure("plot domains must be nonempty and unique")
        if domains != tuple(domain for domain in _DOMAINS if domain in domains):
            raise HardFailure("plot domains are not in canonical order")


@dataclass(frozen=True)
class DensityAnalysis:
    x_centers: np.ndarray
    y_centers: np.ndarray
    grid: np.ndarray
    contour_levels: tuple[float, ...]
    actual_contour_masses: tuple[float, ...]
    histogram_count: int


@dataclass(frozen=True)
class PanelAnalysis:
    domain: str
    uncertainty: torch.Tensor
    absolute_residual: torch.Tensor
    original_count: int
    valid_count: int
    excluded: Mapping[str, int]
    spearman: float | None
    pearson_log10: float | None
    scatter_indices: torch.Tensor
    density: DensityAnalysis
    log_limits: tuple[float, float]


@dataclass(frozen=True)
class PlotAnalysisResult:
    plot_input: PlotInput
    settings: PlotSettings
    panels: tuple[PanelAnalysis, ...]

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(panel.domain for panel in self.panels)


@dataclass(frozen=True)
class PlotConfig:
    input_root: Path
    input_kind: str
    output_root: Path
    settings: PlotSettings


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise HardFailure(f"{label} must be a string-keyed mapping")
    return cast(Mapping[str, object], value)


def _strict_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        detail = unknown[0] if unknown else missing[0]
        raise HardFailure(f"unknown or missing key in {label}: {detail}")


def _path(value: object, parent: Path, label: str) -> Path:
    if type(value) is not str:
        raise HardFailure(f"{label} must be a path string")
    path = Path(cast(str, value))
    return (path if path.is_absolute() else parent / path).resolve()


def load_plot_config(path: str | Path) -> PlotConfig:
    """Load a strict plot-only configuration with no sweep fields."""
    source = Path(path).resolve()
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HardFailure(f"unable to load plot configuration: {exc}") from exc
    root = _mapping(payload, "plot configuration")
    _strict_keys(root, {"schema_version", "input", "output", "plot"}, "root")
    if root["schema_version"] != "upet.fge.plot.v1":
        raise HardFailure("plot schema_version is invalid")
    input_section = _mapping(root["input"], "input")
    output_section = _mapping(root["output"], "output")
    plot_section = _mapping(root["plot"], "plot")
    _strict_keys(input_section, {"root", "kind"}, "input")
    _strict_keys(output_section, {"root"}, "output")
    allowed = {
        "dpi",
        "figure_size",
        "scatter_max_points",
        "scatter_seed",
        "scatter_size",
        "scatter_alpha",
        "grid_size",
        "gaussian_sigma",
        "contour_masses",
        "log_margin",
    }
    if not set(plot_section) <= allowed:
        raise HardFailure(f"unknown plot key: {sorted(set(plot_section) - allowed)[0]}")
    kind = input_section["kind"]
    if kind not in {"completed_fge", "inference"}:
        raise HardFailure("plot input kind is invalid")
    settings_payload = dict(plot_section)
    for key in ("figure_size", "contour_masses"):
        if key in settings_payload:
            value = settings_payload[key]
            if not isinstance(value, list):
                raise HardFailure(f"plot {key} must be a list")
            settings_payload[key] = tuple(value)
    try:
        settings = PlotSettings(**settings_payload)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise HardFailure(f"plot settings are invalid: {exc}") from exc
    return PlotConfig(
        input_root=_path(input_section["root"], source.parent, "input.root"),
        input_kind=cast(str, kind),
        output_root=_path(output_section["root"], source.parent, "output.root"),
        settings=settings,
    )


def _ranks(values: torch.Tensor) -> torch.Tensor:
    order = sorted(
        range(values.numel()), key=lambda index: (float(values[index]), index)
    )
    ranks = torch.empty(values.numel(), dtype=torch.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(
            values[order[start]]
        ):
            end += 1
        ranks[order[start:end]] = ((start + 1) + end) / 2.0
        start = end
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.to(torch.float64) - left.to(torch.float64).mean()
    right = right.to(torch.float64) - right.to(torch.float64).mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _density(
    log_uncertainty: torch.Tensor,
    log_residual: torch.Tensor,
    limits: tuple[float, float],
    settings: PlotSettings,
) -> DensityAnalysis:
    from scipy.ndimage import gaussian_filter

    low, high = limits
    histogram, x_edges, y_edges = np.histogram2d(
        log_uncertainty.numpy(),
        log_residual.numpy(),
        bins=settings.grid_size,
        range=((low, high), (low, high)),
    )
    count = int(histogram.sum())
    if count < 2:
        raise HardFailure("plot density requires at least two points")
    grid = gaussian_filter(
        histogram, sigma=settings.gaussian_sigma, mode="nearest"
    ).astype(np.float64, copy=False)
    grid /= grid.sum()
    descending = np.sort(grid.reshape(-1))[::-1]
    cumulative = np.cumsum(descending)
    thresholds = [
        float(
            descending[min(int(np.searchsorted(cumulative, mass)), len(descending) - 1)]
        )
        for mass in settings.contour_masses
    ]
    levels = tuple(float(value) for value in np.unique(thresholds) if value > 0)
    if len(levels) != len(settings.contour_masses):
        raise HardFailure("plot density contour levels are degenerate")
    return DensityAnalysis(
        x_centers=(x_edges[:-1] + x_edges[1:]) * 0.5,
        y_centers=(y_edges[:-1] + y_edges[1:]) * 0.5,
        grid=grid,
        contour_levels=levels,
        actual_contour_masses=tuple(
            float(grid[grid >= level].sum()) for level in levels
        ),
        histogram_count=count,
    )


def _analyze_panel(panel: PlotPanelInput, settings: PlotSettings) -> PanelAnalysis:
    uncertainty = panel.uncertainty.to(torch.float64)
    residual = panel.absolute_residual.to(torch.float64)
    nan = torch.isnan(uncertainty) | torch.isnan(residual)
    inf = ~nan & (torch.isinf(uncertainty) | torch.isinf(residual))
    finite = ~(nan | inf)
    nonpositive = finite & ((uncertainty <= 0) | (residual <= 0))
    valid = finite & ~nonpositive
    uncertainty = uncertainty[valid]
    residual = residual[valid]
    if uncertainty.numel() < 2:
        raise HardFailure(f"{panel.domain} plot has fewer than two valid pairs")
    log_u = torch.log10(uncertainty)
    log_r = torch.log10(residual)
    low = min(float(log_u.min()), float(log_r.min()))
    high = max(float(log_u.max()), float(log_r.max()))
    padding = max((high - low) * settings.log_margin, settings.log_margin)
    limits = (low - padding, high + padding)
    if uncertainty.numel() <= settings.scatter_max_points:
        indices = torch.arange(uncertainty.numel(), dtype=torch.int64)
    else:
        generator = np.random.default_rng(settings.scatter_seed)
        selected = np.sort(
            generator.choice(
                uncertainty.numel(), settings.scatter_max_points, replace=False
            )
        )
        indices = torch.from_numpy(selected.astype(np.int64, copy=False))
    return PanelAnalysis(
        domain=panel.domain,
        uncertainty=uncertainty,
        absolute_residual=residual,
        original_count=panel.uncertainty.numel(),
        valid_count=uncertainty.numel(),
        excluded={
            "nan": int(nan.sum()),
            "inf": int(inf.sum()),
            "nonpositive": int(nonpositive.sum()),
        },
        spearman=_pearson(_ranks(uncertainty), _ranks(residual)),
        pearson_log10=_pearson(log_u, log_r),
        scatter_indices=indices,
        density=_density(log_u, log_r, limits, settings),
        log_limits=limits,
    )


def analyze_plot_input(input: PlotInput, settings: PlotSettings) -> PlotAnalysisResult:
    """Analyze each physical domain independently without recomputing UQ."""
    return PlotAnalysisResult(
        plot_input=input,
        settings=settings,
        panels=tuple(_analyze_panel(panel, settings) for panel in input.panels),
    )


def _load_torch(path: Path) -> Mapping[str, object]:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure(f"unable to load plot tensor artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise HardFailure(f"plot tensor artifact is not a mapping: {path}")
    return cast(Mapping[str, object], value)


def _tensor(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    value = mapping.get(key)
    if not isinstance(value, torch.Tensor):
        raise HardFailure(f"plot source tensor is missing: {key}")
    return value


def load_completed_fge_plot_input(root: str | Path) -> PlotInput:
    """Reopen the completed MATPES-test result and expose its three plot domains."""
    result_root = Path(root)
    validate_completed_result(result_root)
    prediction = _load_torch(result_root / "prediction" / "test_raw.pt")
    ensemble = _load_torch(
        result_root / "evaluation" / "legacy_equal_weight" / "ensemble.pt"
    )
    uncertainty = _load_torch(
        result_root / "evaluation" / "legacy_equal_weight" / "uncertainty.pt"
    )
    n_atoms = _tensor(prediction, "n_atoms")
    energy_reference = _tensor(prediction, "energy_reference")
    force_reference = _tensor(prediction, "forces_reference")
    stress_reference = _tensor(prediction, "stress_reference")
    energy_uq = cast(Mapping[str, object], uncertainty["energy_per_atom"])
    force_uq = cast(Mapping[str, object], uncertainty["force_component"])
    energy_residual = torch.abs(
        _tensor(ensemble, "energy") / n_atoms - energy_reference / n_atoms
    )
    force_residual = torch.abs(_tensor(ensemble, "forces") - force_reference).reshape(
        -1
    )
    stress_members = tensor_to_voigt_symmetric(_tensor(prediction, "stress_prediction"))
    stress_residual = torch.abs(
        tensor_to_voigt_symmetric(_tensor(ensemble, "stress"))
        - tensor_to_voigt_symmetric(stress_reference)
    ).reshape(-1)
    return PlotInput(
        dataset_label="matpes_test",
        source_kind="completed_fge",
        source_identity=sha256_file(result_root / "result_manifest.json"),
        panels=(
            PlotPanelInput("energy", _tensor(energy_uq, "std"), energy_residual),
            PlotPanelInput(
                "force", _tensor(force_uq, "std").reshape(-1), force_residual
            ),
            PlotPanelInput(
                "stress",
                population_std(stress_members).reshape(-1),
                stress_residual,
            ),
        ),
    )


def load_inference_plot_input(root: str | Path) -> PlotInput:
    """Reopen a completed chunked inference result and concatenate plot vectors."""
    result_root = Path(root)
    validate_inference_result(result_root)
    run_path = result_root / "run_manifest.json"
    try:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        manifest = json.loads(
            (result_root / "uncertainty" / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise HardFailure("unable to load inference plot manifests") from exc
    arrays: dict[str, list[torch.Tensor]] = {}
    for entry in manifest["chunks"]:
        payload = _load_torch(result_root / entry["path"])
        for domain, prefix in (
            ("energy", "energy_per_atom"),
            ("force", "force_component"),
            ("stress", "stress_component"),
        ):
            residual_key = f"{prefix}_absolute_residual"
            if residual_key in payload:
                arrays.setdefault(f"{domain}_uncertainty", []).append(
                    _tensor(payload, f"{prefix}_std")
                )
                arrays.setdefault(f"{domain}_residual", []).append(
                    _tensor(payload, residual_key)
                )
    panels = tuple(
        PlotPanelInput(
            domain,
            torch.cat(arrays[f"{domain}_uncertainty"]),
            torch.cat(arrays[f"{domain}_residual"]),
        )
        for domain in _DOMAINS
        if arrays.get(f"{domain}_uncertainty")
    )
    dataset = run.get("dataset_identity")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("label"), str):
        raise HardFailure("inference plot dataset identity is invalid")
    return PlotInput(
        dataset_label=cast(str, dataset["label"]),
        source_kind="inference",
        source_identity=sha256_file(run_path),
        panels=panels,
    )


__all__ = [
    "DensityAnalysis",
    "PanelAnalysis",
    "PlotAnalysisResult",
    "PlotConfig",
    "PlotInput",
    "PlotPanelInput",
    "PlotSettings",
    "analyze_plot_input",
    "load_completed_fge_plot_input",
    "load_inference_plot_input",
    "load_plot_config",
]
