"""Read-only diagnostics for canonical LLPR evaluation artifacts."""

from __future__ import annotations

import csv
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import yaml
from pydantic import Field, model_validator
from scipy.stats import pearsonr, spearmanr

from .artifacts import (
    atomic_json_dump,
    load_verified_manifest,
    sha256_file,
    stage_identity,
)
from .config import StrictModel, resolve_repo_path


class PlotConfig(StrictModel):
    """Configuration for plotting one completed evaluation."""

    run_root: Path
    output_root: Path
    bin_count: int = Field(default=20, gt=0)
    sample_size: int = Field(default=200_000, gt=0)
    seed: int = 2026
    evaluation_identity: str | None = None

    @model_validator(mode="after")
    def resolve_paths(self) -> "PlotConfig":
        run_root = (
            self.run_root.resolve()
            if self.run_root.is_absolute()
            else resolve_repo_path(self.run_root)
        )
        output_root = (
            self.output_root.resolve()
            if self.output_root.is_absolute()
            else resolve_repo_path(self.output_root)
        )
        if output_root == run_root or run_root in output_root.parents:
            raise ValueError("output_root must be outside run_root")
        object.__setattr__(self, "run_root", run_root)
        object.__setattr__(self, "output_root", output_root)
        return self


def load_plot_config(path: Path) -> PlotConfig:
    """Load and strictly validate plotting configuration."""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("plot configuration must contain a YAML mapping")
    return PlotConfig.model_validate(loaded)


@dataclass(frozen=True)
class FilteredPairs:
    uncertainty: np.ndarray
    absolute_error: np.ndarray
    original_count: int
    dropped_count: int


@dataclass(frozen=True)
class PanelAnalysis:
    count: int
    dropped_count: int
    pearson_log10: float
    spearman_log10: float
    uncertainty_median: float
    absolute_error_median: float


@dataclass(frozen=True)
class ReliabilityBin:
    count: int
    mean_predicted_std: float
    rmse: float
    lower_predicted_std: float
    upper_predicted_std: float


def filter_log_pairs(
    uncertainty: np.ndarray, absolute_error: np.ndarray
) -> FilteredPairs:
    """Keep finite, strictly positive pairs suitable for logarithmic plots."""
    uncertainty = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    absolute_error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    if uncertainty.shape != absolute_error.shape:
        raise ValueError("uncertainty and absolute_error shapes must match")
    keep = (
        np.isfinite(uncertainty)
        & np.isfinite(absolute_error)
        & (uncertainty > 0)
        & (absolute_error > 0)
    )
    return FilteredPairs(
        uncertainty=uncertainty[keep],
        absolute_error=absolute_error[keep],
        original_count=len(uncertainty),
        dropped_count=int(np.count_nonzero(~keep)),
    )


def analyze_panel(uncertainty: np.ndarray, absolute_error: np.ndarray) -> PanelAnalysis:
    """Calculate deterministic diagnostics on valid log-domain pairs."""
    filtered = filter_log_pairs(uncertainty, absolute_error)
    if len(filtered.uncertainty) < 2:
        raise ValueError("at least two valid log-domain pairs are required")
    log_uncertainty = np.log10(filtered.uncertainty)
    log_error = np.log10(filtered.absolute_error)
    return PanelAnalysis(
        count=len(filtered.uncertainty),
        dropped_count=filtered.dropped_count,
        pearson_log10=float(pearsonr(log_uncertainty, log_error).statistic),
        spearman_log10=float(spearmanr(log_uncertainty, log_error).statistic),
        uncertainty_median=float(np.median(filtered.uncertainty)),
        absolute_error_median=float(np.median(filtered.absolute_error)),
    )


def _valid_std_residual(
    predicted_std: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    predicted_std = np.asarray(predicted_std, dtype=np.float64).reshape(-1)
    residual = np.asarray(residual, dtype=np.float64).reshape(-1)
    if predicted_std.shape != residual.shape:
        raise ValueError("predicted_std and residual shapes must match")
    keep = np.isfinite(predicted_std) & np.isfinite(residual) & (predicted_std > 0)
    if not np.any(keep):
        raise ValueError(
            "no finite residuals with positive predicted standard deviation"
        )
    return predicted_std[keep], residual[keep]


def reliability_bins(
    predicted_std: np.ndarray, residual: np.ndarray, *, bin_count: int
) -> tuple[ReliabilityBin, ...]:
    """Bin samples by predicted standard deviation and compare against RMSE."""
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")
    std, errors = _valid_std_residual(predicted_std, residual)
    order = np.argsort(std, kind="stable")
    groups = np.array_split(order, min(bin_count, len(order)))
    return tuple(
        ReliabilityBin(
            count=len(group),
            mean_predicted_std=float(np.mean(std[group])),
            rmse=float(np.sqrt(np.mean(np.square(errors[group])))),
            lower_predicted_std=float(np.min(std[group])),
            upper_predicted_std=float(np.max(std[group])),
        )
        for group in groups
        if len(group)
    )


def standardized_residual_cdf(
    predicted_std: np.ndarray, residual: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted absolute standardized residuals and their empirical CDF."""
    std, errors = _valid_std_residual(predicted_std, residual)
    standardized = np.sort(np.abs(errors) / std)
    empirical = np.arange(1, len(standardized) + 1, dtype=np.float64) / len(
        standardized
    )
    return standardized, empirical


def _find_evaluation(config: PlotConfig) -> tuple[Path, dict[str, object]]:
    candidates: list[tuple[Path, dict[str, object]]] = []
    for path in sorted((config.run_root / "evaluation").glob("*/*/manifest.json")):
        manifest = load_verified_manifest(path, verify_npz=True)
        identity = str(manifest.get("identity", ""))
        if config.evaluation_identity is None or identity == config.evaluation_identity:
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one matching complete evaluation, "
            f"found {len(candidates)}"
        )
    return candidates[0]


def _sample_indices(count: int, sample_size: int, seed: int) -> np.ndarray:
    if count <= sample_size:
        return np.arange(count)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(count, size=sample_size, replace=False))


def _scatter(
    path_base: Path,
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    *,
    title: str,
    seed: int,
    sample_size: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filtered = filter_log_pairs(uncertainty, absolute_error)
    indices = _sample_indices(len(filtered.uncertainty), sample_size, seed)
    figure, axis = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    axis.scatter(
        filtered.uncertainty[indices],
        filtered.absolute_error[indices],
        s=7,
        alpha=0.35,
        linewidths=0,
    )
    axis.set(xscale="log", yscale="log", xlabel="Predicted standard deviation")
    axis.set_ylabel("Absolute error")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        figure.savefig(path_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _reliability_figure(
    path_base: Path,
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    bin_count: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for axis, (name, (std, residual)) in zip(axes, panels.items(), strict=True):
        bins = reliability_bins(std, residual, bin_count=bin_count)
        predicted = np.array([item.mean_predicted_std for item in bins])
        observed = np.array([item.rmse for item in bins])
        limit = float(max(np.max(predicted), np.max(observed)))
        axis.plot([0, limit], [0, limit], "--", color="0.4", label="ideal")
        axis.plot(predicted, observed, "o-", label=name)
        axis.set(
            xlabel="Mean predicted standard deviation",
            ylabel="Observed RMSE",
            title=name,
        )
        axis.grid(alpha=0.2)
        axis.legend()
    for suffix in ("png", "pdf"):
        figure.savefig(path_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _cdf_figure(
    path_base: Path, panels: dict[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.special import erf

    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), constrained_layout=True)
    for axis, (name, (std, residual)) in zip(axes, panels.items(), strict=True):
        standardized, empirical = standardized_residual_cdf(std, residual)
        axis.plot(standardized, empirical, label="empirical")
        reference_x = np.linspace(0, max(4.0, float(standardized[-1])), 400)
        axis.plot(reference_x, erf(reference_x / np.sqrt(2)), "--", label="|N(0,1)|")
        axis.set(
            xlabel="Absolute standardized residual",
            ylabel="Cumulative probability",
            title=name,
            xlim=(0, float(reference_x[-1])),
            ylim=(0, 1),
        )
        axis.grid(alpha=0.2)
        axis.legend()
    for suffix in ("png", "pdf"):
        figure.savefig(path_base.with_suffix(f".{suffix}"), dpi=220)
    plt.close(figure)


def _write_statistics(
    path: Path,
    analyses: dict[str, PanelAnalysis],
    panels: dict[str, tuple[np.ndarray, np.ndarray]],
    bin_count: int,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "panel",
                "count",
                "dropped_count",
                "pearson_log10",
                "spearman_log10",
                "uncertainty_median",
                "absolute_error_median",
                "standardized_abs_median",
                "reliability_bin_count",
            ],
        )
        writer.writeheader()
        for name in ("energy", "force"):
            std, residual = panels[name]
            standardized, _ = standardized_residual_cdf(std, residual)
            row = asdict(analyses[name])
            writer.writerow(
                {
                    "panel": name,
                    **row,
                    "standardized_abs_median": float(np.median(standardized)),
                    "reliability_bin_count": len(
                        reliability_bins(std, residual, bin_count=bin_count)
                    ),
                }
            )


def run_plot(config: PlotConfig) -> Path:
    """Publish figures derived from a complete evaluation without modifying it."""
    evaluation_dir, evaluation_manifest = _find_evaluation(config)
    identity = stage_identity(
        "plot",
        {
            "evaluation_identity": evaluation_manifest["identity"],
            "bin_count": config.bin_count,
            "sample_size": config.sample_size,
            "seed": config.seed,
        },
    )
    plot_id = str(identity["identity"])
    destination = config.output_root / plot_id
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        load_verified_manifest(manifest_path, {"identity": plot_id})
        return destination

    staging = destination.parent / f".{plot_id}.{uuid.uuid4().hex}.staging"
    staging.mkdir(parents=True)
    try:
        with np.load(evaluation_dir / "details.npz", allow_pickle=False) as archive:
            panels = {
                "energy": (
                    archive["energy_calibrated_std"].copy(),
                    archive["energy_residual"].copy(),
                ),
                "force": (
                    archive["force_calibrated_std_component"].copy(),
                    archive["force_residual"].copy(),
                ),
            }
        analyses = {
            name: analyze_panel(std, np.abs(residual))
            for name, (std, residual) in panels.items()
        }
        _scatter(
            staging / "energy_uncertainty_vs_error",
            panels["energy"][0],
            np.abs(panels["energy"][1]),
            title="Energy uncertainty vs absolute error",
            seed=config.seed,
            sample_size=config.sample_size,
        )
        _scatter(
            staging / "force_uncertainty_vs_error",
            panels["force"][0],
            np.abs(panels["force"][1]),
            title="Force uncertainty vs absolute error",
            seed=config.seed + 1,
            sample_size=config.sample_size,
        )
        _reliability_figure(staging / "reliability", panels, config.bin_count)
        _cdf_figure(staging / "standardized_residual", panels)
        _write_statistics(
            staging / "statistics.csv", analyses, panels, config.bin_count
        )
        names = sorted(path.name for path in staging.iterdir() if path.is_file())
        files = {name: sha256_file(staging / name) for name in names}
        atomic_json_dump(
            staging / "manifest.json",
            {
                **identity,
                "status": "complete",
                "evaluation_identity": evaluation_manifest["identity"],
                "statistics": {
                    name: asdict(analysis) for name, analysis in analyses.items()
                },
                "files": files,
            },
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
