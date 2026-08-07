"""Validated analysis and plotting inputs for completed confidence-head runs."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import torch
import yaml

from .artifacts import load_verified_torch
from .metrics import _correlation, _ranks
from .workflows.verify import verify_run


Target = Literal["energy", "force"]
_EXPECTED_NUM_BINS = 50
_FIXED_LINEAR = "fixed_linear_v1"
_ATOM_MEAN = "atom_mean"
_ATOM_MEAN_ERROR = "abs_cartesian_component_mean_v1"


@dataclass(frozen=True)
class PlotSeries:
    """One verified single-target evaluation compacted for plotting."""

    run_dir: Path
    structure_ids: torch.Tensor
    target: Target
    order: int | None
    logits: torch.Tensor
    observed: torch.Tensor
    expected: torch.Tensor
    representatives: torch.Tensor
    stored_metrics: Mapping[str, int | float]
    force_target_mode: str | None


@dataclass(frozen=True)
class BinRow:
    """Observed-error statistics for one predicted argmax bin."""

    bin: int
    sample_count: int
    mean_observed_error: float | None
    median_observed_error: float | None
    std_observed_error: float | None
    q25_observed_error: float | None
    q75_observed_error: float | None
    min_observed_error: float | None
    max_observed_error: float | None
    mean_expected_error: float | None


@dataclass(frozen=True)
class CorrelationRow:
    """Expected-versus-observed energy correlations for one order."""

    order: int
    sample_count: int
    pearson: float
    spearman: float


def _mapping(path: Path, *, yaml_file: bool = False) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_file else json.loads(text)
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"invalid plot input {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"plot input {path} must contain a mapping")
    return value


def _active_target(config: Mapping[str, Any]) -> Target:
    model = config.get("model")
    loss = config.get("loss")
    if not isinstance(model, Mapping) or not isinstance(loss, Mapping):
        raise ValueError("resolved config is missing model or loss settings")
    force = model.get("force")
    energy = model.get("energy")
    if not isinstance(force, Mapping) or not isinstance(energy, Mapping):
        raise ValueError("resolved config is missing confidence branches")
    force_active = (
        bool(force.get("enabled")) and float(loss.get("force_coefficient", 0.0)) > 0.0
    )
    energy_active = (
        bool(energy.get("enabled")) and float(loss.get("energy_coefficient", 0.0)) > 0.0
    )
    if force_active == energy_active:
        raise ValueError("plotting requires exactly one active target")
    return "force" if force_active else "energy"


def _tensor(
    predictions: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if name not in predictions:
        raise ValueError(f"prediction artifact is missing {name}")
    result = torch.as_tensor(predictions[name]).detach().cpu()
    return result.to(dtype=dtype) if dtype is not None else result


def load_plot_series(
    run_dir: Path,
    *,
    expected_num_bins: int = _EXPECTED_NUM_BINS,
) -> PlotSeries:
    """Load a fully verified single-target evaluation without changing it."""

    if expected_num_bins != _EXPECTED_NUM_BINS:
        raise ValueError("UPET result plots require exactly 50 bins")
    root = Path(run_dir).resolve()
    verify_run(root, full=True)
    config = _mapping(root / "resolved_config.yaml", yaml_file=True)
    bins = _mapping(root / "binning.json")
    metrics = _mapping(root / "evaluation" / "metrics.json")
    loaded = load_verified_torch(
        root / "evaluation" / "test_predictions.pt",
        weights_only=False,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError("test_predictions.pt must contain a mapping")
    predictions = cast(Mapping[str, Any], loaded)

    target = _active_target(config)
    model = cast(Mapping[str, Any], config["model"])
    branch_config = model.get(target)
    branch_bins = bins.get(target)
    branch_metrics = metrics.get(target)
    if not isinstance(branch_config, Mapping):
        raise ValueError(f"resolved config is missing {target} model settings")
    if not isinstance(branch_bins, Mapping):
        raise ValueError(f"binning artifact is missing {target} settings")
    if not isinstance(branch_metrics, Mapping):
        raise ValueError(f"metrics artifact is missing {target} settings")
    if config.get("binning", {}).get("algorithm") != _FIXED_LINEAR:
        raise ValueError("plotting requires fixed_linear_v1 binning")
    if branch_bins.get("algorithm") != _FIXED_LINEAR:
        raise ValueError(f"{target} binning must use fixed_linear_v1")
    if (
        int(branch_config.get("num_bins", 0)) != expected_num_bins
        or int(branch_bins.get("num_bins", 0)) != expected_num_bins
    ):
        raise ValueError(f"{target} plotting requires exactly 50 bins")

    logits = _tensor(predictions, f"{target}_logits", dtype=torch.float64)
    observed = _tensor(
        predictions,
        f"{target}_observed_errors",
        dtype=torch.float64,
    ).reshape(-1)
    expected = _tensor(
        predictions,
        f"{target}_expected_errors",
        dtype=torch.float64,
    ).reshape(-1)
    representatives = _tensor(
        predictions,
        f"{target}_representatives",
        dtype=torch.float64,
    ).reshape(-1)
    structure_ids = _tensor(predictions, "structure_ids").reshape(-1)

    if logits.ndim != 2 or logits.shape != (len(observed), expected_num_bins):
        raise ValueError(f"{target} logits must have shape [N, 50]")
    if expected.shape != observed.shape:
        raise ValueError(f"{target} expected and observed errors must have matching N")
    if representatives.numel() != expected_num_bins:
        raise ValueError(f"{target} representatives must contain 50 values")
    declared_representatives = torch.as_tensor(
        branch_bins.get("representatives"), dtype=torch.float64
    ).reshape(-1)
    if declared_representatives.shape != representatives.shape or not torch.allclose(
        declared_representatives,
        representatives,
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError(f"{target} representatives disagree with binning.json")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError(f"{target} logits must be finite")
    if not bool(torch.isfinite(observed).all() and torch.isfinite(expected).all()):
        raise ValueError(f"{target} errors must be finite")
    if bool(torch.any(observed < 0) or torch.any(expected < 0)):
        raise ValueError(f"{target} errors must be non-negative")
    if int(branch_metrics.get("sample_count", -1)) != len(observed):
        raise ValueError(f"{target} metric sample_count disagrees with predictions")

    force_target_mode: str | None = None
    order: int | None = None
    if target == "force":
        force_target_mode = str(predictions.get("force_target_mode"))
        semantics = (
            branch_config.get("target_mode"),
            branch_bins.get("target_mode"),
            force_target_mode,
            branch_bins.get("error_definition"),
            predictions.get("force_error_definition"),
        )
        if semantics != (
            _ATOM_MEAN,
            _ATOM_MEAN,
            _ATOM_MEAN,
            _ATOM_MEAN_ERROR,
            _ATOM_MEAN_ERROR,
        ):
            raise ValueError("force plots require atom_mean error semantics")
    else:
        order = int(branch_config.get("cumulant_order", 0))
        if order not in range(1, 9):
            raise ValueError("energy cumulant order must be between 1 and 8")
        if len(structure_ids) != len(observed):
            raise ValueError("energy structure_ids must match observation count")

    return PlotSeries(
        run_dir=root,
        structure_ids=structure_ids,
        target=target,
        order=order,
        logits=logits,
        observed=observed,
        expected=expected,
        representatives=representatives,
        stored_metrics=cast(Mapping[str, int | float], branch_metrics),
        force_target_mode=force_target_mode,
    )


def bin_rows(series: PlotSeries) -> tuple[BinRow, ...]:
    """Return statistics for every predicted bin, including empty bins."""

    predicted_bins = series.logits.argmax(dim=-1)
    rows: list[BinRow] = []
    quantiles = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64)
    for index in range(series.logits.shape[1]):
        mask = predicted_bins == index
        values = series.observed[mask]
        expected = series.expected[mask]
        if values.numel() == 0:
            rows.append(
                BinRow(
                    bin=index,
                    sample_count=0,
                    mean_observed_error=None,
                    median_observed_error=None,
                    std_observed_error=None,
                    q25_observed_error=None,
                    q75_observed_error=None,
                    min_observed_error=None,
                    max_observed_error=None,
                    mean_expected_error=None,
                )
            )
            continue
        q25, median, q75 = torch.quantile(values, quantiles)
        rows.append(
            BinRow(
                bin=index,
                sample_count=len(values),
                mean_observed_error=float(values.mean()),
                median_observed_error=float(median),
                std_observed_error=float(values.std(unbiased=False)),
                q25_observed_error=float(q25),
                q75_observed_error=float(q75),
                min_observed_error=float(values.min()),
                max_observed_error=float(values.max()),
                mean_expected_error=float(expected.mean()),
            )
        )
    return tuple(rows)


def energy_correlation(
    series: PlotSeries,
    *,
    relative_tolerance: float = 1e-6,
    absolute_tolerance: float = 1e-7,
) -> CorrelationRow:
    """Recompute and verify expected-versus-observed energy correlations."""

    if series.target != "energy" or series.order is None:
        raise ValueError("energy correlation requires an energy series")
    if len(series.expected) < 2:
        raise ValueError("energy correlation requires at least two samples")
    if series.expected.unique().numel() < 2 or series.observed.unique().numel() < 2:
        raise ValueError("energy correlation is undefined for constant arrays")
    pearson = _correlation(series.expected, series.observed)
    spearman = _correlation(_ranks(series.expected), _ranks(series.observed))
    if not math.isfinite(pearson) or not math.isfinite(spearman):
        raise ValueError("energy correlations must be finite")
    for name, value in (("pearson", pearson), ("spearman", spearman)):
        try:
            stored = float(series.stored_metrics[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"stored energy {name} is invalid") from error
        if not math.isclose(
            value,
            stored,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        ):
            raise ValueError(
                f"stored energy {name} disagrees with recomputed value: "
                f"{stored} != {value}"
            )
    return CorrelationRow(
        order=series.order,
        sample_count=len(series.observed),
        pearson=pearson,
        spearman=spearman,
    )
