"""Validated analysis and plotting inputs for completed confidence-head runs."""

from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

import matplotlib
import torch
import yaml


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

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
    verify_run(root, full=True, allow_plots=True)
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


@contextmanager
def _atomic_output(path: Path) -> Iterator[Path]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=target.suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError(f"plot output is empty: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def write_bin_csv(path: Path, rows: Sequence[BinRow]) -> Path:
    """Atomically write one row for every configured bin."""

    if len(rows) != _EXPECTED_NUM_BINS:
        raise ValueError("bin statistics CSV requires exactly 50 rows")
    fieldnames = tuple(BinRow.__dataclass_fields__)
    target = Path(path)
    with _atomic_output(target) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(asdict(row) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
    return target


def write_correlation_csv(
    path: Path,
    rows: Sequence[CorrelationRow],
) -> Path:
    """Atomically write order-sorted energy correlations."""

    ordered = sorted(rows, key=lambda row: row.order)
    if [row.order for row in ordered] != list(range(1, 9)):
        raise ValueError("energy correlation rows must cover orders 1 through 8")
    target = Path(path)
    with _atomic_output(target) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=tuple(CorrelationRow.__dataclass_fields__),
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in ordered)
            handle.flush()
            os.fsync(handle.fileno())
    return target


def _target_ylabel(target: Target) -> str:
    if target == "energy":
        return "Absolute energy error per atom (eV/atom)"
    return "Mean absolute force-component error per atom (eV/Å)"


def _target_title(series: PlotSeries) -> str:
    if series.target == "energy":
        return f"Energy confidence (per atom), order {series.order}"
    return "Force confidence (per-atom Cartesian-component mean)"


def _draw_boxplot(
    axis: Any,
    series: PlotSeries,
    *,
    title: str,
    tick_fontsize: float,
) -> None:
    rows = bin_rows(series)
    predicted = series.logits.argmax(dim=-1)
    positions: list[int] = []
    distributions: list[torch.Tensor] = []
    for row in rows:
        values = series.observed[predicted == row.bin]
        if values.numel():
            positions.append(row.bin)
            distributions.append(values)
    boxes = axis.boxplot(
        [values.numpy() for values in distributions],
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "#b2182b", "linewidth": 1.2},
        whiskerprops={"color": "#4d4d4d", "linewidth": 0.8},
        capprops={"color": "#4d4d4d", "linewidth": 0.8},
        flierprops={
            "marker": ".",
            "markersize": 1.5,
            "markerfacecolor": "#777777",
            "markeredgecolor": "#777777",
            "alpha": 0.4,
        },
    )
    for patch in boxes["boxes"]:
        patch.set_facecolor("#80b1d3")
        patch.set_edgecolor("#2b5c85")
        patch.set_alpha(0.8)
    axis.set_yscale("symlog", linthresh=1e-4)
    axis.set_xlim(-0.75, _EXPECTED_NUM_BINS - 0.25)
    axis.set_xticks(range(_EXPECTED_NUM_BINS))
    axis.set_xticklabels(
        [f"{row.bin}\nn={row.sample_count}" for row in rows],
        rotation=90,
        fontsize=tick_fontsize,
    )
    axis.set_xlabel("Predicted argmax bin")
    axis.set_ylabel(_target_ylabel(series.target))
    axis.set_title(title)
    axis.grid(axis="y", which="both", alpha=0.25, linewidth=0.6)


def _save_figure(figure: Any, path: Path, *, dpi: int | None = None) -> Path:
    target = Path(path)
    with _atomic_output(target) as temporary:
        figure.savefig(
            temporary,
            format=target.suffix.removeprefix("."),
            dpi=dpi,
            bbox_inches="tight",
        )
    return target


def plot_single_boxplot(
    series: PlotSeries,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write one target's statistics plus PNG and vector PDF boxplots."""

    root = Path(output_dir)
    stem = f"test_{series.target}_argmax_bin"
    csv_path = write_bin_csv(root / f"{stem}_statistics.csv", bin_rows(series))
    figure, axis = plt.subplots(figsize=(20, 7))
    try:
        _draw_boxplot(
            axis,
            series,
            title=_target_title(series),
            tick_fontsize=6,
        )
        figure.tight_layout()
        png_path = _save_figure(
            figure,
            root / f"{stem}_boxplot.png",
            dpi=300,
        )
        pdf_path = _save_figure(figure, root / f"{stem}_boxplot.pdf")
    finally:
        plt.close(figure)
    return csv_path, png_path, pdf_path


def plot_combined_energy_boxplots(
    series_by_order: Mapping[int, PlotSeries],
    output_dir: Path,
) -> Path:
    """Write the order 1–8 energy comparison as a 4-by-2 PDF."""

    if set(series_by_order) != set(range(1, 9)):
        raise ValueError("combined energy plot requires orders 1 through 8")
    figure, axes = plt.subplots(4, 2, figsize=(24, 26))
    try:
        for order, axis in zip(range(1, 9), axes.reshape(-1), strict=True):
            series = series_by_order[order]
            if series.target != "energy" or series.order != order:
                raise ValueError(f"invalid energy series for order {order}")
            _draw_boxplot(
                axis,
                series,
                title=f"Energy confidence order {order}",
                tick_fontsize=4,
            )
        figure.suptitle("UPET energy argmax-bin error distributions", fontsize=16)
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        return _save_figure(
            figure,
            Path(output_dir) / "combined_energy_argmax_bin_boxplots.pdf",
        )
    finally:
        plt.close(figure)


def plot_combined_force_boxplot(series: PlotSeries, output_dir: Path) -> Path:
    """Write the force-only per-atom comparison PDF."""

    if series.target != "force" or series.force_target_mode != _ATOM_MEAN:
        raise ValueError("combined force plot requires force atom_mean data")
    figure, axis = plt.subplots(figsize=(20, 7))
    try:
        _draw_boxplot(
            axis,
            series,
            title=_target_title(series),
            tick_fontsize=6,
        )
        figure.tight_layout()
        return _save_figure(
            figure,
            Path(output_dir) / "combined_force_argmax_bin_boxplots.pdf",
        )
    finally:
        plt.close(figure)


def plot_energy_correlations(
    rows: Sequence[CorrelationRow],
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write the no-CI order 1–8 Pearson/Spearman comparison."""

    ordered = sorted(rows, key=lambda row: row.order)
    if [row.order for row in ordered] != list(range(1, 9)):
        raise ValueError("energy correlation plot requires orders 1 through 8")
    values = [value for row in ordered for value in (row.pearson, row.spearman)]
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
        raise ValueError("correlations must be finite and within [-1, 1]")
    root = Path(output_dir)
    stem = "linear_order_correlations_no_ci"
    csv_path = write_correlation_csv(root / f"{stem}.csv", ordered)
    figure, axis = plt.subplots(figsize=(9, 6))
    try:
        orders = [row.order for row in ordered]
        axis.plot(
            orders,
            [row.pearson for row in ordered],
            marker="o",
            linewidth=2,
            label="Pearson",
        )
        axis.plot(
            orders,
            [row.spearman for row in ordered],
            marker="s",
            linewidth=2,
            label="Spearman",
        )
        lower = min(0.0, min(values))
        upper = max(0.0, max(values))
        padding = max(0.03, 0.08 * (upper - lower))
        axis.set_ylim(max(-1.0, lower - padding), min(1.0, upper + padding))
        axis.set_xticks(orders)
        axis.set_xlabel("Energy cumulant order")
        axis.set_ylabel("Correlation of expected and observed error")
        axis.set_title("UPET energy confidence correlations (no CI)")
        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()
        png_path = _save_figure(figure, root / f"{stem}.png", dpi=300)
        pdf_path = _save_figure(figure, root / f"{stem}.pdf")
    finally:
        plt.close(figure)
    return csv_path, png_path, pdf_path


@dataclass(frozen=True)
class CompletedRuns:
    """The exact force-only and energy-order series required for comparison."""

    force: PlotSeries
    energy_by_order: Mapping[int, PlotSeries]


def discover_completed_runs(
    runs_root: Path,
    *,
    explicit_run_dirs: Sequence[Path] = (),
) -> CompletedRuns:
    """Find exactly one force run and one energy run for every order 1–8."""

    root = Path(runs_root).resolve()
    if explicit_run_dirs:
        candidates = tuple(Path(path).resolve() for path in explicit_run_dirs)
    else:
        if not root.is_dir():
            raise ValueError(f"runs root does not exist: {root}")
        candidates = tuple(
            child.resolve()
            for child in sorted(root.iterdir())
            if child.is_dir()
            and (child / "manifest.json").is_file()
            and (child / "evaluation" / "manifest.json").is_file()
        )
    if not candidates:
        raise ValueError(f"no completed plot candidates found under {root}")

    force_series: list[PlotSeries] = []
    energy_by_order: dict[int, PlotSeries] = {}
    for candidate in candidates:
        series = load_plot_series(candidate)
        if series.target == "force":
            force_series.append(series)
            continue
        if series.order is None:
            raise ValueError(f"energy run has no order: {candidate}")
        if series.order in energy_by_order:
            previous = energy_by_order[series.order].run_dir
            raise ValueError(
                f"duplicate energy order {series.order}: {previous}, {candidate}"
            )
        energy_by_order[series.order] = series

    if len(force_series) != 1:
        paths = ", ".join(str(series.run_dir) for series in force_series) or "<none>"
        raise ValueError(f"expected exactly one force run, found: {paths}")
    required_orders = set(range(1, 9))
    missing = sorted(required_orders - set(energy_by_order))
    extra = sorted(set(energy_by_order) - required_orders)
    if missing:
        raise ValueError(f"missing energy orders: {missing}")
    if extra:
        raise ValueError(f"unexpected energy orders: {extra}")
    return CompletedRuns(
        force=force_series[0],
        energy_by_order={
            order: energy_by_order[order] for order in sorted(required_orders)
        },
    )


def _validate_comparable_energy(
    series_by_order: Mapping[int, PlotSeries],
) -> tuple[CorrelationRow, ...]:
    if set(series_by_order) != set(range(1, 9)):
        raise ValueError("energy comparison requires orders 1 through 8")
    reference = series_by_order[1]
    if reference.target != "energy" or reference.order != 1:
        raise ValueError("order 1 reference must be an energy series")
    correlations: list[CorrelationRow] = []
    for order in range(1, 9):
        series = series_by_order[order]
        if series.target != "energy" or series.order != order:
            raise ValueError(f"invalid energy series for order {order}")
        if not torch.equal(series.structure_ids, reference.structure_ids):
            raise ValueError(
                f"energy structure_ids differ between orders 1 and {order}"
            )
        if not torch.equal(series.observed, reference.observed):
            raise ValueError(
                f"observed energy errors differ between orders 1 and {order}"
            )
        if not torch.equal(series.representatives, reference.representatives):
            raise ValueError(
                f"energy bin representatives differ between orders 1 and {order}"
            )
        correlations.append(energy_correlation(series))
    return tuple(correlations)


def energy_series_by_order(
    series: Sequence[PlotSeries],
) -> Mapping[int, PlotSeries]:
    """Index exactly eight energy series by their unique orders."""

    by_order: dict[int, PlotSeries] = {}
    for value in series:
        if value.target != "energy" or value.order is None:
            raise ValueError(f"expected an energy series, got {value.target}")
        if value.order in by_order:
            raise ValueError(f"duplicate energy order {value.order}")
        by_order[value.order] = value
    required = set(range(1, 9))
    missing = sorted(required - set(by_order))
    extra = sorted(set(by_order) - required)
    if missing:
        raise ValueError(f"missing energy orders: {missing}")
    if extra:
        raise ValueError(f"unexpected energy orders: {extra}")
    return {order: by_order[order] for order in range(1, 9)}


def validate_comparable_energy(
    series_by_order: Mapping[int, PlotSeries],
) -> tuple[CorrelationRow, ...]:
    """Validate fair order comparisons and return checked correlations."""

    return _validate_comparable_energy(series_by_order)


def plot_completed_runs(
    runs: CompletedRuns,
    *,
    comparisons_dir: Path,
) -> tuple[Path, ...]:
    """Validate all nine series, then write single-run and comparison plots."""

    if runs.force.target != "force" or runs.force.force_target_mode != _ATOM_MEAN:
        raise ValueError("completed force run must use atom_mean semantics")
    correlations = validate_comparable_energy(runs.energy_by_order)

    artifacts: list[Path] = []
    for order in range(1, 9):
        series = runs.energy_by_order[order]
        artifacts.extend(
            plot_single_boxplot(
                series,
                series.run_dir / "plots" / "argmax_bin_boxplots",
            )
        )
    artifacts.extend(
        plot_single_boxplot(
            runs.force,
            runs.force.run_dir / "plots" / "argmax_bin_boxplots",
        )
    )
    comparison_root = Path(comparisons_dir)
    boxplot_root = comparison_root / "argmax_bin_boxplots"
    correlation_root = comparison_root / "energy_correlations"
    artifacts.append(plot_combined_energy_boxplots(runs.energy_by_order, boxplot_root))
    artifacts.append(plot_combined_force_boxplot(runs.force, boxplot_root))
    artifacts.extend(plot_energy_correlations(correlations, correlation_root))
    return tuple(artifacts)
