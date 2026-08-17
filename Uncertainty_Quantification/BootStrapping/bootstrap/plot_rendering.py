"""Deterministic Carnet-style rendering for Bootstrap campaign plots."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .campaign import CampaignPlotStyle
from .errors import HardFailure
from .plot_analysis import PanelAnalysis
from .plot_source import PanelKey, PlotSource


_ORANGE = "#F28E2B"
_GRAY = "#D9D9D9"
_PNG_METADATA = {"Software": "upet BootStrapping raw STD plotting"}
_PDF_METADATA = {
    "Creator": "upet BootStrapping raw STD plotting",
    "Producer": "upet BootStrapping raw STD plotting",
    "CreationDate": None,
    "ModDate": None,
}
_UNITS = {
    "energy": "eV/atom",
    "force": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
}
_CSV_COLUMNS = (
    "run_label",
    "dataset_label",
    "storage_key",
    "target",
    "shape",
    "original_count",
    "valid_count",
    "excluded_nan",
    "excluded_inf",
    "excluded_zero",
    "excluded_negative",
    "scatter_count",
    "histogram_count",
    "shared_log_low",
    "shared_log_high",
    "spearman_log",
    "pearson_log10",
)


@dataclass(frozen=True)
class PanelStatistics:
    """Compact immutable statistics retained after one panel is rendered."""

    key: PanelKey
    shape: tuple[int, ...]
    original_count: int
    valid_count: int
    excluded_nan: int
    excluded_inf: int
    excluded_zero: int
    excluded_negative: int
    scatter_count: int
    histogram_count: int
    shared_log_low: float
    shared_log_high: float
    spearman_log: float
    pearson_log10: float


def _style() -> dict[str, Any]:
    return {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#BFBFBF",
        "grid.alpha": 0.28,
        "grid.linewidth": 0.55,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.compression": 6,
    }


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise HardFailure(f"plot {label} must be nonempty text")
    if value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise HardFailure(f"plot {label} is not a safe file component")
    return value


def panel_stem(key: PanelKey) -> str:
    """Return the stable file stem for one campaign panel."""

    if not isinstance(key, PanelKey):
        raise HardFailure("plot panel stem requires PanelKey")
    run = _safe_component(key.run_label, "run label")
    dataset = _safe_component(key.dataset_label, "dataset label")
    if key.target not in _UNITS:
        raise HardFailure(f"plot target is not supported: {key.target}")
    return f"{run}__{dataset}__raw_{key.target}_uncertainty_vs_residual"


def build_panel_figure(analysis: PanelAnalysis, style: CampaignPlotStyle) -> Figure:
    """Build one deterministic shared-scale uncertainty-versus-residual figure."""

    if not isinstance(analysis, PanelAnalysis):
        raise HardFailure("plot rendering requires PanelAnalysis")
    if not isinstance(style, CampaignPlotStyle):
        raise HardFailure("plot rendering requires CampaignPlotStyle")
    target = analysis.key.target
    if target not in _UNITS:
        raise HardFailure(f"plot target is not supported: {target}")
    low_log, high_log = analysis.log_limits
    if not all(math.isfinite(value) for value in analysis.log_limits):
        raise HardFailure("plot log limits must be finite")
    if low_log >= high_log:
        raise HardFailure("plot log limits must increase strictly")
    low = 10.0**low_log
    high = 10.0**high_log
    if not all(math.isfinite(value) and value > 0 for value in (low, high)):
        raise HardFailure("plot axis limits are not positive and finite")

    with matplotlib.rc_context(_style()):
        figure, axis = plt.subplots(figsize=style.figure_size)
        diagonal = np.geomspace(low, high, 512)
        axis.fill_between(
            diagonal,
            low,
            diagonal,
            color=_GRAY,
            alpha=0.45,
            label="Residual ≤ uncertainty",
            zorder=0,
        )
        indices = analysis.scatter_indices
        axis.scatter(
            analysis.filtered.uncertainty[indices],
            analysis.filtered.residual[indices],
            s=style.scatter_size,
            alpha=style.scatter_alpha,
            color=_ORANGE,
            edgecolors="none",
            rasterized=True,
            label="Samples",
            zorder=2,
        )
        density = analysis.density
        axis.contour(
            10.0**density.x_centers,
            10.0**density.y_centers,
            density.grid.T,
            levels=density.contour_levels,
            colors=[_ORANGE],
            linewidths=0.9,
            alpha=0.9,
            zorder=3,
        )
        axis.plot(
            diagonal,
            diagonal,
            color="black",
            linewidth=1.1,
            label="Ideal calibration",
            zorder=4,
        )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_aspect("equal", adjustable="box")
        unit = _UNITS[target]
        axis.set_xlabel(f"Raw {target} uncertainty ({unit})")
        axis.set_ylabel(f"Absolute residual ({unit})")
        axis.set_title(f"{analysis.key.run_label} | {analysis.key.dataset_label}")
        axis.text(
            0.03,
            0.97,
            (
                f"Spearman(log) = {analysis.spearman_log:.4f}\n"
                f"Pearson(log10) = {analysis.pearson_log10:.4f}\n"
                f"valid = {analysis.filtered.valid_count:,} / "
                f"{analysis.filtered.original_count:,}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#AAAAAA",
                "alpha": 0.92,
            },
        )
        axis.legend(loc="lower right")
        figure.tight_layout()
    return figure


def _require_output_directory(destination: str | Path) -> Path:
    root = Path(destination)
    if root.is_symlink() or not root.is_dir():
        raise HardFailure(f"plot output directory is missing or unsafe: {root}")
    return root


def render_panel(
    analysis: PanelAnalysis,
    style: CampaignPlotStyle,
    destination: str | Path,
) -> tuple[Path, Path]:
    """Render deterministic PNG and PDF artifacts for one panel."""

    root = _require_output_directory(destination)
    stem = panel_stem(analysis.key)
    png_path = root / f"{stem}.png"
    pdf_path = root / f"{stem}.pdf"
    if any(path.exists() or path.is_symlink() for path in (png_path, pdf_path)):
        raise HardFailure(f"plot panel artifact already exists: {stem}")
    figure = build_panel_figure(analysis, style)
    try:
        figure.savefig(
            png_path,
            format="png",
            dpi=style.dpi,
            metadata=_PNG_METADATA,
        )
        figure.savefig(
            pdf_path,
            format="pdf",
            dpi=style.dpi,
            metadata=_PDF_METADATA,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not render plot panel {stem}: {error}") from error
    finally:
        plt.close(figure)
    return png_path, pdf_path


def _key_document(key: PanelKey) -> dict[str, str]:
    return {
        "run_label": key.run_label,
        "dataset_label": key.dataset_label,
        "storage_key": key.storage_key,
        "target": key.target,
    }


def _source_documents(sources: tuple[PlotSource, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        members = [
            {"index": index, "path": f"{path.parent.name}/{path.name}"}
            for index, path in enumerate(source.member_paths)
        ]
        records.append(
            {
                "key": _key_document(source.key),
                "prediction_manifest_sha256": source.prediction_manifest_sha256,
                "uq_manifest_sha256": source.uq_manifest_sha256,
                "targets_sha256": source.targets_sha256,
                "ordered_members": members,
            }
        )
    return records


def summarize_panel(analysis: PanelAnalysis) -> PanelStatistics:
    """Discard large arrays while retaining all published panel statistics."""

    if not isinstance(analysis, PanelAnalysis):
        raise HardFailure("plot summary requires PanelAnalysis")
    excluded = analysis.filtered.excluded
    return PanelStatistics(
        key=analysis.key,
        shape=tuple(analysis.filtered.uncertainty.shape),
        original_count=analysis.filtered.original_count,
        valid_count=analysis.filtered.valid_count,
        excluded_nan=excluded["nan"],
        excluded_inf=excluded["inf"],
        excluded_zero=excluded["zero"],
        excluded_negative=excluded["negative"],
        scatter_count=int(analysis.scatter_indices.size),
        histogram_count=analysis.density.histogram_count,
        shared_log_low=analysis.log_limits[0],
        shared_log_high=analysis.log_limits[1],
        spearman_log=analysis.spearman_log,
        pearson_log10=analysis.pearson_log10,
    )


def _row_document(item: PanelAnalysis | PanelStatistics) -> dict[str, Any]:
    summary = summarize_panel(item) if isinstance(item, PanelAnalysis) else item
    if not isinstance(summary, PanelStatistics):
        raise HardFailure("plot statistics item is invalid")
    return {
        **_key_document(summary.key),
        "shape": list(summary.shape),
        "original_count": summary.original_count,
        "valid_count": summary.valid_count,
        "excluded_nan": summary.excluded_nan,
        "excluded_inf": summary.excluded_inf,
        "excluded_zero": summary.excluded_zero,
        "excluded_negative": summary.excluded_negative,
        "scatter_count": summary.scatter_count,
        "histogram_count": summary.histogram_count,
        "shared_log_low": summary.shared_log_low,
        "shared_log_high": summary.shared_log_high,
        "spearman_log": summary.spearman_log,
        "pearson_log10": summary.pearson_log10,
    }


def _write_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise HardFailure(f"plot artifact already exists: {path}")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise HardFailure(f"could not write plot artifact {path}: {error}") from error


def render_statistics(
    analyses: tuple[PanelAnalysis | PanelStatistics, ...],
    sources: tuple[PlotSource, ...],
    style: CampaignPlotStyle,
    destination: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic CSV and JSON statistics for ordered panels."""

    root = _require_output_directory(destination)
    if not analyses or len(analyses) != len(sources):
        raise HardFailure(
            "plot statistics require matching nonempty panels and sources"
        )
    if not isinstance(style, CampaignPlotStyle):
        raise HardFailure("plot statistics require CampaignPlotStyle")
    for item, source in zip(analyses, sources, strict=True):
        if item.key != source.key:
            raise HardFailure("plot analysis and source order differ")
    rows = [_row_document(item) for item in analyses]

    csv_lines: list[str] = []
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=_CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        csv_row = dict(row)
        csv_row["shape"] = "x".join(str(value) for value in row["shape"])
        for name in (
            "shared_log_low",
            "shared_log_high",
            "spearman_log",
            "pearson_log10",
        ):
            csv_row[name] = format(float(row[name]), ".17g")
        writer.writerow(csv_row)
    csv_lines.append(buffer.getvalue())

    document = {
        "schema": "upet.bootstrap.plot-statistics/v1",
        "mode": "raw",
        "style": asdict(style),
        "rows": rows,
        "sources": _source_documents(sources),
    }
    try:
        json_payload = (
            json.dumps(
                document,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HardFailure(
            f"plot statistics are not JSON serializable: {error}"
        ) from error

    csv_path = root / "raw_std_statistics.csv"
    json_path = root / "raw_std_statistics.json"
    _write_bytes(csv_path, "".join(csv_lines).encode("utf-8"))
    _write_bytes(json_path, json_payload)
    return csv_path, json_path


__all__ = [
    "PanelStatistics",
    "build_panel_figure",
    "panel_stem",
    "render_panel",
    "render_statistics",
    "summarize_panel",
]
