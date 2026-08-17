from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib import colors

from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
    CampaignPlotStyle,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_analysis import (
    DensityContours,
    PanelAnalysis,
    filter_log_pairs,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_rendering import (
    build_panel_figure,
    panel_stem,
    render_panel,
    render_statistics,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
    PanelKey,
    PlotSource,
)


STYLE = CampaignPlotStyle(
    grid_size=24,
    gaussian_sigma=1.0,
    contour_masses=(0.5, 0.8),
    scatter_max_points=48,
    scatter_seed=20260816,
    scatter_size=4.0,
    scatter_alpha=0.7,
    log_margin=0.05,
    figure_size=(3.0, 3.0),
    dpi=48,
    formats=("png", "pdf"),
)
RUNS = ("full_remote_b8_e8", "lr_1e-4", "lr_1e-6")
DATASETS = (
    ("matpes_test", "test", ("energy", "force", "stress")),
    ("mad_test", "mad_test", ("energy", "force")),
    ("matpes_train", "matpes_train", ("energy", "force", "stress")),
)


def _analysis(target: str, *, run: str = RUNS[0], dataset: str = "matpes_test"):
    log_uncertainty = np.linspace(-2.8, 1.8, 96)
    log_residual = 0.72 * log_uncertainty + 0.18 * np.sin(log_uncertainty * 3.1)
    filtered = filter_log_pairs(
        np.power(10.0, log_uncertainty),
        np.power(10.0, log_residual),
    )
    centers = np.linspace(-2.85, 1.85, 12)
    coordinate = np.linspace(-1.0, 1.0, 12)
    grid = np.exp(-3.0 * (coordinate[:, None] ** 2 + coordinate[None, :] ** 2))
    grid /= grid.sum()
    levels = (float(np.quantile(grid, 0.55)), float(np.quantile(grid, 0.82)))
    density = DensityContours(
        x_centers=centers,
        y_centers=centers.copy(),
        grid=grid,
        contour_levels=levels,
        actual_contour_masses=tuple(
            float(grid[grid >= level].sum()) for level in levels
        ),
        histogram_count=filtered.valid_count,
    )
    storage_key = "test" if dataset == "matpes_test" else dataset
    return PanelAnalysis(
        key=PanelKey(run, dataset, storage_key, target),
        filtered=filtered,
        density=density,
        scatter_indices=np.arange(0, filtered.valid_count, 2, dtype=np.int64),
        log_limits=(-3.0, 2.0),
        spearman_log=0.875,
        pearson_log10=0.8125,
    )


def _source(tmp_path: Path, analysis: PanelAnalysis, marker: str) -> PlotSource:
    members = []
    root = tmp_path / analysis.key.run_label / analysis.key.storage_key
    for index in range(8):
        path = root / "members" / f"member_{index:03d}" / "raw.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{marker}-raw-member-{index}".encode())
        members.append(path)
    return PlotSource(
        key=analysis.key,
        targets_path=root / "targets.npz",
        uq_results_path=root / "uncertainty" / "results.npz",
        member_paths=tuple(members),
        prediction_manifest_sha256="1" * 64,
        uq_manifest_sha256="2" * 64,
        targets_sha256="3" * 64,
    )


def _collection_colors(axis) -> set[str]:
    result: set[str] = set()
    for collection in axis.collections:
        for getter in (collection.get_facecolors, collection.get_edgecolors):
            values = getter()
            for value in values:
                result.add(colors.to_hex(value, keep_alpha=False).lower())
    return result


@pytest.mark.parametrize(
    ("target", "unit"),
    [
        ("energy", "eV/atom"),
        ("force", "eV/Angstrom"),
        ("stress", "eV/Angstrom^3"),
    ],
)
def test_panel_figure_matches_carnet_visual_contract_and_declared_units(
    target: str, unit: str
) -> None:
    figure = build_panel_figure(_analysis(target), STYLE)
    try:
        assert matplotlib.get_backend().lower() == "agg"
        assert len(figure.axes) == 1
        axis = figure.axes[0]
        assert axis.get_xscale() == axis.get_yscale() == "log"
        assert axis.get_aspect() == pytest.approx(1.0)
        assert axis.get_xlim() == pytest.approx((1.0e-3, 1.0e2))
        assert axis.get_ylim() == pytest.approx(axis.get_xlim())
        assert axis.get_xlabel() == f"Raw {target} uncertainty ({unit})"
        assert axis.get_ylabel() == f"Absolute residual ({unit})"

        labels = axis.get_legend_handles_labels()[1]
        assert {"Residual ≤ uncertainty", "Samples", "Ideal calibration"} <= set(labels)
        ideal = next(
            line for line in axis.lines if line.get_label() == "Ideal calibration"
        )
        assert colors.to_hex(ideal.get_color()).lower() == "#000000"
        np.testing.assert_allclose(ideal.get_xdata(), ideal.get_ydata())
        visible_colors = _collection_colors(axis)
        assert "#d9d9d9" in visible_colors
        assert "#f28e2b" in visible_colors

        annotation = "\n".join(text.get_text() for text in axis.texts)
        assert "Spearman(log) = 0.8750" in annotation
        assert "Pearson(log10) = 0.8125" in annotation
        assert "valid = 96 / 96" in annotation
    finally:
        plt.close(figure)


def test_panel_stems_preserve_the_exact_24_panel_campaign_order() -> None:
    keys = tuple(
        PanelKey(run, dataset, storage_key, target)
        for run in RUNS
        for dataset, storage_key, targets in DATASETS
        for target in targets
    )
    expected = tuple(
        f"{key.run_label}__{key.dataset_label}__raw_{key.target}_uncertainty_vs_residual"
        for key in keys
    )

    assert len(keys) == 24
    assert tuple(panel_stem(key) for key in keys) == expected
    assert len(set(expected)) == 24
    assert not any("member_sweep" in stem for stem in expected)


def test_render_panel_writes_deterministic_png_pdf_without_pdf_timestamps(
    tmp_path: Path,
) -> None:
    analysis = _analysis("energy")
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()

    left_paths = render_panel(analysis, STYLE, left)
    right_paths = render_panel(analysis, STYLE, right)

    assert tuple(path.name for path in left_paths) == (
        f"{panel_stem(analysis.key)}.png",
        f"{panel_stem(analysis.key)}.pdf",
    )
    for first, second in zip(left_paths, right_paths, strict=True):
        assert first.read_bytes() == second.read_bytes()
        assert first.stat().st_size > 0
    assert left_paths[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    pdf = left_paths[1].read_bytes()
    assert pdf.startswith(b"%PDF")
    assert b"/CreationDate" not in pdf
    assert b"/ModDate" not in pdf


def test_statistics_csv_json_have_one_semantically_identical_row_per_panel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_rendering

    analyses = tuple(_analysis(target) for target in ("energy", "force", "stress"))
    sources = tuple(
        _source(tmp_path, analysis, marker=analysis.key.target) for analysis in analyses
    )
    output = tmp_path / "statistics"
    output.mkdir()
    monkeypatch.setattr(
        plot_rendering,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("statistics must reuse manifest provenance")
        ),
        raising=False,
    )

    csv_path, json_path = render_statistics(analyses, sources, STYLE, output)

    assert csv_path.name == "raw_std_statistics.csv"
    assert json_path.name == "raw_std_statistics.json"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    document = json.loads(json_path.read_text(encoding="utf-8"))
    json_rows = document["rows"]
    assert len(csv_rows) == len(json_rows) == 3
    for csv_row, json_row, analysis in zip(csv_rows, json_rows, analyses, strict=True):
        assert csv_row["run_label"] == json_row["run_label"] == analysis.key.run_label
        assert csv_row["dataset_label"] == json_row["dataset_label"]
        assert csv_row["storage_key"] == json_row["storage_key"]
        assert csv_row["target"] == json_row["target"] == analysis.key.target
        assert csv_row["shape"] == "x".join(map(str, json_row["shape"]))
        for name in (
            "original_count",
            "valid_count",
            "excluded_nan",
            "excluded_inf",
            "excluded_zero",
            "excluded_negative",
            "scatter_count",
            "histogram_count",
        ):
            assert int(csv_row[name]) == json_row[name]
        for name in (
            "shared_log_low",
            "shared_log_high",
            "spearman_log",
            "pearson_log10",
        ):
            assert float(csv_row[name]) == pytest.approx(json_row[name])

    assert document["schema"] == "upet.bootstrap.plot-statistics/v1"
    assert document["mode"] == "raw"
    assert document["style"] == json.loads(json.dumps(asdict(STYLE)))
    assert len(document["sources"]) == 3
    for record, source in zip(document["sources"], sources, strict=True):
        assert record["key"] == {
            "run_label": source.key.run_label,
            "dataset_label": source.key.dataset_label,
            "storage_key": source.key.storage_key,
            "target": source.key.target,
        }
        assert record["prediction_manifest_sha256"] == "1" * 64
        assert record["uq_manifest_sha256"] == "2" * 64
        assert record["targets_sha256"] == "3" * 64
        assert [member["index"] for member in record["ordered_members"]] == list(
            range(8)
        )
        assert record["ordered_members"] == [
            {
                "index": index,
                "path": f"member_{index:03d}/raw.npz",
            }
            for index in range(8)
        ]
    assert "member_sweep" not in csv_path.read_text(encoding="utf-8")
    assert "member_sweep" not in json_path.read_text(encoding="utf-8")
