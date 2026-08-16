from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.plot_analysis import (
    PlotInput,
    PlotPanelInput,
    PlotSettings,
    analyze_plot_input,
)
from Uncertainty_Quantification.FGE.fge.plot_rendering import render_plot_suite


def _analysis(*, stress: bool):
    base = torch.logspace(-3, 1, 64)
    panels = [
        PlotPanelInput("energy", base, base.flip(0)),
        PlotPanelInput("force", base.repeat(3), base.flip(0).repeat(3)),
    ]
    if stress:
        panels.append(PlotPanelInput("stress", base.repeat(6), base.flip(0).repeat(6)))
    return analyze_plot_input(
        PlotInput(
            "matpes_test" if stress else "mad_test",
            "completed_fge" if stress else "inference",
            "b" * 64,
            tuple(panels),
        ),
        PlotSettings(),
    )


def _snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.iterdir()
        if path.is_file()
    }


def test_matpes_rendering_publishes_three_independent_panels(tmp_path: Path) -> None:
    destination = tmp_path / "matpes_test"

    result = render_plot_suite(_analysis(stress=True), destination)

    assert result == destination
    assert {path.name for path in destination.iterdir()} == {
        "energy_uncertainty_vs_absolute_residual.png",
        "energy_uncertainty_vs_absolute_residual.pdf",
        "force_uncertainty_vs_absolute_residual.png",
        "force_uncertainty_vs_absolute_residual.pdf",
        "stress_uncertainty_vs_absolute_residual.png",
        "stress_uncertainty_vs_absolute_residual.pdf",
        "plot_statistics.json",
        "plot_manifest.json",
    }
    image = mpimg.imread(destination / "energy_uncertainty_vs_absolute_residual.png")
    assert image.shape[:2] == (2100, 2100)
    assert (
        (destination / "energy_uncertainty_vs_absolute_residual.pdf")
        .read_bytes()
        .startswith(b"%PDF")
    )
    manifest = json.loads((destination / "plot_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "PASS"
    assert len(manifest["outputs"]) == 7


def test_mad_rendering_has_no_stress_artifact_and_rerun_is_read_only(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "mad_test"
    analysis = _analysis(stress=False)
    render_plot_suite(analysis, destination)
    before = _snapshot(destination)

    render_plot_suite(analysis, destination)

    assert _snapshot(destination) == before
    assert len(before) == 6
    assert all("stress" not in name for name in before)


def test_existing_conflicting_plot_artifact_is_rejected(tmp_path: Path) -> None:
    destination = tmp_path / "mad_test"
    analysis = _analysis(stress=False)
    render_plot_suite(analysis, destination)
    (destination / "energy_uncertainty_vs_absolute_residual.png").write_bytes(b"bad")

    with pytest.raises(HardFailure, match="conflict|hash"):
        render_plot_suite(analysis, destination)
