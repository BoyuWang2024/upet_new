from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.plot_analysis import (
    PlotInput,
    PlotPanelInput,
    PlotSettings,
    analyze_plot_input,
    load_completed_fge_plot_input,
    load_inference_plot_input,
    load_plot_config,
)
from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.tests.test_inference_validation import _completed
from Uncertainty_Quantification.FGE.tests.test_validation import make_canonical_result


def _plot_input(*, stress: bool = True) -> PlotInput:
    uncertainty = torch.tensor([0.05, 0.1, 0.2, 0.4, 0.8, 1.6], dtype=torch.float32)
    residual = torch.tensor([0.025, 0.2, 0.1, 0.8, 0.4, 3.2], dtype=torch.float32)
    panels = [
        PlotPanelInput("energy", uncertainty, residual),
        PlotPanelInput("force", uncertainty.repeat(3), residual.repeat(3)),
    ]
    if stress:
        panels.append(
            PlotPanelInput("stress", uncertainty.repeat(6), residual.repeat(6))
        )
    return PlotInput(
        dataset_label="matpes_test" if stress else "mad_test",
        source_kind="completed_fge" if stress else "inference",
        source_identity="a" * 64,
        panels=tuple(panels),
    )


def test_plot_analysis_is_immutable_deterministic_and_domain_local() -> None:
    settings = PlotSettings(scatter_max_points=4)
    first = analyze_plot_input(_plot_input(), settings)
    second = analyze_plot_input(_plot_input(), settings)

    assert first.domains == ("energy", "force", "stress")
    assert (
        first.panels[0].scatter_indices.tolist()
        == second.panels[0].scatter_indices.tolist()
    )
    assert first.panels[0].density.grid.shape == (160, 160)
    assert first.panels[0].density.histogram_count == 6
    assert len(first.panels[0].density.contour_levels) == 5
    with pytest.raises(FrozenInstanceError):
        settings.dpi = 72  # type: ignore[misc]


def test_mad_domains_are_exactly_energy_and_force() -> None:
    result = analyze_plot_input(_plot_input(stress=False), PlotSettings())

    assert result.domains == ("energy", "force")


def test_filtering_counts_nonpositive_and_nonfinite_pairs() -> None:
    value = _plot_input(stress=False)
    panel = PlotPanelInput(
        "energy",
        torch.tensor([1.0, 0.0, -1.0, float("nan"), float("inf"), 2.0]),
        torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 4.0]),
    )
    result = analyze_plot_input(
        PlotInput(
            value.dataset_label, value.source_kind, value.source_identity, (panel,)
        ),
        PlotSettings(),
    )

    assert result.panels[0].valid_count == 2
    assert result.panels[0].excluded == {
        "nan": 1,
        "inf": 1,
        "nonpositive": 2,
    }


def test_plot_config_rejects_checkpoint_or_correlation_curve_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "plot.yaml"
    path.write_text(
        """schema_version: upet.fge.plot.v1
input:
  root: input
  kind: inference
output:
  root: output
plot:
  checkpoint_sweep: true
""",
        encoding="utf-8",
    )

    with pytest.raises(HardFailure, match="checkpoint|unknown"):
        load_plot_config(path)


def test_completed_fge_adapter_reopens_formal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import Uncertainty_Quantification.FGE.fge.plot_analysis as plotting

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    monkeypatch.setattr(
        plotting, "validate_completed_result", lambda candidate: candidate
    )

    value = load_completed_fge_plot_input(root)

    assert tuple(panel.domain for panel in value.panels) == (
        "energy",
        "force",
        "stress",
    )
    assert value.source_kind == "completed_fge"


def test_inference_adapter_reopens_chunked_result(tmp_path: Path) -> None:
    root = _completed(tmp_path)
    value = load_inference_plot_input(root)
    assert tuple(panel.domain for panel in value.panels) == (
        "energy",
        "force",
        "stress",
    )
    assert value.dataset_label == "matpes_train"
