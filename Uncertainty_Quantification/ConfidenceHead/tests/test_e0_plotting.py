from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from confidence_head import e0_plotting
from confidence_head.density_plotting import DensitySettings
from confidence_head.e0_plotting import (
    build_variant_series,
    load_e0_density_series,
    publish_e0_density_campaign,
    verify_e0_density_publication,
)
from confidence_head.e0_publication import E0_VARIANTS, publish_e0_campaign
from confidence_head.plot_analysis import PlotSeries
from matplotlib import image as mpl_image
from test_e0_publication import _inputs


SETTINGS = DensitySettings(
    scatter_max_points=20,
    scatter_seed=7,
    grid_size=12,
    gaussian_sigma=0.8,
    contour_masses=(0.5, 0.9),
    log_margin=0.05,
    dpi=40,
)


def _raw_series(tmp_path: Path) -> PlotSeries:
    return PlotSeries(
        run_dir=tmp_path / "energy-1",
        structure_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        target="energy",
        order=1,
        logits=torch.ones((3, 3), dtype=torch.float64),
        observed=torch.tensor([0.3, 0.5, 0.7], dtype=torch.float64),
        expected=torch.tensor([0.2, 0.4, 0.8], dtype=torch.float64),
        representatives=torch.tensor([0.5, 1.5, 2.5], dtype=torch.float64),
        stored_metrics={"sample_count": 3},
        force_target_mode=None,
    )


def test_variant_series_preserves_uq_and_replaces_only_observed(
    tmp_path: Path,
) -> None:
    raw = _raw_series(tmp_path)
    observed = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)

    series = build_variant_series(raw, observed)

    assert torch.equal(series.logits, raw.logits)
    assert torch.equal(series.expected, raw.expected)
    assert torch.equal(series.representatives, raw.representatives)
    assert torch.equal(series.structure_ids, raw.structure_ids)
    assert torch.equal(series.observed, observed)


def test_density_campaign_renders_three_variants_and_eight_orders(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    campaign = publish_e0_campaign(inputs, tmp_path / "campaign")
    source = inputs.test_sources[1] / "predictions.pt"
    before = source.read_bytes()

    series = load_e0_density_series(campaign)
    root = publish_e0_density_campaign(
        campaign,
        tmp_path / "plots",
        SETTINGS,
    )
    manifest = verify_e0_density_publication(root / "manifest.json", full=True)

    assert set(series) == set(E0_VARIANTS)
    assert all(set(orders) == set(range(1, 9)) for orders in series.values())
    assert manifest["status"] == "complete"
    assert source.read_bytes() == before
    assert (
        len(
            list(
                (root / "variants").rglob(
                    "test_energy_expected_vs_observed_density.png"
                )
            )
        )
        == 24
    )
    assert (
        len(
            list(
                (root / "variants").rglob(
                    "test_energy_expected_vs_observed_density.pdf"
                )
            )
        )
        == 24
    )
    assert not list(root.rglob("*force*"))
    assert not list(root.rglob("*boxplot*"))
    assert not list(root.parent.glob(".staging-*"))
    correlations = root / "comparisons" / "three_variant_energy_correlations"
    for suffix in (".csv", ".json", ".png", ".pdf"):
        assert correlations.with_suffix(suffix).is_file()
    correlation_payload = json.loads(
        correlations.with_suffix(".json").read_text(encoding="utf-8")
    )
    assert {row["variant"] for row in correlation_payload["rows"]} == set(E0_VARIANTS)
    assert {row["data_use"] for row in correlation_payload["rows"]} == {
        "uncorrected baseline",
        "test-informed/oracle",
        "validation-only calibration",
    }
    for order in range(1, 9):
        limits_by_variant = {
            variant: tuple(
                json.loads(
                    (
                        root
                        / "variants"
                        / variant
                        / "runs"
                        / f"order-{order}"
                        / "density_analysis.json"
                    ).read_text(encoding="utf-8")
                )["log_limits"]
            )
            for variant in E0_VARIANTS
        }
        assert len(set(limits_by_variant.values())) == 1, limits_by_variant
    for path in root.rglob("*.png"):
        pixels = mpl_image.imread(path)
        assert pixels.ndim in (2, 3)
        assert pixels.size > 0
    for path in root.rglob("*.pdf"):
        assert path.stat().st_size > 0

    correlations.with_suffix(".csv").write_bytes(
        correlations.with_suffix(".csv").read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_e0_density_publication(root / "manifest.json", full=True)


def test_density_campaign_reuses_identical_publication(tmp_path: Path) -> None:
    campaign = publish_e0_campaign(_inputs(tmp_path), tmp_path / "campaign")

    first = publish_e0_density_campaign(campaign, tmp_path / "plots", SETTINGS)
    manifest_before = (first / "manifest.json").read_bytes()
    second = publish_e0_density_campaign(campaign, tmp_path / "plots", SETTINGS)

    assert second == first
    with pytest.raises(ValueError, match="identity conflicts"):
        publish_e0_density_campaign(
            campaign,
            tmp_path / "plots",
            replace(SETTINGS, dpi=SETTINGS.dpi + 1),
        )
    assert (first / "manifest.json").read_bytes() == manifest_before
    assert not list(first.parent.glob(".staging-*"))


def test_density_campaign_cleans_staging_after_render_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = publish_e0_campaign(_inputs(tmp_path), tmp_path / "campaign")

    def fail_render(*args: object, **kwargs: object) -> None:
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(e0_plotting, "render_density_panel", fail_render)
    output = tmp_path / "plots"
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        publish_e0_density_campaign(campaign, output, SETTINGS)

    assert not output.exists()
    assert not list(output.parent.glob(".staging-*"))
