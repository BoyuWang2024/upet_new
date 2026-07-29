import json
from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    atomic_json_dump,
    atomic_npz_save,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr.plotting import (
    PlotConfig,
    analyze_panel,
    filter_log_pairs,
    reliability_bins,
    run_plot,
    standardized_residual_cdf,
)


def test_panel_statistics_are_deterministic() -> None:
    uncertainty = np.array([0.1, 0.2, 0.4, 0.8])
    error = np.array([0.08, 0.3, 0.2, 1.0])

    first = analyze_panel(uncertainty, error)
    second = analyze_panel(uncertainty, error)

    assert first == second
    assert first.count == 4
    assert np.isfinite(first.pearson_log10)
    assert np.isfinite(first.spearman_log10)


def test_nonpositive_and_nonfinite_log_pairs_are_filtered() -> None:
    filtered = filter_log_pairs(
        np.array([1.0, 0.0, np.nan, 2.0, -1.0]),
        np.array([0.5, 1.0, 2.0, np.inf, 3.0]),
    )

    np.testing.assert_array_equal(filtered.uncertainty, np.array([1.0]))
    np.testing.assert_array_equal(filtered.absolute_error, np.array([0.5]))
    assert filtered.original_count == 5
    assert filtered.dropped_count == 4


def test_reliability_bins_and_standardized_cdf() -> None:
    std = np.array([1.0, 2.0, 3.0, 4.0])
    residual = np.array([0.5, -1.0, 6.0, -2.0])

    bins = reliability_bins(std, residual, bin_count=2)
    z, empirical = standardized_residual_cdf(std, residual)

    assert len(bins) == 2
    assert sum(item.count for item in bins) == 4
    assert all(item.mean_predicted_std > 0 for item in bins)
    np.testing.assert_allclose(z, np.array([0.5, 0.5, 0.5, 2.0]))
    np.testing.assert_allclose(empirical, np.array([0.25, 0.5, 0.75, 1.0]))


def _write_evaluation(root: Path) -> Path:
    evaluation = root / "evaluation/cal/eval"
    details = {
        "energy_residual": np.array([0.1, -0.2, 0.3, -0.4]),
        "energy_calibrated_std": np.array([0.15, 0.25, 0.35, 0.45]),
        "force_residual": np.array([0.2, -0.1, 0.4, -0.3, 0.5, -0.6]),
        "force_calibrated_std_component": np.array(
            [0.25, 0.15, 0.45, 0.35, 0.55, 0.65]
        ),
    }
    details_path = evaluation / "details.npz"
    summary_path = evaluation / "summary.json"
    atomic_npz_save(details_path, details)
    atomic_json_dump(summary_path, {"structure_count": 4})
    atomic_json_dump(
        evaluation / "manifest.json",
        {
            "status": "complete",
            "identity": "eval",
            "files": {
                "details.npz": sha256_file(details_path),
                "summary.json": sha256_file(summary_path),
            },
        },
    )
    return evaluation


def test_run_plot_publishes_complete_figures_and_statistics(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    plot_root = tmp_path / "plots"
    _write_evaluation(run_root)

    output = run_plot(
        PlotConfig(
            run_root=run_root,
            output_root=plot_root,
            bin_count=2,
            sample_size=100,
            seed=7,
        )
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "origin" not in manifest
    assert "source_origin" not in manifest
    assert manifest["evaluation_identity"] == "eval"
    assert output.parent == plot_root
    for name in (
        "energy_uncertainty_vs_error.png",
        "energy_uncertainty_vs_error.pdf",
        "force_uncertainty_vs_error.png",
        "force_uncertainty_vs_error.pdf",
        "reliability.png",
        "reliability.pdf",
        "standardized_residual.png",
        "standardized_residual.pdf",
        "statistics.csv",
    ):
        assert (output / name).is_file()
        assert (output / name).stat().st_size > 0


def test_plotting_does_not_modify_evaluation_details(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    evaluation = _write_evaluation(run_root)
    details = evaluation / "details.npz"
    before_hash = sha256_file(details)
    before_mtime = details.stat().st_mtime_ns

    run_plot(
        PlotConfig(
            run_root=run_root,
            output_root=tmp_path / "plots",
            bin_count=2,
            sample_size=2,
            seed=3,
        )
    )

    assert sha256_file(details) == before_hash
    assert details.stat().st_mtime_ns == before_mtime


def test_plot_output_must_be_outside_formal_run(tmp_path: Path) -> None:
    run_root = tmp_path / "run"

    with pytest.raises(ValueError, match="outside run_root"):
        PlotConfig(run_root=run_root, output_root=run_root / "plots")
