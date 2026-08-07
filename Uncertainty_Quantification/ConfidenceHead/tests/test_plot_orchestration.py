from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head import plot_analysis


def _series(
    run_dir: Path,
    *,
    target: plot_analysis.Target,
    order: int | None = None,
) -> plot_analysis.PlotSeries:
    logits = torch.full((4, 50), -10.0, dtype=torch.float64)
    logits[torch.arange(4), torch.tensor([0, 1, 1, 49])] = 10.0
    return plot_analysis.PlotSeries(
        run_dir=run_dir.resolve(),
        structure_ids=torch.tensor([11, 12, 13, 14]),
        target=target,
        order=order,
        logits=logits,
        observed=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        expected=torch.tensor([2.0, 1.0, 4.0, 3.0], dtype=torch.float64),
        representatives=torch.linspace(0.003, 0.297, 50, dtype=torch.float64),
        stored_metrics={"sample_count": 4, "pearson": 0.6, "spearman": 0.6},
        force_target_mode="atom_mean" if target == "force" else None,
    )


def _candidate_runs(
    tmp_path: Path,
) -> tuple[dict[Path, plot_analysis.PlotSeries], Path]:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    candidates: dict[Path, plot_analysis.PlotSeries] = {}
    force_dir = runs_root / "force"
    candidates[force_dir.resolve()] = _series(force_dir, target="force")
    for order in range(1, 9):
        run_dir = runs_root / f"energy-order{order}"
        candidates[run_dir.resolve()] = _series(
            run_dir,
            target="energy",
            order=order,
        )
    for run_dir in candidates:
        (run_dir / "evaluation").mkdir(parents=True)
        (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
        (run_dir / "evaluation" / "manifest.json").write_text("{}", encoding="utf-8")
    return candidates, runs_root


def _use_candidates(
    monkeypatch: pytest.MonkeyPatch,
    candidates: dict[Path, plot_analysis.PlotSeries],
) -> None:
    def load(run_dir: Path) -> plot_analysis.PlotSeries:
        return candidates[Path(run_dir).resolve()]

    monkeypatch.setattr(plot_analysis, "load_plot_series", load)


def test_discover_completed_runs_requires_exact_nine_run_coverage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, runs_root = _candidate_runs(tmp_path)
    _use_candidates(monkeypatch, candidates)

    runs = plot_analysis.discover_completed_runs(runs_root)

    assert runs.force.target == "force"
    assert set(runs.energy_by_order) == set(range(1, 9))


def test_discover_completed_runs_rejects_missing_energy_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, runs_root = _candidate_runs(tmp_path)
    missing = (runs_root / "energy-order8").resolve()
    del candidates[missing]
    (missing / "evaluation" / "manifest.json").unlink()
    _use_candidates(monkeypatch, candidates)

    with pytest.raises(ValueError, match="missing energy orders.*8"):
        plot_analysis.discover_completed_runs(runs_root)


def test_discover_completed_runs_rejects_duplicate_energy_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, runs_root = _candidate_runs(tmp_path)
    duplicate = runs_root / "energy-order4-repeat"
    (duplicate / "evaluation").mkdir(parents=True)
    (duplicate / "manifest.json").write_text("{}", encoding="utf-8")
    (duplicate / "evaluation" / "manifest.json").write_text("{}", encoding="utf-8")
    candidates[duplicate.resolve()] = _series(
        duplicate,
        target="energy",
        order=4,
    )
    _use_candidates(monkeypatch, candidates)

    with pytest.raises(ValueError, match="duplicate energy order 4"):
        plot_analysis.discover_completed_runs(runs_root)


def test_discover_completed_runs_rejects_two_force_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, runs_root = _candidate_runs(tmp_path)
    duplicate = runs_root / "force-repeat"
    (duplicate / "evaluation").mkdir(parents=True)
    (duplicate / "manifest.json").write_text("{}", encoding="utf-8")
    (duplicate / "evaluation" / "manifest.json").write_text("{}", encoding="utf-8")
    candidates[duplicate.resolve()] = _series(duplicate, target="force")
    _use_candidates(monkeypatch, candidates)

    with pytest.raises(ValueError, match="exactly one force run"):
        plot_analysis.discover_completed_runs(runs_root)


def test_discover_completed_runs_ignores_unrelated_incomplete_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates, runs_root = _candidate_runs(tmp_path)
    (runs_root / "failed-history").mkdir()
    _use_candidates(monkeypatch, candidates)

    runs = plot_analysis.discover_completed_runs(runs_root)

    assert len(runs.energy_by_order) == 8


def test_plot_completed_runs_rejects_incomparable_energy_before_writing(
    tmp_path: Path,
) -> None:
    candidates, _ = _candidate_runs(tmp_path)
    force = next(series for series in candidates.values() if series.target == "force")
    energy = {
        series.order: series
        for series in candidates.values()
        if series.target == "energy" and series.order is not None
    }
    energy[8] = replace(
        energy[8],
        observed=torch.tensor([1.0, 2.0, 3.0, 5.0], dtype=torch.float64),
    )
    runs = plot_analysis.CompletedRuns(force=force, energy_by_order=energy)
    comparisons = tmp_path / "outputs" / "comparisons"

    with pytest.raises(ValueError, match="observed energy errors differ"):
        plot_analysis.plot_completed_runs(runs, comparisons_dir=comparisons)

    assert not comparisons.exists()
    assert not any(
        series.run_dir.joinpath("plots").exists() for series in energy.values()
    )


def test_plot_completed_runs_writes_all_single_and_comparison_outputs(
    tmp_path: Path,
) -> None:
    candidates, _ = _candidate_runs(tmp_path)
    force = next(series for series in candidates.values() if series.target == "force")
    energy = {
        series.order: series
        for series in candidates.values()
        if series.target == "energy" and series.order is not None
    }
    runs = plot_analysis.CompletedRuns(force=force, energy_by_order=energy)
    comparisons = tmp_path / "outputs" / "comparisons"

    artifacts = plot_analysis.plot_completed_runs(runs, comparisons_dir=comparisons)

    assert len(artifacts) == 32
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts)
    assert (
        comparisons / "energy_correlations" / "linear_order_correlations_no_ci.csv"
    ).is_file()
    assert (
        comparisons / "argmax_bin_boxplots" / "combined_energy_argmax_bin_boxplots.pdf"
    ).is_file()
    assert (
        force.run_dir
        / "plots"
        / "argmax_bin_boxplots"
        / "test_force_argmax_bin_statistics.csv"
    ).is_file()
