from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import pytest
import torch
import yaml

from Uncertainty_Quantification.ConfidenceHead.confidence_head import plot_analysis


Target = Literal["energy", "force"]


def _write_plot_run(
    tmp_path: Path,
    *,
    target: Target,
    order: int = 1,
    num_bins: int = 50,
    observed: torch.Tensor | None = None,
    expected: torch.Tensor | None = None,
    force_target_mode: str = "atom_mean",
    force_coefficient: float | None = None,
    energy_coefficient: float | None = None,
) -> Path:
    run_dir = tmp_path / f"run-{target}-order{order}"
    evaluation_dir = run_dir / "evaluation"
    evaluation_dir.mkdir(parents=True)

    if force_coefficient is None:
        force_coefficient = 1.0 if target == "force" else 0.0
    if energy_coefficient is None:
        energy_coefficient = 1.0 if target == "energy" else 0.0
    config = {
        "binning": {
            "algorithm": "fixed_linear_v1",
            "force_max_error": 0.5,
            "energy_max_error": 0.3,
        },
        "model": {
            "force": {
                "enabled": True,
                "num_bins": num_bins,
                "target_mode": force_target_mode,
            },
            "energy": {
                "enabled": True,
                "num_bins": num_bins,
                "cumulant_order": order,
            },
        },
        "loss": {
            "force_coefficient": force_coefficient,
            "energy_coefficient": energy_coefficient,
        },
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )

    force_representatives = torch.linspace(0.005, 0.495, num_bins)
    energy_representatives = torch.linspace(0.003, 0.297, num_bins)
    binning = {
        "force": {
            "algorithm": "fixed_linear_v1",
            "num_bins": num_bins,
            "max_error": 0.5,
            "target_mode": force_target_mode,
            "error_definition": "abs_cartesian_component_mean_v1",
            "thresholds": torch.linspace(0.01, 0.49, num_bins - 1).tolist(),
            "representatives": force_representatives.tolist(),
        },
        "energy": {
            "algorithm": "fixed_linear_v1",
            "num_bins": num_bins,
            "max_error": 0.3,
            "thresholds": torch.linspace(0.006, 0.294, num_bins - 1).tolist(),
            "representatives": energy_representatives.tolist(),
        },
    }
    (run_dir / "binning.json").write_text(json.dumps(binning), encoding="utf-8")

    if observed is None:
        observed = torch.tensor([1.0, 2.0, 3.0, 4.0])
    if expected is None:
        expected = torch.tensor([2.0, 1.0, 4.0, 3.0])
    logits = torch.full((len(observed), num_bins), -10.0)
    predicted_bins = torch.tensor([0, 1, 1, num_bins - 1])
    logits[torch.arange(len(observed)), predicted_bins] = 10.0
    representatives = (
        force_representatives if target == "force" else energy_representatives
    )
    predictions: dict[str, object] = {
        "structure_ids": torch.tensor([11, 12, 13, 14]),
        f"{target}_logits": logits,
        f"{target}_labels": predicted_bins,
        f"{target}_observed_errors": observed,
        f"{target}_expected_errors": expected,
        f"{target}_representatives": representatives,
        "atom_offsets": torch.arange(5),
    }
    if target == "force":
        predictions["force_target_mode"] = force_target_mode
        predictions["force_error_definition"] = "abs_cartesian_component_mean_v1"
    torch.save(predictions, evaluation_dir / "test_predictions.pt")

    metrics = {
        target: {
            "sample_count": len(observed),
            "pearson": 0.6,
            "spearman": 0.6,
        }
    }
    (evaluation_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir


@pytest.fixture(autouse=True)
def _accept_synthetic_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plot_analysis, "verify_run", lambda run_dir, full, **kwargs: {})


def test_load_energy_preserves_per_atom_observations(tmp_path: Path) -> None:
    run_dir = _write_plot_run(tmp_path, target="energy", order=7)

    series = plot_analysis.load_plot_series(run_dir)

    assert series.target == "energy"
    assert series.order == 7
    assert series.force_target_mode is None
    assert series.structure_ids.tolist() == [11, 12, 13, 14]
    assert series.observed.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_load_force_keeps_one_atom_mean_sample_per_atom(tmp_path: Path) -> None:
    run_dir = _write_plot_run(tmp_path, target="force")

    series = plot_analysis.load_plot_series(run_dir)

    assert series.target == "force"
    assert series.order is None
    assert series.force_target_mode == "atom_mean"
    assert series.observed.shape == (4,)
    assert series.observed.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_load_rejects_component_force_results(tmp_path: Path) -> None:
    run_dir = _write_plot_run(
        tmp_path,
        target="force",
        force_target_mode="component",
    )

    with pytest.raises(ValueError, match="atom_mean"):
        plot_analysis.load_plot_series(run_dir)


def test_load_requires_exactly_one_active_target(tmp_path: Path) -> None:
    run_dir = _write_plot_run(
        tmp_path,
        target="energy",
        force_coefficient=1.0,
        energy_coefficient=1.0,
    )

    with pytest.raises(ValueError, match="exactly one active target"):
        plot_analysis.load_plot_series(run_dir)


def test_load_rejects_non_50_bin_results(tmp_path: Path) -> None:
    run_dir = _write_plot_run(tmp_path, target="energy", num_bins=49)

    with pytest.raises(ValueError, match="50"):
        plot_analysis.load_plot_series(run_dir)


def test_load_rejects_non_finite_errors(tmp_path: Path) -> None:
    run_dir = _write_plot_run(
        tmp_path,
        target="energy",
        observed=torch.tensor([1.0, 2.0, float("nan"), 4.0]),
    )

    with pytest.raises(ValueError, match="finite"):
        plot_analysis.load_plot_series(run_dir)


def test_bin_rows_include_all_50_bins_and_empty_rows(tmp_path: Path) -> None:
    series = plot_analysis.load_plot_series(_write_plot_run(tmp_path, target="energy"))

    rows = plot_analysis.bin_rows(series)

    assert len(rows) == 50
    assert rows[0].sample_count == 1
    assert rows[0].mean_observed_error == pytest.approx(1.0)
    assert rows[1].sample_count == 2
    assert rows[1].median_observed_error == pytest.approx(2.5)
    assert rows[1].std_observed_error == pytest.approx(0.5)
    assert rows[2].sample_count == 0
    assert rows[2].mean_observed_error is None
    assert rows[2].mean_expected_error is None
    assert rows[49].sample_count == 1
    assert sum(row.sample_count for row in rows) == 4


def test_energy_correlation_matches_hand_calculated_values(tmp_path: Path) -> None:
    series = plot_analysis.load_plot_series(
        _write_plot_run(tmp_path, target="energy", order=3)
    )

    row = plot_analysis.energy_correlation(series)

    assert row.order == 3
    assert row.sample_count == 4
    assert row.pearson == pytest.approx(0.6)
    assert row.spearman == pytest.approx(0.6)


def test_energy_correlation_rejects_stored_metric_disagreement(
    tmp_path: Path,
) -> None:
    run_dir = _write_plot_run(tmp_path, target="energy")
    metrics_path = run_dir / "evaluation" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["energy"]["pearson"] = 0.7
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    series = plot_analysis.load_plot_series(run_dir)

    with pytest.raises(ValueError, match="pearson"):
        plot_analysis.energy_correlation(series)


def test_energy_correlation_rejects_constant_expected_errors(
    tmp_path: Path,
) -> None:
    run_dir = _write_plot_run(
        tmp_path,
        target="energy",
        expected=torch.ones(4),
    )
    series = plot_analysis.load_plot_series(run_dir)

    with pytest.raises(ValueError, match="constant"):
        plot_analysis.energy_correlation(series)
