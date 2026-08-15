from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    atomic_json_dump,
    atomic_npz_save,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr.plot_multi import (
    PlotConfig,
    PlotEvaluationConfig,
    PlotStyleConfig,
    analyze_panel,
    filter_log_pairs,
    run_plot,
    shared_square_log_limits,
)


def _write_evaluation(root: Path, identity: str, scale: float) -> PlotEvaluationConfig:
    evaluation = root / "evaluation/cal" / identity
    arrays = {
        "energy_residual": scale * np.array([0.1, -0.2, 0.3, -0.4]),
        "energy_calibrated_std": scale * np.array([0.15, 0.25, 0.35, 0.45]),
        "force_residual": scale * np.array([0.2, -0.1, 0.4, -0.3, 0.5, -0.6]),
        "force_calibrated_std_component": scale
        * np.array([0.25, 0.15, 0.45, 0.35, 0.55, 0.65]),
    }
    details = evaluation / "details.npz"
    summary = evaluation / "summary.json"
    atomic_npz_save(details, arrays)
    atomic_json_dump(summary, {"structure_count": 4})
    atomic_json_dump(
        evaluation / "manifest.json",
        {
            "status": "complete",
            "identity": identity,
            "files": {
                "details.npz": sha256_file(details),
                "summary.json": sha256_file(summary),
            },
        },
    )
    return PlotEvaluationConfig(
        label=root.name, run_root=root, evaluation_identity=identity
    )


def _style() -> PlotStyleConfig:
    return PlotStyleConfig(grid_size=24, dpi=40)


def test_filtering_limits_and_analysis_are_deterministic() -> None:
    first = (np.array([0.1, 0.2, np.nan]), np.array([0.2, 0.4, 1.0]))
    second = (np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    filtered = filter_log_pairs(*first)
    limits = shared_square_log_limits((first, second), margin=0.05)

    assert filtered.valid_count == 2
    assert filtered.excluded == {"nan": 1, "inf": 0, "zero": 0, "negative": 0}
    assert limits[0] < -1.0
    assert limits[1] > np.log10(4.0)

    uncertainty = np.geomspace(0.01, 1.0, 20)
    error = uncertainty[::-1]
    kwargs = {
        "log_limits": (-2.1, 0.1),
        "grid_size": 24,
        "gaussian_sigma": 1.2,
        "contour_masses": (0.5, 0.7, 0.85, 0.95, 0.99),
        "scatter_max_points": 10,
        "random_seed": 7,
        "coverage_thresholds": (),
    }
    left = analyze_panel(uncertainty, error, **kwargs)
    right = analyze_panel(uncertainty, error, **kwargs)
    np.testing.assert_array_equal(left.scatter_indices, right.scatter_indices)
    assert left.statistics == right.statistics
    assert left.statistics["spearman_log"] == pytest.approx(-1.0)


def test_run_plot_publishes_exactly_fourteen_verified_files(tmp_path: Path) -> None:
    evaluations = tuple(
        _write_evaluation(tmp_path / label, f"eval-{index}", scale)
        for index, (label, scale) in enumerate(
            (("matpes_test", 1.0), ("mad_test", 10.0), ("matpes_train", 100.0))
        )
    )
    output = tmp_path / "plots"
    assert (
        run_plot(
            PlotConfig(evaluations=evaluations, output_root=output, style=_style())
        )
        == output
    )

    expected = {
        *(
            f"llpr_{label}_{target}_uncertainty_vs_residual.{suffix}"
            for label in ("matpes_test", "mad_test", "matpes_train")
            for target in ("energy", "force")
            for suffix in ("png", "pdf")
        ),
        "plotting_statistics.csv",
        "plotting_manifest.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert all(path.stat().st_size > 0 for path in output.iterdir())

    manifest = json.loads((output / "plotting_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert set(manifest["evaluation_identities"]) == {
        "matpes_test",
        "mad_test",
        "matpes_train",
    }
    assert set(manifest["files"]) == expected - {"plotting_manifest.json"}
    for name, digest in manifest["files"].items():
        assert sha256_file(output / name) == digest
    assert str(tmp_path.resolve()) not in json.dumps(manifest)

    with (output / "plotting_statistics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert {(row["dataset"], row["target"]) for row in rows} == {
        (label, target)
        for label in ("matpes_test", "mad_test", "matpes_train")
        for target in ("energy", "force")
    }
    for target in ("energy", "force"):
        limits = {
            (row["log_limit_low"], row["log_limit_high"])
            for row in rows
            if row["target"] == target
        }
        assert len(limits) == 1


def test_plotting_is_read_only_and_validates_configuration(tmp_path: Path) -> None:
    evaluation = _write_evaluation(tmp_path / "matpes_test", "eval", 1.0)
    details = tmp_path / "matpes_test/evaluation/cal/eval/details.npz"
    before = (sha256_file(details), details.stat().st_mtime_ns)
    run_plot(
        PlotConfig(
            evaluations=(evaluation,), output_root=tmp_path / "plots", style=_style()
        )
    )
    assert (sha256_file(details), details.stat().st_mtime_ns) == before

    with pytest.raises(ValueError, match="unique"):
        PlotConfig(
            evaluations=(evaluation, evaluation), output_root=tmp_path / "plots-2"
        )
    with pytest.raises(ValueError, match="label"):
        PlotEvaluationConfig(label="../unsafe", run_root=tmp_path / "one")
    with pytest.raises(ValueError, match="outside"):
        PlotConfig(
            evaluations=(evaluation,),
            output_root=tmp_path / "matpes_test/plots",
        )


def test_existing_destination_survives_failed_render(tmp_path: Path) -> None:
    output = tmp_path / "plots"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("old")
    evaluation = _write_evaluation(tmp_path / "bad", "eval", 1.0)
    details = tmp_path / "bad/evaluation/cal/eval/details.npz"
    with np.load(details) as archive:
        arrays = {
            name: archive[name] for name in archive.files if name != "force_residual"
        }
    atomic_npz_save(details, arrays)
    manifest_path = details.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["details.npz"] = sha256_file(details)
    atomic_json_dump(manifest_path, manifest)

    with pytest.raises(KeyError, match="force_residual"):
        run_plot(
            PlotConfig(evaluations=(evaluation,), output_root=output, style=_style())
        )
    assert marker.read_text() == "old"
    assert {path.name for path in output.iterdir()} == {"keep.txt"}
