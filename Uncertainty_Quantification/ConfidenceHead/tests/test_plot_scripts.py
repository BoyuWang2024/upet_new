from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str) -> ModuleType:
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plot_module() -> ModuleType:
    return importlib.import_module("confidence_head.plot_analysis")


def test_single_plot_script_resolves_run_and_fixed_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script("plot_argmax_bin_boxplots.py")
    plotting = _plot_module()
    run_dir = tmp_path / "run"
    series = object()
    captured: dict[str, object] = {}

    def load(path: Path) -> object:
        captured["run_dir"] = path
        return series

    def plot(value: object, output_dir: Path) -> tuple[Path, ...]:
        captured["series"] = value
        captured["output_dir"] = output_dir
        return (output_dir / "one.png",)

    monkeypatch.setattr(plotting, "load_plot_series", load)
    monkeypatch.setattr(plotting, "plot_single_boxplot", plot)

    assert module.main(["--run-dir", str(run_dir)]) == 0
    assert captured == {
        "run_dir": run_dir.resolve(),
        "series": series,
        "output_dir": run_dir.resolve() / "plots" / "argmax_bin_boxplots",
    }
    assert capsys.readouterr().out.strip().endswith("one.png")


def test_energy_correlation_script_requires_and_forwards_eight_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("plot_energy_order_correlations.py")
    plotting = _plot_module()
    run_dirs = tuple(tmp_path / f"order-{order}" for order in range(1, 9))
    output_dir = tmp_path / "correlations"
    loaded = [object() for _ in range(8)]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        plotting,
        "load_plot_series",
        lambda path: loaded[run_dirs.index(path)],
    )

    def correlations(series: dict[int, object]) -> tuple[str, ...]:
        captured["series"] = series
        return ("rows",)

    def plot(rows: tuple[str, ...], root: Path) -> tuple[Path, ...]:
        captured["rows"] = rows
        captured["output_dir"] = root
        return (root / "correlations.csv",)

    monkeypatch.setattr(
        plotting,
        "energy_series_by_order",
        lambda values: {order: value for order, value in enumerate(values, 1)},
    )
    monkeypatch.setattr(plotting, "validate_comparable_energy", correlations)
    monkeypatch.setattr(plotting, "plot_energy_correlations", plot)

    arguments = [
        item for run_dir in run_dirs for item in ("--run-dir", str(run_dir))
    ] + ["--output-dir", str(output_dir)]
    assert module.main(arguments) == 0
    assert captured == {
        "series": {order: loaded[order - 1] for order in range(1, 9)},
        "rows": ("rows",),
        "output_dir": output_dir.resolve(),
    }


def test_completed_plot_script_forwards_explicit_run_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("plot_completed_runs.py")
    plotting = _plot_module()
    runs_root = tmp_path / "runs"
    output_root = tmp_path / "outputs"
    explicit = tuple(tmp_path / f"selected-{index}" for index in range(9))
    completed = object()
    captured: dict[str, object] = {}

    def discover(
        root: Path,
        *,
        explicit_run_dirs: tuple[Path, ...],
    ) -> object:
        captured["runs_root"] = root
        captured["explicit"] = explicit_run_dirs
        return completed

    def plot(value: object, *, comparisons_dir: Path) -> tuple[Path, ...]:
        captured["completed"] = value
        captured["comparisons_dir"] = comparisons_dir
        return (comparisons_dir / "summary.pdf",)

    monkeypatch.setattr(plotting, "discover_completed_runs", discover)
    monkeypatch.setattr(plotting, "plot_completed_runs", plot)
    arguments = [
        "--runs-root",
        str(runs_root),
        "--output-root",
        str(output_root),
        *[item for run_dir in explicit for item in ("--run-dir", str(run_dir))],
    ]

    assert module.main(arguments) == 0
    assert captured == {
        "runs_root": runs_root.resolve(),
        "explicit": tuple(path.resolve() for path in explicit),
        "completed": completed,
        "comparisons_dir": output_root.resolve() / "comparisons",
    }
