from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize(
    "script",
    ("predict_dataset.py", "evaluate_dataset.py", "plot_dataset.py"),
)
def test_dataset_cli_runs_directly_and_accepts_only_config(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    for forbidden in ("run-all", "train", "checkpoint-sweep", "runtime"):
        assert forbidden not in result.stdout


@pytest.mark.parametrize(
    ("module_name", "callback_name"),
    (
        ("predict_dataset", "predict_inference_dataset"),
        ("evaluate_dataset", "evaluate_inference_dataset"),
    ),
)
def test_inference_cli_calls_only_its_named_stage(
    module_name: str,
    callback_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        f"Uncertainty_Quantification.FGE.scripts.{module_name}"
    )
    config = object()
    calls: list[object] = []
    monkeypatch.setattr(module, "load_inference_config", lambda _: config)
    monkeypatch.setattr(module, callback_name, lambda value: calls.append(value))

    assert module.main(["--config", "input.yaml"]) == 0
    assert calls == [config]


def test_plot_cli_reopens_analyzes_and_renders_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module(
        "Uncertainty_Quantification.FGE.scripts.plot_dataset"
    )
    config = type(
        "Config",
        (),
        {
            "input_kind": "inference",
            "input_root": Path("input"),
            "output_root": Path("output"),
            "settings": object(),
        },
    )()
    plot_input = object()
    analysis = object()
    calls: list[tuple[str, object]] = []

    def load(value: object) -> object:
        calls.append(("load", value))
        return plot_input

    def analyze(value: object, settings: object) -> object:
        del settings
        calls.append(("analyze", value))
        return analysis

    monkeypatch.setattr(module, "load_plot_config", lambda _: config)
    monkeypatch.setattr(
        module,
        "load_inference_plot_input",
        load,
    )
    monkeypatch.setattr(
        module,
        "analyze_plot_input",
        analyze,
    )
    monkeypatch.setattr(
        module,
        "render_plot_suite",
        lambda value, root: calls.append(("render", root)),
    )

    assert module.main(["--config", "plot.yaml"]) == 0
    assert calls == [
        ("load", Path("input")),
        ("analyze", plot_input),
        ("render", Path("output")),
    ]


def test_formal_configs_bind_verified_inputs_without_sweep_fields() -> None:
    from Uncertainty_Quantification.FGE.fge.inference_config import (
        load_inference_config,
    )
    from Uncertainty_Quantification.FGE.fge.plot_analysis import load_plot_config

    mad = load_inference_config(CONFIGS / "inference_mad_test.yaml")
    train = load_inference_config(CONFIGS / "inference_matpes_train.yaml")
    plots = [
        load_plot_config(CONFIGS / f"plot_{label}.yaml")
        for label in ("matpes_test", "mad_test", "matpes_train")
    ]

    assert (
        mad.dataset.expected_sha256
        == "d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b"
    )
    assert mad.dataset.reference_availability["stress"] is False
    assert (
        train.dataset.expected_sha256
        == "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec"
    )
    assert train.dataset.reference_availability["stress"] is True
    assert all(plot.settings.dpi == 300 for plot in plots)
    assert {plot.input_kind for plot in plots} == {"completed_fge", "inference"}


def test_public_package_exports_dataset_pipeline() -> None:
    from Uncertainty_Quantification.FGE import fge

    for name in (
        "load_inference_config",
        "predict_inference_dataset",
        "evaluate_inference_dataset",
        "validate_inference_result",
        "load_plot_config",
        "analyze_plot_input",
        "render_plot_suite",
    ):
        assert hasattr(fge, name)
