from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from confidence_head.external_config import load_external_config
from confidence_head.workflows import external_commands as commands


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
SUBMIT = ROOT / "run" / "submit_external_prediction.sh"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), SCRIPTS / name
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predict_script_delegates_selected_datasets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("predict_external_datasets.py")
    config_path = tmp_path / "external.yaml"
    config_path.write_text("profile: smoke\n", encoding="utf-8")
    loaded = object()
    seen: list[tuple[object, tuple[str, ...]]] = []
    monkeypatch.setattr(
        importlib.import_module("confidence_head.external_config"),
        "load_external_config",
        lambda path: loaded,
    )
    monkeypatch.setattr(
        commands,
        "predict_external_from_config",
        lambda config, names=(): seen.append((config, names)) or (Path("/result"),),
    )

    assert module.main(["--config", str(config_path), "--dataset", "mad_test"]) == 0
    assert seen == [(loaded, ("mad_test",))]


def test_plot_script_only_dispatches_completed_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("plot_prediction_datasets.py")
    config_path = tmp_path / "external.yaml"
    config_path.write_text("profile: smoke\n", encoding="utf-8")
    loaded = object()
    seen: list[tuple[object, tuple[str, ...]]] = []
    monkeypatch.setattr(
        importlib.import_module("confidence_head.external_config"),
        "load_external_config",
        lambda path: loaded,
    )
    monkeypatch.setattr(
        commands,
        "plot_external_from_config",
        lambda config, names=(): (
            seen.append((config, names)) or (Path("/plots/manifest.json"),)
        ),
    )

    assert module.main(["--config", str(config_path)]) == 0
    assert seen == [(loaded, ())]


def test_plot_workflow_uses_evaluation_and_dataset_scoped_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordered = tuple(tmp_path / "runs" / f"run-{index}" for index in range(9))
    runs = SimpleNamespace(ordered=lambda: ordered)
    config = SimpleNamespace(
        datasets={
            "matpes_test": SimpleNamespace(source="existing_evaluation"),
            "mad_test": SimpleNamespace(source="extxyz"),
        },
        runs_root=tmp_path / "runs",
        plots_root=tmp_path / "plots",
    )
    loaded_sources: list[tuple[str, tuple[Path, ...]]] = []
    publications: list[SimpleNamespace] = []
    monkeypatch.setattr(commands, "discover_external_runs", lambda root: runs)

    def load(sources: tuple[Path, ...], name: str):
        loaded_sources.append((name, tuple(sources)))
        return SimpleNamespace(name=name)

    monkeypatch.setattr(commands, "load_dataset_series", load)
    monkeypatch.setattr(
        commands,
        "plot_dataset_suite",
        lambda dataset, root: (
            publications.append(
                SimpleNamespace(
                    dataset=dataset, manifest=root / dataset.name / "manifest.json"
                )
            )
            or publications[-1]
        ),
    )
    monkeypatch.setattr(
        commands,
        "plot_cross_dataset_correlations",
        lambda values, root: pytest.fail(
            "two selected datasets must not make cross plots"
        ),
    )

    manifests = commands.plot_external_from_config(config)

    assert loaded_sources[0] == (
        "matpes_test",
        tuple(run / "evaluation" for run in ordered),
    )
    assert loaded_sources[1] == (
        "mad_test",
        tuple(run / "predictions" / "mad_test" for run in ordered),
    )
    assert manifests == tuple(item.manifest for item in publications)


def test_shipped_external_configs_are_strict_and_release_scoped() -> None:
    assert sorted(path.name for path in CONFIGS.glob("predict_external*.yaml")) == [
        "predict_external_gpu.yaml",
        "predict_external_smoke.yaml",
    ]
    production = load_external_config(CONFIGS / "predict_external_gpu.yaml")
    smoke = load_external_config(CONFIGS / "predict_external_smoke.yaml")

    assert set(production.datasets) == {"matpes_test", "matpes_train", "mad_test"}
    assert set(smoke.datasets) == {"smoke"}
    assert production.device == "cuda"
    assert smoke.device == "cpu"


def test_slurm_template_only_runs_prediction_or_plotting() -> None:
    text = SUBMIT.read_text(encoding="utf-8")
    assert "train.py" not in text
    assert "wandb" not in text.lower()
    assert "predict_external_datasets.py" in text
    assert "plot_prediction_datasets.py" in text
    assert "STAGE" in text
    assert set(
        yaml.safe_load((CONFIGS / "predict_external_gpu.yaml").read_text())["datasets"]
    ) == {
        "matpes_test",
        "matpes_train",
        "mad_test",
    }
