from __future__ import annotations

# mypy: disable-error-code=func-returns-value
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


def test_shipped_e0_postprocessing_files_are_strict_and_remote_scoped() -> None:
    from confidence_head.e0_config import load_e0_config

    prediction = load_external_config(CONFIGS / "predict_r2scan_e0_gpu.yaml")
    campaign = load_e0_config(CONFIGS / "e0_postprocessing_gpu.yaml")
    submit = ROOT / "run" / "submit_e0_postprocessing.sh"
    text = submit.read_text(encoding="utf-8")

    assert prediction.profile == "postprocessing"
    assert prediction.checkpoint.path == Path(
        "/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt"
    )
    assert prediction.checkpoint.expected_sha256 == (
        "879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004"
    )
    assert prediction.datasets["mad_r2scan_val"].path == Path(
        "/home/bywang/code/UQ/upet_new/data/dataset/mad_val_r2scan_filtered.extxyz"
    )
    assert prediction.datasets["mad_r2scan_val"].expected_sha256 == (
        "4f4d4807592d75cfedda4a157850d60fc1428e44762e8debf37c012a4fc060aa"
    )
    assert prediction.datasets["mad_r2scan_test"].path == Path(
        "/home/bywang/code/UQ/upet_new/data/dataset/mad_test_r2scan_filtered.extxyz"
    )
    assert prediction.datasets["mad_r2scan_test"].expected_sha256 == (
        "499b479499eb56d0792360cb8bcb3397b566ac99c4e290ce0e866382c7f4d2ed"
    )
    assert prediction.runs_root == Path(
        "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
        "ConfidenceHead/outputs/runs"
    )
    assert prediction.cache_root == Path(
        "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
        "ConfidenceHead/outputs/e0_postprocessing/prediction_cache"
    )
    assert prediction.plots_root == Path(
        "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
        "Plots/ConfidenceHead/density_scatter_r2scan_e0"
    )
    assert prediction.cache_batch_size == 2
    assert prediction.batch_size == 128
    assert prediction.device == "cuda"
    assert campaign.external_config == CONFIGS / "predict_r2scan_e0_gpu.yaml"
    assert campaign.validation_dataset == "mad_r2scan_val"
    assert campaign.test_dataset == "mad_r2scan_test"
    assert campaign.validation_expected.model_dump() == {
        "structures": 16098,
        "atoms": 310432,
        "elements": 89,
    }
    assert campaign.test_expected.model_dump() == {
        "structures": 16072,
        "atoms": 311657,
        "elements": 89,
    }
    assert campaign.output_root == Path(
        "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
        "ConfidenceHead/outputs/e0_postprocessing/mad_r2scan"
    )
    assert campaign.plots_root == Path(
        "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
        "Plots/ConfidenceHead/density_scatter_r2scan_e0"
    )
    assert campaign.plot.grid_size == 160
    assert "/home/bywang/.conda/envs/upet_new/bin/python" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert 'STAGE="${STAGE:-all}"' in text
    assert "postprocess_r2scan_e0.py" in text
    assert text.count(".py") == 1
    assert "train.py" not in text
    assert "wandb" not in text.lower()
    assert "force" not in text.lower()
    assert "order" not in text.lower()
