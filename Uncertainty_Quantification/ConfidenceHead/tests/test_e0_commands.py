from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from confidence_head.e0_calibration import E0Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def _commands():
    return importlib.import_module("confidence_head.workflows.e0_commands")


def _dataset() -> E0Dataset:
    return E0Dataset(
        structure_ids=torch.tensor([0, 1], dtype=torch.int64),
        atomic_numbers=torch.tensor([1, 6], dtype=torch.int64),
        atom_offsets=torch.tensor([0, 1, 2], dtype=torch.int64),
        composition=torch.eye(2, dtype=torch.float64),
        target_energy_r2scan=torch.tensor([-1.0, -2.0], dtype=torch.float64),
        atomization_energy=torch.tensor([0.1, 0.2], dtype=torch.float64),
    )


def _workflow_config(tmp_path: Path):
    settings = object()
    return SimpleNamespace(
        external_config=tmp_path / "external.yaml",
        validation_dataset="mad_r2scan_val",
        test_dataset="mad_r2scan_test",
        validation_expected=SimpleNamespace(structures=2, atoms=2, elements=2),
        test_expected=SimpleNamespace(structures=2, atoms=2, elements=2),
        output_root=tmp_path / "campaign",
        plots_root=tmp_path / "plots",
        plot=SimpleNamespace(settings=lambda: settings),
    )


def _external_config(tmp_path: Path):
    return SimpleNamespace(
        checkpoint=SimpleNamespace(
            path=tmp_path / "model.ckpt",
            expected_sha256="a" * 64,
        ),
        datasets={
            "mad_r2scan_val": SimpleNamespace(
                source="extxyz",
                path=tmp_path / "val.extxyz",
                expected_sha256="b" * 64,
            ),
            "mad_r2scan_test": SimpleNamespace(
                source="extxyz",
                path=tmp_path / "test.extxyz",
                expected_sha256="c" * 64,
            ),
        },
        runs_root=tmp_path / "outputs" / "runs",
        cache_root=tmp_path / "cache",
        batch_size=128,
        device="cuda",
    )


def _runs(tmp_path: Path):
    return SimpleNamespace(
        force=tmp_path / "outputs" / "runs" / "force-only",
        energy_by_order={
            order: tmp_path / "outputs" / "runs" / f"energy-{order}"
            for order in range(1, 9)
        },
    )


def test_predict_builds_two_shared_caches_and_runs_energy_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _commands()
    config = _workflow_config(tmp_path)
    external = _external_config(tmp_path)
    runs = _runs(tmp_path)
    built: list[str] = []
    calls: list[tuple[Path, str, object, int, str]] = []
    monkeypatch.setattr(commands, "load_external_config", lambda path: external)
    monkeypatch.setattr(
        commands, "discover_energy_runs", lambda root: runs.energy_by_order
    )

    def build(_config: object, name: str) -> object:
        built.append(name)
        return SimpleNamespace(name=name)

    def predict(
        run: Path,
        name: str,
        cache: object,
        *,
        batch_size: int,
        device_name: str,
    ) -> Path:
        calls.append((run, name, cache, batch_size, device_name))
        return run / "predictions" / name

    monkeypatch.setattr(commands, "build_external_dataset_cache", build)
    monkeypatch.setattr(commands, "predict_dataset_run", predict)

    outputs = commands.predict_e0_inputs(config)

    assert built == ["mad_r2scan_val", "mad_r2scan_test"]
    assert len(outputs) == len(calls) == 16
    assert {call[0] for call in calls} == set(runs.energy_by_order.values())
    assert all(call[0] != runs.force for call in calls)
    assert [call[1] for call in calls[:8]] == ["mad_r2scan_val"] * 8
    assert [call[1] for call in calls[8:]] == ["mad_r2scan_test"] * 8
    assert len({id(call[2]) for call in calls[:8]}) == 1
    assert len({id(call[2]) for call in calls[8:]}) == 1
    assert all(call[3:] == (128, "cuda") for call in calls)


def test_postprocess_preflights_inputs_and_loads_checkpoint_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _commands()
    config = _workflow_config(tmp_path)
    external = _external_config(tmp_path)
    runs = _runs(tmp_path)
    dataset = _dataset()
    seen_sha: list[Path] = []
    verified: list[Path] = []
    loaded_datasets: list[tuple[Path, str, tuple[int, ...]]] = []
    checkpoint_calls: list[tuple[Path, str, torch.device, torch.dtype]] = []
    published: list[object] = []
    expected_by_path = {
        external.checkpoint.path: "a" * 64,
        external.datasets["mad_r2scan_val"].path: "b" * 64,
        external.datasets["mad_r2scan_test"].path: "c" * 64,
    }
    monkeypatch.setattr(commands, "load_external_config", lambda path: external)
    monkeypatch.setattr(
        commands, "discover_energy_runs", lambda root: runs.energy_by_order
    )

    def digest(path: Path) -> str:
        path = Path(path)
        seen_sha.append(path)
        return expected_by_path[path]

    monkeypatch.setattr(commands, "sha256_file", digest)
    monkeypatch.setattr(
        commands,
        "verify_external_prediction",
        lambda path, full=True: verified.append(Path(path)) or {"status": "complete"},
    )

    def load_checkpoint(path: Path, expected_sha256: str, device, dtype):
        checkpoint_calls.append((path, expected_sha256, device, dtype))
        return SimpleNamespace(model=object(), sha256=expected_sha256)

    monkeypatch.setattr(commands, "load_upet_checkpoint", load_checkpoint)
    monkeypatch.setattr(
        commands,
        "extract_model_e0",
        lambda model: (
            (1, 6),
            torch.tensor([-1.0, -2.0], dtype=torch.float64),
        ),
    )

    def load_dataset(path: Path, *, expected_sha256: str, atomic_types):
        loaded_datasets.append((path, expected_sha256, tuple(atomic_types)))
        return dataset

    monkeypatch.setattr(commands, "load_e0_dataset", load_dataset)
    monkeypatch.setattr(
        commands,
        "publish_e0_campaign",
        lambda inputs, root: published.append(inputs) or root,
    )

    result = commands.postprocess_e0_from_config(config)

    assert result == config.output_root
    assert seen_sha == list(expected_by_path)
    assert len(verified) == 16
    assert len(checkpoint_calls) == 1
    assert checkpoint_calls[0][2:] == (torch.device("cpu"), torch.float32)
    assert loaded_datasets == [
        (external.datasets["mad_r2scan_val"].path, "b" * 64, (1, 6)),
        (external.datasets["mad_r2scan_test"].path, "c" * 64, (1, 6)),
    ]
    inputs = published[0]
    assert set(inputs.validation_sources) == set(range(1, 9))
    assert set(inputs.test_sources) == set(range(1, 9))
    assert all("force-only" not in str(path) for path in inputs.test_sources.values())


def test_plot_requires_complete_campaign_before_density_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _commands()
    config = _workflow_config(tmp_path)
    calls: list[tuple[str, Path]] = []
    settings = config.plot.settings()
    monkeypatch.setattr(
        commands,
        "verify_e0_campaign",
        lambda path, full=True: (
            calls.append(("verify", Path(path))) or {"status": "complete"}
        ),
    )
    monkeypatch.setattr(
        commands,
        "publish_e0_density_campaign",
        lambda campaign, output, value: (
            calls.append(("publish", Path(campaign))) or output
        ),
    )

    result = commands.plot_e0_from_config(config)

    assert result == config.plots_root
    assert calls == [
        ("verify", config.output_root / "manifest.json"),
        ("publish", config.output_root),
    ]
    assert settings is config.plot.settings()


def test_dispatch_all_runs_predict_postprocess_plot_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commands = _commands()
    config = _workflow_config(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        commands,
        "predict_e0_inputs",
        lambda value: calls.append("predict") or (Path("prediction"),),
    )
    monkeypatch.setattr(
        commands,
        "postprocess_e0_from_config",
        lambda value: calls.append("postprocess") or Path("campaign"),
    )
    monkeypatch.setattr(
        commands,
        "plot_e0_from_config",
        lambda value: calls.append("plot") or Path("plots"),
    )

    assert commands.dispatch_e0_stage(config, "all") == (
        Path("prediction"),
        Path("campaign"),
        Path("plots"),
    )
    assert calls == ["predict", "postprocess", "plot"]
    calls.clear()
    assert commands.dispatch_e0_stage(config, "plot") == (Path("plots"),)
    assert calls == ["plot"]
    with pytest.raises(ValueError, match="stage"):
        commands.dispatch_e0_stage(config, "train")


def test_cli_requires_config_and_dispatches_selected_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = SCRIPTS / "postprocess_r2scan_e0.py"
    spec = importlib.util.spec_from_file_location("postprocess_r2scan_e0", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tmp_path / "e0.yaml"
    config_path.write_text("output_root: campaign\n", encoding="utf-8")
    loaded = object()
    seen: list[tuple[object, str]] = []
    monkeypatch.setattr(
        importlib.import_module("confidence_head.e0_config"),
        "load_e0_config",
        lambda path: loaded,
    )
    monkeypatch.setattr(
        _commands(),
        "dispatch_e0_stage",
        lambda config, stage: seen.append((config, stage)) or (Path("result"),),
    )

    assert module.main(["--config", str(config_path), "--stage", "postprocess"]) == 0
    assert seen == [(loaded, "postprocess")]
    with pytest.raises(SystemExit):
        module.main(["--stage", "plot"])
