from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from confidence_head.density_config import load_density_config
from confidence_head.workflows import density_commands


def _write_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "external_config: external.yaml",
                "output_root: plots/density_scatter",
                "plot:",
                "  scatter_max_points: 20000",
                "  scatter_seed: 20260714",
                "  grid_size: 160",
                "  gaussian_sigma: 1.2",
                "  contour_masses: [0.50, 0.70, 0.85, 0.95, 0.99]",
                "  log_margin: 0.05",
                "  dpi: 300",
            )
        )
        + "\n",
        encoding="utf-8",
    )


def test_density_config_resolves_paths_and_rejects_unknown_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "density.yaml"
    _write_config(path)
    config = load_density_config(path)
    assert config.external_config == (tmp_path / "external.yaml").absolute()
    assert config.output_root == (tmp_path / "plots/density_scatter").absolute()
    assert config.settings().grid_size == 160

    path.write_text(path.read_text() + "unknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_density_config(path)


def test_density_command_uses_existing_sources_without_prediction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "density.yaml"
    _write_config(path)
    config = load_density_config(path)
    external = SimpleNamespace(
        datasets={
            "matpes_train": SimpleNamespace(source="cache_split"),
            "matpes_test": SimpleNamespace(source="existing_evaluation"),
            "mad_test": SimpleNamespace(source="extxyz"),
        },
        runs_root=tmp_path / "runs",
    )
    runs = SimpleNamespace(
        ordered=lambda: tuple(tmp_path / "runs" / f"run-{index}" for index in range(9))
    )
    seen: list[str] = []
    monkeypatch.setattr(density_commands, "load_external_config", lambda path: external)
    monkeypatch.setattr(density_commands, "discover_external_runs", lambda root: runs)
    monkeypatch.setattr(
        density_commands,
        "load_dataset_series",
        lambda sources, name: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(
        density_commands,
        "publish_density_dataset",
        lambda dataset, root, settings: seen.append(dataset.name)
        or SimpleNamespace(dataset=dataset, manifest=root / dataset.name / "manifest.json"),
    )
    monkeypatch.setattr(
        density_commands,
        "publish_density_cross_dataset",
        lambda publications, settings, root: root / "manifest.json",
    )

    manifests = density_commands.plot_density_from_config(config)

    assert seen == ["matpes_train", "matpes_test", "mad_test"]
    assert len(manifests) == 4
