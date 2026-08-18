from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pytest

from confidence_head.workflows import density_commands


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_density_script_forwards_config_and_selected_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "plot_density_scatter", SCRIPTS / "plot_density_scatter.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = tmp_path / "density.yaml"
    config_path.write_text("output_root: plots\n", encoding="utf-8")
    loaded = object()
    seen: list[tuple[object, tuple[str, ...]]] = []
    monkeypatch.setattr(
        importlib.import_module("confidence_head.density_config"),
        "load_density_config",
        lambda path: loaded,
    )
    monkeypatch.setattr(
        density_commands,
        "plot_density_from_config",
        lambda config, names=(): (
            seen.append((config, names)) or (Path("/plots/manifest.json"),)
        ),
    )

    assert module.main(
        ["--config", str(config_path), "--dataset", "mad_test"]
    ) == 0
    assert seen == [(loaded, ("mad_test",))]
