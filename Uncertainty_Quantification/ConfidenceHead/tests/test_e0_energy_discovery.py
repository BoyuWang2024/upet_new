from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from confidence_head.external_prediction import discover_external_runs
from confidence_head.workflows import e0_commands


def _energy_only_runs_root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs" / "runs"
    for order in range(1, 9):
        run = root / f"energy-{order}"
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "run_id": f"energy-{order}",
                }
            ),
            encoding="utf-8",
        )
        (run / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "loss": {
                        "force_coefficient": 0.0,
                        "energy_coefficient": 1.0,
                    },
                    "model": {
                        "force": {
                            "enabled": True,
                            "target_mode": "atom_mean",
                        },
                        "energy": {
                            "enabled": True,
                            "cumulant_order": order,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
    return root


def test_e0_inputs_discover_energy_runs_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = _energy_only_runs_root(tmp_path)
    external = SimpleNamespace(
        datasets={
            "mad_r2scan_val": SimpleNamespace(source="extxyz"),
            "mad_r2scan_test": SimpleNamespace(source="extxyz"),
        },
        runs_root=runs_root,
    )
    config = SimpleNamespace(
        external_config=tmp_path / "external.yaml",
        validation_dataset="mad_r2scan_val",
        test_dataset="mad_r2scan_test",
    )
    monkeypatch.setattr(
        e0_commands,
        "load_external_config",
        lambda path: external,
    )

    loaded, energy_by_order = e0_commands._load_inputs(config)

    assert loaded is external
    assert set(energy_by_order) == set(range(1, 9))
    assert all(
        path.name == f"energy-{order}" for order, path in energy_by_order.items()
    )


def test_existing_external_discovery_still_requires_force(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="force-only run was not found"):
        discover_external_runs(_energy_only_runs_root(tmp_path))
