from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import ConfidenceConfig
from Uncertainty_Quantification.ConfidenceHead.confidence_head.run_naming import (
    build_run_name,
    resolve_run_dir,
)


def _config(tmp_path: Path) -> ConfidenceConfig:
    return ConfidenceConfig.model_validate(
        {
            "profile": "smoke",
            "checkpoint": {"path": tmp_path / "model.ckpt", "expected_sha256": "a" * 64},
            "data": {
                split: {
                    "path": tmp_path / f"{split}.xyz",
                    "expected_sha256": f"{index:064x}",
                }
                for index, split in enumerate(("train", "validation", "test"), 1)
            },
            "binning": {
                "force_num_bins": 3,
                "force_max_error": 0.5,
                "energy_num_bins": 3,
                "energy_max_error": 0.3,
            },
            "model": {
                "force": {"hidden_dims": [4], "num_bins": 3},
                "energy": {"hidden_dims": [4], "num_bins": 3, "cumulant_order": 2},
            },
            "loss": {"force_coefficient": 1.0, "energy_coefficient": 1.5},
            "run": {"output_root": tmp_path / "outputs", "name_prefix": "demo"},
        }
    )


def test_run_name_encodes_branch_semantics(tmp_path: Path) -> None:
    assert build_run_name(_config(tmp_path)) == (
        "demo_fixed-linear_f3-fmax0.5-fw1-fmlp4_"
        "e3-emax0.3-ew1.5-emlp4-order2"
    )


def test_run_name_changes_when_energy_head_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    changed = config.model_copy(
        update={
            "model": config.model.model_copy(
                update={
                    "energy": config.model.energy.model_copy(
                        update={"hidden_dims": (9,)}
                    )
                }
            )
        }
    )
    assert build_run_name(changed) != build_run_name(config)


def test_resolve_run_dir_stays_beneath_runs_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert resolve_run_dir(config) == (
        config.run.output_root / "runs" / build_run_name(config)
    ).resolve()
