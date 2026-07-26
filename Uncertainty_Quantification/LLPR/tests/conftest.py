from pathlib import Path
from typing import Any

import pytest
import yaml


@pytest.fixture
def write_llpr_config(tmp_path: Path):
    def write(overrides: dict[str, Any] | None = None) -> Path:
        value: dict[str, Any] = {
            "checkpoint": {
                "path": "data/checkpoint/pet-omatpes-l-v0.1.0.ckpt",
                "expected_sha256": "a" * 64,
            },
            "data": {
                "build": "data/dataset/matpes_n20.extxyz",
                "calibration": "data/dataset/matpes_n20.extxyz",
                "test": "data/dataset/matpes_n20.extxyz",
            },
            "curvature": {
                "energy_loss_weight": 1.0,
                "force_loss_weight": 0.1,
                "energy_huber_delta": 0.015,
                "force_huber_delta": 0.01,
            },
            "calibration": {
                "ridge": {
                    "mode": "fixed",
                    "max_condition_number": 1.0e10,
                    "eta": {"energy": 1.0e-6, "force": 1.0e-6},
                }
            },
            "runtime": {"device": "cpu", "matrix_dtype": "float64"},
            "output": {
                "root": "Uncertainty_Quantification/LLPR/outputs",
                "experiment": "test",
            },
        }
        if overrides:
            value.update(overrides)
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        return path

    return write
