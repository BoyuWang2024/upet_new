from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from Uncertainty_Quantification.LLPR.llpr import calibration, curvature
from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    load_verified_manifest,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr.calibration import (
    resolve_calibration_stage,
)
from Uncertainty_Quantification.LLPR.llpr.config import LLPRConfig, load_llpr_config
from Uncertainty_Quantification.LLPR.llpr.curvature import resolve_curvature_stage


ConfigWriter = Callable[[dict[str, Any] | None], Path]
CURVATURE_ID = "a" * 16
CALIBRATION_ID = "b" * 16


def _write_curvature_stage(path: Path, config: LLPRConfig) -> None:
    path.mkdir(parents=True)
    matrix_path = path / "curvature.npz"
    diagnostics_path = path / "diagnostics.json"
    atomic_npz_save(
        matrix_path,
        {"energy": np.eye(2, dtype=np.float64), "force": np.eye(3)},
    )
    atomic_json_dump(
        diagnostics_path,
        {
            "energy_dimension": 2,
            "force_dimension": 3,
            "total_dimension": 5,
        },
    )
    atomic_json_dump(
        path / "manifest.json",
        {
            "stage": "curvature",
            "status": "complete",
            "identity": CURVATURE_ID,
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
            "layout_hash": "layout",
            "payload": {
                "checkpoint_sha256": config.checkpoint.expected_sha256,
                "curvature": config.curvature.model_dump(mode="json"),
                "matrix_dtype": config.runtime.matrix_dtype,
                "jacobian_backend": config.runtime.jacobian_backend,
                "force_component_chunk_size": (
                    config.runtime.force_component_chunk_size
                ),
            },
            "files": {
                matrix_path.name: sha256_file(matrix_path),
                diagnostics_path.name: sha256_file(diagnostics_path),
            },
        },
    )


def _write_calibration_stage(
    path: Path,
    config: LLPRConfig,
    *,
    curvature_identity: str = CURVATURE_ID,
) -> None:
    path.mkdir(parents=True)
    candidates_path = path / "candidates.json"
    summary_path = path / "summary.json"
    atomic_json_dump(candidates_path, {"energy": [], "force": []})
    atomic_json_dump(summary_path, {"selected": {}})
    atomic_json_dump(
        path / "manifest.json",
        {
            "stage": "calibration",
            "status": "complete",
            "identity": CALIBRATION_ID,
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
            "curvature_identity": curvature_identity,
            "payload": {
                "curvature_identity": curvature_identity,
                "ridge": config.calibration.ridge.model_dump(mode="json"),
            },
            "files": {
                candidates_path.name: sha256_file(candidates_path),
                summary_path.name: sha256_file(summary_path),
            },
        },
    )


def _reuse_config(
    write_llpr_config: ConfigWriter,
    tmp_path: Path,
    *,
    include_calibration: bool,
    calibration_curvature_identity: str = CURVATURE_ID,
) -> LLPRConfig:
    initial_path = write_llpr_config(None)
    initial = load_llpr_config(initial_path)
    curvature_source = tmp_path / "sources/curvature"
    calibration_source = tmp_path / "sources/calibration"
    _write_curvature_stage(curvature_source, initial)
    if include_calibration:
        _write_calibration_stage(
            calibration_source,
            initial,
            curvature_identity=calibration_curvature_identity,
        )

    raw = yaml.safe_load(initial_path.read_text(encoding="utf-8"))
    raw["data"].pop("build")
    raw["output"]["root"] = str(tmp_path / "outputs")
    raw["output"]["experiment"] = "reuse"
    raw["reuse"] = {
        "curvature": {"path": str(curvature_source), "identity": CURVATURE_ID}
    }
    if include_calibration:
        raw["data"].pop("calibration")
        raw["reuse"]["calibration"] = {
            "path": str(calibration_source),
            "identity": CALIBRATION_ID,
        }
    initial_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_llpr_config(initial_path)


def test_curvature_reuse_does_not_call_numerical_build(
    write_llpr_config: ConfigWriter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reuse_config(write_llpr_config, tmp_path, include_calibration=False)

    def fail_numerical_build(unused_config: LLPRConfig) -> Path:
        raise AssertionError("numerical build called")

    monkeypatch.setattr(curvature, "run_build", fail_numerical_build)

    resolved = resolve_curvature_stage(config)

    assert resolved == tmp_path / "outputs/reuse/curvature" / CURVATURE_ID
    assert load_verified_manifest(resolved / "manifest.json", verify_npz=True)


def test_calibration_reuse_does_not_call_numerical_calibration(
    write_llpr_config: ConfigWriter,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _reuse_config(write_llpr_config, tmp_path, include_calibration=True)

    def fail_numerical_calibration(unused_config: LLPRConfig) -> Path:
        raise AssertionError("numerical calibration called")

    monkeypatch.setattr(calibration, "run_calibrate", fail_numerical_calibration)

    resolved = resolve_calibration_stage(config)

    assert resolved == tmp_path / "outputs/reuse/calibration" / CALIBRATION_ID
    manifest = load_verified_manifest(resolved / "manifest.json")
    assert manifest["curvature_identity"] == CURVATURE_ID


def test_calibration_reuse_rejects_different_curvature_binding(
    write_llpr_config: ConfigWriter,
    tmp_path: Path,
) -> None:
    config = _reuse_config(
        write_llpr_config,
        tmp_path,
        include_calibration=True,
        calibration_curvature_identity="c" * 16,
    )

    with pytest.raises(ValueError, match="curvature identity mismatch"):
        resolve_calibration_stage(config)


def test_calibration_reuse_rejects_ridge_payload_mismatch(
    write_llpr_config: ConfigWriter,
    tmp_path: Path,
) -> None:
    config = _reuse_config(write_llpr_config, tmp_path, include_calibration=True)
    assert config.reuse is not None
    assert config.reuse.calibration is not None
    manifest_path = config.reuse.calibration.path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload"]["ridge"]["eta"]["energy"] = 2.0e-6
    atomic_json_dump(manifest_path, manifest)

    with pytest.raises(ValueError, match="payload field 'ridge' mismatch"):
        resolve_calibration_stage(config)
