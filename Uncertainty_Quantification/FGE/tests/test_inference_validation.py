from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.artifacts import sha256_file
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.inference_evaluation import (
    evaluate_inference_dataset,
)
from Uncertainty_Quantification.FGE.fge.inference_only import (
    predict_inference_dataset,
)
from Uncertainty_Quantification.FGE.fge.inference_validation import (
    validate_inference_result,
)
from Uncertainty_Quantification.FGE.tests.test_inference_only import (
    LiteralChunkRuntime,
    _config,
)


def _completed(tmp_path: Path) -> Path:
    config = _config(tmp_path)
    predict_inference_dataset(config, runtime=LiteralChunkRuntime())
    evaluate_inference_dataset(config)
    return config.output.root


def _set_dataset_field(root: Path, key: str, value: object) -> None:
    run_path = root / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    for relative in (
        "prediction/manifest.json",
        "uncertainty/manifest.json",
    ):
        path = root / relative
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest["dataset_identity"][key] = value
        path.write_text(json.dumps(manifest), encoding="utf-8")
        for artifact in run["artifacts"]:
            if artifact["path"] == relative:
                artifact["sha256"] = sha256_file(path)
                artifact["bytes"] = path.stat().st_size
    run["dataset_identity"][key] = value
    run_path.write_text(json.dumps(run), encoding="utf-8")


def test_read_only_validation_recomputes_and_never_mutates_files(
    tmp_path: Path,
) -> None:
    root = _completed(tmp_path)
    before = {
        path.relative_to(root): (
            path.read_bytes(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }

    report = validate_inference_result(root)

    after = {
        path.relative_to(root): (
            path.read_bytes(),
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }
    assert report.status == "PASS"
    assert report.mode == "read_only"
    assert after == before


def test_validation_rejects_metrics_not_from_global_recomputation(
    tmp_path: Path,
) -> None:
    root = _completed(tmp_path)
    path = root / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["domains"]["energy"]["pearson_log10"] = 0.123
    path.write_text(json.dumps(metrics), encoding="utf-8")

    with pytest.raises(HardFailure, match="metrics"):
        validate_inference_result(root)


def test_validation_rejects_missing_duplicate_or_noncontiguous_chunks(
    tmp_path: Path,
) -> None:
    root = _completed(tmp_path)
    path = root / "uncertainty" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["chunks"][1] = dict(manifest["chunks"][0])
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HardFailure, match="chunk"):
        validate_inference_result(root)


def test_validation_rejects_prediction_hash_and_formula_version(
    tmp_path: Path,
) -> None:
    root = _completed(tmp_path)
    run_path = root / "run_manifest.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run["formula_version"] = "sample_std_forbidden"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(HardFailure, match="formula"):
        validate_inference_result(root)


def test_validation_rejects_reference_availability_not_bound_to_chunks(
    tmp_path: Path,
) -> None:
    root = _completed(tmp_path)
    _set_dataset_field(
        root,
        "reference_availability",
        {"energy": True, "forces": True, "stress": False},
    )

    with pytest.raises(HardFailure, match="reference availability"):
        validate_inference_result(root)


def test_validation_rejects_atom_count_not_covered_by_chunks(tmp_path: Path) -> None:
    root = _completed(tmp_path)
    run = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    _set_dataset_field(root, "atom_count", run["dataset_identity"]["atom_count"] + 1)

    with pytest.raises(HardFailure, match="cover the dataset"):
        validate_inference_result(root)


def test_validation_rejects_unlisted_residue(tmp_path: Path) -> None:
    root = _completed(tmp_path)
    (root / "debug.log").write_text("residue", encoding="utf-8")

    with pytest.raises(HardFailure, match="inventory|residue"):
        validate_inference_result(root)
