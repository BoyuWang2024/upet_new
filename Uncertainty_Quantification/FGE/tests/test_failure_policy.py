"""Fault-injection coverage for the formal result failure policy."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.tests.test_validation import make_canonical_result


def _prediction(root: Path) -> dict[str, Any]:
    return torch.load(
        root / "prediction" / "test_raw.pt", weights_only=True, map_location="cpu"
    )


def _save_prediction(root: Path, payload: dict[str, Any]) -> None:
    path = root / "prediction" / "test_raw.pt"
    torch.save(payload, path)
    manifest_path = root / "prediction" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    import hashlib

    manifest["artifact"]["bytes"] = path.stat().st_size
    manifest["artifact"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _corrupt_hash(root: Path) -> None:
    manifest_path = root / "prediction" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _corrupt_member_order(root: Path) -> None:
    manifest_path = root / "prediction" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["member_ids"] = list(reversed(manifest["member_ids"]))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _corrupt_dtype(root: Path) -> None:
    payload = _prediction(root)
    payload["energy_prediction"] = payload["energy_prediction"].to(torch.float64)
    _save_prediction(root, payload)


def _corrupt_shape(root: Path) -> None:
    payload = _prediction(root)
    payload["stress_prediction"] = torch.zeros((2, 1, 9), dtype=torch.float32)
    _save_prediction(root, payload)


def _corrupt_mapping(root: Path) -> None:
    payload = _prediction(root)
    payload["structure_mapping"] = torch.tensor([1], dtype=torch.int64)
    _save_prediction(root, payload)


def _corrupt_reference(root: Path) -> None:
    payload = _prediction(root)
    payload["energy_reference"] = torch.zeros(1, dtype=torch.float32)
    _save_prediction(root, payload)


def _corrupt_formula(root: Path) -> None:
    path = root / "evaluation" / "legacy_equal_weight" / "uncertainty.pt"
    artifact = torch.load(path, weights_only=True, map_location="cpu")
    artifact["formula_version"] = "wrong_formula"
    torch.save(artifact, path)


def _corrupt_forbidden_path(root: Path) -> None:
    path = root / "training" / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["source_path"] = "/old/FGE/outputs/result"
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _corrupt_nonfinite(root: Path) -> None:
    payload = _prediction(root)
    energies = payload["energy_prediction"].clone()
    energies[0, 0] = math.nan
    payload["energy_prediction"] = energies
    _save_prediction(root, payload)


@pytest.mark.parametrize(
    "corrupt",
    [
        _corrupt_hash,
        _corrupt_member_order,
        _corrupt_dtype,
        _corrupt_shape,
        _corrupt_mapping,
        _corrupt_reference,
        _corrupt_formula,
        _corrupt_forbidden_path,
        _corrupt_nonfinite,
    ],
)
def test_validator_hard_fails_each_independent_contract_corruption(
    tmp_path: Path, corrupt: Callable[[Path], None]
) -> None:
    """No formal corruption is downgraded, skipped, or silently repaired."""
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    corrupt(root)

    with pytest.raises(HardFailure):
        validate_result(config, root)
