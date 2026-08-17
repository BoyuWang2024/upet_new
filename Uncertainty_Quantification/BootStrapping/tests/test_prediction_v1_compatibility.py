from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_prediction_publication import _write_existing_v1_test_publication

from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction_publication import (
    validate_prediction_publication,
)


def test_v1_targets_without_structure_limit_mean_complete_dataset(
    tmp_path: Path,
) -> None:
    _write_existing_v1_test_publication(tmp_path)
    root = tmp_path / "predictions" / "test"
    manifest = root / "manifest.json"
    document = json.loads(manifest.read_text(encoding="utf-8"))
    del document["targets"]["structure_limit"]
    manifest.write_text(json.dumps(document), encoding="utf-8")

    audit = validate_prediction_publication(
        root,
        dataset_key="test",
        mode="raw",
        member_count=2,
        reference_targets=("energy", "forces", "stress"),
        structure_limit=None,
    )

    assert audit.manifest_path == manifest
    with pytest.raises(HardFailure, match="structure_limit"):
        validate_prediction_publication(
            root,
            dataset_key="test",
            mode="raw",
            member_count=2,
            reference_targets=("energy", "forces", "stress"),
            structure_limit=1,
        )
