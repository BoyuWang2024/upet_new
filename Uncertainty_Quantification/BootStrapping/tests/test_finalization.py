from __future__ import annotations

from pathlib import Path

import numpy as np


def test_uq_validation_precedes_run_finalization(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
        atomic_write_json,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.finalization import (
        finalize_run,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
        PredictionStore,
        TargetArrays,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        compute_store_uncertainty,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.validation import (
        validate_uq_publication,
    )

    units = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}
    store = PredictionStore(tmp_path / "predictions", split="test", units=units)
    store.write_targets(
        TargetArrays(
            structure_ids=np.array(["s0"]),
            num_atoms=np.array([1]),
            atom_offsets=np.array([0, 1]),
            energy=np.zeros(1),
            forces=np.zeros((1, 3)),
            stress=np.zeros((1, 3, 3)),
        )
    )
    for index, value in enumerate((1.0, 3.0)):
        store.write_member(
            index,
            "raw",
            PredictionArrays(
                energy=np.full(1, value),
                forces=np.full((1, 3), value),
                stress=np.full((1, 3, 3), value),
            ),
        )
    compute_store_uncertainty(
        tmp_path / "predictions",
        tmp_path / "uncertainty",
        split="test",
        mode="raw",
        member_count=2,
        units=units,
    )
    atomic_write_json(
        tmp_path / "run_manifest.json",
        {
            "schema": "upet.bootstrap.run/v1",
            "stages": {"uncertainty": "pending"},
        },
    )

    validate_uq_publication(tmp_path, split="test", mode="raw", member_count=2)
    manifest = finalize_run(
        tmp_path, splits=("test",), modes=("raw",), member_count=2
    )

    assert manifest["stages"]["uncertainty"] == "complete"
    assert manifest["uncertainty_manifests"] == ["uncertainty/test/raw/manifest.json"]
