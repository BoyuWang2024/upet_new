from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


_UNITS = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}


def _write_prediction_split(root: Path, dataset_key: str) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
        PredictionStore,
        TargetArrays,
    )

    store = PredictionStore(root, split=dataset_key, units=_UNITS)
    store.write_targets(
        TargetArrays(
            structure_ids=np.array(["s0"]),
            num_atoms=np.array([1]),
            atom_offsets=np.array([0, 1]),
            energy=np.zeros(1),
            forces=np.zeros((1, 3)),
            stress=None,
        )
    )
    for member_index, value in enumerate((1.0, 3.0)):
        store.write_member(
            member_index,
            "raw",
            PredictionArrays(
                energy=np.full(1, value),
                forces=np.full((1, 3), value),
                stress=np.full((1, 3, 3), value),
            ),
        )


def _campaign(run_roots: tuple[Path, ...], dataset_keys: tuple[str, ...]):
    from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
        CampaignDataset,
        CampaignPrediction,
        CampaignRun,
    )

    return SimpleNamespace(
        runs=tuple(
            CampaignRun(
                label=f"run_{index}",
                config_path=root / "config.yaml",
                run_root=root,
                config=SimpleNamespace(),
            )
            for index, root in enumerate(run_roots)
        ),
        datasets=tuple(
            CampaignDataset(
                label=key,
                storage_key=key,
                path=Path(f"/{key}"),
                reference_targets=("energy", "forces"),
            )
            for key in dataset_keys
        ),
        prediction=CampaignPrediction(
            mode="raw", member_count=2, device="cpu", batch_size=1
        ),
    )


def _publish_verified_test(run_root: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uq_publication import (
        compute_uncertainty_results,
        publish_uncertainty_results,
    )

    results = compute_uncertainty_results(
        run_root / "predictions" / "test", mode="raw", member_count=2
    )
    publish_uncertainty_results(
        run_root / "uncertainty" / "test" / "raw",
        results,
        dataset_key="test",
        mode="raw",
        member_count=2,
        units=_UNITS,
    )


def test_campaign_reuses_verified_test_without_recomputing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches recomputation despite a verified raw test publication."""

    import Uncertainty_Quantification.BootStrapping.bootstrap.uq_campaign as uq_campaign

    run_root = tmp_path / "run"
    _write_prediction_split(run_root / "predictions", "test")
    _publish_verified_test(run_root)

    def _must_not_compute(*args: object, **kwargs: object) -> dict[str, np.ndarray]:
        raise AssertionError("complete UQ publication was recomputed")

    monkeypatch.setattr(uq_campaign, "compute_uncertainty_results", _must_not_compute)
    result = uq_campaign.compute_campaign_uq(
        _campaign((run_root,), ("test",)), run_labels=None, dataset_labels=None
    )

    assert len(result) == 1
    assert result[0].skipped is True
    assert result[0].destination == run_root / "uncertainty" / "test" / "raw"


def test_campaign_manifest_failure_leaves_no_destination_or_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches manifest failures that leak output or staging."""

    import Uncertainty_Quantification.BootStrapping.bootstrap.uq_campaign as uq_campaign
    from Uncertainty_Quantification.BootStrapping.bootstrap import (
        uq_publication as publication,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    run_root = tmp_path / "run"
    _write_prediction_split(run_root / "predictions", "mad_test")

    def _fail_manifest(*args: object, **kwargs: object) -> Path:
        raise HardFailure("simulated manifest write failure")

    monkeypatch.setattr(publication, "atomic_write_json", _fail_manifest)
    with pytest.raises(HardFailure, match="simulated manifest"):
        uq_campaign.compute_campaign_uq(
            _campaign((run_root,), ("mad_test",)),
            run_labels=None,
            dataset_labels=None,
        )

    parent = run_root / "uncertainty" / "mad_test"
    assert not (parent / "raw").exists()
    assert not list(parent.glob(".raw.*.staging"))


def test_campaign_rejects_partial_destination_without_clobbering(
    tmp_path: Path,
) -> None:
    """Catches overwriting of a partial existing UQ publication."""

    import Uncertainty_Quantification.BootStrapping.bootstrap.uq_campaign as uq_campaign
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    run_root = tmp_path / "run"
    _write_prediction_split(run_root / "predictions", "test")
    destination = run_root / "uncertainty" / "test" / "raw"
    destination.mkdir(parents=True)
    sentinel = destination / "results.npz"
    sentinel.write_bytes(b"partial")

    with pytest.raises(HardFailure):
        uq_campaign.compute_campaign_uq(
            _campaign((run_root,), ("test",)), run_labels=None, dataset_labels=None
        )

    assert sentinel.read_bytes() == b"partial"
    assert not list(destination.parent.glob(".raw.*.staging"))


def test_campaign_selection_keeps_declaration_order(tmp_path: Path) -> None:
    """Catches selector handling that changes the deterministic declaration order."""

    import Uncertainty_Quantification.BootStrapping.bootstrap.uq_campaign as uq_campaign

    first = tmp_path / "first"
    second = tmp_path / "second"
    for run_root in (first, second):
        _write_prediction_split(run_root / "predictions", "test")

    result = uq_campaign.compute_campaign_uq(
        _campaign((first, second), ("test",)),
        run_labels=("run_1", "run_0"),
        dataset_labels=("test",),
    )

    assert [(item.run_label, item.dataset_label) for item in result] == [
        ("run_0", "test"),
        ("run_1", "test"),
    ]
