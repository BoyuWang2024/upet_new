from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_member(path: Path, energy: np.ndarray) -> Path:
    np.savez(
        path,
        energy=energy,
        forces=np.repeat(energy[:, None], 3, axis=1),
        stress=np.repeat(energy[:, None, None], 9, axis=1).reshape(-1, 3, 3),
    )
    return path


def test_sample_std_uses_ddof_one(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        streaming_mean_std,
    )

    paths = [
        _write_member(tmp_path / "a.npz", np.array([1.0])),
        _write_member(tmp_path / "b.npz", np.array([3.0])),
    ]
    result = streaming_mean_std(paths, "energy")
    assert result.member_count == 2
    assert result.mean == pytest.approx(np.array([2.0]))
    assert result.std == pytest.approx(np.array([np.sqrt(2.0)]))
    assert result.mean.dtype == np.float64


def test_gmd_excludes_self_and_counts_unordered_pairs(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        pairwise_gmd,
    )

    paths = [
        _write_member(tmp_path / "a.npz", np.array([0.0])),
        _write_member(tmp_path / "b.npz", np.array([2.0])),
        _write_member(tmp_path / "c.npz", np.array([5.0])),
    ]
    assert pairwise_gmd(paths, "energy") == pytest.approx(np.array([10.0 / 3.0]))


def test_scalar_rms_reductions_follow_atom_layout() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        scalar_rms_reductions,
    )

    num_atoms = np.array([2, 1])
    offsets = np.array([0, 2, 3])
    energy = scalar_rms_reductions(
        np.array([4.0, 3.0]), "energy", num_atoms=num_atoms, atom_offsets=offsets
    )
    forces = scalar_rms_reductions(
        np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 6.0]]),
        "forces",
        num_atoms=num_atoms,
        atom_offsets=offsets,
    )
    stress = scalar_rms_reductions(
        np.stack([np.eye(3), np.ones((3, 3))]),
        "stress",
        num_atoms=num_atoms,
        atom_offsets=offsets,
    )
    assert energy == pytest.approx(np.array([2.0, 3.0]))
    assert forces == pytest.approx(np.array([np.sqrt(25 / 3), 1.0, np.sqrt(12)]))
    assert stress == pytest.approx(np.array([np.sqrt(1 / 3), 1.0]))


def test_compute_store_uncertainty_requires_publication_api() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import uncertainty

    assert callable(uncertainty.compute_store_uncertainty)


_UNITS = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}


def _write_prediction_split(
    root: Path, *, dataset_key: str, target_stress: bool
) -> Path:
    """Create a tiny canonical prediction store for publication tests."""
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        PredictionArrays,
        PredictionStore,
        TargetArrays,
    )

    store = PredictionStore(root, split=dataset_key, units=_UNITS)
    store.write_targets(
        TargetArrays(
            structure_ids=np.array(["s0", "s1"]),
            num_atoms=np.array([1, 2]),
            atom_offsets=np.array([0, 1, 3]),
            energy=np.array([0.0, 0.0]),
            forces=np.zeros((3, 3)),
            stress=np.zeros((2, 3, 3)) if target_stress else None,
        )
    )
    for member_index, value in enumerate((1.0, 3.0)):
        store.write_member(
            member_index,
            "raw",
            PredictionArrays(
                energy=np.full(2, value),
                forces=np.full((3, 3), value),
                stress=np.full((2, 3, 3), value),
            ),
        )
    return root / dataset_key


def test_generic_mad_test_without_target_stress_publishes_predicted_stress_uq(
    tmp_path: Path,
) -> None:
    """Catches UQ reductions that incorrectly require reference target stress."""
    from Uncertainty_Quantification.BootStrapping.bootstrap.uq_publication import (
        compute_uncertainty_results,
    )

    split_root = _write_prediction_split(
        tmp_path / "predictions", dataset_key="mad_test", target_stress=False
    )
    results = compute_uncertainty_results(split_root, mode="raw", member_count=2)
    assert results["stress_std"].shape == (2, 3, 3)
    assert results["stress_std"].dtype == np.float64
    assert results["stress_tensor_rms_std"].shape == (2,)


def test_generic_publication_is_manifest_last_and_exactly_auditable(
    tmp_path: Path,
) -> None:
    """Catches noncanonical manifest timing and extra-file audit gaps."""
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.uq_publication import (
        compute_uncertainty_results,
        publish_uncertainty_results,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.validation import (
        validate_uq_publication,
    )

    split_root = _write_prediction_split(
        tmp_path / "predictions", dataset_key="mad_test", target_stress=False
    )
    results = compute_uncertainty_results(split_root, mode="raw", member_count=2)
    publication_root = tmp_path / "uncertainty" / "mad_test" / "raw"
    publication = publish_uncertainty_results(
        publication_root,
        results,
        dataset_key="mad_test",
        mode="raw",
        member_count=2,
        units=_UNITS,
    )
    assert publication.results_path == publication_root / "results.npz"
    assert publication.manifest_path == publication_root / "manifest.json"
    assert sorted(path.name for path in publication_root.iterdir()) == [
        "manifest.json",
        "results.npz",
    ]
    validate_uq_publication(tmp_path, split="mad_test", mode="raw", member_count=2)
    (publication_root / "unexpected.txt").write_text("not canonical", encoding="utf-8")
    with pytest.raises(HardFailure, match="exact"):
        validate_uq_publication(tmp_path, split="mad_test", mode="raw", member_count=2)


def test_compute_store_uncertainty_rejects_unsafe_generic_key(tmp_path: Path) -> None:
    """Catches traversal-capable dataset keys before any publication path is used."""
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.uq_publication import (
        compute_store_uncertainty,
    )

    with pytest.raises(HardFailure, match="lowercase letters"):
        compute_store_uncertainty(
            tmp_path / "predictions",
            tmp_path / "uncertainty",
            split="../escape",
            mode="raw",
            member_count=2,
            units=_UNITS,
        )
