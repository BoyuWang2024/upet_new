from __future__ import annotations

import gc
import json
import weakref
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
    atomic_write_json,
    sha256_file,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
    CampaignConfig,
    CampaignDataset,
    CampaignPlotStyle,
    CampaignPrediction,
    CampaignRun,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
    PredictionArrays,
    PredictionStore,
    TargetArrays,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.uq_publication import (
    compute_uncertainty_results,
    publish_uncertainty_results,
)


_UNITS = {
    "energy": "eV",
    "forces": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
}
_RUN_LABELS = ("run_z", "run_a", "run_m")
_DATASET_SPECS = (
    ("matpes_test", "test", ("energy", "forces", "stress")),
    ("mad_test", "mad_test", ("energy", "forces")),
    ("matpes_train", "matpes_train", ("energy", "forces", "stress")),
)
_MEMBER_COUNT = 8
_COEFFICIENTS = np.array(
    [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float64
) / np.sqrt(24.0)
_ENERGY_MEAN = np.array([4.0, 2.0], dtype=np.float64)
_ENERGY_STD = np.array([2.0, 0.5], dtype=np.float64)
_ENERGY_REFERENCE = np.array([3.5, 1.25], dtype=np.float64)
_FORCE_MEAN = np.array(
    [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
    dtype=np.float64,
)
_FORCE_STD = np.array(
    [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
    dtype=np.float64,
)
_FORCE_RESIDUAL = np.array(
    [[0.05, 0.10, 0.15], [0.20, 0.25, 0.30], [0.35, 0.40, 0.45]],
    dtype=np.float64,
)
_STRESS_MEAN = np.array(
    [
        [[10.0, 2.0, 4.0], [6.0, 20.0, 8.0], [12.0, 14.0, 30.0]],
        [[-3.0, 5.0, 7.0], [1.0, -4.0, 9.0], [11.0, 13.0, -5.0]],
    ],
    dtype=np.float64,
)
_STRESS_DIRECTION = np.array(
    [
        [[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0], [-7.0, -8.0, 9.0]],
        [[2.0, -3.0, 4.0], [5.0, -6.0, -7.0], [8.0, 9.0, 10.0]],
    ],
    dtype=np.float64,
)
_STRESS_REFERENCE = np.array(
    [
        [[9.0, 1.0, 5.0], [5.0, 18.0, 7.0], [9.0, 13.0, 27.0]],
        [[-2.0, 4.0, 8.0], [0.0, -7.0, 6.0], [10.0, 16.0, -1.0]],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class CampaignArtifacts:
    campaign: CampaignConfig
    prediction_manifests: dict[tuple[str, str], Path]
    uq_manifests: dict[tuple[str, str], Path]
    targets_paths: dict[tuple[str, str], Path]


def _targets(*, with_stress: bool) -> TargetArrays:
    return TargetArrays(
        structure_ids=np.array(["s0", "s1"]),
        num_atoms=np.array([2, 1], dtype=np.int64),
        atom_offsets=np.array([0, 2, 3], dtype=np.int64),
        energy=_ENERGY_REFERENCE.copy(),
        forces=_FORCE_MEAN + _FORCE_RESIDUAL,
        stress=_STRESS_REFERENCE.copy() if with_stress else None,
    )


def _member_values(member_index: int) -> PredictionArrays:
    coefficient = _COEFFICIENTS[member_index]
    return PredictionArrays(
        energy=_ENERGY_MEAN + coefficient * _ENERGY_STD,
        forces=_FORCE_MEAN + coefficient * _FORCE_STD,
        stress=_STRESS_MEAN + coefficient * _STRESS_DIRECTION,
    )


def _write_prediction_publication(
    run_root: Path,
    *,
    dataset_label: str,
    storage_key: str,
    reference_targets: tuple[str, ...],
) -> tuple[Path, Path]:
    split_root = run_root / "predictions" / storage_key
    store = PredictionStore.at_split_root(split_root, split=storage_key, units=_UNITS)
    targets = _targets(with_stress="stress" in reference_targets)
    store.write_targets(targets)
    member_records: list[dict[str, object]] = []
    for member_index in range(_MEMBER_COUNT):
        publication = store.write_member(
            member_index, "raw", _member_values(member_index)
        )
        member_records.append(
            {
                "member_index": member_index,
                "mode": "raw",
                "path": publication.path.relative_to(split_root).as_posix(),
                "sha256": publication.sha256,
                "shapes": publication.shapes,
                "dtypes": publication.dtypes,
            }
        )
    document: dict[str, object] = {
        "schema": (
            "upet.bootstrap.predictions/v1"
            if targets.stress is not None
            else "upet.bootstrap.predictions/v2"
        ),
        "split": storage_key,
        "units": _UNITS,
        "member_count": _MEMBER_COUNT,
        "targets": {
            "structure_limit": None,
            "path": "targets.npz",
            "sha256": sha256_file(store.targets_path),
        },
        "members": member_records,
    }
    if targets.stress is None:
        document.update(
            {
                "dataset_key": storage_key,
                "dataset_label": dataset_label,
                "reference_targets": list(reference_targets),
            }
        )
    manifest = atomic_write_json(split_root / "manifest.json", document)
    return manifest, store.targets_path


def _write_uq_publication(run_root: Path, storage_key: str) -> Path:
    results = compute_uncertainty_results(
        run_root / "predictions" / storage_key,
        mode="raw",
        member_count=_MEMBER_COUNT,
    )
    publication = publish_uncertainty_results(
        run_root / "uncertainty" / storage_key / "raw",
        results,
        dataset_key=storage_key,
        mode="raw",
        member_count=_MEMBER_COUNT,
        units=_UNITS,
    )
    return publication.manifest_path


def _build_campaign_artifacts(tmp_path: Path) -> CampaignArtifacts:
    runs: list[CampaignRun] = []
    prediction_manifests: dict[tuple[str, str], Path] = {}
    uq_manifests: dict[tuple[str, str], Path] = {}
    targets_paths: dict[tuple[str, str], Path] = {}
    for run_label in _RUN_LABELS:
        run_root = tmp_path / "runs" / run_label
        runs.append(
            CampaignRun(
                label=run_label,
                config_path=tmp_path / f"{run_label}.yaml",
                run_root=run_root,
                config=SimpleNamespace(),
            )
        )
        for dataset_label, storage_key, reference_targets in _DATASET_SPECS:
            key = (run_label, dataset_label)
            manifest, targets_path = _write_prediction_publication(
                run_root,
                dataset_label=dataset_label,
                storage_key=storage_key,
                reference_targets=reference_targets,
            )
            prediction_manifests[key] = manifest
            targets_paths[key] = targets_path
            uq_manifests[key] = _write_uq_publication(run_root, storage_key)

    datasets = tuple(
        CampaignDataset(
            label=label,
            storage_key=storage_key,
            path=tmp_path / f"{label}.extxyz",
            reference_targets=reference_targets,
        )
        for label, storage_key, reference_targets in _DATASET_SPECS
    )
    campaign = CampaignConfig(
        schema_version=1,
        runs=tuple(runs),
        datasets=datasets,
        prediction=CampaignPrediction(
            mode="raw", member_count=_MEMBER_COUNT, device="cpu", batch_size=2
        ),
        plot=CampaignPlotStyle(
            grid_size=32,
            gaussian_sigma=1.0,
            contour_masses=(0.5, 0.9),
            scatter_max_points=100,
            scatter_seed=2026,
            scatter_size=4.0,
            scatter_alpha=0.7,
            log_margin=0.05,
            figure_size=(4.0, 4.0),
            dpi=72,
            formats=("png", "pdf"),
        ),
        output_root=tmp_path / "plots",
        source_path=tmp_path / "campaign.yaml",
    )
    return CampaignArtifacts(
        campaign=campaign,
        prediction_manifests=prediction_manifests,
        uq_manifests=uq_manifests,
        targets_paths=targets_paths,
    )


@pytest.fixture
def campaign_artifacts(tmp_path: Path) -> CampaignArtifacts:
    return _build_campaign_artifacts(tmp_path)


def _source(sources: tuple[object, ...], run: str, dataset: str, target: str):
    return next(
        source
        for source in sources
        if (
            source.key.run_label,
            source.key.dataset_label,
            source.key.target,
        )
        == (run, dataset, target)
    )


def _literal_symmetric_voigt(values: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    return np.stack(
        (
            symmetric[..., 0, 0],
            symmetric[..., 1, 1],
            symmetric[..., 2, 2],
            symmetric[..., 1, 2],
            symmetric[..., 0, 2],
            symmetric[..., 0, 1],
        ),
        axis=-1,
    )


def test_discover_sources_builds_24_panels_in_campaign_order_and_audits_once(
    campaign_artifacts: CampaignArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_source

    prediction_calls: list[tuple[str, str]] = []
    uq_calls: list[tuple[str, str]] = []
    original_prediction_audit = plot_source.validate_prediction_publication
    original_uq_audit = plot_source.validate_uq_publication

    def prediction_audit(split_root: Path, **kwargs: object):
        prediction_calls.append(
            (Path(split_root).parents[1].name, str(kwargs["dataset_key"]))
        )
        return original_prediction_audit(split_root, **kwargs)

    def uq_audit(run_root: Path, **kwargs: object):
        uq_calls.append((Path(run_root).name, str(kwargs["split"])))
        return original_uq_audit(run_root, **kwargs)

    monkeypatch.setattr(
        plot_source, "validate_prediction_publication", prediction_audit
    )
    monkeypatch.setattr(plot_source, "validate_uq_publication", uq_audit)

    sources = plot_source.discover_plot_sources(campaign_artifacts.campaign)

    expected_keys = []
    for run_label in _RUN_LABELS:
        for dataset_label, _storage_key, reference_targets in _DATASET_SPECS:
            targets = ("energy", "force", "stress")
            if "stress" not in reference_targets:
                targets = targets[:2]
            expected_keys.extend(
                (run_label, dataset_label, target) for target in targets
            )
    assert [
        (source.key.run_label, source.key.dataset_label, source.key.target)
        for source in sources
    ] == expected_keys
    assert len(sources) == 24
    assert Counter(source.key.target for source in sources) == {
        "energy": 9,
        "force": 9,
        "stress": 6,
    }
    assert not any(
        source.key.dataset_label == "mad_test" and source.key.target == "stress"
        for source in sources
    )

    expected_audits = Counter(
        (run_label, storage_key)
        for run_label in _RUN_LABELS
        for _, storage_key, _ in _DATASET_SPECS
    )
    assert Counter(prediction_calls) == expected_audits
    assert Counter(uq_calls) == expected_audits
    assert len(prediction_calls) == len(uq_calls) == 9


def test_sources_preserve_eight_member_order_and_manifest_sha_provenance(
    campaign_artifacts: CampaignArtifacts,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
    )

    sources = discover_plot_sources(campaign_artifacts.campaign)
    for source in sources:
        run = next(
            item
            for item in campaign_artifacts.campaign.runs
            if item.label == source.key.run_label
        )
        expected_members = tuple(
            run.run_root
            / "predictions"
            / source.key.storage_key
            / "members"
            / f"member_{index:03d}"
            / "raw.npz"
            for index in range(_MEMBER_COUNT)
        )
        identity_key = (source.key.run_label, source.key.dataset_label)
        assert source.member_paths == expected_members
        assert source.targets_path == campaign_artifacts.targets_paths[identity_key]
        assert source.uq_results_path == (
            run.run_root
            / "uncertainty"
            / source.key.storage_key
            / "raw"
            / "results.npz"
        )
        assert source.prediction_manifest_sha256 == sha256_file(
            campaign_artifacts.prediction_manifests[identity_key]
        )
        assert source.uq_manifest_sha256 == sha256_file(
            campaign_artifacts.uq_manifests[identity_key]
        )
        assert source.targets_sha256 == sha256_file(source.targets_path)


def test_panel_arrays_match_carnet_energy_and_force_definitions(
    campaign_artifacts: CampaignArtifacts,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
        load_panel_arrays,
    )

    sources = discover_plot_sources(campaign_artifacts.campaign)
    energy = _source(sources, "run_z", "matpes_test", "energy")
    force = _source(sources, "run_z", "matpes_test", "force")

    energy_uncertainty, energy_residual = load_panel_arrays(energy)
    force_uncertainty, force_residual = load_panel_arrays(force)

    assert energy_uncertainty == pytest.approx(np.array([1.0, 0.5]))
    assert energy_residual == pytest.approx(np.array([0.25, 0.75]))
    assert force_uncertainty.shape == force_residual.shape == (3, 3)
    assert force_uncertainty == pytest.approx(_FORCE_STD)
    assert force_residual == pytest.approx(_FORCE_RESIDUAL)


def test_stress_symmetrizes_each_member_before_welford_and_voigt(
    campaign_artifacts: CampaignArtifacts,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
        load_panel_arrays,
        symmetric_voigt,
    )

    sources = discover_plot_sources(campaign_artifacts.campaign)
    stress_source = _source(sources, "run_z", "matpes_test", "stress")
    uncertainty, residual = load_panel_arrays(stress_source)

    members = np.stack(
        [_member_values(index).stress for index in range(_MEMBER_COUNT)], axis=0
    )
    member_voigt = _literal_symmetric_voigt(members)
    expected_uncertainty = np.std(member_voigt, axis=0, ddof=1, dtype=np.float64)
    expected_mean = np.mean(member_voigt, axis=0, dtype=np.float64)
    expected_reference = _literal_symmetric_voigt(_STRESS_REFERENCE)
    expected_residual = np.abs(expected_mean - expected_reference)
    matrix_std = np.std(members, axis=0, ddof=1, dtype=np.float64)

    assert uncertainty.shape == residual.shape == (2, 6)
    assert uncertainty == pytest.approx(expected_uncertainty)
    assert residual == pytest.approx(expected_residual)
    assert not np.allclose(uncertainty, symmetric_voigt(matrix_std))


def test_stress_loading_streams_members_without_retaining_the_ensemble(
    campaign_artifacts: CampaignArtifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_source

    sources = plot_source.discover_plot_sources(campaign_artifacts.campaign)
    stress_source = _source(sources, "run_z", "matpes_train", "stress")
    original_loader = plot_source.load_prediction_arrays
    state = {"live": 0, "peak": 0}
    loaded_paths: list[Path] = []

    class TrackedStress(np.ndarray):
        pass

    def release() -> None:
        state["live"] -= 1

    def tracked_loader(path: Path) -> PredictionArrays:
        values = original_loader(path)
        stress = values.stress.view(TrackedStress)
        state["live"] += 1
        state["peak"] = max(state["peak"], state["live"])
        weakref.finalize(stress, release)
        loaded_paths.append(Path(path))
        return PredictionArrays(values.energy, values.forces, stress)

    monkeypatch.setattr(plot_source, "load_prediction_arrays", tracked_loader)
    plot_source.load_panel_arrays(stress_source)
    gc.collect()

    assert loaded_paths == list(stress_source.member_paths)
    assert state["peak"] <= 2
    assert state["live"] == 0


def test_discovery_rejects_symlinked_prediction_and_uq_ancestors(
    campaign_artifacts: CampaignArtifacts, tmp_path: Path
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
    )

    run_root = campaign_artifacts.campaign.runs[0].run_root
    prediction_root = run_root / "predictions"
    external_predictions = tmp_path / "external_predictions"
    prediction_root.rename(external_predictions)
    prediction_root.symlink_to(external_predictions, target_is_directory=True)

    with pytest.raises(HardFailure, match="symlink"):
        discover_plot_sources(campaign_artifacts.campaign)

    prediction_root.unlink()
    external_predictions.rename(prediction_root)
    uncertainty_root = run_root / "uncertainty"
    external_uncertainty = tmp_path / "external_uncertainty"
    uncertainty_root.rename(external_uncertainty)
    uncertainty_root.symlink_to(external_uncertainty, target_is_directory=True)

    with pytest.raises(HardFailure, match="symlink"):
        discover_plot_sources(campaign_artifacts.campaign)


def test_discovery_rejects_prediction_manifest_with_reordered_members(
    campaign_artifacts: CampaignArtifacts,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
    )

    manifest = campaign_artifacts.prediction_manifests[("run_z", "matpes_test")]
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["members"].reverse()
    manifest.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(HardFailure, match="order"):
        discover_plot_sources(campaign_artifacts.campaign)


def test_discovery_binds_v2_dataset_label_without_breaking_v1(
    campaign_artifacts: CampaignArtifacts,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
    )

    campaign = campaign_artifacts.campaign
    first_run = (campaign.runs[0],)
    v1_campaign = replace(campaign, runs=first_run, datasets=(campaign.datasets[0],))
    v1_sources = discover_plot_sources(v1_campaign)
    assert [source.key.target for source in v1_sources] == [
        "energy",
        "force",
        "stress",
    ]

    manifest = campaign_artifacts.prediction_manifests[("run_z", "mad_test")]
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["schema"] == "upet.bootstrap.predictions/v2"
    document["dataset_label"] = "different_mad_test"
    manifest.write_text(json.dumps(document), encoding="utf-8")
    v2_campaign = replace(campaign, runs=first_run, datasets=(campaign.datasets[1],))

    with pytest.raises(HardFailure, match="dataset label|identity"):
        discover_plot_sources(v2_campaign)


def test_discovery_rejects_bad_prediction_sha_and_partial_uq(
    tmp_path: Path,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
        discover_plot_sources,
    )

    bad_prediction = _build_campaign_artifacts(tmp_path / "bad_prediction")
    prediction_manifest = bad_prediction.prediction_manifests[("run_z", "matpes_test")]
    document = json.loads(prediction_manifest.read_text(encoding="utf-8"))
    document["targets"]["sha256"] = "0" * 64
    prediction_manifest.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HardFailure, match="SHA-256"):
        discover_plot_sources(bad_prediction.campaign)

    partial_uq = _build_campaign_artifacts(tmp_path / "partial_uq")
    partial_uq.uq_manifests[("run_z", "matpes_test")].unlink()
    with pytest.raises(HardFailure, match="UQ publication"):
        discover_plot_sources(partial_uq.campaign)
