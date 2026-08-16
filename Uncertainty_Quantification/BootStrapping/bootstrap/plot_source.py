"""Audited, bounded-memory scientific inputs for Bootstrap campaign plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .artifacts import sha256_file
from .campaign import CampaignConfig
from .errors import HardFailure
from .prediction import load_prediction_arrays, load_target_arrays
from .prediction_publication import validate_prediction_publication
from .validation import validate_uq_publication


PanelTarget = Literal["energy", "force", "stress"]
_TARGET_ORDER: tuple[PanelTarget, ...] = ("energy", "force", "stress")


@dataclass(frozen=True, order=True)
class PanelKey:
    """Stable campaign coordinates for one uncertainty/residual panel."""

    run_label: str
    dataset_label: str
    storage_key: str
    target: PanelTarget


@dataclass(frozen=True)
class PlotSource:
    """Canonical, audited artifacts needed to load one scientific panel."""

    key: PanelKey
    targets_path: Path
    uq_results_path: Path
    member_paths: tuple[Path, ...]
    prediction_manifest_sha256: str
    uq_manifest_sha256: str
    targets_sha256: str


def _absolute_lexical(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _panel_targets(reference_targets: tuple[str, ...]) -> tuple[PanelTarget, ...]:
    if reference_targets == ("energy", "forces"):
        return _TARGET_ORDER[:2]
    if reference_targets == ("energy", "forces", "stress"):
        return _TARGET_ORDER
    raise HardFailure("plot source reference targets are not supported")


def discover_plot_sources(campaign: CampaignConfig) -> tuple[PlotSource, ...]:
    """Audit each campaign input once and emit panels in declaration order."""

    mode = campaign.prediction.mode
    member_count = campaign.prediction.member_count
    if mode != "raw":
        raise HardFailure("plot sources require raw campaign predictions")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
    ):
        raise HardFailure("plot sources require at least two members")

    sources: list[PlotSource] = []
    for run in campaign.runs:
        run_root = _absolute_lexical(run.run_root)
        for dataset in campaign.datasets:
            split_root = run_root / "predictions" / dataset.storage_key
            prediction_audit = validate_prediction_publication(
                split_root,
                dataset_key=dataset.storage_key,
                mode=mode,
                member_count=member_count,
                reference_targets=dataset.reference_targets,
                dataset_label=dataset.label,
                structure_limit=None,
            )
            validate_uq_publication(
                run_root,
                split=dataset.storage_key,
                mode=mode,
                member_count=member_count,
            )

            canonical_split = split_root.resolve()
            canonical_uq = (
                run_root / "uncertainty" / dataset.storage_key / mode
            ).resolve()
            targets_path = canonical_split / "targets.npz"
            uq_results_path = canonical_uq / "results.npz"
            member_paths = tuple(
                canonical_split / "members" / f"member_{index:03d}" / f"{mode}.npz"
                for index in range(member_count)
            )
            prediction_manifest = prediction_audit.manifest_path
            uq_manifest = canonical_uq / "manifest.json"
            prediction_manifest_sha256 = sha256_file(prediction_manifest)
            uq_manifest_sha256 = sha256_file(uq_manifest)
            targets_sha256 = sha256_file(targets_path)

            for target in _panel_targets(dataset.reference_targets):
                sources.append(
                    PlotSource(
                        key=PanelKey(
                            run_label=run.label,
                            dataset_label=dataset.label,
                            storage_key=dataset.storage_key,
                            target=target,
                        ),
                        targets_path=targets_path,
                        uq_results_path=uq_results_path,
                        member_paths=member_paths,
                        prediction_manifest_sha256=prediction_manifest_sha256,
                        uq_manifest_sha256=uq_manifest_sha256,
                        targets_sha256=targets_sha256,
                    )
                )
    return tuple(sources)


def symmetric_voigt(values: NDArray) -> NDArray[np.float64]:
    """Symmetrize trailing 3x3 tensors and return xx,yy,zz,yz,xz,xy."""

    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"stress tensor is not numeric: {error}") from error
    if array.ndim < 2 or array.shape[-2:] != (3, 3):
        raise HardFailure("stress tensor must end with a 3 x 3 layout")
    if not bool(np.isfinite(array).all()):
        raise HardFailure("stress tensor must contain only finite values")

    result = np.empty(array.shape[:-2] + (6,), dtype=np.float64)
    result[..., 0] = array[..., 0, 0]
    result[..., 1] = array[..., 1, 1]
    result[..., 2] = array[..., 2, 2]
    result[..., 3] = 0.5 * (array[..., 1, 2] + array[..., 2, 1])
    result[..., 4] = 0.5 * (array[..., 0, 2] + array[..., 2, 0])
    result[..., 5] = 0.5 * (array[..., 0, 1] + array[..., 1, 0])
    return result


def _load_uq_arrays(
    path: Path, names: tuple[str, ...]
) -> tuple[NDArray[np.float64], ...]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            missing = set(names) - set(archive.files)
            if missing:
                raise HardFailure(
                    f"UQ results are missing array {sorted(missing)[0]}: {path}"
                )
            arrays = tuple(
                np.asarray(archive[name], dtype=np.float64).copy() for name in names
            )
    except HardFailure:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise HardFailure(f"could not load UQ results {path}: {error}") from error
    if any(not bool(np.isfinite(array).all()) for array in arrays):
        raise HardFailure(f"UQ results contain non-finite values: {path}")
    return arrays


def _stress_mean_std(
    source: PlotSource, expected_shape: tuple[int, int]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.zeros(expected_shape, dtype=np.float64)
    m2 = np.zeros(expected_shape, dtype=np.float64)
    count = 0
    for member_path in source.member_paths:
        values = load_prediction_arrays(member_path)
        current = symmetric_voigt(values.stress)
        if current.shape != expected_shape:
            raise HardFailure(
                f"stress prediction does not match target layout: {member_path}"
            )
        count += 1
        delta = current - mean
        mean += delta / count
        delta2 = current - mean
        m2 += delta * delta2
        del values, current, delta, delta2
    if count < 2:
        raise HardFailure("stress uncertainty requires at least two members")
    variance = np.maximum(m2 / (count - 1), 0.0)
    return mean, np.sqrt(variance)


def load_panel_arrays(
    source: PlotSource,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Load one panel's uncertainty and absolute residual arrays."""

    targets = load_target_arrays(source.targets_path)
    if source.key.target == "energy":
        mean, standard_deviation = _load_uq_arrays(
            source.uq_results_path, ("energy_mean", "energy_std")
        )
        atom_counts = np.asarray(targets.num_atoms, dtype=np.float64)
        expected_shape = targets.energy.shape
        if mean.shape != expected_shape or standard_deviation.shape != expected_shape:
            raise HardFailure("energy UQ arrays do not match target layout")
        uncertainty = standard_deviation / atom_counts
        residual = np.abs(mean / atom_counts - targets.energy / atom_counts)
        return uncertainty, np.asarray(residual, dtype=np.float64)

    if source.key.target == "force":
        mean, uncertainty = _load_uq_arrays(
            source.uq_results_path, ("forces_mean", "forces_std")
        )
        if (
            mean.shape != targets.forces.shape
            or uncertainty.shape != targets.forces.shape
        ):
            raise HardFailure("force UQ arrays do not match target layout")
        residual = np.abs(mean - np.asarray(targets.forces, dtype=np.float64))
        return uncertainty, np.asarray(residual, dtype=np.float64)

    if source.key.target == "stress":
        if targets.stress is None:
            raise HardFailure("stress panel requires reference stress")
        reference = symmetric_voigt(targets.stress)
        mean, uncertainty = _stress_mean_std(source, reference.shape)
        residual = np.abs(mean - reference)
        return uncertainty, np.asarray(residual, dtype=np.float64)

    raise HardFailure(f"plot source target is not supported: {source.key.target}")
