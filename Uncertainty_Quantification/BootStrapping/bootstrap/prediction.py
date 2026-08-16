"""Canonical target and member prediction storage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from .artifacts import atomic_write_npz, sha256_file
from .errors import HardFailure
from .identifiers import validate_artifact_key


_TARGET_KEYS = {
    "structure_ids",
    "num_atoms",
    "atom_offsets",
    "energy",
    "forces",
    "stress",
}
_TARGET_KEYS_WITHOUT_STRESS = _TARGET_KEYS - {"stress"}
_PREDICTION_KEYS = {"energy", "forces", "stress"}
_EXPECTED_UNITS = {
    "energy": "eV",
    "forces": "eV/Angstrom",
    "stress": "eV/Angstrom^3",
}


@dataclass(frozen=True)
class TargetArrays:
    structure_ids: NDArray
    num_atoms: NDArray
    atom_offsets: NDArray
    energy: NDArray
    forces: NDArray
    stress: NDArray | None


def reference_targets(targets: TargetArrays) -> tuple[str, ...]:
    """Return the reference fields available for one target store."""

    fields = ["energy", "forces"]
    if targets.stress is not None:
        fields.append("stress")
    return tuple(fields)


@dataclass(frozen=True)
class PredictionArrays:
    energy: NDArray
    forces: NDArray
    stress: NDArray


@dataclass(frozen=True)
class PredictionPublication:
    path: Path
    sha256: str
    member_index: int
    mode: str
    split: str
    shapes: dict[str, tuple[int, ...]]
    dtypes: dict[str, str]
    units: dict[str, str]


def _array(value: object, location: str) -> NDArray:
    if not isinstance(value, np.ndarray):
        raise HardFailure(f"{location} must be a NumPy array")
    return value


def _finite(array: NDArray, location: str) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise HardFailure(f"{location} must be numeric")
    if not bool(np.isfinite(array).all()):
        raise HardFailure(f"{location} must contain only finite values")


def validate_targets(targets: TargetArrays) -> None:
    """Validate one split's IDs, atom layout and reference arrays."""

    ids = _array(targets.structure_ids, "targets.structure_ids")
    num_atoms = _array(targets.num_atoms, "targets.num_atoms")
    offsets = _array(targets.atom_offsets, "targets.atom_offsets")
    energy = _array(targets.energy, "targets.energy")
    forces = _array(targets.forces, "targets.forces")
    if ids.ndim != 1:
        raise HardFailure("targets.structure_ids must be one-dimensional")
    count = len(ids)
    if len(np.unique(ids)) != count:
        raise HardFailure("targets.structure_ids must be unique")
    if (
        num_atoms.shape != (count,)
        or not np.issubdtype(num_atoms.dtype, np.integer)
        or bool((num_atoms < 1).any())
    ):
        raise HardFailure("targets.num_atoms has an invalid layout")
    expected_offsets = np.concatenate(
        [np.array([0], dtype=np.int64), np.cumsum(num_atoms, dtype=np.int64)]
    )
    if offsets.shape != (count + 1,) or not np.array_equal(offsets, expected_offsets):
        raise HardFailure("targets.atom_offsets do not match the atom layout")
    atom_count = int(expected_offsets[-1])
    if energy.shape != (count,):
        raise HardFailure("targets.energy has an invalid layout")
    if forces.shape != (atom_count, 3):
        raise HardFailure("targets.forces has an invalid layout")
    for name, array in (("energy", energy), ("forces", forces)):
        _finite(array, f"targets.{name}")
    if targets.stress is not None:
        stress = _array(targets.stress, "targets.stress")
        if stress.shape != (count, 3, 3):
            raise HardFailure("targets.stress has an invalid layout")
        _finite(stress, "targets.stress")


def validate_predictions(values: PredictionArrays, targets: TargetArrays) -> None:
    """Require exact structural alignment with a validated target store."""

    validate_targets(targets)
    count = len(targets.structure_ids)
    atom_count = int(targets.atom_offsets[-1])
    expected = {
        "energy": (count,),
        "forces": (atom_count, 3),
        "stress": (count, 3, 3),
    }
    for name in sorted(_PREDICTION_KEYS):
        array = _array(getattr(values, name), f"predictions.{name}")
        if array.shape != expected[name]:
            raise HardFailure(f"predictions.{name} does not match target layout")
        _finite(array, f"predictions.{name}")


def _load_npz(path: str | Path, expected: set[str]) -> dict[str, NDArray]:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise HardFailure(f"NPZ array keys do not match schema: {source}")
            return {name: np.array(archive[name], copy=True) for name in expected}
    except HardFailure:
        raise
    except (OSError, ValueError) as error:
        raise HardFailure(
            f"could not load prediction artifact {source}: {error}"
        ) from error


def load_target_arrays(path: str | Path) -> TargetArrays:
    source = Path(path).expanduser().resolve()
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = set(archive.files)
            if keys == _TARGET_KEYS:
                arrays: dict[str, NDArray | None] = {
                    name: np.array(archive[name], copy=True) for name in _TARGET_KEYS
                }
            elif keys == _TARGET_KEYS_WITHOUT_STRESS:
                arrays = {
                    name: np.array(archive[name], copy=True)
                    for name in _TARGET_KEYS_WITHOUT_STRESS
                }
                arrays["stress"] = None
            else:
                raise HardFailure(f"NPZ array keys do not match schema: {source}")
    except HardFailure:
        raise
    except (OSError, ValueError) as error:
        raise HardFailure(
            f"could not load prediction artifact {source}: {error}"
        ) from error
    targets = TargetArrays(**arrays)
    validate_targets(targets)
    return targets


def load_prediction_arrays(path: str | Path) -> PredictionArrays:
    arrays = _load_npz(path, _PREDICTION_KEYS)
    return PredictionArrays(**arrays)


class PredictionStore:
    """Immutable per-split target and member prediction store."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        units: Mapping[str, str],
        direct_split_root: bool = False,
    ) -> None:
        self.split = validate_artifact_key(split, "prediction split")
        if dict(units) != _EXPECTED_UNITS:
            raise HardFailure("prediction units do not match the public schema")
        self.root = Path(root).expanduser().absolute()
        self._direct_split_root = direct_split_root
        self.units = dict(units)

    @classmethod
    def at_split_root(
        cls, split_root: str | Path, *, split: str, units: Mapping[str, str]
    ) -> "PredictionStore":
        """Create a store whose root is already the final split directory."""

        return cls(split_root, split=split, units=units, direct_split_root=True)

    @property
    def split_root(self) -> Path:
        if self._direct_split_root:
            return self.root
        return self.root / self.split

    @property
    def targets_path(self) -> Path:
        return self.split_root / "targets.npz"

    def write_targets(self, targets: TargetArrays) -> Path:
        validate_targets(targets)
        arrays: dict[str, NDArray] = {
            "structure_ids": targets.structure_ids,
            "num_atoms": targets.num_atoms,
            "atom_offsets": targets.atom_offsets,
            "energy": targets.energy,
            "forces": targets.forces,
        }
        if targets.stress is not None:
            arrays["stress"] = targets.stress
        return atomic_write_npz(self.targets_path, **arrays)

    def write_member(
        self, member_index: int, mode: str, values: PredictionArrays
    ) -> PredictionPublication:
        if isinstance(member_index, bool) or member_index < 0:
            raise HardFailure("prediction member_index must be non-negative")
        if mode not in {"raw", "ema"}:
            raise HardFailure("prediction mode must be raw or ema")
        if not self.targets_path.is_file():
            raise HardFailure("prediction targets must be written before members")
        targets = load_target_arrays(self.targets_path)
        validate_predictions(values, targets)
        path = (
            self.split_root / "members" / f"member_{member_index:03d}" / f"{mode}.npz"
        )
        atomic_write_npz(
            path,
            energy=values.energy,
            forces=values.forces,
            stress=values.stress,
        )
        return PredictionPublication(
            path=path,
            sha256=sha256_file(path),
            member_index=member_index,
            mode=mode,
            split=self.split,
            shapes={
                name: tuple(getattr(values, name).shape) for name in _PREDICTION_KEYS
            },
            dtypes={
                name: str(getattr(values, name).dtype) for name in _PREDICTION_KEYS
            },
            units=dict(self.units),
        )
