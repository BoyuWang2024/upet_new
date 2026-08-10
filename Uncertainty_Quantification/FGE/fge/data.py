"""Canonical, source-independent identities for FGE datasets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import HardFailure


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_OBSERVABLES = ("energy", "forces", "stress")


@dataclass(frozen=True)
class DatasetIdentity:
    """Immutable logical provenance for one ordered dataset split."""

    split: str
    structure_ids: tuple[str, ...]
    structure_count: int
    atom_count: int
    content_sha256: str
    target_names: tuple[tuple[str, str], ...]
    units: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.split, str) or not self.split:
            raise HardFailure("dataset split must be a non-empty string")
        if (
            not isinstance(self.structure_count, int)
            or isinstance(self.structure_count, bool)
            or self.structure_count < 1
        ):
            raise HardFailure("dataset structure_count must be positive")
        if (
            not isinstance(self.atom_count, int)
            or isinstance(self.atom_count, bool)
            or self.atom_count < 1
        ):
            raise HardFailure("dataset atom_count must be positive")
        if (
            not isinstance(self.structure_ids, tuple)
            or len(self.structure_ids) != self.structure_count
            or any(
                not isinstance(value, str) or not value for value in self.structure_ids
            )
            or len(set(self.structure_ids)) != len(self.structure_ids)
        ):
            raise HardFailure("dataset structure_ids must be ordered and unique")
        if (
            not isinstance(self.content_sha256, str)
            or _SHA256_RE.fullmatch(self.content_sha256) is None
        ):
            raise HardFailure("dataset content_sha256 must be a lowercase SHA256")
        _validate_roles(self.target_names, "dataset target_names")
        _validate_roles(self.units, "dataset units")


def _validate_roles(value: object, label: str) -> None:
    if not isinstance(value, tuple) or len(value) != len(_OBSERVABLES):
        raise HardFailure(f"{label} must define energy, forces, and stress")
    roles: list[str] = []
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not item[0]
            or not item[1]
        ):
            raise HardFailure(f"{label} must contain non-empty string pairs")
        roles.append(item[0])
    if tuple(roles) != _OBSERVABLES:
        raise HardFailure(f"{label} roles must be energy, forces, and stress in order")
