"""Strict configuration for inference-only FGE dataset runs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, cast

import yaml

from .errors import HardFailure


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_SCHEMA_VERSION = "upet.fge.inference.v1"
_REFERENCE_KEYS = {"energy", "forces", "stress"}
_ALLOWED_KEYS = {
    "root": {
        "schema_version",
        "ensemble",
        "dataset",
        "chunking",
        "output",
        "runtime",
    },
    "ensemble": {
        "root",
        "result_manifest_sha256",
        "base_checkpoint",
        "base_checkpoint_sha256",
        "member_count",
    },
    "dataset": {
        "label",
        "path",
        "expected_sha256",
        "split",
        "reference_availability",
    },
    "chunking": {"max_structures", "max_atoms"},
    "output": {"root"},
    "runtime": {"device", "torch_threads"},
}


@dataclass(frozen=True)
class DatasetSourceConfig:
    label: str
    path: Path
    expected_sha256: str
    split: str
    reference_availability: Mapping[str, bool]


@dataclass(frozen=True)
class ChunkPolicy:
    max_structures: int
    max_atoms: int


@dataclass(frozen=True)
class EnsembleSourceConfig:
    root: Path
    result_manifest_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    member_count: int


@dataclass(frozen=True)
class InferenceOutputConfig:
    root: Path


@dataclass(frozen=True)
class RuntimeInferenceConfig:
    device: str
    torch_threads: int


@dataclass(frozen=True)
class InferenceConfig:
    schema_version: str
    ensemble: EnsembleSourceConfig
    dataset: DatasetSourceConfig
    chunking: ChunkPolicy
    output: InferenceOutputConfig
    runtime: RuntimeInferenceConfig

    def sanitized(self) -> dict[str, object]:
        """Return canonical identity data without machine-specific paths."""
        return {
            "schema_version": self.schema_version,
            "ensemble": {
                "root": {"role": "completed_fge_result"},
                "result_manifest_sha256": self.ensemble.result_manifest_sha256,
                "base_checkpoint": {
                    "role": "base_checkpoint",
                    "sha256": self.ensemble.base_checkpoint_sha256,
                },
                "member_count": self.ensemble.member_count,
            },
            "dataset": {
                "label": self.dataset.label,
                "path": {
                    "role": "inference_dataset",
                    "sha256": self.dataset.expected_sha256,
                },
                "split": self.dataset.split,
                "reference_availability": dict(self.dataset.reference_availability),
            },
            "chunking": {
                "max_structures": self.chunking.max_structures,
                "max_atoms": self.chunking.max_atoms,
            },
            "output": {"root": {"role": "inference_output"}},
            "runtime": {
                "device": self.runtime.device,
                "torch_threads": self.runtime.torch_threads,
            },
        }


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise HardFailure(f"{location} must be a mapping")
    if any(type(key) is not str for key in value):
        raise HardFailure(f"{location} keys must be strings")
    return cast(Mapping[str, object], value)


def _strict_keys(value: Mapping[str, object], location: str) -> None:
    expected = _ALLOWED_KEYS[location]
    actual = set(value)
    unknown = actual - expected
    missing = expected - actual
    if unknown:
        raise HardFailure(f"unknown key in {location}: {sorted(unknown)[0]}")
    if missing:
        raise HardFailure(f"missing key in {location}: {sorted(missing)[0]}")


def _section(root: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = _mapping(root[name], name)
    _strict_keys(value, name)
    return value


def _string(value: object, location: str) -> str:
    if type(value) is not str:
        raise HardFailure(f"{location} must be a string")
    return cast(str, value)


def _sha256(value: object, location: str) -> str:
    digest = _string(value, location)
    if _SHA256_RE.fullmatch(digest) is None:
        raise HardFailure(f"{location} must be a lowercase SHA-256 digest")
    return digest


def _positive_integer(value: object, location: str) -> int:
    if type(value) is not int or value <= 0:
        raise HardFailure(f"{location} must be a positive integer")
    return cast(int, value)


def _path(value: object, location: str, directory: Path) -> Path:
    raw = _string(value, location)
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = directory / candidate
    return candidate.resolve()


def _references(value: object) -> Mapping[str, bool]:
    mapping = _mapping(value, "dataset.reference_availability")
    actual = set(mapping)
    if actual != _REFERENCE_KEYS:
        raise HardFailure(
            "dataset.reference_availability must contain exactly energy, forces, stress"
        )
    if any(type(mapping[key]) is not bool for key in _REFERENCE_KEYS):
        raise HardFailure(
            "dataset.reference_availability values must be strict booleans"
        )
    return MappingProxyType(
        {key: cast(bool, mapping[key]) for key in ("energy", "forces", "stress")}
    )


def load_inference_config(path: str | Path) -> InferenceConfig:
    """Load one exact, path-neutral FGE inference configuration."""
    source = Path(path).resolve()
    try:
        parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise HardFailure(
            f"unable to load inference configuration {source}: {exc}"
        ) from exc

    root = _mapping(parsed, "root")
    _strict_keys(root, "root")
    ensemble = _section(root, "ensemble")
    dataset = _section(root, "dataset")
    chunking = _section(root, "chunking")
    output = _section(root, "output")
    runtime = _section(root, "runtime")

    schema_version = _string(root["schema_version"], "schema_version")
    if schema_version != _SCHEMA_VERSION:
        raise HardFailure("schema_version is fixed for inference configuration")

    member_count = _positive_integer(ensemble["member_count"], "ensemble.member_count")
    if member_count != 8:
        raise HardFailure("ensemble.member_count must be exactly 8")

    label = _string(dataset["label"], "dataset.label")
    if _LABEL_RE.fullmatch(label) is None:
        raise HardFailure("dataset.label is invalid")
    split = _string(dataset["split"], "dataset.split")
    if _LABEL_RE.fullmatch(split) is None:
        raise HardFailure("dataset.split is invalid")

    device = _string(runtime["device"], "runtime.device")
    if device != "cpu":
        raise HardFailure("runtime.device must be cpu")

    directory = source.parent
    return InferenceConfig(
        schema_version=schema_version,
        ensemble=EnsembleSourceConfig(
            root=_path(ensemble["root"], "ensemble.root", directory),
            result_manifest_sha256=_sha256(
                ensemble["result_manifest_sha256"],
                "ensemble.result_manifest_sha256",
            ),
            base_checkpoint=_path(
                ensemble["base_checkpoint"], "ensemble.base_checkpoint", directory
            ),
            base_checkpoint_sha256=_sha256(
                ensemble["base_checkpoint_sha256"],
                "ensemble.base_checkpoint_sha256",
            ),
            member_count=member_count,
        ),
        dataset=DatasetSourceConfig(
            label=label,
            path=_path(dataset["path"], "dataset.path", directory),
            expected_sha256=_sha256(
                dataset["expected_sha256"], "dataset.expected_sha256"
            ),
            split=split,
            reference_availability=_references(dataset["reference_availability"]),
        ),
        chunking=ChunkPolicy(
            max_structures=_positive_integer(
                chunking["max_structures"], "chunking.max_structures"
            ),
            max_atoms=_positive_integer(chunking["max_atoms"], "chunking.max_atoms"),
        ),
        output=InferenceOutputConfig(
            root=_path(output["root"], "output.root", directory)
        ),
        runtime=RuntimeInferenceConfig(
            device=device,
            torch_threads=_positive_integer(
                runtime["torch_threads"], "runtime.torch_threads"
            ),
        ),
    )


__all__ = [
    "ChunkPolicy",
    "DatasetSourceConfig",
    "EnsembleSourceConfig",
    "InferenceConfig",
    "InferenceOutputConfig",
    "RuntimeInferenceConfig",
    "load_inference_config",
]
