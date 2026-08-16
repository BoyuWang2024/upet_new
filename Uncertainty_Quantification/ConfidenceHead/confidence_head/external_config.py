"""Strict configuration for prediction on datasets outside a training run."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, field_validator, model_validator

from .config import (
    REPO_ROOT,
    CheckpointConfig,
    Profile,
    StrictModel,
    UniqueKeySafeLoader,
)


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class ExistingEvaluationSource(StrictModel):
    """Use the complete evaluation already declared by every selected run."""

    source: Literal["existing_evaluation"]
    expected_sha256: Sha256


class CacheSplitSource(StrictModel):
    """Run a trained head over a verified split of its declared raw cache."""

    source: Literal["cache_split"]
    split: str = Field(default="train", min_length=1)
    expected_sha256: Sha256

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: str) -> str:
        if value in {".", ".."} or Path(value).name != value:
            raise ValueError("split must be a safe path component")
        return value


class ExtXYZSource(StrictModel):
    """Build one immutable raw cache from a labelled extxyz dataset."""

    source: Literal["extxyz"]
    path: Path
    expected_sha256: Sha256


DatasetSource = Annotated[
    ExistingEvaluationSource | CacheSplitSource | ExtXYZSource,
    Field(discriminator="source"),
]


class ExternalPredictionConfig(StrictModel):
    """Inputs and publication roots for external ConfidenceHead inference."""

    profile: Profile
    checkpoint: CheckpointConfig
    datasets: dict[str, DatasetSource]
    runs_root: Path
    cache_root: Path
    plots_root: Path
    batch_size: int = Field(default=128, gt=0)
    device: str = Field(default="cpu", min_length=1)

    @field_validator("datasets")
    @classmethod
    def validate_dataset_names(
        cls, value: dict[str, DatasetSource]
    ) -> dict[str, DatasetSource]:
        if not value:
            raise ValueError("datasets must not be empty")
        for name in value:
            if name in {".", ".."} or Path(name).name != name:
                raise ValueError(f"dataset name must be a safe path component: {name!r}")
        return value

    @model_validator(mode="after")
    def validate_production_datasets(self) -> "ExternalPredictionConfig":
        required = {"matpes_test", "matpes_train", "mad_test"}
        if self.profile == "production" and set(self.datasets) != required:
            raise ValueError(
                "production datasets must be exactly matpes_test, matpes_train, "
                "mad_test"
            )
        return self


def _absolute(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).absolute()


def _resolve_paths(raw: dict[str, Any], root: Path) -> None:
    checkpoint = raw.get("checkpoint")
    if isinstance(checkpoint, dict) and "path" in checkpoint:
        checkpoint["path"] = _absolute(checkpoint["path"], root)
    datasets = raw.get("datasets")
    if isinstance(datasets, dict):
        for source in datasets.values():
            if isinstance(source, dict) and "path" in source:
                source["path"] = _absolute(source["path"], root)
    for field in ("runs_root", "cache_root", "plots_root"):
        if field in raw:
            raw[field] = _absolute(raw[field], root)


def load_external_config(
    path: Path,
    repo_root: Path | None = None,
) -> ExternalPredictionConfig:
    """Load a duplicate-key-safe external prediction YAML configuration."""

    root = (REPO_ROOT if repo_root is None else Path(repo_root)).resolve()
    try:
        loaded = yaml.load(Path(path).read_text(encoding="utf-8"), UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid external prediction config {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError("external prediction config must contain a mapping")
    _resolve_paths(loaded, root)
    return ExternalPredictionConfig.model_validate(loaded)
