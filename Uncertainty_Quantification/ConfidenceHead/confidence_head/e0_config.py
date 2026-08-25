"""Strict configuration for MAD r2SCAN E0 postprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator

from .config import StrictModel, UniqueKeySafeLoader
from .density_config import DensityPlotSettings


class ExpectedDatasetCounts(StrictModel):
    """Expected filtered dataset statistics used as a fail-closed guard."""

    structures: int = Field(gt=0)
    atoms: int = Field(gt=0)
    elements: int = Field(gt=0)


class E0PostprocessingConfig(StrictModel):
    """Inputs and publication roots for one E0 correction campaign."""

    external_config: Path
    validation_dataset: str
    test_dataset: str
    validation_expected: ExpectedDatasetCounts
    test_expected: ExpectedDatasetCounts
    output_root: Path
    plots_root: Path
    plot: DensityPlotSettings = Field(default_factory=DensityPlotSettings)

    @field_validator("validation_dataset", "test_dataset")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        if not value or value in {".", ".."} or Path(value).name != value:
            raise ValueError("dataset names must be safe path components")
        return value

    @model_validator(mode="after")
    def validate_distinct_datasets(self) -> "E0PostprocessingConfig":
        if self.validation_dataset == self.test_dataset:
            raise ValueError("validation_dataset and test_dataset must differ")
        return self


def _absolute(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a filesystem path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).absolute()


def load_e0_config(path: Path) -> E0PostprocessingConfig:
    """Load duplicate-key-safe YAML and resolve paths relative to the config."""

    config_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.load(config_path.read_text(encoding="utf-8"), UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"invalid E0 postprocessing config {config_path}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise ValueError("E0 postprocessing config must contain a mapping")
    raw: dict[str, Any] = dict(loaded)
    for name in ("external_config", "output_root", "plots_root"):
        raw[name] = _absolute(raw.get(name), config_path.parent, name)
    return E0PostprocessingConfig.model_validate(raw)
