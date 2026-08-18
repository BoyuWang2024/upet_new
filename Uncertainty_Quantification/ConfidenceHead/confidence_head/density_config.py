"""Strict YAML configuration for ConfidenceHead density plotting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator

from .config import StrictModel, UniqueKeySafeLoader
from .density_plotting import DensitySettings


class DensityPlotSettings(StrictModel):
    scatter_max_points: int = Field(default=20_000, gt=0)
    scatter_seed: int = 20260714
    grid_size: int = Field(default=160, ge=8)
    gaussian_sigma: float = Field(default=1.2, gt=0.0)
    contour_masses: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 0.99)
    log_margin: float = Field(default=0.05, gt=0.0)
    dpi: int = Field(default=300, gt=0)

    @field_validator("contour_masses")
    @classmethod
    def validate_contour_masses(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not 0.0 < value < 1.0 for value in values):
            raise ValueError("contour_masses must be inside (0, 1)")
        if any(left >= right for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("contour_masses must be strictly increasing")
        return values

    def settings(self) -> DensitySettings:
        return DensitySettings(**self.model_dump())


class DensityPlotConfig(StrictModel):
    external_config: Path
    output_root: Path
    plot: DensityPlotSettings = Field(default_factory=DensityPlotSettings)

    def settings(self) -> DensitySettings:
        return self.plot.settings()


def _absolute(value: object, root: Path, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a filesystem path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).absolute()


def load_density_config(path: Path) -> DensityPlotConfig:
    """Load duplicate-key-safe density YAML and resolve its paths."""

    config_path = Path(path).expanduser().resolve()
    try:
        loaded = yaml.load(config_path.read_text(encoding="utf-8"), UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"invalid density plot config {config_path}: {error}"
        ) from error
    if not isinstance(loaded, dict):
        raise ValueError("density plot config must contain a mapping")
    raw: dict[str, Any] = dict(loaded)
    raw["external_config"] = _absolute(
        raw.get("external_config"), config_path.parent, "external_config"
    )
    raw["output_root"] = _absolute(
        raw.get("output_root"), config_path.parent, "output_root"
    )
    return DensityPlotConfig.model_validate(raw)
