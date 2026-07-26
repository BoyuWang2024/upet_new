"""Strict configuration contracts for the LLPR workflow."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_repo_path(raw: str | Path) -> Path:
    """Resolve ``raw`` against the repository root without requiring existence."""
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


class StrictModel(BaseModel):
    """Immutable model that rejects accidental configuration keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetEta(StrictModel):
    energy: float = Field(gt=0)
    force: float = Field(gt=0)


class FileIdentityConfig(StrictModel):
    path: Path
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DataConfig(StrictModel):
    build: Path
    calibration: Path
    test: Path
    build_expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    calibration_expected_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    test_expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class CurvatureConfig(StrictModel):
    energy_loss_weight: float = Field(gt=0)
    force_loss_weight: float = Field(gt=0)
    energy_huber_delta: float = Field(gt=0)
    force_huber_delta: float = Field(gt=0)
    checkpoint_interval: int = Field(default=1, gt=0)


class RidgeFitConfig(StrictModel):
    candidate_multipliers: tuple[float, ...] = (
        1,
        3,
        10,
        30,
        100,
        300,
        1000,
    )
    score: Literal["gaussian_nll"] = "gaussian_nll"

    @model_validator(mode="after")
    def validate_multipliers(self) -> "RidgeFitConfig":
        if not self.candidate_multipliers:
            raise ValueError("candidate_multipliers must not be empty")
        if any(value <= 0 for value in self.candidate_multipliers):
            raise ValueError("candidate_multipliers must all be positive")
        return self


class RidgeConfig(StrictModel):
    mode: Literal["fixed", "fit"]
    max_condition_number: float = Field(default=1.0e10, gt=1)
    eta: TargetEta | None = None
    fit: RidgeFitConfig | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "RidgeConfig":
        if self.mode == "fixed":
            if self.eta is None:
                raise ValueError("fixed mode requires eta")
            if self.fit is not None:
                raise ValueError("fixed mode must not define fit")
        else:
            if self.eta is not None:
                raise ValueError("fit mode must not define eta")
            if self.fit is None:
                raise ValueError("fit mode must define fit")
        return self


class CalibrationConfig(StrictModel):
    ridge: RidgeConfig


class RuntimeConfig(StrictModel):
    device: str = "cpu"
    matrix_dtype: Literal["float64"] = "float64"
    jacobian_backend: Literal["scalar", "batched"] = "scalar"
    force_component_chunk_size: int = Field(default=64, gt=0)


class OutputConfig(StrictModel):
    root: Path
    experiment: str = Field(min_length=1)
    shard_structure_count: int = Field(default=64, gt=0)
    keep_shards: bool = False


class LLPRConfig(StrictModel):
    checkpoint: FileIdentityConfig
    data: DataConfig
    curvature: CurvatureConfig
    calibration: CalibrationConfig
    runtime: RuntimeConfig
    output: OutputConfig


def _normalize_eta(raw: dict[str, Any]) -> None:
    calibration = raw.get("calibration")
    if not isinstance(calibration, dict):
        return
    ridge = calibration.get("ridge")
    if not isinstance(ridge, dict):
        return
    eta = ridge.get("eta")
    if isinstance(eta, (int, float)) and not isinstance(eta, bool):
        ridge["eta"] = {"energy": float(eta), "force": float(eta)}


def _resolve_paths(raw: dict[str, Any]) -> None:
    checkpoint = raw.get("checkpoint")
    if isinstance(checkpoint, dict) and "path" in checkpoint:
        checkpoint["path"] = resolve_repo_path(checkpoint["path"])
    data = raw.get("data")
    if isinstance(data, dict):
        for name in ("build", "calibration", "test"):
            if name in data:
                data[name] = resolve_repo_path(data[name])
    output = raw.get("output")
    if isinstance(output, dict) and "root" in output:
        output["root"] = resolve_repo_path(output["root"])


def load_llpr_config(path: Path) -> LLPRConfig:
    """Load, normalize, and validate one YAML configuration."""
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("LLPR configuration must contain a YAML mapping")
    raw: dict[str, Any] = loaded
    _normalize_eta(raw)
    _resolve_paths(raw)
    return LLPRConfig.model_validate(raw)
