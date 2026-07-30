"""Strict configuration contracts for the confidence-head workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver


REPO_ROOT = Path(__file__).resolve().parents[3]
Profile = Literal["smoke", "production"]


class UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys in every mapping."""


def _construct_unique_mapping(
    loader: UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            hash(key)
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found unhashable key",
                key_node.start_mark,
            ) from error
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeySafeLoader.add_constructor(
    BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class StrictModel(BaseModel):
    """Immutable configuration that rejects accidental keys."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FileIdentityConfig(StrictModel):
    path: Path
    expected_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class CheckpointConfig(FileIdentityConfig):
    pass


class DataConfig(StrictModel):
    train: FileIdentityConfig
    validation: FileIdentityConfig
    test: FileIdentityConfig


class ReadoutConfig(StrictModel):
    energy_prediction: str = Field(default="energy", min_length=1)
    force_prediction: str = Field(
        default="non_conservative_force",
        min_length=1,
    )
    energy_features: str = Field(
        default="mtt::aux::energy_last_layer_features",
        min_length=1,
    )
    force_features: str = Field(
        default="mtt::aux::non_conservative_forces_last_layer_features",
        min_length=1,
    )


class CacheConfig(StrictModel):
    batch_size: int = Field(default=2, gt=0)
    num_workers: int = Field(default=0, ge=0)
    shard_max_atoms: int = Field(default=100_000, gt=0)


class BinningConfig(StrictModel):
    algorithm: Literal["fixed_linear_v1"] = "fixed_linear_v1"
    force_num_bins: int = Field(default=50, ge=3)
    force_max_error: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    energy_num_bins: int = Field(default=50, ge=3)
    energy_max_error: float = Field(default=0.3, gt=0, allow_inf_nan=False)


class ModelConfig(StrictModel):
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    cumulant_order: int = Field(default=3, ge=1, le=8)
    signed_root: bool = True

    @field_validator("hidden_dims")
    @classmethod
    def validate_hidden_dims(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(dimension <= 0 for dimension in value):
            raise ValueError("hidden_dims must contain only positive dimensions")
        return value


class LossConfig(StrictModel):
    force_coefficient: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    energy_coefficient: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    label_smoothing: Literal[0] = 0
    class_weights: None = None


class OptimizerConfig(StrictModel):
    lr: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0, ge=0, allow_inf_nan=False)


class SchedulerConfig(StrictModel):
    monitor: Literal["val/total_loss_ema"] = "val/total_loss_ema"
    factor: float = Field(default=0.5, gt=0, lt=1, allow_inf_nan=False)
    patience: int = Field(default=5, ge=0)
    threshold: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    threshold_mode: Literal["abs"] = "abs"
    cooldown: int = Field(default=0, ge=0)
    min_lr: float = Field(default=1e-6, ge=0, allow_inf_nan=False)


class TrainerConfig(StrictModel):
    max_epochs: int = Field(default=200, gt=0)
    ema_beta: float = Field(default=0.95, ge=0, lt=1, allow_inf_nan=False)
    early_stopping_patience: int = Field(default=15, gt=0)
    min_delta: float = Field(default=1e-4, ge=0, allow_inf_nan=False)
    min_epochs: int = Field(default=3, gt=0)
    monitor: Literal["val/total_loss_ema"] = "val/total_loss_ema"


class RunConfig(StrictModel):
    output_root: Path = Path("Uncertainty_Quantification/ConfidenceHead/outputs")
    seed: int = 1234
    device: str = Field(default="cpu", min_length=1)
    amp: bool = False

    @model_validator(mode="after")
    def validate_cpu_amp(self) -> "RunConfig":
        if self.device.lower().startswith("cpu") and self.amp:
            raise ValueError("amp must be false when device starts with cpu")
        return self


class ConfidenceConfig(StrictModel):
    profile: Profile
    allow_identical_splits: bool = False
    checkpoint: CheckpointConfig
    data: DataConfig
    readouts: ReadoutConfig = Field(default_factory=ReadoutConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    binning: BinningConfig = Field(default_factory=BinningConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    run: RunConfig = Field(default_factory=RunConfig)

    @model_validator(mode="after")
    def validate_split_identities(self) -> "ConfidenceConfig":
        identities = (
            self.data.train.expected_sha256,
            self.data.validation.expected_sha256,
            self.data.test.expected_sha256,
        )
        identities_are_distinct = len(set(identities)) == len(identities)
        if self.profile == "production":
            if self.allow_identical_splits:
                raise ValueError("production forbids allow_identical_splits=true")
            if not identities_are_distinct:
                raise ValueError("production split identities must be distinct")
        elif not identities_are_distinct and not self.allow_identical_splits:
            raise ValueError(
                "identical smoke splits require allow_identical_splits=true"
            )
        return self


def _resolve_path(raw: str | Path, repo_root: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _resolve_config_paths(raw: dict[str, Any], repo_root: Path) -> None:
    checkpoint = raw.get("checkpoint")
    if isinstance(checkpoint, dict) and "path" in checkpoint:
        checkpoint["path"] = _resolve_path(checkpoint["path"], repo_root)

    data = raw.get("data")
    if isinstance(data, dict):
        for split_name in ("train", "validation", "test"):
            split = data.get(split_name)
            if isinstance(split, dict) and "path" in split:
                split["path"] = _resolve_path(split["path"], repo_root)

    run = raw.setdefault("run", {})
    if isinstance(run, dict):
        output_root = run.setdefault(
            "output_root",
            Path("Uncertainty_Quantification/ConfidenceHead/outputs"),
        )
        run["output_root"] = _resolve_path(output_root, repo_root)


def load_config(
    path: Path,
    repo_root: Path | None = None,
) -> ConfidenceConfig:
    """Load one YAML configuration with paths anchored at the repository root."""
    root = (REPO_ROOT if repo_root is None else Path(repo_root)).resolve()
    config_path = _resolve_path(path, root)
    try:
        loaded = yaml.load(
            config_path.read_text(encoding="utf-8"),
            Loader=UniqueKeySafeLoader,
        )
    except yaml.YAMLError as error:
        raise ValueError(f"invalid confidence YAML {config_path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError("confidence configuration must contain a YAML mapping")
    raw: dict[str, Any] = loaded
    _resolve_config_paths(raw, root)
    return ConfidenceConfig.model_validate(raw)
