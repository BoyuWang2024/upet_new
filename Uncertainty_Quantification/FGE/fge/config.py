"""Strict, reproducible configuration loading for the UPET FGE workflow."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, cast

import yaml

from .errors import HardFailure


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PATH_ENVIRONMENT = {
    "UPET_FGE_BASE_CHECKPOINT",
    "UPET_FGE_TRAIN_DATA",
    "UPET_FGE_VAL_DATA",
    "UPET_FGE_TEST_DATA",
    "UPET_FGE_OUTPUT_ROOT",
}
_BASE_CHECKPOINT_SHA256 = (
    "879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004"
)
_TRAIN_DATA_SHA256 = "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec"
_TEST_DATA_SHA256 = "1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d"
_FULL_SHA = {
    "base_checkpoint_sha256": _BASE_CHECKPOINT_SHA256,
    "train_data_sha256": _TRAIN_DATA_SHA256,
    "val_data_sha256": _TRAIN_DATA_SHA256,
    "test_data_sha256": _TEST_DATA_SHA256,
}
_ALLOWED_KEYS = {
    "root": {
        "schema_version",
        "project",
        "paths",
        "identity",
        "data",
        "training",
        "fge",
        "ema",
        "prediction",
        "evaluation",
        "scientific",
    },
    "project": {"name", "method", "backend"},
    "paths": {"base_checkpoint", "train_data", "val_data", "test_data", "output_root"},
    "identity": {
        "base_checkpoint_sha256",
        "train_data_sha256",
        "val_data_sha256",
        "test_data_sha256",
    },
    "data": {
        "format",
        "energy_target",
        "forces_target",
        "stress_target",
        "energy_unit",
        "forces_unit",
        "stress_unit",
    },
    "training": {
        "mode",
        "restart_state",
        "trainable_prefixes",
        "expected_readout_tensor_count",
        "expected_readout_parameter_count",
        "seed",
        "batch_size",
        "validation_batch_size",
        "drop_last",
        "weight_decay",
        "dtype",
        "device",
        "num_workers",
        "resume",
    },
    "fge": {
        "member_count",
        "cycles",
        "epochs_per_cycle",
        "schedule",
        "lr_min",
        "lr_max",
        "rise_fraction",
    },
    "ema": {"enabled", "decay", "member_source", "usage"},
    "prediction": {"split", "batch_size", "compute_stress"},
    "evaluation": {
        "formula_version",
        "metric_schema_version",
        "structure_quantile",
        "constant_tolerance",
        "risk_coverages",
    },
    "scientific": {"training", "evaluation"},
    "scientific.training": {
        "path_feasibility_only",
        "split_leakage",
        "scientific_evaluation",
        "inference_only",
    },
    "scientific.evaluation": {
        "path_feasibility_only",
        "split_leakage",
        "scientific_evaluation",
        "inference_only",
    },
}


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    method: str
    backend: str


@dataclass(frozen=True)
class PathsConfig:
    base_checkpoint: Path
    train_data: Path
    val_data: Path
    test_data: Path
    output_root: Path


@dataclass(frozen=True)
class IdentityConfig:
    base_checkpoint_sha256: str
    train_data_sha256: str
    val_data_sha256: str
    test_data_sha256: str


@dataclass(frozen=True)
class DataConfig:
    format: str
    energy_target: str
    forces_target: str
    stress_target: str
    energy_unit: str
    forces_unit: str
    stress_unit: str


@dataclass(frozen=True)
class TrainingConfig:
    mode: str
    restart_state: str
    trainable_prefixes: tuple[str, ...]
    expected_readout_tensor_count: int
    expected_readout_parameter_count: int
    seed: int
    batch_size: int
    validation_batch_size: int
    drop_last: bool
    weight_decay: float
    dtype: str
    device: str
    num_workers: int
    resume: bool


@dataclass(frozen=True)
class FGEScheduleConfig:
    member_count: int
    cycles: int
    epochs_per_cycle: int
    schedule: str
    lr_min: float
    lr_max: float
    rise_fraction: float


@dataclass(frozen=True)
class EMAConfig:
    enabled: bool
    decay: float
    member_source: str
    usage: str


@dataclass(frozen=True)
class PredictionConfig:
    split: str
    batch_size: int
    compute_stress: bool


@dataclass(frozen=True)
class EvaluationConfig:
    formula_version: str
    metric_schema_version: int
    structure_quantile: float
    constant_tolerance: float
    risk_coverages: tuple[float, ...]


@dataclass(frozen=True)
class ScientificStateConfig:
    path_feasibility_only: bool
    split_leakage: bool
    scientific_evaluation: bool
    inference_only: bool


@dataclass(frozen=True)
class ScientificConfig:
    training: ScientificStateConfig
    evaluation: ScientificStateConfig


@dataclass(frozen=True)
class RuntimeConfig:
    source_path: Path


@dataclass(frozen=True)
class FGEConfig:
    schema_version: str
    project: ProjectConfig
    paths: PathsConfig
    identity: IdentityConfig
    data: DataConfig
    training: TrainingConfig
    fge: FGEScheduleConfig
    ema: EMAConfig
    prediction: PredictionConfig
    evaluation: EvaluationConfig
    scientific: ScientificConfig
    runtime: RuntimeConfig

    def sanitized(self) -> dict[str, object]:
        """Return reproducibility metadata without runtime filesystem locations."""
        payload = cast(dict[str, object], asdict(self))
        payload.pop("runtime")
        payload["paths"] = {
            "base_checkpoint": {
                "role": "base_checkpoint",
                "sha256": self.identity.base_checkpoint_sha256,
            },
            "train_data": {
                "role": "train_data",
                "sha256": self.identity.train_data_sha256,
            },
            "val_data": {"role": "val_data", "sha256": self.identity.val_data_sha256},
            "test_data": {
                "role": "test_data",
                "sha256": self.identity.test_data_sha256,
            },
            "output_root": {"role": "output_root"},
        }
        return payload

    def assert_stage(self, stage: str) -> None:
        """Ensure that *stage* is one of the supported execution stages."""
        if stage not in {"train", "predict", "evaluate"}:
            raise HardFailure(f"unsupported FGE stage: {stage}")


def _mapping(value: object, location: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise HardFailure(f"{location} must be a mapping")
    if any(not isinstance(key, str) for key in value):
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


def _string(value: object, location: str) -> str:
    if type(value) is not str:
        raise HardFailure(f"{location} must be a string")
    return cast(str, value)


def _integer(value: object, location: str) -> int:
    if type(value) is not int:
        raise HardFailure(f"{location} must be an integer")
    return cast(int, value)


def _boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise HardFailure(f"{location} must be a boolean")
    return cast(bool, value)


def _float(value: object, location: str) -> float:
    if type(value) is not float:
        raise HardFailure(f"{location} must be a float")
    return cast(float, value)


def _strings(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise HardFailure(f"{location} must be a list of strings")
    return tuple(cast(str, item) for item in value)


def _floats(value: object, location: str) -> tuple[float, ...]:
    if not isinstance(value, list) or any(type(item) is not float for item in value):
        raise HardFailure(f"{location} must be a list of floats")
    return tuple(cast(float, item) for item in value)


def _path(value: object, location: str, source_directory: Path) -> Path:
    raw = _string(value, location)
    match = re.fullmatch(r"\$\{([^}]+)\}", raw)
    if match is not None:
        name = match.group(1)
        if name not in _PATH_ENVIRONMENT:
            raise HardFailure(f"unsupported environment variable in {location}: {name}")
        if name not in os.environ:
            raise HardFailure(
                f"environment variable is required for {location}: {name}"
            )
        raw = os.environ[name]
    elif "${" in raw:
        raise HardFailure(f"unsupported environment expression in {location}")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = source_directory / candidate
    return candidate.resolve()


def _section(root: Mapping[str, object], name: str) -> Mapping[str, object]:
    section = _mapping(root[name], name)
    _strict_keys(section, name)
    return section


def _state(value: object, location: str) -> ScientificStateConfig:
    section = _mapping(value, location)
    _strict_keys(section, location)
    return ScientificStateConfig(
        path_feasibility_only=_boolean(
            section["path_feasibility_only"], f"{location}.path_feasibility_only"
        ),
        split_leakage=_boolean(section["split_leakage"], f"{location}.split_leakage"),
        scientific_evaluation=_boolean(
            section["scientific_evaluation"], f"{location}.scientific_evaluation"
        ),
        inference_only=_boolean(
            section["inference_only"], f"{location}.inference_only"
        ),
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HardFailure(message)


def _validate_contract(config: FGEConfig) -> None:
    _require(config.schema_version == "upet.fge.v1", "schema_version is fixed")
    _require(
        config.project.name in {"upet_fge_full", "upet_fge_n20_cpu"},
        "project.name is invalid",
    )
    _require(
        (config.project.method, config.project.backend) == ("FGE", "upet"),
        "project method and backend are fixed",
    )
    _require(
        (
            config.data.format,
            config.data.energy_target,
            config.data.forces_target,
            config.data.stress_target,
            config.data.energy_unit,
            config.data.forces_unit,
            config.data.stress_unit,
        )
        == (
            "extxyz",
            "energy",
            "non_conservative_forces",
            "non_conservative_stress",
            "eV",
            "eV/angstrom",
            "eV/angstrom^3",
        ),
        "data contract is fixed",
    )
    _require(
        (
            config.training.mode,
            config.training.restart_state,
            config.training.trainable_prefixes,
            config.training.expected_readout_tensor_count,
            config.training.expected_readout_parameter_count,
            config.training.seed,
            config.training.drop_last,
            config.training.weight_decay,
            config.training.dtype,
            config.training.num_workers,
        )
        == (
            "readout_only_official_upet",
            "model_state_dict",
            ("node_last_layers.", "edge_last_layers."),
            12,
            13338,
            2026,
            True,
            0.0,
            "float32",
            0,
        ),
        "training contract is fixed",
    )
    _require(
        (
            config.fge.schedule,
            config.fge.lr_min,
            config.fge.lr_max,
            config.fge.rise_fraction,
        )
        == ("asymmetric_triangular", 1e-8, 1e-7, 0.2),
        "FGE schedule contract is fixed",
    )
    _require(
        (
            config.ema.enabled,
            config.ema.decay,
            config.ema.member_source,
            config.ema.usage,
        )
        == (True, 0.999, "raw_endpoint", "validation_only"),
        "EMA contract is fixed",
    )
    _require(
        (
            config.prediction.split,
            config.prediction.compute_stress,
            config.evaluation.formula_version,
            config.evaluation.metric_schema_version,
            config.evaluation.structure_quantile,
            config.evaluation.constant_tolerance,
            config.evaluation.risk_coverages,
        )
        == (
            "test",
            True,
            "legacy_upet_fge_v1",
            4,
            0.95,
            1e-12,
            (1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1),
        ),
        "prediction and evaluation contract is fixed",
    )
    is_n20 = config.project.name == "upet_fge_n20_cpu"
    expected_run = (
        (4, 4, "cpu", False, 2, 2, 2, 4)
        if is_n20
        else (16, 16, "cuda", True, 8, 8, 8, 16)
    )
    actual_run = (
        config.training.batch_size,
        config.training.validation_batch_size,
        config.training.device,
        config.training.resume,
        config.fge.member_count,
        config.fge.cycles,
        config.fge.epochs_per_cycle,
        config.prediction.batch_size,
    )
    _require(actual_run == expected_run, "runtime scale contract is invalid")
    expected_training = ScientificStateConfig(True, True, False, False)
    expected_evaluation = ScientificStateConfig(
        True if is_n20 else False,
        True if is_n20 else False,
        False if is_n20 else True,
        True,
    )
    _require(
        config.scientific.training == expected_training,
        "training scientific flags are invalid",
    )
    _require(
        config.scientific.evaluation == expected_evaluation,
        "evaluation scientific flags are invalid",
    )
    identities = asdict(config.identity)
    if is_n20:
        _require(
            identities["base_checkpoint_sha256"] == _FULL_SHA["base_checkpoint_sha256"],
            "n20 base checkpoint identity is invalid",
        )
        _require(
            identities["train_data_sha256"]
            == identities["val_data_sha256"]
            == identities["test_data_sha256"],
            "n20 data identities must match",
        )
    else:
        _require(identities == _FULL_SHA, "full identities are invalid")


def load_config(path: str | Path) -> FGEConfig:
    """Load and validate a single exact UPET FGE configuration document."""
    source_path = Path(path).resolve()
    try:
        parsed = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardFailure(f"unable to load configuration {source_path}: {exc}") from exc
    root = _mapping(parsed, "root")
    _strict_keys(root, "root")
    project = _section(root, "project")
    paths = _section(root, "paths")
    identity = _section(root, "identity")
    data = _section(root, "data")
    training = _section(root, "training")
    fge = _section(root, "fge")
    ema = _section(root, "ema")
    prediction = _section(root, "prediction")
    evaluation = _section(root, "evaluation")
    scientific = _section(root, "scientific")
    config = FGEConfig(
        schema_version=_string(root["schema_version"], "schema_version"),
        project=ProjectConfig(
            *(
                _string(project[key], f"project.{key}")
                for key in ("name", "method", "backend")
            )
        ),
        paths=PathsConfig(
            *(
                _path(paths[key], f"paths.{key}", source_path.parent)
                for key in (
                    "base_checkpoint",
                    "train_data",
                    "val_data",
                    "test_data",
                    "output_root",
                )
            )
        ),
        identity=IdentityConfig(
            *(
                _string(identity[key], f"identity.{key}")
                for key in (
                    "base_checkpoint_sha256",
                    "train_data_sha256",
                    "val_data_sha256",
                    "test_data_sha256",
                )
            )
        ),
        data=DataConfig(
            *(
                _string(data[key], f"data.{key}")
                for key in (
                    "format",
                    "energy_target",
                    "forces_target",
                    "stress_target",
                    "energy_unit",
                    "forces_unit",
                    "stress_unit",
                )
            )
        ),
        training=TrainingConfig(
            _string(training["mode"], "training.mode"),
            _string(training["restart_state"], "training.restart_state"),
            _strings(training["trainable_prefixes"], "training.trainable_prefixes"),
            _integer(
                training["expected_readout_tensor_count"],
                "training.expected_readout_tensor_count",
            ),
            _integer(
                training["expected_readout_parameter_count"],
                "training.expected_readout_parameter_count",
            ),
            _integer(training["seed"], "training.seed"),
            _integer(training["batch_size"], "training.batch_size"),
            _integer(
                training["validation_batch_size"], "training.validation_batch_size"
            ),
            _boolean(training["drop_last"], "training.drop_last"),
            _float(training["weight_decay"], "training.weight_decay"),
            _string(training["dtype"], "training.dtype"),
            _string(training["device"], "training.device"),
            _integer(training["num_workers"], "training.num_workers"),
            _boolean(training["resume"], "training.resume"),
        ),
        fge=FGEScheduleConfig(
            member_count=_integer(fge["member_count"], "fge.member_count"),
            cycles=_integer(fge["cycles"], "fge.cycles"),
            epochs_per_cycle=_integer(fge["epochs_per_cycle"], "fge.epochs_per_cycle"),
            schedule=_string(fge["schedule"], "fge.schedule"),
            lr_min=_float(fge["lr_min"], "fge.lr_min"),
            lr_max=_float(fge["lr_max"], "fge.lr_max"),
            rise_fraction=_float(fge["rise_fraction"], "fge.rise_fraction"),
        ),
        ema=EMAConfig(
            _boolean(ema["enabled"], "ema.enabled"),
            _float(ema["decay"], "ema.decay"),
            _string(ema["member_source"], "ema.member_source"),
            _string(ema["usage"], "ema.usage"),
        ),
        prediction=PredictionConfig(
            _string(prediction["split"], "prediction.split"),
            _integer(prediction["batch_size"], "prediction.batch_size"),
            _boolean(prediction["compute_stress"], "prediction.compute_stress"),
        ),
        evaluation=EvaluationConfig(
            _string(evaluation["formula_version"], "evaluation.formula_version"),
            _integer(
                evaluation["metric_schema_version"], "evaluation.metric_schema_version"
            ),
            _float(evaluation["structure_quantile"], "evaluation.structure_quantile"),
            _float(evaluation["constant_tolerance"], "evaluation.constant_tolerance"),
            _floats(evaluation["risk_coverages"], "evaluation.risk_coverages"),
        ),
        scientific=ScientificConfig(
            _state(scientific["training"], "scientific.training"),
            _state(scientific["evaluation"], "scientific.evaluation"),
        ),
        runtime=RuntimeConfig(source_path=source_path),
    )
    for key, value in asdict(config.identity).items():
        _require(bool(_SHA256_RE.fullmatch(value)), f"{key} must be lowercase SHA256")
    _validate_contract(config)
    return config
