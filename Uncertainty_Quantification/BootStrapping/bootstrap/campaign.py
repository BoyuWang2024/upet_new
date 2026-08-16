"""Strict configuration for a multi-run, multi-dataset Bootstrap campaign."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, TypeVar

import yaml

from .config import BootstrapConfig, load_config
from .errors import HardFailure
from .identifiers import validate_artifact_key


@dataclass(frozen=True)
class CampaignRun:
    label: str
    config_path: Path
    run_root: Path
    config: BootstrapConfig


@dataclass(frozen=True)
class CampaignDataset:
    label: str
    storage_key: str
    path: Path
    reference_targets: tuple[str, ...]


@dataclass(frozen=True)
class CampaignPrediction:
    mode: str
    member_count: int
    device: str
    batch_size: int


@dataclass(frozen=True)
class CampaignPlotStyle:
    grid_size: int
    gaussian_sigma: float
    contour_masses: tuple[float, ...]
    scatter_max_points: int
    scatter_seed: int
    scatter_size: float
    scatter_alpha: float
    log_margin: float
    figure_size: tuple[float, float]
    dpi: int
    formats: tuple[str, ...]


@dataclass(frozen=True)
class CampaignConfig:
    schema_version: int
    runs: tuple[CampaignRun, ...]
    datasets: tuple[CampaignDataset, ...]
    prediction: CampaignPrediction
    plot: CampaignPlotStyle
    output_root: Path
    source_path: Path


T = TypeVar("T")
_REFERENCE_TARGETS = ("energy", "forces", "stress")


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{location} must be a mapping")
    return value


def _keys(value: object, location: str, required: set[str]) -> Mapping[str, Any]:
    result = _mapping(value, location)
    unknown = [key for key in result if key not in required]
    if unknown:
        raise HardFailure(f"unknown key {location}.{sorted(map(str, unknown))[0]}")
    present = {key for key in result if isinstance(key, str)}
    missing = required - present
    if missing:
        raise HardFailure(f"missing key {location}.{sorted(missing)[0]}")
    return result


def _typed(value: object, expected: type[T], location: str) -> T:
    if expected is int and isinstance(value, bool):
        raise HardFailure(f"{location} must be an integer")
    if expected is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)  # type: ignore[return-value]
    if not isinstance(value, expected):
        raise HardFailure(f"{location} must be {expected.__name__}")
    return value


def _nonempty_text(value: object, location: str) -> str:
    result = _typed(value, str, location)
    if not result:
        raise HardFailure(f"{location} must not be empty")
    return result


def _positive_int(value: object, location: str) -> int:
    result = _typed(value, int, location)
    if result < 1:
        raise HardFailure(f"{location} must be positive")
    return result


def _nonnegative_int(value: object, location: str) -> int:
    result = _typed(value, int, location)
    if result < 0:
        raise HardFailure(f"{location} must be non-negative")
    return result


def _positive_float(value: object, location: str) -> float:
    result = _typed(value, float, location)
    if not math.isfinite(result) or result <= 0:
        raise HardFailure(f"{location} must be finite and positive")
    return result


def _path(value: object, base: Path, location: str) -> Path:
    path = Path(_nonempty_text(value, location)).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _unique_labels(items: Sequence[object], kind: str) -> tuple[str, ...]:
    labels: list[str] = []
    for index, item in enumerate(items):
        record = _mapping(item, f"{kind}[{index}]")
        labels.append(_nonempty_text(record.get("label"), f"{kind}[{index}].label"))
    duplicates = {label for label in labels if labels.count(label) > 1}
    if duplicates:
        raise HardFailure(f"duplicate {kind[:-1]} label {sorted(duplicates)[0]}")
    return tuple(labels)


def _reference_targets(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise HardFailure(f"{location} must be a non-empty list")
    targets = tuple(_nonempty_text(item, location) for item in value)
    if len(set(targets)) != len(targets):
        raise HardFailure(f"{location} must not contain duplicates")
    if targets not in (_REFERENCE_TARGETS[:2], _REFERENCE_TARGETS):
        raise HardFailure(f"{location} must be [energy, forces] or [energy, forces, stress]")
    return targets


def _load_prediction(value: object) -> CampaignPrediction:
    record = _keys(
        value,
        "prediction",
        {"mode", "member_count", "device", "batch_size"},
    )
    mode = _nonempty_text(record["mode"], "prediction.mode")
    if mode != "raw":
        raise HardFailure("prediction.mode must be raw")
    member_count = _positive_int(record["member_count"], "prediction.member_count")
    if member_count < 2:
        raise HardFailure("prediction.member_count must be at least 2")
    return CampaignPrediction(
        mode=mode,
        member_count=member_count,
        device=_nonempty_text(record["device"], "prediction.device"),
        batch_size=_positive_int(record["batch_size"], "prediction.batch_size"),
    )


def _load_plot(value: object) -> CampaignPlotStyle:
    record = _keys(
        value,
        "plot",
        {
            "grid_size",
            "gaussian_sigma",
            "contour_masses",
            "scatter_max_points",
            "scatter_seed",
            "scatter_size",
            "scatter_alpha",
            "log_margin",
            "figure_size",
            "dpi",
            "formats",
        },
    )
    raw_masses = record["contour_masses"]
    if not isinstance(raw_masses, list) or not raw_masses:
        raise HardFailure("plot.contour_masses must be a non-empty list")
    masses = tuple(
        _typed(item, float, "plot.contour_masses") for item in raw_masses
    )
    if any(not 0 < mass < 1 for mass in masses) or any(
        left >= right for left, right in zip(masses, masses[1:])
    ):
        raise HardFailure(
            "plot.contour_masses must be strictly increasing inside (0, 1)"
        )
    raw_size = record["figure_size"]
    if not isinstance(raw_size, list) or len(raw_size) != 2:
        raise HardFailure("plot.figure_size must contain two values")
    figure_size = tuple(_positive_float(item, "plot.figure_size") for item in raw_size)
    raw_formats = record["formats"]
    if not isinstance(raw_formats, list):
        raise HardFailure("plot.formats must be a list")
    formats = tuple(_nonempty_text(item, "plot.formats") for item in raw_formats)
    if formats != ("png", "pdf"):
        raise HardFailure("plot.formats must be [png, pdf]")
    scatter_alpha = _positive_float(record["scatter_alpha"], "plot.scatter_alpha")
    if scatter_alpha > 1:
        raise HardFailure("plot.scatter_alpha must not exceed 1")
    return CampaignPlotStyle(
        grid_size=_positive_int(record["grid_size"], "plot.grid_size"),
        gaussian_sigma=_positive_float(record["gaussian_sigma"], "plot.gaussian_sigma"),
        contour_masses=masses,
        scatter_max_points=_positive_int(
            record["scatter_max_points"], "plot.scatter_max_points"
        ),
        scatter_seed=_nonnegative_int(record["scatter_seed"], "plot.scatter_seed"),
        scatter_size=_positive_float(record["scatter_size"], "plot.scatter_size"),
        scatter_alpha=scatter_alpha,
        log_margin=_positive_float(record["log_margin"], "plot.log_margin"),
        figure_size=(figure_size[0], figure_size[1]),
        dpi=_positive_int(record["dpi"], "plot.dpi"),
        formats=formats,
    )


def load_campaign(source: str | Path) -> CampaignConfig:
    """Load a campaign YAML file and validate its complete static contract."""

    source_path = Path(source).expanduser().resolve()
    try:
        document = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise HardFailure(f"could not load campaign {source_path}: {error}") from error
    root = _keys(
        document,
        "campaign",
        {"schema_version", "runs", "datasets", "prediction", "plot", "output_root"},
    )
    schema_version = _typed(root["schema_version"], int, "schema_version")
    if schema_version != 1:
        raise HardFailure("schema_version must be 1")
    prediction = _load_prediction(root["prediction"])
    base = source_path.parent

    raw_runs = root["runs"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise HardFailure("runs must be a non-empty list")
    _unique_labels(raw_runs, "runs")
    runs: list[CampaignRun] = []
    for index, value in enumerate(raw_runs):
        location = f"runs[{index}]"
        record = _keys(value, location, {"label", "config", "root"})
        label = _nonempty_text(record["label"], f"{location}.label")
        config_path = _path(record["config"], base, f"{location}.config")
        config = load_config(config_path)
        if config.experiment.run_id != label:
            raise HardFailure(f"{location} run_id must match its label")
        if config.bootstrap.ensemble_size != prediction.member_count:
            raise HardFailure(
                f"{location} ensemble size must match prediction.member_count"
            )
        runs.append(
            CampaignRun(
                label=label,
                config_path=config_path,
                run_root=_path(record["root"], base, f"{location}.root"),
                config=config,
            )
        )

    raw_datasets = root["datasets"]
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise HardFailure("datasets must be a non-empty list")
    _unique_labels(raw_datasets, "datasets")
    datasets: list[CampaignDataset] = []
    for index, value in enumerate(raw_datasets):
        location = f"datasets[{index}]"
        record = _keys(value, location, {"label", "storage_key", "path", "reference_targets"})
        datasets.append(
            CampaignDataset(
                label=_nonempty_text(record["label"], f"{location}.label"),
                storage_key=validate_artifact_key(
                    record["storage_key"], f"{location}.storage_key"
                ),
                path=_path(record["path"], base, f"{location}.path"),
                reference_targets=_reference_targets(
                    record["reference_targets"], f"{location}.reference_targets"
                ),
            )
        )
    storage_keys = [dataset.storage_key for dataset in datasets]
    duplicates = {key for key in storage_keys if storage_keys.count(key) > 1}
    if duplicates:
        raise HardFailure(f"duplicate dataset storage_key {sorted(duplicates)[0]}")

    return CampaignConfig(
        schema_version=schema_version,
        runs=tuple(runs),
        datasets=tuple(datasets),
        prediction=prediction,
        plot=_load_plot(root["plot"]),
        output_root=_path(root["output_root"], base, "output_root"),
        source_path=source_path,
    )


def _select_items(
    items: tuple[T, ...],
    labels: Sequence[str] | None,
    kind: str,
) -> tuple[T, ...]:
    if labels is None:
        return items
    requested = tuple(labels)
    if not requested:
        raise HardFailure(f"{kind} selection must not be empty")
    if len(set(requested)) != len(requested):
        raise HardFailure(f"{kind} selection must not contain duplicates")
    available = {item.label for item in items}  # type: ignore[attr-defined]
    unknown = set(requested) - available
    if unknown:
        raise HardFailure(f"unknown {kind} label {sorted(unknown)[0]}")
    wanted = set(requested)
    return tuple(item for item in items if item.label in wanted)  # type: ignore[attr-defined]


def select_campaign_items(
    campaign: CampaignConfig,
    run_labels: Sequence[str] | None,
    dataset_labels: Sequence[str] | None,
) -> tuple[tuple[CampaignRun, ...], tuple[CampaignDataset, ...]]:
    """Select campaign runs and datasets while retaining declaration order."""

    return (
        _select_items(campaign.runs, run_labels, "run"),
        _select_items(campaign.datasets, dataset_labels, "dataset"),
    )
