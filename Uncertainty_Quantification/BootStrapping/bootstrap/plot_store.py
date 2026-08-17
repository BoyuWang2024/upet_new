"""Content-addressed, manifest-last publication for campaign plots."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, sha256_file, sibling_staging
from .campaign import CampaignConfig, CampaignDataset
from .errors import HardFailure
from .plot_analysis import analyze_panel_source, scan_shared_log_limits
from .plot_rendering import (
    panel_stem,
    render_panel,
    render_statistics,
    summarize_panel,
)
from .plot_source import PanelKey, PlotSource, discover_plot_sources


_ANALYSIS_CONTRACT = {
    "correlations": ["spearman_log", "pearson_log10"],
    "ddof": 1,
    "energy": "per_atom",
    "filter": "positive_finite_uncertainty_and_residual",
    "force": "component",
    "stress": "symmetric_voigt_xx_yy_zz_yz_xz_xy",
}
_IDENTITY_KEYS = {
    "schema",
    "runs",
    "datasets",
    "mode",
    "member_count",
    "analysis",
    "style",
    "sources",
}
_MANIFEST_KEYS = {
    "schema",
    "status",
    "identity",
    "identity_document",
    "artifacts",
}


@dataclass(frozen=True)
class PlotIdentity:
    digest: str
    document: Mapping[str, Any]


@dataclass(frozen=True)
class PlotPublication:
    plot_dir: Path
    manifest_path: Path
    skipped: bool


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise HardFailure(f"plot identity is not canonical JSON: {error}") from error


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise HardFailure(f"plot {label} must be a SHA256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise HardFailure(f"plot {label} must be a SHA256 digest") from error
    return value.lower()


def _key_document(key: PanelKey) -> dict[str, str]:
    return {
        "run_label": key.run_label,
        "dataset_label": key.dataset_label,
        "storage_key": key.storage_key,
        "target": key.target,
    }


def _dataset_document(dataset: CampaignDataset) -> dict[str, Any]:
    return {
        "label": dataset.label,
        "storage_key": dataset.storage_key,
        "reference_targets": list(dataset.reference_targets),
    }


def _panel_targets(dataset: CampaignDataset) -> tuple[str, ...]:
    if dataset.reference_targets == ("energy", "forces"):
        return "energy", "force"
    if dataset.reference_targets == ("energy", "forces", "stress"):
        return "energy", "force", "stress"
    raise HardFailure("plot campaign reference targets are invalid")


def _expected_keys(campaign: CampaignConfig) -> tuple[PanelKey, ...]:
    return tuple(
        PanelKey(run.label, dataset.label, dataset.storage_key, target)
        for run in campaign.runs
        for dataset in campaign.datasets
        for target in _panel_targets(dataset)
    )


def _validate_campaign_sources(
    campaign: CampaignConfig, sources: Sequence[PlotSource]
) -> tuple[PlotSource, ...]:
    if not isinstance(campaign, CampaignConfig):
        raise HardFailure("plot identity requires CampaignConfig")
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise HardFailure("plot identity sources must be an ordered sequence")
    normalized = tuple(sources)
    expected_keys = _expected_keys(campaign)
    if not normalized or tuple(source.key for source in normalized) != expected_keys:
        raise HardFailure("plot sources do not match campaign declaration order")
    prediction = campaign.prediction
    if prediction.mode != "raw":
        raise HardFailure("plot identity requires raw prediction mode")
    if type(prediction.member_count) is not int or prediction.member_count < 2:
        raise HardFailure("plot identity member count is invalid")
    for source in normalized:
        if not isinstance(source, PlotSource):
            raise HardFailure("plot identity source must use PlotSource")
        if len(source.member_paths) != prediction.member_count:
            raise HardFailure("plot source member count differs from campaign")
        _require_sha256(
            source.prediction_manifest_sha256, "prediction manifest identity"
        )
        _require_sha256(source.uq_manifest_sha256, "UQ manifest identity")
        _require_sha256(source.targets_sha256, "targets identity")
    return normalized


def compute_plot_identity(
    campaign: CampaignConfig, sources: Sequence[PlotSource]
) -> PlotIdentity:
    """Hash every ordered scientific and rendering input to campaign plots."""

    normalized = _validate_campaign_sources(campaign, sources)
    source_documents: list[dict[str, Any]] = []
    for source in normalized:
        members = [
            {
                "index": index,
                "raw_path": f"{member_path.parent.name}/{member_path.name}",
            }
            for index, member_path in enumerate(source.member_paths)
        ]
        source_documents.append(
            {
                "key": _key_document(source.key),
                "prediction_manifest_sha256": source.prediction_manifest_sha256,
                "uq_manifest_sha256": source.uq_manifest_sha256,
                "targets_sha256": source.targets_sha256,
                "ordered_members": members,
            }
        )
    document = {
        "schema": "upet.bootstrap.plot-identity/v1",
        "runs": [run.label for run in campaign.runs],
        "datasets": [_dataset_document(dataset) for dataset in campaign.datasets],
        "mode": campaign.prediction.mode,
        "member_count": campaign.prediction.member_count,
        "analysis": _ANALYSIS_CONTRACT,
        "style": asdict(campaign.plot),
        "sources": source_documents,
    }
    canonical = _canonical_json(document)
    return PlotIdentity(
        digest=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        document=json.loads(canonical),
    )


def _absolute_lexical(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_symlink_components(path: Path, label: str) -> None:
    candidate = _absolute_lexical(path)
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        if os.path.lexists(current) and current.is_symlink():
            raise HardFailure(f"plot {label} contains a symlink: {current}")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HardFailure(f"could not read plot manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise HardFailure("plot manifest root must be an object")
    return value


def _validate_identity_document(value: object, digest: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise HardFailure("plot identity document schema is invalid")
    document = _json_copy(value)
    if document["schema"] != "upet.bootstrap.plot-identity/v1":
        raise HardFailure("plot identity schema is invalid")
    runs = document["runs"]
    if (
        not isinstance(runs, list)
        or not runs
        or not all(type(item) is str and item for item in runs)
        or len(set(runs)) != len(runs)
    ):
        raise HardFailure("plot identity runs are invalid")
    datasets = document["datasets"]
    if not isinstance(datasets, list) or not datasets:
        raise HardFailure("plot identity datasets are invalid")
    dataset_records: list[dict[str, Any]] = []
    for record in datasets:
        if not isinstance(record, Mapping) or set(record) != {
            "label",
            "storage_key",
            "reference_targets",
        }:
            raise HardFailure("plot identity dataset record is invalid")
        if type(record["label"]) is not str or not record["label"]:
            raise HardFailure("plot identity dataset label is invalid")
        if type(record["storage_key"]) is not str or not record["storage_key"]:
            raise HardFailure("plot identity storage key is invalid")
        targets = record["reference_targets"]
        if targets not in (["energy", "forces"], ["energy", "forces", "stress"]):
            raise HardFailure("plot identity reference targets are invalid")
        dataset_records.append(dict(record))
    if document["mode"] != "raw":
        raise HardFailure("plot identity mode must be raw")
    member_count = document["member_count"]
    if type(member_count) is not int or member_count < 2:
        raise HardFailure("plot identity member count is invalid")
    if document["analysis"] != _ANALYSIS_CONTRACT:
        raise HardFailure("plot identity analysis contract is invalid")
    if not isinstance(document["style"], Mapping) or not document["style"]:
        raise HardFailure("plot identity style is invalid")
    sources = document["sources"]
    if not isinstance(sources, list) or not sources:
        raise HardFailure("plot identity sources are invalid")
    expected_keys = [
        {
            "run_label": run,
            "dataset_label": dataset["label"],
            "storage_key": dataset["storage_key"],
            "target": target,
        }
        for run in runs
        for dataset in dataset_records
        for target in (
            ["energy", "force"]
            if dataset["reference_targets"] == ["energy", "forces"]
            else ["energy", "force", "stress"]
        )
    ]
    actual_keys: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping) or set(source) != {
            "key",
            "prediction_manifest_sha256",
            "uq_manifest_sha256",
            "targets_sha256",
            "ordered_members",
        }:
            raise HardFailure("plot identity source record is invalid")
        key = source["key"]
        if not isinstance(key, Mapping) or set(key) != {
            "run_label",
            "dataset_label",
            "storage_key",
            "target",
        }:
            raise HardFailure("plot identity panel key is invalid")
        actual_keys.append(dict(key))
        _require_sha256(
            source["prediction_manifest_sha256"], "prediction manifest identity"
        )
        _require_sha256(source["uq_manifest_sha256"], "UQ manifest identity")
        _require_sha256(source["targets_sha256"], "targets identity")
        members = source["ordered_members"]
        if not isinstance(members, list) or len(members) != member_count:
            raise HardFailure("plot identity ordered members are incomplete")
        for index, member in enumerate(members):
            if not isinstance(member, Mapping) or set(member) != {
                "index",
                "raw_path",
            }:
                raise HardFailure("plot identity member record is invalid")
            if type(member["index"]) is not int or member["index"] != index:
                raise HardFailure("plot identity member order is invalid")
            raw_path = member["raw_path"]
            if type(raw_path) is not str:
                raise HardFailure("plot identity member path is invalid")
            parts = raw_path.split("/")
            if (
                len(parts) != 2
                or not parts[0].startswith("member_")
                or parts[1] != "raw.npz"
            ):
                raise HardFailure("plot identity member path is invalid")
    if actual_keys != expected_keys:
        raise HardFailure("plot identity source order is invalid")
    actual_digest = hashlib.sha256(
        _canonical_json(document).encode("utf-8")
    ).hexdigest()
    if actual_digest != digest:
        raise HardFailure("plot identity document digest does not match")
    return document


def _expected_payload(identity_document: Mapping[str, Any]) -> set[str]:
    names = {"raw_std_statistics.csv", "raw_std_statistics.json"}
    for source in identity_document["sources"]:
        key = PanelKey(**source["key"])
        stem = panel_stem(key)
        names.add(f"{stem}.png")
        names.add(f"{stem}.pdf")
    return names


def validate_plot_publication(
    publication_root: str | Path, *, expected_identity: str | None = None
) -> Mapping[str, Any]:
    """Strictly audit one complete content-addressed plot publication."""

    root = _absolute_lexical(publication_root)
    _reject_symlink_components(root, "publication root")
    if root.is_symlink() or not root.is_dir():
        raise HardFailure("plot publication directory is missing or unsafe")
    manifest_path = root / "plot_manifest.json"
    _reject_symlink_components(manifest_path, "manifest path")
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise HardFailure("plot publication manifest is missing or unsafe")
    manifest = _read_json(manifest_path)
    if set(manifest) != _MANIFEST_KEYS:
        raise HardFailure("plot manifest schema is invalid")
    if manifest["schema"] != "upet.bootstrap.plot-publication/v1":
        raise HardFailure("plot publication schema is invalid")
    if manifest["status"] != "PASS":
        raise HardFailure("plot publication status must be PASS")
    identity = _require_sha256(manifest["identity"], "publication identity")
    if expected_identity is not None and identity != _require_sha256(
        expected_identity, "expected identity"
    ):
        raise HardFailure("plot publication identity differs from request")
    identity_document = _validate_identity_document(
        manifest["identity_document"], identity
    )
    expected_payload = _expected_payload(identity_document)
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_payload:
        raise HardFailure("plot artifact inventory is incomplete or has extras")
    actual_entries = {entry.name for entry in root.iterdir()}
    if actual_entries != expected_payload | {"plot_manifest.json"}:
        raise HardFailure("plot publication contains missing or extra artifacts")
    for name in sorted(expected_payload):
        record = artifacts[name]
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "size",
            "sha256",
        }:
            raise HardFailure(f"plot artifact record is invalid: {name}")
        if record["path"] != name:
            raise HardFailure(f"plot artifact path is noncanonical: {name}")
        if type(record["size"]) is not int or record["size"] <= 0:
            raise HardFailure(f"plot artifact size is invalid: {name}")
        digest = _require_sha256(record["sha256"], f"artifact {name} hash")
        candidate = root / name
        _reject_symlink_components(candidate, f"artifact {name}")
        if candidate.is_symlink():
            raise HardFailure(f"plot artifact is a symlink: {name}")
        try:
            candidate_stat = candidate.stat()
        except OSError as error:
            raise HardFailure(f"plot artifact is missing: {name}") from error
        if not stat.S_ISREG(candidate_stat.st_mode):
            raise HardFailure(f"plot artifact is not a regular file: {name}")
        if candidate_stat.st_size != record["size"]:
            raise HardFailure(f"plot artifact size changed: {name}")
        if sha256_file(candidate) != digest:
            raise HardFailure(f"plot artifact hash changed: {name}")
    return manifest


def _artifact_records(root: Path, names: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(names):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise HardFailure(f"plot artifact was not rendered: {name}")
        records[name] = {
            "path": name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _is_formal_campaign(campaign: CampaignConfig, source_count: int) -> bool:
    return (
        len(campaign.runs) == 3 and len(campaign.datasets) == 3 and source_count == 24
    )


def publish_campaign_plots(campaign: CampaignConfig) -> PlotPublication:
    """Analyze, render, and transactionally publish one campaign plot set."""

    if not isinstance(campaign, CampaignConfig):
        raise HardFailure("plot publication requires CampaignConfig")
    sources = discover_plot_sources(campaign)
    identity = compute_plot_identity(campaign, sources)
    prefix = (
        "raw_std_three_runs_three_datasets"
        if _is_formal_campaign(campaign, len(sources))
        else "raw_std_campaign"
    )
    destination = _absolute_lexical(
        campaign.output_root / f"{prefix}__{identity.digest[:16]}"
    )
    if destination.exists() or destination.is_symlink():
        validate_plot_publication(destination, expected_identity=identity.digest)
        return PlotPublication(
            plot_dir=destination,
            manifest_path=destination / "plot_manifest.json",
            skipped=True,
        )

    with sibling_staging(destination) as staging:
        limits = scan_shared_log_limits(sources, margin=campaign.plot.log_margin)
        summaries = []
        for source in sources:
            analysis = analyze_panel_source(
                source, limits[source.key.target], campaign.plot
            )
            render_panel(analysis, campaign.plot, staging)
            summaries.append(summarize_panel(analysis))
            del analysis
        render_statistics(tuple(summaries), sources, campaign.plot, staging)
        expected_payload = {
            f"{panel_stem(source.key)}.{suffix}"
            for source in sources
            for suffix in campaign.plot.formats
        } | {"raw_std_statistics.csv", "raw_std_statistics.json"}
        actual_payload = {entry.name for entry in staging.iterdir()}
        if actual_payload != expected_payload:
            raise HardFailure("plot renderer produced a missing or extra artifact")
        manifest = {
            "schema": "upet.bootstrap.plot-publication/v1",
            "status": "PASS",
            "identity": identity.digest,
            "identity_document": identity.document,
            "artifacts": _artifact_records(staging, expected_payload),
        }
        atomic_write_json(staging / "plot_manifest.json", manifest)
        validate_plot_publication(staging, expected_identity=identity.digest)

    validate_plot_publication(destination, expected_identity=identity.digest)
    return PlotPublication(
        plot_dir=destination,
        manifest_path=destination / "plot_manifest.json",
        skipped=False,
    )


__all__ = [
    "PlotIdentity",
    "PlotPublication",
    "compute_plot_identity",
    "publish_campaign_plots",
    "validate_plot_publication",
]
