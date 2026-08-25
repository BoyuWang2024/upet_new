"""Energy-only continuous density plots for E0 postprocessing variants."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import torch


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from .artifacts import atomic_write_json, load_verified_torch, sha256_file
from .density_plotting import (
    DensitySettings,
    analyze_density_panel,
    shared_log_limits,
)
from .density_publication import _artifact_descriptors
from .density_rendering import (
    _draw_density_axis,
    _save_figure,
    render_density_panel,
    render_energy_comparison,
)
from .e0_publication import E0_VARIANTS, verify_e0_campaign
from .plot_analysis import PlotSeries


E0_DENSITY_SCHEMA_VERSION = "upet_confidence_e0_density_v1"


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"e0-density-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _mapping(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a mapping")
    return value


def _confined(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("E0 density artifact path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("E0 density artifact escapes publication root")
    return path


def _referenced(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("E0 density input path must be relative")
    return (root / relative).resolve()


def _tensor(
    mapping: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if name not in mapping:
        raise ValueError(f"E0 density input is missing {name}")
    result = torch.as_tensor(mapping[name]).detach().cpu()
    if dtype is not None:
        result = result.to(dtype)
    if result.numel() == 0 or (
        result.is_floating_point() and not bool(torch.isfinite(result).all())
    ):
        raise ValueError(f"E0 density input {name} must be finite and non-empty")
    return result


def build_variant_series(
    raw: PlotSeries,
    observed: torch.Tensor,
) -> PlotSeries:
    """Preserve raw UQ tensors while substituting a corrected observed error."""

    corrected = torch.as_tensor(observed, dtype=torch.float64).reshape(-1)
    if raw.target != "energy" or raw.order not in range(1, 9):
        raise ValueError("E0 density series must be an energy order 1 through 8")
    if corrected.shape != raw.expected.reshape(-1).shape:
        raise ValueError("corrected observed errors must match raw expected errors")
    if not bool(torch.isfinite(corrected).all()) or bool(torch.any(corrected < 0)):
        raise ValueError("corrected observed errors must be finite and non-negative")
    return replace(raw, observed=corrected)


def _load_torch(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    value = load_verified_torch(
        path,
        expected_sha256=expected_sha256,
        weights_only=True,
    )
    if not isinstance(value, Mapping):
        raise ValueError(f"E0 density torch artifact must contain a mapping: {path}")
    return cast(Mapping[str, Any], value)


def load_e0_density_series(
    campaign_root: Path,
) -> dict[str, dict[int, PlotSeries]]:
    """Combine referenced raw UQ with each campaign's corrected observations."""

    root = Path(campaign_root).resolve()
    campaign = verify_e0_campaign(root / "manifest.json", full=True)
    payload = campaign["identity_payload"]
    raw_inputs = payload["raw_inputs"]["test"]
    artifacts = campaign["artifacts"]
    variants = campaign["variants"]
    result: dict[str, dict[int, PlotSeries]] = {}
    for variant in E0_VARIANTS:
        variant_entry = variants[variant]
        data_relative = str(variant_entry["energy_data"])
        data_descriptor = artifacts[data_relative]
        data = _load_torch(
            root / data_relative,
            str(data_descriptor["sha256"]),
        )
        metrics = _mapping(
            root / str(variant_entry["metrics"]),
            context=f"{variant} metrics",
        )
        structure_ids = _tensor(data, "structure_ids").to(torch.int64).reshape(-1)
        observed = _tensor(
            data,
            "energy_observed_errors",
            dtype=torch.float64,
        ).reshape(-1)
        orders: dict[int, PlotSeries] = {}
        for order in range(1, 9):
            binding = raw_inputs[str(order)]
            predictions_descriptor = binding["predictions"]
            predictions_path = _referenced(
                root,
                predictions_descriptor["path"],
            )
            predictions = _load_torch(
                predictions_path,
                str(predictions_descriptor["sha256"]),
            )
            raw_ids = _tensor(predictions, "structure_ids").to(torch.int64).reshape(-1)
            if not torch.equal(raw_ids, structure_ids):
                raise ValueError(f"{variant} structure IDs mismatch raw order {order}")
            logits = _tensor(
                predictions,
                "energy_logits",
                dtype=torch.float64,
            )
            expected = _tensor(
                predictions,
                "energy_expected_errors",
                dtype=torch.float64,
            ).reshape(-1)
            representatives = _tensor(
                predictions,
                "energy_representatives",
                dtype=torch.float64,
            ).reshape(-1)
            if logits.shape != (len(observed), representatives.numel()):
                raise ValueError(f"raw logits shape mismatch for order {order}")
            if expected.shape != observed.shape:
                raise ValueError(f"raw expected shape mismatch for order {order}")
            orders[order] = build_variant_series(
                PlotSeries(
                    run_dir=predictions_path.parents[2],
                    structure_ids=raw_ids,
                    target="energy",
                    order=order,
                    logits=logits,
                    observed=_tensor(
                        predictions,
                        "energy_observed_errors",
                        dtype=torch.float64,
                    ).reshape(-1),
                    expected=expected,
                    representatives=representatives,
                    stored_metrics=cast(
                        Mapping[str, int | float],
                        metrics["uq_by_order"][str(order)],
                    ),
                    force_target_mode=None,
                ),
                observed,
            )
        result[variant] = orders
    return result


def _write_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError("E0 density summary requires rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _comparison_rows(
    series: Mapping[str, Mapping[int, PlotSeries]],
    settings: DensitySettings,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order in range(1, 9):
        values = [series[name][order] for name in E0_VARIANTS]
        limits = shared_log_limits(values, margin=settings.log_margin)
        for name, value in zip(E0_VARIANTS, values, strict=True):
            panel = analyze_density_panel(value, settings, log_limits=limits)
            rows.append(
                {
                    "variant": name,
                    "data_use": (
                        "test-informed/oracle"
                        if name == "direct_mad_e0_test_informed"
                        else (
                            "validation-only calibration"
                            if name == "model_aware_reestimated_val_calibrated"
                            else "uncorrected baseline"
                        )
                    ),
                    "order": order,
                    "sample_count": panel.filtered.valid_count,
                    "pearson_log10": panel.pearson_log10,
                    "spearman_log10": panel.spearman_log10,
                    "log_limit_low": limits[0],
                    "log_limit_high": limits[1],
                }
            )
    return rows


def _render_variant_comparisons(
    series: Mapping[str, Mapping[int, PlotSeries]],
    settings: DensitySettings,
    root: Path,
) -> None:
    for order in range(1, 9):
        values = [series[name][order] for name in E0_VARIANTS]
        limits = shared_log_limits(values, margin=settings.log_margin)
        figure, axes = plt.subplots(1, 3, figsize=(21.0, 7.0))
        try:
            for name, value, axis in zip(
                E0_VARIANTS,
                values,
                axes,
                strict=True,
            ):
                _draw_density_axis(
                    axis,
                    analyze_density_panel(value, settings, log_limits=limits),
                )
                axis.set_title(f"{name}\nEnergy order {order}")
            figure.suptitle(
                f"MAD r2SCAN E0 postprocessing comparison / energy order {order}",
                fontsize=15,
            )
            figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            stem = root / f"order_{order}_three_variant_density"
            _save_figure(figure, stem.with_suffix(".png"), dpi=settings.dpi)
            _save_figure(figure, stem.with_suffix(".pdf"))
        finally:
            plt.close(figure)


def _render_correlation_summary(
    rows: list[dict[str, Any]],
    settings: DensitySettings,
    root: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), sharex=True, sharey=True)
    try:
        for name in E0_VARIANTS:
            selected = [row for row in rows if row["variant"] == name]
            orders = [int(row["order"]) for row in selected]
            axes[0].plot(
                orders,
                [float(row["pearson_log10"]) for row in selected],
                marker="o",
                label=name,
            )
            axes[1].plot(
                orders,
                [float(row["spearman_log10"]) for row in selected],
                marker="s",
                label=name,
            )
        for axis, title in zip(
            axes,
            ("Pearson r (log10)", "Spearman rho (log10)"),
            strict=True,
        ):
            axis.set_xticks(range(1, 9))
            axis.set_ylim(-1.0, 1.0)
            axis.set_xlabel("Energy cumulant order")
            axis.set_ylabel("Expected vs observed correlation")
            axis.set_title(title)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=7)
        figure.tight_layout()
        _save_figure(
            figure,
            root / "three_variant_energy_correlations.png",
            dpi=settings.dpi,
        )
        _save_figure(figure, root / "three_variant_energy_correlations.pdf")
    finally:
        plt.close(figure)


def verify_e0_density_publication(
    manifest_path: Path,
    *,
    full: bool = True,
) -> dict[str, Any]:
    """Verify an E0 density publication and every declared artifact."""

    path = Path(manifest_path).resolve()
    manifest = _mapping(path, context="E0 density manifest")
    payload = manifest.get("identity_payload")
    if (
        manifest.get("schema_version") != E0_DENSITY_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
        or manifest.get("identity") != _identity(payload)
    ):
        raise ValueError("E0 density schema/status/identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("E0 density artifacts are missing")
    for name, descriptor in artifacts.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise ValueError("E0 density artifact descriptor is invalid")
        artifact = _confined(path.parent, descriptor.get("path"))
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"E0 density artifact is missing or empty: {name}")
        expected = descriptor.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"E0 density artifact SHA is invalid: {name}")
        if full and sha256_file(artifact) != expected:
            raise ValueError(f"E0 density artifact SHA mismatch: {name}")
    campaign = payload.get("campaign")
    if not isinstance(campaign, Mapping):
        raise ValueError("E0 density campaign binding is missing")
    campaign_manifest = _referenced(path.parent, campaign.get("path"))
    if full and sha256_file(campaign_manifest) != campaign.get("sha256"):
        raise ValueError("E0 density campaign SHA mismatch")
    verify_e0_campaign(campaign_manifest, full=full)
    return manifest


def publish_e0_density_campaign(
    campaign_root: Path,
    output_root: Path,
    settings: DensitySettings,
) -> Path:
    """Atomically publish three energy-only density suites and comparisons."""

    campaign = Path(campaign_root).resolve()
    campaign_manifest = campaign / "manifest.json"
    verify_e0_campaign(campaign_manifest, full=True)
    series = load_e0_density_series(campaign)
    target = Path(output_root).resolve()
    payload = {
        "algorithm": E0_DENSITY_SCHEMA_VERSION,
        "campaign": {
            "path": Path(os.path.relpath(campaign_manifest, target)).as_posix(),
            "sha256": sha256_file(campaign_manifest),
        },
        "variants": list(E0_VARIANTS),
        "orders": list(range(1, 9)),
        "settings": asdict(settings),
    }
    identity = _identity(payload)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_e0_density_publication(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError("E0 density identity conflicts with existing output")
        return target
    if target.exists():
        raise ValueError("E0 density target exists without a complete identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".staging-{target.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        limits_by_order = {
            order: shared_log_limits(
                [series[name][order] for name in E0_VARIANTS],
                margin=settings.log_margin,
            )
            for order in range(1, 9)
        }
        for name in E0_VARIANTS:
            values = series[name]
            variant_root = staging / "variants" / name
            for order in range(1, 9):
                render_density_panel(
                    values[order],
                    settings,
                    variant_root / "runs" / f"order-{order}",
                    log_limits=limits_by_order[order],
                )
            render_energy_comparison(
                values,
                settings,
                variant_root / "comparisons",
            )
        comparisons = staging / "comparisons"
        rows = _comparison_rows(series, settings)
        _write_rows(comparisons / "three_variant_energy_correlations.csv", rows)
        atomic_write_json(
            comparisons / "three_variant_energy_correlations.json",
            {
                "variants": list(E0_VARIANTS),
                "orders": list(range(1, 9)),
                "rows": rows,
            },
        )
        _render_variant_comparisons(series, settings, comparisons)
        _render_correlation_summary(rows, settings, comparisons)
        manifest = {
            "schema_version": E0_DENSITY_SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "identity_payload": payload,
            "artifacts": _artifact_descriptors(staging),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        verify_e0_density_publication(staging / "manifest.json", full=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target
