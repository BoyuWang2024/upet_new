from __future__ import annotations

import gc
import hashlib
import json
import weakref
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import sha256_file
from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
    CampaignConfig,
    CampaignDataset,
    CampaignPlotStyle,
    CampaignPrediction,
    CampaignRun,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_rendering import panel_stem
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
    PanelKey,
    PlotSource,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_store import (
    compute_plot_identity,
    publish_campaign_plots,
    validate_plot_publication,
)


RUN_LABELS = ("full_remote_b8_e8", "lr_1e-4", "lr_1e-6")
DATASET_SPECS = (
    ("matpes_test", "test", ("energy", "forces", "stress")),
    ("mad_test", "mad_test", ("energy", "forces")),
    ("matpes_train", "matpes_train", ("energy", "forces", "stress")),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _style() -> CampaignPlotStyle:
    return CampaignPlotStyle(
        grid_size=24,
        gaussian_sigma=1.0,
        contour_masses=(0.5, 0.8),
        scatter_max_points=100,
        scatter_seed=20260816,
        scatter_size=4.0,
        scatter_alpha=0.7,
        log_margin=0.05,
        figure_size=(2.0, 2.0),
        dpi=36,
        formats=("png", "pdf"),
    )


def _campaign(tmp_path: Path) -> CampaignConfig:
    return CampaignConfig(
        schema_version=1,
        runs=tuple(
            CampaignRun(
                label=label,
                config_path=tmp_path / f"{label}.yaml",
                run_root=tmp_path / "runs" / label,
                config=SimpleNamespace(),
            )
            for label in RUN_LABELS
        ),
        datasets=tuple(
            CampaignDataset(
                label=label,
                storage_key=storage_key,
                path=tmp_path / "data" / f"{label}.extxyz",
                reference_targets=reference_targets,
            )
            for label, storage_key, reference_targets in DATASET_SPECS
        ),
        prediction=CampaignPrediction(
            mode="raw", member_count=8, device="cpu", batch_size=2
        ),
        plot=_style(),
        output_root=tmp_path / "plots",
        source_path=tmp_path / "campaign.yaml",
    )


def _sources(tmp_path: Path) -> tuple[PlotSource, ...]:
    result = []
    for run_label in RUN_LABELS:
        for dataset_label, storage_key, reference_targets in DATASET_SPECS:
            root = tmp_path / "runs" / run_label / "predictions" / storage_key
            members = []
            for index in range(8):
                path = root / "members" / f"member_{index:03d}" / "raw.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(
                    f"{run_label}:{dataset_label}:raw:{index}".encode("utf-8")
                )
                members.append(path)
            targets = ("energy", "force", "stress")
            if "stress" not in reference_targets:
                targets = targets[:2]
            for target in targets:
                result.append(
                    PlotSource(
                        key=PanelKey(run_label, dataset_label, storage_key, target),
                        targets_path=root / "targets.npz",
                        uq_results_path=(
                            tmp_path
                            / "runs"
                            / run_label
                            / "uncertainty"
                            / storage_key
                            / "raw"
                            / "results.npz"
                        ),
                        member_paths=tuple(members),
                        prediction_manifest_sha256=_digest(
                            f"prediction:{run_label}:{dataset_label}"
                        ),
                        uq_manifest_sha256=_digest(f"uq:{run_label}:{dataset_label}"),
                        targets_sha256=_digest(f"targets:{run_label}:{dataset_label}"),
                    )
                )
    return tuple(result)


@pytest.fixture
def campaign_sources(tmp_path: Path) -> tuple[CampaignConfig, tuple[PlotSource, ...]]:
    return _campaign(tmp_path), _sources(tmp_path)


def _expected_names(sources: tuple[PlotSource, ...]) -> set[str]:
    images = {
        f"{panel_stem(source.key)}.{suffix}"
        for source in sources
        for suffix in ("png", "pdf")
    }
    return images | {
        "raw_std_statistics.csv",
        "raw_std_statistics.json",
        "plot_manifest.json",
    }


def _install_fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    campaign: CampaignConfig,
    sources: tuple[PlotSource, ...],
    *,
    events: list[str] | None = None,
    fail_at: int | None = None,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_store

    limits = {"energy": (-4.0, 2.0), "force": (-5.0, 1.0), "stress": (-3.0, 3.0)}

    monkeypatch.setattr(
        plot_store,
        "discover_plot_sources",
        lambda actual: sources if actual is campaign else (),
    )

    def scan(actual, *, margin: float):
        assert actual == sources
        assert margin == campaign.plot.log_margin
        return limits

    def analyze(source, log_limits, style):
        assert log_limits == limits[source.key.target]
        assert style is campaign.plot
        return SimpleNamespace(key=source.key)

    calls = {"count": 0}

    def render(analysis, style, destination: Path):
        assert style is campaign.plot
        calls["count"] += 1
        if fail_at is not None and calls["count"] == fail_at:
            raise RuntimeError("synthetic render failure")
        if events is not None:
            events.append(f"panel:{analysis.key.target}")
        stem = destination / panel_stem(analysis.key)
        png = stem.with_suffix(".png")
        pdf = stem.with_suffix(".pdf")
        png.write_bytes(f"PNG:{panel_stem(analysis.key)}".encode())
        pdf.write_bytes(f"PDF:{panel_stem(analysis.key)}".encode())
        return png, pdf

    def statistics(analyses, actual_sources, style, destination: Path):
        assert tuple(item.key for item in analyses) == tuple(
            source.key for source in sources
        )
        assert actual_sources == sources
        assert style is campaign.plot
        if events is not None:
            events.append("statistics")
        csv_path = destination / "raw_std_statistics.csv"
        json_path = destination / "raw_std_statistics.json"
        csv_path.write_text(
            "target\n" + "\n".join(source.key.target for source in sources),
            encoding="utf-8",
        )
        json_path.write_text(
            json.dumps({"rows": [source.key.target for source in sources]}) + "\n",
            encoding="utf-8",
        )
        return csv_path, json_path

    monkeypatch.setattr(plot_store, "scan_shared_log_limits", scan)
    monkeypatch.setattr(plot_store, "analyze_panel_source", analyze)
    monkeypatch.setattr(plot_store, "render_panel", render)
    monkeypatch.setattr(plot_store, "render_statistics", statistics)
    monkeypatch.setattr(
        plot_store,
        "summarize_panel",
        lambda analysis: SimpleNamespace(key=analysis.key),
        raising=False,
    )


def test_identity_binds_ordered_campaign_sources_members_analysis_and_style(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
) -> None:
    campaign, sources = campaign_sources
    identity = compute_plot_identity(campaign, sources)
    document = identity.document

    assert len(identity.digest) == 64
    int(identity.digest, 16)
    assert document["schema"] == "upet.bootstrap.plot-identity/v1"
    assert document["runs"] == list(RUN_LABELS)
    assert document["datasets"] == [
        {
            "label": label,
            "storage_key": storage_key,
            "reference_targets": list(reference_targets),
        }
        for label, storage_key, reference_targets in DATASET_SPECS
    ]
    assert document["mode"] == "raw"
    assert document["member_count"] == 8
    assert document["style"] == json.loads(json.dumps(asdict(campaign.plot)))
    assert document["analysis"] == {
        "correlations": ["spearman_log", "pearson_log10"],
        "ddof": 1,
        "energy": "per_atom",
        "filter": "positive_finite_uncertainty_and_residual",
        "force": "component",
        "stress": "symmetric_voigt_xx_yy_zz_yz_xz_xy",
    }
    assert len(document["sources"]) == 24
    for record, source in zip(document["sources"], sources, strict=True):
        assert record["key"] == {
            "run_label": source.key.run_label,
            "dataset_label": source.key.dataset_label,
            "storage_key": source.key.storage_key,
            "target": source.key.target,
        }
        assert record["prediction_manifest_sha256"] == (
            source.prediction_manifest_sha256
        )
        assert record["uq_manifest_sha256"] == source.uq_manifest_sha256
        assert record["targets_sha256"] == source.targets_sha256
        assert [item["index"] for item in record["ordered_members"]] == list(range(8))
        assert record["ordered_members"] == [
            {
                "index": index,
                "raw_path": f"member_{index:03d}/raw.npz",
            }
            for index in range(8)
        ]
    assert str(campaign.source_path.parent) not in json.dumps(document)


def test_identity_changes_for_all_bound_inputs_and_is_repeatable(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
) -> None:
    campaign, sources = campaign_sources
    baseline = compute_plot_identity(campaign, sources)
    assert compute_plot_identity(campaign, sources) == baseline
    first = sources[0]
    variants = (
        (campaign, (replace(first, prediction_manifest_sha256="a" * 64), *sources[1:])),
        (campaign, (replace(first, uq_manifest_sha256="b" * 64), *sources[1:])),
        (campaign, (replace(first, targets_sha256="c" * 64), *sources[1:])),
        (
            campaign,
            (
                replace(first, member_paths=tuple(reversed(first.member_paths))),
                *sources[1:],
            ),
        ),
        (
            replace(
                campaign,
                plot=replace(
                    campaign.plot, scatter_seed=campaign.plot.scatter_seed + 1
                ),
            ),
            sources,
        ),
    )
    for actual_campaign, actual_sources in variants:
        assert (
            compute_plot_identity(actual_campaign, actual_sources).digest
            != baseline.digest
        )


def test_identity_does_not_reread_member_payloads(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_store

    campaign, sources = campaign_sources
    monkeypatch.setattr(
        plot_store,
        "sha256_file",
        lambda _path: (_ for _ in ()).throw(
            AssertionError("identity must reuse prediction manifest provenance")
        ),
    )
    compute_plot_identity(campaign, sources)


def test_identity_rejects_non_raw_or_member_count_mismatch(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
) -> None:
    campaign, sources = campaign_sources
    ema = replace(campaign, prediction=replace(campaign.prediction, mode="ema"))
    wrong_count = replace(
        campaign, prediction=replace(campaign.prediction, member_count=7)
    )
    with pytest.raises(HardFailure, match="raw|mode"):
        compute_plot_identity(ema, sources)
    with pytest.raises(HardFailure, match="member|count"):
        compute_plot_identity(wrong_count, sources)


def test_formal_publication_has_exactly_51_files_and_manifest_is_last(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_store

    campaign, sources = campaign_sources
    events: list[str] = []
    _install_fake_pipeline(monkeypatch, campaign, sources, events=events)
    original_write_json = plot_store.atomic_write_json

    def record_manifest(path: Path, document):
        assert Path(path).name == "plot_manifest.json"
        events.append("manifest")
        return original_write_json(path, document)

    monkeypatch.setattr(plot_store, "atomic_write_json", record_manifest)
    publication = publish_campaign_plots(campaign)
    identity = compute_plot_identity(campaign, sources)
    names = {path.name for path in publication.plot_dir.iterdir()}

    assert publication.skipped is False
    assert publication.plot_dir.name == (
        f"raw_std_three_runs_three_datasets__{identity.digest[:16]}"
    )
    assert names == _expected_names(sources)
    assert len(names) == 51
    assert sum(name.endswith(".png") for name in names) == 24
    assert sum(name.endswith(".pdf") for name in names) == 24
    assert not any("member_sweep" in name for name in names)
    assert len([event for event in events if event.startswith("panel:")]) == 24
    assert events[-2:] == ["statistics", "manifest"]

    manifest = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "upet.bootstrap.plot-publication/v1"
    assert manifest["status"] == "PASS"
    assert manifest["identity"] == identity.digest
    assert manifest["identity_document"] == identity.document
    assert len(manifest["artifacts"]) == 50
    assert "plot_manifest.json" not in manifest["artifacts"]
    for name, record in manifest["artifacts"].items():
        path = publication.plot_dir / name
        assert record == {
            "path": name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    validate_plot_publication(publication.plot_dir, expected_identity=identity.digest)


def test_generic_smoke_uses_generic_content_addressed_name(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, sources = campaign_sources
    smoke = replace(
        campaign,
        runs=campaign.runs[:1],
        datasets=campaign.datasets[:1],
        output_root=campaign.output_root / "smoke",
    )
    smoke_sources = sources[:3]
    _install_fake_pipeline(monkeypatch, smoke, smoke_sources)
    publication = publish_campaign_plots(smoke)
    identity = compute_plot_identity(smoke, smoke_sources)

    assert publication.plot_dir.name == f"raw_std_campaign__{identity.digest[:16]}"
    assert {path.name for path in publication.plot_dir.iterdir()} == _expected_names(
        smoke_sources
    )
    assert len(tuple(publication.plot_dir.iterdir())) == 9


def test_complete_identity_is_zero_write_reuse(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_store

    campaign, sources = campaign_sources
    _install_fake_pipeline(monkeypatch, campaign, sources)
    first = publish_campaign_plots(campaign)
    before = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in first.plot_dir.iterdir()
    }

    def must_not_render(*_args, **_kwargs):
        raise AssertionError("complete identity must be reused before rendering")

    monkeypatch.setattr(plot_store, "render_panel", must_not_render)
    second = publish_campaign_plots(campaign)
    after = {
        path.name: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in second.plot_dir.iterdir()
    }
    assert second.skipped is True
    assert second.plot_dir == first.plot_dir
    assert after == before


def test_publication_streams_full_panel_analyses_before_statistics(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import plot_store

    campaign, sources = campaign_sources
    _install_fake_pipeline(monkeypatch, campaign, sources)
    references: list[weakref.ReferenceType[object]] = []
    peak_alive = 0

    class TrackedAnalysis:
        def __init__(self, key: PanelKey) -> None:
            self.key = key

    def analyze(source, log_limits, style):
        nonlocal peak_alive
        assert style is campaign.plot
        assert (
            log_limits
            in {
                "energy": (-4.0, 2.0),
                "force": (-5.0, 1.0),
                "stress": (-3.0, 3.0),
            }.values()
        )
        gc.collect()
        peak_alive = max(
            peak_alive,
            sum(reference() is not None for reference in references),
        )
        result = TrackedAnalysis(source.key)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(plot_store, "analyze_panel_source", analyze)
    monkeypatch.setattr(
        plot_store,
        "summarize_panel",
        lambda analysis: SimpleNamespace(key=analysis.key),
        raising=False,
    )

    publish_campaign_plots(campaign)

    assert peak_alive <= 1


def test_failure_cleans_staging_and_partial_target_is_not_clobbered(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, sources = campaign_sources
    _install_fake_pipeline(monkeypatch, campaign, sources, fail_at=4)
    identity = compute_plot_identity(campaign, sources)
    destination = campaign.output_root / (
        f"raw_std_three_runs_three_datasets__{identity.digest[:16]}"
    )
    with pytest.raises(RuntimeError, match="synthetic render failure"):
        publish_campaign_plots(campaign)
    assert not destination.exists()
    assert not list(campaign.output_root.glob(f".{destination.name}.*.staging"))

    destination.mkdir(parents=True)
    sentinel = destination / "unfinished.txt"
    sentinel.write_bytes(b"preserve me")
    with pytest.raises(HardFailure, match="publication|manifest|artifact|exist"):
        publish_campaign_plots(campaign)
    assert sentinel.read_bytes() == b"preserve me"
    assert {path.name for path in destination.iterdir()} == {"unfinished.txt"}


@pytest.mark.parametrize("field", ["path", "size", "sha256"])
def test_validator_rejects_tampered_manifest_records(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    campaign, sources = campaign_sources
    _install_fake_pipeline(monkeypatch, campaign, sources)
    publication = publish_campaign_plots(campaign)
    document = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    name = next(iter(document["artifacts"]))
    document["artifacts"][name][field] = (
        "different.png" if field == "path" else (1 if field == "size" else "0" * 64)
    )
    publication.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HardFailure, match="path|size|hash|SHA|artifact"):
        validate_plot_publication(publication.plot_dir)


def test_validator_rejects_extra_tampered_missing_and_symlinked_artifacts(
    tmp_path: Path,
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, sources = campaign_sources
    _install_fake_pipeline(monkeypatch, campaign, sources)
    publication = publish_campaign_plots(campaign)
    victim = next(
        path for path in publication.plot_dir.iterdir() if path.suffix == ".png"
    )
    original = victim.read_bytes()

    victim.write_bytes(b"tampered")
    with pytest.raises(HardFailure, match="size|hash|artifact"):
        validate_plot_publication(publication.plot_dir)
    victim.write_bytes(original)

    extra = publication.plot_dir / "extra.txt"
    extra.write_text("unexpected", encoding="utf-8")
    with pytest.raises(HardFailure, match="extra|inventory|artifact"):
        validate_plot_publication(publication.plot_dir)
    extra.unlink()

    victim.unlink()
    with pytest.raises(HardFailure, match="missing|inventory|artifact"):
        validate_plot_publication(publication.plot_dir)
    outside = tmp_path / "outside.png"
    outside.write_bytes(original)
    victim.symlink_to(outside)
    with pytest.raises(HardFailure, match="symlink|unsafe|artifact"):
        validate_plot_publication(publication.plot_dir)


def test_identity_conflict_or_manifest_tampering_is_hard_failure(
    campaign_sources: tuple[CampaignConfig, tuple[PlotSource, ...]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign, sources = campaign_sources
    _install_fake_pipeline(monkeypatch, campaign, sources)
    publication = publish_campaign_plots(campaign)
    document = json.loads(publication.manifest_path.read_text(encoding="utf-8"))
    document["identity"] = "0" * 64
    publication.manifest_path.write_text(json.dumps(document), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in publication.plot_dir.iterdir()}

    with pytest.raises(HardFailure, match="identity|manifest|publication"):
        publish_campaign_plots(campaign)
    assert {
        path.name: path.read_bytes() for path in publication.plot_dir.iterdir()
    } == before
