from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import SCHEMA_VERSION
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import ConfidenceConfig
from Uncertainty_Quantification.ConfidenceHead.confidence_head.run_naming import build_run_name
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows import commands


def _config(tmp_path: Path, *, resume_from: Path | None = None) -> ConfidenceConfig:
    return ConfidenceConfig.model_validate(
        {
            "profile": "smoke",
            "checkpoint": {"path": tmp_path / "model.ckpt", "expected_sha256": "a" * 64},
            "data": {
                split: {
                    "path": tmp_path / f"{split}.xyz",
                    "expected_sha256": f"{index:064x}",
                }
                for index, split in enumerate(("train", "validation", "test"), 1)
            },
            "binning": {
                "force_num_bins": 3,
                "force_max_error": 0.5,
                "energy_num_bins": 3,
                "energy_max_error": 0.3,
            },
            "model": {
                "force": {"hidden_dims": [4], "num_bins": 3},
                "energy": {"hidden_dims": [4], "num_bins": 3, "cumulant_order": 2},
            },
            "loss": {"force_coefficient": 1.0, "energy_coefficient": 1.5},
            "trainer": {"resume_from": resume_from},
            "run": {"output_root": tmp_path / "outputs", "name_prefix": "demo"},
        }
    )


def _manifest(config: ConfidenceConfig) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity_payload": {
            "checkpoint": {"sha256": config.checkpoint.expected_sha256},
            "splits": {
                split: {"sha256": getattr(config.data, split).expected_sha256}
                for split in ("train", "validation", "test")
            },
            "outputs": config.readouts.model_dump(),
            "execution": {
                "device": config.run.device,
                "model_dtype": "float32",
                "system_dtype": "float32",
                "autocast": config.run.amp,
                "autocast_dtype": None,
            },
        },
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_resolve_unique_cache_manifest_rejects_missing_match(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = _manifest(config)
    manifest["status"] = "incomplete"
    _write_manifest(config.run.output_root / "cache" / "candidate" / "manifest.json", manifest)

    with pytest.raises(ValueError, match="matching cache manifest was not found"):
        commands.resolve_unique_cache_manifest(config)


def test_resolve_unique_cache_manifest_requires_exactly_one_match(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _manifest(config)
    _write_manifest(config.run.output_root / "cache" / "one" / "manifest.json", manifest)
    _write_manifest(config.run.output_root / "cache" / "two" / "manifest.json", manifest)

    with pytest.raises(ValueError, match="multiple matching cache manifests"):
        commands.resolve_unique_cache_manifest(config)


def test_resolve_unique_cache_manifest_returns_matching_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = _write_manifest(
        config.run.output_root / "cache" / "only" / "manifest.json",
        _manifest(config),
    )
    assert commands.resolve_unique_cache_manifest(config) == path.resolve()


def test_build_cache_from_config_returns_low_level_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    expected = Path("/cache/manifest.json")
    monkeypatch.setattr(commands, "build_cache", lambda value: expected)
    assert commands.build_cache_from_config(config) == expected


def test_train_from_config_passes_derived_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        commands,
        "resolve_unique_cache_manifest",
        lambda value: Path("/cache/manifest.json"),
    )
    monkeypatch.setattr(
        commands,
        "train_run",
        lambda value, **kwargs: seen.update(kwargs) or Path("/run"),
    )

    assert commands.train_from_config(config) == Path("/run")
    assert seen == {
        "cache_manifest_path": Path("/cache/manifest.json"),
        "run_name": build_run_name(config),
        "resume_from": config.trainer.resume_from,
    }


def test_evaluate_from_config_defaults_to_best_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        commands,
        "resolve_unique_cache_manifest",
        lambda value: Path("/cache/manifest.json"),
    )
    monkeypatch.setattr(commands, "resolve_run_dir", lambda value: Path("/run"))
    monkeypatch.setattr(
        commands,
        "evaluate_run",
        lambda value, **kwargs: seen.update(kwargs) or Path("/run/evaluation"),
    )

    assert commands.evaluate_from_config(config) == Path("/run/evaluation")
    assert seen == {
        "cache_manifest_path": Path("/cache/manifest.json"),
        "checkpoint_path": None,
    }


def test_verify_from_config_requests_full_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        commands,
        "resolve_unique_cache_manifest",
        lambda value: Path("/cache/manifest.json"),
    )
    monkeypatch.setattr(commands, "resolve_run_dir", lambda value: Path("/run"))
    monkeypatch.setattr(
        commands,
        "verify_run",
        lambda value, **kwargs: seen.update(kwargs) or {"status": "ok"},
    )

    assert commands.verify_from_config(config) == {"status": "ok"}
    assert seen == {"full": True}


def test_train_from_config_rejects_resume_outside_derived_run_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path, resume_from=tmp_path / "foreign" / "last.pt")
    monkeypatch.setattr(
        commands,
        "resolve_unique_cache_manifest",
        lambda value: Path("/cache/manifest.json"),
    )

    with pytest.raises(
        ValueError,
        match="resume checkpoint must be within derived run directory",
    ):
        commands.train_from_config(config)
