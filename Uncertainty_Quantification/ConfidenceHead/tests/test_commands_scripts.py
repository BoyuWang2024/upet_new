from __future__ import annotations

import importlib
import importlib.util
import json
import runpy
from pathlib import Path
import subprocess
import sys

import pytest

from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import SCHEMA_VERSION
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import ConfidenceConfig
from Uncertainty_Quantification.ConfidenceHead.confidence_head.identity import cache_id
from Uncertainty_Quantification.ConfidenceHead.confidence_head.run_naming import build_run_name
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows import commands


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name.removesuffix(".py"), SCRIPTS / name
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path, *, resume_from: Path | None = None) -> ConfidenceConfig:
    return ConfidenceConfig.model_validate(
        {
            "profile": "smoke",
            "checkpoint": {
                "path": tmp_path / "model.ckpt",
                "expected_sha256": "a" * 64,
            },
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
                "energy": {
                    "hidden_dims": [4],
                    "num_bins": 3,
                    "cumulant_order": 2,
                },
            },
            "loss": {"force_coefficient": 1.0, "energy_coefficient": 1.5},
            "trainer": {"resume_from": resume_from},
            "run": {"output_root": tmp_path / "outputs", "name_prefix": "demo"},
        }
    )


def _manifest(config: ConfidenceConfig) -> dict[str, object]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"sha256": config.checkpoint.expected_sha256},
        "splits": {
            split: {
                "sha256": getattr(config.data, split).expected_sha256,
                "structure_count": 2,
                "atom_count": 3,
                "force_component_count": 9,
            }
            for split in ("train", "validation", "test")
        },
        "outputs": config.readouts.model_dump(),
        "features": {"force_dim": 2, "energy_dim": 3, "dtype": "float32"},
        "cache": {
            "batch_size": config.cache.batch_size,
            "shard_max_atoms": config.cache.shard_max_atoms,
        },
        "execution": {
            "device": config.run.device,
            "model_dtype": "float32",
            "system_dtype": "float32",
            "autocast": config.run.amp,
            "autocast_dtype": None,
        },
        "versions": {
            "python": "3.11.0",
            "torch": "2.0.0",
            "metatomic": "not-installed",
            "metatrain": "not-installed",
            "upet": "not-installed",
            "upet_git": "unknown",
        },
    }
    identity = cache_id(
        {"schema_version": SCHEMA_VERSION, "identity_payload": payload}
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "cache_id": identity,
        "identity_payload": payload,
        "splits": {},
    }


def _write_manifest(path: Path, manifest: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _refresh_identity(manifest: dict[str, object]) -> None:
    identity = cache_id(
        {
            "schema_version": SCHEMA_VERSION,
            "identity_payload": manifest["identity_payload"],
        }
    )
    manifest["identity"] = identity
    manifest["cache_id"] = identity


def test_resolve_unique_cache_manifest_rejects_missing_match(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest = _manifest(config)
    manifest["status"] = "incomplete"
    _write_manifest(
        config.run.output_root / "cache" / "candidate" / "manifest.json",
        manifest,
    )

    with pytest.raises(ValueError, match="matching cache manifest was not found"):
        commands.resolve_unique_cache_manifest(config)


def test_resolve_unique_cache_manifest_requires_exactly_one_match(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _manifest(config)
    _write_manifest(
        config.run.output_root / "cache" / "one" / "manifest.json",
        manifest,
    )
    _write_manifest(
        config.run.output_root / "cache" / "two" / "manifest.json",
        manifest,
    )

    with pytest.raises(ValueError, match="multiple matching cache manifests"):
        commands.resolve_unique_cache_manifest(config)


def test_resolve_unique_cache_manifest_returns_matching_path(tmp_path: Path) -> None:
    config = _config(tmp_path)
    path = _write_manifest(
        config.run.output_root / "cache" / "only" / "manifest.json",
        _manifest(config),
    )

    assert commands.resolve_unique_cache_manifest(config) == path.resolve()


def test_resolve_unique_cache_manifest_rejects_inconsistent_cache_id(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _manifest(config)
    manifest["cache_id"] = "cache-tampered"
    _write_manifest(
        config.run.output_root / "cache" / "candidate" / "manifest.json",
        manifest,
    )

    with pytest.raises(ValueError, match="matching cache manifest was not found"):
        commands.resolve_unique_cache_manifest(config)


def test_resolve_unique_cache_manifest_rejects_mismatched_cache_policy(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    manifest = _manifest(config)
    payload = manifest["identity_payload"]
    assert isinstance(payload, dict)
    cache = payload["cache"]
    assert isinstance(cache, dict)
    cache["batch_size"] = config.cache.batch_size + 1
    _refresh_identity(manifest)
    _write_manifest(
        config.run.output_root / "cache" / "candidate" / "manifest.json",
        manifest,
    )

    with pytest.raises(ValueError, match="matching cache manifest was not found"):
        commands.resolve_unique_cache_manifest(config)


def test_resolve_unique_cache_manifest_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    external = tmp_path / "external"
    _write_manifest(external / "manifest.json", _manifest(config))
    cache_root = config.run.output_root / "cache"
    cache_root.mkdir(parents=True)
    (cache_root / "escaped").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="matching cache manifest was not found"):
        commands.resolve_unique_cache_manifest(config)


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
    cache_manifest = _manifest(config)
    _write_manifest(
        config.run.output_root / "cache" / "only" / "manifest.json",
        cache_manifest,
    )
    run_dir = commands.resolve_run_dir(config)
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"cache_id": cache_manifest["cache_id"]}),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        commands,
        "verify_run",
        lambda value, **kwargs: seen.update(kwargs) or {"status": "ok"},
    )

    assert commands.verify_from_config(config) == {"status": "ok"}
    assert seen == {"full": True}


def test_verify_from_config_rejects_run_cache_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    _write_manifest(
        config.run.output_root / "cache" / "only" / "manifest.json",
        _manifest(config),
    )
    run_dir = commands.resolve_run_dir(config)
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"cache_id": "cache-other"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        commands,
        "verify_run",
        lambda value, **kwargs: pytest.fail("verify_run must not be called"),
    )

    with pytest.raises(ValueError, match="run cache_id does not match selected cache"):
        commands.verify_from_config(config)


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

@pytest.mark.parametrize(
    "script",
    ["build_cache.py", "train.py", "evaluate.py", "verify.py"],
)
def test_script_help_exposes_only_config(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout


def test_train_script_loads_config_and_dispatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("train.py")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("profile: smoke\n", encoding="utf-8")
    loaded_config = object()
    seen: list[object] = []
    monkeypatch.setattr(
        importlib.import_module("confidence_head.config"),
        "load_config",
        lambda path: loaded_config,
    )
    monkeypatch.setattr(
        importlib.import_module("confidence_head.workflows.commands"),
        "train_from_config",
        lambda config: seen.append(config) or Path("/run"),
    )

    assert module.main(["--config", str(config_path)]) == 0
    assert seen == [loaded_config]


@pytest.mark.parametrize(
    "script",
    ["build_cache.py", "train.py", "evaluate.py", "verify.py"],
)
def test_script_help_does_not_import_torch(script: str) -> None:
    script_path = SCRIPTS / script
    probe = "\n".join(
        (
            "import runpy",
            "import sys",
            f"sys.argv = [{str(script_path)!r}, '--help']",
            "try:",
            f"    runpy.run_path({str(script_path)!r}, run_name='__main__')",
            "except SystemExit as error:",
            "    assert error.code == 0",
            "assert 'torch' not in sys.modules",
        )
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--config" in result.stdout
