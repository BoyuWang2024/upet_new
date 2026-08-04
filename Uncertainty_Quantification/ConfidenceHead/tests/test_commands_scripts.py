from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import (
    SCHEMA_VERSION,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    ConfidenceConfig,
    load_config,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.identity import cache_id
from Uncertainty_Quantification.ConfidenceHead.confidence_head.run_naming import (
    build_run_name,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.workflows import commands


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.mark.parametrize(
    "name",
    ["n20_local_cpu.yaml", "n20_cpu.yaml", "full_gpu.yaml"],
)
def test_shipped_configs_explicitly_select_atom_mean(name: str) -> None:
    path = CONFIGS / name
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert raw["model"]["force"]["target_mode"] == "atom_mean"
    config = load_config(path)
    assert config.model.force.target_mode == "atom_mean"
    assert config.binning.force_max_error == 0.5
    assert config.binning.energy_max_error == 0.3
    assert config.loss.force_coefficient == 1.0
    assert config.loss.energy_coefficient == 1.5
    assert config.scheduler.monitor == "val/total_loss_ema"
    assert config.trainer.monitor == "val/total_loss_ema"


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
                "force_max_error": 0.5,
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
        "cache": {"batch_size": config.cache.batch_size},
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
    identity = cache_id({"schema_version": SCHEMA_VERSION, "identity_payload": payload})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "cache_id": identity,
        "identity_payload": payload,
        "splits": {},
    }


def _run_manifest(
    config: ConfidenceConfig,
    cache_manifest: dict[str, object],
) -> dict[str, object]:
    bins = commands._bin_payload(
        commands.fixed_linear_binning(
            config.model.force.num_bins,
            config.binning.force_max_error,
        ),
        commands.fixed_linear_binning(
            config.model.energy.num_bins,
            config.binning.energy_max_error,
        ),
        config.model.force.target_mode,
    )
    identity, run_identity, _ = commands._identities(
        config,
        cache_manifest,
        bins,
    )
    return {
        "schema_version": commands.RUN_SCHEMA_VERSION,
        "status": "complete",
        "identity": run_identity,
        "run_id": run_identity,
        "config_id": identity.config_id,
        "cache_id": identity.cache_id,
        "binning_id": identity.binning_id,
        "model_loss_id": identity.model_loss_id,
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
    cache_manifest = _manifest(config)
    cache_path = _write_manifest(
        config.run.output_root / "cache" / "only" / "manifest.json",
        cache_manifest,
    )
    run_dir = commands.resolve_run_dir(config)
    run_dir.mkdir(parents=True)
    _write_manifest(run_dir / "manifest.json", _run_manifest(config, cache_manifest))
    seen: dict[str, object] = {}
    monkeypatch.setattr(
        commands,
        "evaluate_run",
        lambda value, **kwargs: seen.update(kwargs) or run_dir / "evaluation",
    )

    assert commands.evaluate_from_config(config) == run_dir / "evaluation"
    assert seen == {
        "cache_manifest_path": cache_path.resolve(),
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
    _write_manifest(
        run_dir / "manifest.json",
        _run_manifest(config, cache_manifest),
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
    cache_manifest = _manifest(config)
    _write_manifest(
        config.run.output_root / "cache" / "only" / "manifest.json",
        cache_manifest,
    )
    run_dir = commands.resolve_run_dir(config)
    run_dir.mkdir(parents=True)
    run_manifest = _run_manifest(config, cache_manifest)
    run_manifest["cache_id"] = "cache-other"
    _write_manifest(run_dir / "manifest.json", run_manifest)
    monkeypatch.setattr(
        commands,
        "verify_run",
        lambda value, **kwargs: pytest.fail("verify_run must not be called"),
    )

    with pytest.raises(ValueError, match="configured run cache_id identity mismatch"):
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
    ("script", "command_name"),
    [
        ("build_cache.py", "build_cache_from_config"),
        ("train.py", "train_from_config"),
        ("evaluate.py", "evaluate_from_config"),
        ("verify.py", "verify_from_config"),
    ],
)
def test_script_resolves_relative_config_from_current_directory(
    script: str,
    command_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(script)
    config_path = tmp_path / "config.yaml"
    config_path.write_text("profile: smoke\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    loaded_config = object()
    seen_paths: list[Path] = []
    config_module = importlib.import_module("confidence_head.config")
    command_module = importlib.import_module("confidence_head.workflows.commands")
    monkeypatch.setattr(
        config_module,
        "load_config",
        lambda path: seen_paths.append(path) or loaded_config,
    )
    monkeypatch.setattr(
        command_module,
        command_name,
        lambda config: Path("/artifact"),
    )

    assert module.main(["--config", "config.yaml"]) == 0
    assert seen_paths == [config_path.resolve()]


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
