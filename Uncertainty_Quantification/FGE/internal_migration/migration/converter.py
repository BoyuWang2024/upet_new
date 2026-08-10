"""Atomic, compute-free conversion of one authenticated legacy FGE run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ...fge.artifacts import (
    atomic_torch_save,
    atomic_write_json,
    atomic_write_yaml,
    sha256_file,
    sibling_staging,
)
from ...fge.config import FGEConfig
from ...fge.errors import HardFailure
from ...fge.manifests import build_prediction_manifest, build_training_manifest
from ...fge.members import READOUT_PREFIXES, SCHEMA_VERSION
from ...fge.validation import schema_signature, validate_result
from .legacy_reader import (
    LegacyExpectations,
    LegacyRun,
    SourceSnapshot,
    read_legacy_run,
    verify_source_unchanged,
)
from .normalization import evaluation_values


_DEFAULT_MEMBER_SHA256 = {
    "member_001": "c9376c9d06aaef652602cfd30c20aaeb2c20b547f2cdc2aded7b7554231e44cb",
    "member_002": "96239b6766252168da329435fa19f78ec0801903aae5f192cf42561f845b08a7",
    "member_003": "6afaaec932d44eea9e3d7f5460b2d4fb16142a7acca5668a324ff5c98d4c8ef3",
    "member_004": "c6e688442a7afef456de36282d3a6d998d15ac9b328242d9c27a8d8d64b865e0",
    "member_005": "415306b16c60aa400ce75d2e18fabec9f61cd19d7ced4ec2f380055fe00b1546",
    "member_006": "68c4a04418022b08d7120eae03862fdb47c42beabc3ac63adf1168065a290cbd",
    "member_007": "1a99ac819c408d65e8e0d3e411f023b5642a285d7fbbe661b09894c9f708980a",
    "member_008": "5b91297faa2a1f0ec95d48f728e33687efd4fe3e6f6773c5de2f985ac6671521",
}
_UNAVAILABLE = {"status": "unavailable"}
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _migration_code_identity(value: Mapping[str, object]) -> dict[str, str]:
    if set(value) != {"commit", "dirty_sha256"}:
        raise HardFailure("migration code identity has an invalid schema")
    commit, dirty = value.get("commit"), value.get("dirty_sha256")
    if (
        not isinstance(commit, str)
        or _COMMIT.fullmatch(commit) is None
        or not isinstance(dirty, str)
        or _SHA256.fullmatch(dirty) is None
    ):
        raise HardFailure("migration code identity has invalid digests")
    return {"commit": commit, "dirty_sha256": dirty}


def _repo_code_identity(repo_root: Path | None = None) -> dict[str, str]:
    """Return the exact commit and a content digest of every dirty repository byte."""

    cwd = Path(repo_root or __file__).resolve()
    if cwd.is_file():
        cwd = cwd.parent
    try:
        root = Path(
            subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "."],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
    except (OSError, subprocess.CalledProcessError) as exc:
        raise HardFailure("repository code identity is unavailable") from exc
    digest = hashlib.sha256(tracked)
    for encoded in sorted(item for item in untracked if item):
        candidate = root / os.fsdecode(encoded)
        if not candidate.is_file() or candidate.is_symlink():
            raise HardFailure("repository identity contains an unsafe untracked path")
        digest.update(encoded)
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
    return {"commit": commit, "dirty_sha256": digest.hexdigest()}


def _safe_absolute(path: Path, label: str) -> Path:
    candidate = Path(path).absolute()
    for item in (candidate, *candidate.parents):
        if item.is_symlink():
            raise HardFailure(f"{label} must not contain a symbolic link")
    return candidate.resolve()


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _default_expectations() -> LegacyExpectations:
    return LegacyExpectations(
        member_count=8,
        chunk_count=969,
        member_sha256=_DEFAULT_MEMBER_SHA256,
        test_data_sha256=(
            "1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d"
        ),
        manifest_path="matpes_test_full/fge_manifest.json",
        members_directory=("n20_official_readout_schema_v4_path_validation/members"),
        prediction_directory="matpes_test_full/predictions/test",
        uncertainty_path="matpes_test_full/uncertainty/test/uncertainty.pt",
        metrics_path="matpes_test_full/evaluation/test_final_metrics.json",
        risk_coverage_paths={
            "energy_total_std": (
                "matpes_test_full/evaluation/risk_coverage_test_energy_total_std.csv"
            ),
            "energy_per_atom_std": (
                "matpes_test_full/evaluation/risk_coverage_test_energy_atom_std.csv"
            ),
            "force_atom_vector_std": (
                "matpes_test_full/evaluation/"
                "risk_coverage_test_force_atom_vector_std.csv"
            ),
            "force_structure_q95_std": (
                "matpes_test_full/evaluation/"
                "risk_coverage_test_force_structure_q95_std.csv"
            ),
        },
    )


def _config_identity(config_resolved: Mapping[str, object]) -> dict[str, str]:
    encoded = json.dumps(
        dict(config_resolved), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest()}


def _state(path: Path) -> dict[str, torch.Tensor]:
    try:
        import metatomic.torch  # noqa: F401

        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as exc:
        raise HardFailure(
            f"unable to load authenticated checkpoint: {path.name}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise HardFailure("legacy checkpoint must be a mapping")
    state = raw.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise HardFailure("legacy checkpoint has no model_state_dict")
    result: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if name == "finetune_config" and isinstance(value, Mapping):
            continue
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise HardFailure("legacy checkpoint state contains unsupported data")
        result[name] = value.detach().cpu()
    return result


def _readout_names(state: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
    names = tuple(name for name in state if name.startswith(READOUT_PREFIXES))
    if len(names) != 12 or sum(state[name].numel() for name in names) != 13_338:
        raise HardFailure("legacy checkpoint violates the 12-tensor readout contract")
    if any(
        not state[name].is_floating_point()
        or not bool(torch.isfinite(state[name]).all())
        for name in names
    ):
        raise HardFailure("legacy readout tensors must be finite floating tensors")
    return names


def _same_tensor(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(left, right)
    )


def _frozen_identity(
    state: Mapping[str, torch.Tensor], readout_names: Sequence[str]
) -> dict[str, str]:
    digest = hashlib.sha256()
    readout = set(readout_names)
    for name in sorted(set(state) - readout):
        tensor = state[name].contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest()}


def _extract_members(
    staging: Path,
    run: LegacyRun,
    base_state: Mapping[str, torch.Tensor],
    base_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, str]]:
    readout_names = _readout_names(base_state)
    members_dir = staging / "training" / "members"
    members_dir.mkdir(parents=True, exist_ok=True)
    manifest_members: list[dict[str, object]] = []
    audit_mapping: list[dict[str, object]] = []
    for member in run.members:
        member_state = _state(member.checkpoint_path)
        if set(member_state) != set(base_state):
            raise HardFailure("legacy member state keys differ from base")
        for name in set(base_state) - set(readout_names):
            if not _same_tensor(member_state[name], base_state[name]):
                raise HardFailure(f"legacy non-readout state drift: {member.member_id}")
        for name in readout_names:
            if (
                member_state[name].dtype != base_state[name].dtype
                or member_state[name].shape != base_state[name].shape
                or not bool(torch.isfinite(member_state[name]).all())
            ):
                raise HardFailure(
                    "legacy member readout metadata or values are invalid"
                )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "member_id": int(member.member_id.removeprefix("member_")),
            "cycle": member.cycle,
            "global_step": member.global_step,
            "base_sha256": base_sha256,
            "tensors": [
                {
                    "name": name,
                    "dtype": str(member_state[name].dtype),
                    "shape": list(member_state[name].shape),
                    "value": member_state[name].clone(),
                }
                for name in readout_names
            ],
        }
        target = members_dir / f"{member.member_id}.pt"
        atomic_torch_save(target, payload)
        target_sha = sha256_file(target)
        manifest_members.append(
            {
                "member_id": member.member_id,
                "sha256": target_sha,
                "cycle": member.cycle,
                "endpoint_global_step": member.global_step,
            }
        )
        audit_mapping.append(
            {
                "member_id": member.member_id,
                "source_checkpoint": str(member.checkpoint_path),
                "source_sha256": member.sha256,
                "a3_path": f"training/members/{member.member_id}.pt",
                "a3_sha256": target_sha,
            }
        )
    return manifest_members, audit_mapping, _frozen_identity(base_state, readout_names)


def _dependencies() -> dict[str, str]:
    result = {"torch": torch.__version__}
    try:
        result["metatrain"] = importlib.metadata.version("metatrain")
    except importlib.metadata.PackageNotFoundError as exc:
        raise HardFailure("metatrain dependency identity is unavailable") from exc
    return result


def _preflight_documents(
    staging: Path,
    config: FGEConfig,
    training: Mapping[str, object],
    config_identity: Mapping[str, str],
) -> None:
    data = training["data_identities"]
    checkpoint = training["checkpoint_identity"]
    if not isinstance(data, Mapping) or not isinstance(checkpoint, Mapping):
        raise HardFailure("training identities are invalid")
    identity = {
        "inputs": {
            "base_checkpoint": checkpoint["sha256"],
            "train_data": data["train"]["sha256"],
            "val_data": data["val"]["sha256"],
            "test_data": data["test"]["sha256"],
        },
        "targets": {
            "energy": config.data.energy_target,
            "forces": config.data.forces_target,
            "stress": config.data.stress_target,
        },
        "units": {
            "energy": config.data.energy_unit,
            "forces": config.data.forces_unit,
            "stress": config.data.stress_unit,
        },
        "model_contract": training["model_contract"],
        "member_count": training["member_count"],
        "runtime": {
            "device": config.training.device,
            "dtype": config.training.dtype,
        },
    }
    for stage in ("train", "predict", "evaluate"):
        flags = (
            config.scientific.training
            if stage == "train"
            else config.scientific.evaluation
        )
        atomic_write_json(
            staging / "preflight" / f"{stage}.json",
            {
                "stage": stage,
                "status": "PASS",
                "basis": "canonical_artifacts",
                "identity": identity,
                "scientific_flags": {
                    "path_feasibility_only": flags.path_feasibility_only,
                    "split_leakage": flags.split_leakage,
                    "scientific_evaluation": flags.scientific_evaluation,
                    "inference_only": flags.inference_only,
                },
                "config_identity": dict(config_identity),
            },
        )


def _write_formal_tree(
    staging: Path,
    run: LegacyRun,
    config: FGEConfig,
    base_checkpoint: Path,
    code_identity: Mapping[str, object],
) -> list[dict[str, object]]:
    base_sha = sha256_file(base_checkpoint)
    if base_sha != config.identity.base_checkpoint_sha256:
        raise HardFailure("base checkpoint SHA256 differs from configuration")
    base_state = _state(base_checkpoint)
    config_resolved = config.sanitized()
    config_identity = _config_identity(config_resolved)
    atomic_write_yaml(staging / "config_resolved.yaml", config_resolved)
    manifest_members, audit_mapping, frozen_identity = _extract_members(
        staging, run, base_state, base_sha
    )
    training = build_training_manifest(
        project_name=config.project.name,
        config_resolved=config_resolved,
        config_identity=config_identity,
        checkpoint_identity={"sha256": base_sha},
        data_identities={
            "train": {"sha256": config.identity.train_data_sha256},
            "val": {"sha256": config.identity.val_data_sha256},
            "test": {"sha256": config.identity.test_data_sha256},
        },
        model_contract={
            "readout_tensor_count": 12,
            "readout_parameter_count": 13_338,
        },
        frozen_fingerprint_identity=frozen_identity,
        dependency_snapshot=_dependencies(),
        scientific_flags={
            "path_feasibility_only": config.scientific.training.path_feasibility_only,
            "split_leakage": config.scientific.training.split_leakage,
            "scientific_evaluation": config.scientific.training.scientific_evaluation,
            "inference_only": config.scientific.training.inference_only,
        },
        artifact_writer_code_identity=code_identity,
        validator_code_identity=code_identity,
        training_code_identity=_UNAVAILABLE,
        member_count=len(run.members),
        members=manifest_members,
    )
    atomic_write_json(staging / "training" / "manifest.json", training)
    _preflight_documents(staging, config, training, config_identity)
    prediction_path = staging / "prediction" / "test_raw.pt"
    atomic_torch_save(prediction_path, run.prediction)
    prediction_shape = run.prediction["statistics"]
    if not isinstance(prediction_shape, Mapping):
        raise HardFailure("canonical prediction statistics are invalid")
    target_names = run.prediction["target_names"]
    units = run.prediction["units"]
    if not isinstance(target_names, Mapping) or not isinstance(units, Mapping):
        raise HardFailure("canonical prediction metadata is invalid")
    prediction_manifest = build_prediction_manifest(
        root=staging,
        prediction_path=prediction_path,
        member_ids=tuple(member.member_id for member in run.members),
        shape=prediction_shape,
        config_identity=config_identity,
        test_data_identity={"sha256": config.identity.test_data_sha256},
        target_names=target_names,
        units=units,
        artifact_writer_code_identity=code_identity,
        validator_code_identity=code_identity,
    )
    atomic_write_json(staging / "prediction" / "manifest.json", prediction_manifest)
    ensemble, uncertainty, metrics, report_inputs = evaluation_values(run, config)
    evaluation = staging / "evaluation" / "legacy_equal_weight"
    atomic_torch_save(evaluation / "ensemble.pt", ensemble)
    atomic_torch_save(evaluation / "uncertainty.pt", dict(uncertainty))
    atomic_write_json(evaluation / "metrics.json", metrics)
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "report.md").write_text(
        "# Canonical FGE report\n"
        + json.dumps(dict(report_inputs), sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validate_result(config, staging, publish_completion=True)
    completed = validate_result(config, staging)
    if completed.mode != "read_only":
        raise HardFailure("completed staging did not enter read-only validation mode")
    return audit_mapping


def _snapshot_json(snapshot: SourceSnapshot) -> dict[str, str]:
    return dict(snapshot.files)


def write_external_audit(
    audit_root: Path, experiment: str, payload: Mapping[str, Any]
) -> Path:
    """Atomically publish the durable authorization record outside formal output."""

    root = _safe_absolute(Path(audit_root), "audit path")
    if Path(experiment).name != experiment:
        raise HardFailure("external audit experiment escapes audit root")
    destination = _safe_absolute(root / experiment, "audit path")
    if destination.exists():
        raise HardFailure("external migration audit destination already exists")
    with sibling_staging(destination) as staging:
        atomic_write_json(staging / "audit.json", payload)
    return destination / "audit.json"


def convert_legacy_run(
    source: Path,
    destination: Path,
    audit_root: Path,
    config: FGEConfig,
    base_checkpoint: Path,
    *,
    expected: LegacyExpectations | None = None,
) -> Path:
    """Validate, stage, audit and atomically publish one legacy result."""

    final = _safe_absolute(Path(destination), "artifact destination")
    safe_audit_root = _safe_absolute(Path(audit_root), "audit path")
    audit_destination = _safe_absolute(safe_audit_root / final.name, "audit path")
    if (
        _contains(final, safe_audit_root)
        or _contains(safe_audit_root, final)
        or _contains(final, audit_destination)
        or _contains(audit_destination, final)
    ):
        raise HardFailure("artifact destination and audit must be separate trees")
    if final.exists():
        raise HardFailure(f"artifact destination already exists: {final}")
    identity = _migration_code_identity(_repo_code_identity())
    run = read_legacy_run(Path(source), expected or _default_expectations())
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    published = False
    try:
        mapping = _write_formal_tree(
            staging, run, config, Path(base_checkpoint), identity
        )
        verify_source_unchanged(run.source_snapshot)
        before = _snapshot_json(run.source_snapshot)
        after = {
            relative: sha256_file(run.source_snapshot.root / relative)
            for relative in before
        }
        if after != before:
            raise HardFailure("legacy source changed before publication")
        audit = {
            "schema_version": "upet.fge.external-migration-audit.v1",
            "source_root": str(run.source_snapshot.root),
            "source_hashes_before": before,
            "source_hashes_after": after,
            "source_to_a3": mapping,
            "artifact_writer_code_identity": dict(identity),
            "validator_code_identity": dict(identity),
            "canonical_staging_signature": schema_signature(staging),
            "expected_final_destination": str(final),
            "publication_authorized": True,
        }
        write_external_audit(Path(audit_root), final.name, audit)
        if final.exists():
            raise HardFailure(f"artifact destination already exists: {final}")
        try:
            os.replace(staging, final)
        except OSError as exc:
            raise HardFailure(
                "unable to publish canonical migration atomically"
            ) from exc
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return final
