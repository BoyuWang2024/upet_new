"""No-compute conversion into the public checkpoint/prediction schema."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...bootstrap.artifacts import (
    atomic_write_json,
    copy_file_exact,
    sha256_file,
    sibling_staging,
)
from ...bootstrap.errors import HardFailure
from ...bootstrap.prediction import PredictionStore
from .legacy_reader import inspect_legacy_run, load_legacy_payload
from .normalization import assert_same_targets, concatenate_legacy_chunks


@dataclass(frozen=True)
class MigrationPublication:
    destination: Path
    member_count: int
    splits: tuple[str, ...]


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _checkpoint_capabilities(path: Path) -> dict[str, object]:
    payload = load_legacy_payload(path)
    raw = payload.get("raw_state")
    ema = payload.get("ema_state")
    inference_ready = isinstance(raw, dict) and bool(raw)
    if ema is not None and not isinstance(ema, dict):
        raise HardFailure(f"checkpoint EMA state is invalid: {path}")
    return {
        "epoch": payload.get("epoch"),
        "validation_loss": payload.get("validation_loss"),
        "inference_ready": inference_ready,
        "resume_ready": all(
            key in payload
            for key in ("optimizer_state_dict", "python_rng_state", "torch_rng_state")
        ),
    }


def convert_legacy_run(
    source: str | Path,
    destination: str | Path,
    config: object,
    audit_root: str | Path,
) -> MigrationPublication:
    """Copy checkpoints exactly and normalize predictions without any computation."""

    source_path = Path(source).expanduser().resolve()
    destination_path = Path(destination).expanduser().resolve()
    audit_path = Path(audit_root).expanduser().resolve()
    if _contains(source_path, destination_path) or _contains(
        destination_path, source_path
    ):
        raise HardFailure("source and destination must be separate directory trees")
    if _contains(destination_path, audit_path):
        raise HardFailure("audit_root must be outside the formal destination")
    audit = inspect_legacy_run(source_path, config)
    try:
        units = {
            "energy": config.data.units.energy,  # type: ignore[attr-defined]
            "forces": config.data.units.forces,  # type: ignore[attr-defined]
            "stress": config.data.units.stress,  # type: ignore[attr-defined]
        }
    except AttributeError as error:
        raise HardFailure("adapter units are missing") from error

    audit_records: list[dict[str, object]] = []
    with sibling_staging(destination_path) as staging:
        member_manifests: list[dict[str, object]] = []
        for zero_index, member in enumerate(audit.members):
            checkpoint_records: dict[str, object] = {}
            for kind, source_checkpoint in member.checkpoints.items():
                target = (
                    staging
                    / "members"
                    / f"member_{zero_index:03d}"
                    / "checkpoints"
                    / f"{kind}.pt"
                )
                copy_file_exact(source_checkpoint, target)
                if sha256_file(source_checkpoint) != sha256_file(target):
                    raise HardFailure(f"checkpoint copy SHA-256 mismatch: {target}")
                checkpoint_records[kind] = {
                    "path": f"checkpoints/{kind}.pt",
                    "sha256": sha256_file(target),
                    "size_bytes": target.stat().st_size,
                    **_checkpoint_capabilities(source_checkpoint),
                }
                audit_records.append(
                    {
                        "source": str(source_checkpoint),
                        "destination": str(target.relative_to(staging)),
                        "sha256": sha256_file(target),
                    }
                )
            member_manifest = {
                "schema": "upet.bootstrap.member/v1",
                "member_index": zero_index,
                "checkpoints": checkpoint_records,
            }
            atomic_write_json(
                staging / "members" / f"member_{zero_index:03d}" / "manifest.json",
                member_manifest,
            )
            member_manifests.append(member_manifest)

        split_manifests: list[dict[str, object]] = []
        for split in audit.splits:
            store = PredictionStore(staging / "predictions", split=split, units=units)
            canonical_targets = None
            publications = []
            for zero_index, member in enumerate(audit.members):
                targets, values = concatenate_legacy_chunks(member.chunks[split])
                if canonical_targets is None:
                    canonical_targets = targets
                    store.write_targets(targets)
                else:
                    assert_same_targets(canonical_targets, targets)
                publication = store.write_member(zero_index, "raw", values)
                publications.append(
                    {
                        "member_index": zero_index,
                        "mode": "raw",
                        "path": str(publication.path.relative_to(store.split_root)),
                        "sha256": publication.sha256,
                        "shapes": publication.shapes,
                        "dtypes": publication.dtypes,
                    }
                )
            assert canonical_targets is not None
            split_manifest = {
                "schema": "upet.bootstrap.predictions/v1",
                "split": split,
                "units": units,
                "member_count": len(audit.members),
                "targets": {
                    "path": "targets.npz",
                    "sha256": sha256_file(store.targets_path),
                },
                "members": publications,
            }
            atomic_write_json(store.split_root / "manifest.json", split_manifest)
            split_manifests.append(split_manifest)

        atomic_write_json(
            staging / "run_manifest.json",
            {
                "schema": "upet.bootstrap.run/v1",
                "member_count": len(audit.members),
                "parameter_modes": ["raw"],
                "stages": {
                    "checkpoints": "complete",
                    "predictions": "complete",
                    "uncertainty": "pending",
                },
                "members": [
                    f"members/member_{index:03d}/manifest.json"
                    for index in range(len(member_manifests))
                ],
                "prediction_splits": [item["split"] for item in split_manifests],
            },
        )

    audit_path.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        audit_path / "migration_audit.json",
        {
            "source": str(source_path),
            "destination": str(destination_path),
            "checkpoint_copies": audit_records,
        },
    )
    return MigrationPublication(destination_path, len(audit.members), audit.splits)
