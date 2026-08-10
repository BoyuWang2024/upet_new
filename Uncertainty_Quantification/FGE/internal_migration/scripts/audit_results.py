from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from Uncertainty_Quantification.FGE.fge.artifacts import sha256_file
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import (
        schema_signature,
        validate_completed_result,
    )
    from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
        _contains,
        _safe_absolute,
    )
    from Uncertainty_Quantification.FGE.internal_migration.scripts._cli import run
else:
    from ...fge.artifacts import sha256_file
    from ...fge.errors import HardFailure
    from ...fge.validation import schema_signature, validate_completed_result
    from ..migration.converter import _contains, _safe_absolute
    from ._cli import run


_AUDIT_KEYS = {
    "schema_version",
    "source_root",
    "source_hashes_before",
    "source_hashes_after",
    "source_to_a3",
    "artifact_writer_code_identity",
    "validator_code_identity",
    "canonical_staging_signature",
    "expected_final_destination",
    "publication_authorized",
}
_MAPPING_KEYS = {
    "member_id",
    "source_checkpoint",
    "source_sha256",
    "a3_path",
    "a3_sha256",
}
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def _safe_relative(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str):
        raise HardFailure(f"{label} path is invalid")
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise HardFailure(f"{label} path escapes root")
    candidate = root / relative
    for item in (candidate, *candidate.parents):
        if item == root.parent:
            break
        if item.is_symlink():
            raise HardFailure(f"{label} must not contain a symbolic link")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HardFailure(f"{label} path escapes root") from exc
    if not resolved.is_file():
        raise HardFailure(f"{label} is not a regular file")
    return resolved


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise HardFailure(f"{label} must be a SHA256")
    return value


def _identity(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"commit", "dirty_sha256"}:
        raise HardFailure(f"{label} has an invalid schema")
    commit = value.get("commit")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise HardFailure(f"{label} commit is invalid")
    return {"commit": commit, "dirty_sha256": _sha(value.get("dirty_sha256"), label)}


def _json(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardFailure(f"{label} cannot be loaded") from exc
    if not isinstance(value, Mapping):
        raise HardFailure(f"{label} must be a mapping")
    return value


def _canonical_source_member(
    source_files: Mapping[str, Path], member_id: str
) -> tuple[str, Path]:
    expected_name = f"{member_id}.ckpt"
    matches = [
        (relative, candidate)
        for relative, candidate in source_files.items()
        if Path(relative).name == expected_name
    ]
    if len(matches) != 1:
        raise HardFailure("source checkpoint inventory is not unique for member")
    return matches[0]


def audit(destination: Path, audit_root: Path) -> None:
    final = _safe_absolute(destination, "destination")
    root = _safe_absolute(audit_root, "audit root")
    if _contains(final, root) or _contains(root, final):
        raise HardFailure("destination and audit root must be separate trees")
    path = _safe_relative(root, f"{final.name}/audit.json", "external audit")
    document = _json(path, "external audit")
    if set(document) != _AUDIT_KEYS:
        raise HardFailure("external audit has an invalid key inventory")
    if (
        document.get("schema_version") != "upet.fge.external-migration-audit.v1"
        or document.get("publication_authorized") is not True
        or document.get("expected_final_destination") != str(final)
    ):
        raise HardFailure("external audit is not bound to this destination")

    validate_completed_result(final)
    training = _json(final / "training/manifest.json", "training manifest")
    formal_members = training.get("members")
    member_count = training.get("member_count")
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or not isinstance(formal_members, list)
        or len(formal_members) != member_count
    ):
        raise HardFailure("validated training member inventory is invalid")

    source_value = document.get("source_root")
    if not isinstance(source_value, str) or not Path(source_value).is_absolute():
        raise HardFailure("external audit source root is invalid")
    source = _safe_absolute(Path(source_value), "source root")
    before, after = (
        document.get("source_hashes_before"),
        document.get("source_hashes_after"),
    )
    if not isinstance(before, Mapping) or not before or before != after:
        raise HardFailure("external audit source identity is invalid")
    source_files: dict[str, Path] = {}
    for relative, digest in before.items():
        if not isinstance(relative, str):
            raise HardFailure("external audit source inventory is invalid")
        candidate = _safe_relative(source, relative, "source file")
        if sha256_file(candidate) != _sha(digest, "source digest"):
            raise HardFailure("external audit source has changed")
        source_files[relative] = candidate

    mapping = document.get("source_to_a3")
    if not isinstance(mapping, list) or len(mapping) != member_count:
        raise HardFailure("external audit A3 mapping length is invalid")
    for index, (item, formal_member) in enumerate(
        zip(mapping, formal_members, strict=True), start=1
    ):
        if not isinstance(item, Mapping) or set(item) != _MAPPING_KEYS:
            raise HardFailure("external audit A3 mapping schema is invalid")
        if not isinstance(formal_member, Mapping) or set(formal_member) != {
            "member_id",
            "sha256",
            "cycle",
            "endpoint_global_step",
        }:
            raise HardFailure("validated training member entry is invalid")
        expected_id = f"member_{index:03d}"
        if (
            item.get("member_id") != expected_id
            or formal_member.get("member_id") != expected_id
        ):
            raise HardFailure("external audit member order is invalid")
        source_relative, expected_source = _canonical_source_member(
            source_files, expected_id
        )
        source_checkpoint = item.get("source_checkpoint")
        if source_checkpoint != str(expected_source):
            raise HardFailure("source checkpoint path differs from canonical member")
        source_path = _safe_absolute(expected_source, "source checkpoint")
        expected_source_sha = _sha(before[source_relative], "source checkpoint")
        if (
            _sha(item.get("source_sha256"), "source checkpoint") != expected_source_sha
            or sha256_file(source_path) != expected_source_sha
        ):
            raise HardFailure("source checkpoint SHA differs from canonical member")
        expected_a3 = f"training/members/member_{index:03d}.pt"
        if item.get("a3_path") != expected_a3:
            raise HardFailure("external audit A3 path is not canonical")
        a3 = _safe_relative(final, item.get("a3_path"), "A3 artifact")
        a3_sha = _sha(item.get("a3_sha256"), "A3 artifact")
        if formal_member.get("sha256") != a3_sha or sha256_file(a3) != a3_sha:
            raise HardFailure("A3 artifact SHA differs from validated training member")

    writer = _identity(document.get("artifact_writer_code_identity"), "writer identity")
    validator = _identity(document.get("validator_code_identity"), "validator identity")
    prediction = _json(final / "prediction/manifest.json", "prediction manifest")
    result = _json(final / "result_manifest.json", "result manifest")
    for formal in (training, prediction, result):
        if (
            formal.get("artifact_writer_code_identity") != writer
            or formal.get("validator_code_identity") != validator
        ):
            raise HardFailure(
                "external audit code identity differs from formal artifacts"
            )
    if document.get("canonical_staging_signature") != schema_signature(final):
        raise HardFailure("external audit schema signature differs from destination")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recheck a legacy migration audit")
    parser.add_argument("--destination", required=True)
    parser.add_argument("--audit-root", required=True)
    return parser


def _command(argv: Sequence[str] | None) -> None:
    args = build_parser().parse_args(argv)
    audit(Path(args.destination), Path(args.audit_root))


def main(argv: Sequence[str] | None = None) -> int:
    return run(_command, argv)


if __name__ == "__main__":
    raise SystemExit(main())
