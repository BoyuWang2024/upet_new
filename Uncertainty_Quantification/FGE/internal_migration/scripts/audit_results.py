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
    from Uncertainty_Quantification.FGE.internal_migration.scripts._cli import run
else:
    from ...fge.artifacts import sha256_file
    from ...fge.errors import HardFailure
    from ...fge.validation import schema_signature, validate_completed_result
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


def _safe_absolute(path: Path, label: str) -> Path:
    candidate = Path(path).absolute()
    for item in (candidate, *candidate.parents):
        if item.is_symlink():
            raise HardFailure(f"{label} must not contain a symbolic link")
    return candidate.resolve()


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


def audit(destination: Path, audit_root: Path) -> None:
    final = _safe_absolute(destination, "destination")
    root = _safe_absolute(audit_root, "audit root")
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
    if not isinstance(mapping, list) or not mapping:
        raise HardFailure("external audit A3 mapping is invalid")
    for index, item in enumerate(mapping, start=1):
        if not isinstance(item, Mapping) or set(item) != _MAPPING_KEYS:
            raise HardFailure("external audit A3 mapping schema is invalid")
        if item.get("member_id") != f"member_{index:03d}":
            raise HardFailure("external audit member order is invalid")
        source_checkpoint = item.get("source_checkpoint")
        if not isinstance(source_checkpoint, str):
            raise HardFailure("external audit source checkpoint is invalid")
        source_path = _safe_absolute(Path(source_checkpoint), "source checkpoint")
        try:
            source_relative = source_path.relative_to(source).as_posix()
        except ValueError as exc:
            raise HardFailure("source checkpoint escapes source root") from exc
        if source_files.get(source_relative) != source_path:
            raise HardFailure("source checkpoint is absent from source inventory")
        if sha256_file(source_path) != _sha(
            item.get("source_sha256"), "source checkpoint"
        ):
            raise HardFailure("source checkpoint SHA differs")
        expected_a3 = f"training/members/member_{index:03d}.pt"
        if item.get("a3_path") != expected_a3:
            raise HardFailure("external audit A3 path is not canonical")
        a3 = _safe_relative(final, item.get("a3_path"), "A3 artifact")
        if sha256_file(a3) != _sha(item.get("a3_sha256"), "A3 artifact"):
            raise HardFailure("A3 artifact SHA differs")

    writer = _identity(document.get("artifact_writer_code_identity"), "writer identity")
    validator = _identity(document.get("validator_code_identity"), "validator identity")
    validate_completed_result(final)
    training = _json(final / "training/manifest.json", "training manifest")
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
