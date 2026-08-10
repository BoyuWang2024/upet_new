"""Atomic artifact IO and canonical result locations for FGE."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import yaml

from .errors import HardFailure


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 digest of one regular file."""
    source = Path(path)
    if not source.is_file():
        raise HardFailure(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _sibling_temporary_file(destination: Path) -> Iterator[Path]:
    """Yield a same-filesystem temporary path, retaining failures for audit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    yield temporary


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def atomic_write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Atomically write strict JSON through a sibling temporary file."""
    destination = Path(path)
    try:
        document = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise HardFailure(f"non-finite JSON or unsupported value: {exc}") from exc
    with _sibling_temporary_file(destination) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)


def atomic_write_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a deterministic UTF-8 YAML document."""
    destination = Path(path)
    try:
        document = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as exc:
        raise HardFailure(f"unable to serialize YAML: {exc}") from exc
    with _sibling_temporary_file(destination) as temporary:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(document)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    """Atomically store a torch payload through a sibling temporary file."""
    destination = Path(path)
    with _sibling_temporary_file(destination) as temporary:
        torch.save(payload, temporary)
        _fsync(temporary)
        os.replace(temporary, destination)


def assert_safe_result_path(root: str | Path, path: str | Path) -> None:
    """Reject a formal output path whose existing ancestors escape via symlinks."""
    result_root = Path(root).absolute()
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise HardFailure(f"formal artifact destination is a symlink: {candidate}")
    try:
        relative = candidate.relative_to(result_root)
    except ValueError as exc:
        raise HardFailure(f"formal artifact path escapes result root: {path}") from exc
    ancestor = result_root
    if ancestor.exists() and ancestor.is_symlink():
        raise HardFailure("formal result root must not be a symlink")
    for part in relative.parts[:-1]:
        ancestor = ancestor / part
        if ancestor.exists() and ancestor.is_symlink():
            raise HardFailure(f"formal artifact ancestor is a symlink: {ancestor}")
    try:
        candidate.parent.resolve().relative_to(result_root.resolve())
    except ValueError as exc:
        raise HardFailure(f"formal artifact path escapes result root: {path}") from exc


def normalize_artifact_path(root: str | Path, path: str | Path) -> str:
    """Return a POSIX relative artifact path or reject paths outside *root*."""
    result_root = Path(root).resolve()
    candidate = Path(path).resolve()
    try:
        relative = candidate.relative_to(result_root)
    except ValueError as exc:
        raise HardFailure(f"artifact path must be inside result root: {path}") from exc
    if relative == Path("."):
        raise HardFailure(f"artifact path must name a file inside result root: {path}")
    return relative.as_posix()


@dataclass(frozen=True)
class ExperimentLayout:
    """Fixed paths for a single formal FGE experiment result."""

    root: Path

    @property
    def preflight_dir(self) -> Path:
        return self.root / "preflight"

    @property
    def training_dir(self) -> Path:
        return self.root / "training"

    @property
    def training_manifest(self) -> Path:
        return self.training_dir / "manifest.json"

    @property
    def prediction_dir(self) -> Path:
        return self.root / "prediction"

    @property
    def prediction_tensor(self) -> Path:
        return self.prediction_dir / "test_raw.pt"

    @property
    def prediction_manifest(self) -> Path:
        return self.prediction_dir / "manifest.json"

    @property
    def evaluation_dir(self) -> Path:
        return self.root / "evaluation"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def result_manifest(self) -> Path:
        return self.root / "result_manifest.json"


@contextmanager
def sibling_staging(destination: str | Path) -> Iterator[Path]:
    """Build a sibling directory and publish it atomically when complete."""
    final_path = Path(destination).resolve()
    if final_path.exists():
        raise HardFailure(f"artifact destination already exists: {final_path}")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final_path.name}.staging-", dir=final_path.parent)
    )
    published = False
    try:
        yield staging
        if final_path.exists():
            raise HardFailure(f"artifact destination already exists: {final_path}")
        os.replace(staging, final_path)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
