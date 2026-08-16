"""Immutable artifact paths and atomic publication primitives."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .errors import HardFailure


def _absolute_lexical(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise HardFailure(f"artifact path contains symlink: {current}")


def _safe_parent(path: Path) -> None:
    _reject_symlink_components(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular, non-symlink file."""

    source = _absolute_lexical(path)
    _reject_symlink_components(source)
    try:
        mode = source.stat().st_mode
    except OSError as error:
        raise HardFailure(f"could not stat artifact {source}: {error}") from error
    if not stat.S_ISREG(mode):
        raise HardFailure(f"artifact is not a regular file: {source}")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as error:
        raise HardFailure(f"could not read artifact {source}: {error}") from error
    return digest.hexdigest()


def _publish_bytes(path: str | Path, payload: bytes) -> Path:
    destination = _absolute_lexical(path)
    _safe_parent(destination)
    if destination.exists() or destination.is_symlink():
        if destination.is_file() and not destination.is_symlink():
            try:
                if destination.read_bytes() == payload:
                    return destination
            except OSError as error:
                raise HardFailure(
                    f"could not verify existing artifact {destination}: {error}"
                ) from error
        raise HardFailure(
            f"artifact already exists with different content: {destination}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if (
                destination.is_file()
                and not destination.is_symlink()
                and destination.read_bytes() == payload
            ):
                return destination
            raise HardFailure(
                f"artifact already exists with different content: {destination}"
            ) from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise HardFailure(
            f"could not publish artifact {destination}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def atomic_write_json(path: str | Path, document: Any) -> Path:
    """Publish canonical UTF-8 JSON without replacing existing content."""

    try:
        payload = (
            json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise HardFailure(f"document is not JSON serializable: {error}") from error
    return _publish_bytes(path, payload)


def atomic_write_yaml(path: str | Path, document: Any) -> Path:
    """Publish deterministic UTF-8 YAML without replacing existing content."""

    try:
        payload = yaml.safe_dump(document, sort_keys=True, allow_unicode=True).encode(
            "utf-8"
        )
    except yaml.YAMLError as error:
        raise HardFailure(f"document is not YAML serializable: {error}") from error
    return _publish_bytes(path, payload)


def atomic_write_npz(path: str | Path, **arrays: Any) -> Path:
    """Publish a NumPy archive without replacing existing content."""

    buffer = io.BytesIO()
    try:
        np.savez(buffer, **arrays)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"could not serialize NumPy archive: {error}") from error
    return _publish_bytes(path, buffer.getvalue())


def atomic_write_torch(path: str | Path, document: Any) -> Path:
    """Publish a Torch document without replacing different existing content."""

    buffer = io.BytesIO()
    try:
        torch.save(document, buffer)
    except (RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not serialize Torch document: {error}") from error
    return _publish_bytes(path, buffer.getvalue())


def atomic_replace_torch(path: str | Path, document: Any) -> Path:
    """Atomically replace the mutable latest-training checkpoint."""

    destination = _absolute_lexical(path)
    _safe_parent(destination)
    buffer = io.BytesIO()
    try:
        torch.save(document, buffer)
    except (RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not serialize Torch document: {error}") from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        raise HardFailure(
            f"could not replace artifact {destination}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def copy_file_exact(source: str | Path, destination: str | Path) -> Path:
    """Stream-copy a file and publish it only when hashes match exactly."""

    source_path = _absolute_lexical(source)
    destination_path = _absolute_lexical(destination)
    source_hash = sha256_file(source_path)
    _safe_parent(destination_path)
    if destination_path.exists() or destination_path.is_symlink():
        if (
            destination_path.is_file()
            and not destination_path.is_symlink()
            and sha256_file(destination_path) == source_hash
        ):
            return destination_path
        raise HardFailure(
            f"artifact already exists with different content: {destination_path}"
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with source_path.open("rb") as source_handle, temporary.open("wb") as target:
            shutil.copyfileobj(source_handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if sha256_file(temporary) != source_hash:
            raise HardFailure(
                f"copied artifact failed SHA-256 verification: {source_path}"
            )
        try:
            os.link(temporary, destination_path)
        except FileExistsError:
            if (
                destination_path.is_file()
                and not destination_path.is_symlink()
                and sha256_file(destination_path) == source_hash
            ):
                return destination_path
            raise HardFailure(
                f"artifact already exists with different content: {destination_path}"
            ) from None
    except OSError as error:
        raise HardFailure(
            f"could not copy artifact {source_path} to {destination_path}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination_path


@contextmanager
def sibling_staging(destination: str | Path) -> Iterator[Path]:
    """Yield same-filesystem staging and publish it without replacement."""

    destination_path = _absolute_lexical(destination)
    _safe_parent(destination_path)
    if destination_path.exists() or destination_path.is_symlink():
        raise HardFailure(f"artifact already exists: {destination_path}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_path.name}.",
            suffix=".staging",
            dir=destination_path.parent,
        )
    )
    try:
        yield staging
        if destination_path.exists() or destination_path.is_symlink():
            raise HardFailure(f"artifact already exists: {destination_path}")
        try:
            staging.rename(destination_path)
            directory_fd = os.open(destination_path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError as error:
            raise HardFailure(
                f"could not publish staging {staging} to {destination_path}: {error}"
            ) from error
    finally:
        if staging.exists():
            shutil.rmtree(staging)


@dataclass(frozen=True)
class ExperimentLayout:
    """Stable public directory layout for a single experiment run."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        root_path = _absolute_lexical(root)
        _reject_symlink_components(root_path)
        object.__setattr__(self, "root", root_path)

    def member_dir(self, member_index: int) -> Path:
        if isinstance(member_index, bool) or member_index < 0:
            raise HardFailure("member_index must be a non-negative integer")
        return self.root / "members" / f"member_{member_index:03d}"

    def split_dir(self, split: str) -> Path:
        if split not in {"val", "test"}:
            raise HardFailure("split must be val or test")
        return self.root / "predictions" / split
