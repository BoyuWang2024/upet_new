"""Atomic artifact IO and canonical result locations for FGE."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import secrets
import stat
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


_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_PROC_FD_ROOT = Path("/proc/self/fd")
_AT_EMPTY_PATH = 0x1000
_AT_FDCWD = -100
_AT_SYMLINK_FOLLOW = 0x400
_RENAME_NOREPLACE = 1
_LIBC = ctypes.CDLL(None, use_errno=True)
_SECURE_ARTIFACT_PRIMITIVES = (
    all(
        operation in os.supports_dir_fd
        for operation in (os.open, os.mkdir, os.rename, os.stat)
    )
    and hasattr(os, "O_DIRECTORY")
    and hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_TMPFILE")
    and hasattr(_LIBC, "linkat")
    and hasattr(_LIBC, "renameat2")
    and _PROC_FD_ROOT.is_dir()
)


def _assert_secure_artifact_primitives() -> None:
    if not _SECURE_ARTIFACT_PRIMITIVES:
        raise HardFailure("race-safe artifact operations are unavailable")


def _open_directory_component(parent_fd: int, component: str, *, create: bool) -> int:
    try:
        return os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise
        try:
            os.mkdir(component, 0o700, dir_fd=parent_fd)
        except FileExistsError:
            pass
        else:
            os.fsync(parent_fd)
        return os.open(component, _DIRECTORY_FLAGS, dir_fd=parent_fd)


def _open_stable_directory(path: Path, *, create: bool) -> int:
    _assert_secure_artifact_primitives()
    absolute = path.absolute()
    parts = absolute.parts
    proc_prefix = (absolute.anchor, "proc", "self", "fd")
    if len(parts) >= 5 and parts[:4] == proc_prefix:
        try:
            current = os.dup(int(parts[4]))
        except (OSError, ValueError) as exc:
            raise HardFailure("artifact proc-fd anchor is invalid") from exc
        if not stat.S_ISDIR(os.fstat(current).st_mode):
            os.close(current)
            raise HardFailure("artifact proc-fd anchor is not a directory")
        components = parts[5:]
    else:
        if not parts or parts[0] != absolute.anchor:
            raise HardFailure("artifact path has no trusted filesystem anchor")
        try:
            current = os.open(absolute.anchor, _DIRECTORY_FLAGS)
        except OSError as exc:
            raise HardFailure("artifact filesystem anchor is unsafe") from exc
        components = parts[1:]
    try:
        for component in components:
            child = _open_directory_component(current, component, create=create)
            os.close(current)
            current = child
        return current
    except BaseException:
        os.close(current)
        raise


def is_bound_directory_path(path: Path) -> bool:
    """Return whether *path* is exactly one live proc-fd directory handle."""
    absolute = path.absolute()
    parts = absolute.parts
    prefix = (absolute.anchor, "proc", "self", "fd")
    if len(parts) != 5 or parts[:4] != prefix:
        return False
    try:
        return stat.S_ISDIR(os.fstat(int(parts[4])).st_mode)
    except (OSError, ValueError):
        return False


def _safe_name(path: Path) -> str:
    name = path.name
    if not name or name in {".", ".."} or Path(name).name != name:
        raise HardFailure("artifact destination name is unsafe")
    return name


def _random_name(prefix: str) -> str:
    return f"{prefix}{secrets.token_hex(16)}"


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _link_open_file(descriptor: int, parent_fd: int, destination_name: str) -> None:
    result = _LIBC.linkat(
        descriptor,
        ctypes.c_char_p(b""),
        parent_fd,
        ctypes.c_char_p(os.fsencode(destination_name)),
        _AT_EMPTY_PATH,
    )
    if result != 0 and ctypes.get_errno() in {errno.ENOENT, errno.EPERM}:
        proc_source = os.fsencode(f"/proc/self/fd/{descriptor}")
        result = _LIBC.linkat(
            _AT_FDCWD,
            ctypes.c_char_p(proc_source),
            parent_fd,
            ctypes.c_char_p(os.fsencode(destination_name)),
            _AT_SYMLINK_FOLLOW,
        )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination_name)


def _rename_directory_noreplace(parent_fd: int, source: str, destination: str) -> None:
    result = _LIBC.renameat2(
        parent_fd,
        ctypes.c_char_p(os.fsencode(source)),
        parent_fd,
        ctypes.c_char_p(os.fsencode(destination)),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def _verify_parent_binding(path: Path, expected_fd: int) -> None:
    actual_fd = _open_stable_directory(path, create=False)
    try:
        if not _same_inode(os.fstat(actual_fd), os.fstat(expected_fd)):
            raise HardFailure("artifact destination parent identity changed")
    finally:
        os.close(actual_fd)


def _verify_entry_binding(
    parent_fd: int, name: str, expected: os.stat_result, *, directory: bool
) -> None:
    flags = _DIRECTORY_FLAGS if directory else os.O_RDONLY | os.O_NOFOLLOW
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        if not _same_inode(os.fstat(descriptor), expected):
            raise HardFailure("published artifact identity differs from staged inode")
    finally:
        os.close(descriptor)


def _read_open_file(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _atomic_store(destination: Path, payload: bytes) -> None:
    parent_fd = _open_stable_directory(destination.parent, create=True)
    descriptor = -1
    try:
        destination_name = _safe_name(destination)
        try:
            existing = os.open(
                destination_name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd
            )
        except FileNotFoundError:
            pass
        else:
            try:
                if not stat.S_ISREG(os.fstat(existing).st_mode):
                    raise HardFailure("artifact destination is not a regular file")
                if _read_open_file(existing) != payload:
                    raise HardFailure("immutable artifact destination already exists")
            finally:
                os.close(existing)
            _verify_parent_binding(destination.parent, parent_fd)
            return
        descriptor = os.open(".", os.O_RDWR | os.O_TMPFILE, 0o600, dir_fd=parent_fd)
        with os.fdopen(os.dup(descriptor), "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        expected = os.fstat(descriptor)
        _link_open_file(descriptor, parent_fd, destination_name)
        _verify_entry_binding(parent_fd, destination_name, expected, directory=False)
        os.fsync(parent_fd)
        _verify_parent_binding(destination.parent, parent_fd)
    except FileExistsError as exc:
        raise HardFailure("immutable artifact destination already exists") from exc
    except OSError as exc:
        raise HardFailure("unable to publish descriptor-bound artifact") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any] | list[Any]) -> None:
    """Atomically write strict JSON through a descriptor-bound sibling."""
    destination = Path(path)
    try:
        document = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except (TypeError, ValueError) as exc:
        raise HardFailure(f"non-finite JSON or unsupported value: {exc}") from exc
    _atomic_store(destination, document.encode("utf-8"))


def atomic_write_yaml(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Atomically write deterministic UTF-8 YAML through a bound sibling."""
    destination = Path(path)
    try:
        document = yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True)
    except yaml.YAMLError as exc:
        raise HardFailure(f"unable to serialize YAML: {exc}") from exc
    _atomic_store(destination, document.encode("utf-8"))


def atomic_torch_save(path: str | Path, payload: Any) -> None:
    """Atomically store a torch payload through a descriptor-bound sibling."""
    stream = io.BytesIO()
    torch.save(payload, stream)
    _atomic_store(Path(path), stream.getvalue())


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
    """Build through a bound directory and retain failed staging for audit."""
    final_path = Path(destination).absolute()
    final_name = _safe_name(final_path)
    parent_fd = _open_stable_directory(final_path.parent, create=True)
    staging_fd = -1
    try:
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise HardFailure(f"artifact destination already exists: {final_path}")
        for _ in range(100):
            staging_name = _random_name(f".{final_name}.staging-")
            try:
                os.mkdir(staging_name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            break
        else:
            raise HardFailure("unable to allocate a unique staging directory")
        staging_fd = _open_directory_component(parent_fd, staging_name, create=False)
        expected = os.fstat(staging_fd)
        yield _PROC_FD_ROOT / str(staging_fd)
        try:
            os.stat(final_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise HardFailure(f"artifact destination already exists: {final_path}")
        try:
            os.fsync(staging_fd)
            _rename_directory_noreplace(parent_fd, staging_name, final_name)
            _verify_entry_binding(parent_fd, final_name, expected, directory=True)
            os.fsync(parent_fd)
            _verify_parent_binding(final_path.parent, parent_fd)
        except OSError as exc:
            raise HardFailure(
                "unable to publish descriptor-bound staging atomically"
            ) from exc
    finally:
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(parent_fd)
