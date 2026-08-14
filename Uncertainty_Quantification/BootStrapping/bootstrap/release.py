"""Deterministic source releases for the formal BootStrapping workflow."""

from __future__ import annotations

import gzip
import os
import stat
import tarfile
import tempfile
from pathlib import Path

from .errors import HardFailure


_PUBLIC_ENTRIES = ("__init__.py", "README.md", "bootstrap", "scripts", "configs")


def _release_files(source: Path) -> tuple[Path, ...]:
    files: list[Path] = []
    for name in _PUBLIC_ENTRIES:
        entry = source / name
        if not entry.exists():
            raise HardFailure(f"release input is missing: {entry}")
        candidates = (entry,) if entry.is_file() else tuple(entry.rglob("*"))
        for path in candidates:
            if path.is_symlink():
                raise HardFailure(f"release input must not contain symlinks: {path}")
            if path.is_dir():
                continue
            try:
                mode = path.stat().st_mode
            except OSError as error:
                raise HardFailure(
                    f"could not stat release input {path}: {error}"
                ) from error
            if not stat.S_ISREG(mode):
                raise HardFailure(f"release input is not a regular file: {path}")
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                continue
            files.append(path)
    return tuple(sorted(files, key=lambda path: path.relative_to(source).as_posix()))


def build_source_release(source_root: str | Path, destination: str | Path) -> Path:
    """Build a deterministic formal source archive with operational tools excluded."""

    source = Path(source_root).expanduser().resolve()
    target = Path(destination).expanduser().absolute()
    if source.is_symlink() or not source.is_dir():
        raise HardFailure("release source must be a regular directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0
            ) as compressed:
                with tarfile.open(fileobj=compressed, mode="w") as archive:
                    for path in _release_files(source):
                        relative = path.relative_to(source)
                        information = archive.gettarinfo(
                            str(path),
                            arcname=(Path("BootStrapping") / relative).as_posix(),
                        )
                        information.uid = 0
                        information.gid = 0
                        information.uname = ""
                        information.gname = ""
                        information.mtime = 0
                        information.mode = 0o644
                        with path.open("rb") as handle:
                            archive.addfile(information, handle)
            raw.flush()
            os.fsync(raw.fileno())
        if target.exists():
            if target.is_file() and target.read_bytes() == temporary.read_bytes():
                return target
            raise HardFailure(
                f"release archive already exists with different content: {target}"
            )
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_file() and target.read_bytes() == temporary.read_bytes():
                return target
            raise HardFailure(
                f"release archive already exists with different content: {target}"
            ) from None
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except HardFailure:
        raise
    except OSError as error:
        raise HardFailure(
            f"could not build release archive {target}: {error}"
        ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return target
