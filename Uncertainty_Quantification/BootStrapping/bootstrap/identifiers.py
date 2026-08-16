"""Validation for names used as artifact directory keys."""

from __future__ import annotations

import re

from .errors import HardFailure

_ARTIFACT_KEY = re.compile(r"[a-z0-9][a-z0-9_]*\Z")


def validate_artifact_key(value: object, location: str) -> str:
    """Return a path-safe artifact key or raise a deterministic failure."""

    if not isinstance(value, str) or _ARTIFACT_KEY.fullmatch(value) is None:
        raise HardFailure(
            f"{location} must use lowercase letters, digits, or underscores"
        )
    return value
