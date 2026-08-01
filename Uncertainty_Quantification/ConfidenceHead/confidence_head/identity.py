"""Canonical identifiers for confidence-head configurations and artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json(payload: Mapping[str, Any]) -> bytes:
    """Encode a mapping as deterministic, compact UTF-8 JSON."""
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")


def stable_id(namespace: str, payload: Mapping[str, Any]) -> str:
    """Return a namespaced identifier derived from canonical JSON."""
    if not namespace:
        raise ValueError("namespace must not be empty")
    digest = hashlib.sha256(canonical_json(payload)).hexdigest()
    return f"{namespace}-{digest[:16]}"


def _legacy_component_identity_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    model = normalized.get("model")
    force = model.get("force") if isinstance(model, dict) else None
    if isinstance(force, dict) and force.get("target_mode") == "component":
        force.pop("target_mode")
    return normalized


def config_id(payload: Mapping[str, Any]) -> str:
    return stable_id("config", _legacy_component_identity_payload(payload))


def cache_id(payload: Mapping[str, Any]) -> str:
    return stable_id("cache", payload)


def binning_id(payload: Mapping[str, Any]) -> str:
    return stable_id("binning", payload)


def model_loss_id(payload: Mapping[str, Any]) -> str:
    return stable_id("model-loss", _legacy_component_identity_payload(payload))


def run_id(payload: Mapping[str, Any]) -> str:
    return stable_id("run", payload)
