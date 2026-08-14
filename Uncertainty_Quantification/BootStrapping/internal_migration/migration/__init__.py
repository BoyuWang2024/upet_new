"""Strict readers and converters for the isolated result adapter."""

from .converter import MigrationPublication, convert_legacy_run
from .legacy_reader import LegacyRunAudit, inspect_legacy_run

__all__ = [
    "LegacyRunAudit",
    "MigrationPublication",
    "convert_legacy_run",
    "inspect_legacy_run",
]
