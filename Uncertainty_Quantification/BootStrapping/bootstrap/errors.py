"""Errors exposed by BootStrapping command-line stages."""


class HardFailure(RuntimeError):
    """A deterministic contract failure that should stop the current stage."""
