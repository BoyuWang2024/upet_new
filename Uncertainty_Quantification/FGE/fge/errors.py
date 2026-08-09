"""Failure types for the FGE workflow."""


class HardFailure(RuntimeError):
    """An unrecoverable violation of the FGE workflow contract."""
