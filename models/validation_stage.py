"""
Validation execution stages.
"""

from enum import StrEnum


class ValidationStage(StrEnum):
    """Execution stages for multi-stage validation short-circuiting."""

    CHEAP = "cheap"
    PARSING = "parsing"
    GLASSES = "glasses"
