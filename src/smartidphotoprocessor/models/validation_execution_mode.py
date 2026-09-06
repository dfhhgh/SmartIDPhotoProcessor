"""
Validation execution modes.
"""

from enum import StrEnum


class ValidationExecutionMode(StrEnum):
    """Controls how the ValidationOrchestrator runs validator stages.

    PRODUCTION: Short-circuits on the first failing stage to minimize
        unnecessary AI inference (FaceParserService).
    DEVELOPMENT: Runs every stage and every validator regardless of failures,
        so every ValidationMetric is collected. Intended for debugging,
        validator calibration, dataset analysis, benchmarking, and threshold
        tuning.
    """

    PRODUCTION = "production"
    DEVELOPMENT = "development"
