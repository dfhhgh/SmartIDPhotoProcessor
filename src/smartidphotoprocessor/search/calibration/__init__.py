"""Threshold calibration and evaluation for reverse search similarity scoring."""

from smartidphotoprocessor.search.calibration.pair_generator import PairGenerator, PairSample
from smartidphotoprocessor.search.calibration.similarity_evaluator import EvaluatedPair, SimilarityEvaluator
from smartidphotoprocessor.search.calibration.threshold_evaluator import (
    DistributionStats,
    ThresholdEvaluator,
    ThresholdMetrics,
)
from smartidphotoprocessor.search.calibration.calibration_report import CalibrationReporter, CalibrationSummary

__all__ = [
    "PairGenerator",
    "PairSample",
    "SimilarityEvaluator",
    "EvaluatedPair",
    "ThresholdEvaluator",
    "DistributionStats",
    "ThresholdMetrics",
    "CalibrationReporter",
    "CalibrationSummary",
]
