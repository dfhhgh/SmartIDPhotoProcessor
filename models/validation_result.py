"""
Validation result model.
"""

from dataclasses import dataclass, field

from models.validation_metric import ValidationMetric


@dataclass(slots=True)
class ValidationResult:
    """
    Represents the overall validation result.
    """

    passed: bool

    overall_score: float

    metrics: list[ValidationMetric] = field(
        default_factory=list,
    )