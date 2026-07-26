"""
Validation result model.
"""

from dataclasses import dataclass

from models.validation_metric import ValidationMetric


@dataclass(slots=True)
class ValidationResult:
    """Represents the result of validating one image."""

    metrics: list[ValidationMetric]

    def __post_init__(self) -> None:
        """Validate the result values after initialization."""
        if not isinstance(
            self.metrics,
            list,
        ):
            raise TypeError(
                "Metrics must be a list."
            )
        if not self.metrics:
            raise ValueError(
                "Metrics list cannot be empty."
            )

        if not all(
            isinstance(
                metric,
                ValidationMetric,
            )
            for metric in self.metrics
        ):
            raise TypeError(
                "Every metric must be a ValidationMetric."
            )

    @property
    def is_valid(self) -> bool:
        """Return True when every validation metric passed."""
        return all(
            metric.passed
            for metric in self.metrics
        )

    @property
    def passed_metrics(self) -> list[ValidationMetric]:
        """Return a new list containing only passed metrics."""
        return [
            metric
            for metric in self.metrics
            if metric.passed
        ]

    @property
    def failed_metrics(self) -> list[ValidationMetric]:
        """Return a new list containing only failed metrics."""
        return [
            metric
            for metric in self.metrics
            if not metric.passed
        ]
