"""
Validation metric model.
"""

from dataclasses import dataclass

from models.validation_type import ValidationType


@dataclass(slots=True)
class ValidationMetric:
    """
    Represents the result of a single validation rule.
    """

    type: ValidationType

    passed: bool

    score: float

    message: str = ""

    def __post_init__(self) -> None:
        """
        Validate the metric values.
        """

        if not isinstance(
            self.type,
            ValidationType,
        ):
            raise TypeError(
                "Validation type must be a ValidationType."
            )

        if not isinstance(
            self.passed,
            bool,
        ):
            raise TypeError(
                "Passed flag must be a bool."
            )

        if isinstance(self.score, bool):
            raise TypeError(
                "Score must be numeric."
            )

        if not isinstance(
            self.score,
            (int, float),
        ):
            raise TypeError(
                "Score must be numeric."
            )

        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError(
                "Score must be between 0.0 and 1.0."
            )

        if not isinstance(
            self.message,
            str,
        ):
            raise TypeError(
                "Message must be a string."
            )

        self.message = self.message.strip()

        self.score = float(self.score)
