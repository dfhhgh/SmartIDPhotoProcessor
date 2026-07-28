"""
Eyewear prediction model.
"""

from dataclasses import dataclass

from models.eyewear_type import EyewearType


@dataclass(slots=True)
class EyewearPrediction:
    """
    Represents the output of an eyewear classifier.

    This is a pure domain model that encapsulates the predicted
    eyewear category and its associated confidence score.
    It contains no inference logic, validation policy,
    or machine learning framework-specific behavior.
    """

    eyewear_type: EyewearType

    confidence: float

    def __post_init__(self) -> None:
        """
        Validate the prediction values.
        """

        if not isinstance(
            self.eyewear_type,
            EyewearType,
        ):
            raise TypeError(
                "Eyewear type must be an EyewearType."
            )

        if isinstance(self.confidence, bool):
            raise TypeError(
                "Confidence must be numeric."
            )

        if not isinstance(
            self.confidence,
            (int, float),
        ):
            raise TypeError(
                "Confidence must be numeric."
            )

        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(
                "Confidence must be between 0.0 and 1.0."
            )

        self.confidence = float(self.confidence)
