"""
Glasses validator.
"""

import numpy as np
from insightface.app.common import Face

from config.constants import (
    GLASSES_FAILURE_MESSAGE,
    GLASSES_SUCCESS_MESSAGE,
)
from models.eyewear_type import EyewearType
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from services.eyewear_classifier import EyewearClassifier
from validators.base_validator import BaseValidator


class GlassesValidator(BaseValidator):
    """Validates whether detected eyewear is acceptable for ID processing.

    Delegates eyewear classification to an injected EyewearClassifier and
    enforces the document policy regarding eyewear. Normal eyeglasses and
    bare faces are accepted; sunglasses are rejected.
    """

    _ACCEPTED_EYEWEAR: frozenset[EyewearType] = frozenset(
        {
            EyewearType.NONE,
            EyewearType.CLEAR_GLASSES,
            EyewearType.PRESCRIPTION_GLASSES,
        },
    )

    def __init__(
        self,
        classifier: EyewearClassifier,
    ) -> None:
        """Initialise the validator with a classifier dependency.

        Args:
            classifier: EyewearClassifier instance used to classify
                eyewear on the detected face.

        Raises:
            TypeError: If classifier is not an EyewearClassifier.
        """
        if not isinstance(
            classifier,
            EyewearClassifier,
        ):
            raise TypeError(
                "Classifier must be an EyewearClassifier."
            )

        self._classifier = classifier

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate eyewear policy using the injected classifier.

        Args:
            image: Image data to validate.
            face: Detected face required for eyewear classification.
            parsing_result: Optional face parsing result. Unused by this validator.

        Returns:
            A ValidationMetric using ValidationType.GLASSES, with the
            classifier's confidence as the score and a message indicating
            whether the detected eyewear is acceptable.

        Raises:
            TypeError: If image is not a NumPy array, or face is not a
                Face instance.
            ValueError: If image is None/empty, or face is None.
        """
        _ = parsing_result
        if image is None:
            raise ValueError(
                "Image must not be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "Image must be a numpy array."
            )

        if image.size == 0:
            raise ValueError(
                "Image must not be empty."
            )

        if face is None:
            raise ValueError(
                "Face must not be None."
            )

        if not isinstance(
            face,
            Face,
        ):
            raise TypeError(
                "Face must be a Face instance."
            )

        prediction = self._classifier.classify(
            image=image,
            face=face,
        )

        passed = prediction.eyewear_type in self._ACCEPTED_EYEWEAR

        message = (
            GLASSES_SUCCESS_MESSAGE
            if passed
            else GLASSES_FAILURE_MESSAGE
        )

        return ValidationMetric(
            type=ValidationType.GLASSES,
            passed=passed,
            score=prediction.confidence,
            message=message,
        )
