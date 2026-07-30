"""
Blur validator.
"""

import cv2
import numpy as np
from insightface.app.common import Face

from config.constants import (
    BLUR_MAX_EXPECTED_VALUE,
    BLUR_THRESHOLD,
)
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.base_validator import BaseValidator


class BlurValidator(BaseValidator):
    """Validates whether an image is sharp enough for ID processing."""

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate image sharpness using variance of the Laplacian.

        Args:
            image: Image data to validate.
            face: Optional detected face. Unused by this validator.
            parsing_result: Optional face parsing result. Unused by this validator.

        Returns:
            A normalized score clamped to the range [0.0, 1.0]..

        Raises:
            TypeError: If image is not a NumPy array.
            ValueError: If image is None or empty.
        """
        _ = face
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

        grayscale_image = self._convert_to_grayscale(
            image=image,
        )
        variance = float(
            cv2.Laplacian(
                grayscale_image,
                cv2.CV_64F,
            ).var()
        )
        score = self._normalize_variance(
            variance=variance,
        )
        passed = bool(
            variance >= BLUR_THRESHOLD
        )
        message = (
            "Image sharpness is acceptable."
            if passed
            else "Image is too blurry for reliable processing."
        )

        return ValidationMetric(
            type=ValidationType.BLUR,
            passed=passed,
            score=score,
            message=message,
        )

    def _convert_to_grayscale(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Convert an image to grayscale.

        Args:
            image: Image data to convert.

        Returns:
            Grayscale image data.

        Raises:
            ValueError: If image dimensions or channel count are unsupported.
        """
        if image.ndim == 2:
            return image

        if image.ndim != 3:
            raise ValueError(
                "Image must be grayscale, BGR, or BGRA."
            )

        channel_count = image.shape[2]

        if channel_count == 3:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY,
            )

        if channel_count == 4:
            return cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY,
            )

        raise ValueError(
            "Image must be grayscale, BGR, or BGRA."
        )

    def _normalize_variance(
        self,
        variance: float,
    ) -> float:
        """Normalize Laplacian variance into a bounded blur score.

        Args:
            variance: Raw variance of the Laplacian.

        Returns:
            Normalized score between the configured minimum and maximum score.
        """
        score = variance / BLUR_MAX_EXPECTED_VALUE

        return float(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            )
        )