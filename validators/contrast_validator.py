"""
Contrast validator.
"""

import cv2
import numpy as np
from insightface.app.common import Face

from config.constants import (
    CONTRAST_MAX_EXPECTED_VALUE,
    CONTRAST_MIN_THRESHOLD,
)
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.base_validator import BaseValidator


class ContrastValidator(BaseValidator):
    """Validates whether an image has acceptable contrast for ID processing."""

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate image contrast using standard deviation of grayscale intensity.

        Args:
            image: Image data to validate.
            face: Optional detected face. Unused by this validator.
            parsing_result: Optional face parsing result. Unused by this validator.

        Returns:
            A ValidationMetric containing a quality score clamped to the
            range [0.0, 1.0], where 1.0 indicates high contrast.

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
        contrast = float(
            grayscale_image.std()
        )
        score = self._normalize_contrast(
            contrast=contrast,
        )
        passed = bool(
            contrast >= CONTRAST_MIN_THRESHOLD
        )
        message = self._build_message(
            passed=passed,
        )

        return ValidationMetric(
            type=ValidationType.CONTRAST,
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

    def _normalize_contrast(
        self,
        contrast: float,
    ) -> float:
        """Normalize contrast into a bounded quality score.

        The score increases linearly with contrast and is clamped to the
        range [0.0, 1.0].

        Args:
            contrast: Standard deviation of grayscale intensity.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = contrast / CONTRAST_MAX_EXPECTED_VALUE

        return float(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            )
        )

    def _build_message(
        self,
        passed: bool,
    ) -> str:
        """Build a human-readable message for the validation result.

        Args:
            passed: Whether the contrast check passed.

        Returns:
            A descriptive message string.
        """
        if passed:
            return "Image contrast is acceptable."

        return "Image contrast is too low."
