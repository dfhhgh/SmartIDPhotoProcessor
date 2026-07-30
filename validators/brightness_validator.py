"""
Brightness validator.
"""

import cv2
import numpy as np
from insightface.app.common import Face

from config.constants import (
    BRIGHTNESS_MAX_THRESHOLD,
    BRIGHTNESS_MIN_THRESHOLD,
)
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.base_validator import BaseValidator


class BrightnessValidator(BaseValidator):
    """Validates whether an image has acceptable brightness for ID processing."""

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate image brightness using mean grayscale intensity.

        Args:
            image: Image data to validate.
            face: Optional detected face. Unused by this validator.
            parsing_result: Optional face parsing result. Unused by this validator.

        Returns:
            A ValidationMetric containing a quality score clamped to the
            range [0.0, 1.0], where 1.0 indicates brightness near the
            centre of the acceptable range.

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
        mean_intensity = float(
            grayscale_image.mean()
        )
        score = self._normalize_intensity(
            intensity=mean_intensity,
        )
        passed = bool(
            BRIGHTNESS_MIN_THRESHOLD
            <= mean_intensity
            <= BRIGHTNESS_MAX_THRESHOLD
        )
        message = self._build_message(
            passed=passed,
            mean_intensity=mean_intensity,
        )

        return ValidationMetric(
            type=ValidationType.BRIGHTNESS,
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

    def _normalize_intensity(
        self,
        intensity: float,
    ) -> float:
        """Normalize mean intensity into a brightness quality score.

        The score equals 1.0 at the midpoint of the acceptable range and
        decreases linearly to 0.5 at both boundaries, continuing toward
        0.0 for extreme brightness values.

        Args:
            intensity: Mean grayscale intensity value.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        midpoint = (
            BRIGHTNESS_MIN_THRESHOLD + BRIGHTNESS_MAX_THRESHOLD
        ) / 2.0
        half_range = (
            BRIGHTNESS_MAX_THRESHOLD - BRIGHTNESS_MIN_THRESHOLD
        ) / 2.0

        distance = abs(intensity - midpoint)
        score = 1.0 - 0.5 * (distance / half_range)

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
        mean_intensity: float,
    ) -> str:
        """Build a human-readable message for the validation result.

        Args:
            passed: Whether the brightness check passed.
            mean_intensity: Mean grayscale intensity value.

        Returns:
            A descriptive message string.
        """
        if passed:
            return "Image brightness is acceptable."

        if mean_intensity < BRIGHTNESS_MIN_THRESHOLD:
            return "Image is too dark."

        return "Image is too bright."
