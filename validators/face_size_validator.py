"""
Face size validator.
"""

import numpy as np
from insightface.app.common import Face

from config.constants import (
    FACE_SIZE_IDEAL_RATIO,
    FACE_SIZE_MAX_RATIO,
    FACE_SIZE_MIN_RATIO,
    FLOAT_COMPARISON_EPSILON
)
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.base_validator import BaseValidator


class FaceSizeValidator(BaseValidator):
    """Validates whether the detected face occupies an acceptable proportion of the image."""

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
        parsing_result: FaceParsingResult | None = None,
    ) -> ValidationMetric:
        """Validate face size using the ratio of face area to image area.

        Args:
            image: Image data to validate.
            face: Detected face with bounding box information.
            parsing_result: Optional face parsing result. Unused by this validator.

        Returns:
            A ValidationMetric containing a quality score clamped to the
            range [0.0, 1.0], where 1.0 indicates ideal face size.

        Raises:
            TypeError: If image is not a NumPy array.
            ValueError: If image is None, empty, or face is None.
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

        face_ratio = self._compute_face_ratio(
            image=image,
            face=face,
        )
        score = self._normalize_face_ratio(
            face_ratio=face_ratio,
        )
        
        
        passed = (
            FACE_SIZE_MIN_RATIO - FLOAT_COMPARISON_EPSILON
            <= face_ratio
            <= FACE_SIZE_MAX_RATIO + FLOAT_COMPARISON_EPSILON
        )
        message = self._build_message(
            passed=passed,
            face_ratio=face_ratio,
        )

        return ValidationMetric(
            type=ValidationType.FACE_SIZE,
            passed=passed,
            score=score,
            message=message,
        )

    def _compute_face_ratio(
        self,
        image: np.ndarray,
        face: Face,
    ) -> float:
        """Compute the ratio of face area to image area.

        Args:
            image: Image data used to determine image dimensions.
            face: Detected face with bounding box.

        Returns:
            Face area ratio as a float between 0.0 and 1.0.

        Raises:
            ValueError: If bounding box has non-positive dimensions.
        """
        image_height, image_width = image.shape[:2]
        image_area = float(
            image_width * image_height
        )

        x1, y1, x2, y2 = face.bbox
        face_width = float(x2 - x1)
        face_height = float(y2 - y1)

        if face_width <= 0 or face_height <= 0:
            raise ValueError(
                "Face bounding box must have positive dimensions."
            )

        face_area = face_width * face_height

        return face_area / image_area

    def _normalize_face_ratio(
        self,
        face_ratio: float,
    ) -> float:
        """Normalize face ratio into a bounded quality score.

        The score equals 1.0 at the ideal face ratio and decreases
        linearly to 0.5 at both boundaries, continuing toward 0.0 for
        extreme face sizes.

        Args:
            face_ratio: Ratio of face area to image area.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        ideal_ratio = FACE_SIZE_IDEAL_RATIO
        half_range = (
            FACE_SIZE_MAX_RATIO - FACE_SIZE_MIN_RATIO
        ) / 2.0

        distance = abs(face_ratio - ideal_ratio)
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
        face_ratio: float,
    ) -> str:
        """Build a human-readable message for the validation result.

        Args:
            passed: Whether the face size check passed.
            face_ratio: Computed face area ratio.

        Returns:
            A descriptive message string.
        """
        if passed:
            return "Face size is acceptable."

        if face_ratio < FACE_SIZE_MIN_RATIO:
            return "Face is too small."

        return "Face is too large."
