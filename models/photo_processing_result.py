"""
Photo processing result model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from insightface.app.common import Face

from models.validation_result import ValidationResult


@dataclass(slots=True)
class PhotoProcessingResult:
    """
    Represents the final output produced after processing one photo.

    This is a pure domain model that encapsulates the aggregated validation
    decision, selected primary face, aligned face image, and optional
    final cropped ID photo.
    """

    validation_result: ValidationResult
    selected_face: Face
    aligned_image: np.ndarray
    cropped_image: np.ndarray | None

    def __post_init__(self) -> None:
        """
        Validate the photo processing result fields.
        """
        if not isinstance(
            self.validation_result,
            ValidationResult,
        ):
            raise TypeError(
                "Validation result must be a ValidationResult instance."
            )

        if self.selected_face is None:
            raise ValueError(
                "Selected face cannot be None."
            )

        if not isinstance(
            self.selected_face,
            Face,
        ):
            raise TypeError(
                "Selected face must be a Face instance."
            )

        if not isinstance(
            self.aligned_image,
            np.ndarray,
        ):
            raise TypeError(
                "Aligned image must be a numpy ndarray."
            )

        if self.aligned_image.size == 0:
            raise ValueError(
                "Aligned image must not be empty."
            )

        if self.aligned_image.ndim != 3:
            raise ValueError(
                "Aligned image must be a 3-dimensional image."
            )

        if self.cropped_image is not None:
            if not isinstance(
                self.cropped_image,
                np.ndarray,
            ):
                raise TypeError(
                    "Cropped image must be a numpy ndarray or None."
                )

            if self.cropped_image.size == 0:
                raise ValueError(
                    "Cropped image must not be empty."
                )

            if self.cropped_image.ndim != 3:
                raise ValueError(
                    "Cropped image must be a 3-dimensional image."
                )
