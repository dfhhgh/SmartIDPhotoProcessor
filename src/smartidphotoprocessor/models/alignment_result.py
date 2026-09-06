"""
Alignment result domain model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from insightface.app.common import Face


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    """
    Represents the complete output of the face alignment stage.

    The aligned image and aligned face always belong to the same aligned
    coordinate system. The affine transform records how cropped-image
    coordinates were mapped into aligned-image coordinates.
    """

    aligned_image: np.ndarray
    aligned_face: Face
    transform: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.aligned_image, np.ndarray):
            raise TypeError("Aligned image must be a numpy ndarray.")

        if self.aligned_image.size == 0:
            raise ValueError("Aligned image must not be empty.")

        if self.aligned_image.ndim != 3:
            raise ValueError("Aligned image must be a 3-dimensional image.")

        if not isinstance(self.aligned_face, Face):
            raise TypeError("Aligned face must be a Face instance.")

        if not isinstance(self.transform, np.ndarray):
            raise TypeError("Transform must be a numpy ndarray.")

        if self.transform.shape != (2, 3):
            raise ValueError("Transform must have shape (2, 3).")
