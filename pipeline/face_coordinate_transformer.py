"""
Face coordinate transformation module.

Responsible exclusively for translating InsightFace Face coordinates
(bounding boxes and facial landmarks) from the original image coordinate
system into the cropped image coordinate system.
"""

from __future__ import annotations

import logging

import numpy as np
from insightface.app.common import Face

from exceptions.face_exceptions import FaceCoordinateTransformationError

logger = logging.getLogger(__name__)


class FaceCoordinateTransformer:
    """
    Transforms InsightFace Face coordinates into cropped image coordinates.

    Operates purely on spatial coordinates (bbox and kps) by subtracting
    the top-left crop offset (crop_x, crop_y). It leaves all other face
    attributes and image data entirely untouched.
    """

    def transform(
        self,
        face: Face,
        crop_x: int,
        crop_y: int,
    ) -> Face:
        """
        Translate a face's bounding box and landmarks into cropped coordinates.

        Args:
            face:
                Original InsightFace Face object in original coordinate space.
            crop_x:
                Top-left X coordinate of the crop relative to original image.
            crop_y:
                Top-left Y coordinate of the crop relative to original image.

        Returns:
            A new, independent Face object with translated bbox and kps.

        Raises:
            FaceCoordinateTransformationError:
                If face validation fails or transformation encounters an error.
        """
        try:
            self._validate_face(face)

            # Create a shallow copy wrapping the underlying dictionary to preserve
            # all original attributes without mutating the original Face instance.
            transformed_face = Face(dict(face))

            transformed_face.bbox = self._translate_bbox(face.bbox, crop_x, crop_y)
            transformed_face.kps = self._translate_landmarks(face.kps, crop_x, crop_y)

            logger.info("Face coordinates transformed successfully.")
            return transformed_face

        except FaceCoordinateTransformationError:
            raise

        except Exception as error:
            logger.exception("Failed to transform face coordinates.")
            raise FaceCoordinateTransformationError(
                "Failed to transform face coordinates."
            ) from error

    def _validate_face(self, face: Face | None) -> None:
        """
        Validate that the input face and its spatial attributes are well-formed.

        Raises:
            FaceCoordinateTransformationError:
                If validation fails.
        """
        if face is None:
            logger.error("Face cannot be None.")
            raise FaceCoordinateTransformationError("Face cannot be None.")

        bbox = getattr(face, "bbox", None)
        if bbox is None:
            logger.error("Face bounding box is missing.")
            raise FaceCoordinateTransformationError("Face bounding box is missing.")

        if not isinstance(bbox, np.ndarray):
            logger.error("Face bounding box must be a NumPy ndarray.")
            raise FaceCoordinateTransformationError("Face bounding box must be a NumPy ndarray.")

        if bbox.shape != (4,):
            logger.error("Face bounding box must have shape (4,).")
            raise FaceCoordinateTransformationError("Face bounding box must have shape (4,).")

        kps = getattr(face, "kps", None)
        if kps is None:
            logger.error("Face landmarks (kps) are missing.")
            raise FaceCoordinateTransformationError("Face landmarks (kps) are missing.")

        if not isinstance(kps, np.ndarray):
            logger.error("Face landmarks (kps) must be a NumPy ndarray.")
            raise FaceCoordinateTransformationError("Face landmarks (kps) must be a NumPy ndarray.")

        if kps.shape != (5, 2):
            logger.error("Face landmarks (kps) must have shape (5, 2).")
            raise FaceCoordinateTransformationError("Face landmarks (kps) must have shape (5, 2).")

    def _translate_bbox(
        self,
        bbox: np.ndarray,
        crop_x: int,
        crop_y: int,
    ) -> np.ndarray:
        """
        Translate bounding box coordinates by subtracting crop origin offset.
        """
        translated = bbox.copy()
        translated[0] -= crop_x
        translated[1] -= crop_y
        translated[2] -= crop_x
        translated[3] -= crop_y
        return translated

    def _translate_landmarks(
        self,
        kps: np.ndarray,
        crop_x: int,
        crop_y: int,
    ) -> np.ndarray:
        """
        Translate facial landmark coordinates by subtracting crop origin offset.
        """
        translated = kps.copy()
        translated[:, 0] -= crop_x
        translated[:, 1] -= crop_y
        return translated
