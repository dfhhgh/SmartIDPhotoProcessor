"""
Face alignment module.

This module is responsible for aligning the selected face
using facial landmarks and a similarity transform.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from insightface.app.common import Face

from config.constants import ALIGNED_FACE_SIZE, ARCFACE_TEMPLATE
from exceptions.face_exceptions import FaceAlignmentError

logger = logging.getLogger(__name__)


class FaceAligner:
    """
    Align a detected face using facial landmarks.

    The face is aligned using a similarity transform
    computed from the detected facial landmarks.
    """

    def align(
        self,
        image: np.ndarray,
        face: Face,
    ) -> np.ndarray:
        """
        Align the selected face.

        Args:
            image:
                Original image.

            face:
                Selected InsightFace face.

        Returns:
            The aligned face image.

        Raises:
            FaceAlignmentError:
                If face alignment fails.
        """
        try:
            self._validate_image(image)

            source = self._get_source_landmarks(face)
            target = self._get_target_landmarks()

            transform = self._compute_similarity_transform(
                source,
                target,
            )

            aligned_image = self._apply_transform(
                image,
                transform,
            )

            logger.info(
                "Face aligned successfully."
            )

            return aligned_image

        except FaceAlignmentError:
            raise

        except Exception as error:
            logger.exception(
                "Failed to align face."
            )

            raise FaceAlignmentError(
                "Failed to align face."
            ) from error

    def _validate_image(
        self,
        image: np.ndarray,
    ) -> None:
        """
        Validate the input image.

        Args:
            image:
                Original image.

        Raises:
            FaceAlignmentError:
                If the input image is invalid.
        """

        if image is None:
            logger.error(
                "Input image cannot be None."
            )

            raise FaceAlignmentError(
                "Input image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            logger.error(
                "Input image must be a NumPy ndarray."
            )

            raise FaceAlignmentError(
                "Input image must be a NumPy ndarray."
            )

        if image.size == 0:
            logger.error(
                "Input image cannot be empty."
            )

            raise FaceAlignmentError(
                "Input image cannot be empty."
            )

        if image.ndim not in (2, 3):
            logger.error(
                "Input image must have 2 or 3 dimensions."
            )

            raise FaceAlignmentError(
                "Input image must have 2 or 3 dimensions."
            )

    def _validate_landmarks(
        self,
        face: Face,
    ) -> None:
        """
        Validate the detected facial landmarks.

        Args:
            face:
                Selected InsightFace face.

        Raises:
            FaceAlignmentError:
                If the facial landmarks are invalid.
        """

        landmarks = face.kps

        if landmarks is None:
            logger.error(
                "Face landmarks cannot be None."
            )

            raise FaceAlignmentError(
                "Face landmarks cannot be None."
            )

        if not isinstance(landmarks, np.ndarray):
            logger.error(
                "Facial landmarks must be a NumPy ndarray."
            )

            raise FaceAlignmentError(
                "Facial landmarks must be a NumPy ndarray."
            )

        if landmarks.shape != (5, 2):
            logger.error(
                "Facial landmarks must have shape (5, 2)."
            )

            raise FaceAlignmentError(
                "Facial landmarks must have shape (5, 2)."
            )

        if np.isnan(landmarks).any():
            logger.error(
                "Facial landmarks cannot contain NaN values."
            )

            raise FaceAlignmentError(
                "Facial landmarks cannot contain NaN values."
            )

        if np.isinf(landmarks).any():
            logger.error(
                "Facial landmarks cannot contain infinite values."
            )

            raise FaceAlignmentError(
                "Facial landmarks cannot contain infinite values."
            )

    def _get_source_landmarks(
        self,
        face: Face,
    ) -> np.ndarray:
        """
        Return the detected facial landmarks.

        Args:
            face:
                Selected InsightFace face.

        Returns:
            Detected facial landmarks as a float32 array.

        Raises:
            FaceAlignmentError:
                If the facial landmarks are invalid.
        """
        self._validate_landmarks(face)
        landmarks = face.kps.astype(np.float32)
        return landmarks

    def _get_target_landmarks(
        self,
    ) -> np.ndarray:
        """
        Return the target landmark template.

        Returns:
            Target facial landmark template as a float32 array.
        """
        return ARCFACE_TEMPLATE.copy()

    def _compute_similarity_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the similarity transformation matrix.

        Args:
            source:
                Source facial landmarks.

            target:
                Target facial landmarks.

        Returns:
            A 2x3 affine transformation matrix.

        Raises:
            FaceAlignmentError:
                If the transform cannot be computed.
        """
        if not isinstance(source, np.ndarray):
            logger.error(
                "Source landmarks must be a NumPy ndarray."
            )

            raise FaceAlignmentError(
                "Source landmarks must be a NumPy ndarray."
            )

        if not isinstance(target, np.ndarray):
            logger.error(
                "Target landmarks must be a NumPy ndarray."
            )

            raise FaceAlignmentError(
                "Target landmarks must be a NumPy ndarray."
            )

        if source.shape != (5, 2):
            logger.error(
                "Source landmarks must have shape (5, 2)."
            )

            raise FaceAlignmentError(
                "Source landmarks must have shape (5, 2)."
            )

        if target.shape != (5, 2):
            logger.error(
                "Target landmarks must have shape (5, 2)."
            )

            raise FaceAlignmentError(
                "Target landmarks must have shape (5, 2)."
            )

        transform, _ = cv2.estimateAffinePartial2D(
            source,
            target,
        )

        if transform is None:
            logger.error(
                "Similarity transform could not be computed."
            )

            raise FaceAlignmentError(
                "Similarity transform could not be computed."
            )

        if transform.shape != (2, 3):
            logger.error(
                "Similarity transform must have shape (2, 3)."
            )

            raise FaceAlignmentError(
                "Similarity transform must have shape (2, 3)."
            )

        return transform.astype(np.float32)

    def _apply_transform(
        self,
        image: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """
        Apply the similarity transform to the image.

        Args:
            image:
                Original image.

            transform:
                A 2x3 affine transformation matrix.

        Returns:
            The aligned face image.

        Raises:
            FaceAlignmentError:
                If the transform cannot be applied.
        """
        if not isinstance(transform, np.ndarray):
            logger.error(
                "Similarity transform must be a NumPy ndarray."
            )

            raise FaceAlignmentError(
                "Similarity transform must be a NumPy ndarray."
            )

        if transform.shape != (2, 3):
            logger.error(
                "Similarity transform must have shape (2, 3)."
            )

            raise FaceAlignmentError(
                "Similarity transform must have shape (2, 3)."
            )

        try:
            aligned_image = cv2.warpAffine(
                image,
                transform,
                ALIGNED_FACE_SIZE,
            )

            if aligned_image is None:
                logger.error(
                    "Aligned image cannot be None."
                )

                raise FaceAlignmentError(
                    "Aligned image cannot be None."
                )

            if not isinstance(aligned_image, np.ndarray):
                logger.error(
                    "Aligned image must be a NumPy ndarray."
                )

                raise FaceAlignmentError(
                    "Aligned image must be a NumPy ndarray."
                )

            if aligned_image.size == 0:
                logger.error(
                    "Aligned image cannot be empty."
                )

                raise FaceAlignmentError(
                    "Aligned image cannot be empty."
                )

            return aligned_image

        except cv2.error as exc:
            logger.exception(
                "Similarity transform could not be applied."
            )

            raise FaceAlignmentError(
                "Similarity transform could not be applied."
            ) from exc
