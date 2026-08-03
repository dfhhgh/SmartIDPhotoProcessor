"""
Face cropping module.

This module is responsible for cropping the selected face
from the original image while preserving suitable margins
around the face.
"""

from __future__ import annotations

import logging
from utils.bbox import validate_bbox

import numpy as np
from insightface.app.common import Face

from config.constants import (
    TOP_PADDING_RATIO,
    BOTTOM_PADDING_RATIO,
    SIDE_PADDING_RATIO,
    TARGET_CROP_ASPECT_RATIO,
)
from exceptions.face_exceptions import FaceCroppingError
from models.crop_result import CropResult

logger = logging.getLogger(__name__)


class FaceCropper:
    """
    Crop the selected face from an image.

    The cropped region includes configurable padding around
    the detected face to produce an ID-photo style crop.
    """

    def crop(
        self,
        image: np.ndarray,
        face: Face,
    ) -> CropResult:
        """
        Crop the selected face from the original image.

        Args:
            image:
                Original image.

            face:
                Selected InsightFace face.

        Returns:
            A CropResult containing the cropped face image and crop origin coordinates.

        Raises:
            FaceCroppingError:
                If the crop cannot be produced.
        """

        try:
            self._validate_image(image)

            x1, y1, x2, y2 = validate_bbox(face.bbox)

            x1, y1, x2, y2 = self._expand_bbox(
                x1,
                y1,
                x2,
                y2,
                image.shape,
            )

            x1, y1, x2, y2 = self._clamp_bbox(
                x1,
                y1,
                x2,
                y2,
                image.shape,
            )

            cropped_image = self._crop_image(
                image,
                x1,
                y1,
                x2,
                y2,
            )

            logger.info(
                "Face cropped successfully."
            )

            return CropResult(
                image=cropped_image,
                crop_x=x1,
                crop_y=y1,
            )

        except FaceCroppingError:
            raise

        except Exception as error:
            logger.exception(
                "Failed to crop face."
            )

            raise FaceCroppingError(
                "Failed to crop face."
            ) from error

    def _clamp_bbox(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        image_shape: tuple[int, ...],
    ) -> tuple[int, int, int, int]:
        """
        Clamp the bounding box to the image boundaries.

        Args:
            x1:
                Left coordinate.

            y1:
                Top coordinate.

            x2:
                Right coordinate.

            y2:
                Bottom coordinate.

            image_shape:
                Shape of the image.

        Returns:
            A tuple containing:
            (x1, y1, x2, y2)
        """

        image_height, image_width = image_shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(image_width, x2)
        y2 = min(image_height, y2)

        return (
            x1,
            y1,
            x2,
            y2,
        )

    def _crop_image(
        self,
        image: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> np.ndarray:
        """
        Crop the image using the provided bounding box.

        Args:
            image:
                Original image.

            x1:
                Left coordinate.

            y1:
                Top coordinate.

            x2:
                Right coordinate.

            y2:
                Bottom coordinate.

        Returns:
            The cropped face image.
        """

        return image[y1:y2, x1:x2]

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
            FaceCroppingError:
                If the input image is invalid.
        """

        if image is None:
            logger.error(
                "Input image cannot be None."
            )

            raise FaceCroppingError(
                "Input image cannot be None."
            )

        if not isinstance(image, np.ndarray):
            logger.exception(
                "Input image must be a NumPy ndarray."
            )

            raise FaceCroppingError(
                "Input image must be a NumPy ndarray."
            )

        if image.size == 0:
            logger.exception(
                "Input image cannot be empty."
            )

            raise FaceCroppingError(
                "Input image cannot be empty."
            )

        if image.ndim not in (2, 3):
            logger.exception(
                "Input image must have 2 or 3 dimensions."
            )

            raise FaceCroppingError(
                "Input image must have 2 or 3 dimensions."
            )

    def _expand_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        image_shape: tuple[int, ...] | None = None,
    ) -> tuple[int, int, int, int]:
        """
        Expand the face bounding box using configurable padding ratios and
        the target crop aspect ratio.
        """

        face_width = x2 - x1
        face_height = y2 - y1

        horizontal_padding = face_width * SIDE_PADDING_RATIO
        top_padding = face_height * TOP_PADDING_RATIO
        bottom_padding = face_height * BOTTOM_PADDING_RATIO

        new_x1 = x1 - horizontal_padding
        new_y1 = y1 - top_padding
        new_x2 = x2 + horizontal_padding
        new_y2 = y2 + bottom_padding

        expanded_width = new_x2 - new_x1
        expanded_height = new_y2 - new_y1
        current_ratio = expanded_width / expanded_height if expanded_height > 0 else float("inf")

        if current_ratio < TARGET_CROP_ASPECT_RATIO:
            target_width = max(expanded_width, int(np.ceil(expanded_height * TARGET_CROP_ASPECT_RATIO)))
            delta = target_width - expanded_width
            if delta > 0:
                ideal_left = delta // 2
                ideal_right = delta - ideal_left
                if image_shape is not None and len(image_shape) >= 2:
                    left_capacity = max(0, int(np.floor(new_x1)))
                    right_capacity = max(0, int(np.floor(image_shape[1] - new_x2)))
                    left_expand = min(ideal_left, left_capacity)
                    right_expand = min(ideal_right, right_capacity)
                    remaining = delta - (left_expand + right_expand)
                    if remaining > 0:
                        if right_capacity > right_expand:
                            right_expand += min(remaining, right_capacity - right_expand)
                        elif left_capacity > left_expand:
                            left_expand += min(remaining, left_capacity - left_expand)
                    new_x1 -= left_expand
                    new_x2 += right_expand
                else:
                    new_x1 -= ideal_left
                    new_x2 += ideal_right
        elif current_ratio > TARGET_CROP_ASPECT_RATIO:
            target_height = max(expanded_height, int(np.ceil(expanded_width / TARGET_CROP_ASPECT_RATIO)))
            delta = target_height - expanded_height
            if delta > 0:
                ideal_top = delta // 2
                ideal_bottom = delta - ideal_top
                if image_shape is not None and len(image_shape) >= 2:
                    top_capacity = max(0, int(np.floor(new_y1)))
                    bottom_capacity = max(0, int(np.floor(image_shape[0] - new_y2)))
                    top_expand = min(ideal_top, top_capacity)
                    bottom_expand = min(ideal_bottom, bottom_capacity)
                    remaining = delta - (top_expand + bottom_expand)
                    if remaining > 0:
                        if bottom_capacity > bottom_expand:
                            bottom_expand += min(remaining, bottom_capacity - bottom_expand)
                        elif top_capacity > top_expand:
                            top_expand += min(remaining, top_capacity - top_expand)
                    new_y1 -= top_expand
                    new_y2 += bottom_expand
                else:
                    new_y1 -= ideal_top
                    new_y2 += ideal_bottom

        return (
            round(new_x1),
            round(new_y1),
            round(new_x2),
            round(new_y2),
        )
