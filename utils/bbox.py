"""
Bounding box utility functions.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from exceptions.face_exceptions import FacePipelineError

logger = logging.getLogger(__name__)


from collections.abc import Sequence

def validate_bbox(
    bbox: Sequence[float] | None,
) -> tuple[float, float, float, float]:
    """
    Validate and return bounding box coordinates.

    Args:
        bbox:
            Bounding box coordinates in the format:
            (x1, y1, x2, y2)

    Returns:
        A tuple containing:
        (x1, y1, x2, y2)

    Raises:
        FacePipelineError:
            If the bounding box is invalid.
    """

    if bbox is None:
        logger.error(
            "Face bounding box is missing."
        )

        raise FacePipelineError(
            "Face bounding box is missing."
        )

    if len(bbox) != 4:
        logger.error(
            "Face bounding box must contain exactly four coordinates."
        )

        raise FacePipelineError(
            "Face bounding box must contain exactly four coordinates."
        )

    x1, y1, x2, y2 = bbox

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        logger.error(
            "Face bounding box has invalid dimensions."
        )

        raise FacePipelineError(
            "Face bounding box has invalid dimensions."
        )

    return x1, y1, x2, y2