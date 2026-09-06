"""
Crop result domain model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True,slots=True,)
class CropResult:
    """
    Represents the output of the face cropping process, containing
    the cropped image array and the final top-left crop origin coordinates.
    """

    image: np.ndarray

    crop_x: int

    crop_y: int
