"""
Immutable semantic wrapper around a BiSeNet face-parsing segmentation mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Collection, Mapping

import numpy as np
import numpy.typing as npt

from models.parsing.face_part import FacePart


@dataclass(frozen=True, slots=True)
class FaceParsingResult:
    """
    Semantic representation of a face-parsing segmentation mask.

    Hides all raw NumPy mask operations behind a typed query API.
    All pixel counts are eagerly cached during construction so that
    repeated queries are O(1).
    """

    mask: np.ndarray
    """2-D integer segmentation mask whose values correspond to FacePart IDs."""

    image_height: int
    """Height (rows) of the source image in pixels."""

    image_width: int
    """Width (columns) of the source image in pixels."""

    _part_pixel_counts: Mapping[FacePart, int] = field(
        default=MappingProxyType({}),
        init=False,
        repr=False,
    )
    """Eagerly computed per-class pixel counts (internal cache)."""

    def __post_init__(self) -> None:
        """Validate inputs and eagerly compute the pixel-count cache."""
        self._validate_mask()
        self._validate_dimensions()
        self._validate_shape_consistency()

        object.__setattr__(
            self,
            "_part_pixel_counts",
            self._compute_part_pixel_counts(),
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _validate_mask(self) -> None:
        """Ensure the mask is a valid 2-D integer array."""
        if not isinstance(self.mask, np.ndarray):
            raise TypeError(
                "Mask must be a numpy.ndarray."
            )

        if self.mask.ndim != 2:
            raise ValueError(
                f"Mask must be 2-D, got {self.mask.ndim}-D."
            )

        if not np.issubdtype(self.mask.dtype, np.integer):
            raise TypeError(
                f"Mask dtype must be an integer type, got {self.mask.dtype}."
            )

    def _validate_dimensions(self) -> None:
        """Ensure image dimensions are positive integers."""
        if not isinstance(self.image_height, int) or self.image_height <= 0:
            raise ValueError(
                f"Image height must be a positive integer, got {self.image_height!r}."
            )

        if not isinstance(self.image_width, int) or self.image_width <= 0:
            raise ValueError(
                f"Image width must be a positive integer, got {self.image_width!r}."
            )

    def _validate_shape_consistency(self) -> None:
        """Ensure mask shape matches the declared image dimensions."""
        expected = (self.image_height, self.image_width)

        if self.mask.shape != expected:
            raise ValueError(
                f"Mask shape {self.mask.shape} does not match "
                f"declared dimensions {expected}."
            )

    def _compute_part_pixel_counts(self) -> MappingProxyType[FacePart, int]:
        """
        Single-pass pixel-count computation for every FacePart via
        np.bincount.

        Returns an immutable MappingProxyType so the frozen dataclass
        cannot be mutated through the cache.
        """
        raw_counts = np.bincount(
            self.mask.ravel(),
            minlength=len(FacePart),
        )

        counts = {
            part: int(raw_counts[part])
            for part in FacePart
        }

        return MappingProxyType(counts)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_part(self, part: FacePart) -> bool:
        """
        Return True if the mask contains at least one pixel of *part*.
        """
        if not isinstance(part, FacePart):
            raise TypeError(
                f"part must be a FacePart, got {type(part).__name__}."
            )

        return self._part_pixel_counts.get(part, 0) > 0

    def contains_any(self, parts: Collection[FacePart]) -> bool:
        """
        Return True if the mask contains at least one pixel of
        any part in *parts*.
        """
        self._validate_parts(parts)

        return any(
            self._part_pixel_counts.get(part, 0) > 0
            for part in parts
        )

    def contains_all(self, parts: Collection[FacePart]) -> bool:
        """
        Return True if the mask contains at least one pixel of
        every part in *parts*.
        """
        self._validate_parts(parts)

        return all(
            self._part_pixel_counts.get(part, 0) > 0
            for part in parts
        )

    def part_mask(self, part: FacePart) -> npt.NDArray[np.bool_]:
        """
        Return a newly computed boolean mask where True marks pixels
        belonging to *part*.  The result is NOT cached.
        """
        if not isinstance(part, FacePart):
            raise TypeError(
                f"part must be a FacePart, got {type(part).__name__}."
            )

        return self.mask == part

    def part_area(self, part: FacePart) -> int:
        """
        Return the cached pixel count for *part*.
        """
        if not isinstance(part, FacePart):
            raise TypeError(
                f"part must be a FacePart, got {type(part).__name__}."
            )

        return self._part_pixel_counts.get(part, 0)

    def part_ratio(self, part: FacePart) -> float:
        """
        Return the fraction of total image pixels occupied by *part*.

        Result = part_area(part) / total_pixels()
        """
        if not isinstance(part, FacePart):
            raise TypeError(
                f"part must be a FacePart, got {type(part).__name__}."
            )

        total = self.total_pixels()

        if total == 0:
            return 0.0

        return self._part_pixel_counts.get(part, 0) / total

    # ------------------------------------------------------------------
    # Composite region queries
    # ------------------------------------------------------------------

    def has_visible_mouth_region(
        self,
        mouth_min_ratio: float = 0.0,
        upper_lip_min_ratio: float = 0.0,
        lower_lip_min_ratio: float = 0.0,
    ) -> bool:
        """Return True when the semantic mouth region is sufficiently visible.

        The mouth region is considered visible when either:

        *Case 1* — The MOUTH class itself is sufficiently visible
        (part_ratio >= *mouth_min_ratio*).

        *Case 2* — Both UPPER_LIP and LOWER_LIP are individually
        sufficiently visible (part_ratio >= their respective thresholds).
        This covers closed-mouth poses where BiSeNet correctly predicts
        the lips but not the inner oral cavity.

        Args:
            mouth_min_ratio: Minimum acceptable part_ratio for MOUTH.
            upper_lip_min_ratio: Minimum acceptable part_ratio for UPPER_LIP.
            lower_lip_min_ratio: Minimum acceptable part_ratio for LOWER_LIP.

        Returns:
            True when the mouth region passes either case.
        """
        # Case 1: MOUTH itself is sufficiently visible.
        if (
            self.has_part(FacePart.MOUTH)
            and self.part_ratio(FacePart.MOUTH) >= mouth_min_ratio
        ):
            return True

        # Case 2: Both lips are sufficiently visible.
        if (
            self.has_part(FacePart.UPPER_LIP)
            and self.has_part(FacePart.LOWER_LIP)
            and self.part_ratio(FacePart.UPPER_LIP) >= upper_lip_min_ratio
            and self.part_ratio(FacePart.LOWER_LIP) >= lower_lip_min_ratio
        ):
            return True

        return False

    def total_pixels(self) -> int:
        """Return the total number of pixels in the mask."""
        return self.mask.size

    def image_size(self) -> tuple[int, int]:
        """Return (height, width) of the source image."""
        return (self.image_height, self.image_width)

    def mask_shape(self) -> tuple[int, ...]:
        """Return the shape of the segmentation mask."""
        return self.mask.shape

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_parts(parts: Collection[FacePart]) -> None:
        """Ensure every element in *parts* is a FacePart."""
        for part in parts:
            if not isinstance(part, FacePart):
                raise TypeError(
                    f"Every element must be a FacePart, got {type(part).__name__}."
                )
