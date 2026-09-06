"""
Semantic face-part labels produced by the BiSeNet face-parsing model
trained on the CelebAMask-HQ dataset (19 classes).
"""

from __future__ import annotations

from enum import IntEnum
from typing import FrozenSet


class FacePart(IntEnum):
    """
    Every semantic class in the CelebAMask-HQ face-parsing label map.

    The integer values are the exact pixel labels stored in the dataset
    annotation masks and predicted by BiSeNet at inference time.
    """

    BACKGROUND = 0
    """Non-face pixels / image background."""

    SKIN = 1
    """General facial skin region."""

    LEFT_BROW = 2
    """Left eyebrow (subject's left, viewer's right)."""

    RIGHT_BROW = 3
    """Right eyebrow (subject's right, viewer's left)."""

    LEFT_EYE = 4
    """Left eye including sclera, iris, and pupil."""

    RIGHT_EYE = 5
    """Right eye including sclera, iris, and pupil."""

    EYE_GLASS = 6
    """Eyeglasses / sunglasses frames and lenses."""

    LEFT_EAR = 7
    """Left ear."""

    RIGHT_EAR = 8
    """Right ear."""

    EAR_RING = 9
    """Earrings and other ear accessories."""

    NOSE = 10
    """Nose including bridge, tip, and nostrils."""

    MOUTH = 11
    """Inner mouth / oral cavity (visible teeth and tongue)."""

    UPPER_LIP = 12
    """Upper lip vermilion border and surface."""

    LOWER_LIP = 13
    """Lower lip vermilion border and surface."""

    NECK = 14
    """Neck region."""

    NECKLACE = 15
    """Necklaces, chains, and other neck accessories."""

    CLOTH = 16
    """Clothing visible in the crop."""

    HAIR = 17
    """Hair on the head."""

    HAT = 18
    """Hats, caps, headscarves, and other headwear."""

    # ------------------------------------------------------------------
    # Helper sets for common grouping tasks
    # ------------------------------------------------------------------

    @classmethod
    def facial_region(cls) -> FrozenSet[FacePart]:
        """Parts that constitute the face proper (excluding accessories/background)."""
        return frozenset({
            cls.SKIN,
            cls.LEFT_BROW,
            cls.RIGHT_BROW,
            cls.LEFT_EYE,
            cls.RIGHT_EYE,
            cls.NOSE,
            cls.MOUTH,
            cls.UPPER_LIP,
            cls.LOWER_LIP,
        })

    @classmethod
    def brows(cls) -> FrozenSet[FacePart]:
        """Eyebrow regions."""
        return frozenset({
            cls.LEFT_BROW,
            cls.RIGHT_BROW,
        })

    @classmethod
    def eyes(cls) -> FrozenSet[FacePart]:
        """Anatomical eye regions (sclera, iris, pupil)."""
        return frozenset({
            cls.LEFT_EYE,
            cls.RIGHT_EYE,
        })

    @classmethod
    def eye_related(cls) -> FrozenSet[FacePart]:
        """Eyes together with eyeglass accessories."""
        return frozenset({
            cls.LEFT_EYE,
            cls.RIGHT_EYE,
            cls.EYE_GLASS,
        })

    @classmethod
    def nose_region(cls) -> FrozenSet[FacePart]:
        """Nose region."""
        return frozenset({
            cls.NOSE,
        })

    @classmethod
    def mouth_region(cls) -> FrozenSet[FacePart]:
        """Mouth and lip regions."""
        return frozenset({
            cls.MOUTH,
            cls.UPPER_LIP,
            cls.LOWER_LIP,
        })

    @classmethod
    def lips(cls) -> FrozenSet[FacePart]:
        """Lip regions."""
        return frozenset({
            cls.UPPER_LIP,
            cls.LOWER_LIP,
        })

    @classmethod
    def ears(cls) -> FrozenSet[FacePart]:
        """Ear regions and ear accessories."""
        return frozenset({
            cls.LEFT_EAR,
            cls.RIGHT_EAR,
            cls.EAR_RING,
        })

    @classmethod
    def accessories(cls) -> FrozenSet[FacePart]:
        """Non-anatomical accessories (glasses, jewellery, headwear)."""
        return frozenset({
            cls.EYE_GLASS,
            cls.EAR_RING,
            cls.NECKLACE,
            cls.HAT,
        })

    @classmethod
    def non_face(cls) -> FrozenSet[FacePart]:
        """Background and clothing — not part of the face or head."""
        return frozenset({
            cls.BACKGROUND,
            cls.CLOTH,
        })

    @property
    def is_anatomical(self: FacePart) -> bool:
        """``True`` if this part is a real anatomical region (not background or accessory)."""
        return (
            self in self.facial_region()
            or self in (self.LEFT_EAR, self.RIGHT_EAR, self.NECK, self.HAIR)
        )
