"""
Eyewear type domain model.
"""

from enum import Enum


class EyewearType(Enum):
    """
    Categories of eyewear that may be detected on a subject's face.

    This enum is a pure domain model representing the eyewear categories
    required for validation policy decisions. It contains no business logic,
    conversion routines, or validation rules, and remains strictly independent
    from any machine learning framework or classification model.
    """

    NONE = "none"
    CLEAR_GLASSES = "clear_glasses"
    PRESCRIPTION_GLASSES = "prescription_glasses"
    SUNGLASSES = "sunglasses"
