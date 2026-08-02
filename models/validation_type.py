"""
Validation rule types.
"""

from enum import StrEnum


class ValidationType(StrEnum):
    """
    Supported validation rule types.
    """

    BLUR = "blur"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    FACE_SIZE = "face_size"
    HEAD_POSE = "head_pose"
    FACE_VISIBILITY = "face_visibility"
    OCCLUSION = "occlusion"
    GLASSES = "glasses"
    FACE_AMBIGUITY = "face_ambiguity"
