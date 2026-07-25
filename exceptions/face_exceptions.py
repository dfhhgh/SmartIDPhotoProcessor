"""
Custom exceptions for the face processing pipeline.
"""


class FacePipelineError(Exception):
    """
    Base exception for all face pipeline errors.
    """


class FaceDetectionError(FacePipelineError):
    """
    Raised when face detection fails.
    """

class FaceSelectionError(FacePipelineError):
    """
    Raised when selecting the primary face fails.
    """

class FaceCroppingError(FacePipelineError):
    """
    Raised when face cropping fails.
    """