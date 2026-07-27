"""
Validator factory.

Constructs the default validator pipeline without orchestrating execution.
The returned tuple is consumed by the future ValidationOrchestrator.
"""

from validators.base_validator import BaseValidator
from validators.blur_validator import BlurValidator
from validators.brightness_validator import BrightnessValidator
from validators.contrast_validator import ContrastValidator
from validators.face_size_validator import FaceSizeValidator
from validators.head_pose_validator import HeadPoseValidator
from validators.face_visibility_validator import FaceVisibilityValidator
from validators.occlusion_validator import OcclusionValidator


def create_default_validators() -> tuple[BaseValidator, ...]:
    """Create the default validator pipeline.

    Returns:
        An immutable tuple containing one instance of every built-in
        validator, ordered so that image-level checks run before
        face-level checks, and face-level checks run before
        parsing-dependent checks.
    """
    return (
        BlurValidator(),
        BrightnessValidator(),
        ContrastValidator(),
        FaceSizeValidator(),
        HeadPoseValidator(),
        FaceVisibilityValidator(),
        OcclusionValidator(),
    )
