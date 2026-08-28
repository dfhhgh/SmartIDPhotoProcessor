import cv2
import numpy as np

# -------------------------------
# Primary Face Selection
# -------------------------------

MIN_FACE_AREA_RATIO = 0.10

AREA_WEIGHT = 0.40
CENTER_WEIGHT = 0.35
QUALITY_WEIGHT = 0.15
POSE_WEIGHT = 0.10


# -------------------------------
# Face Validation
# -------------------------------

MAX_VALID_FACES = 1

MIN_SECONDARY_FACE_RATIO = 0.20


# -------------------------------
# Face Selection Ambiguity & Reliability
# -------------------------------

# Initial conservative default values for a university student ID system.
# These values are initial defaults that should later be calibrated on a
# representative student-ID dataset.
FACE_SELECTION_AMBIGUITY_MAX_RATIO = 0.80
FACE_SELECTION_MIN_PRIMARY_SCORE = 0.25
FACE_SELECTION_MIN_COMPETITIVE_SCORE = 0.20
FACE_SELECTION_MIN_SCORE_MARGIN = 0.05


# -------------------------------
# Face Size Validation
# -------------------------------

# Initial conservative default values for a university student ID system.
# These thresholds define the acceptable range for the face area ratio
# (face_area / image_area). The ideal ratio equals the midpoint of the
# acceptable range, which is used to compute the quality score.
# They are not scientifically fixed and should be calibrated later using
# a representative dataset of real student ID photos.
FACE_SIZE_MIN_RATIO = 0.08
FACE_SIZE_IDEAL_RATIO = 0.40
FACE_SIZE_MAX_RATIO = 0.65
FLOAT_COMPARISON_EPSILON = 1e-6

# -------------------------------
# Blur Validation
# -------------------------------

# Initial conservative default values for a university student ID system.
# These values are designed to reject only obviously blurry images, not to
# require professional-quality photos. They are not scientifically fixed and
# should be calibrated later using a representative dataset of real student ID
# photos.
BLUR_THRESHOLD = 60.0
BLUR_MAX_EXPECTED_VALUE = 1000.0


# -------------------------------
# Brightness Validation
# -------------------------------

# Conservative default values for a university student ID system.
# These thresholds reject only extremely dark or washed-out photos and
# are not scientifically fixed. They should be calibrated later using a
# representative dataset of real student ID photos.
BRIGHTNESS_MIN_THRESHOLD = 40.0
BRIGHTNESS_MAX_THRESHOLD = 220.0
BRIGHTNESS_MAX_EXPECTED_VALUE = 255.0


# -------------------------------
# Face Alignment
# -------------------------------

FACE_ALIGNMENT_SIZE = (112, 112)
ALIGNED_FACE_SIZE = FACE_ALIGNMENT_SIZE
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


# -------------------------------
# Face Export (Optional)
# -------------------------------

OUTPUT_WIDTH = 600
OUTPUT_HEIGHT = 800
TARGET_CROP_ASPECT_RATIO = OUTPUT_WIDTH / OUTPUT_HEIGHT
EXPORT_INTERPOLATION_METHOD = cv2.INTER_AREA
MIN_EXPORT_WIDTH = 1
MIN_EXPORT_HEIGHT = 1
MAX_EXPORT_WIDTH = 10000
MAX_EXPORT_HEIGHT = 10000
MAX_ALLOWED_UPSCALE = 2.0
SAFE_UPSCALE_FACTOR = 1.5

# -------------------------------
# Face Cropping
# -------------------------------

# Padding ratios for face cropping, calibrated for official student ID /
# passport style photographs.
#
# InsightFace bounding boxes typically span from the eyebrows/forehead to the chin.
# To transform this into a standard ID portrait containing the complete head,
# forehead, hair/head covering, visible ears, neck, and shoulders, asymmetric
# and calibrated padding ratios are applied:
#
# - TOP_PADDING_RATIO (0.45): Provides generous space above the detected
#   bounding box top to fully include the forehead, top of the head, and hair or
#   head covering.
# - BOTTOM_PADDING_RATIO (0.75): Provides substantial padding below the chin
#   to capture the neck and a portion of the shoulders, adhering to passport/ID
#   composition guidelines.
# - SIDE_PADDING_RATIO (0.30): Provides moderate horizontal padding on both
#   sides to encompass ears (when visible) and side hair while avoiding
#   excessive lateral background.
TOP_PADDING_RATIO = 0.45
BOTTOM_PADDING_RATIO = 0.75
SIDE_PADDING_RATIO = 0.30


# Contrast validation thresholds.
#
# Minimum acceptable grayscale standard deviation for student ID photos.
# Images below this threshold are considered low contrast.
CONTRAST_MIN_THRESHOLD = 30.0

# Expected upper bound for contrast normalization.
# Used only to normalize the quality score into the range [0.0, 1.0].
# Values above this threshold are clamped to 1.0.
CONTRAST_MAX_EXPECTED_VALUE = 100.0

from models.parsing.face_part import FacePart

# -------------------------------
# Face Visibility Validation
# -------------------------------

# Mandatory anatomical regions checked via simple has_part() queries.
# Composite semantic regions (mouth, eyebrows) are intentionally excluded.
FACE_VISIBILITY_REQUIRED_PARTS: tuple[FacePart, ...] = (
    FacePart.LEFT_EYE,
    FacePart.RIGHT_EYE,
    FacePart.NOSE,
)

# Number of composite semantic regions checked alongside the individual
# required parts: mouth region (MOUTH / UPPER_LIP / LOWER_LIP) and
# two eyebrow regions (LEFT_BROW / RIGHT_BROW, each with eye fallback).
# Add new composite regions here rather than hardcoding offsets.
FACE_VISIBILITY_COMPOSITE_REGION_COUNT: int = 3

# Minimum acceptable ratio of (part pixel area / total image area) for each
# individual required region (eyes, nose). These are relative to the whole
# frame, not the face bounding box, so they are intentionally small.
# Initial conservative defaults; calibrate later using a representative
# dataset of real student ID photos.
FACE_VISIBILITY_REQUIRED_PART_THRESHOLDS: dict[FacePart, float] = {
    FacePart.LEFT_EYE: 0.0015,
    FacePart.RIGHT_EYE: 0.0015,
    FacePart.NOSE: 0.0050,
}

# Minimum ratios for composite eyebrow regions (brow and eye fallback).
FACE_VISIBILITY_EYEBROW_THRESHOLDS: dict[FacePart, float] = {
    FacePart.LEFT_BROW: 0.0010,
    FacePart.RIGHT_BROW: 0.0010,
    FacePart.LEFT_EYE: 0.0015,
    FacePart.RIGHT_EYE: 0.0015,
}

# Minimum ratios for composite mouth region (MOUTH and lip fallback).
FACE_VISIBILITY_MOUTH_THRESHOLDS: dict[FacePart, float] = {
    FacePart.MOUTH: 0.0008,
    FacePart.UPPER_LIP: 0.0020,
    FacePart.LOWER_LIP: 0.0020,
}

# Combined lookup used by the validator for simple part-ratio checks.
# Merges required-part thresholds with eyebrow eye-fallback thresholds
# so the validator can do a single dict lookup per part.
FACE_VISIBILITY_MIN_PART_RATIOS: dict[FacePart, float] = {
    **FACE_VISIBILITY_REQUIRED_PART_THRESHOLDS,
    **FACE_VISIBILITY_EYEBROW_THRESHOLDS,
    **FACE_VISIBILITY_MOUTH_THRESHOLDS,
}

# Fraction of a region's full scoring weight deducted when the region is
# present but below its minimum visibility ratio (vs. fully missing).
FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR = 0.5

# -------------------------------
# Head Pose Validation
# -------------------------------

# Initial conservative default values for a university student ID system.
# These thresholds define the acceptable absolute deviation, in degrees,
# for each pose axis relative to a perfectly frontal (0.0) pose. They are
# not scientifically fixed and should be calibrated later using a
# representative dataset of real student ID photos.
HEAD_POSE_PITCH_MAX_DEGREES = 20.0
HEAD_POSE_YAW_MAX_DEGREES = 22.0
HEAD_POSE_ROLL_MAX_DEGREES = 20.0

# -------------------------------
# Occlusion Validation
# -------------------------------

# Semantic classes that are never acceptable in a student ID photo.
#
# Notably absent:
#   - FacePart.EYE_GLASS: normal eyeglasses are allowed. The face parser
#     produces EYE_GLASS as a semantic class; OcclusionValidator must not
#     treat eyeglasses as an occlusion.
#   - FacePart.HAIR: hair is allowed. Hair covering the eyes already
#     reduces eye visibility, which FaceVisibilityValidator handles.
#
# New prohibited classes can be appended here without changing
# OcclusionValidator itself.
OCCLUSION_PROHIBITED_PARTS: tuple[FacePart, ...] = ()

# -------------------------------
# Semantic Evidence Fusion
# -------------------------------

# Weights for continuous weighted evidence fusion combining parser,
# landmarks, pose, and occlusion.
SEMANTIC_PARSER_WEIGHT = 0.35
SEMANTIC_LANDMARK_WEIGHT = 0.20
SEMANTIC_POSE_WEIGHT = 0.20
SEMANTIC_OCCLUSION_WEIGHT = 0.10

# Final normalized confidence threshold required to treat a semantic region
# as visible or acceptable.
SEMANTIC_DECISION_THRESHOLD = 0.50
