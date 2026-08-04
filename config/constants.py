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
FACE_SIZE_MIN_RATIO = 0.15
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

# Mandatory anatomical regions for a valid student ID photograph.
# Note: MOUTH, UPPER_LIP, LOWER_LIP are handled as a single composite
# region by FaceParsingResult.has_visible_mouth_region() and are
# intentionally excluded from this list.
FACE_VISIBILITY_REQUIRED_PARTS: tuple[FacePart, ...] = (
    FacePart.LEFT_EYE,
    FacePart.RIGHT_EYE,
    FacePart.LEFT_BROW,
    FacePart.RIGHT_BROW,
    FacePart.NOSE,
)

# Minimum acceptable ratio of (part pixel area / TOTAL IMAGE area) for each
# mandatory region. These are relative to the whole frame, not the face
# bounding box, so they are intentionally small: a typical ID photo face
# occupies roughly FACE_SIZE_MIN_RATIO to FACE_SIZE_MAX_RATIO of the image
# (see Face Size Validation above), and each individual feature is only a
# small fraction of the face itself.
# Initial conservative defaults for a university student ID system; not
# scientifically fixed and should be calibrated later using a
# representative dataset of real student ID photos.
FACE_VISIBILITY_MIN_PART_RATIOS: dict[FacePart, float] = {
    FacePart.LEFT_EYE: 0.0015,
    FacePart.RIGHT_EYE: 0.0015,
    FacePart.LEFT_BROW: 0.0010,
    FacePart.RIGHT_BROW: 0.0010,
    FacePart.NOSE: 0.0050,
    FacePart.MOUTH: 0.0008,
    FacePart.UPPER_LIP: 0.0020,
    FacePart.LOWER_LIP: 0.0020,
}

# Fraction of a region's full scoring weight that is deducted when the
# region is present but below its minimum visibility ratio, as opposed to
# being entirely missing (which deducts the region's full weight).
FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR = 0.5

# -------------------------------
# Head Pose Validation
# -------------------------------

# Initial conservative default values for a university student ID system.
# These thresholds define the acceptable absolute deviation, in degrees,
# for each pose axis relative to a perfectly frontal (0.0) pose. They are
# not scientifically fixed and should be calibrated later using a
# representative dataset of real student ID photos.
HEAD_POSE_PITCH_MAX_DEGREES = 15.0
HEAD_POSE_YAW_MAX_DEGREES = 15.0
HEAD_POSE_ROLL_MAX_DEGREES = 10.0

# -------------------------------
# Occlusion Validation
# -------------------------------

# Semantic classes that are never acceptable in a student ID photo.
#
# Notably absent:
#   - FacePart.EYE_GLASS: normal eyeglasses are allowed. A future
#     GlassesValidator will distinguish normal glasses from sunglasses;
#     OcclusionValidator must not treat eyeglasses as an occlusion.
#   - FacePart.HAIR: hair is allowed. Hair covering the eyes already
#     reduces eye visibility, which FaceVisibilityValidator handles.
#
# New prohibited classes can be appended here without changing
# OcclusionValidator itself.
OCCLUSION_PROHIBITED_PARTS: tuple[FacePart, ...] = (
    FacePart.HAT,
)

# -------------------------------
# Glasses Validation
# -------------------------------

GLASSES_SUCCESS_MESSAGE = "Acceptable eyewear detected."
GLASSES_FAILURE_MESSAGE = "Sunglasses are not permitted."
# Decision thresholds for the glasses-detector binary classifiers.
# Both classifiers output a probability in [0.0, 1.0]; a value strictly
# greater than the threshold is treated as a positive detection.
# Not scientifically fixed; calibrate later against a representative
# dataset of real student ID photos.
GLASSES_SUNGLASSES_PROBABILITY_THRESHOLD = 0.5
GLASSES_EYEGLASSES_PROBABILITY_THRESHOLD = 0.5
