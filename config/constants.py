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

# -------------------------------
# Face Cropping
# -------------------------------

TOP_PADDING_RATIO = 0.25
BOTTOM_PADDING_RATIO = 0.30
SIDE_PADDING_RATIO = 0.18
