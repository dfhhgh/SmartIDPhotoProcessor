"""Tests for the FaceSizeValidator."""

import numpy as np
import pytest

from config.constants import (
    FACE_SIZE_IDEAL_RATIO,
    FACE_SIZE_MAX_RATIO,
    FACE_SIZE_MIN_RATIO,
)
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.face_size_validator import FaceSizeValidator

_FACE_SIZE_HALF_RANGE = (
    FACE_SIZE_MAX_RATIO - FACE_SIZE_MIN_RATIO
) / 2.0

_IMAGE_SIZE = 100


# ------------------------------------------------------------------
# Image Helpers
# ------------------------------------------------------------------


def _create_image(
    height: int,
    width: int,
) -> np.ndarray:
    """Create a deterministic RGB image."""
    return np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.uint8,
    )


# ------------------------------------------------------------------
# Face Helpers
# ------------------------------------------------------------------


class _Face:
    """Lightweight mock for InsightFace Face objects."""

    def __init__(
        self,
        bbox: np.ndarray,
    ) -> None:
        self.bbox = bbox


def _create_face(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> _Face:
    """Create a mock face with a bounding box."""
    return _Face(
        bbox=np.array(
            [
                x1,
                y1,
                x2,
                y2,
            ],
            dtype=np.float32,
        ),
    )


def _create_face_for_ratio(
    face_ratio: float,
    image_size: int = _IMAGE_SIZE,
) -> _Face:
    """Create a centered face with the desired area ratio to the image.

    Computes a square bounding box centered in the image such that
    face_area / image_area equals the requested ratio.

    Args:
        face_ratio: Desired ratio of face area to image area.
        image_size: Width and height of the square image.

    Returns:
        A mock face with the computed bounding box.
    """
    image_area = float(
        image_size * image_size,
    )
    face_area = face_ratio * image_area
    side_length = np.sqrt(
        face_area,
    )
    offset = (
        float(image_size) - side_length
    ) / 2.0

    return _create_face(
        x1=offset,
        y1=offset,
        x2=float(image_size) - offset,
        y2=float(image_size) - offset,
    )


# ------------------------------------------------------------------
# Score Helpers
# ------------------------------------------------------------------


def _expected_score(
    face_ratio: float,
) -> float:
    """Compute the expected quality score for a given face ratio.

    Mirrors the normalization in FaceSizeValidator._normalize_face_ratio().
    Kept minimal; avoids full reimplementation of production logic.
    """
    distance = abs(
        face_ratio - FACE_SIZE_IDEAL_RATIO,
    )
    score = 1.0 - 0.5 * (
        distance / _FACE_SIZE_HALF_RANGE
    )

    return float(
        min(
            max(
                score,
                0.0,
            ),
            1.0,
        )
    )


# ------------------------------------------------------------------
# Assertion Helpers
# ------------------------------------------------------------------


def _assert_normalized_score(
    score: float,
) -> None:
    """Assert that a validation score is normalized."""
    assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------
# Fixture
# ------------------------------------------------------------------


@pytest.fixture
def validator() -> FaceSizeValidator:
    """Create a FaceSizeValidator instance."""
    return FaceSizeValidator()


# ------------------------------------------------------------------
# Input Validation Tests
# ------------------------------------------------------------------


def test_validate_none_image_raises_value_error(
    validator: FaceSizeValidator,
):
    """Verify that None image input raises ValueError."""
    # Arrange
    image = None
    face = _create_face(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=face,
        )


def test_validate_non_numpy_image_raises_type_error(
    validator: FaceSizeValidator,
):
    """Verify that non-NumPy image input raises TypeError."""
    # Arrange
    image = [
        [
            0,
            0,
            0,
        ],
    ]
    face = _create_face(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0,
    )

    # Act & Assert
    with pytest.raises(TypeError):
        validator.validate(
            image=image,
            face=face,
        )


def test_validate_empty_image_raises_value_error(
    validator: FaceSizeValidator,
):
    """Verify that an empty NumPy image raises ValueError."""
    # Arrange
    image = np.array(
        [],
        dtype=np.uint8,
    )
    face = _create_face(
        x1=0.0,
        y1=0.0,
        x2=100.0,
        y2=100.0,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=face,
        )


def test_validate_none_face_raises_value_error(
    validator: FaceSizeValidator,
):
    """Verify that None face input raises ValueError."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=None,
        )


# ------------------------------------------------------------------
# Bounding Box Validation Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "bbox",
    [
        pytest.param(
            (
                50.0,
                0.0,
                50.0,
                100.0,
            ),
            id="zero_width",
        ),
        pytest.param(
            (
                0.0,
                50.0,
                100.0,
                50.0,
            ),
            id="zero_height",
        ),
        pytest.param(
            (
                100.0,
                0.0,
                0.0,
                100.0,
            ),
            id="negative_width",
        ),
        pytest.param(
            (
                0.0,
                100.0,
                100.0,
                0.0,
            ),
            id="negative_height",
        ),
    ],
)
def test_validate_invalid_bbox_raises_value_error(
    validator: FaceSizeValidator,
    bbox: tuple[float, float, float, float],
):
    """Verify that invalid bounding boxes raise ValueError."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face(
        x1=bbox[0],
        y1=bbox[1],
        x2=bbox[2],
        y2=bbox[3],
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=face,
        )


# ------------------------------------------------------------------
# Validation Metric Tests
# ------------------------------------------------------------------


def test_validate_returns_validation_metric(
    validator: FaceSizeValidator,
):
    """Verify that validation returns a proper ValidationMetric."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_IDEAL_RATIO,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.FACE_SIZE
    assert isinstance(
        metric.passed,
        bool,
    )
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )
    assert isinstance(
        metric.message,
        str,
    )


# ------------------------------------------------------------------
# Face Ratio Calculation Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "face_coords, expected_ratio",
    [
        pytest.param(
            (
                40.0,
                40.0,
                60.0,
                60.0,
            ),
            (20.0 * 20.0) / (100.0 * 100.0),
            id="small_face",
        ),
        pytest.param(
            (
                10.0,
                10.0,
                90.0,
                90.0,
            ),
            (80.0 * 80.0) / (100.0 * 100.0),
            id="large_face",
        ),
        pytest.param(
            (
                0.0,
                0.0,
                100.0,
                100.0,
            ),
            (100.0 * 100.0) / (100.0 * 100.0),
            id="full_image",
        ),
    ],
)
def test_validate_computes_expected_score_from_face_ratio(
    validator: FaceSizeValidator,
    face_coords: tuple[float, float, float, float],
    expected_ratio: float,
):
    """Verify correct ratio computation for various face sizes."""
    # Arrange
    image = _create_image(
        height=100,
        width=100,
    )
    face = _create_face(
        x1=face_coords[0],
        y1=face_coords[1],
        x2=face_coords[2],
        y2=face_coords[3],
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.score == pytest.approx(
        _expected_score(expected_ratio),
    )


# ------------------------------------------------------------------
# Direct Ratio Computation Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "face_ratio, image_size",
    [
        pytest.param(
            0.10,
            100,
            id="small_face_small_image",
        ),
        pytest.param(
            FACE_SIZE_IDEAL_RATIO,
            100,
            id="ideal_face_small_image",
        ),
        pytest.param(
            0.70,
            200,
            id="large_face_large_image",
        ),
    ],
)
def test_compute_face_ratio_directly(
    validator: FaceSizeValidator,
    face_ratio: float,
    image_size: int,
):
    """Verify _compute_face_ratio returns the expected geometric ratio."""
    # Arrange
    image = _create_image(
        height=image_size,
        width=image_size,
    )
    face = _create_face_for_ratio(
        face_ratio=face_ratio,
        image_size=image_size,
    )

    # Act
    result = validator._compute_face_ratio(
        image=image,
        face=face,
    )

    # Assert
    assert result == pytest.approx(
        face_ratio,
    )


# ------------------------------------------------------------------
# Normalization Tests
# ------------------------------------------------------------------


def test_ideal_ratio_produces_highest_score(
    validator: FaceSizeValidator,
):
    """Verify that ideal face ratio yields the highest quality score."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_IDEAL_RATIO,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.score == pytest.approx(
        1.0,
    )
    assert metric.passed is True


def test_boundary_at_minimum_produces_expected_score(
    validator: FaceSizeValidator,
):
    """Verify that the minimum boundary produces the expected score."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_MIN_RATIO,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.score == pytest.approx(
        _expected_score(FACE_SIZE_MIN_RATIO),
    )
    assert metric.passed is True


def test_boundary_at_maximum_produces_expected_score(
    validator: FaceSizeValidator,
):
    """Verify that the maximum boundary produces the expected score."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_MAX_RATIO,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.score == pytest.approx(
        _expected_score(FACE_SIZE_MAX_RATIO),
    )
    assert metric.passed is True


@pytest.mark.parametrize(
    "face_ratio",
    [
        pytest.param(
            FACE_SIZE_MIN_RATIO - 0.05,
            id="below_minimum",
        ),
        pytest.param(
            FACE_SIZE_MAX_RATIO + 0.05,
            id="above_maximum",
        ),
    ],
)
def test_outside_boundary_produces_low_score(
    validator: FaceSizeValidator,
    face_ratio: float,
):
    """Verify that face ratios outside boundaries produce scores below 0.5."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=face_ratio,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.score < 0.5


# ------------------------------------------------------------------
# Symmetry Tests
# ------------------------------------------------------------------


def test_normalization_is_symmetric(
    validator: FaceSizeValidator,
):
    """Verify that ratios equally distant from ideal produce equal scores."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    delta = 0.10

    below_face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_IDEAL_RATIO - delta,
    )
    above_face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_IDEAL_RATIO + delta,
    )

    # Act
    below_metric = validator.validate(
        image=image,
        face=below_face,
    )
    above_metric = validator.validate(
        image=image,
        face=above_face,
    )

    # Assert
    assert below_metric.score == pytest.approx(
        above_metric.score,
    )


# ------------------------------------------------------------------
# Monotonicity Tests
# ------------------------------------------------------------------


def test_score_increases_as_face_approaches_ideal(
    validator: FaceSizeValidator,
):
    """Verify that score increases while approaching the ideal ratio."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    ratios = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
    ]
    faces = [
        _create_face_for_ratio(ratio)
        for ratio in ratios
    ]

    # Act
    scores = [
        validator.validate(
            image=image,
            face=face,
        ).score
        for face in faces
    ]

    # Assert
    for i in range(
        len(scores) - 1,
    ):
        assert scores[i] < scores[i + 1]


def test_score_decreases_as_face_moves_away_from_ideal(
    validator: FaceSizeValidator,
):
    """Verify that score decreases while moving away from the ideal ratio."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    ratios = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    ]
    faces = [
        _create_face_for_ratio(ratio)
        for ratio in ratios
    ]

    # Act
    scores = [
        validator.validate(
            image=image,
            face=face,
        ).score
        for face in faces
    ]

    # Assert
    for i in range(
        len(scores) - 1,
    ):
        assert scores[i] > scores[i + 1]


# ------------------------------------------------------------------
# Boundary / Threshold Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "face_ratio, expected_passed, expected_message",
    [
        pytest.param(
            0.05,
            False,
            "Face is too small.",
            id="very_small_face",
        ),
        pytest.param(
            FACE_SIZE_MIN_RATIO - 0.01,
            False,
            "Face is too small.",
            id="just_below_min",
        ),
        pytest.param(
            FACE_SIZE_MIN_RATIO,
            True,
            "Face size is acceptable.",
            id="at_min",
        ),
        pytest.param(
            FACE_SIZE_IDEAL_RATIO,
            True,
            "Face size is acceptable.",
            id="at_ideal",
        ),
        pytest.param(
            FACE_SIZE_MAX_RATIO,
            True,
            "Face size is acceptable.",
            id="at_max",
        ),
        pytest.param(
            FACE_SIZE_MAX_RATIO + 0.01,
            False,
            "Face is too large.",
            id="just_above_max",
        ),
        pytest.param(
            0.80,
            False,
            "Face is too large.",
            id="very_large_face",
        ),
    ],
)
def test_validate_face_size_threshold(
    validator: FaceSizeValidator,
    face_ratio: float,
    expected_passed: bool,
    expected_message: str,
):
    """Verify pass/fail outcome and message for representative face sizes."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=face_ratio,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.type == ValidationType.FACE_SIZE
    assert metric.passed is expected_passed
    _assert_normalized_score(
        metric.score,
    )
    assert metric.message == expected_message


# ------------------------------------------------------------------
# Score Calculation Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "face_ratio",
    [
        pytest.param(
            0.05,
            id="very_small",
        ),
        pytest.param(
            FACE_SIZE_MIN_RATIO,
            id="at_min",
        ),
        pytest.param(
            0.25,
            id="below_ideal",
        ),
        pytest.param(
            FACE_SIZE_IDEAL_RATIO,
            id="at_ideal",
        ),
        pytest.param(
            0.55,
            id="above_ideal",
        ),
        pytest.param(
            FACE_SIZE_MAX_RATIO,
            id="at_max",
        ),
        pytest.param(
            0.80,
            id="very_large",
        ),
    ],
)
def test_validate_score_matches_expected_calculation(
    validator: FaceSizeValidator,
    face_ratio: float,
):
    """Verify that the score matches the expected linear formula."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=face_ratio,
    )
    expected = _expected_score(
        face_ratio,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.score == pytest.approx(
        expected,
    )


# ------------------------------------------------------------------
# Determinism Tests
# ------------------------------------------------------------------


def test_validate_is_deterministic(
    validator: FaceSizeValidator,
):
    """Verify that calling validate multiple times produces identical results."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_IDEAL_RATIO,
    )

    # Act
    metric1 = validator.validate(
        image=image,
        face=face,
    )
    metric2 = validator.validate(
        image=image,
        face=face,
    )
    metric3 = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric1.type == metric2.type == metric3.type
    assert metric1.passed == metric2.passed == metric3.passed
    assert metric1.score == metric2.score == metric3.score
    assert metric1.message == metric2.message == metric3.message


# ------------------------------------------------------------------
# Score Type Tests
# ------------------------------------------------------------------


def test_validate_score_type_is_float(
    validator: FaceSizeValidator,
):
    """Verify that the returned score is a native Python float."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_IDEAL_RATIO,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert isinstance(
        metric.score,
        float,
    )


# ------------------------------------------------------------------
# Non-Square Image Tests
# ------------------------------------------------------------------


def test_validate_rectangular_image(
    validator: FaceSizeValidator,
):
    """Verify that validation works correctly with non-square images."""
    # Arrange
    image = _create_image(
        height=200,
        width=100,
    )
    face = _create_face(
        x1=10.0,
        y1=40.0,
        x2=90.0,
        y2=160.0,
    )
    expected_ratio = (80.0 * 120.0) / (100.0 * 200.0)

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.type == ValidationType.FACE_SIZE
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )
    assert metric.score == pytest.approx(
        _expected_score(expected_ratio),
    )


def test_validate_uses_face_coordinates_matching_the_provided_image(
    validator: FaceSizeValidator,
):
    """Face ratio must be computed from the face bbox and image in one coordinate system."""
    aligned_image = _create_image(
        height=112,
        width=112,
    )
    aligned_face = _create_face(
        x1=28.0,
        y1=28.0,
        x2=84.0,
        y2=84.0,
    )
    expected_ratio = (56.0 * 56.0) / (112.0 * 112.0)

    metric = validator.validate(
        image=aligned_image,
        face=aligned_face,
    )

    assert metric.score == pytest.approx(
        _expected_score(expected_ratio),
    )


# ------------------------------------------------------------------
# Calibration Regression Tests (Phase 5 face-size calibration)
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "face_ratio, expected_passed, expected_message",
    [
        pytest.param(
            0.093,
            True,
            "Face size is acceptable.",
            id="borderline_093_now_passes",
        ),
        pytest.param(
            FACE_SIZE_MIN_RATIO - 0.01,
            False,
            "Face is too small.",
            id="just_below_new_min",
        ),
        pytest.param(
            FACE_SIZE_MIN_RATIO,
            True,
            "Face size is acceptable.",
            id="exactly_at_new_min",
        ),
        pytest.param(
            FACE_SIZE_MIN_RATIO + 0.01,
            True,
            "Face size is acceptable.",
            id="just_above_new_min",
        ),
        pytest.param(
            0.112,
            True,
            "Face size is acceptable.",
            id="hijab2_ratio",
        ),
        pytest.param(
            FACE_SIZE_IDEAL_RATIO,
            True,
            "Face size is acceptable.",
            id="at_ideal",
        ),
        pytest.param(
            FACE_SIZE_MAX_RATIO,
            True,
            "Face size is acceptable.",
            id="at_max",
        ),
        pytest.param(
            0.80,
            False,
            "Face is too large.",
            id="clearly_oversized",
        ),
        pytest.param(
            0.05,
            False,
            "Face is too small.",
            id="clearly_undersized",
        ),
    ],
)
def test_calibrated_face_size_threshold(
    validator: FaceSizeValidator,
    face_ratio: float,
    expected_passed: bool,
    expected_message: str,
):
    """Verify calibrated pass/fail and message for representative face sizes."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=face_ratio,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.type == ValidationType.FACE_SIZE
    assert metric.passed is expected_passed
    _assert_normalized_score(metric.score)
    assert metric.message == expected_message


def test_calibrated_score_at_min_boundary(
    validator: FaceSizeValidator,
):
    """Verify that score at the new minimum boundary matches expected formula."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=FACE_SIZE_MIN_RATIO,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    expected = _expected_score(FACE_SIZE_MIN_RATIO)
    assert metric.score == pytest.approx(expected)
    assert metric.passed is True


def test_calibrated_hijab_portrait_accepted(
    validator: FaceSizeValidator,
):
    """Verify that a hijab portrait with face_ratio ~0.112 is accepted."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=0.112,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(
        _expected_score(0.112),
    )


def test_calibrated_genuinely_small_face_rejected(
    validator: FaceSizeValidator,
):
    """Verify that genuinely small faces (ratio < 0.07) are still rejected."""
    # Arrange
    image = _create_image(
        height=_IMAGE_SIZE,
        width=_IMAGE_SIZE,
    )
    face = _create_face_for_ratio(
        face_ratio=0.05,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.message == "Face is too small."


# ------------------------------------------------------------------
# Crop-Space Validation Tests
# ------------------------------------------------------------------


def test_crop_space_ratio_computed_from_crop_image_and_crop_face(
    validator: FaceSizeValidator,
):
    """When crop_image and crop_face are provided, ratio uses crop space."""
    # Arrange
    crop_image = _create_image(height=400, width=300)
    crop_face = _create_face(x1=50.0, y1=80.0, x2=250.0, y2=320.0)
    expected_ratio = (200.0 * 240.0) / (300.0 * 400.0)

    # Act
    metric = validator.validate(
        image=crop_image,
        face=crop_face,
        crop_image=crop_image,
        crop_face=crop_face,
    )

    # Assert
    assert metric.score == pytest.approx(_expected_score(expected_ratio))


def test_crop_space_ignores_original_image_dimensions(
    validator: FaceSizeValidator,
):
    """Crop-space validation must use crop dimensions, not original image dimensions."""
    # Arrange: original is 4000x6000, crop is 200x300
    original_image = _create_image(height=6000, width=4000)
    original_face = _create_face(x1=1500.0, y1=2000.0, x2=2500.0, y2=4000.0)
    # original ratio: (1000 * 2000) / (4000 * 6000) = 0.0833

    crop_image = _create_image(height=300, width=200)
    crop_face = _create_face(x1=20.0, y1=30.0, x2=180.0, y2=270.0)
    # crop ratio: (160 * 240) / (200 * 300) = 0.64

    # Act
    metric = validator.validate(
        image=original_image,
        face=original_face,
        crop_image=crop_image,
        crop_face=crop_face,
    )

    # Assert: score must match crop-space ratio, not original-space ratio
    assert metric.score == pytest.approx(_expected_score(0.64))


def test_crop_space_fallback_to_original_when_crop_not_provided(
    validator: FaceSizeValidator,
):
    """When crop_image/crop_face are None, validator uses image/face as before."""
    image = _create_image(height=_IMAGE_SIZE, width=_IMAGE_SIZE)
    face = _create_face_for_ratio(face_ratio=FACE_SIZE_IDEAL_RATIO)

    # Act — no crop_image or crop_face
    metric = validator.validate(image=image, face=face)

    # Assert — uses image/face, which have ideal ratio
    assert metric.score == pytest.approx(1.0)
    assert metric.passed is True


def test_crop_space_fallback_when_only_crop_image_provided(
    validator: FaceSizeValidator,
):
    """When only crop_image is provided (without crop_face), falls back to image/face."""
    image = _create_image(height=_IMAGE_SIZE, width=_IMAGE_SIZE)
    face = _create_face_for_ratio(face_ratio=FACE_SIZE_IDEAL_RATIO)
    crop_image = _create_image(height=100, width=100)

    # Act — only crop_image, no crop_face
    metric = validator.validate(image=image, face=face, crop_image=crop_image)

    # Assert — falls back to image/face
    assert metric.score == pytest.approx(1.0)


def test_crop_space_fallback_when_only_crop_face_provided(
    validator: FaceSizeValidator,
):
    """When only crop_face is provided (without crop_image), falls back to image/face."""
    image = _create_image(height=_IMAGE_SIZE, width=_IMAGE_SIZE)
    face = _create_face_for_ratio(face_ratio=FACE_SIZE_IDEAL_RATIO)
    crop_face = _create_face(x1=10.0, y1=10.0, x2=90.0, y2=90.0)

    # Act — only crop_face, no crop_image
    metric = validator.validate(image=image, face=face, crop_face=crop_face)

    # Assert — falls back to image/face
    assert metric.score == pytest.approx(1.0)


def test_crop_space_pass_fail_uses_correct_thresholds(
    validator: FaceSizeValidator,
):
    """Crop-space ratio is evaluated against the same MIN/MAX thresholds."""
    # Arrange: face_ratio in crop space = 0.05, below MIN_RATIO (0.08)
    crop_image = _create_image(height=1000, width=1000)
    crop_face = _create_face_for_ratio(face_ratio=0.05, image_size=1000)

    # Act
    metric = validator.validate(
        image=crop_image,
        face=crop_face,
        crop_image=crop_image,
        crop_face=crop_face,
    )

    # Assert
    assert metric.passed is False
    assert metric.message == "Face is too small."


def test_crop_space_score_formula_unchanged(
    validator: FaceSizeValidator,
):
    """Score formula is the same regardless of coordinate space."""
    crop_image = _create_image(height=100, width=100)
    crop_face = _create_face_for_ratio(face_ratio=0.30, image_size=100)

    # Act
    metric = validator.validate(
        image=crop_image,
        face=crop_face,
        crop_image=crop_image,
        crop_face=crop_face,
    )

    # Assert
    assert metric.score == pytest.approx(_expected_score(0.30))


def test_crop_space_coordinate_system_safety(
    validator: FaceSizeValidator,
):
    """Verify crop-space ratio differs from original-space when coordinate systems diverge."""
    # Original: small face in large image → low ratio
    original_image = _create_image(height=6000, width=4000)
    original_face = _create_face(x1=1800.0, y1=1100.0, x2=2700.0, y2=2200.0)
    # original_ratio = (900 * 1100) / (4000 * 6000) = 0.04125

    # Crop: face occupies large portion of small crop
    crop_image = _create_image(height=2372, width=1780)
    crop_face = _create_face(x1=463.0, y1=485.0, x2=1315.0, y2=1563.0)
    # crop_ratio = (852 * 1078) / (1780 * 2372) = 0.2175

    # Act
    metric = validator.validate(
        image=original_image,
        face=original_face,
        crop_image=crop_image,
        crop_face=crop_face,
    )

    # Assert: uses crop-space ratio (0.2175), not original-space (0.04125)
    assert metric.passed is True
    assert metric.score == pytest.approx(_expected_score(0.2175), abs=0.01)
