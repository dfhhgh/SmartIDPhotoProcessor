"""Tests for the HeadPoseValidator."""

import math

import numpy as np
import pytest
from unittest.mock import MagicMock

from config.constants import (
    HEAD_POSE_PITCH_MAX_DEGREES,
    HEAD_POSE_ROLL_MAX_DEGREES,
    HEAD_POSE_YAW_MAX_DEGREES,
)
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.head_pose_validator import HeadPoseValidator


def _make_face(pose) -> MagicMock:
    """Create a MagicMock standing in for an InsightFace Face with a pose."""
    face = MagicMock()
    face.pose = pose
    return face


def _make_face_without_pose() -> MagicMock:
    """Create a MagicMock standing in for a Face with no pose attribute at all."""
    return MagicMock(spec=[])


def _assert_normalized_score(score: float) -> None:
    """Assert that a validation score is normalized."""
    assert 0.0 <= score <= 1.0


def _expected_axis_score(
    angle: float,
    max_degrees: float,
) -> float:
    """Compute the expected per-axis normalized score using the documented formula."""
    score = 1.0 - 0.5 * (abs(angle) / max_degrees)

    return max(
        min(
            score,
            1.0,
        ),
        0.0,
    )


def _expected_score(
    pitch: float,
    yaw: float,
    roll: float,
) -> float:
    """Compute the expected combined score for a full pose."""
    pitch_score = _expected_axis_score(pitch, HEAD_POSE_PITCH_MAX_DEGREES)
    yaw_score = _expected_axis_score(yaw, HEAD_POSE_YAW_MAX_DEGREES)
    roll_score = _expected_axis_score(roll, HEAD_POSE_ROLL_MAX_DEGREES)

    return (pitch_score + yaw_score + roll_score) / 3.0


@pytest.fixture
def validator() -> HeadPoseValidator:
    """Create a HeadPoseValidator instance."""
    return HeadPoseValidator()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a deterministic placeholder BGR image."""
    return np.zeros(
        (
            64,
            64,
            3,
        ),
        dtype=np.uint8,
    )


# ================================================================== #
# validate() — input validation
# ================================================================== #


def test_validate_none_image_raises_value_error(
    validator: HeadPoseValidator,
):
    """Verify that None image input raises ValueError with expected message."""
    # Arrange
    face = _make_face((0.0, 0.0, 0.0))

    # Act & Assert
    with pytest.raises(ValueError, match="Image must not be None."):
        validator.validate(
            image=None,
            face=face,
        )


def test_validate_non_numpy_image_raises_type_error(
    validator: HeadPoseValidator,
):
    """Verify that non-NumPy image input raises TypeError with expected message."""
    # Arrange
    image = [
        [
            0,
            255,
        ],
    ]
    face = _make_face((0.0, 0.0, 0.0))

    # Act & Assert
    with pytest.raises(TypeError, match="Image must be a numpy array."):
        validator.validate(
            image=image,
            face=face,
        )


def test_validate_empty_image_raises_value_error(
    validator: HeadPoseValidator,
):
    """Verify that an empty NumPy image raises ValueError with expected message."""
    # Arrange
    image = np.array(
        [],
        dtype=np.uint8,
    )
    face = _make_face((0.0, 0.0, 0.0))

    # Act & Assert
    with pytest.raises(ValueError, match="Image must not be empty."):
        validator.validate(
            image=image,
            face=face,
        )


def test_validate_none_face_raises_value_error(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify that a None face raises ValueError with expected message."""
    # Act & Assert
    with pytest.raises(ValueError, match="Face must not be None."):
        validator.validate(
            image=sample_image,
            face=None,
        )


# ================================================================== #
# _extract_pose() — valid inputs
# ================================================================== #


def test_extract_pose_valid_tuple_returns_expected_values(
    validator: HeadPoseValidator,
):
    """Verify that a valid pose tuple is extracted correctly."""
    # Arrange
    face = _make_face((1.5, -2.5, 3.5))

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert pitch == pytest.approx(1.5)
    assert yaw == pytest.approx(-2.5)
    assert roll == pytest.approx(3.5)


def test_extract_pose_returns_native_python_floats(
    validator: HeadPoseValidator,
):
    """Verify that returned pose values are native Python floats."""
    # Arrange
    face = _make_face((1, 2, 3))

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert type(pitch) is float
    assert type(yaw) is float
    assert type(roll) is float


def test_extract_pose_accepts_list(
    validator: HeadPoseValidator,
):
    """Verify that a pose provided as a list is accepted."""
    # Arrange
    face = _make_face([4.0, 5.0, 6.0])

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert (pitch, yaw, roll) == (4.0, 5.0, 6.0)


def test_extract_pose_accepts_numpy_array(
    validator: HeadPoseValidator,
):
    """Verify that a pose provided as a NumPy array is accepted."""
    # Arrange
    face = _make_face(np.array([1.0, 2.0, 3.0]))

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert (pitch, yaw, roll) == (1.0, 2.0, 3.0)
    assert type(pitch) is float


def test_extract_pose_accepts_numpy_scalar_values(
    validator: HeadPoseValidator,
):
    """Verify that individual NumPy scalar pose values are converted to floats."""
    # Arrange
    face = _make_face(
        (
            np.float32(1.25),
            np.float64(-2.5),
            np.int32(3),
        )
    )

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert pitch == pytest.approx(1.25)
    assert yaw == pytest.approx(-2.5)
    assert roll == pytest.approx(3.0)
    assert type(pitch) is float
    assert type(yaw) is float
    assert type(roll) is float


def test_extract_pose_accepts_integer_values(
    validator: HeadPoseValidator,
):
    """Verify that integer pose values are converted to floats."""
    # Arrange
    face = _make_face((1, -2, 3))

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert (pitch, yaw, roll) == (1.0, -2.0, 3.0)
    assert type(pitch) is float
    assert type(yaw) is float
    assert type(roll) is float


def test_extract_pose_accepts_float_values(
    validator: HeadPoseValidator,
):
    """Verify that float pose values are preserved."""
    # Arrange
    face = _make_face((1.1, -2.2, 3.3))

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert pitch == pytest.approx(1.1)
    assert yaw == pytest.approx(-2.2)
    assert roll == pytest.approx(3.3)


def test_extract_pose_accepts_zero_values(
    validator: HeadPoseValidator,
):
    """Verify that an all-zero pose (perfectly frontal) is accepted."""
    # Arrange
    face = _make_face((0.0, 0.0, 0.0))

    # Act
    pitch, yaw, roll = validator._extract_pose(
        face=face,
    )

    # Assert
    assert (pitch, yaw, roll) == (0.0, 0.0, 0.0)


# ================================================================== #
# _extract_pose() — missing / None pose
# ================================================================== #


def test_extract_pose_missing_pose_attribute_raises_value_error(
    validator: HeadPoseValidator,
):
    """Verify that a face with no pose attribute at all raises ValueError."""
    # Arrange
    face = _make_face_without_pose()

    # Act & Assert
    with pytest.raises(ValueError, match="Face pose must not be None."):
        validator._extract_pose(
            face=face,
        )


def test_extract_pose_none_pose_raises_value_error(
    validator: HeadPoseValidator,
):
    """Verify that an explicit None pose raises ValueError."""
    # Arrange
    face = _make_face(None)

    # Act & Assert
    with pytest.raises(ValueError, match="Face pose must not be None."):
        validator._extract_pose(
            face=face,
        )


# ================================================================== #
# _extract_pose() — wrong length / non-iterable
# ================================================================== #


@pytest.mark.parametrize(
    "invalid_pose",
    [
        pytest.param((1.0, 2.0), id="two_values"),
        pytest.param((1.0,), id="one_value"),
        pytest.param((), id="empty_tuple"),
        pytest.param((1.0, 2.0, 3.0, 4.0), id="four_values"),
        pytest.param([1.0, 2.0, 3.0, 4.0, 5.0], id="five_values_list"),
    ],
)
def test_extract_pose_wrong_length_raises_value_error(
    validator: HeadPoseValidator,
    invalid_pose,
):
    """Verify that a pose with fewer or more than three values raises ValueError."""
    # Arrange
    face = _make_face(invalid_pose)

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Face pose must contain exactly three values",
    ):
        validator._extract_pose(
            face=face,
        )


@pytest.mark.parametrize(
    "non_iterable_pose",
    [
        pytest.param(42, id="integer"),
        pytest.param(3.14, id="float"),
        pytest.param(True, id="boolean"),
    ],
)
def test_extract_pose_non_iterable_raises_value_error(
    validator: HeadPoseValidator,
    non_iterable_pose,
):
    """Verify that a non-iterable pose value raises ValueError."""
    # Arrange
    face = _make_face(non_iterable_pose)

    # Act & Assert
    with pytest.raises(
        ValueError,
        match="Face pose must contain exactly three values",
    ):
        validator._extract_pose(
            face=face,
        )


# ================================================================== #
# _extract_pose() — non-numeric values
# ================================================================== #


@pytest.mark.parametrize(
    "invalid_pose",
    [
        pytest.param(("a", "b", "c"), id="all_strings"),
        pytest.param((1.0, "yaw", 3.0), id="middle_string"),
        pytest.param((None, 2.0, 3.0), id="none_value"),
        pytest.param((1.0, 2.0, [3.0]), id="nested_list"),
        pytest.param((1.0, 2.0, object()), id="unsupported_object"),
    ],
)
def test_extract_pose_non_numeric_values_raise_type_error(
    validator: HeadPoseValidator,
    invalid_pose,
):
    """Verify that non-numeric pose values raise TypeError."""
    # Arrange
    face = _make_face(invalid_pose)

    # Act & Assert
    with pytest.raises(TypeError, match="Face pose values must be numeric."):
        validator._extract_pose(
            face=face,
        )


# ================================================================== #
# _extract_pose() — non-finite values
# ================================================================== #


@pytest.mark.parametrize(
    "invalid_pose",
    [
        pytest.param((math.nan, 0.0, 0.0), id="pitch_nan"),
        pytest.param((0.0, math.nan, 0.0), id="yaw_nan"),
        pytest.param((0.0, 0.0, math.nan), id="roll_nan"),
        pytest.param((math.inf, 0.0, 0.0), id="pitch_positive_inf"),
        pytest.param((0.0, math.inf, 0.0), id="yaw_positive_inf"),
        pytest.param((0.0, 0.0, math.inf), id="roll_positive_inf"),
        pytest.param((-math.inf, 0.0, 0.0), id="pitch_negative_inf"),
        pytest.param((0.0, -math.inf, 0.0), id="yaw_negative_inf"),
        pytest.param((0.0, 0.0, -math.inf), id="roll_negative_inf"),
        pytest.param((math.nan, math.inf, -math.inf), id="all_non_finite"),
    ],
)
def test_extract_pose_non_finite_values_raise_type_error(
    validator: HeadPoseValidator,
    invalid_pose,
):
    """Verify that NaN or +/-Inf pose values raise TypeError."""
    # Arrange
    face = _make_face(invalid_pose)

    # Act & Assert
    with pytest.raises(
        TypeError,
        match="Face pose values must be finite numbers.",
    ):
        validator._extract_pose(
            face=face,
        )


# ================================================================== #
# validate() — combined axis outcomes
# ================================================================== #


def test_validate_fully_valid_pose_passes(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify that a perfectly frontal pose passes with a perfect score."""
    # Arrange
    face = _make_face((0.0, 0.0, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.HEAD_POSE
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)
    assert metric.message == "Head pose is acceptable."


def test_validate_pitch_only_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when only pitch exceeds its threshold (looking down)."""
    # Arrange
    pitch = HEAD_POSE_PITCH_MAX_DEGREES + 5.0
    face = _make_face((pitch, 0.0, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(pitch, 0.0, 0.0)
    )
    assert metric.message == "Head is tilted too far downward."


def test_validate_yaw_only_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when only yaw exceeds its threshold (turned right)."""
    # Arrange
    yaw = HEAD_POSE_YAW_MAX_DEGREES + 5.0
    face = _make_face((0.0, yaw, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(0.0, yaw, 0.0)
    )
    assert metric.message == "Head is turned too far right."


def test_validate_roll_only_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when only roll exceeds its threshold."""
    # Arrange
    roll = HEAD_POSE_ROLL_MAX_DEGREES + 5.0
    face = _make_face((0.0, 0.0, roll))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(0.0, 0.0, roll)
    )
    assert metric.message == "Head roll exceeds the acceptable limit."


def test_validate_pitch_and_yaw_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when pitch and yaw both exceed their thresholds."""
    # Arrange
    pitch = HEAD_POSE_PITCH_MAX_DEGREES + 5.0
    yaw = HEAD_POSE_YAW_MAX_DEGREES + 5.0
    face = _make_face((pitch, yaw, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(pitch, yaw, 0.0)
    )
    assert metric.message == (
        "Head is turned too far right. Head is tilted too far downward."
    )


def test_validate_pitch_and_roll_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when pitch and roll both exceed their thresholds."""
    # Arrange
    pitch = -(HEAD_POSE_PITCH_MAX_DEGREES + 5.0)
    roll = HEAD_POSE_ROLL_MAX_DEGREES + 5.0
    face = _make_face((pitch, 0.0, roll))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(pitch, 0.0, roll)
    )
    assert metric.message == (
        "Head is tilted too far upward. Head roll exceeds the acceptable limit."
    )


def test_validate_yaw_and_roll_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when yaw and roll both exceed their thresholds."""
    # Arrange
    yaw = -(HEAD_POSE_YAW_MAX_DEGREES + 5.0)
    roll = HEAD_POSE_ROLL_MAX_DEGREES + 5.0
    face = _make_face((0.0, yaw, roll))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(0.0, yaw, roll)
    )
    assert metric.message == (
        "Head is turned too far left. Head roll exceeds the acceptable limit."
    )


def test_validate_all_axes_invalid(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify behaviour when pitch, yaw, and roll all exceed their thresholds."""
    # Arrange
    pitch = HEAD_POSE_PITCH_MAX_DEGREES + 5.0
    yaw = HEAD_POSE_YAW_MAX_DEGREES + 5.0
    roll = HEAD_POSE_ROLL_MAX_DEGREES + 5.0
    face = _make_face((pitch, yaw, roll))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(pitch, yaw, roll)
    )
    assert metric.message == (
        "Head is turned too far right. "
        "Head is tilted too far downward. "
        "Head roll exceeds the acceptable limit."
    )


def test_validate_returns_validation_metric_type(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
):
    """Verify that validate() always returns a ValidationMetric of the correct type."""
    # Arrange
    face = _make_face((0.0, 0.0, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.HEAD_POSE
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


# ================================================================== #
# Boundary tests — per axis
# ================================================================== #


@pytest.mark.parametrize(
    "pitch, expected_passed",
    [
        pytest.param(HEAD_POSE_PITCH_MAX_DEGREES, True, id="exactly_at_positive_threshold"),
        pytest.param(HEAD_POSE_PITCH_MAX_DEGREES - 0.001, True, id="just_below_positive_threshold"),
        pytest.param(HEAD_POSE_PITCH_MAX_DEGREES + 0.001, False, id="just_above_positive_threshold"),
        pytest.param(-HEAD_POSE_PITCH_MAX_DEGREES, True, id="exactly_at_negative_threshold"),
        pytest.param(-HEAD_POSE_PITCH_MAX_DEGREES + 0.001, True, id="just_below_negative_threshold_magnitude"),
        pytest.param(-HEAD_POSE_PITCH_MAX_DEGREES - 0.001, False, id="just_above_negative_threshold_magnitude"),
    ],
)
def test_validate_pitch_boundary(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
    pitch: float,
    expected_passed: bool,
):
    """Verify pass/fail behaviour at, just below, and just above the pitch threshold."""
    # Arrange
    face = _make_face((pitch, 0.0, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is expected_passed


@pytest.mark.parametrize(
    "yaw, expected_passed",
    [
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES, True, id="exactly_at_positive_threshold"),
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES - 0.001, True, id="just_below_positive_threshold"),
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES + 0.001, False, id="just_above_positive_threshold"),
        pytest.param(-HEAD_POSE_YAW_MAX_DEGREES, True, id="exactly_at_negative_threshold"),
        pytest.param(-HEAD_POSE_YAW_MAX_DEGREES + 0.001, True, id="just_below_negative_threshold_magnitude"),
        pytest.param(-HEAD_POSE_YAW_MAX_DEGREES - 0.001, False, id="just_above_negative_threshold_magnitude"),
    ],
)
def test_validate_yaw_boundary(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
    yaw: float,
    expected_passed: bool,
):
    """Verify pass/fail behaviour at, just below, and just above the yaw threshold."""
    # Arrange
    face = _make_face((0.0, yaw, 0.0))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is expected_passed


@pytest.mark.parametrize(
    "roll, expected_passed",
    [
        pytest.param(HEAD_POSE_ROLL_MAX_DEGREES, True, id="exactly_at_positive_threshold"),
        pytest.param(HEAD_POSE_ROLL_MAX_DEGREES - 0.001, True, id="just_below_positive_threshold"),
        pytest.param(HEAD_POSE_ROLL_MAX_DEGREES + 0.001, False, id="just_above_positive_threshold"),
        pytest.param(-HEAD_POSE_ROLL_MAX_DEGREES, True, id="exactly_at_negative_threshold"),
        pytest.param(-HEAD_POSE_ROLL_MAX_DEGREES + 0.001, True, id="just_below_negative_threshold_magnitude"),
        pytest.param(-HEAD_POSE_ROLL_MAX_DEGREES - 0.001, False, id="just_above_negative_threshold_magnitude"),
    ],
)
def test_validate_roll_boundary(
    validator: HeadPoseValidator,
    sample_image: np.ndarray,
    roll: float,
    expected_passed: bool,
):
    """Verify pass/fail behaviour at, just below, and just above the roll threshold."""
    # Arrange
    face = _make_face((0.0, 0.0, roll))

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
    )

    # Assert
    assert metric.passed is expected_passed


# ================================================================== #
# _compute_score()
# ================================================================== #


def test_compute_score_perfect_pose_returns_one(
    validator: HeadPoseValidator,
):
    """Verify that a perfectly frontal pose yields a score of 1.0."""
    # Act
    score = validator._compute_score(
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
    )

    # Assert
    assert score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "pitch, yaw, roll",
    [
        pytest.param(HEAD_POSE_PITCH_MAX_DEGREES + 5.0, 0.0, 0.0, id="pitch_deviation_only"),
        pytest.param(0.0, HEAD_POSE_YAW_MAX_DEGREES + 5.0, 0.0, id="yaw_deviation_only"),
        pytest.param(0.0, 0.0, HEAD_POSE_ROLL_MAX_DEGREES + 5.0, id="roll_deviation_only"),
    ],
)
def test_compute_score_single_axis_deviation(
    validator: HeadPoseValidator,
    pitch: float,
    yaw: float,
    roll: float,
):
    """Verify that a single-axis deviation reduces the score by the expected amount."""
    # Act
    score = validator._compute_score(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )

    # Assert
    assert score == pytest.approx(
        _expected_score(pitch, yaw, roll)
    )
    assert score < 1.0


def test_compute_score_mixed_deviations(
    validator: HeadPoseValidator,
):
    """Verify that mixed deviations across axes combine into the expected average."""
    # Arrange
    pitch = 5.0
    yaw = -8.0
    roll = 3.0

    # Act
    score = validator._compute_score(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )

    # Assert
    assert score == pytest.approx(
        _expected_score(pitch, yaw, roll)
    )
    _assert_normalized_score(score)


def test_compute_score_large_deviations_reduce_score_significantly(
    validator: HeadPoseValidator,
):
    """Verify that large deviations produce a noticeably reduced score."""
    # Arrange
    pitch = HEAD_POSE_PITCH_MAX_DEGREES * 2
    yaw = HEAD_POSE_YAW_MAX_DEGREES * 2
    roll = HEAD_POSE_ROLL_MAX_DEGREES * 2

    # Act
    score = validator._compute_score(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )

    # Assert
    assert score == pytest.approx(0.0)


def test_compute_score_extreme_deviations_clamp_to_zero(
    validator: HeadPoseValidator,
):
    """Verify that extreme deviations clamp the score at exactly 0.0, never negative."""
    # Arrange
    pitch = HEAD_POSE_PITCH_MAX_DEGREES * 100
    yaw = HEAD_POSE_YAW_MAX_DEGREES * 100
    roll = HEAD_POSE_ROLL_MAX_DEGREES * 100

    # Act
    score = validator._compute_score(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )

    # Assert
    assert score == pytest.approx(0.0)
    assert score >= 0.0


@pytest.mark.parametrize(
    "pitch, yaw, roll",
    [
        pytest.param(0.0, 0.0, 0.0, id="zero"),
        pytest.param(5.0, -5.0, 5.0, id="small_mixed"),
        pytest.param(1000.0, -1000.0, 1000.0, id="extreme_mixed"),
        pytest.param(HEAD_POSE_PITCH_MAX_DEGREES, HEAD_POSE_YAW_MAX_DEGREES, HEAD_POSE_ROLL_MAX_DEGREES, id="all_at_threshold"),
    ],
)
def test_compute_score_always_within_bounds(
    validator: HeadPoseValidator,
    pitch: float,
    yaw: float,
    roll: float,
):
    """Verify that the computed score always stays within [0.0, 1.0]."""
    # Act
    score = validator._compute_score(
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )

    # Assert
    _assert_normalized_score(score)


# ================================================================== #
# _normalize_angle()
# ================================================================== #


def test_normalize_angle_zero_returns_one(
    validator: HeadPoseValidator,
):
    """Verify that a zero-degree angle yields the maximum score of 1.0."""
    # Act
    score = validator._normalize_angle(
        angle=0.0,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert score == pytest.approx(1.0)


def test_normalize_angle_half_threshold_returns_expected_score(
    validator: HeadPoseValidator,
):
    """Verify the score at half of the configured threshold."""
    # Arrange
    half_angle = HEAD_POSE_YAW_MAX_DEGREES / 2.0

    # Act
    score = validator._normalize_angle(
        angle=half_angle,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert score == pytest.approx(0.75)


def test_normalize_angle_exact_threshold_returns_half(
    validator: HeadPoseValidator,
):
    """Verify that the score at exactly the threshold is 0.5."""
    # Act
    score = validator._normalize_angle(
        angle=HEAD_POSE_YAW_MAX_DEGREES,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert score == pytest.approx(0.5)


def test_normalize_angle_above_threshold_returns_below_half(
    validator: HeadPoseValidator,
):
    """Verify that an angle beyond the threshold scores below 0.5."""
    # Act
    score = validator._normalize_angle(
        angle=HEAD_POSE_YAW_MAX_DEGREES * 1.5,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert score < 0.5
    _assert_normalized_score(score)


def test_normalize_angle_very_large_angle_clamps_to_zero(
    validator: HeadPoseValidator,
):
    """Verify that a very large angle clamps the score to exactly 0.0."""
    # Act
    score = validator._normalize_angle(
        angle=HEAD_POSE_YAW_MAX_DEGREES * 1000,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert score == pytest.approx(0.0)


def test_normalize_angle_negative_angle_behaves_like_positive(
    validator: HeadPoseValidator,
):
    """Verify that a negative angle is scored using its absolute value."""
    # Act
    score = validator._normalize_angle(
        angle=-HEAD_POSE_YAW_MAX_DEGREES / 2.0,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert score == pytest.approx(0.75)


@pytest.mark.parametrize(
    "magnitude",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(3.0, id="small"),
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES, id="at_threshold"),
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES * 3, id="large"),
    ],
)
def test_normalize_angle_symmetry(
    validator: HeadPoseValidator,
    magnitude: float,
):
    """Verify that normalize(+x) equals normalize(-x) for various magnitudes."""
    # Act
    positive_score = validator._normalize_angle(
        angle=magnitude,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )
    negative_score = validator._normalize_angle(
        angle=-magnitude,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    assert positive_score == pytest.approx(negative_score)


def test_normalize_angle_monotonically_decreases_with_magnitude(
    validator: HeadPoseValidator,
):
    """Verify that the score strictly decreases as the angle magnitude increases,
    up to the point where it clamps at 0.0."""
    # Arrange
    magnitudes = [0.0, 2.0, 5.0, 8.0, HEAD_POSE_YAW_MAX_DEGREES]

    # Act
    scores = [
        validator._normalize_angle(
            angle=magnitude,
            max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
        )
        for magnitude in magnitudes
    ]

    # Assert
    assert scores == sorted(
        scores,
        reverse=True,
    )
    assert len(set(scores)) == len(scores)


@pytest.mark.parametrize(
    "angle",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(1.0, id="small_positive"),
        pytest.param(-1.0, id="small_negative"),
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES, id="at_threshold"),
        pytest.param(HEAD_POSE_YAW_MAX_DEGREES * 50, id="huge_positive"),
        pytest.param(-HEAD_POSE_YAW_MAX_DEGREES * 50, id="huge_negative"),
    ],
)
def test_normalize_angle_never_exceeds_one_or_drops_below_zero(
    validator: HeadPoseValidator,
    angle: float,
):
    """Verify that the normalized score is always clamped within [0.0, 1.0]."""
    # Act
    score = validator._normalize_angle(
        angle=angle,
        max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
    )

    # Assert
    _assert_normalized_score(score)


# ================================================================== #
# _build_message()
# ================================================================== #


def test_build_message_all_valid(
    validator: HeadPoseValidator,
):
    """Verify the success message when every axis is within threshold."""
    # Act
    message = validator._build_message(
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        pitch_valid=True,
        yaw_valid=True,
        roll_valid=True,
    )

    # Assert
    assert message == "Head pose is acceptable."


def test_build_message_head_turned_right(
    validator: HeadPoseValidator,
):
    """Verify the message for a positive (rightward) yaw violation."""
    # Act
    message = validator._build_message(
        pitch=0.0,
        yaw=20.0,
        roll=0.0,
        pitch_valid=True,
        yaw_valid=False,
        roll_valid=True,
    )

    # Assert
    assert message == "Head is turned too far right."


def test_build_message_head_turned_left(
    validator: HeadPoseValidator,
):
    """Verify the message for a negative (leftward) yaw violation."""
    # Act
    message = validator._build_message(
        pitch=0.0,
        yaw=-20.0,
        roll=0.0,
        pitch_valid=True,
        yaw_valid=False,
        roll_valid=True,
    )

    # Assert
    assert message == "Head is turned too far left."


def test_build_message_head_tilted_upward(
    validator: HeadPoseValidator,
):
    """Verify the message for a negative pitch violation (tilted upward)."""
    # Act
    message = validator._build_message(
        pitch=-20.0,
        yaw=0.0,
        roll=0.0,
        pitch_valid=False,
        yaw_valid=True,
        roll_valid=True,
    )

    # Assert
    assert message == "Head is tilted too far upward."


def test_build_message_head_tilted_downward(
    validator: HeadPoseValidator,
):
    """Verify the message for a positive pitch violation (tilted downward)."""
    # Act
    message = validator._build_message(
        pitch=20.0,
        yaw=0.0,
        roll=0.0,
        pitch_valid=False,
        yaw_valid=True,
        roll_valid=True,
    )

    # Assert
    assert message == "Head is tilted too far downward."


@pytest.mark.parametrize(
    "roll",
    [
        pytest.param(20.0, id="positive_roll"),
        pytest.param(-20.0, id="negative_roll"),
    ],
)
def test_build_message_roll_exceeded(
    validator: HeadPoseValidator,
    roll: float,
):
    """Verify the roll-violation message, regardless of roll sign."""
    # Act
    message = validator._build_message(
        pitch=0.0,
        yaw=0.0,
        roll=roll,
        pitch_valid=True,
        yaw_valid=True,
        roll_valid=False,
    )

    # Assert
    assert message == "Head roll exceeds the acceptable limit."


def test_build_message_multiple_failures_are_combined_in_order(
    validator: HeadPoseValidator,
):
    """Verify that combined failures are concatenated in yaw, pitch, roll order."""
    # Act
    message = validator._build_message(
        pitch=20.0,
        yaw=20.0,
        roll=20.0,
        pitch_valid=False,
        yaw_valid=False,
        roll_valid=False,
    )

    # Assert
    assert message == (
        "Head is turned too far right. "
        "Head is tilted too far downward. "
        "Head roll exceeds the acceptable limit."
    )


def test_build_message_yaw_and_pitch_failures_preserve_order(
    validator: HeadPoseValidator,
):
    """Verify ordering when only yaw and pitch fail (yaw message precedes pitch)."""
    # Act
    message = validator._build_message(
        pitch=-20.0,
        yaw=-20.0,
        roll=0.0,
        pitch_valid=False,
        yaw_valid=False,
        roll_valid=True,
    )

    # Assert
    assert message == (
        "Head is turned too far left. Head is tilted too far upward."
    )