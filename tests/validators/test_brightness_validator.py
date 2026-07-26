"""Tests for the BrightnessValidator."""

import numpy as np
import pytest

from config.constants import (
    BRIGHTNESS_MAX_THRESHOLD,
    BRIGHTNESS_MIN_THRESHOLD,
)
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.brightness_validator import BrightnessValidator

_BRIGHTNESS_MIDPOINT = (
    BRIGHTNESS_MIN_THRESHOLD + BRIGHTNESS_MAX_THRESHOLD
) / 2.0

_BRIGHTNESS_HALF_RANGE = (
    BRIGHTNESS_MAX_THRESHOLD - BRIGHTNESS_MIN_THRESHOLD
) / 2.0


def _create_grayscale_image(
    intensity: int,
) -> np.ndarray:
    """Create a deterministic constant grayscale image."""
    return np.full(
        (
            64,
            64,
        ),
        intensity,
        dtype=np.uint8,
    )


def _create_bgr_image(
    intensity: int,
) -> np.ndarray:
    """Create a deterministic constant BGR image."""
    return np.full(
        (
            64,
            64,
            3,
        ),
        intensity,
        dtype=np.uint8,
    )


def _create_bgra_image(
    intensity: int,
) -> np.ndarray:
    """Create a deterministic constant BGRA image."""
    return np.full(
        (
            64,
            64,
            4,
        ),
        intensity,
        dtype=np.uint8,
    )


def _assert_normalized_score(score: float) -> None:
    """Assert that a validation score is normalized."""
    assert 0.0 <= score <= 1.0


def _expected_score(
    intensity: float,
) -> float:
    """Compute the expected quality score for a given intensity."""
    distance = abs(intensity - _BRIGHTNESS_MIDPOINT)
    score = 1.0 - 0.5 * (distance / _BRIGHTNESS_HALF_RANGE)

    return max(
        min(
            score,
            1.0,
        ),
        0.0,
    )


@pytest.fixture
def validator() -> BrightnessValidator:
    """Create a BrightnessValidator instance."""
    return BrightnessValidator()


def test_validate_none_image_raises_value_error(
    validator: BrightnessValidator,
):
    """Verify that None image input raises ValueError."""
    # Arrange
    image = None

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
        )


def test_validate_non_numpy_image_raises_type_error(
    validator: BrightnessValidator,
):
    """Verify that non-NumPy image input raises TypeError."""
    # Arrange
    image = [
        [
            0,
            255,
        ],
    ]

    # Act & Assert
    with pytest.raises(TypeError):
        validator.validate(
            image=image,
        )


def test_validate_empty_image_raises_value_error(
    validator: BrightnessValidator,
):
    """Verify that an empty NumPy image raises ValueError."""
    # Arrange
    image = np.array(
        [],
        dtype=np.uint8,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
        )


def test_validate_unsupported_image_dimensions_raises_value_error(
    validator: BrightnessValidator,
):
    """Verify that unsupported image dimensions raise ValueError."""
    # Arrange
    image = np.zeros(
        (
            16,
            16,
            3,
            1,
        ),
        dtype=np.uint8,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
        )


def test_validate_unsupported_channel_count_raises_value_error(
    validator: BrightnessValidator,
):
    """Verify that unsupported channel counts raise ValueError."""
    # Arrange
    image = np.zeros(
        (
            16,
            16,
            2,
        ),
        dtype=np.uint8,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
        )


def test_validate_grayscale_image_returns_validation_metric(
    validator: BrightnessValidator,
):
    """Verify that grayscale image validation returns a metric."""
    # Arrange
    image = _create_grayscale_image(
        intensity=130,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.BRIGHTNESS
    assert metric.passed is True
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


def test_validate_bgr_image_returns_validation_metric(
    validator: BrightnessValidator,
):
    """Verify that BGR image validation returns a metric."""
    # Arrange
    image = _create_bgr_image(
        intensity=130,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.BRIGHTNESS
    assert metric.passed is True
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


def test_validate_bgra_image_returns_validation_metric(
    validator: BrightnessValidator,
):
    """Verify that BGRA image validation returns a metric."""
    # Arrange
    image = _create_bgra_image(
        intensity=130,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.BRIGHTNESS
    assert metric.passed is True
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


@pytest.mark.parametrize(
    "intensity, expected_passed, expected_message",
    [
        pytest.param(
            10,
            False,
            "Image is too dark.",
            id="darker_than_min",
        ),
        pytest.param(
            int(BRIGHTNESS_MIN_THRESHOLD),
            True,
            "Image brightness is acceptable.",
            id="at_min_threshold",
        ),
        pytest.param(
            int(_BRIGHTNESS_MIDPOINT),
            True,
            "Image brightness is acceptable.",
            id="at_midpoint",
        ),
        pytest.param(
            int(BRIGHTNESS_MAX_THRESHOLD),
            True,
            "Image brightness is acceptable.",
            id="at_max_threshold",
        ),
        pytest.param(
            250,
            False,
            "Image is too bright.",
            id="brighter_than_max",
        ),
    ],
)
def test_validate_brightness_threshold(
    validator: BrightnessValidator,
    intensity: int,
    expected_passed: bool,
    expected_message: str,
):
    """Verify pass/fail outcome and message for representative intensities."""
    # Arrange
    image = _create_grayscale_image(
        intensity=intensity,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.type == ValidationType.BRIGHTNESS
    assert metric.passed is expected_passed
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )
    assert metric.message == expected_message


@pytest.mark.parametrize(
    "intensity",
    [
        pytest.param(0, id="black"),
        pytest.param(20, id="very_dark"),
        pytest.param(40, id="min_threshold"),
        pytest.param(80, id="below_midpoint"),
        pytest.param(100, id="lower_acceptable"),
        pytest.param(130, id="midpoint"),
        pytest.param(160, id="upper_acceptable"),
        pytest.param(180, id="above_midpoint"),
        pytest.param(220, id="max_threshold"),
        pytest.param(240, id="very_bright"),
        pytest.param(255, id="white"),
    ],
)
def test_validate_score_is_normalized(
    validator: BrightnessValidator,
    intensity: int,
):
    """Verify that the score is within [0.0, 1.0] across the full range."""
    # Arrange
    image = _create_grayscale_image(
        intensity=intensity,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    _assert_normalized_score(
        metric.score,
    )


@pytest.mark.parametrize(
    "intensity",
    [
        pytest.param(10, id="very_dark"),
        pytest.param(40, id="min_threshold"),
        pytest.param(80, id="below_midpoint"),
        pytest.param(100, id="lower_acceptable"),
        pytest.param(130, id="midpoint"),
        pytest.param(160, id="upper_acceptable"),
        pytest.param(180, id="above_midpoint"),
        pytest.param(220, id="max_threshold"),
        pytest.param(250, id="very_bright"),
    ],
)
def test_validate_score_matches_expected_calculation(
    validator: BrightnessValidator,
    intensity: int,
):
    """Verify that the score matches the expected linear formula."""
    # Arrange
    image = _create_grayscale_image(
        intensity=intensity,
    )
    expected = _expected_score(
        intensity,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.score == pytest.approx(
        expected,
    )


def test_validate_midpoint_brightness_produces_highest_score(
    validator: BrightnessValidator,
):
    """Verify that midpoint brightness yields the highest quality score."""
    # Arrange
    midpoint_image = _create_grayscale_image(
        intensity=int(_BRIGHTNESS_MIDPOINT),
    )
    darker_image = _create_grayscale_image(
        intensity=int(BRIGHTNESS_MIN_THRESHOLD),
    )
    brighter_image = _create_grayscale_image(
        intensity=int(BRIGHTNESS_MAX_THRESHOLD),
    )

    # Act
    midpoint_metric = validator.validate(
        image=midpoint_image,
    )
    darker_metric = validator.validate(
        image=darker_image,
    )
    brighter_metric = validator.validate(
        image=brighter_image,
    )

    # Assert
    assert midpoint_metric.score > darker_metric.score
    assert midpoint_metric.score > brighter_metric.score
    assert midpoint_metric.score == pytest.approx(
        1.0,
    )


def test_validate_boundary_brightness_produces_half_score(
    validator: BrightnessValidator,
):
    """Verify that boundary brightness yields a score of approximately 0.5."""
    # Arrange
    min_image = _create_grayscale_image(
        intensity=int(BRIGHTNESS_MIN_THRESHOLD),
    )
    max_image = _create_grayscale_image(
        intensity=int(BRIGHTNESS_MAX_THRESHOLD),
    )

    # Act
    min_metric = validator.validate(
        image=min_image,
    )
    max_metric = validator.validate(
        image=max_image,
    )

    # Assert
    assert min_metric.score == pytest.approx(
        0.5,
    )
    assert max_metric.score == pytest.approx(
        0.5,
    )


@pytest.mark.parametrize(
    "intensity",
    [
        pytest.param(0, id="black"),
        pytest.param(10, id="very_dark"),
        pytest.param(250, id="very_bright"),
        pytest.param(255, id="white"),
    ],
)
def test_validate_out_of_range_score_below_half(
    validator: BrightnessValidator,
    intensity: int,
):
    """Verify that out-of-range images produce a score below 0.5."""
    # Arrange
    image = _create_grayscale_image(
        intensity=intensity,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.score < 0.5


def test_validate_score_type_is_float(
    validator: BrightnessValidator,
):
    """Verify that the returned score is a native Python float."""
    # Arrange
    image = _create_grayscale_image(
        intensity=130,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert isinstance(
        metric.score,
        float,
    )
