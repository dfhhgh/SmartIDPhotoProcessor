"""Tests for the BlurValidator."""

import cv2
import numpy as np
import pytest

from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.blur_validator import BlurValidator


def _create_sharp_grayscale_image() -> np.ndarray:
    """Create a deterministic sharp checkerboard image."""
    image = np.zeros(
        (
            128,
            128,
        ),
        dtype=np.uint8,
    )

    for row in range(
        0,
        image.shape[0],
        16,
    ):
        for column in range(
            0,
            image.shape[1],
            16,
        ):
            if (row // 16 + column // 16) % 2 == 0:
                image[
                    row:row + 16,
                    column:column + 16,
                ] = 255

    return image


def _create_blurry_grayscale_image() -> np.ndarray:
    """Create a deterministic heavily blurred checkerboard image."""
    return cv2.GaussianBlur(
        _create_sharp_grayscale_image(),
        (
            31,
            31,
        ),
        10,
    )


def _assert_normalized_score(score: float) -> None:
    """Assert that a validation score is normalized."""
    assert 0.0 <= score <= 1.0


@pytest.fixture
def validator() -> BlurValidator:
    """Create a BlurValidator instance."""
    return BlurValidator()


def test_validate_none_image_raises_value_error(
    validator: BlurValidator,
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
    validator: BlurValidator,
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
    validator: BlurValidator,
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
    validator: BlurValidator,
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
    validator: BlurValidator,
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
    validator: BlurValidator,
):
    """Verify that grayscale image validation returns a metric."""
    # Arrange
    image = _create_sharp_grayscale_image()

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.BLUR
    _assert_normalized_score(
        metric.score,
    )


def test_validate_bgr_image_returns_validation_metric(
    validator: BlurValidator,
):
    """Verify that BGR image validation returns a metric."""
    # Arrange
    image = cv2.cvtColor(
        _create_sharp_grayscale_image(),
        cv2.COLOR_GRAY2BGR,
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
    assert metric.type == ValidationType.BLUR
    _assert_normalized_score(
        metric.score,
    )


def test_validate_bgra_image_returns_validation_metric(
    validator: BlurValidator,
):
    """Verify that BGRA image validation returns a metric."""
    # Arrange
    image = cv2.cvtColor(
        _create_sharp_grayscale_image(),
        cv2.COLOR_GRAY2BGRA,
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
    assert metric.type == ValidationType.BLUR
    _assert_normalized_score(
        metric.score,
    )


def test_validate_sharp_image_passes(
    validator: BlurValidator,
):
    """Verify that a sufficiently sharp image passes blur validation."""
    # Arrange
    image = _create_sharp_grayscale_image()

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.type == ValidationType.BLUR
    assert metric.passed is True
    _assert_normalized_score(
        metric.score,
    )
    assert metric.message == "Image sharpness is acceptable."


def test_validate_blurry_image_fails(
    validator: BlurValidator,
):
    """Verify that a heavily blurred image fails blur validation."""
    # Arrange
    image = _create_blurry_grayscale_image()

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.type == ValidationType.BLUR
    assert metric.passed is False
    _assert_normalized_score(
        metric.score,
    )
    assert metric.message == "Image is too blurry for reliable processing."


def test_validate_sharp_image_score_is_normalized(
    validator: BlurValidator,
):
    """Verify that sharp image validation returns a normalized score."""
    # Arrange
    image = _create_sharp_grayscale_image()

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    _assert_normalized_score(
        metric.score,
    )


def test_validate_blurry_image_score_is_normalized(
    validator: BlurValidator,
):
    """Verify that blurry image validation returns a normalized score."""
    # Arrange
    image = _create_blurry_grayscale_image()

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    _assert_normalized_score(
        metric.score,
    )
