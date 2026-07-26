"""Tests for the ContrastValidator."""

import numpy as np
import pytest

from config.constants import (
    CONTRAST_MAX_EXPECTED_VALUE,
)
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.contrast_validator import ContrastValidator


def _create_grayscale_image(
    values: tuple[int, ...] | int,
    shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Create a deterministic grayscale image.

    Args:
        values: A single intensity or a tuple of two intensities for a
            horizontal stripe pattern.
        shape: Image dimensions as (height, width).

    Returns:
        A grayscale uint8 image.
    """
    if isinstance(
        values,
        int,
    ):
        return np.full(
            shape,
            values,
            dtype=np.uint8,
        )

    low, high = values
    image = np.empty(
        shape,
        dtype=np.uint8,
    )
    stripe_height = shape[0] // 2
    image[
        :stripe_height,
        :,
    ] = low
    image[
        stripe_height:,
        :,
    ] = high
    return image


def _create_bgr_image(
    values: tuple[int, ...] | int,
    shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Create a deterministic BGR image.

    Args:
        values: A single intensity or a tuple of two intensities for a
            horizontal stripe pattern.
        shape: Image dimensions as (height, width).

    Returns:
        A BGR uint8 image.
    """
    grayscale = _create_grayscale_image(
        values=values,
        shape=shape,
    )
    return np.stack(
        [
            grayscale,
            grayscale,
            grayscale,
        ],
        axis=-1,
    )


def _create_bgra_image(
    values: tuple[int, ...] | int,
    shape: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """Create a deterministic BGRA image.

    Args:
        values: A single intensity or a tuple of two intensities for a
            horizontal stripe pattern.
        shape: Image dimensions as (height, width).

    Returns:
        A BGRA uint8 image.
    """
    grayscale = _create_grayscale_image(
        values=values,
        shape=shape,
    )
    alpha = np.full(
        shape,
        255,
        dtype=np.uint8,
    )
    return np.stack(
        [
            grayscale,
            grayscale,
            grayscale,
            alpha,
        ],
        axis=-1,
    )


def _assert_normalized_score(
    score: float,
) -> None:
    """Assert that a validation score is normalized."""
    assert 0.0 <= score <= 1.0


def _expected_score(
    contrast: float,
) -> float:
    """Compute the expected quality score for a given contrast.

    Implements the same linear normalization as ContrastValidator:
    score = contrast / CONTRAST_MAX_EXPECTED_VALUE, clamped to [0.0, 1.0].
    """
    score = contrast / CONTRAST_MAX_EXPECTED_VALUE

    return max(
        min(
            score,
            1.0,
        ),
        0.0,
    )


@pytest.fixture
def validator() -> ContrastValidator:
    """Create a ContrastValidator instance."""
    return ContrastValidator()


# ------------------------------------------------------------------
# Validation Tests
# ------------------------------------------------------------------


def test_validate_none_image_raises_value_error(
    validator: ContrastValidator,
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
    validator: ContrastValidator,
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
    validator: ContrastValidator,
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
    validator: ContrastValidator,
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
    validator: ContrastValidator,
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


# ------------------------------------------------------------------
# Image Format Tests
# ------------------------------------------------------------------


def test_validate_grayscale_image_returns_validation_metric(
    validator: ContrastValidator,
):
    """Verify that grayscale image validation returns a metric."""
    # Arrange
    image = _create_grayscale_image(
        values=(0, 255),
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
    assert metric.type == ValidationType.CONTRAST
    assert metric.passed is True
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


def test_validate_bgr_image_returns_validation_metric(
    validator: ContrastValidator,
):
    """Verify that BGR image validation returns a metric."""
    # Arrange
    image = _create_bgr_image(
        values=(0, 255),
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
    assert metric.type == ValidationType.CONTRAST
    assert metric.passed is True
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


def test_validate_bgra_image_returns_validation_metric(
    validator: ContrastValidator,
):
    """Verify that BGRA image validation returns a metric."""
    # Arrange
    image = _create_bgra_image(
        values=(0, 255),
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
    assert metric.type == ValidationType.CONTRAST
    assert metric.passed is True
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )


# ------------------------------------------------------------------
# Contrast Threshold Tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected_passed, expected_message",
    [
        pytest.param(
            (0, 20),
            False,
            "Image contrast is too low.",
            id="very_low_contrast",
        ),
        pytest.param(
            (0, 60),
            True,
            "Image contrast is acceptable.",
            id="exactly_at_min_threshold",
        ),
        pytest.param(
            (0, 100),
            True,
            "Image contrast is acceptable.",
            id="moderate_contrast",
        ),
        pytest.param(
            (0, 200),
            True,
            "Image contrast is acceptable.",
            id="high_contrast",
        ),
        pytest.param(
            (0, 255),
            True,
            "Image contrast is acceptable.",
            id="very_high_contrast",
        ),
    ],
)
def test_validate_contrast_threshold(
    validator: ContrastValidator,
    values: tuple[int, int],
    expected_passed: bool,
    expected_message: str,
):
    """Verify pass/fail outcome and message for representative contrasts."""
    # Arrange
    image = _create_grayscale_image(
        values=values,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.type == ValidationType.CONTRAST
    assert metric.passed is expected_passed
    assert isinstance(
        metric.score,
        float,
    )
    _assert_normalized_score(
        metric.score,
    )
    assert metric.message == expected_message


# ------------------------------------------------------------------
# Synthetic Image / Score Tests
# ------------------------------------------------------------------


def test_validate_constant_image_produces_zero_contrast(
    validator: ContrastValidator,
):
    """Verify that a constant image yields a contrast score of 0.0."""
    # Arrange
    image = _create_grayscale_image(
        values=128,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        0.0,
    )


@pytest.mark.parametrize(
    "values",
    [
        pytest.param(
            (0, 20),
            id="very_low_contrast",
        ),
        pytest.param(
            (0, 60),
            id="min_threshold",
        ),
        pytest.param(
            (0, 100),
            id="moderate_contrast",
        ),
        pytest.param(
            (0, 160),
            id="high_contrast",
        ),
        pytest.param(
            (0, 200),
            id="max_expected_value",
        ),
        pytest.param(
            (0, 255),
            id="above_max_expected_value",
        ),
    ],
)
def test_validate_score_matches_expected_normalization(
    validator: ContrastValidator,
    values: tuple[int, int],
):
    """Verify that the score matches the expected linear formula."""
    # Arrange
    image = _create_grayscale_image(
        values=values,
    )
    expected_contrast = float(
        image.std(),
    )
    expected = _expected_score(
        expected_contrast,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.score == pytest.approx(
        expected,
    )


def test_validate_score_increases_monotonically_with_contrast(
    validator: ContrastValidator,
):
    """Verify that higher contrast always produces a higher or equal score."""
    # Arrange
    images = [
        _create_grayscale_image(values=128),
        _create_grayscale_image(values=(0, 20)),
        _create_grayscale_image(values=(0, 60)),
        _create_grayscale_image(values=(0, 100)),
        _create_grayscale_image(values=(0, 200)),
        _create_grayscale_image(values=(0, 255)),
    ]

    # Act
    scores = [
        validator.validate(image=image).score
        for image in images
    ]

    # Assert
    for i in range(
        len(scores) - 1,
    ):
        assert scores[i] <= scores[i + 1]


# ------------------------------------------------------------------
# Boundary Tests
# ------------------------------------------------------------------


def test_validate_contrast_below_threshold_fails(
    validator: ContrastValidator,
):
    """Verify that contrast below the minimum threshold fails."""
    # Arrange
    image = _create_grayscale_image(
        values=(0, 20),
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.passed is False
    assert metric.message == "Image contrast is too low."


def test_validate_contrast_exactly_at_threshold_passes(
    validator: ContrastValidator,
):
    """Verify that contrast exactly at the minimum threshold passes."""
    # Arrange
    image = _create_grayscale_image(
        values=(0, 60),
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.passed is True
    assert metric.message == "Image contrast is acceptable."


def test_validate_contrast_above_threshold_passes(
    validator: ContrastValidator,
):
    """Verify that contrast above the minimum threshold passes."""
    # Arrange
    image = _create_grayscale_image(
        values=(0, 100),
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.passed is True
    assert metric.message == "Image contrast is acceptable."


# ------------------------------------------------------------------
# Message Tests
# ------------------------------------------------------------------


def test_validate_pass_message(
    validator: ContrastValidator,
):
    """Verify the exact passed message."""
    # Arrange
    image = _create_grayscale_image(
        values=(0, 255),
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.message == "Image contrast is acceptable."


def test_validate_fail_message(
    validator: ContrastValidator,
):
    """Verify the exact failed message."""
    # Arrange
    image = _create_grayscale_image(
        values=128,
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    assert metric.message == "Image contrast is too low."


# ------------------------------------------------------------------
# Stripe Pattern Determinism Tests
# ------------------------------------------------------------------


def test_validate_stripe_pattern_produces_known_contrast(
    validator: ContrastValidator,
):
    """Verify that a simple stripe pattern produces a predictable std."""
    # Arrange
    image = _create_grayscale_image(
        values=(0, 60),
    )

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    expected_contrast = float(
        image.std(),
    )
    expected = _expected_score(expected_contrast)

    assert metric.score == pytest.approx(expected)
    assert metric.passed is True


def test_validate_checkerboard_pattern_high_contrast(
    validator: ContrastValidator,
):
    """Verify that a checkerboard of 0 and 255 yields high contrast."""
    # Arrange
    image = np.zeros(
        (
            64,
            64,
        ),
        dtype=np.uint8,
    )
    image[
        ::2,
        ::2,
    ] = 255
    image[
        1::2,
        1::2,
    ] = 255

    # Act
    metric = validator.validate(
        image=image,
    )

    # Assert
    expected_contrast = float(image.std())
    expected = _expected_score(expected_contrast)

    assert metric.score == pytest.approx(expected)