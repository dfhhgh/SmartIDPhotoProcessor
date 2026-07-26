"""Tests for the ValidationMetric model."""

import pytest

from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType


def test_create_validation_metric_success():
    """Verify that a valid validation metric is created successfully."""
    # Arrange
    validation_type = ValidationType.BLUR
    passed = True
    score = 0.75
    message = "Image is sharp enough."

    # Act
    metric = ValidationMetric(
        type=validation_type,
        passed=passed,
        score=score,
        message=message,
    )

    # Assert
    assert metric.type is validation_type
    assert metric.passed is passed
    assert metric.score == score
    assert metric.message == message


def test_invalid_validation_type_raises_type_error():
    """Verify that a non-ValidationType validation type raises TypeError."""
    # Arrange
    validation_type = "blur"

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationMetric(
            type=validation_type,
            passed=True,
            score=0.75,
            message="Image is sharp enough.",
        )


def test_invalid_passed_type_raises_type_error():
    """Verify that a non-bool passed value raises TypeError."""
    # Arrange
    passed = 1

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationMetric(
            type=ValidationType.BLUR,
            passed=passed,
            score=0.75,
            message="Image is sharp enough.",
        )


def test_bool_score_raises_type_error():
    """Verify that a bool score raises TypeError."""
    # Arrange
    score = True

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationMetric(
            type=ValidationType.BLUR,
            passed=True,
            score=score,
            message="Image is sharp enough.",
        )


def test_non_numeric_score_raises_type_error():
    """Verify that a non-numeric score raises TypeError."""
    # Arrange
    score = "0.75"

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationMetric(
            type=ValidationType.BLUR,
            passed=True,
            score=score,
            message="Image is sharp enough.",
        )


def test_score_below_zero_raises_value_error():
    """Verify that a score below zero raises ValueError."""
    # Arrange
    score = -0.01

    # Act & Assert
    with pytest.raises(ValueError):
        ValidationMetric(
            type=ValidationType.BLUR,
            passed=True,
            score=score,
            message="Image is sharp enough.",
        )


def test_score_above_one_raises_value_error():
    """Verify that a score above one raises ValueError."""
    # Arrange
    score = 1.01

    # Act & Assert
    with pytest.raises(ValueError):
        ValidationMetric(
            type=ValidationType.BLUR,
            passed=True,
            score=score,
            message="Image is sharp enough.",
        )


def test_invalid_message_type_raises_type_error():
    """Verify that a non-string message raises TypeError."""
    # Arrange
    message = 123

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationMetric(
            type=ValidationType.BLUR,
            passed=True,
            score=0.75,
            message=message,
        )


def test_message_is_trimmed():
    """Verify that leading and trailing whitespace is removed from message."""
    # Arrange
    message = "   blurry image   "

    # Act
    metric = ValidationMetric(
        type=ValidationType.BLUR,
        passed=False,
        score=0.25,
        message=message,
    )

    # Assert
    assert metric.message == "blurry image"


def test_score_is_converted_to_float():
    """Verify integer scores are converted to float."""

    metric = ValidationMetric(
        type=ValidationType.BLUR,
        passed=True,
        score=1,
    )

    assert isinstance(metric.score, float)
    assert metric.score == 1.0