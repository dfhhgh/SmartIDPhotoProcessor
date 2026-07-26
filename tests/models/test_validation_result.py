"""Tests for the ValidationResult model."""

import pytest

from models.validation_metric import ValidationMetric
from models.validation_result import ValidationResult
from models.validation_type import ValidationType


def _create_metric(
    validation_type: ValidationType = ValidationType.BLUR,
    passed: bool = True,
    score: float = 1.0,
    message: str = "Valid metric.",
) -> ValidationMetric:
    """Create a ValidationMetric for ValidationResult tests."""
    return ValidationMetric(
        type=validation_type,
        passed=passed,
        score=score,
        message=message,
    )


def test_create_validation_result_success():
    """Verify that a valid validation result is created successfully."""
    # Arrange
    metrics = [
        _create_metric(),
    ]

    # Act
    result = ValidationResult(
        metrics=metrics,
    )

    # Assert
    assert result.metrics == metrics
    assert result.metrics is metrics


def test_metrics_must_be_list():
    """Verify that metrics must be provided as a list."""
    # Arrange
    metrics = (
        _create_metric(),
    )

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationResult(
            metrics=metrics,
        )


def test_empty_metrics_list_raises_value_error():
    """Verify that an empty metrics list raises ValueError."""
    # Arrange
    metrics = []

    # Act & Assert
    with pytest.raises(ValueError):
        ValidationResult(
            metrics=metrics,
        )


def test_invalid_metric_type_raises_type_error():
    """Verify that every metrics item must be a ValidationMetric."""
    # Arrange
    metrics = [
        _create_metric(),
        object(),
    ]

    # Act & Assert
    with pytest.raises(TypeError):
        ValidationResult(
            metrics=metrics,
        )


def test_is_valid_returns_true_when_all_metrics_pass():
    """Verify that is_valid is True when all metrics pass."""
    # Arrange
    metrics = [
        _create_metric(
            validation_type=ValidationType.BLUR,
            passed=True,
        ),
        _create_metric(
            validation_type=ValidationType.BRIGHTNESS,
            passed=True,
        ),
    ]
    result = ValidationResult(
        metrics=metrics,
    )

    # Act
    is_valid = result.is_valid

    # Assert
    assert is_valid is True


def test_is_valid_returns_false_when_any_metric_fails():
    """Verify that is_valid is False when any metric fails."""
    # Arrange
    metrics = [
        _create_metric(
            validation_type=ValidationType.BLUR,
            passed=True,
        ),
        _create_metric(
            validation_type=ValidationType.BRIGHTNESS,
            passed=False,
            score=0.25,
        ),
    ]
    result = ValidationResult(
        metrics=metrics,
    )

    # Act
    is_valid = result.is_valid

    # Assert
    assert is_valid is False


def test_passed_metrics_returns_only_passed_metrics():
    """Verify that passed_metrics returns only passed metrics."""
    # Arrange
    passed_metric = _create_metric(
        validation_type=ValidationType.BLUR,
        passed=True,
    )
    failed_metric = _create_metric(
        validation_type=ValidationType.BRIGHTNESS,
        passed=False,
        score=0.25,
    )
    result = ValidationResult(
        metrics=[
            passed_metric,
            failed_metric,
        ],
    )

    # Act
    passed_metrics = result.passed_metrics

    # Assert
    assert len(passed_metrics) == 1
    assert passed_metrics == [
        passed_metric,
    ]
    assert failed_metric not in passed_metrics


def test_failed_metrics_returns_only_failed_metrics():
    """Verify that failed_metrics returns only failed metrics."""
    # Arrange
    passed_metric = _create_metric(
        validation_type=ValidationType.BLUR,
        passed=True,
    )
    failed_metric = _create_metric(
        validation_type=ValidationType.BRIGHTNESS,
        passed=False,
        score=0.25,
    )
    result = ValidationResult(
        metrics=[
            passed_metric,
            failed_metric,
        ],
    )

    # Act
    failed_metrics = result.failed_metrics

    # Assert
    assert len(failed_metrics) == 1
    assert failed_metrics == [
        failed_metric,
    ]
    assert passed_metric not in failed_metrics


def test_passed_metrics_returns_new_list():
    """Verify that passed_metrics returns a new list instance."""
    # Arrange
    passed_metric = _create_metric()
    result = ValidationResult(
        metrics=[
            passed_metric,
        ],
    )

    # Act
    passed_metrics = result.passed_metrics

    # Assert
    assert passed_metrics == [
        passed_metric,
    ]
    assert passed_metrics is not result.metrics


def test_failed_metrics_returns_new_list():
    """Verify that failed_metrics returns a new list instance."""
    # Arrange
    failed_metric = _create_metric(
        passed=False,
        score=0.25,
    )
    result = ValidationResult(
        metrics=[
            failed_metric,
        ],
    )

    # Act
    failed_metrics = result.failed_metrics

    # Assert
    assert failed_metrics == [
        failed_metric,
    ]
    assert failed_metrics is not result.metrics
