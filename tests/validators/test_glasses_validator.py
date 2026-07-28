"""Tests for the GlassesValidator."""

from unittest.mock import Mock

import numpy as np
import pytest
from insightface.app.common import Face

from config.constants import (
    GLASSES_FAILURE_MESSAGE,
    GLASSES_SUCCESS_MESSAGE,
)
from models.eyewear_prediction import EyewearPrediction
from models.eyewear_type import EyewearType
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from services.eyewear_classifier import EyewearClassifier
from validators.glasses_validator import GlassesValidator


def _create_valid_image() -> np.ndarray:
    """Create a valid test image."""
    return np.zeros(
        (
            100,
            100,
            3,
        ),
        dtype=np.uint8,
    )


def _create_mock_face() -> Mock:
    """Create a mock Face object."""
    return Mock(spec=Face)


def _create_mock_classifier(
    eyewear_type: EyewearType,
    confidence: float,
) -> Mock:
    """Create a mock EyewearClassifier returning the specified prediction."""
    mock_classifier = Mock(
        spec=EyewearClassifier,
    )
    mock_classifier.classify.return_value = EyewearPrediction(
        eyewear_type=eyewear_type,
        confidence=confidence,
    )
    return mock_classifier


@pytest.fixture
def mock_face() -> Mock:
    """Create a mock Face for testing."""
    return _create_mock_face()


@pytest.fixture
def image() -> np.ndarray:
    """Create a valid test image."""
    return _create_valid_image()


@pytest.fixture
def validator() -> GlassesValidator:
    """Create a GlassesValidator with a mock classifier."""
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    return GlassesValidator(
        classifier=mock_classifier,
    )


def test_none_eyewear_is_accepted(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that EyewearType.NONE is accepted."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=0.95,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    assert metric.type == ValidationType.GLASSES
    assert metric.passed is True
    assert metric.score == pytest.approx(
        0.95,
    )
    assert metric.message == GLASSES_SUCCESS_MESSAGE


def test_clear_glasses_are_accepted(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that EyewearType.CLEAR_GLASSES is accepted."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.CLEAR_GLASSES,
        confidence=0.88,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    assert metric.type == ValidationType.GLASSES
    assert metric.passed is True
    assert metric.score == pytest.approx(
        0.88,
    )
    assert metric.message == GLASSES_SUCCESS_MESSAGE


def test_prescription_glasses_are_accepted(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that EyewearType.PRESCRIPTION_GLASSES is accepted."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.PRESCRIPTION_GLASSES,
        confidence=0.92,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    assert metric.type == ValidationType.GLASSES
    assert metric.passed is True
    assert metric.score == pytest.approx(
        0.92,
    )
    assert metric.message == GLASSES_SUCCESS_MESSAGE


def test_sunglasses_are_rejected(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that EyewearType.SUNGLASSES is rejected."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.SUNGLASSES,
        confidence=0.97,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    assert metric.type == ValidationType.GLASSES
    assert metric.passed is False
    assert metric.score == pytest.approx(
        0.97,
    )
    assert metric.message == GLASSES_FAILURE_MESSAGE


def test_classifier_is_called_once(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that classify() is called exactly once with correct arguments."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    mock_classifier.classify.assert_called_once_with(
        image=image,
        face=mock_face,
    )


def test_validate_none_image_raises_value_error(
    mock_face: Mock,
):
    """Verify that None image input raises ValueError."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )
    image = None

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=mock_face,
        )


def test_validate_non_numpy_image_raises_type_error(
    mock_face: Mock,
):
    """Verify that non-NumPy image input raises TypeError."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )
    image = "not_an_image"

    # Act & Assert
    with pytest.raises(TypeError):
        validator.validate(
            image=image,
            face=mock_face,
        )


def test_validate_empty_image_raises_value_error(
    mock_face: Mock,
):
    """Verify that an empty NumPy image raises ValueError."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )
    image = np.array(
        [],
        dtype=np.uint8,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=mock_face,
        )


def test_validate_none_face_raises_value_error(
    image: np.ndarray,
):
    """Verify that None face input raises ValueError."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )
    face = None

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            face=face,
        )


def test_validate_invalid_face_type_raises_type_error(
    image: np.ndarray,
):
    """Verify that invalid face type raises TypeError."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.NONE,
        confidence=1.0,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )
    face = "not_a_face"

    # Act & Assert
    with pytest.raises(TypeError):
        validator.validate(
            image=image,
            face=face,
        )


def test_constructor_rejects_non_classifier_dependency():
    """Verify that constructor rejects non-EyewearClassifier dependencies."""
    # Arrange
    invalid_classifier = "not_a_classifier"

    # Act & Assert
    with pytest.raises(TypeError):
        GlassesValidator(
            classifier=invalid_classifier,
        )


def test_metric_always_uses_glasses_type_for_success(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that ValidationType.GLASSES is used for success cases."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.CLEAR_GLASSES,
        confidence=0.90,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    assert metric.type == ValidationType.GLASSES


def test_metric_always_uses_glasses_type_for_failure(
    image: np.ndarray,
    mock_face: Mock,
):
    """Verify that ValidationType.GLASSES is used for failure cases."""
    # Arrange
    mock_classifier = _create_mock_classifier(
        eyewear_type=EyewearType.SUNGLASSES,
        confidence=0.95,
    )
    validator = GlassesValidator(
        classifier=mock_classifier,
    )

    # Act
    metric = validator.validate(
        image=image,
        face=mock_face,
    )

    # Assert
    assert metric.type == ValidationType.GLASSES
