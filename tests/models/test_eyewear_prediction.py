"""Tests for the EyewearPrediction model."""

import pytest

from models.eyewear_prediction import EyewearPrediction
from models.eyewear_type import EyewearType


def test_create_valid_prediction():
    """Verify that a valid prediction is created successfully."""
    # Arrange
    eyewear_type = EyewearType.SUNGLASSES
    confidence = 0.95

    # Act
    prediction = EyewearPrediction(
        eyewear_type=eyewear_type,
        confidence=confidence,
    )

    # Assert
    assert prediction.eyewear_type is eyewear_type
    assert prediction.confidence == confidence


def test_integer_confidence_is_converted_to_float():
    """Verify that integer confidence is converted to float."""
    # Arrange
    confidence = 1

    # Act
    prediction = EyewearPrediction(
        eyewear_type=EyewearType.NONE,
        confidence=confidence,
    )

    # Assert
    assert isinstance(prediction.confidence, float)
    assert prediction.confidence == 1.0


def test_confidence_at_lower_boundary():
    """Verify that confidence of 0.0 is accepted."""
    # Act
    prediction = EyewearPrediction(
        eyewear_type=EyewearType.CLEAR_GLASSES,
        confidence=0.0,
    )

    # Assert
    assert prediction.confidence == 0.0


def test_confidence_at_upper_boundary():
    """Verify that confidence of 1.0 is accepted."""
    # Act
    prediction = EyewearPrediction(
        eyewear_type=EyewearType.PRESCRIPTION_GLASSES,
        confidence=1.0,
    )

    # Assert
    assert prediction.confidence == 1.0


def test_confidence_below_zero_raises_value_error():
    """Verify that a negative confidence raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError):
        EyewearPrediction(
            eyewear_type=EyewearType.NONE,
            confidence=-0.01,
        )


def test_confidence_above_one_raises_value_error():
    """Verify that confidence above 1.0 raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError):
        EyewearPrediction(
            eyewear_type=EyewearType.NONE,
            confidence=1.01,
        )


def test_bool_confidence_raises_type_error():
    """Verify that a boolean confidence raises TypeError."""
    # Act & Assert
    with pytest.raises(TypeError):
        EyewearPrediction(
            eyewear_type=EyewearType.NONE,
            confidence=True,
        )


def test_string_confidence_raises_type_error():
    """Verify that a string confidence raises TypeError."""
    # Act & Assert
    with pytest.raises(TypeError):
        EyewearPrediction(
            eyewear_type=EyewearType.NONE,
            confidence="0.8",
        )


def test_none_confidence_raises_type_error():
    """Verify that a None confidence raises TypeError."""
    # Act & Assert
    with pytest.raises(TypeError):
        EyewearPrediction(
            eyewear_type=EyewearType.NONE,
            confidence=None,
        )


def test_invalid_eyewear_type_raises_type_error():
    """Verify that a non-EyewearType eyewear_type raises TypeError."""
    # Act & Assert
    with pytest.raises(TypeError):
        EyewearPrediction(
            eyewear_type="sunglasses",
            confidence=0.9,
        )
