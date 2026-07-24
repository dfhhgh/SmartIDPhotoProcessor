import numpy as np
import pytest

from unittest.mock import MagicMock, patch

from pipeline.detector import FaceDetector
from exceptions.face_exceptions import FaceDetectionError

@pytest.fixture
def detector():
    """
    Create a FaceDetector instance with a mocked InsightFace model.
    """

    mock_model = MagicMock()

    with patch(
        "pipeline.detector.FaceService.get_model",
        return_value=mock_model,
    ):
        detector = FaceDetector()

    return detector, mock_model


def test_detect_raises_value_error_when_image_is_none(detector):
    """
    Test that detect() raises ValueError when the input image is None.
    """

    face_detector, _ = detector

    with pytest.raises(
        ValueError,
        match="Input image cannot be None.",
    ):
        face_detector.detect(None)

def test_detect_raises_type_error_for_invalid_input_type(detector):
    """
    Test that detect() raises TypeError when the input is not a NumPy array.
    """

    face_detector, _ = detector

    with pytest.raises(
        TypeError,
        match="Input image must be a NumPy ndarray.",
    ):
        face_detector.detect("image.jpg")


def test_detect_raises_value_error_for_empty_array(detector):
    """
    Test that detect() raises ValueError when the input image is empty.
    """

    face_detector, _ = detector

    empty_image = np.array([])

    with pytest.raises(
        ValueError,
        match="Input image cannot be empty.",
    ):
        face_detector.detect(empty_image)


def test_detect_raises_value_error_for_invalid_image_dimensions(detector):
    """
    Test that detect() raises ValueError when the image has invalid dimensions.
    """

    face_detector, _ = detector

    invalid_image = np.zeros((1, 2, 3, 4), dtype=np.uint8)

    with pytest.raises(
        ValueError,
        match="Input image must have 2 or 3 dimensions.",
    ):
        face_detector.detect(invalid_image)

def test_detect_returns_detected_faces(detector):
    """
    Test that detect() returns the faces detected by the model.
    """

    face_detector, mock_model = detector

    image = np.zeros((640, 640, 3), dtype=np.uint8)

    fake_faces = [MagicMock(), MagicMock()]

    mock_model.get.return_value = fake_faces

    result = face_detector.detect(image)

    assert result == fake_faces
    mock_model.get.assert_called_once_with(image)


def test_detect_raises_face_detection_error_when_model_fails(detector):
    """
    Test that detect() raises FaceDetectionError
    when the underlying model fails.
    """

    face_detector, mock_model = detector

    image = np.zeros((640, 640, 3), dtype=np.uint8)

    mock_model.get.side_effect = RuntimeError("ONNX Runtime failed")

    with pytest.raises(
        FaceDetectionError,
        match="Failed to detect faces.",
    ):
        face_detector.detect(image)

    mock_model.get.assert_called_once_with(image)