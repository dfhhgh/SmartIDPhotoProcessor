"""Tests for PhotoValidationPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from insightface.app.common import Face
from models.photo_processing_result import PhotoProcessingResult
from models.validation_metric import ValidationMetric
from models.validation_result import ValidationResult
from models.validation_type import ValidationType
from pipeline.photo_validation_pipeline import PhotoValidationPipeline


@pytest.fixture
def valid_image() -> np.ndarray:
    """Create a valid test image."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


def test_pipeline_execution_flow(valid_image):
    """Verify that validate() coordinates all components in the correct sequence."""
    mock_detector = MagicMock()
    mock_faces = [MagicMock()]
    mock_detector.detect.return_value = mock_faces

    mock_selector = MagicMock()
    mock_selected_face = Face()
    mock_selected_face.bbox = np.array([0, 0, 10, 10], dtype=np.float32)
    mock_selector.select.return_value = mock_selected_face

    mock_cropper = MagicMock()
    mock_cropped_image = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_cropper.crop.return_value = mock_cropped_image

    mock_aligner = MagicMock()
    mock_aligned_image = np.zeros((112, 112, 3), dtype=np.uint8)
    mock_aligner.align.return_value = mock_aligned_image

    mock_parser = MagicMock()
    mock_parsing_result = MagicMock()
    mock_parser.parse.return_value = mock_parsing_result

    mock_orchestrator = MagicMock()
    mock_validation_result = ValidationResult(
        metrics=[
            ValidationMetric(
                type=ValidationType.BLUR,
                passed=True,
                score=1.0,
                message="ok",
            )
        ]
    )
    mock_orchestrator.validate.return_value = mock_validation_result

    pipeline = PhotoValidationPipeline(
        detector=mock_detector,
        selector=mock_selector,
        cropper=mock_cropper,
        aligner=mock_aligner,
        parser_service=mock_parser,
        orchestrator=mock_orchestrator,
    )

    result = pipeline.validate(valid_image)

    assert isinstance(result, PhotoProcessingResult)
    assert result.validation_result is mock_validation_result
    assert result.selected_face is mock_selected_face
    assert result.aligned_image is mock_aligned_image
    assert result.cropped_image is mock_cropped_image

    mock_detector.detect.assert_called_once_with(valid_image)
    mock_selector.select.assert_called_once_with(mock_faces, valid_image.shape)
    mock_aligner.align.assert_called_once_with(valid_image, mock_selected_face)
    mock_parser.parse.assert_called_once_with(mock_aligned_image)
    mock_orchestrator.validate.assert_called_once_with(
        image=mock_aligned_image,
        face=mock_selected_face,
        parsing_result=mock_parsing_result,
    )
