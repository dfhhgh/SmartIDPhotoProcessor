"""Tests for PhotoValidationPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from insightface.app.common import Face
from models.alignment_result import AlignmentResult
from models.crop_result import CropResult
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
    mock_cropped_array = np.zeros((100, 100, 3), dtype=np.uint8)
    mock_crop_result = CropResult(image=mock_cropped_array, crop_x=10, crop_y=20)
    mock_cropper.crop.return_value = mock_crop_result

    mock_transformer = MagicMock()
    mock_transformed_face = Face()
    mock_transformer.transform.return_value = mock_transformed_face

    mock_aligner = MagicMock()
    mock_aligned_image = np.zeros((112, 112, 3), dtype=np.uint8)
    mock_aligned_face = Face()
    mock_aligned_face.bbox = np.array([20, 20, 80, 80], dtype=np.float32)
    mock_alignment_result = AlignmentResult(
        aligned_image=mock_aligned_image,
        aligned_face=mock_aligned_face,
        transform=np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
            ],
            dtype=np.float32,
        ),
    )
    mock_aligner.align.return_value = mock_alignment_result

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
        coordinate_transformer=mock_transformer,
        aligner=mock_aligner,
        parser_service=mock_parser,
        orchestrator=mock_orchestrator,
    )

    result = pipeline.validate(valid_image)

    assert isinstance(result, PhotoProcessingResult)
    assert result.validation_result is mock_validation_result
    assert result.selected_face is mock_selected_face
    assert result.aligned_image is mock_aligned_image
    assert result.cropped_image is mock_cropped_array

    mock_detector.detect.assert_called_once_with(valid_image)
    mock_selector.select.assert_called_once_with(mock_faces, valid_image.shape)
    mock_cropper.crop.assert_called_once_with(valid_image, mock_selected_face)
    mock_transformer.transform.assert_called_once_with(
        mock_selected_face,
        mock_crop_result.crop_x,
        mock_crop_result.crop_y,
    )
    mock_aligner.align.assert_called_once_with(mock_cropped_array, mock_transformed_face)
    mock_parser.parse.assert_called_once_with(mock_aligned_image)
    mock_orchestrator.validate.assert_called_once_with(
        image=mock_aligned_image,
        face=mock_aligned_face,
        parsing_result=mock_parsing_result,
    )


def test_pipeline_does_not_mix_crop_space_face_with_aligned_image(valid_image):
    """Validation must receive the aligned face, not the crop-space face."""
    mock_detector = MagicMock()
    mock_faces = [MagicMock()]
    mock_detector.detect.return_value = mock_faces

    mock_selector = MagicMock()
    selected_face = Face()
    selected_face.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
    mock_selector.select.return_value = selected_face

    mock_cropper = MagicMock()
    cropped_image = np.zeros((120, 120, 3), dtype=np.uint8)
    mock_cropper.crop.return_value = CropResult(
        image=cropped_image,
        crop_x=5,
        crop_y=7,
    )

    mock_transformer = MagicMock()
    crop_space_face = Face()
    crop_space_face.bbox = np.array([5, 3, 85, 83], dtype=np.float32)
    mock_transformer.transform.return_value = crop_space_face

    aligned_image = np.zeros((112, 112, 3), dtype=np.uint8)
    aligned_face = Face()
    aligned_face.bbox = np.array([24, 20, 88, 92], dtype=np.float32)
    alignment_result = AlignmentResult(
        aligned_image=aligned_image,
        aligned_face=aligned_face,
        transform=np.array(
            [
                [0.8, 0, 20],
                [0, 0.8, 18],
            ],
            dtype=np.float32,
        ),
    )
    mock_aligner = MagicMock()
    mock_aligner.align.return_value = alignment_result

    mock_parser = MagicMock()
    parsing_result = MagicMock()
    mock_parser.parse.return_value = parsing_result

    validation_result = ValidationResult(
        metrics=[
            ValidationMetric(
                type=ValidationType.BLUR,
                passed=True,
                score=1.0,
            )
        ]
    )
    mock_orchestrator = MagicMock()
    mock_orchestrator.validate.return_value = validation_result

    pipeline = PhotoValidationPipeline(
        detector=mock_detector,
        selector=mock_selector,
        cropper=mock_cropper,
        coordinate_transformer=mock_transformer,
        aligner=mock_aligner,
        parser_service=mock_parser,
        orchestrator=mock_orchestrator,
    )

    pipeline.validate(valid_image)

    mock_orchestrator.validate.assert_called_once_with(
        image=aligned_image,
        face=aligned_face,
        parsing_result=parsing_result,
    )
    assert mock_orchestrator.validate.call_args.kwargs["face"] is not crop_space_face
