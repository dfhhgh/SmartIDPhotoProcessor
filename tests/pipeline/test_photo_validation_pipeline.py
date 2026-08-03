"""Tests for PhotoValidationPipeline."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from insightface.app.common import Face
from models.alignment_result import AlignmentResult
from models.crop_result import CropResult
from models.photo_processing_result import PhotoProcessingResult
from models.ranked_face import RankedFace
from models.selection_result import SelectionResult
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
    mock_selection_result = SelectionResult(
        selected_face=mock_selected_face,
        selected_score=0.9,
        second_best_score=None,
        score_margin=0.9,
        ambiguity_ratio=0.0,
        detected_faces_count=1,
        ranked_faces=(RankedFace(face=mock_selected_face, score=0.9),),
    )
    mock_selector.select.return_value = mock_selection_result

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


def test_pipeline_does_not_mix_crop_space_face_with_aligned_image(valid_image):
    """Validation must receive the aligned face, not the crop-space face."""
    mock_detector = MagicMock()
    mock_faces = [MagicMock()]
    mock_detector.detect.return_value = mock_faces

    mock_selector = MagicMock()
    selected_face = Face()
    selected_face.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
    selection_result = SelectionResult(
        selected_face=selected_face,
        selected_score=0.9,
        second_best_score=None,
        score_margin=0.9,
        ambiguity_ratio=0.0,
        detected_faces_count=1,
        ranked_faces=(RankedFace(face=selected_face, score=0.9),),
    )
    mock_selector.select.return_value = selection_result

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

    mock_cropper.crop.assert_called_once_with(valid_image, selected_face)
    mock_transformer.transform.assert_called_once_with(selected_face, 5, 7)
    mock_aligner.align.assert_called_once_with(cropped_image, crop_space_face)
    mock_orchestrator.validate.assert_called_once_with(
        image=aligned_image,
        face=aligned_face,
        parsing_result=parsing_result,
        original_image=valid_image,
        original_face=selected_face,
    )


def test_pipeline_stops_before_crop_when_selection_is_ambiguous(valid_image):
    """Ambiguous SelectionResult must not reach Face-only pipeline stages."""
    mock_detector = MagicMock()
    mock_faces = [MagicMock(), MagicMock()]
    mock_detector.detect.return_value = mock_faces

    selected_face = Face()
    selected_face.bbox = np.array([10, 10, 90, 90], dtype=np.float32)
    runner_up_face = Face()
    runner_up_face.bbox = np.array([20, 20, 100, 100], dtype=np.float32)
    selection_result = SelectionResult(
        selected_face=selected_face,
        selected_score=0.80,
        second_best_score=0.75,
        score_margin=0.05,
        ambiguity_ratio=0.9375,
        detected_faces_count=2,
        ranked_faces=(
            RankedFace(face=selected_face, score=0.80),
            RankedFace(face=runner_up_face, score=0.75),
        ),
    )

    mock_selector = MagicMock()
    mock_selector.select.return_value = selection_result

    ambiguity_metric = ValidationMetric(
        type=ValidationType.FACE_AMBIGUITY,
        passed=False,
        score=0.1,
        message="Ambiguous face selection.",
    )
    mock_ambiguity_validator = MagicMock()
    mock_ambiguity_validator.validate.return_value = ambiguity_metric

    mock_cropper = MagicMock()
    mock_transformer = MagicMock()
    mock_aligner = MagicMock()
    mock_parser = MagicMock()
    mock_orchestrator = MagicMock()

    pipeline = PhotoValidationPipeline(
        detector=mock_detector,
        selector=mock_selector,
        cropper=mock_cropper,
        coordinate_transformer=mock_transformer,
        aligner=mock_aligner,
        parser_service=mock_parser,
        orchestrator=mock_orchestrator,
        ambiguity_validator=mock_ambiguity_validator,
    )

    result = pipeline.validate(valid_image)

    assert result.selected_face is selected_face
    assert result.aligned_image is None
    assert result.cropped_image is None
    assert result.validation_result.metrics == [ambiguity_metric]
    mock_ambiguity_validator.validate.assert_called_once_with(selection_result)
    mock_cropper.crop.assert_not_called()
    mock_transformer.transform.assert_not_called()
    mock_aligner.align.assert_not_called()
    mock_parser.parse.assert_not_called()
    mock_orchestrator.validate.assert_not_called()
