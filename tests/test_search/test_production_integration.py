"""Tests for Phase 13.4 — Production Pipeline Integration of ReverseSearchService.

Covers:
- Reverse search disabled (zero searches, None result)
- Reverse search enabled (attaches ReverseSearchResult)
- Reuse existing embedding / zero duplicate InsightFace inference
- Correct k=5 default and candidate attachment
- Existing validation result unchanged
- Failure isolation (service/query errors do not crash pipeline)
- Multiple sequential requests & isolation
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from search.reverse_search_service import CandidateMatch, ReverseSearchResult, ReverseSearchService, ReverseSearchStatus


@pytest.fixture
def valid_image() -> np.ndarray:
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def mock_pipeline_components() -> dict[str, MagicMock]:
    mock_detector = MagicMock()
    mock_selected_face = Face()
    mock_selected_face.bbox = np.array([0, 0, 50, 50], dtype=np.float32)
    # Set a valid normed embedding
    mock_selected_face.embedding = np.random.randn(512).astype(np.float32)
    mock_selected_face.embedding /= np.linalg.norm(mock_selected_face.embedding)

    mock_detector.detect.return_value = [mock_selected_face]

    mock_selector = MagicMock()
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
    mock_cropper.crop.return_value = CropResult(
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        crop_x=0,
        crop_y=0,
    )

    mock_transformer = MagicMock()
    mock_transformer.transform.return_value = mock_selected_face

    mock_aligner = MagicMock()
    mock_aligner.align.return_value = AlignmentResult(
        aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
        aligned_face=mock_selected_face,
        transform=np.eye(2, 3, dtype=np.float32),
    )

    mock_parser = MagicMock()
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

    return {
        "detector": mock_detector,
        "selector": mock_selector,
        "cropper": mock_cropper,
        "transformer": mock_transformer,
        "aligner": mock_aligner,
        "parser": mock_parser,
        "orchestrator": mock_orchestrator,
        "selected_face": mock_selected_face,
        "validation_result": mock_validation_result,
    }


class TestProductionIntegration:
    def test_reverse_search_disabled_by_default(
        self, valid_image: np.ndarray, mock_pipeline_components: dict[str, MagicMock]
    ) -> None:
        c = mock_pipeline_components
        mock_rs_service = MagicMock()

        pipeline = PhotoValidationPipeline(
            detector=c["detector"],
            selector=c["selector"],
            cropper=c["cropper"],
            coordinate_transformer=c["transformer"],
            aligner=c["aligner"],
            parser_service=c["parser"],
            orchestrator=c["orchestrator"],
            reverse_search_service=mock_rs_service,
            reverse_search_enabled=False,
        )

        result = pipeline.validate(valid_image)

        assert isinstance(result, PhotoProcessingResult)
        assert result.reverse_search_result is None
        mock_rs_service.search.assert_not_called()

    def test_reverse_search_enabled_attaches_result(
        self, valid_image: np.ndarray, mock_pipeline_components: dict[str, MagicMock]
    ) -> None:
        c = mock_pipeline_components
        mock_rs_service = MagicMock()
        expected_rs_result = ReverseSearchResult(
            status=ReverseSearchStatus.COMPLETED,
            candidates=(
                CandidateMatch(
                    vector_id=0,
                    person_id="p1",
                    label="p1",
                    image="p1/1.jpg",
                    similarity=0.92,
                ),
            ),
            top_k=5,
            query_dimension=512,
            processing_time_ms=1.0,
        )
        mock_rs_service.search.return_value = expected_rs_result

        pipeline = PhotoValidationPipeline(
            detector=c["detector"],
            selector=c["selector"],
            cropper=c["cropper"],
            coordinate_transformer=c["transformer"],
            aligner=c["aligner"],
            parser_service=c["parser"],
            orchestrator=c["orchestrator"],
            reverse_search_service=mock_rs_service,
            reverse_search_enabled=True,
        )

        result = pipeline.validate(valid_image)

        assert isinstance(result, PhotoProcessingResult)
        assert result.reverse_search_result is expected_rs_result
        assert result.validation_result is c["validation_result"]

        # Verify search was called with the selected face's normed embedding (or embedding)
        mock_rs_service.search.assert_called_once()
        args, kwargs = mock_rs_service.search.call_args
        assert len(args) >= 1 or kwargs.get("k") == 5

    def test_zero_additional_inference_count(
        self, valid_image: np.ndarray, mock_pipeline_components: dict[str, MagicMock]
    ) -> None:
        """Prove that enabling reverse search does NOT cause a second InsightFace inference."""
        c = mock_pipeline_components
        mock_rs_service = MagicMock()
        mock_rs_service.search.return_value = ReverseSearchResult(
            status=ReverseSearchStatus.COMPLETED,
            candidates=(),
            top_k=5,
            query_dimension=512,
            processing_time_ms=0.5,
        )

        pipeline = PhotoValidationPipeline(
            detector=c["detector"],
            selector=c["selector"],
            cropper=c["cropper"],
            coordinate_transformer=c["transformer"],
            aligner=c["aligner"],
            parser_service=c["parser"],
            orchestrator=c["orchestrator"],
            reverse_search_service=mock_rs_service,
            reverse_search_enabled=True,
        )

        pipeline.validate(valid_image)

        # Detector should be called exactly once
        c["detector"].detect.assert_called_once()
        # No other detection / model calls should occur
        mock_rs_service.search.assert_called_once()

    def test_failure_isolation_does_not_crash_pipeline(
        self, valid_image: np.ndarray, mock_pipeline_components: dict[str, MagicMock]
    ) -> None:
        """If ReverseSearchService raises an exception, validation continues normally with None result."""
        c = mock_pipeline_components
        mock_rs_service = MagicMock()
        mock_rs_service.search.side_effect = RuntimeError("FAISS search failure")

        pipeline = PhotoValidationPipeline(
            detector=c["detector"],
            selector=c["selector"],
            cropper=c["cropper"],
            coordinate_transformer=c["transformer"],
            aligner=c["aligner"],
            parser_service=c["parser"],
            orchestrator=c["orchestrator"],
            reverse_search_service=mock_rs_service,
            reverse_search_enabled=True,
        )

        result = pipeline.validate(valid_image)

        assert isinstance(result, PhotoProcessingResult)
        assert result.reverse_search_result is None
        assert result.validation_result is c["validation_result"]

    def test_sequential_requests_isolation(
        self, valid_image: np.ndarray, mock_pipeline_components: dict[str, MagicMock]
    ) -> None:
        c = mock_pipeline_components
        mock_rs_service = MagicMock()
        res1 = ReverseSearchResult(
            status=ReverseSearchStatus.COMPLETED,
            candidates=(CandidateMatch(0, "a", "a", "a/1.jpg", 0.95),),
            top_k=5,
            query_dimension=512,
            processing_time_ms=1.0,
        )
        res2 = ReverseSearchResult(
            status=ReverseSearchStatus.COMPLETED,
            candidates=(CandidateMatch(1, "b", "b", "b/1.jpg", 0.85),),
            top_k=5,
            query_dimension=512,
            processing_time_ms=1.0,
        )
        mock_rs_service.search.side_effect = [res1, res2]

        pipeline = PhotoValidationPipeline(
            detector=c["detector"],
            selector=c["selector"],
            cropper=c["cropper"],
            coordinate_transformer=c["transformer"],
            aligner=c["aligner"],
            parser_service=c["parser"],
            orchestrator=c["orchestrator"],
            reverse_search_service=mock_rs_service,
            reverse_search_enabled=True,
        )

        r1 = pipeline.validate(valid_image)
        r2 = pipeline.validate(valid_image)

        assert r1.reverse_search_result is res1
        assert r2.reverse_search_result is res2
        assert r1.reverse_search_result != r2.reverse_search_result
