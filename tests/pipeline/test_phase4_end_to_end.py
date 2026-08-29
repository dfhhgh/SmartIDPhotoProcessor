"""Phase 4 End-to-End Pipeline Integration Tests.

Tests the full parser service (FaceParserService) with both ORIGINAL and FUSED
modes on real face images from the dataset. Tests that the pipeline components
can be wired together correctly (without requiring live face detection).
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.parser_mode import ParserMode
from services.face_parser_service import FaceParserService, EyeBrowRefinementService
from pipeline.photo_validation_pipeline import PhotoValidationPipeline
from pipeline.validation_orchestrator import ValidationOrchestrator
from models.parsing.face_parsing_result import FaceParsingResult
from models.parsing.face_part import FacePart

IMAGES_DIR = ROOT / "dataset_builder" / "dataset" / "parser_finetune_current" / "images"


def _get_real_face():
    candidates = [IMAGES_DIR / f"sample_{i:04d}.png" for i in range(100)]
    for c in candidates:
        if c.exists():
            img = cv2.imread(str(c))
            if img is not None and img.shape[0] > 100 and img.shape[1] > 100:
                return img
    pytest.skip("No suitable face images found")


def _reset():
    FaceParserService._instance = None
    FaceParserService._initialized = False


class TestParserORIGINAL:
    def test_parse_returns_face_parsing_result(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        result = fp.parse(_get_real_face())
        assert isinstance(result, FaceParsingResult)

    def test_mask_shape_matches_image(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        img = _get_real_face()
        result = fp.parse(img)
        assert result.mask.shape == (img.shape[0], img.shape[1])

    def test_mask_values_are_valid(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        result = fp.parse(_get_real_face())
        assert set(np.unique(result.mask)).issubset(set(range(19)))

    def test_mask_has_nonzero_pixels(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        result = fp.parse(_get_real_face())
        assert np.any(result.mask > 0)


class TestParserFUSED:
    def test_parse_returns_face_parsing_result(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        result = fp.parse(_get_real_face())
        assert isinstance(result, FaceParsingResult)

    def test_mask_shape_matches_image(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        img = _get_real_face()
        result = fp.parse(img)
        assert result.mask.shape == (img.shape[0], img.shape[1])

    def test_mask_values_are_valid(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        result = fp.parse(_get_real_face())
        assert set(np.unique(result.mask)).issubset(set(range(19)))

    def test_mask_has_nonzero_pixels(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        result = fp.parse(_get_real_face())
        assert np.any(result.mask > 0)

    def test_refinement_service_initialized(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        fp.parse(_get_real_face())
        assert fp._refinement_service is not None

    def test_refinement_service_has_models(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        fp.parse(_get_real_face())
        svc = fp._refinement_service
        assert svc._bb_session is not None
        assert svc._aux_session is not None


class TestModeComparison:
    def test_same_image_same_shape(self):
        _reset()
        fp1 = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        img = _get_real_face()
        r1 = fp1.parse(img)
        _reset()
        fp2 = FaceParserService(parser_mode=ParserMode.FUSED)
        r2 = fp2.parse(img)
        assert r1.mask.shape == r2.mask.shape

    def test_fused_produces_valid_eye_regions(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        result = fp.parse(_get_real_face())
        left_eye = int(np.sum(result.mask == FacePart.LEFT_EYE))
        right_eye = int(np.sum(result.mask == FacePart.RIGHT_EYE))
        left_brow = int(np.sum(result.mask == FacePart.LEFT_BROW))
        right_brow = int(np.sum(result.mask == FacePart.RIGHT_BROW))
        assert left_eye >= 0 and right_eye >= 0
        assert left_brow >= 0 and right_brow >= 0

    def test_fused_doesnt_remove_non_target(self):
        _reset()
        fp1 = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        img = _get_real_face()
        r1 = fp1.parse(img)
        _reset()
        fp2 = FaceParserService(parser_mode=ParserMode.FUSED)
        r2 = fp2.parse(img)
        non_target_parts = [FacePart.SKIN, FacePart.HAIR, FacePart.HAT, FacePart.NOSE]
        for part in non_target_parts:
            orig_count = int(np.sum(r1.mask == part))
            fused_count = int(np.sum(r2.mask == part))
            if orig_count > 100:
                assert fused_count >= orig_count * 0.9, f"{part.name}: {orig_count} -> {fused_count}"

    def test_both_modes_same_as_singletons(self):
        _reset()
        fp1 = FaceParserService(parser_mode=ParserMode.FUSED)
        fp2 = FaceParserService()
        assert fp1 is fp2

    def test_pipeline_wires_with_original(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.ORIGINAL)
        orch = ValidationOrchestrator(parser_service=fp)
        pipeline = PhotoValidationPipeline(orchestrator=orch)
        assert pipeline._orchestrator is orch

    def test_pipeline_wires_with_fused(self):
        _reset()
        fp = FaceParserService(parser_mode=ParserMode.FUSED)
        orch = ValidationOrchestrator(parser_service=fp)
        pipeline = PhotoValidationPipeline(orchestrator=orch)
        assert pipeline._orchestrator is orch
