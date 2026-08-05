"""
Regression tests for Phase 1 semantic evidence fusion enhancements.
"""

from __future__ import annotations

import numpy as np
import pytest
import types

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from reasoning.semantic_engine import SemanticEvidenceEngine, SemanticEvidence


def _make_parsing_result(parts: dict[FacePart, int], height: int = 100, width: int = 100) -> FaceParsingResult:
    mask = np.zeros((height, width), dtype=np.int32)
    flat = mask.ravel()
    cursor = 0
    for part, count in parts.items():
        flat[cursor:cursor + count] = int(part)
        cursor += count
    mask = flat.reshape((height, width))
    return FaceParsingResult(mask=mask, image_height=height, image_width=width)


class TestPhase1Regression:
    def test_eyebrow_false_negative_blending(self) -> None:
        # Parser misses eyebrow (0 pixels), but eye is present with sufficient pixels,
        # valid landmark, and frontal pose. Confidence should blend above threshold.
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 50})
        face = types.SimpleNamespace()
        face.kps = np.array([[10.0, 10.0], [20.0, 20.0], [0,0], [0,0], [0,0]], dtype=np.float32)
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE) is True

    def test_closed_mouth_lips_blending(self) -> None:
        # Parser misses inner MOUTH, but predicts upper and lower lips sufficiently.
        parsing_result = _make_parsing_result({
            FacePart.UPPER_LIP: 30,
            FacePart.LOWER_LIP: 30,
        })
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        assert engine.is_mouth_visible() is True

    def test_transparent_glasses_eye_blending(self) -> None:
        # Parser misses eye, but EYE_GLASS is present and landmark is valid.
        parsing_result = _make_parsing_result({FacePart.EYE_GLASS: 100})
        face = types.SimpleNamespace()
        face.kps = np.array([[10.0, 10.0], [20.0, 20.0], [0,0], [0,0], [0,0]], dtype=np.float32)

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_eye_visible(FacePart.LEFT_EYE) is True

    def test_hijab_allowed_head_covering(self) -> None:
        # HAT is present, but mandatory facial features are visible and pose is frontal.
        parsing_result = _make_parsing_result({
            FacePart.HAT: 500,
            FacePart.LEFT_EYE: 50,
            FacePart.RIGHT_EYE: 50,
            FacePart.NOSE: 100,
            FacePart.UPPER_LIP: 30,
            FacePart.LOWER_LIP: 30,
        })
        face = types.SimpleNamespace()
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_head_covering_prohibited() is False

    def test_cap_prohibited_head_covering(self) -> None:
        # HAT is present, and mandatory facial features are missing.
        parsing_result = _make_parsing_result({
            FacePart.HAT: 800,
        })
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        assert engine.is_head_covering_prohibited() is True

    def test_smooth_confidence_normalization(self) -> None:
        engine = SemanticEvidenceEngine(parsing_result=_make_parsing_result({}))
        # Test ratio normalization
        assert engine._normalize_ratio(0.0015, 0.0015) == 1.0
        assert engine._normalize_ratio(0.00075, 0.0015) == 0.5
        assert engine._normalize_ratio(0.0, 0.0015) == 0.0
