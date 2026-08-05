"""
Unit tests for SemanticEvidenceEngine.
"""

from __future__ import annotations

import numpy as np
import pytest
import types

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from reasoning.semantic_engine import SemanticEvidenceEngine


def _make_parsing_result(parts: dict[FacePart, int], height: int = 10, width: int = 10) -> FaceParsingResult:
    mask = np.zeros((height, width), dtype=np.int32)
    flat = mask.ravel()
    cursor = 0
    for part, count in parts.items():
        flat[cursor:cursor + count] = int(part)
        cursor += count
    mask = flat.reshape((height, width))
    return FaceParsingResult(mask=mask, image_height=height, image_width=width)


class TestSemanticEvidenceEngine:
    def test_init_validates_parsing_result(self) -> None:
        with pytest.raises(TypeError):
            SemanticEvidenceEngine(parsing_result=None)  # type: ignore

    def test_is_eye_visible_direct_parser(self) -> None:
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 10})
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        assert engine.is_eye_visible(FacePart.LEFT_EYE) is True
        assert engine.is_eye_visible(FacePart.RIGHT_EYE) is False

    def test_is_eye_visible_landmark_override_with_glasses(self) -> None:
        parsing_result = _make_parsing_result({FacePart.EYE_GLASS: 10})
        face = types.SimpleNamespace()
        face.kps = np.array([[0.0, 0.0], [50.0, 50.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        
        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        # Left eye is index 1 in kps
        assert engine.is_eye_visible(FacePart.LEFT_EYE) is True
        # Right eye is index 0 in kps
        assert engine.is_eye_visible(FacePart.RIGHT_EYE) is True

    def test_is_eyebrow_visible_direct_parser(self) -> None:
        parsing_result = _make_parsing_result({FacePart.LEFT_BROW: 10})
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        assert engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE) is True

    def test_is_eyebrow_visible_fusion_fallback(self) -> None:
        # Parser misses left brow, but has left eye, valid landmark, frontal pose
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 10})
        face = types.SimpleNamespace()
        face.kps = np.array([[0.0, 0.0], [50.0, 50.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE) is True

    def test_is_head_covering_prohibited_when_no_hat(self) -> None:
        parsing_result = _make_parsing_result({FacePart.SKIN: 50})
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        assert engine.is_head_covering_prohibited() is False

    def test_is_head_covering_prohibited_allowed_hijab(self) -> None:
        # HAT present, but all mandatory regions visible and pose frontal -> allowed (not prohibited)
        parsing_result = _make_parsing_result({
            FacePart.HAT: 20,
            FacePart.LEFT_EYE: 5,
            FacePart.RIGHT_EYE: 5,
            FacePart.NOSE: 5,
            FacePart.MOUTH: 5,
        })
        face = types.SimpleNamespace()
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_head_covering_prohibited() is False

    def test_is_head_covering_prohibited_forbidden_cap(self) -> None:
        # HAT present, but mandatory regions missing -> prohibited
        parsing_result = _make_parsing_result({
            FacePart.HAT: 50,
        })
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        assert engine.is_head_covering_prohibited() is True
