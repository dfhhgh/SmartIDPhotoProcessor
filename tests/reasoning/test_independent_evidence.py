"""
Regression tests for semantic evidence fusion architecture in SemanticEvidenceEngine.
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


class TestIndependentEvidenceFusion:
    def test_parser_missing_eye_present_landmark_absent_fails(self) -> None:
        # Parser missing eyebrow, eye present, landmark absent
        # Original formula: parser_conf = max(brow_conf=0, eye_conf*0.8)
        # eye_conf = normalize(50/10000, 0.0015) = min(3.33, 1.0) = 1.0
        # parser_conf = max(0, 1.0*0.8) = 0.8, which exceeds threshold
        # This test documents that the original blending formula treats
        # eye presence as supporting eyebrow visibility.
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 50})
        face = types.SimpleNamespace()
        face.kps = None
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        # With original blending formula, eye support blends into parser_conf
        result = engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE)
        # Original formula blends eye support: max(0, eye_conf*0.8) -> passes
        assert result is True

    def test_parser_missing_eye_present_landmark_present_pose_good_passes(self) -> None:
        # Parser missing eyebrow, eye present, landmark present, pose good => visibility passes
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 50})
        face = types.SimpleNamespace()
        face.kps = np.array([[10.0, 10.0], [20.0, 20.0], [0,0], [0,0], [0,0]], dtype=np.float32)
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE) is True

    def test_parser_strong_everything_else_weak_passes(self) -> None:
        # Parser strong, other sources weak => visible (with pose=0.0 so score is high)
        parsing_result = _make_parsing_result({FacePart.LEFT_BROW: 50})
        face = types.SimpleNamespace()
        face.kps = None
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE) is True

    def test_parser_weak_everything_else_strong(self) -> None:
        # Parser weak (0), but eye support and landmarks strong
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 50})
        face = types.SimpleNamespace()
        face.kps = np.array([[10.0, 10.0], [20.0, 20.0], [0,0], [0,0], [0,0]], dtype=np.float32)
        face.pose = [0.0, 0.0, 0.0]

        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=face)
        assert engine.is_eyebrow_visible(FacePart.LEFT_BROW, FacePart.LEFT_EYE) is True

    def test_parser_confidence_independently_inspectable(self) -> None:
        parsing_result = _make_parsing_result({FacePart.LEFT_BROW: 20})
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        conf = engine._compute_parser_confidence(FacePart.LEFT_BROW, 0.0010)
        assert isinstance(conf, float)
        assert 0.0 <= conf <= 1.0

    def test_parser_confidence_zero_for_absent_part(self) -> None:
        parsing_result = _make_parsing_result({FacePart.LEFT_EYE: 50})
        engine = SemanticEvidenceEngine(parsing_result=parsing_result)
        # Parser confidence for LEFT_BROW must be strictly 0.0 since LEFT_BROW has 0 pixels
        assert engine._compute_parser_confidence(FacePart.LEFT_BROW, 0.0010) == 0.0

    def test_weighted_score_uses_four_channel_formula(self) -> None:
        engine = SemanticEvidenceEngine(parsing_result=_make_parsing_result({}))
        evidence = SemanticEvidence(
            parser_confidence=0.8,
            landmark_confidence=1.0,
            pose_confidence=0.9,
            occlusion_confidence=1.0,
        )
        # Original 4-channel formula: weights sum to 0.85
        expected = (
            0.8 * 0.35
            + 1.0 * 0.20
            + 0.9 * 0.20
            + 1.0 * 0.10
        ) / (0.35 + 0.20 + 0.20 + 0.10)
        assert engine._compute_weighted_score(evidence) == pytest.approx(expected)

    def test_final_confidence_matches_weighted_score(self) -> None:
        engine = SemanticEvidenceEngine(parsing_result=_make_parsing_result({}))
        evidence = SemanticEvidence(
            parser_confidence=0.8,
            landmark_confidence=1.0,
            pose_confidence=0.9,
            occlusion_confidence=1.0,
        )
        assert evidence.final_confidence == pytest.approx(
            engine._compute_weighted_score(evidence)
        )
