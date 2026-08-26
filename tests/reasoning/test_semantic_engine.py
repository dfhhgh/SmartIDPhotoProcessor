"""
Unit tests for SemanticEvidenceEngine.
"""

from __future__ import annotations

import numpy as np
import pytest
import types

from insightface.app.common import Face

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


def _make_face_with_landmarks(pose: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Face:
    """Create a synthetic Face with valid 5-point landmarks."""
    face = Face()
    face.kps = np.array(
        [[55.0, 50.0], [55.0, 50.0], [55.0, 70.0], [40.0, 85.0], [70.0, 85.0]],
        dtype=np.float32,
    )
    face.pose = pose
    return face


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


# ------------------------------------------------------------------
# Phase 5 Reporting Consistency Regression Tests
# ------------------------------------------------------------------


class TestReportingConsistency:
    """Verify that compute_eye_evidence() and is_eye_visible() produce
    semantically identical results for all glasses scenarios."""

    def test_no_glasses_paths_agree(self) -> None:
        """TEST 1: No glasses — normal eye pixels, both paths agree."""
        pr = _make_parsing_result({FacePart.LEFT_EYE: 20, FacePart.RIGHT_EYE: 20, FacePart.NOSE: 50})
        engine = SemanticEvidenceEngine(parsing_result=pr, face=_make_face_with_landmarks())
        ev = engine.compute_eye_evidence(FacePart.LEFT_EYE)
        vis = engine.is_eye_visible(FacePart.LEFT_EYE)
        assert ev.parser_confidence > 0.0
        assert ev.passed is vis

    def test_transparent_glasses_065_fallback_reflected(self) -> None:
        """TEST 2: Transparent glasses — 0.65 fallback reflected in both paths."""
        pr = _make_parsing_result({FacePart.EYE_GLASS: 50, FacePart.NOSE: 50})
        engine = SemanticEvidenceEngine(parsing_result=pr, face=_make_face_with_landmarks())
        ev = engine.compute_eye_evidence(FacePart.LEFT_EYE)
        vis = engine.is_eye_visible(FacePart.LEFT_EYE)
        assert ev.parser_confidence == pytest.approx(0.65)
        assert ev.passed is vis

    def test_opaque_sunglasses_no_landmarks_paths_agree(self) -> None:
        """TEST 3: Opaque sunglasses — EYE_GLASS present, no valid landmarks."""
        pr = _make_parsing_result({FacePart.EYE_GLASS: 50, FacePart.NOSE: 50})
        face = _make_face_with_landmarks()
        face.kps = np.array(
            [[float("nan"), float("nan")], [float("nan"), float("nan")], [0, 0], [0, 0], [0, 0]],
            dtype=np.float32,
        )
        engine = SemanticEvidenceEngine(parsing_result=pr, face=face)
        ev = engine.compute_eye_evidence(FacePart.LEFT_EYE)
        vis = engine.is_eye_visible(FacePart.LEFT_EYE)
        assert ev.parser_confidence == pytest.approx(0.0)
        assert ev.passed is vis

    def test_semi_transparent_sunglasses_paths_agree(self) -> None:
        """TEST 4: Semi-transparent sunglasses — some eye pixels visible."""
        pr = _make_parsing_result(
            {FacePart.EYE_GLASS: 30, FacePart.LEFT_EYE: 5, FacePart.RIGHT_EYE: 5, FacePart.NOSE: 50}
        )
        engine = SemanticEvidenceEngine(parsing_result=pr, face=_make_face_with_landmarks())
        ev = engine.compute_eye_evidence(FacePart.LEFT_EYE)
        vis = engine.is_eye_visible(FacePart.LEFT_EYE)
        assert ev.parser_confidence > 0.0
        assert ev.passed is vis

    def test_borderline_pose_065_tips_balance(self) -> None:
        """TEST 5: Borderline pose — 0.65 fallback could tip the balance.
        Without 0.65: parser=0 → score < 0.50 → FAIL.
        With 0.65: parser=0.65 → score > 0.50 → PASS.
        Both paths must now agree."""
        pr = _make_parsing_result({FacePart.EYE_GLASS: 50, FacePart.NOSE: 50})
        # roll=9.0 → roll_score = 1 - 9/10 = 0.1 (near threshold)
        engine = SemanticEvidenceEngine(parsing_result=pr, face=_make_face_with_landmarks(pose=(0.0, 0.0, 9.0)))
        ev = engine.compute_eye_evidence(FacePart.LEFT_EYE)
        vis = engine.is_eye_visible(FacePart.LEFT_EYE)
        assert ev.parser_confidence == pytest.approx(0.65)
        assert ev.passed is vis, (
            f"Borderline paths disagree: evidence={ev.passed}, visible={vis}"
        )

    def test_compute_eye_evidence_uses_effective_parser_confidence(self) -> None:
        """Verify compute_eye_evidence uses the same fallback as is_eye_visible."""
        pr = _make_parsing_result({FacePart.EYE_GLASS: 50, FacePart.NOSE: 50})
        engine = SemanticEvidenceEngine(parsing_result=pr, face=_make_face_with_landmarks())
        ev = engine.compute_eye_evidence(FacePart.LEFT_EYE)
        effective = engine._compute_effective_parser_confidence(FacePart.LEFT_EYE)
        assert ev.parser_confidence == pytest.approx(effective)

    def test_effective_parser_confidence_no_glasses(self) -> None:
        """_compute_effective_parser_confidence returns raw value when no glasses."""
        pr = _make_parsing_result({FacePart.LEFT_EYE: 20})
        engine = SemanticEvidenceEngine(parsing_result=pr, face=_make_face_with_landmarks())
        effective = engine._compute_effective_parser_confidence(FacePart.LEFT_EYE)
        raw = engine._compute_parser_confidence(FacePart.LEFT_EYE, 0.0015)
        assert effective == pytest.approx(raw)

    def test_effective_parser_confidence_glasses_no_landmarks(self) -> None:
        """_compute_effective_parser_confidence returns 0.0 when landmarks invalid."""
        pr = _make_parsing_result({FacePart.EYE_GLASS: 50})
        face = _make_face_with_landmarks()
        face.kps = None
        engine = SemanticEvidenceEngine(parsing_result=pr, face=face)
        effective = engine._compute_effective_parser_confidence(FacePart.LEFT_EYE)
        assert effective == pytest.approx(0.0)
