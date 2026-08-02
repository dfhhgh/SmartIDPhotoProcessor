"""Tests for the FaceAmbiguityValidator."""

import pytest

from models.ranked_face import RankedFace
from models.selection_result import SelectionResult
from models.validation_type import ValidationType
from validators.face_ambiguity_validator import FaceAmbiguityValidator
from tests.factories import create_face


def test_ambiguity_validator_single_face_passes():
    """Verify that selection with a single face passes ambiguity validation."""
    validator = FaceAmbiguityValidator()
    face = create_face()
    ranked_face = RankedFace(face=face, score=0.9)
    result = SelectionResult(
        selected_face=face,
        selected_score=0.9,
        second_best_score=None,
        score_margin=0.9,
        ambiguity_ratio=0.0,
        detected_faces_count=1,
        ranked_faces=(ranked_face,),
    )

    metric = validator.validate(result)
    assert metric.passed is True
    assert metric.type == ValidationType.FACE_AMBIGUITY
    assert metric.score == 1.0


def test_ambiguity_validator_weak_runner_up_passes():
    """Verify that a weak distant runner-up (billboard/poster) does not trigger ambiguity rejection."""
    validator = FaceAmbiguityValidator()
    face_a = create_face()
    face_b = create_face()
    ranked_a = RankedFace(face=face_a, score=0.85)
    ranked_b = RankedFace(face=face_b, score=0.10)
    
    result = SelectionResult(
        selected_face=face_a,
        selected_score=0.85,
        second_best_score=0.10,
        score_margin=0.75,
        ambiguity_ratio=0.1177,
        detected_faces_count=2,
        ranked_faces=(ranked_a, ranked_b),
    )

    metric = validator.validate(result)
    assert metric.passed is True


def test_ambiguity_validator_genuine_ambiguity_fails():
    """Verify that two strong competing faces with high ambiguity ratio fail validation."""
    validator = FaceAmbiguityValidator()
    face_a = create_face()
    face_b = create_face()
    ranked_a = RankedFace(face=face_a, score=0.80)
    ranked_b = RankedFace(face=face_b, score=0.70)
    
    result = SelectionResult(
        selected_face=face_a,
        selected_score=0.80,
        second_best_score=0.70,
        score_margin=0.10,
        ambiguity_ratio=0.70 / 0.80,
        detected_faces_count=2,
        ranked_faces=(ranked_a, ranked_b),
    )

    metric = validator.validate(result)
    assert metric.passed is False
    assert metric.type == ValidationType.FACE_AMBIGUITY
