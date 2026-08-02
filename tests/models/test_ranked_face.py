"""Tests for the RankedFace model."""

import pytest

from models.ranked_face import RankedFace
from tests.factories import create_face


def test_create_valid_ranked_face():
    """Verify that a valid RankedFace is created successfully."""
    face = create_face()
    score = 0.85
    ranked_face = RankedFace(face=face, score=score)

    assert ranked_face.face is face
    assert ranked_face.score == 0.85


def test_ranked_face_score_must_be_numeric():
    """Verify that non-numeric score raises TypeError."""
    face = create_face()
    with pytest.raises(TypeError):
        RankedFace(face=face, score="0.85")


def test_ranked_face_score_must_be_in_range():
    """Verify that out-of-range score raises ValueError."""
    face = create_face()
    with pytest.raises(ValueError):
        RankedFace(face=face, score=1.05)


def test_ranked_face_none_face_raises_value_error():
    """Verify that None face raises ValueError."""
    with pytest.raises(ValueError):
        RankedFace(face=None, score=0.85)
