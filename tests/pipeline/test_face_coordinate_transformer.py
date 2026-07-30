"""Tests for FaceCoordinateTransformer."""

from __future__ import annotations

import numpy as np
import pytest
from insightface.app.common import Face

from exceptions.face_exceptions import FaceCoordinateTransformationError
from pipeline.face_coordinate_transformer import FaceCoordinateTransformer


@pytest.fixture
def transformer() -> FaceCoordinateTransformer:
    return FaceCoordinateTransformer()


@pytest.fixture
def sample_face() -> Face:
    face = Face()
    face.bbox = np.array([100.0, 150.0, 300.0, 350.0], dtype=np.float32)
    face.kps = np.array(
        [
            [120.0, 180.0],
            [280.0, 180.0],
            [200.0, 250.0],
            [140.0, 300.0],
            [260.0, 300.0],
        ],
        dtype=np.float32,
    )
    face.det_score = 0.99
    face.pose = [1.0, 2.0, 3.0]
    return face


def test_successful_bbox_translation(transformer, sample_face):
    """Verify that bounding box coordinates are correctly translated by crop offset."""
    crop_x, crop_y = 50, 100
    transformed = transformer.transform(sample_face, crop_x, crop_y)

    expected_bbox = np.array([50.0, 50.0, 250.0, 250.0], dtype=np.float32)
    np.testing.assert_array_equal(transformed.bbox, expected_bbox)


def test_successful_landmark_translation(transformer, sample_face):
    """Verify that landmark coordinates (kps) are correctly translated by crop offset."""
    crop_x, crop_y = 50, 100
    transformed = transformer.transform(sample_face, crop_x, crop_y)

    expected_kps = np.array(
        [
            [70.0, 80.0],
            [230.0, 80.0],
            [150.0, 150.0],
            [90.0, 200.0],
            [210.0, 200.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed.kps, expected_kps)


def test_original_face_remains_unchanged(transformer, sample_face):
    """Verify that the original Face object is immutable and unaffected by transformation."""
    original_bbox = sample_face.bbox.copy()
    original_kps = sample_face.kps.copy()

    crop_x, crop_y = 50, 100
    _ = transformer.transform(sample_face, crop_x, crop_y)

    np.testing.assert_array_equal(sample_face.bbox, original_bbox)
    np.testing.assert_array_equal(sample_face.kps, original_kps)


def test_object_independence(transformer, sample_face):
    """Verify that the returned Face object and its NumPy arrays are completely independent."""
    crop_x, crop_y = 50, 100
    transformed = transformer.transform(sample_face, crop_x, crop_y)

    assert transformed is not sample_face
    assert transformed.bbox is not sample_face.bbox
    assert transformed.kps is not sample_face.kps


def test_other_attributes_preserved(transformer, sample_face):
    """Verify non-spatial attributes (det_score, pose) remain intact on the transformed face."""
    crop_x, crop_y = 50, 100
    transformed = transformer.transform(sample_face, crop_x, crop_y)

    assert transformed.det_score == 0.99
    assert transformed.pose == [1.0, 2.0, 3.0]


def test_none_face_raises_error(transformer):
    """Verify that passing None raises FaceCoordinateTransformationError."""
    with pytest.raises(FaceCoordinateTransformationError, match="Face cannot be None"):
        transformer.transform(None, 0, 0)  # type: ignore[arg-type]


def test_missing_bbox_raises_error(transformer):
    """Verify that a Face missing bbox raises FaceCoordinateTransformationError."""
    face = Face()
    face.kps = np.zeros((5, 2), dtype=np.float32)
    face.bbox = None

    with pytest.raises(FaceCoordinateTransformationError, match="Face bounding box is missing"):
        transformer.transform(face, 0, 0)


def test_missing_kps_raises_error(transformer):
    """Verify that a Face missing kps raises FaceCoordinateTransformationError."""
    face = Face()
    face.bbox = np.zeros((4,), dtype=np.float32)
    face.kps = None

    with pytest.raises(FaceCoordinateTransformationError, match="Face landmarks.*are missing"):
        transformer.transform(face, 0, 0)


def test_invalid_bbox_type_raises_error(transformer):
    """Verify that non-ndarray bbox raises FaceCoordinateTransformationError."""
    face = Face()
    face.bbox = [0, 0, 10, 10]
    face.kps = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(FaceCoordinateTransformationError, match="Face bounding box must be a NumPy ndarray"):
        transformer.transform(face, 0, 0)


def test_invalid_kps_type_raises_error(transformer):
    """Verify that non-ndarray kps raises FaceCoordinateTransformationError."""
    face = Face()
    face.bbox = np.zeros((4,), dtype=np.float32)
    face.kps = [[0, 0]] * 5

    with pytest.raises(FaceCoordinateTransformationError, match="Face landmarks.*must be a NumPy ndarray"):
        transformer.transform(face, 0, 0)


def test_invalid_bbox_shape_raises_error(transformer):
    """Verify that bbox with incorrect shape raises FaceCoordinateTransformationError."""
    face = Face()
    face.bbox = np.zeros((5,), dtype=np.float32)
    face.kps = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(FaceCoordinateTransformationError, match=r"Face bounding box must have shape \(4,\)"):
        transformer.transform(face, 0, 0)


def test_invalid_kps_shape_raises_error(transformer):
    """Verify that kps with incorrect shape raises FaceCoordinateTransformationError."""
    face = Face()
    face.bbox = np.zeros((4,), dtype=np.float32)
    face.kps = np.zeros((5, 3), dtype=np.float32)

    with pytest.raises(FaceCoordinateTransformationError, match=r"Face landmarks.*must have shape \(5, 2\)"):
        transformer.transform(face, 0, 0)
