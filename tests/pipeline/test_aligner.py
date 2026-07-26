import cv2
import numpy as np
import pytest

from config.constants import ALIGNED_FACE_SIZE, ARCFACE_TEMPLATE
from pipeline.aligner import FaceAligner
from exceptions.face_exceptions import FaceAlignmentError
from unittest.mock import MagicMock, patch
@pytest.fixture
def aligner():
    return FaceAligner()

def test_validate_image_success(
    aligner: FaceAligner,
):
    image = np.zeros(
        (112, 112, 3),
        dtype=np.uint8,
    )

    aligner._validate_image(image)

from unittest.mock import MagicMock


@pytest.fixture
def valid_face():
    face = MagicMock()

    face.kps = np.array(
        [
            [38.0, 51.0],
            [73.0, 51.0],
            [56.0, 71.0],
            [41.0, 92.0],
            [70.0, 92.0],
        ],
        dtype=np.float32,
    )

    return face





def test_validate_image_none(
    aligner: FaceAligner,
):
    with pytest.raises(
        FaceAlignmentError,
        match="Input image cannot be None.",
    ):
        aligner._validate_image(None)

def test_validate_image_invalid_type(
    aligner: FaceAligner,
):
    with pytest.raises(
        FaceAlignmentError,
        match="Input image must be a NumPy ndarray.",
    ):
        aligner._validate_image("image")


def test_validate_image_empty(
    aligner: FaceAligner,
):
    image = np.array([])

    with pytest.raises(
        FaceAlignmentError,
        match="Input image cannot be empty.",
    ):
        aligner._validate_image(image)


def test_validate_image_invalid_dimensions(
    aligner: FaceAligner,
):
    image = np.zeros(
        (2, 2, 2, 2),
        dtype=np.uint8,
    )

    with pytest.raises(
        FaceAlignmentError,
        match="Input image must have 2 or 3 dimensions.",
    ):
        aligner._validate_image(image)

def test_validate_landmarks_success(
    aligner: FaceAligner,
    valid_face,
):
    aligner._validate_landmarks(valid_face)


def test_validate_landmarks_none(
    aligner: FaceAligner,
    valid_face,
):
    valid_face.kps = None

    with pytest.raises(
        FaceAlignmentError,
        match="Face landmarks cannot be None.",
    ):
        aligner._validate_landmarks(valid_face)


def test_validate_landmarks_invalid_type(
    aligner: FaceAligner,
    valid_face,
):
    valid_face.kps = "invalid"

    with pytest.raises(
        FaceAlignmentError,
        match="Facial landmarks must be a NumPy ndarray.",
    ):
        aligner._validate_landmarks(valid_face)


def test_validate_landmarks_invalid_shape(
    aligner: FaceAligner,
    valid_face,
):
    valid_face.kps = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        FaceAlignmentError,
        match="Facial landmarks must have shape \\(5, 2\\).",
    ):
        aligner._validate_landmarks(valid_face)


def test_validate_landmarks_contains_nan(
    aligner: FaceAligner,
    valid_face,
):
    valid_face.kps[0, 0] = np.nan

    with pytest.raises(
        FaceAlignmentError,
        match="Facial landmarks cannot contain NaN values.",
    ):
        aligner._validate_landmarks(valid_face)


def test_validate_landmarks_contains_inf(
    aligner: FaceAligner,
    valid_face,
):
    valid_face.kps[0, 0] = np.inf

    with pytest.raises(
        FaceAlignmentError,
        match="Facial landmarks cannot contain infinite values.",
    ):
        aligner._validate_landmarks(valid_face)


def test_get_source_landmarks_success(
    aligner: FaceAligner,
    valid_face,
):
    result = aligner._get_source_landmarks(valid_face)

    np.testing.assert_array_equal(
        result,
        valid_face.kps.astype(np.float32),
    )

    assert result.dtype == np.float32

def test_get_target_landmarks_success(
    aligner: FaceAligner,
):
    result = aligner._get_target_landmarks()

    np.testing.assert_array_equal(
        result,
        ARCFACE_TEMPLATE,
    )

    assert result.dtype == np.float32

def test_get_target_landmarks_returns_copy(
    aligner: FaceAligner,
):
    result = aligner._get_target_landmarks()

    assert result is not ARCFACE_TEMPLATE



def test_compute_similarity_transform_success(
    aligner: FaceAligner,
):
    source = np.array(
        [
            [38, 51],
            [73, 51],
            [56, 71],
            [41, 92],
            [70, 92],
        ],
        dtype=np.float32,
    )

    target = source.copy()

    expected_transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    with patch(
        "pipeline.aligner.cv2.estimateAffinePartial2D",
        return_value=(expected_transform, None),
    ) as mock_estimate:

        result = aligner._compute_similarity_transform(
            source,
            target,
        )

    mock_estimate.assert_called_once_with(
        source,
        target,
    )

    np.testing.assert_array_equal(
        result,
        expected_transform,
    )

    assert result.dtype == np.float32

def test_compute_similarity_transform_invalid_source_type(
    aligner: FaceAligner,
):
    target = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        FaceAlignmentError,
        match="Source landmarks must be a NumPy ndarray.",
    ):
        aligner._compute_similarity_transform(
            "invalid",
            target,
        )


def test_compute_similarity_transform_invalid_target_type(
    aligner: FaceAligner,
):
    source = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        FaceAlignmentError,
        match="Target landmarks must be a NumPy ndarray.",
    ):
        aligner._compute_similarity_transform(
            source,
            "invalid",
        )


def test_compute_similarity_transform_invalid_source_shape(
    aligner: FaceAligner,
):
    source = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    target = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        FaceAlignmentError,
        match=r"Source landmarks must have shape \(5, 2\)\.",
    ):
        aligner._compute_similarity_transform(
            source,
            target,
        )


def test_compute_similarity_transform_invalid_target_shape(
    aligner: FaceAligner,
):
    source = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    target = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    with pytest.raises(
        FaceAlignmentError,
        match=r"Target landmarks must have shape \(5, 2\)\.",
    ):
        aligner._compute_similarity_transform(
            source,
            target,
        )

def test_compute_similarity_transform_transform_none(
    aligner: FaceAligner,
):
    source = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    target = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    with patch(
        "pipeline.aligner.cv2.estimateAffinePartial2D",
        return_value=(None, None),
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Similarity transform could not be computed.",
        ):
            aligner._compute_similarity_transform(
                source,
                target,
            )


def test_compute_similarity_transform_invalid_transform_shape(
    aligner: FaceAligner,
):
    source = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    target = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    invalid_transform = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    with patch(
        "pipeline.aligner.cv2.estimateAffinePartial2D",
        return_value=(invalid_transform, None),
    ):
        with pytest.raises(
            FaceAlignmentError,
            match=r"Similarity transform must have shape \(2, 3\)\.",
        ):
            aligner._compute_similarity_transform(
                source,
                target,
            )


def test_apply_transform_success(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    expected_image = np.zeros(
        (
            ALIGNED_FACE_SIZE[1],
            ALIGNED_FACE_SIZE[0],
            3,
        ),
        dtype=np.uint8,
    )

    with patch(
        "pipeline.aligner.cv2.warpAffine",
        return_value=expected_image,
    ) as mock_warp:

        result = aligner._apply_transform(
            image,
            transform,
        )

    mock_warp.assert_called_once_with(
        image,
        transform,
        ALIGNED_FACE_SIZE,
    )

    np.testing.assert_array_equal(
        result,
        expected_image,
    )

def test_apply_transform_invalid_transform_type(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        FaceAlignmentError,
        match="Similarity transform must be a NumPy ndarray.",
    ):
        aligner._apply_transform(
            image,
            "invalid",
        )

def test_apply_transform_invalid_transform_shape(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = np.zeros(
        (3, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        FaceAlignmentError,
        match=r"Similarity transform must have shape \(2, 3\)\.",
    ):
        aligner._apply_transform(
            image,
            transform,
        )

def test_apply_transform_aligned_image_none(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    with patch(
        "pipeline.aligner.cv2.warpAffine",
        return_value=None,
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Aligned image cannot be None.",
        ):
            aligner._apply_transform(
                image,
                transform,
            )


def test_apply_transform_invalid_aligned_image_type(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    with patch(
        "pipeline.aligner.cv2.warpAffine",
        return_value="invalid",
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Aligned image must be a NumPy ndarray.",
        ):
            aligner._apply_transform(
                image,
                transform,
            )



def test_apply_transform_empty_aligned_image(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    empty_image = np.array(
        [],
        dtype=np.uint8,
    )

    with patch(
        "pipeline.aligner.cv2.warpAffine",
        return_value=empty_image,
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Aligned image cannot be empty.",
        ):
            aligner._apply_transform(
                image,
                transform,
            )


def test_apply_transform_warp_affine_error(
    aligner: FaceAligner,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    with patch(
        "pipeline.aligner.cv2.warpAffine",
        side_effect=cv2.error("OpenCV error"),
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Similarity transform could not be applied.",
        ):
            aligner._apply_transform(
                image,
                transform,
            )



def test_align_success(
    aligner: FaceAligner,
    valid_face: MagicMock,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    source = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    target = np.zeros(
        (5, 2),
        dtype=np.float32,
    )

    transform = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.float32,
    )

    expected_image = np.zeros(
        (
            ALIGNED_FACE_SIZE[1],
            ALIGNED_FACE_SIZE[0],
            3,
        ),
        dtype=np.uint8,
    )

    with (
        patch.object(
            aligner,
            "_validate_image",
        ) as mock_validate,

        patch.object(
            aligner,
            "_get_source_landmarks",
            return_value=source,
        ) as mock_source,

        patch.object(
            aligner,
            "_get_target_landmarks",
            return_value=target,
        ) as mock_target,

        patch.object(
            aligner,
            "_compute_similarity_transform",
            return_value=transform,
        ) as mock_compute,

        patch.object(
            aligner,
            "_apply_transform",
            return_value=expected_image,
        ) as mock_apply,
    ):

        result = aligner.align(
            image,
            valid_face,
        )

    mock_validate.assert_called_once_with(image)

    mock_source.assert_called_once_with(valid_face)

    mock_target.assert_called_once_with()

    mock_compute.assert_called_once_with(
        source,
        target,
    )

    mock_apply.assert_called_once_with(
        image,
        transform,
    )

    np.testing.assert_array_equal(
        result,
        expected_image,
    )


def test_align_reraises_face_alignment_error(
    aligner: FaceAligner,
    valid_face: MagicMock,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    with patch.object(
        aligner,
        "_validate_image",
        side_effect=FaceAlignmentError(
            "Validation failed.",
        ),
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Validation failed.",
        ):
            aligner.align(
                image,
                valid_face,
            )

def test_align_wraps_unexpected_exception(
    aligner: FaceAligner,
    valid_face: MagicMock,
):
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    with patch.object(
        aligner,
        "_validate_image",
        side_effect=RuntimeError(
            "Unexpected error.",
        ),
    ):
        with pytest.raises(
            FaceAlignmentError,
            match="Failed to align face.",
        ) as exc_info:
            aligner.align(
                image,
                valid_face,
            )

    assert isinstance(
        exc_info.value.__cause__,
        RuntimeError,
    )

    assert str(
        exc_info.value.__cause__,
    ) == "Unexpected error."