import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from pipeline.cropper import FaceCropper
from exceptions.face_exceptions import FaceCroppingError
from models.crop_result import CropResult

@pytest.fixture
def cropper():
    return FaceCropper()


def test_validate_image_accepts_valid_image(
    cropper: FaceCropper,
):
    image = np.zeros(
        (640, 480, 3),
        dtype=np.uint8,
    )

    cropper._validate_image(image)


def test_validate_image_raises_for_none(
    cropper: FaceCropper,
):
    with pytest.raises(
        FaceCroppingError,
    ):
        cropper._validate_image(None)


def test_validate_image_raises_for_invalid_type(
    cropper: FaceCropper,
):
    with pytest.raises(
        FaceCroppingError,
    ):
        cropper._validate_image([])


def test_validate_image_raises_for_empty_image(
    cropper: FaceCropper,
):
    image = np.array([])

    with pytest.raises(
        FaceCroppingError,
    ):
        cropper._validate_image(image)


def test_validate_image_raises_for_invalid_dimensions(
    cropper: FaceCropper,
):
    image = np.array(5)

    with pytest.raises(
        FaceCroppingError,
    ):
        cropper._validate_image(image)


def test_expand_bbox_returns_expected_coordinates(
    cropper: FaceCropper,
):
    result = cropper._expand_bbox(
        100,
        100,
        200,
        200,
    )

    assert result == (
        70,
        55,
        230,
        275,
    )

def test_clamp_bbox_returns_same_coordinates_when_inside_image(
    cropper: FaceCropper,
):
    result = cropper._clamp_bbox(
        50,
        40,
        150,
        180,
        (300, 200, 3),
    )

    assert result == (
        50,
        40,
        150,
        180,
    )


def test_clamp_bbox_clamps_negative_coordinates(
    cropper: FaceCropper,
):
    result = cropper._clamp_bbox(
        -20,
        -10,
        150,
        180,
        (300, 200, 3),
    )

    assert result == (
        0,
        0,
        150,
        180,
    )


def test_clamp_bbox_clamps_coordinates_exceeding_image_size(
    cropper: FaceCropper,
):
    result = cropper._clamp_bbox(
        20,
        30,
        250,
        350,
        (300, 200, 3),
    )

    assert result == (
        20,
        30,
        200,
        300,
    )


def test_clamp_bbox_clamps_negative_and_exceeding_coordinates(
    cropper: FaceCropper,
):
    result = cropper._clamp_bbox(
        -50,
        -30,
        400,
        500,
        (300, 200, 3),
    )

    assert result == (
        0,
        0,
        200,
        300,
    )


def test_crop_image_returns_expected_region(
    cropper: FaceCropper,
):
    image = np.arange(
        100,
        dtype=np.uint8,
    ).reshape(
        10,
        10,
    )

    cropped = cropper._crop_image(
        image,
        2,
        3,
        6,
        8,
    )

    expected = image[
        3:8,
        2:6,
    ]

    assert np.array_equal(
        cropped,
        expected,
    )



def test_crop_image_returns_expected_shape(
    cropper: FaceCropper,
):
    image = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    cropped = cropper._crop_image(
        image,
        50,
        20,
        150,
        80,
    )

    assert cropped.shape == (
        60,
        100,
        3,
    )





def test_crop_returns_cropped_image(
    cropper: FaceCropper,
):
    image = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    face = MagicMock()
    face.bbox = [100, 100, 200, 200]

    result = cropper.crop(
        image,
        face,
    )

    assert isinstance(result, CropResult)
    assert isinstance(result.image, np.ndarray)
    assert result.image.ndim == 3
    assert result.image.size > 0
    assert isinstance(result.crop_x, int)
    assert isinstance(result.crop_y, int)

    expected = image[
        result.crop_y : result.crop_y + result.image.shape[0],
        result.crop_x : result.crop_x + result.image.shape[1],
    ]
    assert np.array_equal(result.image, expected)

def test_crop_raises_for_invalid_image(
    cropper: FaceCropper,
):
    face = MagicMock()
    face.bbox = [10, 10, 50, 50]

    with pytest.raises(
        FaceCroppingError,
    ):
        cropper.crop(
            None,
            face,
        )


def test_crop_raises_for_invalid_bbox(
    cropper: FaceCropper,
):
    image = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    face = MagicMock()
    face.bbox = None

    with pytest.raises(
        FaceCroppingError,
    ):
        cropper.crop(
            image,
            face,
        )


def test_crop_wraps_unexpected_exception(
    cropper: FaceCropper,
):
    image = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    face = MagicMock()
    face.bbox = [10, 10, 50, 50]

    with patch.object(
        cropper,
        "_crop_image",
        side_effect=RuntimeError("Unexpected error"),
    ):
        with pytest.raises(
            FaceCroppingError,
        ):
            cropper.crop(
                image,
                face,
            )
