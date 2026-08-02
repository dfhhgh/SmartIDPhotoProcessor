from pipeline.selector import FaceSelector

from tests.factories import create_face
import pytest

from exceptions.face_exceptions import FaceSelectionError
from models.selection_result import SelectionResult


def test_select_returns_single_face_when_only_one_face_exists():
    # Arrange
    selector = FaceSelector()

    face = create_face()

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act
    selection_result = selector.select(
        faces=[face],
        image_shape=image_shape,
    )

    # Assert
    assert isinstance(selection_result, SelectionResult)
    assert selection_result.selected_face is face
    assert selection_result.detected_faces_count == 1
    assert selection_result.second_best_score is None


def test_select_returns_face_with_highest_final_score():
    # Arrange
    selector = FaceSelector()

    image_shape = (
        1000,
        1000,
        3,
    )

    face_a = create_face(
        bbox=(
            250,
            250,
            750,
            750,
        ),
        det_score=0.98,
    )

    face_b = create_face(
        bbox=(
            20,
            20,
            150,
            150,
        ),
        det_score=0.99,
    )

    face_c = create_face(
        bbox=(
            300,
            300,
            600,
            600,
        ),
        det_score=0.40,
    )

    # Act
    selection_result = selector.select(
        faces=[
            face_a,
            face_b,
            face_c,
        ],
        image_shape=image_shape,
    )

    # Assert
    assert isinstance(selection_result, SelectionResult)
    assert selection_result.selected_face is face_a
    assert selection_result.detected_faces_count == 3


def test_select_raises_face_selection_error_when_faces_list_is_empty():
    # Arrange
    selector = FaceSelector()

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
        FaceSelectionError,
        match="No faces were provided for selection.",
    ):
        selector.select(
            faces=[],
            image_shape=image_shape,
        )



def test_select_raises_face_selection_error_when_face_bbox_is_none():
    # Arrange
    selector = FaceSelector()

    face = create_face(
        bbox=None,
    )

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
        FaceSelectionError,
        match="Face bounding box is missing.",
    ):
        selector.select(
            faces=[face],
            image_shape=image_shape,
        )





def test_select_raises_face_selection_error_when_face_bbox_has_invalid_length():
    # Arrange
    selector = FaceSelector()

    face = create_face(
        bbox=(
            100,
            100,
            200,
        ),
    )

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
        FaceSelectionError,
        match="Face bounding box must contain exactly four coordinates.",
    ):
        selector.select(
            faces=[face],
            image_shape=image_shape,
        )




def test_select_raises_face_selection_error_when_face_bbox_has_non_positive_width():
    # Arrange
    selector = FaceSelector()

    face = create_face(
        bbox=(
            300,
            100,
            300,
            400,
        ),
    )

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
    FaceSelectionError,
    match="Face bounding box has invalid dimensions.",
    ):
        selector.select(
            faces=[face],
            image_shape=image_shape,
        )


def test_select_raises_face_selection_error_when_image_height_is_non_positive():
    # Arrange
    selector = FaceSelector()

    face = create_face()

    image_shape = (
        0,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
        FaceSelectionError,
        match="Image dimensions must be positive.",
    ):
        selector.select(
            faces=[face],
            image_shape=image_shape,
        )

def test_select_raises_face_selection_error_when_face_detection_confidence_is_missing():
    # Arrange
    selector = FaceSelector()

    face = create_face(
        det_score=None,
    )

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
        FaceSelectionError,
        match="Face detection confidence is missing.",
    ):
        selector.select(
            faces=[face],
            image_shape=image_shape,
        )

def test_select_raises_face_selection_error_when_face_detection_confidence_is_less_than_zero():
    # Arrange
    selector = FaceSelector()

    face = create_face(
        det_score=-0.1,
    )

    image_shape = (
        1000,
        1000,
        3,
    )

    # Act & Assert
    with pytest.raises(
        FaceSelectionError,
        match="Face detection confidence must be between 0.0 and 1.0.",
    ):
        selector.select(
            faces=[face],
            image_shape=image_shape,
        )
