"""
Unit tests for OcclusionValidator.
"""

import numpy as np
import pytest

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_type import ValidationType
from validators.occlusion_validator import OcclusionValidator


# ----------------------------------------------------------------------
# Fixtures / helpers
# ----------------------------------------------------------------------

IMAGE_HEIGHT = 10
IMAGE_WIDTH = 10


def _make_image() -> np.ndarray:
    """Return a small valid BGR-like uint8 image."""
    return np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)


def _make_parsing_result(parts: dict[FacePart, int]) -> FaceParsingResult:
    """Build a FaceParsingResult whose mask contains the given parts.

    Args:
        parts: Mapping of FacePart -> number of pixels to assign to that
            part. Remaining pixels are left as BACKGROUND. The total
            pixel budget is IMAGE_HEIGHT * IMAGE_WIDTH.

    Returns:
        A FaceParsingResult wrapping the constructed mask.
    """
    mask = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.int32,
    )

    flat = mask.ravel()
    cursor = 0

    for part, count in parts.items():
        flat[cursor:cursor + count] = int(part)
        cursor += count

    if cursor > flat.size:
        raise ValueError("Requested more pixels than the mask can hold.")

    mask = flat.reshape((IMAGE_HEIGHT, IMAGE_WIDTH))

    return FaceParsingResult(
        mask=mask,
        image_height=IMAGE_HEIGHT,
        image_width=IMAGE_WIDTH,
    )


@pytest.fixture
def validator() -> OcclusionValidator:
    return OcclusionValidator()


@pytest.fixture
def clean_parsing_result() -> FaceParsingResult:
    """Parsing result with a normal face and no prohibited occlusions."""
    return _make_parsing_result(
        {
            FacePart.SKIN: 40,
            FacePart.LEFT_EYE: 5,
            FacePart.RIGHT_EYE: 5,
        }
    )


@pytest.fixture
def hat_parsing_result() -> FaceParsingResult:
    """Parsing result containing a prohibited HAT region."""
    return _make_parsing_result(
        {
            FacePart.SKIN: 40,
            FacePart.HAT: 10,
        }
    )


@pytest.fixture
def glasses_parsing_result() -> FaceParsingResult:
    """Parsing result containing eyeglasses only (must be allowed)."""
    return _make_parsing_result(
        {
            FacePart.SKIN: 40,
            FacePart.EYE_GLASS: 10,
        }
    )


@pytest.fixture
def hair_parsing_result() -> FaceParsingResult:
    """Parsing result containing hair only (must be allowed)."""
    return _make_parsing_result(
        {
            FacePart.SKIN: 40,
            FacePart.HAIR: 20,
        }
    )


# ----------------------------------------------------------------------
# Input validation
# ----------------------------------------------------------------------

class TestInputValidation:
    def test_none_image_raises_value_error(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate(
                image=None,
                parsing_result=clean_parsing_result,
            )

    def test_non_ndarray_image_raises_type_error(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        with pytest.raises(TypeError):
            validator.validate(
                image="not-an-array",
                parsing_result=clean_parsing_result,
            )

    def test_empty_image_raises_value_error(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate(
                image=np.array([], dtype=np.uint8),
                parsing_result=clean_parsing_result,
            )

    def test_none_parsing_result_raises_value_error(
        self,
        validator: OcclusionValidator,
    ) -> None:
        with pytest.raises(ValueError):
            validator.validate(
                image=_make_image(),
                parsing_result=None,
            )

    def test_invalid_parsing_result_type_raises_type_error(
        self,
        validator: OcclusionValidator,
    ) -> None:
        with pytest.raises(TypeError):
            validator.validate(
                image=_make_image(),
                parsing_result={"not": "a parsing result"},
            )

    def test_face_argument_is_ignored(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        # Passing an arbitrary object as `face` must not affect the result.
        result_without_face = validator.validate(
            image=_make_image(),
            parsing_result=clean_parsing_result,
        )
        result_with_face = validator.validate(
            image=_make_image(),
            face=object(),
            parsing_result=clean_parsing_result,
        )

        assert result_without_face.passed == result_with_face.passed
        assert result_without_face.score == result_with_face.score
        assert result_without_face.message == result_with_face.message


# ----------------------------------------------------------------------
# _find_prohibited_occlusions
# ----------------------------------------------------------------------

class TestFindProhibitedOcclusions:
    def test_returns_empty_tuple_when_clean(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        assert validator._find_prohibited_occlusions(
            parsing_result=clean_parsing_result,
        ) == ()

    def test_detects_hat(
        self,
        validator: OcclusionValidator,
        hat_parsing_result: FaceParsingResult,
    ) -> None:
        assert validator._find_prohibited_occlusions(
            parsing_result=hat_parsing_result,
        ) == (FacePart.HAT,)

    def test_ignores_eyeglasses(
        self,
        validator: OcclusionValidator,
        glasses_parsing_result: FaceParsingResult,
    ) -> None:
        assert validator._find_prohibited_occlusions(
            parsing_result=glasses_parsing_result,
        ) == ()

    def test_ignores_hair(
        self,
        validator: OcclusionValidator,
        hair_parsing_result: FaceParsingResult,
    ) -> None:
        assert validator._find_prohibited_occlusions(
            parsing_result=hair_parsing_result,
        ) == ()

    def test_ignores_hair_and_glasses_together(
        self,
        validator: OcclusionValidator,
    ) -> None:
        parsing_result = _make_parsing_result(
            {
                FacePart.SKIN: 30,
                FacePart.HAIR: 20,
                FacePart.EYE_GLASS: 10,
            }
        )

        assert validator._find_prohibited_occlusions(
            parsing_result=parsing_result,
        ) == ()

    def test_hat_detected_alongside_allowed_parts(
        self,
        validator: OcclusionValidator,
    ) -> None:
        parsing_result = _make_parsing_result(
            {
                FacePart.SKIN: 20,
                FacePart.HAIR: 20,
                FacePart.EYE_GLASS: 10,
                FacePart.HAT: 10,
            }
        )

        assert validator._find_prohibited_occlusions(
            parsing_result=parsing_result,
        ) == (FacePart.HAT,)


# ----------------------------------------------------------------------
# _compute_score
# ----------------------------------------------------------------------

class TestComputeScore:
    def test_perfect_score_when_no_occlusions(
        self,
        validator: OcclusionValidator,
    ) -> None:
        assert validator._compute_score(prohibited_occlusions=()) == 1.0

    def test_score_drops_when_hat_detected(
        self,
        validator: OcclusionValidator,
    ) -> None:
        score = validator._compute_score(
            prohibited_occlusions=(FacePart.HAT,),
        )

        assert score < 1.0
        assert score == pytest.approx(0.0)

    def test_score_is_float(
        self,
        validator: OcclusionValidator,
    ) -> None:
        score = validator._compute_score(prohibited_occlusions=())
        assert isinstance(score, float)


# ----------------------------------------------------------------------
# _build_message
# ----------------------------------------------------------------------

class TestBuildMessage:
    def test_message_when_clean(
        self,
        validator: OcclusionValidator,
    ) -> None:
        assert validator._build_message(
            prohibited_occlusions=(),
        ) == "No prohibited occlusions detected."

    def test_message_when_hat_detected(
        self,
        validator: OcclusionValidator,
    ) -> None:
        assert validator._build_message(
            prohibited_occlusions=(FacePart.HAT,),
        ) == "Hat detected."


# ----------------------------------------------------------------------
# _describe_part
# ----------------------------------------------------------------------

class TestDescribePart:
    def test_single_word_part_name(
        self,
        validator: OcclusionValidator,
    ) -> None:
        assert validator._describe_part(FacePart.HAT) == "Hat"

    def test_multi_word_part_name(
        self,
        validator: OcclusionValidator,
    ) -> None:
        assert validator._describe_part(FacePart.LEFT_EAR) == "Left Ear"

    def test_upper_lip_part_name(
        self,
        validator: OcclusionValidator,
    ) -> None:
        assert validator._describe_part(FacePart.UPPER_LIP) == "Upper Lip"


# ----------------------------------------------------------------------
# Full validate() workflow
# ----------------------------------------------------------------------

class TestValidateWorkflow:
    def test_clean_image_passes_with_perfect_score(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        result = validator.validate(
            image=_make_image(),
            parsing_result=clean_parsing_result,
        )

        assert result.type == ValidationType.OCCLUSION
        assert result.passed is True
        assert result.score == 1.0
        assert result.message == "No prohibited occlusions detected."

    def test_hat_fails_validation(
        self,
        validator: OcclusionValidator,
        hat_parsing_result: FaceParsingResult,
    ) -> None:
        result = validator.validate(
            image=_make_image(),
            parsing_result=hat_parsing_result,
        )

        assert result.type == ValidationType.OCCLUSION
        assert result.passed is False
        assert result.score == pytest.approx(0.0)
        assert result.message == "Hat detected."

    def test_glasses_alone_pass_validation(
        self,
        validator: OcclusionValidator,
        glasses_parsing_result: FaceParsingResult,
    ) -> None:
        result = validator.validate(
            image=_make_image(),
            parsing_result=glasses_parsing_result,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_hair_alone_passes_validation(
        self,
        validator: OcclusionValidator,
        hair_parsing_result: FaceParsingResult,
    ) -> None:
        result = validator.validate(
            image=_make_image(),
            parsing_result=hair_parsing_result,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_hat_with_allowed_accessories_still_fails(
        self,
        validator: OcclusionValidator,
    ) -> None:
        parsing_result = _make_parsing_result(
            {
                FacePart.SKIN: 20,
                FacePart.HAIR: 20,
                FacePart.EYE_GLASS: 10,
                FacePart.HAT: 10,
            }
        )

        result = validator.validate(
            image=_make_image(),
            parsing_result=parsing_result,
        )

        assert result.passed is False
        assert result.message == "Hat detected."

    def test_returns_validation_metric_instance(
        self,
        validator: OcclusionValidator,
        clean_parsing_result: FaceParsingResult,
    ) -> None:
        result = validator.validate(
            image=_make_image(),
            parsing_result=clean_parsing_result,
        )

        assert 0.0 <= result.score <= 1.0
        assert isinstance(result.passed, bool)
        assert isinstance(result.message, str)


# ----------------------------------------------------------------------
# Boundary cases
# ----------------------------------------------------------------------

class TestBoundaryCases:
    def test_single_pixel_of_hat_is_detected(
        self,
        validator: OcclusionValidator,
    ) -> None:
        parsing_result = _make_parsing_result(
            {
                FacePart.SKIN: 99,
                FacePart.HAT: 1,
            }
        )

        result = validator.validate(
            image=_make_image(),
            parsing_result=parsing_result,
        )

        assert result.passed is False

    def test_entirely_background_mask_passes(
        self,
        validator: OcclusionValidator,
    ) -> None:
        parsing_result = _make_parsing_result({})

        result = validator.validate(
            image=_make_image(),
            parsing_result=parsing_result,
        )

        assert result.passed is True
        assert result.score == 1.0

    def test_1x1_image_and_mask_are_handled(
        self,
        validator: OcclusionValidator,
    ) -> None:
        image = np.zeros((1, 1, 3), dtype=np.uint8)
        mask = np.array([[int(FacePart.HAT)]], dtype=np.int32)
        parsing_result = FaceParsingResult(
            mask=mask,
            image_height=1,
            image_width=1,
        )

        result = validator.validate(
            image=image,
            parsing_result=parsing_result,
        )

        assert result.passed is False
        assert result.message == "Hat detected."