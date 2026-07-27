"""Tests for the FaceVisibilityValidator."""

import numpy as np
import pytest

from config.constants import (
    FACE_VISIBILITY_MIN_PART_RATIOS,
    FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR,
    FACE_VISIBILITY_REQUIRED_PARTS,
)
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.face_visibility_validator import FaceVisibilityValidator

_IMAGE_HEIGHT = 100
_IMAGE_WIDTH = 100
_TOTAL_PIXELS = _IMAGE_HEIGHT * _IMAGE_WIDTH

_TOTAL_REQUIRED_PARTS = len(FACE_VISIBILITY_REQUIRED_PARTS)
_PART_WEIGHT = 1.0 / _TOTAL_REQUIRED_PARTS

# Exact pixel counts corresponding to each part's configured minimum ratio,
# derived directly from constants.py so these tests stay correct even if
# the thresholds are recalibrated later.
_THRESHOLD_PIXELS: dict[FacePart, int] = {
    part: round(FACE_VISIBILITY_MIN_PART_RATIOS[part] * _TOTAL_PIXELS)
    for part in FACE_VISIBILITY_REQUIRED_PARTS
}

# The boundary tests below rely on _THRESHOLD_PIXELS[part] / _TOTAL_PIXELS
# reproducing FACE_VISIBILITY_MIN_PART_RATIOS[part] *exactly*, and on that
# count being at least 1 pixel (so "threshold - 1" is a meaningfully smaller,
# still-non-negative count). Both hold today only because every configured
# ratio is a clean multiple of 1 / _TOTAL_PIXELS. If constants.py is ever
# recalibrated to a ratio that isn't (e.g. 0.00033), `round()` would silently
# shift the "exact threshold" test onto the wrong pixel count, or a ratio
# rounding to 0 pixels would make "just below threshold" indistinguishable
# from "missing". Fail loudly here instead of letting a boundary test pass
# or fail for the wrong reason.
for _part in FACE_VISIBILITY_REQUIRED_PARTS:
    assert _THRESHOLD_PIXELS[_part] >= 1, (
        f"{_part.name}'s minimum ratio rounds to 0 pixels at "
        f"_TOTAL_PIXELS={_TOTAL_PIXELS}; the 'just below threshold' boundary "
        "tests need a positive pixel count to be meaningful. Increase "
        "_TOTAL_PIXELS."
    )
    assert _THRESHOLD_PIXELS[_part] / _TOTAL_PIXELS == pytest.approx(
        FACE_VISIBILITY_MIN_PART_RATIOS[_part],
        abs=1e-9,
    ), (
        f"{_part.name}'s minimum ratio "
        f"({FACE_VISIBILITY_MIN_PART_RATIOS[_part]}) is not exactly "
        f"representable at _TOTAL_PIXELS={_TOTAL_PIXELS}; the 'exactly at "
        "threshold' boundary test would silently exercise a rounded value "
        "instead of the real configured threshold. Increase _TOTAL_PIXELS "
        "or adjust the ratio's precision."
    )

_EXPECTED_LABELS: dict[FacePart, str] = {
    FacePart.LEFT_EYE: "Left eye",
    FacePart.RIGHT_EYE: "Right eye",
    FacePart.LEFT_BROW: "Left eyebrow",
    FacePart.RIGHT_BROW: "Right eyebrow",
    FacePart.NOSE: "Nose",
    FacePart.MOUTH: "Mouth",
    FacePart.UPPER_LIP: "Upper lip",
    FacePart.LOWER_LIP: "Lower lip",
}


def _build_parsing_result(
    part_pixel_counts: dict[FacePart, int],
) -> FaceParsingResult:
    """Build a real FaceParsingResult from explicit per-part pixel counts.

    Parts not present in *part_pixel_counts* (or given a count of 0) are
    entirely absent from the mask, so ``has_part`` returns False for them.
    All remaining pixels default to FacePart.BACKGROUND.
    """
    flat_mask = np.zeros(_TOTAL_PIXELS, dtype=np.int32)
    offset = 0

    for part, count in part_pixel_counts.items():
        if count <= 0:
            continue

        assert offset + count <= _TOTAL_PIXELS, (
            f"Test setup error: requested {offset + count} pixels across "
            f"parts, but the synthetic image only has {_TOTAL_PIXELS}. "
            "Assigning would silently truncate this part's pixel count "
            "instead of raising, masking the mistake. Shrink the requested "
            "counts or enlarge _IMAGE_HEIGHT / _IMAGE_WIDTH."
        )

        flat_mask[offset:offset + count] = int(part)
        offset += count

    mask = flat_mask.reshape(_IMAGE_HEIGHT, _IMAGE_WIDTH)

    return FaceParsingResult(
        mask=mask,
        image_height=_IMAGE_HEIGHT,
        image_width=_IMAGE_WIDTH,
    )


def _all_sufficient_counts(margin: int = 5) -> dict[FacePart, int]:
    """Pixel counts for every mandatory part, comfortably above threshold."""
    return {
        part: _THRESHOLD_PIXELS[part] + margin
        for part in FACE_VISIBILITY_REQUIRED_PARTS
    }


def _expected_score(
    missing_count: int,
    insufficient_count: int,
) -> float:
    """Compute the expected visibility score for given failure counts."""
    score = (
        1.0
        - missing_count * _PART_WEIGHT
        - insufficient_count * _PART_WEIGHT * FACE_VISIBILITY_PARTIAL_PENALTY_FACTOR
    )

    return max(
        min(
            score,
            1.0,
        ),
        0.0,
    )


def _assert_normalized_score(score: float) -> None:
    """Assert that a validation score is normalized."""
    assert 0.0 <= score <= 1.0


@pytest.fixture
def validator() -> FaceVisibilityValidator:
    """Create a FaceVisibilityValidator instance."""
    return FaceVisibilityValidator()


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a deterministic placeholder BGR image."""
    return np.zeros(
        (
            _IMAGE_HEIGHT,
            _IMAGE_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )


@pytest.fixture
def all_visible_parsing_result() -> FaceParsingResult:
    """A parsing result where every mandatory part is sufficiently visible."""
    return _build_parsing_result(_all_sufficient_counts())


# ================================================================== #
# Input validation
# ================================================================== #


def test_validate_none_image_raises_value_error(
    validator: FaceVisibilityValidator,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that None image input raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=None,
            parsing_result=all_visible_parsing_result,
        )


def test_validate_non_numpy_image_raises_type_error(
    validator: FaceVisibilityValidator,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that non-NumPy image input raises TypeError."""
    # Arrange
    image = [
        [
            0,
            255,
        ],
    ]

    # Act & Assert
    with pytest.raises(TypeError):
        validator.validate(
            image=image,
            parsing_result=all_visible_parsing_result,
        )


def test_validate_empty_image_raises_value_error(
    validator: FaceVisibilityValidator,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that an empty NumPy image raises ValueError."""
    # Arrange
    image = np.array(
        [],
        dtype=np.uint8,
    )

    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=image,
            parsing_result=all_visible_parsing_result,
        )


def test_validate_none_parsing_result_raises_value_error(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that a None parsing result raises ValueError."""
    # Act & Assert
    with pytest.raises(ValueError):
        validator.validate(
            image=sample_image,
            parsing_result=None,
        )


@pytest.mark.parametrize(
    "invalid_parsing_result",
    [
        pytest.param(
            {
                "left_eye": 15,
            },
            id="plain_dict",
        ),
        pytest.param(
            "not-a-parsing-result",
            id="string",
        ),
        pytest.param(
            12345,
            id="integer",
        ),
    ],
)
def test_validate_invalid_parsing_result_type_raises_type_error(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    invalid_parsing_result,
):
    """Verify that a non-FaceParsingResult parsing_result raises TypeError."""
    # Act & Assert
    with pytest.raises(TypeError):
        validator.validate(
            image=sample_image,
            parsing_result=invalid_parsing_result,
        )


def test_validate_ignores_face_argument(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that `face` has no effect on the result, per its documented
    role as an unused parameter kept only for signature compatibility with
    BaseValidator."""
    # Arrange
    baseline = validator.validate(
        image=sample_image,
        parsing_result=all_visible_parsing_result,
    )

    # Act
    with_face = validator.validate(
        image=sample_image,
        face=object(),
        parsing_result=all_visible_parsing_result,
    )

    # Assert
    assert with_face.passed == baseline.passed
    assert with_face.score == pytest.approx(baseline.score)
    assert with_face.message == baseline.message


# ================================================================== #
# validate() — all parts visible
# ================================================================== #


def test_validate_all_parts_visible_returns_validation_metric(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that a fully visible face returns a correctly typed metric."""
    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=all_visible_parsing_result,
    )

    # Assert
    assert isinstance(
        metric,
        ValidationMetric,
    )
    assert metric.type == ValidationType.FACE_VISIBILITY


def test_validate_all_parts_visible_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that a fully visible face passes with a perfect score."""
    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=all_visible_parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)
    assert metric.message == "All required facial features are sufficiently visible."


def test_validate_non_mandatory_parts_do_not_affect_result(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that non-mandatory parts (missing or tiny) are ignored."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[FacePart.HAIR] = 0
    counts[FacePart.EAR_RING] = 1
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)
    assert metric.message == "All required facial features are sufficiently visible."


# ================================================================== #
# validate() — missing parts
# ================================================================== #


@pytest.mark.parametrize(
    "missing_part",
    list(FACE_VISIBILITY_REQUIRED_PARTS),
    ids=lambda part: part.name,
)
def test_validate_single_missing_part_fails_with_expected_message(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    missing_part: FacePart,
):
    """Verify that any single missing mandatory part fails validation."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[missing_part]
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )
    assert metric.message == f"{_EXPECTED_LABELS[missing_part]} is not visible."


def test_validate_multiple_missing_parts_reduces_score_accordingly(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that multiple missing parts each contribute their own penalty."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_BROW]
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=2,
            insufficient_count=0,
        )
    )
    assert metric.message == (
        "Left eye is not visible. Right eyebrow is not visible."
    )


def test_validate_all_parts_missing_clamps_score_to_zero(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that every mandatory part missing clamps the score to 0.0."""
    # Arrange
    parsing_result = _build_parsing_result({})

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(0.0)
    _assert_normalized_score(
        metric.score,
    )


def test_validate_only_one_visible_part_reports_all_others_missing(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify correct behaviour when only a single mandatory part is present."""
    # Arrange
    counts = {FacePart.NOSE: _THRESHOLD_PIXELS[FacePart.NOSE] + 5}
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=_TOTAL_REQUIRED_PARTS - 1,
            insufficient_count=0,
        )
    )
    assert "Nose is not visible." not in metric.message


# ================================================================== #
# validate() — insufficient parts
# ================================================================== #


@pytest.mark.parametrize(
    "insufficient_part",
    list(FACE_VISIBILITY_REQUIRED_PARTS),
    ids=lambda part: part.name,
)
def test_validate_single_insufficient_part_fails_with_expected_message(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    insufficient_part: FacePart,
):
    """Verify that a present-but-too-small mandatory part fails validation."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[insufficient_part] = max(
        _THRESHOLD_PIXELS[insufficient_part] - 1,
        1,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=0,
            insufficient_count=1,
        )
    )
    assert metric.message == (
        f"{_EXPECTED_LABELS[insufficient_part]} visibility is below "
        "the required threshold."
    )


def test_validate_multiple_insufficient_parts_reduces_score_accordingly(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that multiple insufficient parts each contribute their own penalty."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[FacePart.MOUTH] = max(
        _THRESHOLD_PIXELS[FacePart.MOUTH] - 1,
        1,
    )
    counts[FacePart.UPPER_LIP] = max(
        _THRESHOLD_PIXELS[FacePart.UPPER_LIP] - 1,
        1,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=0,
            insufficient_count=2,
        )
    )
    assert metric.message == (
        "Mouth visibility is below the required threshold. "
        "Upper lip visibility is below the required threshold."
    )


# ================================================================== #
# validate() — missing and insufficient combined
# ================================================================== #


def test_validate_missing_and_insufficient_parts_combined(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify combined missing/insufficient failures produce the correct
    score and a concatenated, ordered message."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.LEFT_BROW]
    counts[FacePart.LOWER_LIP] = max(
        _THRESHOLD_PIXELS[FacePart.LOWER_LIP] - 1,
        1,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=1,
        )
    )
    assert metric.message == (
        "Left eyebrow is not visible. "
        "Lower lip visibility is below the required threshold."
    )


def test_validate_missing_part_is_not_double_penalized_as_insufficient(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that a missing part is never also reported as insufficient,
    since its ratio (0.0) would otherwise also fail the ratio check."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.message.count("Right eye") == 1
    assert "Right eye is not visible." in metric.message
    assert "Right eye visibility is below the required threshold." not in metric.message
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )


# ================================================================== #
# Boundary tests around minimum visibility ratios
# ================================================================== #


@pytest.mark.parametrize(
    "boundary_part",
    list(FACE_VISIBILITY_REQUIRED_PARTS),
    ids=lambda part: part.name,
)
def test_validate_exactly_at_threshold_is_sufficient(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    boundary_part: FacePart,
):
    """Verify that a ratio exactly equal to the threshold is considered
    sufficiently visible (the comparison is strictly-less-than)."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[boundary_part] = _THRESHOLD_PIXELS[boundary_part]
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "boundary_part",
    list(FACE_VISIBILITY_REQUIRED_PARTS),
    ids=lambda part: part.name,
)
def test_validate_just_below_threshold_is_insufficient(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    boundary_part: FacePart,
):
    """Verify that a ratio one pixel below the threshold fails the check."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[boundary_part] = _THRESHOLD_PIXELS[boundary_part] - 1
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=0,
            insufficient_count=1,
        )
    )


@pytest.mark.parametrize(
    "boundary_part",
    list(FACE_VISIBILITY_REQUIRED_PARTS),
    ids=lambda part: part.name,
)
def test_validate_just_above_threshold_is_sufficient(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    boundary_part: FacePart,
):
    """Verify that a ratio one pixel above the threshold passes the check."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[boundary_part] = _THRESHOLD_PIXELS[boundary_part] + 1
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


# ================================================================== #
# Score normalization
# ================================================================== #


def test_validate_score_never_negative(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that the score cannot go below 0.0 even with maximal failures."""
    # Arrange
    parsing_result = _build_parsing_result({})

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.score >= 0.0


def test_validate_score_never_exceeds_one(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that the score cannot exceed 1.0 for a fully visible face."""
    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=all_visible_parsing_result,
    )

    # Assert
    assert metric.score <= 1.0


@pytest.mark.parametrize(
    "missing_count, insufficient_count",
    [
        pytest.param(0, 0, id="no_failures"),
        pytest.param(1, 0, id="one_missing"),
        pytest.param(0, 1, id="one_insufficient"),
        pytest.param(3, 2, id="mixed_failures"),
        pytest.param(8, 0, id="all_missing"),
    ],
)
def test_validate_score_is_always_normalized(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    missing_count: int,
    insufficient_count: int,
):
    """Verify that the score stays within [0.0, 1.0] across representative
    combinations of missing and insufficient parts."""
    # Arrange
    required_parts = list(FACE_VISIBILITY_REQUIRED_PARTS)
    counts = _all_sufficient_counts()

    for part in required_parts[:missing_count]:
        del counts[part]

    for part in required_parts[missing_count:missing_count + insufficient_count]:
        counts[part] = max(
            _THRESHOLD_PIXELS[part] - 1,
            1,
        )

    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    _assert_normalized_score(
        metric.score,
    )


# ================================================================== #
# Helper method: _find_missing_parts()
# ================================================================== #


def test_find_missing_parts_returns_empty_when_all_present(
    validator: FaceVisibilityValidator,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that no parts are reported missing when all are present."""
    # Act
    missing_parts = validator._find_missing_parts(
        parsing_result=all_visible_parsing_result,
    )

    # Assert
    assert missing_parts == ()


def test_find_missing_parts_detects_absent_parts(
    validator: FaceVisibilityValidator,
):
    """Verify that absent mandatory parts are correctly identified."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.NOSE]
    del counts[FacePart.MOUTH]
    parsing_result = _build_parsing_result(counts)

    # Act
    missing_parts = validator._find_missing_parts(
        parsing_result=parsing_result,
    )

    # Assert
    assert set(missing_parts) == {FacePart.NOSE, FacePart.MOUTH}


def test_find_missing_parts_ignores_non_mandatory_parts(
    validator: FaceVisibilityValidator,
):
    """Verify that a missing non-mandatory part is never reported."""
    # Arrange
    counts = _all_sufficient_counts()
    parsing_result = _build_parsing_result(counts)

    # Act
    missing_parts = validator._find_missing_parts(
        parsing_result=parsing_result,
    )

    # Assert
    assert FacePart.HAIR not in missing_parts
    assert FacePart.EAR_RING not in missing_parts


# ================================================================== #
# Helper method: _find_insufficient_parts()
# ================================================================== #


def test_find_insufficient_parts_returns_empty_when_all_sufficient(
    validator: FaceVisibilityValidator,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that no parts are reported insufficient when all meet threshold."""
    # Act
    insufficient_parts = validator._find_insufficient_parts(
        parsing_result=all_visible_parsing_result,
        missing_parts=(),
    )

    # Assert
    assert insufficient_parts == ()


def test_find_insufficient_parts_detects_undersized_parts(
    validator: FaceVisibilityValidator,
):
    """Verify that present-but-too-small mandatory parts are identified."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[FacePart.LEFT_BROW] = max(
        _THRESHOLD_PIXELS[FacePart.LEFT_BROW] - 1,
        1,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    insufficient_parts = validator._find_insufficient_parts(
        parsing_result=parsing_result,
        missing_parts=(),
    )

    # Assert
    assert insufficient_parts == (FacePart.LEFT_BROW,)


def test_find_insufficient_parts_excludes_missing_parts(
    validator: FaceVisibilityValidator,
):
    """Verify that a part already reported as missing is excluded here,
    preventing the same failure from being counted twice."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.RIGHT_BROW]
    parsing_result = _build_parsing_result(counts)

    # Act
    missing_parts = validator._find_missing_parts(
        parsing_result=parsing_result,
    )
    insufficient_parts = validator._find_insufficient_parts(
        parsing_result=parsing_result,
        missing_parts=missing_parts,
    )

    # Assert
    assert FacePart.RIGHT_BROW in missing_parts
    assert FacePart.RIGHT_BROW not in insufficient_parts


# ================================================================== #
# Helper method: _compute_score()
# ================================================================== #


@pytest.mark.parametrize(
    "missing_parts, insufficient_parts",
    [
        pytest.param((), (), id="no_failures"),
        pytest.param(
            (FacePart.LEFT_EYE,),
            (),
            id="one_missing",
        ),
        pytest.param(
            (),
            (FacePart.NOSE,),
            id="one_insufficient",
        ),
        pytest.param(
            (FacePart.LEFT_EYE, FacePart.RIGHT_EYE),
            (FacePart.NOSE,),
            id="mixed_failures",
        ),
        pytest.param(
            tuple(FACE_VISIBILITY_REQUIRED_PARTS),
            (),
            id="all_missing",
        ),
    ],
)
def test_compute_score_matches_expected_formula(
    validator: FaceVisibilityValidator,
    missing_parts: tuple[FacePart, ...],
    insufficient_parts: tuple[FacePart, ...],
):
    """Verify that _compute_score applies the documented weighting formula."""
    # Act
    score = validator._compute_score(
        missing_parts=missing_parts,
        insufficient_parts=insufficient_parts,
    )

    # Assert
    assert score == pytest.approx(
        _expected_score(
            missing_count=len(missing_parts),
            insufficient_count=len(insufficient_parts),
        )
    )
    _assert_normalized_score(score)


def test_compute_score_clamps_at_zero_for_excessive_penalties(
    validator: FaceVisibilityValidator,
):
    """Verify that combined missing and insufficient penalties cannot
    push the score below 0.0."""
    # Arrange
    missing_parts = tuple(FACE_VISIBILITY_REQUIRED_PARTS)
    insufficient_parts = ()

    # Act
    score = validator._compute_score(
        missing_parts=missing_parts,
        insufficient_parts=insufficient_parts,
    )

    # Assert
    assert score == pytest.approx(0.0)


# ================================================================== #
# Helper method: _build_message()
# ================================================================== #


def test_build_message_all_visible(
    validator: FaceVisibilityValidator,
):
    """Verify the success message when there are no failures."""
    # Act
    message = validator._build_message(
        missing_parts=(),
        insufficient_parts=(),
    )

    # Assert
    assert message == "All required facial features are sufficiently visible."


def test_build_message_single_missing_part(
    validator: FaceVisibilityValidator,
):
    """Verify the message for a single missing part."""
    # Act
    message = validator._build_message(
        missing_parts=(FacePart.LEFT_EYE,),
        insufficient_parts=(),
    )

    # Assert
    assert message == "Left eye is not visible."


def test_build_message_single_insufficient_part(
    validator: FaceVisibilityValidator,
):
    """Verify the message for a single insufficient part."""
    # Act
    message = validator._build_message(
        missing_parts=(),
        insufficient_parts=(FacePart.RIGHT_BROW,),
    )

    # Assert
    assert message == "Right eyebrow visibility is below the required threshold."


def test_build_message_orders_missing_before_insufficient(
    validator: FaceVisibilityValidator,
):
    """Verify that missing-part messages are concatenated before
    insufficient-part messages, regardless of argument content."""
    # Act
    message = validator._build_message(
        missing_parts=(FacePart.NOSE,),
        insufficient_parts=(FacePart.MOUTH,),
    )

    # Assert
    assert message == (
        "Nose is not visible. Mouth visibility is below the required threshold."
    )


def test_build_message_combines_multiple_failures_of_each_kind(
    validator: FaceVisibilityValidator,
):
    """Verify a concise combined message for several failures of each kind."""
    # Act
    message = validator._build_message(
        missing_parts=(FacePart.LEFT_EYE, FacePart.RIGHT_EYE),
        insufficient_parts=(FacePart.UPPER_LIP, FacePart.LOWER_LIP),
    )

    # Assert
    assert message == (
        "Left eye is not visible. "
        "Right eye is not visible. "
        "Upper lip visibility is below the required threshold. "
        "Lower lip visibility is below the required threshold."
    )


# ================================================================== #
# Helper method: _describe_part()
# ================================================================== #


@pytest.mark.parametrize(
    "part",
    list(FACE_VISIBILITY_REQUIRED_PARTS),
    ids=lambda part: part.name,
)
def test_describe_part_returns_expected_label(
    validator: FaceVisibilityValidator,
    part: FacePart,
):
    """Verify that every mandatory part maps to its documented display label."""
    # Act
    label = validator._describe_part(part)

    # Assert
    assert label == _EXPECTED_LABELS[part]