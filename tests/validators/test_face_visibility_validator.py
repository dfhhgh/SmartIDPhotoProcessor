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

# The mouth region counts as a single mandatory semantic unit alongside the
# individual required parts, giving 6 total checkable regions.
_MOUTH_REGION_PART_COUNT = 1
_TOTAL_REQUIRED_PARTS = len(FACE_VISIBILITY_REQUIRED_PARTS) + _MOUTH_REGION_PART_COUNT
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

# Mouth-region threshold pixels (used by the composite check).
_MOUTH_THRESHOLD_PIXELS = round(
    FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.MOUTH] * _TOTAL_PIXELS
)
_UPPER_LIP_THRESHOLD_PIXELS = round(
    FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.UPPER_LIP] * _TOTAL_PIXELS
)
_LOWER_LIP_THRESHOLD_PIXELS = round(
    FACE_VISIBILITY_MIN_PART_RATIOS[FacePart.LOWER_LIP] * _TOTAL_PIXELS
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
    """Pixel counts for every mandatory part, comfortably above threshold.

    Includes MOUTH so the composite mouth-region check passes via Case 1.
    """
    counts = {
        part: _THRESHOLD_PIXELS[part] + margin
        for part in FACE_VISIBILITY_REQUIRED_PARTS
    }
    # Satisfy the composite mouth-region check (Case 1: MOUTH above threshold).
    counts[FacePart.MOUTH] = _MOUTH_THRESHOLD_PIXELS + margin
    return counts


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


def test_validate_face_argument_affects_result_with_invalid_face(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that the `face` argument is used for landmark-based eye
    validation. An invalid face object (no kps) should not alter the result
    when all parts are already detected by the parser."""
    # Arrange
    baseline = validator.validate(
        image=sample_image,
        parsing_result=all_visible_parsing_result,
    )

    # Act
    with_invalid_face = validator.validate(
        image=sample_image,
        face=object(),
        parsing_result=all_visible_parsing_result,
    )

    # Assert
    assert with_invalid_face.passed == baseline.passed
    assert with_invalid_face.score == pytest.approx(baseline.score)
    assert with_invalid_face.message == baseline.message


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
    counts[FacePart.LEFT_BROW] = max(
        _THRESHOLD_PIXELS[FacePart.LEFT_BROW] - 1,
        1,
    )
    counts[FacePart.RIGHT_BROW] = max(
        _THRESHOLD_PIXELS[FacePart.RIGHT_BROW] - 1,
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
        "Left eyebrow visibility is below the required threshold. "
        "Right eyebrow visibility is below the required threshold."
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
    counts[FacePart.RIGHT_BROW] = max(
        _THRESHOLD_PIXELS[FacePart.RIGHT_BROW] - 1,
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
        "Right eyebrow visibility is below the required threshold."
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
    del counts[FacePart.LEFT_EYE]
    parsing_result = _build_parsing_result(counts)

    # Act
    missing_parts = validator._find_missing_parts(
        parsing_result=parsing_result,
    )

    # Assert
    assert set(missing_parts) == {FacePart.NOSE, FacePart.LEFT_EYE}


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


def test_find_insufficient_parts_excludes_landmark_overridden_parts(
    validator: FaceVisibilityValidator,
):
    """Verify that parts overridden by valid landmarks are excluded from
    the insufficient-parts check, just like missing parts."""
    # Arrange
    counts = _all_sufficient_counts()
    counts[FacePart.LEFT_EYE] = 0
    counts[FacePart.RIGHT_EYE] = 0
    parsing_result = _build_parsing_result(counts)

    # Act
    insufficient_parts = validator._find_insufficient_parts(
        parsing_result=parsing_result,
        missing_parts=(),
        landmark_overridden_parts=frozenset({FacePart.LEFT_EYE, FacePart.RIGHT_EYE}),
    )

    # Assert
    assert FacePart.LEFT_EYE not in insufficient_parts
    assert FacePart.RIGHT_EYE not in insufficient_parts


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
    # Include MOUTH to represent the missing mouth region (6th check).
    missing_parts = tuple(FACE_VISIBILITY_REQUIRED_PARTS) + (FacePart.MOUTH,)
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


# ================================================================== #
# Mouth region composite check (regression)
# ================================================================== #


def _counts_with_mouth_region(
    *,
    mouth: int = 0,
    upper_lip: int = 0,
    lower_lip: int = 0,
    margin: int = 5,
) -> dict[FacePart, int]:
    """Build pixel counts with explicit mouth-region control.

    All non-mouth required parts are set above threshold. The mouth
    region is configured via the keyword arguments.
    """
    counts = {
        part: _THRESHOLD_PIXELS[part] + margin
        for part in FACE_VISIBILITY_REQUIRED_PARTS
    }
    if mouth > 0:
        counts[FacePart.MOUTH] = mouth
    if upper_lip > 0:
        counts[FacePart.UPPER_LIP] = upper_lip
    if lower_lip > 0:
        counts[FacePart.LOWER_LIP] = lower_lip
    return counts


def test_validate_closed_mouth_with_both_lips_visible_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: closed mouth with no MOUTH but both lips above threshold
    must pass validation."""
    # Arrange
    counts = _counts_with_mouth_region(
        upper_lip=_UPPER_LIP_THRESHOLD_PIXELS + 5,
        lower_lip=_LOWER_LIP_THRESHOLD_PIXELS + 5,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


def test_validate_open_mouth_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: open mouth with MOUTH present passes validation."""
    # Arrange
    counts = _counts_with_mouth_region(
        mouth=_MOUTH_THRESHOLD_PIXELS + 5,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


def test_validate_mouth_region_alone_above_threshold_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: MOUTH above threshold passes even without lips."""
    # Arrange
    counts = _counts_with_mouth_region(
        mouth=_MOUTH_THRESHOLD_PIXELS + 5,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert "Mouth" not in metric.message or metric.passed


def test_validate_mask_covering_mouth_fails(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: no MOUTH and no lips must fail with 'Mouth is not visible.'"""
    # Arrange — no mouth region parts at all
    counts = _counts_with_mouth_region()
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Mouth is not visible." in metric.message


def test_validate_cropped_lower_face_fails(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: cropped lower face with no mouth region fails."""
    # Arrange — only one lip present (partial crop)
    counts = _counts_with_mouth_region(
        upper_lip=_UPPER_LIP_THRESHOLD_PIXELS + 5,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Mouth is not visible." in metric.message


def test_validate_only_one_lip_visible_fails(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: only UPPER_LIP or only LOWER_LIP must fail."""
    # Arrange — only upper lip
    counts = _counts_with_mouth_region(
        upper_lip=_UPPER_LIP_THRESHOLD_PIXELS + 5,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Mouth is not visible." in metric.message


def test_validate_lips_below_threshold_fail(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: both lips present but below threshold must fail."""
    # Arrange
    counts = _counts_with_mouth_region(
        upper_lip=max(_UPPER_LIP_THRESHOLD_PIXELS - 1, 1),
        lower_lip=max(_LOWER_LIP_THRESHOLD_PIXELS - 1, 1),
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Mouth is not visible." in metric.message


def test_validate_mouth_region_exact_threshold_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Regression: lips at exact threshold must pass (>= comparison)."""
    # Arrange — MOUTH absent, both lips at exact threshold
    counts = _counts_with_mouth_region(
        upper_lip=_UPPER_LIP_THRESHOLD_PIXELS,
        lower_lip=_LOWER_LIP_THRESHOLD_PIXELS,
    )
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


def test_validate_mouth_region_score_penalty(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that a missing mouth region deducts exactly one part's weight."""
    # Arrange — all required parts present, mouth region missing
    counts = {
        part: _THRESHOLD_PIXELS[part] + 5
        for part in FACE_VISIBILITY_REQUIRED_PARTS
    }
    # No MOUTH, no lips → mouth region fails
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert metric.score == pytest.approx(
        _expected_score(missing_count=1, insufficient_count=0)
    )
    assert "Mouth is not visible." in metric.message


# ================================================================== #
# Landmark-based eye visibility override
# ================================================================== #


def _make_face_with_valid_landmarks() -> "object":
    """Create a mock Face object with valid InsightFace landmarks."""
    import types

    face = types.SimpleNamespace()
    face.kps = np.array(
        [
            [50.0, 50.0],
            [80.0, 50.0],
            [65.0, 70.0],
            [55.0, 85.0],
            [75.0, 85.0],
        ],
        dtype=np.float32,
    )
    return face


def _make_face_with_nan_landmarks() -> "object":
    """Create a mock Face object with NaN in eye landmark."""
    import types

    face = types.SimpleNamespace()
    face.kps = np.array(
        [
            [np.nan, 50.0],
            [80.0, 50.0],
            [65.0, 70.0],
            [55.0, 85.0],
            [75.0, 85.0],
        ],
        dtype=np.float32,
    )
    return face


def _make_face_with_inf_landmarks() -> "object":
    """Create a mock Face object with Inf in eye landmark."""
    import types

    face = types.SimpleNamespace()
    face.kps = np.array(
        [
            [np.inf, 50.0],
            [80.0, 50.0],
            [65.0, 70.0],
            [55.0, 85.0],
            [75.0, 85.0],
        ],
        dtype=np.float32,
    )
    return face


def _make_face_with_wrong_shape_kps() -> "object":
    """Create a mock Face object with incorrectly shaped landmarks."""
    import types

    face = types.SimpleNamespace()
    face.kps = np.zeros((3, 2), dtype=np.float32)
    return face


def _all_sufficient_counts_with_eye_glass(
    margin: int = 5,
) -> dict[FacePart, int]:
    """Pixel counts for every mandatory part plus EYE_GLASS."""
    counts = _all_sufficient_counts(margin=margin)
    counts[FacePart.EYE_GLASS] = 500
    return counts


# ------------------------------------------------------------------ #
# Parser detects eyes: no override needed
# ------------------------------------------------------------------ #


def test_validate_parser_detects_eyes_without_glasses_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that eyes detected by the parser pass without needing landmarks."""
    # Arrange
    counts = _all_sufficient_counts()
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


def test_validate_parser_detects_eyes_with_glasses_passes(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Verify that eyes detected by the parser pass even when glasses are present."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Prescription glasses: parser misses eyes + landmarks override
# ------------------------------------------------------------------ #


def test_validate_prescription_glasses_both_eyes_missed_but_landmarks_valid(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes due to transparent prescription glasses
    but EYE_GLASS is present and valid landmarks exist, both eyes should
    be treated as visible."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


def test_validate_prescription_glasses_left_eye_missed_landmark_valid(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses only the left eye due to prescription glasses
    but EYE_GLASS is present and a valid landmark exists, the left eye
    should be treated as visible."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


def test_validate_prescription_glasses_right_eye_missed_landmark_valid(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses only the right eye due to prescription glasses
    but EYE_GLASS is present and a valid landmark exists, the right eye
    should be treated as visible."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Parser misses eyes + no landmarks: still fails
# ------------------------------------------------------------------ #


def test_validate_parser_misses_eyes_no_glasses_no_landmarks_fails(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes and no glasses are present,
    the eyes should be reported as missing regardless of landmarks."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Left eye is not visible." in metric.message
    assert "Right eye is not visible." in metric.message


def test_validate_parser_misses_eyes_glasses_present_but_no_face(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes, glasses are present, but no face
    object is provided, the eyes should still be reported as missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        face=None,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Left eye is not visible." in metric.message
    assert "Right eye is not visible." in metric.message


def test_validate_parser_misses_eyes_glasses_present_kps_none(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes, glasses are present, but face.kps is
    None, the eyes should still be reported as missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    import types

    face = types.SimpleNamespace()
    face.kps = None

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Left eye is not visible." in metric.message
    assert "Right eye is not visible." in metric.message


def test_validate_parser_misses_eyes_glasses_present_kps_wrong_shape(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes, glasses are present, but face.kps has
    wrong shape, the eyes should still be reported as missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_wrong_shape_kps()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Left eye is not visible." in metric.message
    assert "Right eye is not visible." in metric.message


def test_validate_parser_misses_eyes_glasses_present_kps_nan(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes, glasses are present, but the right eye
    landmark contains NaN (index 0), LEFT_EYE (index 1) is overridden by
    its valid landmark, and only RIGHT_EYE is reported as missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_nan_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Right eye is not visible." in metric.message
    assert "Left eye is not visible." not in metric.message
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )


def test_validate_parser_misses_eyes_glasses_present_kps_inf(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When parser misses both eyes, glasses are present, but the right eye
    landmark contains Inf (index 0), LEFT_EYE (index 1) is overridden by
    its valid landmark, and only RIGHT_EYE is reported as missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_inf_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Right eye is not visible." in metric.message
    assert "Left eye is not visible." not in metric.message
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )


# ------------------------------------------------------------------ #
# EyeGlass present but non-eye part missing: no override
# ------------------------------------------------------------------ #


def test_validate_eye_glass_present_nose_missing_not_overridden(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When EYE_GLASS is present but a non-eye part (nose) is missing,
    the landmark override should not apply to non-eye parts."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.NOSE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Nose is not visible." in metric.message
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )


# ------------------------------------------------------------------ #
# Mixed: one eye landmark valid, one invalid
# ------------------------------------------------------------------ #


def test_validate_one_eye_landmark_valid_one_nan(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When both eyes are missed by parser, glasses are present, but only
    the RIGHT_EYE landmark (index 0) is NaN, only the LEFT_EYE (index 1)
    is overridden. RIGHT_EYE remains missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_nan_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Right eye is not visible." in metric.message
    assert "Left eye is not visible." not in metric.message
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )


def test_validate_one_eye_landmark_valid_one_wrong_shape(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """When both eyes are missed by parser, glasses are present, but the
    RIGHT_EYE landmark (index 0) has NaN coords, only the LEFT_EYE (index 1)
    is overridden. RIGHT_EYE remains missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    import types

    face = types.SimpleNamespace()
    face.kps = np.array(
        [
            [np.nan, np.nan],
            [80.0, 50.0],
            [65.0, 70.0],
            [55.0, 85.0],
            [75.0, 85.0],
        ],
        dtype=np.float32,
    )

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Right eye is not visible." in metric.message
    assert "Left eye is not visible." not in metric.message
    assert metric.score == pytest.approx(
        _expected_score(
            missing_count=1,
            insufficient_count=0,
        )
    )


# ------------------------------------------------------------------ #
# Thick black frames
# ------------------------------------------------------------------ #


def test_validate_thick_black_frames_parser_misses_eyes_landmarks_valid(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Thick black frames cause the parser to miss eyes, but valid
    landmarks confirm they are visible."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Transparent glasses
# ------------------------------------------------------------------ #


def test_validate_transparent_glasses_parser_misses_eyes_landmarks_valid(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Transparent glasses cause the parser to miss eyes, but valid
    landmarks confirm they are visible."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is True
    assert metric.score == pytest.approx(1.0)


# ------------------------------------------------------------------ #
# Sunglasses: should fail (EYE_GLASS present but eyes genuinely occluded)
# ------------------------------------------------------------------ #


def test_validate_sunglasses_parser_misses_eyes_no_landmarks_fails(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Sunglasses cause the parser to miss eyes. EYE_GLASS is present.
    Without valid landmarks, the eyes are correctly reported as missing.
    (Sunglasses rejection is the responsibility of GlassesValidator.)"""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    # Act
    metric = validator.validate(
        image=sample_image,
        face=None,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Left eye is not visible." in metric.message
    assert "Right eye is not visible." in metric.message


def test_validate_sunglasses_parser_misses_eyes_landmarks_missing_fails(
    validator: FaceVisibilityValidator,
    sample_image: np.ndarray,
):
    """Sunglasses cause the parser to miss eyes. EYE_GLASS is present.
    Without valid landmarks, the eyes are correctly reported as missing."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    import types

    face = types.SimpleNamespace()
    face.kps = None

    # Act
    metric = validator.validate(
        image=sample_image,
        face=face,
        parsing_result=parsing_result,
    )

    # Assert
    assert metric.passed is False
    assert "Left eye is not visible." in metric.message
    assert "Right eye is not visible." in metric.message


# ------------------------------------------------------------------ #
# _has_valid_eye_landmark() helper
# ------------------------------------------------------------------ #


def test_has_valid_eye_landmark_returns_false_for_none_face(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns False when face is None."""
    # Act
    result = validator._has_valid_eye_landmark(
        face=None,
        part=FacePart.LEFT_EYE,
    )

    # Assert
    assert result is False


def test_has_valid_eye_landmark_returns_false_when_kps_is_none(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns False when kps is None."""
    # Arrange
    import types

    face = types.SimpleNamespace()
    face.kps = None

    # Act
    result = validator._has_valid_eye_landmark(
        face=face,
        part=FacePart.LEFT_EYE,
    )

    # Assert
    assert result is False


def test_has_valid_eye_landmark_returns_false_when_kps_wrong_shape(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns False for wrong kps shape."""
    # Arrange
    face = _make_face_with_wrong_shape_kps()

    # Act
    result = validator._has_valid_eye_landmark(
        face=face,
        part=FacePart.LEFT_EYE,
    )

    # Assert
    assert result is False


def test_has_valid_eye_landmark_returns_false_for_nan(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns False when eye landmark is NaN."""
    # Arrange
    face = _make_face_with_nan_landmarks()

    # Act
    result = validator._has_valid_eye_landmark(
        face=face,
        part=FacePart.RIGHT_EYE,
    )

    # Assert
    assert result is False


def test_has_valid_eye_landmark_returns_false_for_inf(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns False when eye landmark is Inf."""
    # Arrange
    face = _make_face_with_inf_landmarks()

    # Act
    result = validator._has_valid_eye_landmark(
        face=face,
        part=FacePart.RIGHT_EYE,
    )

    # Assert
    assert result is False


def test_has_valid_eye_landmark_returns_true_for_valid_left_eye(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns True for valid left eye."""
    # Arrange
    face = _make_face_with_valid_landmarks()

    # Act
    result = validator._has_valid_eye_landmark(
        face=face,
        part=FacePart.LEFT_EYE,
    )

    # Assert
    assert result is True


def test_has_valid_eye_landmark_returns_true_for_valid_right_eye(
    validator: FaceVisibilityValidator,
):
    """Verify that _has_valid_eye_landmark returns True for valid right eye."""
    # Arrange
    face = _make_face_with_valid_landmarks()

    # Act
    result = validator._has_valid_eye_landmark(
        face=face,
        part=FacePart.RIGHT_EYE,
    )

    # Assert
    assert result is True


# ------------------------------------------------------------------ #
# _find_missing_parts_with_landmark_override() helper
# ------------------------------------------------------------------ #


def test_find_missing_parts_with_landmark_override_no_parts_missing(
    validator: FaceVisibilityValidator,
    all_visible_parsing_result: FaceParsingResult,
):
    """Verify that no parts are reported missing when all are present."""
    # Act
    missing, overridden = validator._find_missing_parts_with_landmark_override(
        parsing_result=all_visible_parsing_result,
        face=_make_face_with_valid_landmarks(),
    )

    # Assert
    assert missing == ()
    assert overridden == frozenset()


def test_find_missing_parts_with_landmark_override_no_glasses_still_reports_missing(
    validator: FaceVisibilityValidator,
):
    """Verify that without glasses, missing eyes are still reported."""
    # Arrange
    counts = _all_sufficient_counts()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    missing, overridden = validator._find_missing_parts_with_landmark_override(
        parsing_result=parsing_result,
        face=face,
    )

    # Assert
    assert set(missing) == {FacePart.LEFT_EYE, FacePart.RIGHT_EYE}
    assert overridden == frozenset()


def test_find_missing_parts_with_landmark_override_glasses_valid_landmarks_eyes_not_missing(
    validator: FaceVisibilityValidator,
):
    """Verify that with glasses and valid landmarks, missing eyes are
    removed from the missing list."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    missing, overridden = validator._find_missing_parts_with_landmark_override(
        parsing_result=parsing_result,
        face=face,
    )

    # Assert
    assert missing == ()
    assert overridden == frozenset({FacePart.LEFT_EYE, FacePart.RIGHT_EYE})


def test_find_missing_parts_with_landmark_override_glasses_no_landmarks_eyes_missing(
    validator: FaceVisibilityValidator,
):
    """Verify that with glasses but no face, missing eyes are still reported."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.LEFT_EYE]
    del counts[FacePart.RIGHT_EYE]
    parsing_result = _build_parsing_result(counts)

    # Act
    missing, overridden = validator._find_missing_parts_with_landmark_override(
        parsing_result=parsing_result,
        face=None,
    )

    # Assert
    assert set(missing) == {FacePart.LEFT_EYE, FacePart.RIGHT_EYE}
    assert overridden == frozenset()


def test_find_missing_parts_with_landmark_override_glasses_non_eye_parts_not_affected(
    validator: FaceVisibilityValidator,
):
    """Verify that with glasses, non-eye missing parts are still reported."""
    # Arrange
    counts = _all_sufficient_counts_with_eye_glass()
    del counts[FacePart.NOSE]
    # Ensure mouth region is still visible so only NOSE is missing.
    counts[FacePart.MOUTH] = _MOUTH_THRESHOLD_PIXELS + 5
    parsing_result = _build_parsing_result(counts)
    face = _make_face_with_valid_landmarks()

    # Act
    missing, overridden = validator._find_missing_parts_with_landmark_override(
        parsing_result=parsing_result,
        face=face,
    )

    # Assert
    assert set(missing) == {FacePart.NOSE}
    assert overridden == frozenset()