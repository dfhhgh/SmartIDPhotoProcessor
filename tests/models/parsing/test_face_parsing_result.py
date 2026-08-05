"""Behavioral tests for FaceParsingResult public query API."""

import numpy as np
import pytest

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult


# ------------------------------------------------------------------
# Realistic test fixture
# ------------------------------------------------------------------

def _make_result() -> FaceParsingResult:
    """Create a 4x6 parsing result with multiple semantic labels."""
    mask = np.array([
        [FacePart.BACKGROUND, FacePart.BACKGROUND, FacePart.HAIR,      FacePart.HAIR,      FacePart.BACKGROUND, FacePart.BACKGROUND],
        [FacePart.BACKGROUND, FacePart.SKIN,       FacePart.LEFT_EYE,  FacePart.RIGHT_EYE, FacePart.SKIN,       FacePart.BACKGROUND],
        [FacePart.BACKGROUND, FacePart.SKIN,       FacePart.NOSE,      FacePart.NOSE,      FacePart.SKIN,       FacePart.BACKGROUND],
        [FacePart.BACKGROUND, FacePart.UPPER_LIP,  FacePart.MOUTH,     FacePart.MOUTH,     FacePart.LOWER_LIP,  FacePart.BACKGROUND],
    ], dtype=np.uint8)
    return FaceParsingResult(mask=mask, image_height=4, image_width=6)


# ------------------------------------------------------------------
# 1. has_part()
# ------------------------------------------------------------------

class TestHasPart:
    """Tests for has_part() boolean query."""

    def test_returns_true_when_part_present(self):
        result = _make_result()
        assert result.has_part(FacePart.SKIN) is True

    def test_returns_false_when_part_absent(self):
        result = _make_result()
        assert result.has_part(FacePart.EYE_GLASS) is False

    def test_returns_true_for_single_pixel_part(self):
        result = _make_result()
        assert result.has_part(FacePart.UPPER_LIP) is True

    def test_returns_true_for_background(self):
        result = _make_result()
        assert result.has_part(FacePart.BACKGROUND) is True


# ------------------------------------------------------------------
# 2. contains_any()
# ------------------------------------------------------------------

class TestContainsAny:
    """Tests for contains_any() collection query."""

    def test_returns_true_when_all_parts_present(self):
        result = _make_result()
        assert result.contains_any([FacePart.SKIN, FacePart.NOSE]) is True

    def test_returns_true_when_some_parts_present(self):
        result = _make_result()
        assert result.contains_any([FacePart.SKIN, FacePart.HAT]) is True

    def test_returns_false_when_no_parts_present(self):
        result = _make_result()
        assert result.contains_any([FacePart.HAT, FacePart.NECKLACE]) is False

    def test_accepts_frozenset(self):
        result = _make_result()
        assert result.contains_any(frozenset({FacePart.HAT})) is False

    def test_accepts_tuple(self):
        result = _make_result()
        assert result.contains_any((FacePart.NOSE,)) is True


# ------------------------------------------------------------------
# 3. contains_all()
# ------------------------------------------------------------------

class TestContainsAll:
    """Tests for contains_all() collection query."""

    def test_returns_true_when_all_parts_present(self):
        result = _make_result()
        assert result.contains_all([FacePart.SKIN, FacePart.NOSE, FacePart.MOUTH]) is True

    def test_returns_false_when_one_part_missing(self):
        result = _make_result()
        assert result.contains_all([FacePart.SKIN, FacePart.HAT]) is False

    def test_returns_false_when_all_parts_missing(self):
        result = _make_result()
        assert result.contains_all([FacePart.HAT, FacePart.NECKLACE]) is False

    def test_returns_true_for_single_present_part(self):
        result = _make_result()
        assert result.contains_all([FacePart.NOSE]) is True

    def test_returns_true_for_empty_collection(self):
        result = _make_result()
        assert result.contains_all([]) is True


# ------------------------------------------------------------------
# 4. part_area()
# ------------------------------------------------------------------

class TestPartArea:
    """Tests for part_area() pixel counting."""

    def test_counts_background_pixels(self):
        result = _make_result()
        assert result.part_area(FacePart.BACKGROUND) == 10

    def test_counts_skin_pixels(self):
        result = _make_result()
        assert result.part_area(FacePart.SKIN) == 4

    def test_counts_nose_pixels(self):
        result = _make_result()
        assert result.part_area(FacePart.NOSE) == 2

    def test_counts_mouth_pixels(self):
        result = _make_result()
        assert result.part_area(FacePart.MOUTH) == 2

    def test_returns_zero_for_absent_part(self):
        result = _make_result()
        assert result.part_area(FacePart.HAT) == 0

    def test_returns_zero_for_ear_ring(self):
        result = _make_result()
        assert result.part_area(FacePart.EAR_RING) == 0

    def test_returns_int_type(self):
        result = _make_result()
        assert isinstance(result.part_area(FacePart.SKIN), int)


# ------------------------------------------------------------------
# 5. part_ratio()
# ------------------------------------------------------------------

class TestPartRatio:
    """Tests for part_ratio() fraction computation."""

    def test_ratio_is_area_divided_by_total_pixels(self):
        result = _make_result()
        expected = result.part_area(FacePart.SKIN) / result.total_pixels()
        assert result.part_ratio(FacePart.SKIN) == pytest.approx(expected)

    def test_background_ratio(self):
        result = _make_result()
        assert result.part_ratio(FacePart.BACKGROUND) == pytest.approx(10 / 24)

    def test_absent_part_has_zero_ratio(self):
        result = _make_result()
        assert result.part_ratio(FacePart.HAT) == pytest.approx(0.0)

    def test_single_pixel_ratio(self):
        result = _make_result()
        assert result.part_ratio(FacePart.UPPER_LIP) == pytest.approx(1 / 24)

    def test_returns_float_type(self):
        result = _make_result()
        assert isinstance(result.part_ratio(FacePart.SKIN), float)


# ------------------------------------------------------------------
# 6. part_mask()
# ------------------------------------------------------------------

class TestPartMask:
    """Tests for part_mask() boolean mask generation."""

    def test_returns_boolean_array(self):
        result = _make_result()
        mask = result.part_mask(FacePart.SKIN)
        assert mask.dtype == np.bool_

    def test_true_positions_match_mask_values(self):
        result = _make_result()
        boolean_mask = result.part_mask(FacePart.NOSE)
        expected = result.mask == FacePart.NOSE
        assert np.array_equal(boolean_mask, expected)

    def test_true_count_equals_part_area(self):
        result = _make_result()
        boolean_mask = result.part_mask(FacePart.MOUTH)
        assert boolean_mask.sum() == result.part_area(FacePart.MOUTH)

    def test_absent_part_produces_all_false(self):
        result = _make_result()
        boolean_mask = result.part_mask(FacePart.HAT)
        assert not boolean_mask.any()

    def test_shape_matches_mask_shape(self):
        result = _make_result()
        boolean_mask = result.part_mask(FacePart.SKIN)
        assert boolean_mask.shape == result.mask.shape


# ------------------------------------------------------------------
# 7. total_pixels()
# ------------------------------------------------------------------

class TestTotalPixels:
    """Tests for total_pixels() metadata query."""

    def test_returns_height_times_width(self):
        result = _make_result()
        assert result.total_pixels() == 4 * 6

    def test_returns_int_type(self):
        result = _make_result()
        assert isinstance(result.total_pixels(), int)


# ------------------------------------------------------------------
# 8. image_size()
# ------------------------------------------------------------------

class TestImageSize:
    """Tests for image_size() metadata query."""

    def test_returns_height_width_tuple(self):
        result = _make_result()
        assert result.image_size() == (4, 6)

    def test_returns_tuple_type(self):
        result = _make_result()
        assert isinstance(result.image_size(), tuple)

    def test_length_is_two(self):
        result = _make_result()
        assert len(result.image_size()) == 2


# ------------------------------------------------------------------
# 9. mask_shape()
# ------------------------------------------------------------------

class TestMaskShape:
    """Tests for mask_shape() metadata query."""

    def test_returns_2d_shape(self):
        result = _make_result()
        assert result.mask_shape() == (4, 6)

    def test_matches_mask_shape(self):
        result = _make_result()
        assert result.mask_shape() == result.mask.shape


# ------------------------------------------------------------------
# 10. Cross-method consistency
# ------------------------------------------------------------------

class TestConsistency:
    """Tests verifying consistency between related methods."""

    def test_part_area_matches_part_mask_sum(self):
        result = _make_result()
        for part in FacePart:
            assert result.part_area(part) == result.part_mask(part).sum()

    def test_part_ratio_matches_part_area_divided_by_total(self):
        result = _make_result()
        total = result.total_pixels()
        for part in FacePart:
            assert result.part_ratio(part) == pytest.approx(
                result.part_area(part) / total
            )

    def test_has_part_consistent_with_part_area(self):
        result = _make_result()
        for part in FacePart:
            assert result.has_part(part) == (result.part_area(part) > 0)

    def test_contains_any_consistent_with_has_part(self):
        result = _make_result()
        parts = [FacePart.SKIN, FacePart.HAT, FacePart.NOSE]
        expected = any(result.has_part(p) for p in parts)
        assert result.contains_any(parts) == expected

    def test_contains_all_consistent_with_has_part(self):
        result = _make_result()
        parts = [FacePart.SKIN, FacePart.NOSE]
        expected = all(result.has_part(p) for p in parts)
        assert result.contains_all(parts) == expected

    def test_image_size_matches_dimensions(self):
        result = _make_result()
        height, width = result.image_size()
        assert height == result.image_height
        assert width == result.image_width

    def test_total_pixels_matches_image_size(self):
        result = _make_result()
        height, width = result.image_size()
        assert result.total_pixels() == height * width

    def test_all_areas_sum_to_total_pixels(self):
        result = _make_result()
        total = sum(result.part_area(part) for part in FacePart)
        assert total == result.total_pixels()


# ------------------------------------------------------------------
# 11. has_visible_mouth_region()
# ------------------------------------------------------------------

class TestHasVisibleMouthRegion:
    """Tests for the composite mouth-region visibility check."""

    def _make_result_with_parts(
        self,
        parts: dict[FacePart, int],
        height: int = 10,
        width: int = 10,
    ) -> FaceParsingResult:
        """Build a FaceParsingResult with explicit pixel counts."""
        total = height * width
        flat_mask = np.zeros(total, dtype=np.int32)
        offset = 0
        for part, count in parts.items():
            if count <= 0:
                continue
            flat_mask[offset:offset + count] = int(part)
            offset += count
        mask = flat_mask.reshape(height, width)
        return FaceParsingResult(
            mask=mask,
            image_height=height,
            image_width=width,
        )

    # -- Case 1: MOUTH present and above threshold --

    def test_mouth_present_above_threshold_returns_true(self):
        result = self._make_result_with_parts({FacePart.MOUTH: 50})
        assert result.has_visible_mouth_region(mouth_min_ratio=0.1) is True

    def test_mouth_present_at_exact_threshold_returns_true(self):
        # 5 / 100 = 0.05 exactly
        result = self._make_result_with_parts(
            {FacePart.MOUTH: 5}, height=10, width=10,
        )
        assert result.has_visible_mouth_region(mouth_min_ratio=0.05) is True

    def test_mouth_present_below_threshold_falls_through_to_case2(self):
        # MOUTH below threshold, but lips may satisfy case 2
        result = self._make_result_with_parts(
            {FacePart.MOUTH: 1, FacePart.UPPER_LIP: 50, FacePart.LOWER_LIP: 50},
            height=10, width=10,
        )
        # mouth_min_ratio=1.0 means MOUTH would need 100 pixels (impossible here)
        # but case 2 saves it
        assert result.has_visible_mouth_region(
            mouth_min_ratio=1.0, upper_lip_min_ratio=0.1, lower_lip_min_ratio=0.1,
        ) is True

    def test_mouth_present_ignores_lip_absence(self):
        result = self._make_result_with_parts({FacePart.MOUTH: 50})
        assert result.has_visible_mouth_region(mouth_min_ratio=0.1) is True

    # -- Case 2: Both lips present and above threshold --

    def test_both_lips_present_above_threshold_returns_true(self):
        result = self._make_result_with_parts(
            {FacePart.UPPER_LIP: 50, FacePart.LOWER_LIP: 50},
            height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            upper_lip_min_ratio=0.1, lower_lip_min_ratio=0.1,
        ) is True

    def test_both_lips_at_exact_threshold_returns_true(self):
        # 5 / 100 = 0.05 exactly
        result = self._make_result_with_parts(
            {FacePart.UPPER_LIP: 5, FacePart.LOWER_LIP: 5},
            height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            upper_lip_min_ratio=0.05, lower_lip_min_ratio=0.05,
        ) is True

    def test_both_lips_below_threshold_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.UPPER_LIP: 1, FacePart.LOWER_LIP: 1},
            height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            upper_lip_min_ratio=0.1, lower_lip_min_ratio=0.1,
        ) is False

    # -- Partial lip presence --

    def test_only_upper_lip_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.UPPER_LIP: 50}, height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            upper_lip_min_ratio=0.1, lower_lip_min_ratio=0.1,
        ) is False

    def test_only_lower_lip_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.LOWER_LIP: 50}, height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            upper_lip_min_ratio=0.1, lower_lip_min_ratio=0.1,
        ) is False

    # -- Neither case satisfied --

    def test_no_mouth_no_lips_returns_false(self):
        result = self._make_result_with_parts({})
        assert result.has_visible_mouth_region() is False

    def test_mouth_absent_one_lip_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.UPPER_LIP: 50}, height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            mouth_min_ratio=0.1, upper_lip_min_ratio=0.1, lower_lip_min_ratio=0.1,
        ) is False

    # -- Custom thresholds --

    def test_custom_thresholds_are_respected(self):
        # MOUTH at 0.04, threshold at 0.05 -> fails case 1
        # UPPER_LIP at 0.06, LOWER_LIP at 0.06, threshold at 0.05 -> passes case 2
        result = self._make_result_with_parts(
            {FacePart.MOUTH: 4, FacePart.UPPER_LIP: 6, FacePart.LOWER_LIP: 6},
            height=10, width=10,
        )
        assert result.has_visible_mouth_region(
            mouth_min_ratio=0.05, upper_lip_min_ratio=0.05, lower_lip_min_ratio=0.05,
        ) is True

    # -- Default thresholds (all 0.0) --

    def test_default_thresholds_accept_any_presence(self):
        result = self._make_result_with_parts({FacePart.MOUTH: 1})
        assert result.has_visible_mouth_region() is True

    # -- Return type --

    def test_returns_bool_type(self):
        result = self._make_result_with_parts({})
        assert isinstance(result.has_visible_mouth_region(), bool)


# ------------------------------------------------------------------
# 12. has_visible_left_eyebrow() / has_visible_right_eyebrow()
# ------------------------------------------------------------------


class TestHasVisibleEyebrow:
    """Tests for the composite eyebrow visibility check."""

    def _make_result_with_parts(
        self,
        parts: dict[FacePart, int],
        height: int = 10,
        width: int = 10,
    ) -> FaceParsingResult:
        """Build a FaceParsingResult with explicit pixel counts."""
        total = height * width
        flat_mask = np.zeros(total, dtype=np.int32)
        offset = 0
        for part, count in parts.items():
            if count <= 0:
                continue
            flat_mask[offset:offset + count] = int(part)
            offset += count
        mask = flat_mask.reshape(height, width)
        return FaceParsingResult(
            mask=mask,
            image_height=height,
            image_width=width,
        )

    # -- Case 1: Parser detects brow above threshold --

    def test_left_brow_present_above_threshold_returns_true(self):
        result = self._make_result_with_parts({FacePart.LEFT_BROW: 50})
        assert result.has_visible_left_eyebrow(brow_min_ratio=0.1) is True

    def test_right_brow_present_above_threshold_returns_true(self):
        result = self._make_result_with_parts({FacePart.RIGHT_BROW: 50})
        assert result.has_visible_right_eyebrow(brow_min_ratio=0.1) is True

    def test_left_brow_at_exact_threshold_returns_true(self):
        # 5 / 100 = 0.05 exactly
        result = self._make_result_with_parts(
            {FacePart.LEFT_BROW: 5}, height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(brow_min_ratio=0.05) is True

    def test_left_brow_below_threshold_falls_through_to_case2(self):
        result = self._make_result_with_parts(
            {FacePart.LEFT_BROW: 1, FacePart.LEFT_EYE: 50},
            height=10, width=10,
        )
        # brow_min_ratio=1.0 means LEFT_BROW would need 100 pixels
        # but case 2 saves it if eye + landmark boolean exist
        assert result.has_visible_left_eyebrow(
            brow_min_ratio=1.0, eye_min_ratio=0.1,
            left_eye_landmark_valid=True,
        ) is True

    def test_right_brow_below_threshold_falls_through_to_case2(self):
        result = self._make_result_with_parts(
            {FacePart.RIGHT_BROW: 1, FacePart.RIGHT_EYE: 50},
            height=10, width=10,
        )
        assert result.has_visible_right_eyebrow(
            brow_min_ratio=1.0, eye_min_ratio=0.1,
            right_eye_landmark_valid=True,
        ) is True

    # -- Case 2: Parser misses brow, eye exists, landmark valid --

    def test_left_brow_missing_eye_present_landmark_valid_returns_true(self):
        result = self._make_result_with_parts(
            {FacePart.LEFT_EYE: 50}, height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(
            eye_min_ratio=0.1, left_eye_landmark_valid=True,
        ) is True

    def test_right_brow_missing_eye_present_landmark_valid_returns_true(self):
        result = self._make_result_with_parts(
            {FacePart.RIGHT_EYE: 50}, height=10, width=10,
        )
        assert result.has_visible_right_eyebrow(
            eye_min_ratio=0.1, right_eye_landmark_valid=True,
        ) is True

    def test_left_brow_missing_eye_present_landmark_invalid_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.LEFT_EYE: 50}, height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(
            eye_min_ratio=0.1, left_eye_landmark_valid=False,
        ) is False

    def test_right_brow_missing_eye_present_landmark_invalid_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.RIGHT_EYE: 50}, height=10, width=10,
        )
        assert result.has_visible_right_eyebrow(
            eye_min_ratio=0.1, right_eye_landmark_valid=False,
        ) is False

    # -- Parser misses brow, no eye --

    def test_left_brow_missing_no_eye_returns_false(self):
        result = self._make_result_with_parts({})
        assert result.has_visible_left_eyebrow(
            eye_min_ratio=0.1, left_eye_landmark_valid=True,
        ) is False

    def test_right_brow_missing_no_eye_returns_false(self):
        result = self._make_result_with_parts({})
        assert result.has_visible_right_eyebrow(
            eye_min_ratio=0.1, right_eye_landmark_valid=True,
        ) is False

    # -- Parser misses brow, eye present but landmark invalid --

    def test_left_brow_missing_eye_present_no_landmark_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.LEFT_EYE: 50}, height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(
            eye_min_ratio=0.1, left_eye_landmark_valid=False,
        ) is False

    def test_right_brow_missing_eye_present_no_landmark_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.RIGHT_EYE: 50}, height=10, width=10,
        )
        assert result.has_visible_right_eyebrow(
            eye_min_ratio=0.1, right_eye_landmark_valid=False,
        ) is False

    # -- Parser misses brow, eye present but below threshold --

    def test_left_brow_missing_eye_below_threshold_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.LEFT_EYE: 1}, height=10, width=10,
        )
        # eye_min_ratio=1.0 means LEFT_EYE would need 100 pixels
        assert result.has_visible_left_eyebrow(
            eye_min_ratio=1.0, left_eye_landmark_valid=True,
        ) is False

    def test_right_brow_missing_eye_below_threshold_returns_false(self):
        result = self._make_result_with_parts(
            {FacePart.RIGHT_EYE: 1}, height=10, width=10,
        )
        assert result.has_visible_right_eyebrow(
            eye_min_ratio=1.0, right_eye_landmark_valid=True,
        ) is False

    # -- Neither case satisfied --

    def test_no_brow_no_eye_no_landmark_returns_false(self):
        result = self._make_result_with_parts({})
        assert result.has_visible_left_eyebrow() is False
        assert result.has_visible_right_eyebrow() is False

    # -- Opposite eyebrow does not affect the check --

    def test_left_brow_missing_right_brow_still_missing(self):
        """Parser missing left brow is not saved by right brow existing."""
        result = self._make_result_with_parts(
            {FacePart.RIGHT_BROW: 50}, height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(brow_min_ratio=0.1) is False

    def test_right_brow_missing_left_brow_still_missing(self):
        """Parser missing right brow is not saved by left brow existing."""
        result = self._make_result_with_parts(
            {FacePart.LEFT_BROW: 50}, height=10, width=10,
        )
        assert result.has_visible_right_eyebrow(brow_min_ratio=0.1) is False

    # -- Custom thresholds --

    def test_custom_thresholds_are_respected(self):
        # LEFT_BROW at 0.04, threshold at 0.05 -> fails case 1
        # LEFT_EYE at 0.06, threshold at 0.05 -> passes case 2
        result = self._make_result_with_parts(
            {FacePart.LEFT_BROW: 4, FacePart.LEFT_EYE: 6},
            height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(
            brow_min_ratio=0.05, eye_min_ratio=0.05,
            left_eye_landmark_valid=True,
        ) is True

    # -- Default thresholds (all 0.0) --

    def test_default_thresholds_accept_any_presence(self):
        result = self._make_result_with_parts({FacePart.LEFT_BROW: 1})
        assert result.has_visible_left_eyebrow() is True
        result = self._make_result_with_parts({FacePart.RIGHT_BROW: 1})
        assert result.has_visible_right_eyebrow() is True

    # -- Return type --

    def test_returns_bool_type(self):
        result = self._make_result_with_parts({})
        assert isinstance(result.has_visible_left_eyebrow(), bool)
        assert isinstance(result.has_visible_right_eyebrow(), bool)

    # -- Both eyebrows visible --

    def test_both_eyebrows_visible_returns_true(self):
        result = self._make_result_with_parts(
            {FacePart.LEFT_BROW: 50, FacePart.RIGHT_BROW: 50},
            height=10, width=10,
        )
        assert result.has_visible_left_eyebrow(brow_min_ratio=0.1) is True
        assert result.has_visible_right_eyebrow(brow_min_ratio=0.1) is True

    # -- Both eyebrows missing --

    def test_both_eyebrows_missing_returns_false(self):
        result = self._make_result_with_parts({})
        assert result.has_visible_left_eyebrow() is False
        assert result.has_visible_right_eyebrow() is False
