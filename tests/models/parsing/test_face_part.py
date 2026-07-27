"""Tests for the FacePart enum."""

from enum import IntEnum

import pytest

from models.parsing.face_part import FacePart


# ------------------------------------------------------------------
# 1. Enum Integrity
# ------------------------------------------------------------------


def test_total_enum_member_count():
    """CelebAMask-HQ defines exactly 19 semantic classes."""
    assert len(FacePart) == 19


def test_enum_values_are_unique():
    """Every enum member must map to a distinct integer value."""
    values = [member.value for member in FacePart]
    assert len(values) == len(set(values))


def test_face_part_is_int_enum():
    """FacePart must inherit from IntEnum for mask arithmetic."""
    assert issubclass(FacePart, IntEnum)


def test_expected_celebamask_hq_mapping():
    """Verify the exact numeric ID for every semantic class."""
    expected = {
        "BACKGROUND": 0,
        "SKIN": 1,
        "LEFT_BROW": 2,
        "RIGHT_BROW": 3,
        "LEFT_EYE": 4,
        "RIGHT_EYE": 5,
        "EYE_GLASS": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "EAR_RING": 9,
        "NOSE": 10,
        "MOUTH": 11,
        "UPPER_LIP": 12,
        "LOWER_LIP": 13,
        "NECK": 14,
        "NECKLACE": 15,
        "CLOTH": 16,
        "HAIR": 17,
        "HAT": 18,
    }

    for name, value in expected.items():
        member = FacePart[name]
        assert member == value, (
            f"FacePart.{name} should be {value}, got {member}"
        )


# ------------------------------------------------------------------
# 2. Helper Methods
# ------------------------------------------------------------------

ALL_HELPERS = [
    FacePart.facial_region(),
    FacePart.brows(),
    FacePart.eyes(),
    FacePart.eye_related(),
    FacePart.nose_region(),
    FacePart.mouth_region(),
    FacePart.lips(),
    FacePart.ears(),
    FacePart.accessories(),
    FacePart.non_face(),
]


def test_facial_region_returns_expected_members():
    """facial_region() must contain the 9 core face parts."""
    expected = {
        FacePart.SKIN,
        FacePart.LEFT_BROW,
        FacePart.RIGHT_BROW,
        FacePart.LEFT_EYE,
        FacePart.RIGHT_EYE,
        FacePart.NOSE,
        FacePart.MOUTH,
        FacePart.UPPER_LIP,
        FacePart.LOWER_LIP,
    }

    result = FacePart.facial_region()

    assert result == expected


def test_brows_returns_left_and_right():
    """brows() must contain both eyebrow parts."""
    expected = {FacePart.LEFT_BROW, FacePart.RIGHT_BROW}

    result = FacePart.brows()

    assert result == expected


def test_eyes_returns_anatomical_eyes_only():
    """eyes() must contain only LEFT_EYE and RIGHT_EYE."""
    expected = {FacePart.LEFT_EYE, FacePart.RIGHT_EYE}

    result = FacePart.eyes()

    assert result == expected


def test_eye_related_includes_glasses():
    """eye_related() must contain eyes plus EYE_GLASS."""
    expected = {
        FacePart.LEFT_EYE,
        FacePart.RIGHT_EYE,
        FacePart.EYE_GLASS,
    }

    result = FacePart.eye_related()

    assert result == expected


def test_nose_region_returns_nose():
    """nose_region() must contain only NOSE."""
    expected = {FacePart.NOSE}

    result = FacePart.nose_region()

    assert result == expected


def test_mouth_region_returns_mouth_and_lips():
    """mouth_region() must contain MOUTH, UPPER_LIP, LOWER_LIP."""
    expected = {
        FacePart.MOUTH,
        FacePart.UPPER_LIP,
        FacePart.LOWER_LIP,
    }

    result = FacePart.mouth_region()

    assert result == expected


def test_lips_returns_upper_and_lower():
    """lips() must contain UPPER_LIP and LOWER_LIP."""
    expected = {FacePart.UPPER_LIP, FacePart.LOWER_LIP}

    result = FacePart.lips()

    assert result == expected


def test_ears_returns_ears_and_ear_ring():
    """ears() must contain LEFT_EAR, RIGHT_EAR, EAR_RING."""
    expected = {
        FacePart.LEFT_EAR,
        FacePart.RIGHT_EAR,
        FacePart.EAR_RING,
    }

    result = FacePart.ears()

    assert result == expected


def test_accessories_returns_all_accessory_parts():
    """accessories() must contain EYE_GLASS, EAR_RING, NECKLACE, HAT."""
    expected = {
        FacePart.EYE_GLASS,
        FacePart.EAR_RING,
        FacePart.NECKLACE,
        FacePart.HAT,
    }

    result = FacePart.accessories()

    assert result == expected


def test_non_face_returns_background_and_cloth():
    """non_face() must contain BACKGROUND and CLOTH."""
    expected = {FacePart.BACKGROUND, FacePart.CLOTH}

    result = FacePart.non_face()

    assert result == expected


# ------------------------------------------------------------------
# 3. Semantic Correctness
# ------------------------------------------------------------------


def test_all_helper_collections_are_frozensets():
    """Every helper method must return a frozenset."""
    for collection in ALL_HELPERS:
        assert isinstance(collection, frozenset)


def test_all_helper_collections_contain_only_face_parts():
    """Every element in every helper collection must be a FacePart."""
    for collection in ALL_HELPERS:
        for member in collection:
            assert isinstance(member, FacePart)


def test_eye_related_differs_from_eyes_by_glasses_only():
    """eye_related() must be exactly eyes() plus EYE_GLASS."""
    difference = FacePart.eye_related() - FacePart.eyes()

    assert difference == {FacePart.EYE_GLASS}


def test_mouth_region_differs_from_lips_by_mouth():
    """mouth_region() must be exactly lips() plus MOUTH."""
    difference = FacePart.mouth_region() - FacePart.lips()

    assert difference == {FacePart.MOUTH}


def test_facial_region_includes_eyes_brows_nose_mouth_lips():
    """facial_region() must be the union of its sub-regions."""
    expected = (
        FacePart.eyes()
        | FacePart.brows()
        | FacePart.nose_region()
        | FacePart.mouth_region()
        | {FacePart.SKIN}
    )

    assert FacePart.facial_region() == expected


def test_facial_region_excludes_accessories():
    """facial_region() must not contain any accessory parts."""
    assert FacePart.facial_region().isdisjoint(FacePart.accessories())


def test_non_face_excludes_facial_region():
    """non_face() must not overlap with facial_region()."""
    assert FacePart.non_face().isdisjoint(FacePart.facial_region())


def test_accessories_excludes_non_face():
    """accessories() must not overlap with non_face()."""
    assert FacePart.accessories().isdisjoint(FacePart.non_face())


# ------------------------------------------------------------------
# 4. Predicate Methods
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "part",
    [
        FacePart.SKIN,
        FacePart.LEFT_BROW,
        FacePart.RIGHT_BROW,
        FacePart.LEFT_EYE,
        FacePart.RIGHT_EYE,
        FacePart.NOSE,
        FacePart.MOUTH,
        FacePart.UPPER_LIP,
        FacePart.LOWER_LIP,
        FacePart.LEFT_EAR,
        FacePart.RIGHT_EAR,
        FacePart.NECK,
        FacePart.HAIR,
    ],
    ids=lambda p: p.name,
)
def test_anatomical_parts_are_anatomical(part):
    """All real anatomical regions must be flagged as anatomical."""
    assert part.is_anatomical is True


@pytest.mark.parametrize(
    "part",
    [
        FacePart.BACKGROUND,
        FacePart.EYE_GLASS,
        FacePart.EAR_RING,
        FacePart.NECKLACE,
        FacePart.CLOTH,
        FacePart.HAT,
    ],
    ids=lambda p: p.name,
)
def test_non_anatomical_parts_are_not_anatomical(part):
    """Background, clothing, and accessories must not be anatomical."""
    assert part.is_anatomical is False


# ------------------------------------------------------------------
# 5. Immutability
# ------------------------------------------------------------------


def test_helper_collections_are_immutable():
    """Attempting to mutate a helper collection must raise an error."""
    for collection in ALL_HELPERS:
        with pytest.raises(AttributeError):
            collection.add(FacePart.BACKGROUND)


def test_helper_collections_do_not_support_discard():
    """Attempting to discard from a helper collection must raise an error."""
    for collection in ALL_HELPERS:
        with pytest.raises(AttributeError):
            collection.discard(FacePart.SKIN)
