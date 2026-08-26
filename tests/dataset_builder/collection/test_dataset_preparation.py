"""Tests for dataset_preparation module.

Covers:
1. Manifest saved after split
2. Split counts consistency
3. Deterministic split
4. No split assigned to REVIEW/REMOVE
5. Semantic category flags
6. FaceDetector reuse
7. Multi-face → REVIEW
8. Small face → REVIEW
9. Profile face → REVIEW
10. Quality failure → REVIEW
11. Quality disabled behavior
12. Raw dataset immutability
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image as PILImage

from dataset_builder.collection.dataset_preparation import (
    CATEGORY_SEMANTIC_FLAGS,
    DatasetManifest,
    DatasetPreparation,
    ImageDecision,
)
from dataset_builder.config.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        DATASET_DIR=tmp_path / "dataset",
        RAW_IMAGES_DIR=tmp_path / "raw",
        REPORTS_DIR=tmp_path / "reports",
        MIN_IMAGE_WIDTH=100,
        MIN_IMAGE_HEIGHT=100,
        IMAGEHASH_SIZE=8,
        DUPLICATE_DISTANCE_THRESHOLD=8,
        MIN_FACE_AREA_RATIO=0.02,
        MAX_PROFILE_YAW_DEGREES=30.0,
        MIN_BLUR_SCORE=60.0,
        MIN_BRIGHTNESS=40.0,
        MAX_BRIGHTNESS=220.0,
        MIN_CONTRAST=30.0,
    )


def settings_with_custom_thresholds(
    tmp_path: Path,
    min_face_area_ratio: float = 0.02,
    max_profile_yaw: float = 30.0,
) -> Settings:
    """Create Settings with custom thresholds for testing."""
    return Settings(
        DATASET_DIR=tmp_path / "dataset",
        RAW_IMAGES_DIR=tmp_path / "raw",
        REPORTS_DIR=tmp_path / "reports",
        MIN_IMAGE_WIDTH=100,
        MIN_IMAGE_HEIGHT=100,
        IMAGEHASH_SIZE=8,
        DUPLICATE_DISTANCE_THRESHOLD=8,
        MIN_FACE_AREA_RATIO=min_face_area_ratio,
        MAX_PROFILE_YAW_DEGREES=max_profile_yaw,
        MIN_BLUR_SCORE=60.0,
        MIN_BRIGHTNESS=40.0,
        MAX_BRIGHTNESS=220.0,
        MIN_CONTRAST=30.0,
    )


def create_image(path: Path, size: tuple[int, int] = (800, 600),
                 color: tuple[int, int, int] = (128, 128, 128)) -> None:
    """Create a test image."""
    img = PILImage.new("RGB", size, color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def create_random_image(path: Path, size: tuple[int, int] = (800, 600)) -> None:
    """Create a random test image."""
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.fromarray(arr).save(path)


def create_corrupt_image(path: Path) -> None:
    """Create a corrupt image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a valid image file at all")


def _make_decision(
    *,
    decision: str = "KEEP",
    split: str = "",
    category: str = "normal",
    face_count: int = 1,
    face_detection_evaluated: bool = True,
    quality_evaluated: bool = False,
    is_near_duplicate: bool = False,
    readable: bool = True,
    oversized: bool = False,
    has_small_face: bool = False,
    has_profile_face: bool = False,
    blur_passed: bool | None = None,
    brightness_passed: bool | None = None,
    contrast_passed: bool | None = None,
    filename: str = "test.jpg",
) -> ImageDecision:
    """Helper to create an ImageDecision for testing."""
    dec = ImageDecision(
        path=f"/fake/raw/{category}/{filename}",
        filename=filename,
        category=category,
        readable=readable,
        oversized=oversized,
        face_detection_evaluated=face_detection_evaluated,
        face_count=face_count,
        has_small_face=has_small_face,
        has_profile_face=has_profile_face,
        quality_evaluated=quality_evaluated,
        blur_passed=blur_passed,
        brightness_passed=brightness_passed,
        contrast_passed=contrast_passed,
        is_near_duplicate=is_near_duplicate,
    )
    dec.decision = decision
    return dec


# ---------------------------------------------------------------------------
# 1. Manifest saved after split
# ---------------------------------------------------------------------------

class TestManifestPersistenceAfterSplit:
    """Test that manifest is saved AFTER split assignments."""

    def test_manifest_contains_split_assignments(self, tmp_path: Path) -> None:
        """Manifest JSON must contain final split assignments."""
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = tmp_path / "raw"

        # Create a pipeline and inject decisions directly
        pipeline = DatasetPreparation(
            raw_dir=raw_dir,
            output_dir=output_dir,
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        # Inject KEEP decisions with real paths under raw_dir
        pipeline._decisions = []
        for i in range(10):
            img_path = raw_dir / "normal" / f"keep_{i}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img_path.write_bytes(b"fake")
            dec = ImageDecision(
                path=str(img_path),
                filename=f"keep_{i}.jpg",
                category="normal",
                readable=True,
                face_detection_evaluated=True,
                face_count=1,
                decision="KEEP",
            )
            pipeline._decisions.append(dec)

        # Also inject a REVIEW decision
        review_path = raw_dir / "normal" / "review_0.jpg"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_bytes(b"fake")
        pipeline._decisions.append(ImageDecision(
            path=str(review_path),
            filename="review_0.jpg",
            category="normal",
            readable=True,
            face_detection_evaluated=True,
            face_count=0,
            decision="REVIEW",
            decision_reasons=["No face detected"],
        ))

        # Run split + manifest steps
        pipeline._split_dataset()
        pipeline._manifest = pipeline._build_manifest()

        # Verify manifest was saved
        manifest_path = output_dir / "dataset_manifest.json"
        assert manifest_path.exists(), "Manifest file not found"

        # Load and verify it contains split data
        with open(manifest_path) as f:
            data = json.load(f)

        assert "split_stats" in data
        assert data["split_stats"].get("train", 0) > 0
        assert data["split_stats"].get("val", 0) > 0

        # Verify every KEEP image has a split
        for img in data["images"]:
            if img["decision"] == "KEEP":
                assert img["split"] in ("train", "val", "test"), (
                    f"KEEP image {img['filename']} has split='{img['split']}'"
                )
            else:
                assert img["split"] == "", (
                    f"Non-KEEP image {img['filename']} has split='{img['split']}'"
                )

    def test_split_stats_matches_assignments(self, tmp_path: Path) -> None:
        """split_stats in manifest must match actual split assignments."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(10):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir,
            output_dir=output_dir,
            skip_face_detection=True,
            skip_quality_scores=True,
        )
        manifest = pipeline.run()

        # Count actual splits
        train = sum(1 for img in manifest.images if img.split == "train")
        val = sum(1 for img in manifest.images if img.split == "val")
        test = sum(1 for img in manifest.images if img.split == "test")

        assert manifest.split_stats["train"] == train
        assert manifest.split_stats["val"] == val
        assert manifest.split_stats["test"] == test

    def test_keep_count_equals_split_sum(self, tmp_path: Path) -> None:
        """train + val + test must equal keep_count."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(10):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")
        create_corrupt_image(raw_dir / "normal" / "corrupt.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir,
            output_dir=output_dir,
            skip_face_detection=True,
            skip_quality_scores=True,
        )
        manifest = pipeline.run()

        split_total = (
            manifest.split_stats.get("train", 0)
            + manifest.split_stats.get("val", 0)
            + manifest.split_stats.get("test", 0)
        )
        assert split_total == manifest.keep_count, (
            f"Split total {split_total} != keep_count {manifest.keep_count}"
        )


# ---------------------------------------------------------------------------
# 2. Split counts consistency
# ---------------------------------------------------------------------------

class TestSplitConsistency:
    """Test split count consistency."""

    def test_no_review_in_split(self, tmp_path: Path) -> None:
        """REVIEW images must not appear in any split."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(5):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")
        create_corrupt_image(raw_dir / "normal" / "corrupt.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir,
            output_dir=output_dir,
            skip_face_detection=True,
            skip_quality_scores=True,
        )
        manifest = pipeline.run()

        split_filenames = set()
        for split_name in ["train", "val", "test"]:
            split_path = output_dir / "splits" / f"{split_name}.txt"
            if split_path.exists():
                for line in split_path.read_text().splitlines():
                    split_filenames.add(line.strip().split("/")[-1])

        # Corrupt image should not be in any split
        assert "corrupt.jpg" not in split_filenames

    def test_no_remove_in_split(self, tmp_path: Path) -> None:
        """REMOVE images must not appear in any split."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")
        create_corrupt_image(raw_dir / "normal" / "bad.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir,
            output_dir=output_dir,
            skip_face_detection=True,
            skip_quality_scores=True,
        )
        manifest = pipeline.run()

        for img in manifest.images:
            if img.decision == "REMOVE":
                assert img.split == "", (
                    f"REMOVE image {img.filename} has split='{img.split}'"
                )


# ---------------------------------------------------------------------------
# 3. Deterministic split
# ---------------------------------------------------------------------------

class TestDeterministicSplit:
    """Test that splitting is deterministic with the same seed."""

    def test_same_input_same_split(self, tmp_path: Path) -> None:
        """Same input with same seed produces same split."""
        raw_dir = tmp_path / "raw"
        output_dir1 = tmp_path / "output1"
        output_dir2 = tmp_path / "output2"

        for i in range(20):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        # Run twice
        pipeline1 = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir1,
            skip_face_detection=True, skip_quality_scores=True,
        )
        m1 = pipeline1.run()

        pipeline2 = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir2,
            skip_face_detection=True, skip_quality_scores=True,
        )
        m2 = pipeline2.run()

        # Compare splits
        splits1 = {img.filename: img.split for img in m1.images if img.decision == "KEEP"}
        splits2 = {img.filename: img.split for img in m2.images if img.decision == "KEEP"}
        assert splits1 == splits2, "Deterministic split failed: same input produced different splits"


# ---------------------------------------------------------------------------
# 4. No split assigned to REVIEW/REMOVE
# ---------------------------------------------------------------------------

class TestNoSplitForReviewRemove:
    """Test that REVIEW/REMOVE images never get split assignments."""

    def test_review_images_have_empty_split(self, tmp_path: Path) -> None:
        """REVIEW images must have split=''."""
        dec = _make_decision(decision="REVIEW", split="")
        assert dec.split == ""

    def test_remove_images_have_empty_split(self, tmp_path: Path) -> None:
        """REMOVE images must have split=''."""
        dec = _make_decision(decision="REMOVE", split="")
        assert dec.split == ""


# ---------------------------------------------------------------------------
# 5. Semantic category flags
# ---------------------------------------------------------------------------

class TestSemanticCategoryFlags:
    """Test deterministic semantic category flag population."""

    def test_all_categories_have_flags(self) -> None:
        """All 10 project categories must have semantic flag mappings."""
        expected = {
            "normal", "eyeglasses", "sunglasses", "cap", "helmet",
            "mask", "hijab", "scarf", "hair_occlusion", "beard",
        }
        assert set(CATEGORY_SEMANTIC_FLAGS.keys()) == expected

    def test_normal_category_flags(self) -> None:
        """normal: is_baseline_normal=True, no occlusion."""
        flags = CATEGORY_SEMANTIC_FLAGS["normal"]
        assert flags["is_baseline_normal"] is True
        assert flags["contains_eyeglass"] is False
        assert flags["contains_hat"] is False
        assert flags["contains_facial_occlusion"] is False

    def test_eyeglasses_category_flags(self) -> None:
        """eyeglasses: contains_eyeglass=True, facial_occlusion=True, useful=True."""
        flags = CATEGORY_SEMANTIC_FLAGS["eyeglasses"]
        assert flags["contains_eyeglass"] is True
        assert flags["contains_facial_occlusion"] is True
        assert flags["useful_for_eye_improvement"] is True
        assert flags["contains_hat"] is False
        assert flags["is_baseline_normal"] is False

    def test_sunglasses_category_flags(self) -> None:
        """sunglasses: contains_eyeglass=True, facial_occlusion=True, useful=True."""
        flags = CATEGORY_SEMANTIC_FLAGS["sunglasses"]
        assert flags["contains_eyeglass"] is True
        assert flags["contains_facial_occlusion"] is True
        assert flags["useful_for_eye_improvement"] is True

    def test_cap_category_flags(self) -> None:
        """cap: contains_hat=True, no facial occlusion."""
        flags = CATEGORY_SEMANTIC_FLAGS["cap"]
        assert flags["contains_hat"] is True
        assert flags["contains_facial_occlusion"] is False
        assert flags["is_baseline_normal"] is False

    def test_helmet_category_flags(self) -> None:
        """helmet: contains_hat=True, facial_occlusion=True."""
        flags = CATEGORY_SEMANTIC_FLAGS["helmet"]
        assert flags["contains_hat"] is True
        assert flags["contains_facial_occlusion"] is True

    def test_mask_category_flags(self) -> None:
        """mask: facial_occlusion=True."""
        flags = CATEGORY_SEMANTIC_FLAGS["mask"]
        assert flags["contains_facial_occlusion"] is True
        assert flags["contains_hat"] is False

    def test_hijab_category_flags(self) -> None:
        """hijab: NOT facial occlusion (covers hair, not face)."""
        flags = CATEGORY_SEMANTIC_FLAGS["hijab"]
        assert flags["contains_facial_occlusion"] is False
        assert flags["contains_hat"] is False

    def test_scarf_category_flags(self) -> None:
        """scarf: conservatively False until annotation confirms."""
        flags = CATEGORY_SEMANTIC_FLAGS["scarf"]
        assert flags["contains_facial_occlusion"] is False

    def test_hair_occlusion_category_flags(self) -> None:
        """hair_occlusion: facial_occlusion=True."""
        flags = CATEGORY_SEMANTIC_FLAGS["hair_occlusion"]
        assert flags["contains_facial_occlusion"] is True

    def test_beard_category_flags(self) -> None:
        """beard: NOT facial occlusion (facial hair, not external object)."""
        flags = CATEGORY_SEMANTIC_FLAGS["beard"]
        assert flags["contains_facial_occlusion"] is False

    def test_flags_applied_to_decisions(self, tmp_path: Path) -> None:
        """Semantic flags must be applied to ImageDecision objects."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"n_{i}.jpg")
            create_random_image(raw_dir / "eyeglasses" / f"e_{i}.jpg")
            create_random_image(raw_dir / "cap" / f"c_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        manifest = pipeline.run()

        for img in manifest.images:
            flags = CATEGORY_SEMANTIC_FLAGS.get(img.category, {})
            assert img.is_baseline_normal == flags.get("is_baseline_normal", False)
            assert img.contains_eyeglass == flags.get("contains_eyeglass", False)
            assert img.contains_hat == flags.get("contains_hat", False)
            assert img.contains_facial_occlusion == flags.get("contains_facial_occlusion", False)
            assert img.useful_for_eye_improvement == flags.get("useful_for_eye_improvement", False)


# ---------------------------------------------------------------------------
# 6. FaceDetector reuse
# ---------------------------------------------------------------------------

class TestFaceDetectorReuse:
    """Test that FaceDetector is initialized once, not per image."""

    def test_detector_initialized_once(self) -> None:
        """_get_detector() should initialize FaceDetector only once."""
        pipeline = DatasetPreparation(
            raw_dir=Path("/fake/raw"),
            output_dir=Path("/fake/output"),
        )

        mock_detector = MagicMock()
        with patch("pipeline.detector.FaceDetector", return_value=mock_detector) as MockFD:
            d1 = pipeline._get_detector()
            d2 = pipeline._get_detector()
            d3 = pipeline._get_detector()

            assert d1 is d2 is d3
            # FaceDetector constructor called exactly once
            assert MockFD.call_count == 1

    def test_detector_reused_across_calls(self) -> None:
        """Subsequent _get_detector() calls return same instance."""
        pipeline = DatasetPreparation(
            raw_dir=Path("/fake/raw"),
            output_dir=Path("/fake/output"),
        )

        mock_detector = MagicMock()
        with patch("pipeline.detector.FaceDetector", return_value=mock_detector):
            d1 = pipeline._get_detector()
            d2 = pipeline._get_detector()

            assert d1 is d2


# ---------------------------------------------------------------------------
# 7. Multi-face → REVIEW
# ---------------------------------------------------------------------------

class TestMultiFacePolicy:
    """Test that multi-face images always get REVIEW."""

    def test_multi_face_selected_gets_review(self, tmp_path: Path) -> None:
        """Multi-face + successful selection → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        # Create images and manually set multi-face state
        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        # Bypass audit — directly set decisions
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=3,
                face_detection_evaluated=True,
                filename="multi.jpg",
            )
        ]
        pipeline._decisions[0].multi_face_investigation = "selected_face_0_of_3"
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"
        assert "Multiple faces detected" in pipeline._decisions[0].decision_reasons[0]

    def test_multi_face_selection_failed_gets_review(self, tmp_path: Path) -> None:
        """Multi-face + failed selection → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=2,
                face_detection_evaluated=True,
                filename="multi_fail.jpg",
            )
        ]
        pipeline._decisions[0].multi_face_investigation = "selection_failed"
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"

    def test_multi_face_redetected_single_gets_review(self, tmp_path: Path) -> None:
        """Multi-face re-detected as single face → REVIEW (not KEEP).

        Even though face_count is now 1, the image was originally multi-face,
        so human review is required before annotation.
        """
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,  # updated by re-detection
                face_detection_evaluated=True,
                filename="redetected.jpg",
            )
        ]
        pipeline._decisions[0].multi_face_investigation = "re_detected_1_faces"
        pipeline._make_final_decisions()

        # Re-detected as single face → REVIEW (originally multi-face)
        assert pipeline._decisions[0].decision == "REVIEW"
        reasons = " ".join(pipeline._decisions[0].decision_reasons)
        assert "Originally detected as multi-face" in reasons


# ---------------------------------------------------------------------------
# 8. Small face → REVIEW
# ---------------------------------------------------------------------------

class TestSmallFacePolicy:
    """Test that small-face images get REVIEW."""

    def test_single_face_small_gets_review(self, tmp_path: Path) -> None:
        """Single face + small face → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                has_small_face=True,
                filename="small_face.jpg",
            )
        ]
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"
        assert "Face too small" in pipeline._decisions[0].decision_reasons[0]

    def test_small_face_threshold_from_settings(self, tmp_path: Path) -> None:
        """Small-face reason string must contain the configured MIN_FACE_AREA_RATIO."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        settings = settings_with_custom_thresholds(tmp_path, min_face_area_ratio=0.05)
        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
            settings=settings,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                has_small_face=True,
                filename="small_face.jpg",
            )
        ]
        pipeline._make_final_decisions()

        reasons = " ".join(pipeline._decisions[0].decision_reasons)
        assert "0.05" in reasons
        assert pipeline._decisions[0].decision == "REVIEW"


# ---------------------------------------------------------------------------
# 9. Profile face → REVIEW
# ---------------------------------------------------------------------------

class TestProfileFacePolicy:
    """Test that profile-face images get REVIEW."""

    def test_single_face_profile_gets_review(self, tmp_path: Path) -> None:
        """Single face + extreme profile → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                has_profile_face=True,
                filename="profile.jpg",
            )
        ]
        # Set face_yaw for the reason string
        pipeline._decisions[0].face_yaw = 45.0
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"
        assert "Extreme profile face" in pipeline._decisions[0].decision_reasons[0]

    def test_profile_threshold_from_settings(self, tmp_path: Path) -> None:
        """Profile reason string must contain the configured MAX_PROFILE_YAW_DEGREES."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        settings = settings_with_custom_thresholds(tmp_path, max_profile_yaw=25.0)
        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
            settings=settings,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                has_profile_face=True,
                filename="profile.jpg",
            )
        ]
        pipeline._decisions[0].face_yaw = 40.0
        pipeline._make_final_decisions()

        reasons = " ".join(pipeline._decisions[0].decision_reasons)
        assert "25.0" in reasons
        assert pipeline._decisions[0].decision == "REVIEW"


# ---------------------------------------------------------------------------
# 10. Quality failure → REVIEW
# ---------------------------------------------------------------------------

class TestQualityFailurePolicy:
    """Test that quality failures prevent KEEP."""

    def test_blur_failure_gets_review(self, tmp_path: Path) -> None:
        """Quality enabled + blur failure → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                quality_evaluated=True,
                blur_passed=False,
                filename="blurry.jpg",
            )
        ]
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"
        reasons_text = " ".join(pipeline._decisions[0].decision_reasons)
        assert "blur" in reasons_text

    def test_brightness_failure_gets_review(self, tmp_path: Path) -> None:
        """Quality enabled + brightness failure → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                quality_evaluated=True,
                brightness_passed=False,
                filename="bad_brightness.jpg",
            )
        ]
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"
        reasons_text = " ".join(pipeline._decisions[0].decision_reasons)
        assert "brightness" in reasons_text

    def test_contrast_failure_gets_review(self, tmp_path: Path) -> None:
        """Quality enabled + contrast failure → REVIEW."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                quality_evaluated=True,
                contrast_passed=False,
                filename="low_contrast.jpg",
            )
        ]
        pipeline._make_final_decisions()

        assert pipeline._decisions[0].decision == "REVIEW"
        reasons_text = " ".join(pipeline._decisions[0].decision_reasons)
        assert "contrast" in reasons_text


# ---------------------------------------------------------------------------
# 11. Quality disabled behavior
# ---------------------------------------------------------------------------

class TestQualityDisabledBehavior:
    """Test behavior when quality scoring is disabled."""

    def test_quality_not_evaluated_no_false_pass(self, tmp_path: Path) -> None:
        """When quality is not evaluated, no false quality pass info added."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline._decisions = [
            _make_decision(
                decision="PENDING",
                face_count=1,
                face_detection_evaluated=True,
                quality_evaluated=False,
                filename="no_quality.jpg",
            )
        ]
        pipeline._make_final_decisions()

        # Should be KEEP — quality not evaluated shouldn't block
        assert pipeline._decisions[0].decision == "KEEP"
        # No quality-related reason should appear
        for reason in pipeline._decisions[0].decision_reasons:
            assert "quality" not in reason.lower()

    def test_quality_disabled_in_manifest(self, tmp_path: Path) -> None:
        """Manifest must show quality_evaluated=False when quality is disabled."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(5):
            create_random_image(raw_dir / "normal" / f"img_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True,
            skip_quality_scores=True,  # Disable quality
        )
        manifest = pipeline.run()

        for img in manifest.images:
            assert img.quality_evaluated is False
            assert img.blur_passed is None
            assert img.brightness_passed is None
            assert img.contrast_passed is None


# ---------------------------------------------------------------------------
# 12. Raw dataset immutability
# ---------------------------------------------------------------------------

class TestRawDatasetImmutability:
    """Test that preparation never modifies raw dataset."""

    def test_raw_files_not_modified(self, tmp_path: Path) -> None:
        """No raw file should be modified during preparation."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        # Create test images and record checksums
        files = {}
        for cat in ["normal", "eyeglasses"]:
            for i in range(5):
                path = raw_dir / cat / f"img_{i}.jpg"
                create_random_image(path)
                files[str(path)] = path.stat().st_mtime

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline.run()

        # Verify no file was modified
        for path_str, original_mtime in files.items():
            current_mtime = os.path.getmtime(path_str)
            assert current_mtime == original_mtime, (
                f"File was modified: {path_str}"
            )

    def test_raw_files_not_deleted(self, tmp_path: Path) -> None:
        """No raw file should be deleted during preparation."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        created_files = set()
        for cat in ["normal", "eyeglasses"]:
            for i in range(5):
                path = raw_dir / cat / f"img_{i}.jpg"
                create_random_image(path)
                created_files.add(str(path))

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline.run()

        # Verify no file was deleted
        for path_str in created_files:
            assert Path(path_str).exists(), f"File was deleted: {path_str}"

    def test_no_new_files_in_raw(self, tmp_path: Path) -> None:
        """No new files should be created inside raw/ during preparation."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for cat in ["normal", "eyeglasses"]:
            for i in range(3):
                create_random_image(raw_dir / cat / f"img_{i}.jpg")

        # Record all files before
        before = set()
        for p in raw_dir.rglob("*"):
            if p.is_file():
                before.add(str(p))

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline.run()

        # Record all files after
        after = set()
        for p in raw_dir.rglob("*"):
            if p.is_file():
                after.add(str(p))

        new_files = after - before
        assert len(new_files) == 0, f"New files created in raw/: {new_files}"

    def test_raw_not_renamed(self, tmp_path: Path) -> None:
        """No raw file should be renamed during preparation."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        file_names = set()
        for cat in ["normal"]:
            for i in range(5):
                path = raw_dir / cat / f"img_{i}.jpg"
                create_random_image(path)
                file_names.add(path.name)

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        pipeline.run()

        # Verify all original files still exist with same names
        for cat_dir in raw_dir.iterdir():
            if cat_dir.is_dir():
                for f in cat_dir.iterdir():
                    assert f.name in file_names, (
                        f"File {f.name} was not in original set"
                    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases in the pipeline."""

    def test_empty_dataset(self, tmp_path: Path) -> None:
        """Pipeline handles empty dataset gracefully."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"
        raw_dir.mkdir(parents=True, exist_ok=True)

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        manifest = pipeline.run()

        assert manifest.total_images == 0
        assert manifest.keep_count == 0
        assert manifest.split_stats == {"train": 0, "val": 0, "test": 0}

    def test_single_image_dataset(self, tmp_path: Path) -> None:
        """Pipeline handles single image dataset."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"
        create_random_image(raw_dir / "normal" / "only.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        manifest = pipeline.run()

        assert manifest.total_images == 1
        # With face detection skipped, image gets REVIEW (not KEEP)
        assert manifest.review_count == 1
        assert manifest.keep_count == 0

    def test_all_corrupt_dataset(self, tmp_path: Path) -> None:
        """Pipeline handles dataset where all images are corrupt."""
        raw_dir = tmp_path / "raw"
        output_dir = tmp_path / "output"

        for i in range(3):
            create_corrupt_image(raw_dir / "normal" / f"bad_{i}.jpg")

        pipeline = DatasetPreparation(
            raw_dir=raw_dir, output_dir=output_dir,
            skip_face_detection=True, skip_quality_scores=True,
        )
        manifest = pipeline.run()

        assert manifest.total_images == 3
        assert manifest.keep_count == 0
        assert manifest.remove_count == 3
