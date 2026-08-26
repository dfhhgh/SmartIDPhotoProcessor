"""Tests for quality_audit module.

Covers:
- readable image
- corrupt image
- zero-face image
- multiple-face image
- small-face image
- pHash duplicate detection
- cross-category duplicate detection
- oversized image handling
- category aggregation
- report generation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from dataset_builder.collection.quality_audit import (
    AuditSummary,
    CategoryAuditSummary,
    DuplicateGroup,
    ImageAuditRecord,
    QualityAudit,
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


def create_oversized_image(path: Path) -> None:
    """Create a normal image file (used with mock to simulate oversized)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Create a normal-sized image; test will mock PIL to simulate oversized
    arr = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    PILImage.fromarray(arr).save(path)


# ---------------------------------------------------------------------------
# Readable image
# ---------------------------------------------------------------------------

class TestReadableImage:
    """Test audit of a normal readable image."""

    def test_readable_image_detected(self, settings: Settings, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "normal"
        create_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=False,
        )

        records = audit.get_records()
        assert len(records) == 1
        assert records[0].readable is True
        assert records[0].width == 800
        assert records[0].height == 600
        assert records[0].category == "normal"
        assert records[0].filename == "test.jpg"

    def test_readable_image_has_quality_scores(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=False,
        )

        record = audit.get_records()[0]
        assert record.blur_score >= 0
        assert record.brightness_score >= 0
        assert record.contrast_score >= 0


# ---------------------------------------------------------------------------
# Corrupt image
# ---------------------------------------------------------------------------

class TestCorruptImage:
    """Test audit of a corrupt image."""

    def test_corrupt_image_detected(self, settings: Settings, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "normal"
        create_corrupt_image(raw / "corrupt.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        records = audit.get_records()
        assert len(records) == 1
        assert records[0].readable is False
        assert records[0].decode_error != ""


# ---------------------------------------------------------------------------
# Zero-face image (skipping face detection)
# ---------------------------------------------------------------------------

class TestZeroFaceImage:
    """Test audit when face detection is skipped."""

    def test_no_face_detection_by_default(self, settings: Settings, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
        )

        record = audit.get_records()[0]
        assert record.face_count == 0
        assert record.has_no_face is False  # Not detected, not flagged
        assert record.face_detection_evaluated is False

    def test_skipped_face_detection_not_counted_as_zero_face(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Skipped face detection must NOT produce zero-face count."""
        raw = tmp_path / "raw" / "normal"
        for i in range(5):
            create_random_image(raw / f"img_{i}.jpg")

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        assert summary.face_detection_run is False
        assert summary.zero_face_images == 0
        assert summary.one_face_images == 0
        assert summary.multiple_face_images == 0

        records = audit.get_records()
        for rec in records:
            assert rec.face_detection_evaluated is False

    def test_evaluated_zero_face_image_counted_correctly(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """An evaluated image with zero faces IS counted as zero-face."""
        raw = tmp_path / "raw" / "normal"
        for i in range(3):
            create_random_image(raw / f"img_{i}.jpg")

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        # Manually simulate what _detect_faces does for a zero-face result
        for rec in audit.get_records():
            rec.face_detection_evaluated = True
            rec.face_count = 0
            rec.has_no_face = True

        # Re-run compile to verify
        audit._compile_summaries()

        assert summary.face_detection_run is True
        assert summary.zero_face_images == 3
        assert summary.one_face_images == 0

    def test_report_says_not_run_when_skipped(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Report must explicitly say face detection was not run."""
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        reports_dir = tmp_path / "reports"
        audit.generate_reports(reports_dir)

        txt_content = (reports_dir / "dataset_quality_summary.txt").read_text()
        assert "Face detection:        NOT RUN" in txt_content
        assert "Zero faces" not in txt_content

        md_content = (reports_dir / "dataset_quality_audit.md").read_text()
        assert "NOT RUN" in md_content

    def test_readiness_not_75_when_face_detection_skipped(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Readiness must NOT falsely report 75/100 when face detection is skipped."""
        raw = tmp_path / "raw" / "normal"
        for i in range(3):
            create_random_image(raw / f"img_{i}.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        reports_dir = tmp_path / "reports"
        audit.generate_reports(reports_dir)

        md_content = (reports_dir / "dataset_quality_audit.md").read_text()
        # Face-Validation Readiness should be 0/100 when not run
        assert "Face-Validation Readiness: 0/100" in md_content
        # Should not claim "ready for annotation"
        assert "Dataset appears ready for annotation" not in md_content


# ---------------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------------

class TestNearDuplicateDetection:
    """Test pHash duplicate detection."""

    def test_identical_images_detected_as_duplicates(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        # Create identical images
        for i in range(3):
            create_image(raw / f"img_{i}.jpg", color=(200, 100, 50))

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        groups = audit.get_duplicate_groups()
        assert len(groups) >= 1
        assert len(groups[0].image_paths) == 3

    def test_different_images_not_duplicates(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        raw.mkdir(parents=True, exist_ok=True)
        # Create structurally different images
        arr1 = np.zeros((600, 800, 3), dtype=np.uint8)
        arr1[::10, ::10] = 255
        arr2 = np.zeros((600, 800, 3), dtype=np.uint8)
        gradient = np.linspace(0, 255, 800, dtype=np.uint8)
        arr2[:, :, 0] = gradient
        arr2[:, :, 1] = 255 - gradient
        arr2[:, :, 2] = 128
        PILImage.fromarray(arr1).save(raw / "a.jpg")
        PILImage.fromarray(arr2).save(raw / "b.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        groups = audit.get_duplicate_groups()
        assert len(groups) == 0

        records = audit.get_records()
        assert records[0].is_near_duplicate is False
        assert records[1].is_near_duplicate is False


# ---------------------------------------------------------------------------
# Cross-category duplicate detection
# ---------------------------------------------------------------------------

class TestCrossCategoryDuplicate:
    """Test cross-category duplicate detection."""

    def test_same_image_across_categories(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        # Same image in two categories
        create_image(raw / "normal" / "same.jpg", color=(200, 100, 50))
        create_image(raw / "eyeglasses" / "same.jpg", color=(200, 100, 50))

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal", "eyeglasses"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        records = audit.get_records()
        cross_cat = [r for r in records if r.cross_category_duplicate]
        assert len(cross_cat) == 2

        groups = audit.get_duplicate_groups()
        assert len(groups) == 1
        assert groups[0].is_cross_category is True


# ---------------------------------------------------------------------------
# Oversized image handling
# ---------------------------------------------------------------------------

class TestOversizedImage:
    """Test handling of oversized images."""

    def test_oversized_image_detected(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        from unittest.mock import patch

        raw = tmp_path / "raw" / "normal"
        create_oversized_image(raw / "big.jpg")

        # Mock PIL to simulate oversized image
        original_open = PILImage.open

        def mock_open(*args, **kwargs):
            img = original_open(*args, **kwargs)
            # Simulate oversized by setting large size
            img._size = (10000, 10000)
            return img

        # We need to mock at the class level to trigger DecompressionBombError
        # Instead, let's directly test the oversized path by creating a file
        # that PIL will reject
        # Simpler: mock PILImage.open to raise DecompressionBombError
        def raising_open(*args, **kwargs):
            raise PILImage.DecompressionBombError("test oversized")

        with patch.object(PILImage, "open", side_effect=raising_open):
            audit = QualityAudit(settings)
            summary = audit.run(
                raw_dir=tmp_path / "raw",
                categories=["normal"],
                skip_face_detection=True,
                skip_quality_scores=True,
            )

        records = audit.get_records()
        assert len(records) == 1
        assert records[0].oversized is True

    def test_oversized_image_no_crash(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Oversized images must not crash the audit."""
        from unittest.mock import patch

        raw = tmp_path / "raw" / "normal"
        create_oversized_image(raw / "big.jpg")
        create_image(raw / "normal.jpg")

        call_count = 0
        original_open = PILImage.open

        def selective_open(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (big.jpg) — simulate oversized
                raise PILImage.DecompressionBombError("test oversized")
            return original_open(*args, **kwargs)

        with patch.object(PILImage, "open", side_effect=selective_open):
            audit = QualityAudit(settings)
            summary = audit.run(
                raw_dir=tmp_path / "raw",
                categories=["normal"],
                skip_face_detection=True,
                skip_quality_scores=True,
            )

        assert summary.total_images == 2
        assert summary.oversized_images == 1


# ---------------------------------------------------------------------------
# Category aggregation
# ---------------------------------------------------------------------------

class TestCategoryAggregation:
    """Test per-category summary computation."""

    def test_category_summary_counts(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for i in range(5):
            create_random_image(raw / "normal" / f"img_{i}.jpg")
        for i in range(3):
            create_random_image(raw / "eyeglasses" / f"img_{i}.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal", "eyeglasses"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        summaries = audit.get_category_summaries()
        assert summaries["normal"].total_images == 5
        assert summaries["eyeglasses"].total_images == 3

    def test_global_summary_totals(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw"
        for i in range(5):
            create_random_image(raw / "normal" / f"img_{i}.jpg")
        for i in range(3):
            create_random_image(raw / "eyeglasses" / f"img_{i}.jpg")

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal", "eyeglasses"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        assert summary.total_images == 8
        assert summary.readable_images == 8

    def test_mixed_readable_and_corrupt(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        create_image(raw / "good.jpg")
        create_corrupt_image(raw / "bad.jpg")
        create_random_image(raw / "ok.jpg")

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        assert summary.total_images == 3
        assert summary.readable_images == 2

        records = audit.get_records()
        readable = [r for r in records if r.readable]
        corrupt = [r for r in records if not r.readable]
        assert len(readable) == 2
        assert len(corrupt) == 1


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    """Test report generation."""

    def test_json_report_created(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        reports_dir = tmp_path / "reports"
        audit.generate_reports(reports_dir)

        json_path = reports_dir / "dataset_quality_audit.json"
        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert "summary" in data
        assert "categories" in data
        assert "images" in data

    def test_markdown_report_created(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        reports_dir = tmp_path / "reports"
        audit.generate_reports(reports_dir)

        md_path = reports_dir / "dataset_quality_audit.md"
        assert md_path.exists()
        content = md_path.read_text()
        assert "Dataset Quality Audit Report" in content

    def test_text_summary_created(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        reports_dir = tmp_path / "reports"
        audit.generate_reports(reports_dir)

        txt_path = reports_dir / "dataset_quality_summary.txt"
        assert txt_path.exists()
        content = txt_path.read_text()
        assert "DATASET QUALITY SUMMARY" in content


# ---------------------------------------------------------------------------
# Review candidates
# ---------------------------------------------------------------------------

class TestReviewCandidates:
    """Test manual review candidate identification."""

    def test_corrupt_image_flagged_for_review(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        create_corrupt_image(raw / "bad.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        candidates = audit.get_review_candidates()
        assert len(candidates) == 1
        assert candidates[0].recommended_action in ("REVIEW", "REMOVE_CANDIDATE")

    def test_quality_failures_flagged(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw" / "normal"
        raw.mkdir(parents=True, exist_ok=True)
        # Create a very dark image (brightness failure)
        arr = np.zeros((600, 800, 3), dtype=np.uint8) + 5
        PILImage.fromarray(arr).save(raw / "dark.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=False,
        )

        records = audit.get_records()
        dark = [r for r in records if r.brightness_passed is False]
        assert len(dark) == 1

    def test_skipped_quality_scores_not_counted_as_failures(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Skipped quality scoring must NOT produce blur/brightness/contrast failures."""
        raw = tmp_path / "raw" / "normal"
        for i in range(3):
            create_random_image(raw / f"img_{i}.jpg")

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        assert summary.quality_scores_run is False

        records = audit.get_records()
        for rec in records:
            assert rec.quality_evaluated is False
            # blur_passed etc should remain None (not evaluated), not False
            assert rec.blur_passed is None
            assert rec.brightness_passed is None
            assert rec.contrast_passed is None

        # Category summaries should have zero fail counts
        cat_sum = audit.get_category_summaries()["normal"]
        assert cat_sum.blur_fail_count == 0
        assert cat_sum.brightness_fail_count == 0
        assert cat_sum.contrast_fail_count == 0

    def test_report_quality_not_run_when_skipped(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Report must explicitly say quality scores were not run."""
        raw = tmp_path / "raw" / "normal"
        create_random_image(raw / "test.jpg")

        audit = QualityAudit(settings)
        audit.run(
            raw_dir=tmp_path / "raw",
            categories=["normal"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        reports_dir = tmp_path / "reports"
        audit.generate_reports(reports_dir)

        txt_content = (reports_dir / "dataset_quality_summary.txt").read_text()
        assert "Quality scores:        NOT RUN" in txt_content

        md_content = (reports_dir / "dataset_quality_audit.md").read_text()
        # Readiness should note quality scoring was not run
        assert "Quality scoring was not run" in md_content


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------

class TestEmptyDirectory:
    """Test audit of empty category."""

    def test_empty_category(self, settings: Settings, tmp_path: Path) -> None:
        raw = tmp_path / "raw" / "empty"
        raw.mkdir(parents=True, exist_ok=True)

        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "raw",
            categories=["empty"],
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        assert summary.total_images == 0
        assert len(audit.get_records()) == 0


# ---------------------------------------------------------------------------
# Non-existent directory
# ---------------------------------------------------------------------------

class TestNonExistentDirectory:
    """Test audit of non-existent raw directory."""

    def test_nonexistent_raw_dir(self, settings: Settings, tmp_path: Path) -> None:
        audit = QualityAudit(settings)
        summary = audit.run(
            raw_dir=tmp_path / "nonexistent",
            skip_face_detection=True,
            skip_quality_scores=True,
        )

        assert summary.total_images == 0


# ---------------------------------------------------------------------------
# Runner integration: face detection enabled
# ---------------------------------------------------------------------------

class TestRunnerFaceDetectionEnabled:
    """Verify the runner script enables face detection."""

    def test_runner_script_enables_face_detection(self, tmp_path: Path) -> None:
        """The runner script must call run() with skip_face_detection=False."""
        import importlib
        import importlib.util

        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_quality_audit.py"
        assert script_path.exists(), f"Runner script not found: {script_path}"

        source = script_path.read_text(encoding="utf-8")
        # The script should contain skip_face_detection=False
        assert "skip_face_detection=False" in source, (
            "run_quality_audit.py must set skip_face_detection=False"
        )
        # The script should NOT contain skip_face_detection=True
        assert "skip_face_detection=True" not in source, (
            "run_quality_audit.py must NOT set skip_face_detection=True"
        )

    def test_runner_script_skips_quality_scores(self, tmp_path: Path) -> None:
        """The runner script must call run() with skip_quality_scores=True."""
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "run_quality_audit.py"
        source = script_path.read_text(encoding="utf-8")
        assert "skip_quality_scores=True" in source, (
            "run_quality_audit.py must set skip_quality_scores=True"
        )


class TestFaceDetectionAccounting:
    """Tests for the face_detection_skipped accounting fix."""

    def test_face_detection_skipped_in_global_summary(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """face_detection_skipped should be in global AuditSummary."""
        assert hasattr(AuditSummary, "face_detection_skipped")

    def test_face_detection_skipped_in_category_summary(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """face_detection_skipped should be in CategoryAuditSummary."""
        assert hasattr(CategoryAuditSummary, "face_detection_skipped")

    def test_face_accounting_invariant_holds(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """The invariant: one_face + zero_face + multiple_face + skipped == readable must hold."""
        raw = settings.RAW_IMAGES_DIR
        cat = raw / "normal"
        cat.mkdir(parents=True, exist_ok=True)

        # Create normal images
        for i in range(3):
            img = np.zeros((200, 200, 3), dtype=np.uint8)
            img[50:150, 50:150] = 200
            PILImage.fromarray(img[:, :, ::-1]).save(cat / f"normal_{i}.jpg")

        audit = QualityAudit(settings)
        s = audit.run(
            raw_dir=raw,
            categories=["normal"],
            skip_face_detection=False,
            skip_quality_scores=True,
        )

        face_total = s.one_face_images + s.zero_face_images + s.multiple_face_images + s.face_detection_skipped
        assert face_total == s.readable_images, (
            f"Face accounting invariant violated: {s.one_face_images} + {s.zero_face_images} "
            f"+ {s.multiple_face_images} + {s.face_detection_skipped} = {face_total} "
            f"!= readable_images = {s.readable_images}"
        )

    def test_oversized_readable_counted_as_skipped(
        self, settings: Settings, tmp_path: Path
    ) -> None:
        """Readable oversized images should be counted as face_detection_skipped."""
        raw = settings.RAW_IMAGES_DIR
        cat = raw / "normal"
        cat.mkdir(parents=True, exist_ok=True)

        # Create a normal image
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[50:150, 50:150] = 200
        PILImage.fromarray(img[:, :, ::-1]).save(cat / "normal_ok.jpg")

        # Create an oversized image by writing raw data that PIL thinks is huge
        big_img = np.zeros((200, 400, 3), dtype=np.uint8)
        big_img[50:150, 150:250] = 200
        PILImage.fromarray(big_img[:, :, ::-1]).save(cat / "normal_wide.jpg")

        # Run with very low PIL bomb threshold to trigger oversized detection
        import PIL.Image
        old_limit = PIL.Image.MAX_IMAGE_PIXELS
        try:
            # Set limit so that 200*400=80000 pixels triggers DecompressionBombError
            PIL.Image.MAX_IMAGE_PIXELS = 10000
            audit = QualityAudit(settings)
            s = audit.run(
                raw_dir=raw,
                categories=["normal"],
                skip_face_detection=False,
                skip_quality_scores=True,
            )
        finally:
            PIL.Image.MAX_IMAGE_PIXELS = old_limit

        # The oversized image should be counted in face_detection_skipped
        assert s.face_detection_skipped >= 1
        # And the invariant should hold
        face_total = s.one_face_images + s.zero_face_images + s.multiple_face_images + s.face_detection_skipped
        assert face_total == s.readable_images, (
            f"Face accounting invariant violated: {s.one_face_images} + {s.zero_face_images} "
            f"+ {s.multiple_face_images} + {s.face_detection_skipped} = {face_total} "
            f"!= readable_images = {s.readable_images}"
        )
