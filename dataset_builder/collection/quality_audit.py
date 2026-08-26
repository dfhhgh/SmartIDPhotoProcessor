"""
Dataset quality audit module.

Analyzes all images under raw/<category>/ and produces structured
per-image audit records with face detection, quality scoring,
duplicate detection, and category-level summaries.

READ-ONLY with respect to the dataset — never modifies files.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
from PIL import Image as PILImage

from dataset_builder.config.settings import Settings


# ------------------------------------------------------------------
# Per-image audit record
# ------------------------------------------------------------------

@dataclass
class ImageAuditRecord:
    """Structured audit record for a single image."""

    path: str
    """Absolute path to the image file."""

    category: str
    """Category derived from parent directory."""

    filename: str
    """Image filename."""

    width: int = 0
    """Image width in pixels."""

    height: int = 0
    """Image height in pixels."""

    pixel_count: int = 0
    """Total pixel count (width * height)."""

    image_format: str = ""
    """Image format (JPEG, PNG, etc.)."""

    file_size: int = 0
    """File size in bytes."""

    readable: bool = False
    """Whether the image can be decoded."""

    decode_error: str = ""
    """Error message if decode failed."""

    oversized: bool = False
    """Whether the image exceeds PIL safety limits."""

    # Face detection
    face_detection_evaluated: bool = False
    """Whether face detection was actually performed on this image."""

    face_count: int = 0
    """Number of detected faces."""

    largest_face_bbox: list[float] | None = None
    """Largest face bounding box [x1, y1, x2, y2]."""

    largest_face_area_ratio: float = 0.0
    """Largest face area / image area."""

    largest_face_det_score: float = 0.0
    """Largest face detection confidence."""

    face_yaw: float | None = None
    """Largest face yaw in degrees (None if unavailable)."""

    face_pitch: float | None = None
    """Largest face pitch in degrees (None if unavailable)."""

    has_multiple_faces: bool = False
    """Whether multiple faces were detected."""

    has_no_face: bool = False
    """Whether zero faces were detected."""

    has_small_face: bool = False
    """Whether the face is below MIN_FACE_AREA_RATIO."""

    has_profile_face: bool = False
    """Whether the face exceeds MAX_PROFILE_YAW_DEGREES."""

    # Quality scores (from production validators if available)
    quality_evaluated: bool = False
    """Whether quality scoring was actually performed on this image."""

    blur_score: float = -1.0
    """Laplacian variance (higher = sharper). -1 if unavailable."""

    brightness_score: float = -1.0
    """Mean grayscale intensity (0-255). -1 if unavailable."""

    contrast_score: float = -1.0
    """StdDev of grayscale intensity. -1 if unavailable."""

    blur_passed: bool | None = None
    """Whether blur meets threshold. None if not evaluated."""

    brightness_passed: bool | None = None
    """Whether brightness is in range. None if not evaluated."""

    contrast_passed: bool | None = None
    """Whether contrast meets threshold. None if not evaluated."""

    # Perceptual hash
    phash: str = ""
    """Perceptual hash hex string."""

    phash_computed: bool = False
    """Whether pHash was successfully computed."""

    # Duplicate tracking
    duplicate_group_id: int = -1
    """Group ID for near-duplicate clusters. -1 if not a duplicate."""

    duplicate_distances: list[int] = field(default_factory=list)
    """Hamming distances to other images in the same duplicate group."""

    is_exact_duplicate: bool = False
    """Whether an identical (distance=0) duplicate exists."""

    is_near_duplicate: bool = False
    """Whether a near-duplicate (distance<=threshold) exists."""

    cross_category_duplicate: bool = False
    """Whether the nearest duplicate is in a different category."""

    nearest_duplicate_path: str = ""
    """Path to the nearest duplicate image."""

    nearest_duplicate_distance: int = -1
    """Hamming distance to the nearest duplicate."""

    nearest_duplicate_category: str = ""
    """Category of the nearest duplicate."""

    # Audit status
    audit_status: str = "pending"
    """Audit status: pass, warn, fail, error."""

    category_review_required: bool = False
    """Whether the image needs manual category verification."""

    review_reasons: list[str] = field(default_factory=list)
    """Reasons requiring manual review."""

    recommended_action: str = "KEEP"
    """Recommended action: KEEP, REVIEW, REMOVE_CANDIDATE."""

    # Fine-tuning relevance
    useful_for_eye_improvement: bool = False
    """Whether image is useful for LEFT_EYE/RIGHT_EYE improvement."""

    contains_eyeglass: bool = False
    """Whether image contains EYE_GLASS semantic part."""

    contains_hat: bool = False
    """Whether image contains HAT semantic part."""

    contains_facial_occlusion: bool = False
    """Whether image contains facial occlusion."""

    is_baseline_normal: bool = False
    """Whether image is a clean baseline/normal image."""


@dataclass
class DuplicateGroup:
    """A group of near-duplicate images."""

    group_id: int
    """Unique group identifier."""

    image_paths: list[str] = field(default_factory=list)
    """Paths to all images in this group."""

    categories: list[str] = field(default_factory=list)
    """Categories of images in this group."""

    distances: list[int] = field(default_factory=list)
    """Pairwise Hamming distances."""

    is_cross_category: bool = False
    """Whether the group spans multiple categories."""

    best_image_path: str = ""
    """Path to the recommended image (highest quality)."""


@dataclass
class CategoryAuditSummary:
    """Audit summary for a single category."""

    category: str
    total_images: int = 0
    readable_images: int = 0
    oversized_images: int = 0
    one_face_images: int = 0
    zero_face_images: int = 0
    multiple_face_images: int = 0
    face_detection_skipped: int = 0
    small_face_images: int = 0
    profile_face_images: int = 0
    near_duplicate_count: int = 0
    cross_category_duplicate_count: int = 0
    blur_fail_count: int = 0
    brightness_fail_count: int = 0
    contrast_fail_count: int = 0
    review_required_count: int = 0
    clean_candidate_count: int = 0
    avg_blur_score: float = 0.0
    avg_brightness_score: float = 0.0
    avg_contrast_score: float = 0.0
    useful_for_eye_improvement: int = 0
    contains_eyeglass: int = 0
    contains_hat: int = 0
    contains_facial_occlusion: int = 0
    is_baseline_normal: int = 0


@dataclass
class AuditSummary:
    """Global audit summary."""

    total_images: int = 0
    readable_images: int = 0
    oversized_images: int = 0
    face_detection_run: bool = False
    quality_scores_run: bool = False
    one_face_images: int = 0
    zero_face_images: int = 0
    multiple_face_images: int = 0
    face_detection_skipped: int = 0
    """Readable images skipped by face detection (e.g. oversized)."""
    small_face_images: int = 0
    profile_face_images: int = 0
    total_near_duplicates: int = 0
    total_cross_category_duplicates: int = 0
    total_duplicate_groups: int = 0
    review_required_count: int = 0
    clean_candidate_count: int = 0
    oversized_image_paths: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Quality audit engine
# ------------------------------------------------------------------

class QualityAudit:
    """Dataset quality audit engine.

    Analyzes all images under raw/<category>/ and produces structured
    per-image audit records. READ-ONLY with respect to the dataset.

    Parameters
    ----------
    settings:
        Application settings.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._records: list[ImageAuditRecord] = []
        self._duplicate_groups: list[DuplicateGroup] = []
        self._category_summaries: dict[str, CategoryAuditSummary] = {}
        self._summary = AuditSummary()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        raw_dir: Path | None = None,
        categories: list[str] | None = None,
        skip_face_detection: bool = False,
        skip_quality_scores: bool = False,
    ) -> AuditSummary:
        """Run the full quality audit.

        Parameters
        ----------
        raw_dir:
            Root raw directory. If None, uses Settings.RAW_IMAGES_DIR.
        categories:
            Categories to audit. If None, audits all discovered categories.
        skip_face_detection:
            If True, skip face detection (much faster).
        skip_quality_scores:
            If True, skip blur/brightness/contrast scoring.

        Returns
        -------
        AuditSummary
            Global audit summary.
        """
        base_dir = raw_dir or self._settings.RAW_IMAGES_DIR

        if not base_dir.exists():
            return self._summary

        # Discover categories
        if categories is None:
            categories = sorted([
                d.name for d in base_dir.iterdir()
                if d.is_dir()
            ])

        # Phase 1: Scan all images
        print("Phase 1: Scanning images...")
        self._scan_images(base_dir, categories)

        # Phase 2: Perceptual hash computation
        print("Phase 2: Computing perceptual hashes...")
        self._compute_hashes()

        # Phase 3: Near-duplicate detection
        print("Phase 3: Detecting near-duplicates...")
        self._detect_duplicates()

        # Phase 4: Face detection (optional)
        if not skip_face_detection:
            print("Phase 4: Running face detection...")
            self._detect_faces()
        else:
            print("Phase 4: Skipping face detection.")

        # Phase 5: Quality scoring (optional)
        if not skip_quality_scores:
            print("Phase 5: Computing quality scores...")
            self._compute_quality_scores()
        else:
            print("Phase 5: Skipping quality scores.")

        # Phase 6: Category correctness hints
        print("Phase 6: Analyzing category correctness...")
        self._analyze_category_correctness()

        # Phase 7: Compile summaries
        print("Phase 7: Compiling summaries...")
        self._compile_summaries()

        return self._summary

    def get_records(self) -> list[ImageAuditRecord]:
        """Return all per-image audit records."""
        return self._records

    def get_duplicate_groups(self) -> list[DuplicateGroup]:
        """Return all duplicate groups."""
        return self._duplicate_groups

    def get_category_summaries(self) -> dict[str, CategoryAuditSummary]:
        """Return per-category audit summaries."""
        return self._category_summaries

    def get_summary(self) -> AuditSummary:
        """Return global audit summary."""
        return self._summary

    def get_review_candidates(self) -> list[ImageAuditRecord]:
        """Return images requiring manual review."""
        return [r for r in self._records if r.recommended_action != "KEEP"]

    def get_oversized_images(self) -> list[ImageAuditRecord]:
        """Return oversized images."""
        return [r for r in self._records if r.oversized]

    # ------------------------------------------------------------------
    # Phase 1: Scan images
    # ------------------------------------------------------------------

    def _scan_images(
        self, base_dir: Path, categories: list[str]
    ) -> None:
        """Scan all images and create initial audit records."""
        for cat in categories:
            cat_dir = base_dir / cat
            if not cat_dir.exists():
                continue

            for file_path in sorted(cat_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in self._settings.SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                record = self._create_record(file_path, cat)
                self._records.append(record)

        print(f"  Scanned {len(self._records)} images across {len(categories)} categories.")

    def _create_record(self, file_path: Path, category: str) -> ImageAuditRecord:
        """Create an initial audit record for an image."""
        record = ImageAuditRecord(
            path=str(file_path),
            category=category,
            filename=file_path.name,
        )

        # File size
        try:
            record.file_size = file_path.stat().st_size
        except OSError:
            record.file_size = 0

        # Try to decode with PIL (for format info and oversized detection)
        try:
            with PILImage.open(file_path) as img:
                record.image_format = img.format or ""
                record.width, record.height = img.size
                record.pixel_count = record.width * record.height
                record.readable = True
        except PILImage.DecompressionBombError:
            # Oversized image — get dimensions via cv2 instead
            record.oversized = True
            self._read_oversized_dimensions(record, file_path)
        except Exception as e:
            record.decode_error = str(e)

        # Also try cv2 for validation
        if record.readable:
            try:
                img = cv2.imread(str(file_path))
                if img is None:
                    record.readable = False
                    record.decode_error = "cv2.imread returned None"
                else:
                    h, w = img.shape[:2]
                    if record.width == 0:
                        record.width = w
                        record.height = h
                        record.pixel_count = w * h
            except Exception as e:
                record.readable = False
                record.decode_error = str(e)

        return record

    def _read_oversized_dimensions(self, record: ImageAuditRecord, file_path: Path) -> None:
        """Read dimensions of an oversized image using cv2."""
        try:
            # Use cv2 which doesn't have PIL's decompression limit
            img = cv2.imread(str(file_path))
            if img is not None:
                record.height, record.width = img.shape[:2]
                record.pixel_count = record.width * record.height
                record.readable = True
            else:
                # Try to get dimensions from file header without full decode
                with open(file_path, "rb") as f:
                    header = f.read(32)
                record.decode_error = "cv2.imread returned None for oversized image"
        except Exception as e:
            record.decode_error = f"Failed to read oversized image: {e}"

    # ------------------------------------------------------------------
    # Phase 2: Perceptual hash computation
    # ------------------------------------------------------------------

    def _compute_hashes(self) -> None:
        """Compute pHash for all readable images."""
        count = 0
        for record in self._records:
            if not record.readable:
                continue

            h = self._compute_phash(Path(record.path))
            if h is not None:
                record.phash = str(h)
                record.phash_computed = True
                count += 1

        print(f"  Computed pHash for {count}/{len(self._records)} images.")

    def _compute_phash(self, file_path: Path) -> imagehash.ImageHash | None:
        """Compute perceptual hash for an image file."""
        if not file_path.exists():
            return None
        try:
            # Handle oversized images
            old_limit = PILImage.MAX_IMAGE_PIXELS
            try:
                PILImage.MAX_IMAGE_PIXELS = None
                with PILImage.open(file_path) as img:
                    return imagehash.phash(
                        img, hash_size=self._settings.IMAGEHASH_SIZE
                    )
            except PILImage.DecompressionBombError:
                # For oversized images, resize before hashing
                return self._compute_phash_oversized(file_path)
            finally:
                PILImage.MAX_IMAGE_PIXELS = old_limit
        except Exception:
            return None

    def _compute_phash_oversized(self, file_path: Path) -> imagehash.ImageHash | None:
        """Compute pHash for an oversized image by resizing first."""
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                return None
            # Resize to a reasonable size for hashing
            h, w = img.shape[:2]
            scale = min(1024 / w, 1024 / h, 1.0)
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return imagehash.phash(pil_img, hash_size=self._settings.IMAGEHASH_SIZE)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Phase 3: Near-duplicate detection
    # ------------------------------------------------------------------

    def _detect_duplicates(self) -> None:
        """Detect near-duplicates using pHash with union-find clustering."""
        # Build hash index
        hash_records = [
            (i, r) for i, r in enumerate(self._records)
            if r.phash_computed
        ]

        if len(hash_records) < 2:
            print("  Not enough hashable images for duplicate detection.")
            return

        # Union-Find for clustering
        parent = list(range(len(self._records)))
        rank = [0] * len(self._records)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        # Pairwise comparison (O(N^2) — acceptable for <10k images)
        threshold = self._settings.DUPLICATE_DISTANCE_THRESHOLD
        pairs_compared = 0

        for i, (idx_a, rec_a) in enumerate(hash_records):
            hash_a = imagehash.hex_to_hash(rec_a.phash)
            for j in range(i + 1, len(hash_records)):
                idx_b, rec_b = hash_records[j]
                hash_b = imagehash.hex_to_hash(rec_b.phash)
                distance = hash_a - hash_b
                pairs_compared += 1

                if distance <= threshold:
                    union(idx_a, idx_b)

                    # Track nearest duplicate for each record
                    if rec_a.nearest_duplicate_distance < 0 or distance < rec_a.nearest_duplicate_distance:
                        rec_a.nearest_duplicate_distance = distance
                        rec_a.nearest_duplicate_path = rec_b.path
                        rec_a.nearest_duplicate_category = rec_b.category
                    if rec_b.nearest_duplicate_distance < 0 or distance < rec_b.nearest_duplicate_distance:
                        rec_b.nearest_duplicate_distance = distance
                        rec_b.nearest_duplicate_path = rec_a.path
                        rec_b.nearest_duplicate_category = rec_a.category

        # Build duplicate groups
        groups: dict[int, list[int]] = {}
        for i, (idx, _) in enumerate(hash_records):
            root = find(idx)
            if root not in groups:
                groups[root] = []
            groups[root].append(idx)

        group_id = 0
        for root, members in groups.items():
            if len(members) < 2:
                continue

            group = DuplicateGroup(group_id=group_id)
            categories_seen = set()

            for idx in members:
                rec = self._records[idx]
                group.image_paths.append(rec.path)
                group.categories.append(rec.category)
                categories_seen.add(rec.category)

                # Mark records
                rec.duplicate_group_id = group_id
                rec.is_near_duplicate = True

                if rec.nearest_duplicate_distance == 0:
                    rec.is_exact_duplicate = True

                if rec.nearest_duplicate_category != rec.category and rec.nearest_duplicate_category:
                    rec.cross_category_duplicate = True

            group.is_cross_category = len(categories_seen) > 1
            self._duplicate_groups.append(group)

            # Mark cross-category in group
            if group.is_cross_category:
                for idx in members:
                    self._records[idx].cross_category_duplicate = True

            group_id += 1

        total_dupes = sum(1 for r in self._records if r.is_near_duplicate)
        cross_cat = sum(1 for r in self._records if r.cross_category_duplicate)
        print(f"  Found {len(self._duplicate_groups)} duplicate groups "
              f"({total_dupes} images, {cross_cat} cross-category).")

    # ------------------------------------------------------------------
    # Phase 4: Face detection
    # ------------------------------------------------------------------

    def _detect_faces(self) -> None:
        """Run face detection on all readable images."""
        try:
            from pipeline.detector import FaceDetector
            detector = FaceDetector()
        except ImportError:
            print("  FaceDetector not available — skipping face detection.")
            return
        except Exception as e:
            print(f"  FaceDetector init failed: {e} — skipping face detection.")
            return

        count = 0
        for record in self._records:
            if not record.readable or record.oversized:
                continue

            try:
                img = cv2.imread(record.path)
                if img is None:
                    continue

                faces = detector.detect(img)
                record.face_count = len(faces)
                record.face_detection_evaluated = True

                if len(faces) == 0:
                    record.has_no_face = True
                elif len(faces) > 1:
                    record.has_multiple_faces = True

                if len(faces) >= 1:
                    # Find largest face
                    largest = max(
                        faces,
                        key=lambda f: (
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                            if hasattr(f, "bbox") else 0
                        ),
                    )

                    if hasattr(largest, "bbox"):
                        x1, y1, x2, y2 = largest.bbox[:4]
                        record.largest_face_bbox = [float(x1), float(y1), float(x2), float(y2)]
                        face_area = (x2 - x1) * (y2 - y1)
                        img_area = record.width * record.height
                        if img_area > 0:
                            record.largest_face_area_ratio = face_area / img_area

                    if hasattr(largest, "det_score"):
                        record.largest_face_det_score = float(largest.det_score)

                    if hasattr(largest, "pose") and largest.pose is not None:
                        try:
                            pitch, yaw, roll = largest.pose
                            record.face_pitch = float(pitch)
                            record.face_yaw = float(yaw)
                        except (TypeError, ValueError):
                            pass

                    # Check face size
                    if record.largest_face_area_ratio < self._settings.MIN_FACE_AREA_RATIO:
                        record.has_small_face = True

                    # Check profile face
                    if record.face_yaw is not None:
                        if abs(record.face_yaw) > self._settings.MAX_PROFILE_YAW_DEGREES:
                            record.has_profile_face = True

                count += 1

            except Exception:
                continue

        print(f"  Detected faces in {count} images.")

    # ------------------------------------------------------------------
    # Phase 5: Quality scores
    # ------------------------------------------------------------------

    def _compute_quality_scores(self) -> None:
        """Compute blur, brightness, and contrast scores."""
        count = 0
        for record in self._records:
            if not record.readable or record.oversized:
                continue

            try:
                img = cv2.imread(record.path)
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Blur (Laplacian variance)
                record.blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                record.blur_passed = record.blur_score >= self._settings.MIN_BLUR_SCORE

                # Brightness (mean intensity)
                record.brightness_score = float(np.mean(gray))
                record.brightness_passed = (
                    self._settings.MIN_BRIGHTNESS
                    <= record.brightness_score
                    <= self._settings.MAX_BRIGHTNESS
                )

                # Contrast (stddev)
                record.contrast_score = float(np.std(gray))
                record.contrast_passed = record.contrast_score >= self._settings.MIN_CONTRAST

                record.quality_evaluated = True
                count += 1

            except Exception:
                continue

        print(f"  Computed quality scores for {count} images.")

    # ------------------------------------------------------------------
    # Phase 6: Category correctness hints
    # ------------------------------------------------------------------

    def _analyze_category_correctness(self) -> None:
        """Analyze category correctness and flag review candidates."""
        # Categories that need semantic verification
        occlusion_categories = {
            "helmet", "mask", "cap", "scarf", "sunglasses",
            "eyeglasses", "hijab", "hair_occlusion",
        }

        for record in self._records:
            reasons = []

            # Non-readable images always need review
            if not record.readable:
                reasons.append(f"Unreadable/corrupt image: {record.decode_error}")
                record.review_reasons = reasons
                record.category_review_required = True
                record.recommended_action = "REVIEW"
                continue

            # Check if occlusion category has face (expected for most)
            if record.category in occlusion_categories:
                if record.has_no_face:
                    reasons.append(f"{record.category}: no face detected — may not contain the intended condition")
                elif record.has_multiple_faces:
                    reasons.append(f"{record.category}: multiple faces — review for category correctness")

            # For normal category, check for potential occlusions
            if record.category == "normal":
                if record.has_no_face:
                    reasons.append("normal: no face detected — unexpected for baseline")

            # Oversized images need review
            if record.oversized:
                reasons.append("Oversized image — verify quality after resize")

            # Quality failures
            if record.blur_passed is False:
                reasons.append("Blurry image (below threshold)")
            if record.brightness_passed is False:
                reasons.append("Poor brightness (outside acceptable range)")
            if record.contrast_passed is False:
                reasons.append("Low contrast (below threshold)")

            # Small face
            if record.has_small_face:
                reasons.append("Face too small (below minimum area ratio)")

            # Profile face
            if record.has_profile_face:
                reasons.append("Extreme profile face (high yaw angle)")

            # Near-duplicate
            if record.is_near_duplicate:
                reasons.append(f"Near-duplicate of {record.nearest_duplicate_path}")

            if reasons:
                record.review_reasons = reasons
                record.category_review_required = True
                if any("no face" in r or "Blurry" in r or "Oversized" in r or "Near-duplicate" in r
                       for r in reasons):
                    record.recommended_action = "REVIEW"
                else:
                    record.recommended_action = "REVIEW"

    # ------------------------------------------------------------------
    # Phase 7: Compile summaries
    # ------------------------------------------------------------------

    def _compile_summaries(self) -> None:
        """Compile per-category and global summaries."""
        # Determine which phases were actually run
        any_face_evaluated = any(r.face_detection_evaluated for r in self._records)
        any_quality_evaluated = any(r.quality_evaluated for r in self._records)

        # Group records by category
        by_category: dict[str, list[ImageAuditRecord]] = {}
        for rec in self._records:
            by_category.setdefault(rec.category, []).append(rec)

        for cat, records in by_category.items():
            summary = CategoryAuditSummary(category=cat)
            summary.total_images = len(records)

            for rec in records:
                if rec.readable:
                    summary.readable_images += 1
                if rec.oversized:
                    summary.oversized_images += 1

                # Face stats — only count when face detection was actually performed
                if rec.face_detection_evaluated:
                    if rec.face_count == 1:
                        summary.one_face_images += 1
                    elif rec.face_count == 0:
                        summary.zero_face_images += 1
                    elif rec.face_count > 1:
                        summary.multiple_face_images += 1

                    if rec.has_small_face:
                        summary.small_face_images += 1
                    if rec.has_profile_face:
                        summary.profile_face_images += 1
                elif rec.readable:
                    # Readable but face detection was not evaluated (oversized or init failure)
                    summary.face_detection_skipped += 1

                # Duplicate stats
                if rec.is_near_duplicate:
                    summary.near_duplicate_count += 1
                if rec.cross_category_duplicate:
                    summary.cross_category_duplicate_count += 1

                # Quality stats — only count when quality scoring was actually performed
                if rec.quality_evaluated:
                    if rec.blur_passed is False:
                        summary.blur_fail_count += 1
                    if rec.brightness_passed is False:
                        summary.brightness_fail_count += 1
                    if rec.contrast_passed is False:
                        summary.contrast_fail_count += 1

                # Review
                if rec.category_review_required:
                    summary.review_required_count += 1
                if rec.recommended_action == "KEEP":
                    summary.clean_candidate_count += 1

                # Fine-tuning relevance
                if rec.useful_for_eye_improvement:
                    summary.useful_for_eye_improvement += 1
                if rec.contains_eyeglass:
                    summary.contains_eyeglass += 1
                if rec.contains_hat:
                    summary.contains_hat += 1
                if rec.contains_facial_occlusion:
                    summary.contains_facial_occlusion += 1
                if rec.is_baseline_normal:
                    summary.is_baseline_normal += 1

            # Compute averages (only from evaluated records)
            blur_scores = [r.blur_score for r in records if r.quality_evaluated and r.blur_score >= 0]
            brightness_scores = [r.brightness_score for r in records if r.quality_evaluated and r.brightness_score >= 0]
            contrast_scores = [r.contrast_score for r in records if r.quality_evaluated and r.contrast_score >= 0]

            summary.avg_blur_score = sum(blur_scores) / len(blur_scores) if blur_scores else 0.0
            summary.avg_brightness_score = sum(brightness_scores) / len(brightness_scores) if brightness_scores else 0.0
            summary.avg_contrast_score = sum(contrast_scores) / len(contrast_scores) if contrast_scores else 0.0

            self._category_summaries[cat] = summary

        # Global summary
        self._summary.total_images = len(self._records)
        self._summary.readable_images = sum(s.readable_images for s in self._category_summaries.values())
        self._summary.oversized_images = sum(s.oversized_images for s in self._category_summaries.values())
        self._summary.one_face_images = sum(s.one_face_images for s in self._category_summaries.values())
        self._summary.zero_face_images = sum(s.zero_face_images for s in self._category_summaries.values())
        self._summary.multiple_face_images = sum(s.multiple_face_images for s in self._category_summaries.values())
        self._summary.face_detection_skipped = sum(s.face_detection_skipped for s in self._category_summaries.values())
        self._summary.small_face_images = sum(s.small_face_images for s in self._category_summaries.values())
        self._summary.profile_face_images = sum(s.profile_face_images for s in self._category_summaries.values())
        self._summary.total_near_duplicates = sum(s.near_duplicate_count for s in self._category_summaries.values())
        self._summary.total_cross_category_duplicates = sum(s.cross_category_duplicate_count for s in self._category_summaries.values())
        self._summary.total_duplicate_groups = len(self._duplicate_groups)
        self._summary.review_required_count = sum(s.review_required_count for s in self._category_summaries.values())
        self._summary.clean_candidate_count = sum(s.clean_candidate_count for s in self._category_summaries.values())
        self._summary.face_detection_run = any_face_evaluated
        self._summary.quality_scores_run = any_quality_evaluated
        self._summary.oversized_image_paths = [
            r.path for r in self._records if r.oversized
        ]

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_reports(self, reports_dir: Path) -> None:
        """Generate all output reports.

        Parameters
        ----------
        reports_dir:
            Directory to write reports to.
        """
        reports_dir.mkdir(parents=True, exist_ok=True)

        self._generate_json(reports_dir / "dataset_quality_audit.json")
        self._generate_markdown(reports_dir / "dataset_quality_audit.md")
        self._generate_text_summary(reports_dir / "dataset_quality_summary.txt")

        print(f"Reports written to {reports_dir}")

    def _generate_json(self, path: Path) -> None:
        """Generate machine-readable JSON report."""
        data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": asdict(self._summary),
            "categories": {
                cat: asdict(s) for cat, s in self._category_summaries.items()
            },
            "duplicate_groups": [asdict(g) for g in self._duplicate_groups],
            "images": [asdict(r) for r in self._records],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

        print(f"  JSON report: {path}")

    def _generate_markdown(self, path: Path) -> None:
        """Generate human-readable Markdown report."""
        lines: list[str] = []
        s = self._summary

        lines.append("# Dataset Quality Audit Report")
        lines.append("")
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")

        # 1. Executive summary
        lines.append("## 1. Executive Summary")
        lines.append("")
        lines.append(f"- **Total images**: {s.total_images}")
        lines.append(f"- **Readable images**: {s.readable_images}")
        lines.append(f"- **Oversized images**: {s.oversized_images}")
        lines.append(f"- **Clean candidates**: {s.clean_candidate_count}")
        lines.append(f"- **Manual review required**: {s.review_required_count}")
        lines.append(f"- **Duplicate groups**: {s.total_duplicate_groups}")
        lines.append(f"- **Near-duplicate images**: {s.total_near_duplicates}")
        lines.append(f"- **Cross-category duplicates**: {s.total_cross_category_duplicates}")
        lines.append("")

        # 2. Dataset distribution
        lines.append("## 2. Dataset Distribution")
        lines.append("")
        lines.append("| Category | Images | % of Total |")
        lines.append("|----------|--------|------------|")
        for cat in sorted(self._category_summaries.keys()):
            cs = self._category_summaries[cat]
            pct = (cs.total_images / s.total_images * 100) if s.total_images > 0 else 0
            lines.append(f"| {cat} | {cs.total_images} | {pct:.1f}% |")
        lines.append(f"| **TOTAL** | **{s.total_images}** | **100%** |")
        lines.append("")

        # 3. Image validity
        lines.append("## 3. Image Validity Statistics")
        lines.append("")
        lines.append(f"- Readable: {s.readable_images}/{s.total_images} ({s.readable_images/s.total_images*100:.1f}%)")
        lines.append(f"- Oversized: {s.oversized_images}")
        lines.append("")

        # 4. Face detection
        lines.append("## 4. Face Detection Statistics")
        lines.append("")
        if s.face_detection_run:
            lines.append(f"- Exactly 1 face: {s.one_face_images}")
            lines.append(f"- Zero faces: {s.zero_face_images}")
            lines.append(f"- Multiple faces: {s.multiple_face_images}")
            if s.face_detection_skipped > 0:
                lines.append(f"- Face detection skipped (oversized/init failure): {s.face_detection_skipped}")
        else:
            lines.append("- Face detection: **NOT RUN**")
        lines.append("")

        # 5. Multiple face details
        if s.face_detection_run:
            lines.append("## 5. Multiple-Face Statistics")
            lines.append("")
            lines.append("| Category | Multiple Faces | Zero Faces | Small Face | Profile Face |")
            lines.append("|----------|---------------|------------|------------|--------------|")
            for cat in sorted(self._category_summaries.keys()):
                cs = self._category_summaries[cat]
                lines.append(f"| {cat} | {cs.multiple_face_images} | {cs.zero_face_images} | {cs.small_face_images} | {cs.profile_face_images} |")
            lines.append("")

            # 6. Small face
            lines.append("## 6. Small-Face Statistics")
            lines.append("")
            lines.append(f"- Total small faces: {s.small_face_images}")
            lines.append("")

            # 7. Profile/pose
            lines.append("## 7. Profile/Pose Statistics")
            lines.append("")
            lines.append(f"- Total profile faces: {s.profile_face_images}")
            lines.append("")
        else:
            lines.append("## 5. Multiple-Face Statistics")
            lines.append("")
            lines.append("- Face detection was not run — statistics unavailable.")
            lines.append("")
            lines.append("## 6. Small-Face Statistics")
            lines.append("")
            lines.append("- Face detection was not run — statistics unavailable.")
            lines.append("")
            lines.append("## 7. Profile/Pose Statistics")
            lines.append("")
            lines.append("- Face detection was not run — statistics unavailable.")
            lines.append("")

        # 8. Near-duplicate analysis
        lines.append("## 8. Near-Duplicate Analysis")
        lines.append("")
        lines.append(f"- Duplicate groups: {s.total_duplicate_groups}")
        lines.append(f"- Near-duplicate images: {s.total_near_duplicates}")
        lines.append(f"- Cross-category duplicates: {s.total_cross_category_duplicates}")
        lines.append("")

        if self._duplicate_groups:
            lines.append("### Duplicate Groups")
            lines.append("")
            for group in self._duplicate_groups[:20]:  # Show first 20
                cross = " [CROSS-CATEGORY]" if group.is_cross_category else ""
                lines.append(f"- **Group {group.group_id}**{cross}: {len(group.image_paths)} images")
                for p in group.image_paths:
                    lines.append(f"  - `{p}`")
            if len(self._duplicate_groups) > 20:
                lines.append(f"- ... and {len(self._duplicate_groups) - 20} more groups")
            lines.append("")

        # 9. Cross-category duplicates
        lines.append("## 9. Cross-Category Duplicate Analysis")
        lines.append("")
        cross_cat_records = [r for r in self._records if r.cross_category_duplicate]
        if cross_cat_records:
            lines.append(f"Found {len(cross_cat_records)} images with cross-category duplicates:")
            lines.append("")
            for rec in cross_cat_records[:20]:
                lines.append(f"- `{rec.filename}` ({rec.category}) -> `{Path(rec.nearest_duplicate_path).name}` ({rec.nearest_duplicate_category}), distance={rec.nearest_duplicate_distance}")
            if len(cross_cat_records) > 20:
                lines.append(f"- ... and {len(cross_cat_records) - 20} more")
        else:
            lines.append("No cross-category duplicates found.")
        lines.append("")

        # 10. Category quality table
        lines.append("## 10. Category-Level Quality Table")
        lines.append("")
        if s.face_detection_run:
            lines.append("| Category | Total | Readable | 1-Face | 0-Face | Multi | Small | Profile | Dupes | Review | Clean |")
            lines.append("|----------|-------|----------|--------|--------|-------|-------|---------|-------|--------|-------|")
            for cat in sorted(self._category_summaries.keys()):
                cs = self._category_summaries[cat]
                lines.append(
                    f"| {cat} | {cs.total_images} | {cs.readable_images} | "
                    f"{cs.one_face_images} | {cs.zero_face_images} | "
                    f"{cs.multiple_face_images} | {cs.small_face_images} | "
                    f"{cs.profile_face_images} | {cs.near_duplicate_count} | "
                    f"{cs.review_required_count} | {cs.clean_candidate_count} |"
                )
        else:
            lines.append("| Category | Total | Readable | Dupes | Review | Clean |")
            lines.append("|----------|-------|----------|-------|--------|-------|")
            for cat in sorted(self._category_summaries.keys()):
                cs = self._category_summaries[cat]
                lines.append(
                    f"| {cat} | {cs.total_images} | {cs.readable_images} | "
                    f"{cs.near_duplicate_count} | "
                    f"{cs.review_required_count} | {cs.clean_candidate_count} |"
                )
        lines.append("")

        # 11. Manual review candidates
        lines.append("## 11. Manual-Review Candidates")
        lines.append("")
        review_candidates = self.get_review_candidates()
        if review_candidates:
            lines.append(f"Found {len(review_candidates)} images requiring manual review:")
            lines.append("")
            lines.append("| Category | Image | Reasons | Action |")
            lines.append("|----------|-------|---------|--------|")
            for rec in review_candidates[:50]:
                reasons_str = "; ".join(rec.review_reasons[:3])
                lines.append(f"| {rec.category} | `{rec.filename}` | {reasons_str} | {rec.recommended_action} |")
            if len(review_candidates) > 50:
                lines.append(f"- ... and {len(review_candidates) - 50} more")
        else:
            lines.append("No manual review candidates found.")
        lines.append("")

        # 12. Oversized images
        lines.append("## 12. Oversized-Image Findings")
        lines.append("")
        oversized = self.get_oversized_images()
        if oversized:
            lines.append(f"Found {len(oversized)} oversized images:")
            lines.append("")
            lines.append("| Category | Image | Dimensions | Pixels | Readable |")
            lines.append("|----------|-------|------------|--------|----------|")
            for rec in oversized:
                lines.append(f"| {rec.category} | `{rec.filename}` | {rec.width}x{rec.height} | {rec.pixel_count:,} | {rec.readable} |")
        else:
            lines.append("No oversized images found.")
        lines.append("")

        # 13. Recommended cleaning actions
        lines.append("## 13. Recommended Cleaning Actions")
        lines.append("")
        action_num = 1
        if s.review_required_count > 0:
            lines.append(f"{action_num}. Review {s.review_required_count} images flagged for manual inspection")
            action_num += 1
        if s.total_near_duplicates > 0:
            lines.append(f"{action_num}. Resolve {s.total_near_duplicates} near-duplicate images ({s.total_duplicate_groups} groups)")
            action_num += 1
        if s.total_cross_category_duplicates > 0:
            lines.append(f"{action_num}. Resolve {s.total_cross_category_duplicates} cross-category duplicates")
            action_num += 1
        if s.oversized_images > 0:
            lines.append(f"{action_num}. Handle {s.oversized_images} oversized images (resize or remove)")
            action_num += 1
        if s.face_detection_run and s.zero_face_images > 0:
            lines.append(f"{action_num}. Verify {s.zero_face_images} zero-face images belong in their categories")
            action_num += 1
        if s.face_detection_run and s.multiple_face_images > 0:
            lines.append(f"{action_num}. Review {s.multiple_face_images} multiple-face images")
            action_num += 1
        if not s.face_detection_run:
            lines.append(f"{action_num}. Run face detection to validate face counts per category")
            action_num += 1
        if not s.quality_scores_run:
            lines.append(f"{action_num}. Run quality scoring to validate blur/brightness/contrast")
            action_num += 1
        lines.append("")

        # 14. Readiness Assessment
        lines.append("## 14. Readiness Assessment")
        lines.append("")

        # Structural readiness (25 pts: readable, 25 pts: no duplicates, 50 pts: no review)
        structural_score = 0
        structural_issues = []
        if s.readable_images == s.total_images and s.total_images > 0:
            structural_score += 25
        else:
            structural_issues.append(f"{s.total_images - s.readable_images} unreadable images")
        if s.total_near_duplicates == 0:
            structural_score += 25
        else:
            structural_issues.append(f"{s.total_near_duplicates} near-duplicate images")
        if s.review_required_count == 0:
            structural_score += 50
        else:
            structural_issues.append(f"{s.review_required_count} images require review")
        lines.append(f"### Structural Readiness: {structural_score}/100")
        if structural_issues:
            for issue in structural_issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- All structural checks passed")
        lines.append("")

        # Face-validation readiness (50 pts: face detection run, 50 pts: all pass)
        face_score = 0
        face_issues = []
        if s.face_detection_run:
            face_score += 50
            if s.zero_face_images == 0:
                face_score += 25
            else:
                face_issues.append(f"{s.zero_face_images} zero-face images")
            if s.multiple_face_images == 0:
                face_score += 25
            else:
                face_issues.append(f"{s.multiple_face_images} multiple-face images")
        else:
            face_issues.append("Face detection was not run")
        lines.append(f"### Face-Validation Readiness: {face_score}/100")
        if face_issues:
            for issue in face_issues:
                lines.append(f"- {issue}")
        else:
            lines.append("- All face-validation checks passed")
        lines.append("")

        # Annotation readiness (requires both structural + face)
        annotation_score = (structural_score + face_score) // 2
        lines.append(f"### Annotation Readiness: {annotation_score}/100")
        if not s.face_detection_run:
            lines.append("- Cannot fully assess without face detection results")
        elif structural_issues or face_issues:
            lines.append("- Issues remain — resolve before annotation")
        else:
            lines.append("- Dataset appears ready for annotation")
        lines.append("")

        # Fine-tuning readiness (requires face + quality)
        quality_ready = s.quality_scores_run
        if quality_ready:
            quality_pass = s.review_required_count == 0
        else:
            quality_pass = False
        finetune_score = annotation_score
        if quality_ready and quality_pass:
            finetune_score = min(100, annotation_score + 20)
        lines.append(f"### Fine-Tuning Readiness: {finetune_score}/100")
        if not s.quality_scores_run:
            lines.append("- Quality scoring was not run — fine-tuning readiness incomplete")
        elif s.review_required_count > 0:
            lines.append("- Images still require review — resolve before fine-tuning")
        elif not s.face_detection_run:
            lines.append("- Face detection required for fine-tuning readiness")
        else:
            lines.append("- Dataset appears ready for fine-tuning")
        lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  Markdown report: {path}")

    def _generate_text_summary(self, path: Path) -> None:
        """Generate plain text summary."""
        lines: list[str] = []
        s = self._summary

        lines.append("=" * 60)
        lines.append("DATASET QUALITY SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Total images:          {s.total_images}")
        lines.append(f"Readable:              {s.readable_images}")
        lines.append(f"Oversized:             {s.oversized_images}")
        lines.append(f"")
        lines.append(f"Face detection:        {'NOT RUN' if not s.face_detection_run else 'RUN'}")
        if s.face_detection_run:
            lines.append(f"  Exactly 1 face:      {s.one_face_images}")
            lines.append(f"  Zero faces:          {s.zero_face_images}")
            lines.append(f"  Multiple faces:      {s.multiple_face_images}")
            lines.append(f"  Small face:          {s.small_face_images}")
            lines.append(f"  Profile face:        {s.profile_face_images}")
        lines.append(f"")
        lines.append(f"Quality scores:        {'NOT RUN' if not s.quality_scores_run else 'RUN'}")
        if s.quality_scores_run:
            lines.append(f"  Clean candidates:    {s.clean_candidate_count}")
            lines.append(f"  Review required:     {s.review_required_count}")
        lines.append(f"")
        lines.append(f"Duplicates:")
        lines.append(f"  Groups:              {s.total_duplicate_groups}")
        lines.append(f"  Near-duplicate imgs: {s.total_near_duplicates}")
        lines.append(f"  Cross-category:      {s.total_cross_category_duplicates}")
        lines.append(f"")
        lines.append("Category breakdown:")
        lines.append("-" * 40)
        for cat in sorted(self._category_summaries.keys()):
            cs = self._category_summaries[cat]
            face_str = f"{cs.one_face_images:3d} 1-face, {cs.zero_face_images:3d} 0-face"
            if not s.face_detection_run:
                face_str = "  NOT RUN"
            lines.append(f"  {cat:20s}: {cs.total_images:4d} imgs, "
                         f"{face_str}, "
                         f"{cs.near_duplicate_count:3d} dupes")
        lines.append("")
        lines.append("=" * 60)

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"  Text summary: {path}")
