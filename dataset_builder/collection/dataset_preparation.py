"""Dataset preparation pipeline for parser fine-tuning.

Extends QualityAudit with:
- Zero-face / multi-face investigation
- Oversized image handling
- Quality scoring
- Final decision layer (KEEP/REVIEW/REMOVE)
- Semantic category flags (deterministic, category-based)
- Dataset manifest generation
- Train/val/test split
- Statistics reporting

READ-ONLY with respect to dataset/raw/ — never modifies source images.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from dataset_builder.config.settings import Settings
from .quality_audit import QualityAudit

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Semantic category mapping
# ------------------------------------------------------------------
# These flags are semantic/category hints derived from the category
# name.  They are NOT proof that the image visually contains the
# condition.  They exist to provide deterministic category-level
# metadata for the annotation pipeline.
#
# Do NOT use these flags as a substitute for annotation.
# Do NOT claim that every image in a category is visually correct.
# ------------------------------------------------------------------

CATEGORY_SEMANTIC_FLAGS: dict[str, dict[str, bool]] = {
    "normal": {
        "is_baseline_normal": True,
        "contains_eyeglass": False,
        "contains_hat": False,
        "contains_facial_occlusion": False,
        "useful_for_eye_improvement": False,
    },
    "eyeglasses": {
        "is_baseline_normal": False,
        "contains_eyeglass": True,
        "contains_hat": False,
        "contains_facial_occlusion": True,
        "useful_for_eye_improvement": True,
    },
    "sunglasses": {
        "is_baseline_normal": False,
        "contains_eyeglass": True,
        "contains_hat": False,
        "contains_facial_occlusion": True,
        "useful_for_eye_improvement": True,
    },
    "cap": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": True,
        "contains_facial_occlusion": False,
        "useful_for_eye_improvement": False,
    },
    "helmet": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": True,
        "contains_facial_occlusion": True,
        "useful_for_eye_improvement": False,
    },
    "mask": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": False,
        "contains_facial_occlusion": True,
        "useful_for_eye_improvement": False,
    },
    "hijab": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": False,
        # Hijab covers hair, not facial features — not facial occlusion.
        "contains_facial_occlusion": False,
        "useful_for_eye_improvement": False,
    },
    "scarf": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": False,
        # Scarf may or may not occlude face; conservatively False
        # until annotation confirms.
        "contains_facial_occlusion": False,
        "useful_for_eye_improvement": False,
    },
    "hair_occlusion": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": False,
        "contains_facial_occlusion": True,
        "useful_for_eye_improvement": False,
    },
    "beard": {
        "is_baseline_normal": False,
        "contains_eyeglass": False,
        "contains_hat": False,
        # Beard is facial hair, not an occlusion in this project's
        # semantic sense (occlusion = external object covering face).
        "contains_facial_occlusion": False,
        "useful_for_eye_improvement": False,
    },
}


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class ImageDecision:
    """Final decision for a single image."""

    path: str
    filename: str
    category: str

    # Structural
    readable: bool = True
    oversized: bool = False
    width: int = 0
    height: int = 0

    # Face detection
    face_detection_evaluated: bool = False
    face_count: int = 0
    largest_face_bbox: list[float] = field(default_factory=list)
    largest_face_area_ratio: float = 0.0
    largest_face_det_score: float = 0.0
    face_yaw: float | None = None
    face_pitch: float | None = None

    # Investigation flags
    zero_face_investigation: str = ""
    """Investigation result for zero-face images."""
    multi_face_investigation: str = ""
    """Investigation result for multi-face images."""
    multi_face_selected: int = 0
    """Which face was selected (0-indexed) for multi-face images."""
    has_small_face: bool = False
    """Whether the face is below MIN_FACE_AREA_RATIO (from audit)."""
    has_profile_face: bool = False
    """Whether the face exceeds MAX_PROFILE_YAW_DEGREES (from audit)."""

    # Quality scores
    quality_evaluated: bool = False
    blur_score: float = -1.0
    brightness_score: float = -1.0
    contrast_score: float = -1.0
    blur_passed: bool | None = None
    brightness_passed: bool | None = None
    contrast_passed: bool | None = None

    # Duplicates
    is_near_duplicate: bool = False
    nearest_duplicate_path: str = ""
    nearest_duplicate_distance: float = -1.0
    cross_category_duplicate: bool = False

    # Fine-tuning relevance (semantic/category hints — see CATEGORY_SEMANTIC_FLAGS)
    useful_for_eye_improvement: bool = False
    contains_eyeglass: bool = False
    contains_hat: bool = False
    contains_facial_occlusion: bool = False
    is_baseline_normal: bool = False

    # Final decision
    decision: str = "PENDING"
    """KEEP, REVIEW, REMOVE, or PENDING."""
    decision_reasons: list[str] = field(default_factory=list)
    """Traceable reasons for the decision."""

    # Split assignment
    split: str = ""
    """train, val, or test.  Empty for REVIEW/REMOVE images."""


@dataclass
class DatasetManifest:
    """Complete dataset manifest with per-image decisions."""

    generated_at: str = ""
    total_images: int = 0
    keep_count: int = 0
    review_count: int = 0
    remove_count: int = 0
    pending_count: int = 0

    # Category breakdown
    category_stats: dict[str, dict[str, int]] = field(default_factory=dict)

    # Split counts
    split_stats: dict[str, int] = field(default_factory=dict)

    # All decisions
    images: list[ImageDecision] = field(default_factory=list)


@dataclass
class InvestigationReport:
    """Investigation report for zero-face / multi-face images."""

    report_type: str = ""
    total_investigated: int = 0
    findings: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Dataset preparation engine
# ------------------------------------------------------------------

class DatasetPreparation:
    """Full dataset preparation pipeline.

    Orchestrates:
    1. Structural audit (via QualityAudit)
    2. Face detection
    3. Semantic flag population
    4. Zero-face investigation
    5. Multi-face investigation
    6. Oversized handling
    7. Final decisions
    8. Split assignment
    9. Manifest generation (AFTER split)
    10. Report generation
    11. Statistics
    """

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        *,
        skip_face_detection: bool = False,
        skip_quality_scores: bool = False,
        settings: Settings | None = None,
    ):
        self._raw_dir = raw_dir
        self._output_dir = output_dir
        self._skip_face_detection = skip_face_detection
        self._skip_quality_scores = skip_quality_scores

        self._settings: Settings = settings or Settings(
            DATASET_DIR=output_dir,
            RAW_IMAGES_DIR=raw_dir,
            REPORTS_DIR=output_dir / "reports",
        )
        self._audit: QualityAudit | None = None
        self._decisions: list[ImageDecision] = []
        self._manifest: DatasetManifest | None = None
        self._zero_face_report: InvestigationReport | None = None
        self._multi_face_report: InvestigationReport | None = None
        self._detector = None
        """Reusable FaceDetector instance (lazy-initialized)."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> DatasetManifest:
        """Run the full preparation pipeline."""
        print("=" * 60)
        print("DATASET PREPARATION PIPELINE")
        print("=" * 60)

        # Step 1: Run quality audit
        print("\n--- Step 1: Quality Audit ---")
        if self._settings is None:
            self._settings = Settings(
                DATASET_DIR=self._output_dir,
                RAW_IMAGES_DIR=self._raw_dir,
                REPORTS_DIR=self._output_dir / "reports",
            )
        self._audit = QualityAudit(self._settings)
        self._audit.run(
            raw_dir=self._raw_dir,
            skip_face_detection=self._skip_face_detection,
            skip_quality_scores=self._skip_quality_scores,
        )

        # Step 2: Build initial decisions from audit
        print("\n--- Step 2: Building Initial Decisions ---")
        self._build_decisions_from_audit()

        # Step 3: Populate semantic category flags
        print("\n--- Step 3: Populating Semantic Category Flags ---")
        self._populate_semantic_flags()

        # Step 4: Investigate zero-face images
        if not self._skip_face_detection:
            print("\n--- Step 4: Investigating Zero-Face Images ---")
            self._investigate_zero_face()

            print("\n--- Step 5: Investigating Multi-Face Images ---")
            self._investigate_multi_face()

        # Step 5: Handle oversized images
        print("\n--- Step 6: Handling Oversized Images ---")
        self._handle_oversized()

        # Step 6: Final decisions
        print("\n--- Step 7: Making Final Decisions ---")
        self._make_final_decisions()

        # Step 7: Split dataset
        print("\n--- Step 8: Splitting Dataset ---")
        self._split_dataset()

        # Step 8: Build and persist manifest (AFTER split)
        print("\n--- Step 9: Generating Manifest ---")
        self._manifest = self._build_manifest()

        # Step 9: Generate reports
        print("\n--- Step 10: Generating Reports ---")
        self._generate_reports()

        # Step 10: Statistics
        print("\n--- Step 11: Final Statistics ---")
        self._print_statistics()

        print("\n" + "=" * 60)
        print("PREPARATION COMPLETE")
        print("=" * 60)

        return self._manifest

    def _get_detector(self):
        """Return a shared FaceDetector instance, initializing once."""
        if self._detector is None:
            from pipeline.detector import FaceDetector
            self._detector = FaceDetector()
        return self._detector

    # ------------------------------------------------------------------
    # Step 2: Build decisions from audit
    # ------------------------------------------------------------------

    def _build_decisions_from_audit(self) -> None:
        """Convert audit records into ImageDecision objects."""
        if self._audit is None:
            return

        for rec in self._audit.get_records():
            if not rec.readable:
                decision = ImageDecision(
                    path=rec.path,
                    filename=rec.filename,
                    category=rec.category,
                    readable=False,
                    decision="REMOVE",
                    decision_reasons=["Unreadable/corrupt image"],
                )
            else:
                decision = ImageDecision(
                    path=rec.path,
                    filename=rec.filename,
                    category=rec.category,
                    readable=True,
                    oversized=rec.oversized,
                    width=rec.width,
                    height=rec.height,
                    # Face detection
                    face_detection_evaluated=rec.face_detection_evaluated,
                    face_count=rec.face_count,
                    largest_face_bbox=rec.largest_face_bbox or [],
                    largest_face_area_ratio=rec.largest_face_area_ratio,
                    largest_face_det_score=rec.largest_face_det_score,
                    face_yaw=rec.face_yaw,
                    face_pitch=rec.face_pitch,
                    # Investigation
                    zero_face_investigation="",
                    multi_face_investigation="",
                    multi_face_selected=0,
                    has_small_face=rec.has_small_face,
                    has_profile_face=rec.has_profile_face,
                    # Quality scores
                    quality_evaluated=rec.quality_evaluated,
                    blur_score=rec.blur_score,
                    brightness_score=rec.brightness_score,
                    contrast_score=rec.contrast_score,
                    blur_passed=rec.blur_passed,
                    brightness_passed=rec.brightness_passed,
                    contrast_passed=rec.contrast_passed,
                    # Duplicates
                    is_near_duplicate=rec.is_near_duplicate,
                    nearest_duplicate_path=rec.nearest_duplicate_path,
                    nearest_duplicate_distance=rec.nearest_duplicate_distance,
                    cross_category_duplicate=rec.cross_category_duplicate,
                )
                decision.decision = "PENDING"

            self._decisions.append(decision)

        print(f"  Built {len(self._decisions)} initial decisions.")

    # ------------------------------------------------------------------
    # Step 3: Populate semantic category flags
    # ------------------------------------------------------------------

    def _populate_semantic_flags(self) -> None:
        """Set per-image semantic flags deterministically from the category name.

        These are category-level hints, NOT visual proof.
        See CATEGORY_SEMANTIC_FLAGS for the mapping.
        """
        count = 0
        for dec in self._decisions:
            flags = CATEGORY_SEMANTIC_FLAGS.get(dec.category, {})
            if flags:
                dec.is_baseline_normal = flags.get("is_baseline_normal", False)
                dec.contains_eyeglass = flags.get("contains_eyeglass", False)
                dec.contains_hat = flags.get("contains_hat", False)
                dec.contains_facial_occlusion = flags.get("contains_facial_occlusion", False)
                dec.useful_for_eye_improvement = flags.get("useful_for_eye_improvement", False)
                count += 1
        print(f"  Populated semantic flags for {count} images.")

    # ------------------------------------------------------------------
    # Step 4: Investigate zero-face images
    # ------------------------------------------------------------------

    def _investigate_zero_face(self) -> None:
        """Investigate zero-face images with multiple detection strategies.

        Uses a single shared FaceDetector instance (reused across all images
        and scales) to avoid reloading InsightFace/ONNX models per image.
        """
        if self._audit is None:
            return

        zero_face = [d for d in self._decisions
                     if d.readable and not d.oversized and d.face_detection_evaluated
                     and d.face_count == 0]

        if not zero_face:
            print("  No zero-face images to investigate.")
            return

        print(f"  Investigating {len(zero_face)} zero-face images...")

        detector = self._get_detector()
        scales = [0.5, 0.75, 1.5, 2.0]
        results = []

        for dec in zero_face:
            img = cv2.imread(dec.path)
            if img is None:
                dec.zero_face_investigation = "unreadable"
                continue

            h, w = img.shape[:2]
            found_any = False

            for scale in scales:
                try:
                    if scale != 1.0:
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        if new_w < 10 or new_h < 10:
                            continue
                        resized = cv2.resize(img, (new_w, new_h))
                    else:
                        resized = img

                    faces = detector.detect(resized)
                    if faces:
                        dec.face_count = len(faces)
                        dec.face_detection_evaluated = True
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
                            # Scale back to original coordinates
                            x1, y1, x2, y2 = x1/scale, y1/scale, x2/scale, y2/scale
                            dec.largest_face_bbox = [float(x1), float(y1), float(x2), float(y2)]
                            face_area = (x2 - x1) * (y2 - y1)
                            img_area = w * h
                            if img_area > 0:
                                dec.largest_face_area_ratio = face_area / img_area
                        if hasattr(largest, "det_score"):
                            dec.largest_face_det_score = float(largest.det_score)
                        if hasattr(largest, "pose") and largest.pose is not None:
                            try:
                                pitch, yaw, roll = largest.pose
                                dec.face_pitch = float(pitch)
                                dec.face_yaw = float(yaw)
                            except (TypeError, ValueError):
                                pass

                        found_any = True
                        dec.zero_face_investigation = f"detected_at_scale_{scale}"
                        break
                except Exception:
                    continue

            if not found_any:
                # Classify why no face was found
                if dec.contains_eyeglass:
                    dec.zero_face_investigation = "eyeglass_occlusion"
                elif dec.contains_facial_occlusion:
                    dec.zero_face_investigation = "facial_occlusion"
                elif dec.contains_hat:
                    dec.zero_face_investigation = "hat_occlusion"
                else:
                    dec.zero_face_investigation = "no_detection"

            results.append({
                "filename": dec.filename,
                "category": dec.category,
                "investigation": dec.zero_face_investigation,
                "final_face_count": dec.face_count,
            })

        # Build report
        self._zero_face_report = InvestigationReport(
            report_type="zero_face_investigation",
            total_investigated=len(zero_face),
            findings=results,
        )

        # Count outcomes
        outcomes: dict[str, int] = {}
        for r in results:
            inv = r["investigation"]
            outcomes[inv] = outcomes.get(inv, 0) + 1

        print("  Zero-face investigation results:")
        for inv, count in sorted(outcomes.items()):
            print(f"    {inv}: {count}")

    # ------------------------------------------------------------------
    # Step 5: Investigate multi-face images
    # ------------------------------------------------------------------

    def _investigate_multi_face(self) -> None:
        """Investigate multi-face images — try to select the best face.

        Uses a single shared FaceDetector instance (reused across all images)
        to avoid reloading InsightFace/ONNX models per image.
        """
        if self._audit is None:
            return

        multi_face = [d for d in self._decisions
                      if d.readable and not d.oversized and d.face_detection_evaluated
                      and d.face_count > 1]

        if not multi_face:
            print("  No multi-face images to investigate.")
            return

        print(f"  Investigating {len(multi_face)} multi-face images...")

        detector = self._get_detector()
        results = []

        for dec in multi_face:
            img = cv2.imread(dec.path)
            if img is None:
                dec.multi_face_investigation = "unreadable"
                continue

            try:
                from pipeline.selector import FaceSelector

                faces = detector.detect(img)

                if len(faces) <= 1:
                    # Re-detection gave 0 or 1 face — might be intermittent
                    dec.multi_face_investigation = f"re_detected_{len(faces)}_faces"
                    if len(faces) == 1:
                        dec.face_count = 1
                    continue

                selector = FaceSelector()
                selection = selector.select(faces, img.shape)

                if selection and selection.selected_face is not None:
                    selected_idx = selection.selected_index if hasattr(selection, "selected_index") else 0
                    dec.multi_face_selected = selected_idx
                    dec.multi_face_investigation = f"selected_face_{selected_idx}_of_{len(faces)}"

                    # Update face info for the selected face
                    selected = selection.selected_face
                    if hasattr(selected, "bbox"):
                        x1, y1, x2, y2 = selected.bbox[:4]
                        dec.largest_face_bbox = [float(x1), float(y1), float(x2), float(y2)]
                        face_area = (x2 - x1) * (y2 - y1)
                        img_area = dec.width * dec.height
                        if img_area > 0:
                            dec.largest_face_area_ratio = face_area / img_area
                    if hasattr(selected, "det_score"):
                        dec.largest_face_det_score = float(selected.det_score)
                    if hasattr(selected, "pose") and selected.pose is not None:
                        try:
                            pitch, yaw, roll = selected.pose
                            dec.face_pitch = float(pitch)
                            dec.face_yaw = float(yaw)
                        except (TypeError, ValueError):
                            pass
                else:
                    dec.multi_face_investigation = "selection_failed"

            except Exception as e:
                dec.multi_face_investigation = f"error: {e}"

            results.append({
                "filename": dec.filename,
                "category": dec.category,
                "investigation": dec.multi_face_investigation,
                "original_face_count": dec.face_count,
                "selected_index": dec.multi_face_selected,
            })

        # Build report
        self._multi_face_report = InvestigationReport(
            report_type="multi_face_investigation",
            total_investigated=len(multi_face),
            findings=results,
        )

        outcomes: dict[str, int] = {}
        for r in results:
            inv = r["investigation"]
            outcomes[inv] = outcomes.get(inv, 0) + 1

        print("  Multi-face investigation results:")
        for inv, count in sorted(outcomes.items()):
            print(f"    {inv}: {count}")

    # ------------------------------------------------------------------
    # Step 6: Handle oversized images
    # ------------------------------------------------------------------

    def _handle_oversized(self) -> None:
        """Flag oversized images and note they need preprocessing."""
        oversized = [d for d in self._decisions if d.oversized]

        if not oversized:
            print("  No oversized images.")
            return

        print(f"  Found {len(oversized)} oversized image(s):")
        for dec in oversized:
            print(f"    {dec.filename} ({dec.width}x{dec.height})")
            dec.decision_reasons.append(
                f"Oversized image ({dec.width}x{dec.height}) — needs preprocessing before use"
            )

    # ------------------------------------------------------------------
    # Step 7: Final decisions
    # ------------------------------------------------------------------

    def _make_final_decisions(self) -> None:
        """Apply decision rules to all images.

        Decision policy:
        - Unreadable → REMOVE
        - No face after investigation → REMOVE (or REVIEW if occlusion-related)
        - Multiple faces → always REVIEW
        - Small face → REVIEW
        - Extreme profile → REVIEW
        - Oversized → REVIEW
        - Near-duplicate → REVIEW
        - Quality failure → REVIEW
        - Exactly one face, no blocking issues → KEEP
        """
        for dec in self._decisions:
            reasons = list(dec.decision_reasons)
            decision = "PENDING"

            if not dec.readable:
                decision = "REMOVE"
                reasons.append("Unreadable/corrupt image")

            elif dec.oversized:
                decision = "REVIEW"
                reasons.append("Oversized image — needs preprocessing")

            elif dec.face_detection_evaluated:
                if dec.face_count == 0:
                    if dec.zero_face_investigation.startswith("detected_at_scale_"):
                        # Face found with alternate scale — treat as single-face
                        decision = "KEEP"
                        reasons.append(f"Face found with alternate detection: {dec.zero_face_investigation}")
                    elif dec.zero_face_investigation in (
                        "eyeglass_occlusion", "facial_occlusion", "hat_occlusion",
                    ):
                        decision = "REVIEW"
                        reasons.append(f"Zero face due to occlusion: {dec.zero_face_investigation}")
                    else:
                        decision = "REMOVE"
                        reasons.append("No face detected after investigation")

                elif dec.face_count > 1:
                    # Multiple faces always require REVIEW
                    if dec.multi_face_investigation.startswith("selected_face_"):
                        decision = "REVIEW"
                        reasons.append(
                            "Multiple faces detected; primary face selected, "
                            "but manual review required before annotation."
                        )
                    elif dec.multi_face_investigation == "selection_failed":
                        decision = "REVIEW"
                        reasons.append("Multi-face — face selection failed")
                    elif dec.multi_face_investigation.startswith("re_detected_"):
                        decision = "REVIEW"
                        reasons.append(f"Multi-face — re-detected {dec.face_count} faces")
                    else:
                        decision = "REVIEW"
                        reasons.append(f"Multi-face — needs review: {dec.multi_face_investigation}")

                else:
                    # Exactly 1 face — check for re-detection from multi-face first
                    if dec.multi_face_investigation.startswith("re_detected_"):
                        # Originally multi-face, re-detected as single — still REVIEW
                        decision = "REVIEW"
                        reasons.append(
                            "Originally detected as multi-face; re-detection found one face, "
                            "manual review required before annotation."
                        )
                    elif dec.has_small_face:
                        decision = "REVIEW"
                        reasons.append(
                            f"Face too small (area ratio={dec.largest_face_area_ratio:.4f}, "
                            f"threshold={self._settings.MIN_FACE_AREA_RATIO})"
                        )
                    elif dec.has_profile_face:
                        decision = "REVIEW"
                        reasons.append(
                            f"Extreme profile face (yaw={dec.face_yaw:.1f}°, "
                            f"threshold={self._settings.MAX_PROFILE_YAW_DEGREES}°)"
                        )
                    else:
                        decision = "KEEP"
                        reasons.append("Single face detected")

            else:
                # Face detection was not evaluated
                decision = "REVIEW"
                reasons.append("Face detection not evaluated")

            # Duplicate check
            if dec.is_near_duplicate and decision == "KEEP":
                decision = "REVIEW"
                reasons.append(f"Near-duplicate of {Path(dec.nearest_duplicate_path).name}")

            # Quality check
            if decision == "KEEP" and dec.quality_evaluated:
                quality_failures = []
                if dec.blur_passed is False:
                    quality_failures.append("blur")
                if dec.brightness_passed is False:
                    quality_failures.append("brightness")
                if dec.contrast_passed is False:
                    quality_failures.append("contrast")
                if quality_failures:
                    decision = "REVIEW"
                    reasons.append(f"Quality failures: {', '.join(quality_failures)}")

            dec.decision = decision
            dec.decision_reasons = reasons

    # ------------------------------------------------------------------
    # Step 8: Split dataset
    # ------------------------------------------------------------------

    def _split_dataset(self) -> None:
        """Assign train/val/test splits to KEEP images only.

        Deterministic: seed=42, per-category splitting, no data leakage.
        REVIEW/REMOVE images receive split="".
        """
        keep_images = [d for d in self._decisions if d.decision == "KEEP"]

        if not keep_images:
            print("  No KEEP images to split.")
            return

        # Group by category
        by_category: dict[str, list[ImageDecision]] = {}
        for dec in keep_images:
            by_category.setdefault(dec.category, []).append(dec)

        # Deterministic split with fixed seed
        rng = np.random.RandomState(42)

        train_ratio = 0.70
        val_ratio = 0.15
        test_ratio = 0.15

        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

        for cat, images in by_category.items():
            n = len(images)

            if n == 0:
                continue

            indices = list(range(n))
            rng.shuffle(indices)

            if n >= 3:
                n_train = int(n * train_ratio)
                n_val = int(n * val_ratio)
                n_test = n - n_train - n_val
                # Guard against rounding issues
                if n_test < 0:
                    n_train += n_test
                    n_test = 0
            elif n == 1:
                n_train, n_val, n_test = 1, 0, 0
            else:  # n == 2
                n_train, n_val, n_test = 1, 1, 0

            for i, idx in enumerate(indices):
                if i < n_train:
                    images[idx].split = "train"
                    split_counts["train"] += 1
                elif i < n_train + n_val:
                    images[idx].split = "val"
                    split_counts["val"] += 1
                else:
                    images[idx].split = "test"
                    split_counts["test"] += 1

        # Ensure REVIEW/REMOVE images have no split
        for dec in self._decisions:
            if dec.decision != "KEEP":
                dec.split = ""

        # Write split files
        splits_dir = self._output_dir / "splits"
        splits_dir.mkdir(parents=True, exist_ok=True)

        for split_name in ["train", "val", "test"]:
            split_images = [d for d in self._decisions if d.split == split_name]
            split_path = splits_dir / f"{split_name}.txt"
            with open(split_path, "w", encoding="utf-8") as f:
                for img in split_images:
                    rel_path = Path(img.path).relative_to(self._raw_dir.parent)
                    f.write(f"{rel_path}\n")

            print(f"  {split_name}: {len(split_images)} images -> {split_path}")

    # ------------------------------------------------------------------
    # Step 9: Build manifest (AFTER split)
    # ------------------------------------------------------------------

    def _build_manifest(self) -> DatasetManifest:
        """Build the dataset manifest.

        Called AFTER _split_dataset so that split assignments are included.
        """
        manifest = DatasetManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            total_images=len(self._decisions),
        )

        # Count decisions
        for dec in self._decisions:
            if dec.decision == "KEEP":
                manifest.keep_count += 1
            elif dec.decision == "REVIEW":
                manifest.review_count += 1
            elif dec.decision == "REMOVE":
                manifest.remove_count += 1
            else:
                manifest.pending_count += 1

            # Category stats
            cat = dec.category
            if cat not in manifest.category_stats:
                manifest.category_stats[cat] = {
                    "total": 0, "keep": 0, "review": 0, "remove": 0,
                }
            manifest.category_stats[cat]["total"] += 1
            if dec.decision == "KEEP":
                manifest.category_stats[cat]["keep"] += 1
            elif dec.decision == "REVIEW":
                manifest.category_stats[cat]["review"] += 1
            elif dec.decision == "REMOVE":
                manifest.category_stats[cat]["remove"] += 1

        manifest.images = self._decisions

        # Compute split_stats from actual assignments
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for dec in self._decisions:
            if dec.split in split_counts:
                split_counts[dec.split] += 1
        manifest.split_stats = split_counts

        # Save manifest
        manifest_path = self._output_dir / "dataset_manifest.json"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2, ensure_ascii=False, default=str)
        print(f"  Manifest saved: {manifest_path}")

        return manifest

    # ------------------------------------------------------------------
    # Step 10: Generate reports
    # ------------------------------------------------------------------

    def _generate_reports(self) -> None:
        """Generate all output reports."""
        reports_dir = self._output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Summary text report
        summary_path = reports_dir / "dataset_preparation_summary.txt"
        self._generate_summary_text(summary_path)

        # Investigation reports
        if self._zero_face_report:
            zero_path = reports_dir / "zero_face_investigation.json"
            with open(zero_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self._zero_face_report), f, indent=2, default=str)
            print(f"  Zero-face report: {zero_path}")

        if self._multi_face_report:
            multi_path = reports_dir / "multi_face_investigation.json"
            with open(multi_path, "w", encoding="utf-8") as f:
                json.dump(asdict(self._multi_face_report), f, indent=2, default=str)
            print(f"  Multi-face report: {multi_path}")

        # Decision breakdown
        decisions_path = reports_dir / "decision_breakdown.json"
        breakdown: dict[str, list[dict]] = {
            "keep": [],
            "review": [],
            "remove": [],
        }
        for dec in self._decisions:
            entry = {
                "filename": dec.filename,
                "category": dec.category,
                "decision": dec.decision,
                "reasons": dec.decision_reasons,
            }
            if dec.decision in breakdown:
                breakdown[dec.decision].append(entry)

        with open(decisions_path, "w", encoding="utf-8") as f:
            json.dump(breakdown, f, indent=2, ensure_ascii=False)
        print(f"  Decision breakdown: {decisions_path}")

    def _generate_summary_text(self, path: Path) -> None:
        """Generate a plain-text summary report."""
        lines = []
        lines.append("DATASET PREPARATION SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")

        m = self._manifest
        if m:
            lines.append(f"Total images:   {m.total_images}")
            lines.append(f"KEEP:           {m.keep_count}")
            lines.append(f"REVIEW:         {m.review_count}")
            lines.append(f"REMOVE:         {m.remove_count}")
            lines.append(f"PENDING:        {m.pending_count}")
            lines.append("")

            lines.append("Category Breakdown:")
            lines.append(f"{'Category':<20} {'Total':>6} {'Keep':>6} {'Review':>6} {'Remove':>6}")
            lines.append("-" * 50)
            for cat in sorted(m.category_stats.keys()):
                cs = m.category_stats[cat]
                lines.append(
                    f"{cat:<20} {cs['total']:>6} {cs['keep']:>6} "
                    f"{cs['review']:>6} {cs['remove']:>6}"
                )
            lines.append("")

            if m.split_stats:
                lines.append("Split Statistics:")
                for split_name, count in sorted(m.split_stats.items()):
                    lines.append(f"  {split_name}: {count}")
                lines.append("")

        # Investigation summary
        if self._zero_face_report:
            lines.append("Zero-Face Investigation:")
            lines.append(f"  Investigated: {self._zero_face_report.total_investigated}")
            outcomes: dict[str, int] = {}
            for finding in self._zero_face_report.findings:
                inv = finding["investigation"]
                outcomes[inv] = outcomes.get(inv, 0) + 1
            for inv, count in sorted(outcomes.items()):
                lines.append(f"    {inv}: {count}")
            lines.append("")

        if self._multi_face_report:
            lines.append("Multi-Face Investigation:")
            lines.append(f"  Investigated: {self._multi_face_report.total_investigated}")
            outcomes2: dict[str, int] = {}
            for finding in self._multi_face_report.findings:
                inv = finding["investigation"]
                outcomes2[inv] = outcomes2.get(inv, 0) + 1
            for inv, count in sorted(outcomes2.items()):
                lines.append(f"    {inv}: {count}")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  Summary report: {path}")

    # ------------------------------------------------------------------
    # Step 11: Statistics
    # ------------------------------------------------------------------

    def _print_statistics(self) -> None:
        """Print final statistics."""
        if not self._manifest:
            return

        m = self._manifest
        print(f"\n  Total images:   {m.total_images}")
        print(f"  KEEP:           {m.keep_count}")
        print(f"  REVIEW:         {m.review_count}")
        print(f"  REMOVE:         {m.remove_count}")
        print(f"  PENDING:        {m.pending_count}")

        if m.split_stats:
            print("\n  Split counts:")
            for split_name, count in sorted(m.split_stats.items()):
                print(f"    {split_name}: {count}")

        print(f"\n  Reports saved to: {self._output_dir / 'reports'}")
