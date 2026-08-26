"""Current dataset annotation preparation adapter.

Reads the authoritative dataset manifest (dataset_manifest.json),
filters KEEP images, runs face alignment + BiSeNet mask generation,
and produces the annotation-ready dataset under parser_finetune_current/.

Usage:
    python scripts/prepare_annotation.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_builder.dataset.parser_finetune.face_pipeline import align_face
from dataset_builder.dataset.parser_finetune.mask_generation import (
    generate_mask,
    save_mask,
)
from dataset_builder.dataset.parser_finetune.metadata import (
    SampleMetadata,
    save_class_mapping,
    save_sample_metadata,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune"
    / "dataset_manifest.json"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
)


# ── Data classes ─────────────────────────────────────────────────────


@dataclass
class PreparationResult:
    sample_id: str
    source_path: str
    source_category: str
    filename: str
    split: str
    aligned_path: str = ""
    initial_mask_path: str = ""
    mask_path: str = ""
    metadata_path: str = ""
    aligned: bool = False
    masked: bool = False
    error: str = ""


@dataclass
class PreparationReport:
    generated_at: str = ""
    total_keep: int = 0
    prepared_successfully: int = 0
    alignment_failures: int = 0
    mask_generation_failures: int = 0
    missing_outputs: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    category_distribution: dict[str, int] = field(default_factory=dict)
    errors: list[dict[str, str]] = field(default_factory=list)
    manifest_validation: str = "NOT_RUN"
    split_validation: str = "NOT_RUN"
    metadata_validation: str = "NOT_RUN"
    class_mapping_validation: str = "NOT_RUN"
    artifact_validation: str = "NOT_RUN"
    final_status: str = "NOT_READY_FOR_ANNOTATION"
    validation_issues: list[str] = field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────────


def _load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def _keep_images(manifest: dict) -> list[dict]:
    """Return sorted KEEP images from the manifest."""
    keep = [img for img in manifest["images"] if img["decision"] == "KEEP"]
    keep.sort(key=lambda img: (img["category"], img["filename"]))
    return keep


def _assign_sample_ids(keep_images: list[dict]) -> dict[str, str]:
    """Build filename→sample_id mapping (deterministic)."""
    mapping: dict[str, str] = {}
    for i, img in enumerate(keep_images):
        key = f"{img['category']}/{img['filename']}"
        mapping[key] = f"sample_{i:04d}"
    return mapping


def _ensure_dirs(output_dir: Path) -> None:
    for sub in [
        "images",
        "initial_masks",
        "masks",
        "metadata",
        "face_data",
        "splits",
        "annotation/metadata",
        "annotation/corrected_masks",
        "annotation/qa_reports",
        "annotation/overlays",
        "reports",
    ]:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def _check_artifacts(sample_id: str, output_dir: Path) -> dict[str, bool]:
    """Check which artifacts exist for a sample."""
    return {
        "aligned": (output_dir / "images" / f"{sample_id}.png").exists(),
        "initial_mask": (output_dir / "initial_masks" / f"{sample_id}.png").exists(),
        "mask": (output_dir / "masks" / f"{sample_id}.png").exists(),
        "metadata": (output_dir / "metadata" / f"{sample_id}.json").exists(),
        "face_data": (output_dir / "face_data" / f"{sample_id}.json").exists(),
    }


# ── Core processing ──────────────────────────────────────────────────


def _process_image(
    img_entry: dict,
    sample_id: str,
    output_dir: Path,
    skip_existing: bool = True,
) -> PreparationResult:
    """Process a single KEEP image: align → mask → metadata.

    Handles partial processing for idempotent reruns:
    - Skips any artifact that already exists
    - Regenerates only missing artifacts
    - Uses face_data sidecar to recover metadata without re-running alignment
    """
    source_path = img_entry["path"]
    category = img_entry["category"]
    filename = img_entry["filename"]
    split = img_entry.get("split", "")

    result = PreparationResult(
        sample_id=sample_id,
        source_path=source_path,
        source_category=category,
        filename=filename,
        split=split,
    )

    aligned_path = output_dir / "images" / f"{sample_id}.png"
    initial_mask_path = output_dir / "initial_masks" / f"{sample_id}.png"
    mask_path = output_dir / "masks" / f"{sample_id}.png"

    result.aligned_path = str(aligned_path)
    result.initial_mask_path = str(initial_mask_path)
    result.mask_path = str(mask_path)

    artifacts = _check_artifacts(sample_id, output_dir)

    # All artifacts exist → skip completely
    if skip_existing and all(artifacts.values()):
        result.aligned = True
        result.masked = True
        result.metadata_path = str(output_dir / "metadata" / f"{sample_id}.json")
        return result

    # ── Step 1: Face alignment ──────────────────────────────────────
    alignment = None
    if artifacts["aligned"]:
        result.aligned = True
        # Load face data from sidecar for metadata generation
        face_data_path = output_dir / "face_data" / f"{sample_id}.json"
        if face_data_path.exists():
            with open(face_data_path) as f:
                fd = json.load(f)

            class _FaceData:
                pass

            alignment = _FaceData()
            alignment.face_bbox = fd["face_bbox"]
            alignment.face_kps = fd["face_kps"]
            alignment.detection_score = fd["detection_score"]
            alignment.face_area_ratio = fd["face_area_ratio"]
            alignment.aligned_image = cv2.imread(str(aligned_path))
    else:
        alignment = align_face(Path(source_path), sample_id, category)
        if alignment is None:
            result.error = "alignment_failed"
            return result

        result.aligned = True
        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(aligned_path), alignment.aligned_image)

        # Save face data sidecar for idempotent metadata recovery
        face_data = {
            "face_bbox": alignment.face_bbox,
            "face_kps": alignment.face_kps,
            "detection_score": alignment.detection_score,
            "face_area_ratio": alignment.face_area_ratio,
        }
        face_data_dir = output_dir / "face_data"
        face_data_dir.mkdir(parents=True, exist_ok=True)
        with open(face_data_dir / f"{sample_id}.json", "w") as f:
            json.dump(face_data, f)

    # ── Step 2: BiSeNet mask generation ─────────────────────────────
    if artifacts["initial_mask"]:
        result.masked = True
    else:
        # Use alignment.aligned_image if available, else read from disk
        aligned_image = None
        if alignment is not None and hasattr(alignment, "aligned_image"):
            aligned_image = alignment.aligned_image
        elif artifacts["aligned"]:
            aligned_image = cv2.imread(str(aligned_path))

        if aligned_image is None:
            result.error = "alignment_failed"
            return result

        mask = generate_mask(aligned_image)
        if mask is None:
            result.error = "mask_generation_failed"
            return result

        result.masked = True
        save_mask(mask, initial_mask_path)
        save_mask(mask, mask_path)

    # ── Step 3: Metadata ────────────────────────────────────────────
    if artifacts["metadata"]:
        result.metadata_path = str(output_dir / "metadata" / f"{sample_id}.json")
        return result

    if alignment is None:
        result.error = "metadata_failed_no_face_data"
        return result

    metadata = SampleMetadata(
        sample_id=sample_id,
        source_image=source_path,
        source_category=category,
        aligned_image=str(aligned_path),
        initial_mask=str(initial_mask_path),
        ground_truth_mask=str(mask_path),
        face_bbox=alignment.face_bbox,
        face_kps=alignment.face_kps,
        image_width=alignment.aligned_image.shape[1],
        image_height=alignment.aligned_image.shape[0],
        detection_score=alignment.detection_score,
        face_area_ratio=alignment.face_area_ratio,
        selection_reason="quality_pass",
        quality_status="annotation_pending",
        split=split,
    )
    save_sample_metadata(metadata, output_dir / "metadata")
    result.metadata_path = str(output_dir / "metadata" / f"{sample_id}.json")

    return result


# ── Manifest and splits ──────────────────────────────────────────────


def _build_manifest(
    results: list[PreparationResult],
    output_dir: Path,
    manifest: dict,
) -> dict:
    """Build the annotation-ready manifest with required fields."""
    samples = []
    category_dist: dict[str, int] = {}
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for r in results:
        if not r.aligned or not r.masked:
            continue
        samples.append({
            "sample_id": r.sample_id,
            "source_category": r.source_category,
            "source_filename": r.filename,
            "aligned_image": str(Path(r.aligned_path).resolve()),
            "initial_mask": str(Path(r.initial_mask_path).resolve()),
            "ground_truth_mask": str(Path(r.mask_path).resolve()),
            "split": r.split,
            "annotation_status": "annotation_pending",
            "ground_truth_status": "initial_model_mask_pending_correction",
        })
        category_dist[r.source_category] = (
            category_dist.get(r.source_category, 0) + 1
        )
        if r.split in split_counts:
            split_counts[r.split] += 1

    annotation_manifest = {
        "version": "current_v1",
        "annotation_status": "annotation_pending",
        "total_samples": len(samples),
        "splits": split_counts,
        "category_distribution": category_dist,
        "samples": samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    manifest_path = output_dir / "annotation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(annotation_manifest, f, indent=2, ensure_ascii=False)

    return annotation_manifest


def _write_split_files(
    results: list[PreparationResult],
    output_dir: Path,
) -> dict[str, list[str]]:
    """Write train/val/test split files with sample IDs."""
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for r in results:
        if not r.aligned or not r.masked:
            continue
        if r.split in splits:
            splits[r.split].append(r.sample_id)

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name, ids in splits.items():
        split_path = splits_dir / f"{split_name}.txt"
        with open(split_path, "w", encoding="utf-8") as f:
            for sid in sorted(ids):
                f.write(sid + "\n")

    return splits


# ── Validation ───────────────────────────────────────────────────────


def _validate_artifacts(
    results: list[PreparationResult], output_dir: Path
) -> tuple[str, list[str]]:
    """Validate aligned images, masks for every successful sample."""
    successful = [r for r in results if r.aligned and r.masked]
    issues: list[str] = []

    for r in successful:
        img_path = Path(r.aligned_path)
        if not img_path.exists():
            issues.append(f"{r.sample_id}: aligned image missing")
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            issues.append(f"{r.sample_id}: aligned image unreadable")
        elif img.shape != (112, 112, 3):
            issues.append(
                f"{r.sample_id}: aligned image shape {img.shape}, expected (112, 112, 3)"
            )
        elif img.dtype != np.uint8:
            issues.append(
                f"{r.sample_id}: aligned image dtype {img.dtype}, expected uint8"
            )

    for r in successful:
        mask_path = Path(r.initial_mask_path)
        if not mask_path.exists():
            issues.append(f"{r.sample_id}: initial mask missing")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            issues.append(f"{r.sample_id}: initial mask unreadable")
        elif mask.shape != (112, 112):
            issues.append(
                f"{r.sample_id}: mask shape {mask.shape}, expected (112, 112)"
            )
        elif mask.dtype != np.uint8:
            issues.append(
                f"{r.sample_id}: mask dtype {mask.dtype}, expected uint8"
            )
        elif int(mask.min()) < 0 or int(mask.max()) > 18:
            issues.append(
                f"{r.sample_id}: mask values [{mask.min()}, {mask.max()}], expected 0-18"
            )

    status = "PASS" if not issues else "FAIL"
    return status, issues


def _validate_splits(
    results: list[PreparationResult],
    output_dir: Path,
) -> tuple[str, list[str]]:
    """Validate split files and exclusivity invariants."""
    successful = [r for r in results if r.aligned and r.masked]
    issues: list[str] = []

    # Build expected splits from results
    expected_splits: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for r in successful:
        if r.split in expected_splits:
            expected_splits[r.split].add(r.sample_id)

    # Read actual split files
    actual_splits: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for split_name in ("train", "val", "test"):
        split_path = output_dir / "splits" / f"{split_name}.txt"
        if not split_path.exists():
            issues.append(f"Split file missing: {split_name}.txt")
            continue
        for line in split_path.read_text(encoding="utf-8").splitlines():
            sid = line.strip()
            if sid:
                actual_splits[split_name].add(sid)

    # Check: every successful sample appears in exactly one split
    all_sample_ids = {r.sample_id for r in successful}
    all_split_ids = set()
    for split_name, ids in actual_splits.items():
        for sid in ids:
            if sid in all_split_ids:
                issues.append(f"{sid}: appears in multiple splits")
            all_split_ids.add(sid)

    # Check: no successful sample is missing from a split
    missing_from_split = all_sample_ids - all_split_ids
    for sid in sorted(missing_from_split):
        issues.append(f"{sid}: successful but missing from all splits")

    # Check: no extra IDs in split files
    extra_in_splits = all_split_ids - all_sample_ids
    for sid in sorted(extra_in_splits):
        issues.append(f"{sid}: in split file but not a successful sample")

    # Check: train + val + test == successful sample count
    split_total = sum(len(ids) for ids in actual_splits.values())
    if split_total != len(successful):
        issues.append(
            f"Split total {split_total} != successful count {len(successful)}"
        )

    # Check: split files contain only valid sample IDs (alphanumeric + underscore)
    for split_name, ids in actual_splits.items():
        for sid in ids:
            if not sid.startswith("sample_"):
                issues.append(
                    f"{sid}: invalid sample ID format in {split_name}.txt"
                )

    status = "PASS" if not issues else "FAIL"
    return status, issues


def _validate_manifest(
    results: list[PreparationResult],
    output_dir: Path,
) -> tuple[str, list[str]]:
    """Validate annotation manifest consistency."""
    successful = [r for r in results if r.aligned and r.masked]
    issues: list[str] = []

    manifest_path = output_dir / "annotation_manifest.json"
    if not manifest_path.exists():
        issues.append("annotation_manifest.json not found")
        return "FAIL", issues

    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)

    manifest_samples = m.get("samples", [])
    manifest_ids = [s["sample_id"] for s in manifest_samples]
    manifest_id_set = set(manifest_ids)
    successful_ids = {r.sample_id for r in successful}

    # Check: no duplicate sample IDs in manifest
    if len(manifest_ids) != len(manifest_id_set):
        issues.append(
            f"Duplicate sample IDs in manifest: {len(manifest_ids)} total, {len(manifest_id_set)} unique"
        )

    # Check: manifest sample count == successful count
    if m.get("total_samples", 0) != len(successful):
        issues.append(
            f"Manifest total_samples {m.get('total_samples', 0)} != successful count {len(successful)}"
        )

    # Check: manifest IDs == successful IDs
    missing_from_manifest = successful_ids - manifest_id_set
    for sid in sorted(missing_from_manifest):
        issues.append(f"{sid}: successful but missing from manifest")

    extra_in_manifest = manifest_id_set - successful_ids
    for sid in sorted(extra_in_manifest):
        issues.append(f"{sid}: in manifest but not a successful sample")

    # Check: every manifest sample has required fields
    required_fields = [
        "sample_id", "source_category", "source_filename",
        "aligned_image", "initial_mask", "ground_truth_mask",
        "split", "annotation_status",
    ]
    for sample in manifest_samples:
        sid = sample.get("sample_id", "?")
        for field_name in required_fields:
            if field_name not in sample or not sample[field_name]:
                issues.append(f"{sid}: manifest sample missing field '{field_name}'")

    # Check: manifest split counts match actual split files
    split_counts = m.get("splits", {})
    for split_name in ("train", "val", "test"):
        split_path = output_dir / "splits" / f"{split_name}.txt"
        if split_path.exists():
            file_count = sum(
                1 for line in split_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            manifest_count = split_counts.get(split_name, 0)
            if manifest_count != file_count:
                issues.append(
                    f"Manifest split '{split_name}' count {manifest_count} != file count {file_count}"
                )

    status = "PASS" if not issues else "FAIL"
    return status, issues


def _validate_metadata(
    results: list[PreparationResult],
    output_dir: Path,
) -> tuple[str, list[str]]:
    """Validate metadata files for every successful sample."""
    successful = [r for r in results if r.aligned and r.masked]
    issues: list[str] = []

    # Load manifest for split lookup
    manifest_path = output_dir / "annotation_manifest.json"
    manifest_splits: dict[str, str] = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
        for s in m.get("samples", []):
            manifest_splits[s["sample_id"]] = s.get("split", "")

    for r in successful:
        meta_path = Path(r.metadata_path)
        if not meta_path.exists():
            issues.append(f"{r.sample_id}: metadata file missing")
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        # Check sample_id matches
        if meta.get("sample_id") != r.sample_id:
            issues.append(
                f"{r.sample_id}: metadata sample_id mismatch '{meta.get('sample_id')}'"
            )

        # Check split matches manifest
        expected_split = manifest_splits.get(r.sample_id, r.split)
        if meta.get("split") != expected_split:
            issues.append(
                f"{r.sample_id}: metadata split '{meta.get('split')}' != manifest split '{expected_split}'"
            )

        # Check aligned_image path
        if meta.get("aligned_image") != r.aligned_path:
            issues.append(f"{r.sample_id}: metadata aligned_image path mismatch")

        # Check initial_mask path
        if meta.get("initial_mask") != r.initial_mask_path:
            issues.append(f"{r.sample_id}: metadata initial_mask path mismatch")

        # Check ground_truth_mask path
        if meta.get("ground_truth_mask") != r.mask_path:
            issues.append(f"{r.sample_id}: metadata ground_truth_mask path mismatch")

    status = "PASS" if not issues else "FAIL"
    return status, issues


def _validate_class_mapping(output_dir: Path) -> tuple[str, list[str]]:
    """Validate class_mapping.json exists and has expected 19 classes."""
    issues: list[str] = []
    mapping_path = output_dir / "metadata" / "class_mapping.json"

    if not mapping_path.exists():
        issues.append("class_mapping.json not found")
        return "FAIL", issues

    with open(mapping_path, encoding="utf-8") as f:
        mapping = json.load(f)

    expected_ids = set(range(19))
    actual_ids = set(mapping.keys()) if isinstance(mapping, dict) else set()

    # Handle string keys from JSON
    if actual_ids and all(isinstance(k, str) for k in actual_ids):
        actual_ids = {int(k) for k in actual_ids}

    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing:
            issues.append(f"Missing class IDs: {sorted(missing)}")
        if extra:
            issues.append(f"Extra class IDs: {sorted(extra)}")

    if len(mapping) != 19:
        issues.append(f"Expected 19 classes, got {len(mapping)}")

    status = "PASS" if not issues else "FAIL"
    return status, issues


# ── Reports ──────────────────────────────────────────────────────────


def _write_reports(
    report: PreparationReport,
    output_dir: Path,
) -> None:
    """Write preparation reports with all validation results."""
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Determine final status
    all_pass = all(
        v == "PASS"
        for v in [
            report.manifest_validation,
            report.split_validation,
            report.metadata_validation,
            report.class_mapping_validation,
            report.artifact_validation,
        ]
    )
    report.final_status = "READY_FOR_ANNOTATION" if all_pass else "NOT_READY_FOR_ANNOTATION"

    # JSON report
    json_path = reports_dir / "annotation_preparation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)

    # Text summary
    txt_path = reports_dir / "annotation_preparation_summary.txt"
    lines = [
        "ANNOTATION PREPARATION SUMMARY",
        "=" * 60,
        f"Generated: {report.generated_at}",
        "",
        f"Total KEEP:              {report.total_keep}",
        f"Prepared successfully:   {report.prepared_successfully}",
        f"Alignment failures:      {report.alignment_failures}",
        f"Mask-generation failures:{report.mask_generation_failures}",
        f"Missing outputs:         {report.missing_outputs}",
        "",
        f"Train: {report.train_count}",
        f"Val:   {report.val_count}",
        f"Test:  {report.test_count}",
        "",
        "Category distribution:",
    ]
    for cat, count in sorted(report.category_distribution.items()):
        lines.append(f"  {cat}: {count}")

    if report.errors:
        lines.append("")
        lines.append(f"Errors ({len(report.errors)}):")
        for err in report.errors[:20]:
            lines.append(f"  {err.get('sample_id', '?')}: {err.get('error', '?')}")
        if len(report.errors) > 20:
            lines.append(f"  ... and {len(report.errors) - 20} more")

    lines.append("")
    lines.append("VALIDATION RESULTS")
    lines.append("-" * 40)
    lines.append(f"Manifest validation:    {report.manifest_validation}")
    lines.append(f"Split validation:       {report.split_validation}")
    lines.append(f"Metadata validation:    {report.metadata_validation}")
    lines.append(f"Class mapping:          {report.class_mapping_validation}")
    lines.append(f"Artifact validation:    {report.artifact_validation}")

    if report.validation_issues:
        lines.append("")
        lines.append(f"Validation issues ({len(report.validation_issues)}):")
        for issue in report.validation_issues[:30]:
            lines.append(f"  - {issue}")
        if len(report.validation_issues) > 30:
            lines.append(f"  ... and {len(report.validation_issues) - 30} more")

    lines.append("")
    lines.append("=" * 60)
    lines.append(f"FINAL STATUS: {report.final_status}")
    lines.append("=" * 60)

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Main ─────────────────────────────────────────────────────────────


def prepare(manifest_path: Path | None = None, output_dir: Path | None = None) -> int:
    """Run the annotation preparation pipeline.

    Returns 0 on success, non-zero if failures occurred.
    """
    if manifest_path is None:
        manifest_path = MANIFEST_PATH
    if output_dir is None:
        output_dir = OUTPUT_DIR

    t_start = time.perf_counter()
    logger.info("=" * 60)
    logger.info("ANNOTATION PREPARATION")
    logger.info("Manifest: %s", manifest_path)
    logger.info("Output:   %s", output_dir)
    logger.info("=" * 60)

    # Load and filter
    manifest = _load_manifest(manifest_path)
    keep_images = _keep_images(manifest)
    sample_id_map = _assign_sample_ids(keep_images)

    logger.info("Total KEEP images: %d", len(keep_images))

    # Ensure output structure
    _ensure_dirs(output_dir)

    # Process each image
    results: list[PreparationResult] = []
    alignment_failures = 0
    mask_failures = 0

    for i, img_entry in enumerate(keep_images):
        key = f"{img_entry['category']}/{img_entry['filename']}"
        sample_id = sample_id_map[key]

        result = _process_image(img_entry, sample_id, output_dir)
        results.append(result)

        if result.error == "alignment_failed":
            alignment_failures += 1
        elif result.error == "mask_generation_failed":
            mask_failures += 1

        if (i + 1) % 50 == 0:
            logger.info("  Processed %d / %d", i + 1, len(keep_images))

    # Build manifest and splits
    logger.info("Building annotation manifest...")
    annotation_manifest = _build_manifest(results, output_dir, manifest)

    logger.info("Writing split files...")
    split_counts = _write_split_files(results, output_dir)

    # Class mapping
    logger.info("Saving class mapping...")
    save_class_mapping(output_dir=output_dir / "metadata")

    # ── Validations ─────────────────────────────────────────────────
    logger.info("Running validations...")

    artifact_status, artifact_issues = _validate_artifacts(results, output_dir)
    split_status, split_issues = _validate_splits(results, output_dir)
    manifest_status, manifest_issues = _validate_manifest(results, output_dir)
    metadata_status, metadata_issues = _validate_metadata(results, output_dir)
    class_mapping_status, class_mapping_issues = _validate_class_mapping(output_dir)

    all_issues = (
        artifact_issues + split_issues + manifest_issues
        + metadata_issues + class_mapping_issues
    )

    # Build report
    successful = [r for r in results if r.aligned and r.masked]
    missing = [
        r for r in results
        if r.aligned and r.masked and not Path(r.metadata_path).exists()
    ]

    category_dist: dict[str, int] = {}
    for r in successful:
        category_dist[r.source_category] = (
            category_dist.get(r.source_category, 0) + 1
        )

    report = PreparationReport(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        total_keep=len(keep_images),
        prepared_successfully=len(successful),
        alignment_failures=alignment_failures,
        mask_generation_failures=mask_failures,
        missing_outputs=len(missing),
        train_count=split_counts.get("train", 0),
        val_count=split_counts.get("val", 0),
        test_count=split_counts.get("test", 0),
        category_distribution=category_dist,
        errors=[
            {"sample_id": r.sample_id, "error": r.error}
            for r in results
            if r.error
        ],
        manifest_validation=manifest_status,
        split_validation=split_status,
        metadata_validation=metadata_status,
        class_mapping_validation=class_mapping_status,
        artifact_validation=artifact_status,
        validation_issues=all_issues,
    )

    _write_reports(report, output_dir)

    # Summary
    elapsed = time.perf_counter() - t_start
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPARATION COMPLETE (%.1f s)", elapsed)
    logger.info("=" * 60)
    logger.info("Total KEEP:              %d", report.total_keep)
    logger.info("Prepared successfully:   %d", report.prepared_successfully)
    logger.info("Alignment failures:      %d", report.alignment_failures)
    logger.info("Mask-generation failures:%d", report.mask_generation_failures)
    logger.info("Missing outputs:         %d", report.missing_outputs)
    logger.info("Train: %d  Val: %d  Test: %d",
                report.train_count, report.val_count, report.test_count)
    logger.info("Manifest validation:     %s", report.manifest_validation)
    logger.info("Split validation:        %s", report.split_validation)
    logger.info("Metadata validation:     %s", report.metadata_validation)
    logger.info("Class mapping:           %s", report.class_mapping_validation)
    logger.info("Artifact validation:     %s", report.artifact_validation)
    logger.info("FINAL STATUS:            %s", report.final_status)
    logger.info("")

    # Exit code
    failures = alignment_failures + mask_failures
    if report.final_status != "READY_FOR_ANNOTATION":
        logger.warning("Not ready for annotation. %d validation issues.", len(all_issues))
        return 1
    if failures > 0:
        logger.warning("%d preparation failures occurred.", failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(prepare())
