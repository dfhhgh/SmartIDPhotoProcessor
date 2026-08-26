"""Run face detection audit on the real 931-image raw dataset.

Face detection: ENABLED
Quality scoring: DISABLED
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_builder.config.settings import Settings
from dataset_builder.collection.quality_audit import QualityAudit


def main() -> None:
    settings = Settings()
    raw_dir = settings.PROJECT_ROOT / "dataset" / "raw"

    if not raw_dir.exists():
        print(f"ERROR: Raw dataset directory not found: {raw_dir}")
        sys.exit(1)

    print(f"Raw dataset directory: {raw_dir}")
    print()

    audit = QualityAudit(settings)

    t0 = time.time()
    summary = audit.run(
        raw_dir=raw_dir,
        categories=None,
        skip_face_detection=False,
        skip_quality_scores=True,
    )
    elapsed = time.time() - t0

    print()
    print(f"Face detection audit completed in {elapsed:.1f}s")
    print()

    # Print summary
    print("=" * 60)
    print("DATASET QUALITY AUDIT SUMMARY (face detection enabled)")
    print("=" * 60)
    print(f"Total images:              {summary.total_images}")
    print(f"Readable:                  {summary.readable_images}")
    print(f"Oversized:                 {summary.oversized_images}")
    print(f"Face detection:            {'NOT RUN' if not summary.face_detection_run else 'RUN'}")
    if summary.face_detection_run:
        print(f"  One face:                {summary.one_face_images}")
        print(f"  Zero faces:              {summary.zero_face_images}")
        print(f"  Multiple faces:          {summary.multiple_face_images}")
    print(f"Quality scores:            {'NOT RUN' if not summary.quality_scores_run else 'RUN'}")
    print(f"Near-duplicate groups:     {summary.total_duplicate_groups}")
    print(f"Near-duplicate images:     {summary.total_near_duplicates}")
    print(f"Cross-category duplicates: {summary.total_cross_category_duplicates}")
    print(f"Review required:           {summary.review_required_count}")
    print(f"Clean candidates:          {summary.clean_candidate_count}")
    print()

    # Print category breakdown
    print("CATEGORY BREAKDOWN:")
    print("-" * 60)
    cat_sums = audit.get_category_summaries()
    for cat, cat_sum in sorted(cat_sums.items()):
        print(
            f"  {cat:20s}: total={cat_sum.total_images:4d}  "
            f"readable={cat_sum.readable_images:4d}  "
            f"1face={cat_sum.one_face_images:4d}  "
            f"0face={cat_sum.zero_face_images:4d}  "
            f"multi={cat_sum.multiple_face_images:4d}  "
            f"small={cat_sum.small_face_images:4d}  "
            f"profile={cat_sum.profile_face_images:4d}  "
            f"review={cat_sum.review_required_count:4d}  "
            f"clean={cat_sum.clean_candidate_count:4d}"
        )
    print()

    # Print oversized image paths
    if summary.oversized_image_paths:
        print(f"OVERSIZED IMAGES ({len(summary.oversized_image_paths)}):")
        for p in summary.oversized_image_paths:
            print(f"  {p}")
        print()

    # Generate reports
    output_dir = settings.PROJECT_ROOT.parent / "reports"
    output_dir.mkdir(exist_ok=True)

    audit.generate_reports(output_dir)
    print(f"Reports generated in {output_dir}/")

    # Print review candidates summary
    candidates = audit.get_review_candidates()
    print()
    print(f"MANUAL REVIEW CANDIDATES: {len(candidates)}")
    if candidates:
        reasons: dict[str, int] = {}
        for c in candidates:
            for r in c.review_reasons:
                reasons[r] = reasons.get(r, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  [{count:3d}] {reason}")

    # Verify dataset integrity
    print()
    print("DATASET INTEGRITY CHECK:")
    supported = settings.SUPPORTED_IMAGE_EXTENSIONS
    total_files = sum(
        1 for _ in raw_dir.rglob("*")
        if _.is_file() and _.suffix.lower() in supported
    )
    print(f"  Image files on disk: {total_files}")
    print(f"  Images audited: {summary.total_images}")
    assert total_files == summary.total_images, (
        f"MISMATCH: {total_files} files vs {summary.total_images} audited"
    )
    print("  PASS: File count matches")


if __name__ == "__main__":
    main()
