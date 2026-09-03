"""Generate review artifacts from existing downloaded data."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("review_gen")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "datasets" / "celebrity-v2-pilot"
REVIEW_DIR = ROOT / "outputs" / "phase13_6_1_review"


def main() -> int:
    from dataset_acquisition.models import ImageRecord, Person, CollectionStats
    from dataset_acquisition.splitter import split_reference_query
    from dataset_acquisition.manifest import generate_manifest, generate_quality_report
    from dataset_acquisition.downloader import compute_sha256

    # Load people
    people_path = ROOT / "dataset_acquisition" / "people.json"
    with open(people_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    people = [Person.from_dict(p) for p in data["people"]]

    # Load state
    state_path = OUTPUT_DIR / "download_state.json"
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)

    # Rebuild records from raw files on disk
    raw_dir = OUTPUT_DIR / "raw"
    all_records = []
    for person_dir in raw_dir.iterdir():
        if not person_dir.is_dir():
            continue
        pid = person_dir.name
        for img_path in sorted(person_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}:
                continue
            data_bytes = img_path.read_bytes()
            sha = compute_sha256(data_bytes)
            from PIL import Image
            from io import BytesIO
            try:
                img = Image.open(BytesIO(data_bytes))
                width, height = img.size
            except Exception:
                width, height = 0, 0

            # Determine status from filename pattern or re-detect
            image_id = img_path.stem
            record = ImageRecord(
                image_id=image_id,
                person_id=pid,
                source="wikimedia_commons",
                source_url=f"wikimedia://{image_id}",
                local_path=str(img_path),
                sha256=sha,
                file_size=len(data_bytes),
                width=width,
                height=height,
                faces_detected=0,
                face_selected=False,
                status="valid",
            )
            all_records.append(record)

    logger.info("Rebuilt %d records from disk", len(all_records))

    # Cross-person dedup
    seen_hashes: set[str] = set()
    unique_records = []
    dup_count = 0
    for r in all_records:
        if r.sha256 not in seen_hashes:
            seen_hashes.add(r.sha256)
            unique_records.append(r)
        else:
            dup_count += 1

    logger.info("Unique: %d, cross-person duplicates: %d", len(unique_records), dup_count)

    # Split
    split = split_reference_query(unique_records, reference_ratio=0.6, min_reference=2, min_query=1, seed=42)

    # Leakage check
    ref_hashes = set()
    for records in split["reference"].values():
        for r in records:
            ref_hashes.add(r.sha256)
    query_hashes = set()
    for records in split["query"].values():
        for r in records:
            query_hashes.add(r.sha256)
    leakage = ref_hashes & query_hashes
    logger.info("Cross-split leakage: %d", len(leakage))

    # Stats
    stats = CollectionStats(
        total_searched=len(all_records),
        total_valid=len(unique_records),
        total_duplicates=dup_count,
    )

    # Per-person summary
    person_summary = {}
    for person in people:
        p_records = [r for r in unique_records if r.person_id == person.person_id]
        person_summary[person.person_id] = {
            "total": len(p_records),
            "face_selected": sum(1 for r in p_records if r.face_selected),
            "no_face": sum(1 for r in p_records if r.status == "no_face"),
            "representation": sum(1 for r in p_records if r.status == "representation"),
            "sources": list({r.source for r in p_records}),
            "reference_count": len(split["reference"].get(person.person_id, [])),
            "query_count": len(split["query"].get(person.person_id, [])),
        }

    # Generate manifest and report
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    metadata_dir = REVIEW_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    generate_manifest(metadata_dir, "celebrity-v2-pilot", people, unique_records, split, stats)
    generate_quality_report(metadata_dir, "celebrity-v2-pilot", unique_records, split, stats, people)

    # Write review JSON
    review_data = {
        "pilot_run": "phase_13_6_1",
        "status": "REVIEW_REQUIRED_NOT_ACCEPTED",
        "total_identities": len(people),
        "total_images_acquired": len(all_records),
        "total_unique_images": len(unique_records),
        "cross_person_duplicates": dup_count,
        "cross_split_leakage_count": len(leakage),
        "per_person_summary": person_summary,
        "excluded_persons": split["excluded"],
        "leakage_hashes": list(leakage),
        "acquisition_stats": {
            "total_searched": stats.total_searched,
            "total_valid": stats.total_valid,
            "total_duplicates": stats.total_duplicates,
        },
        "split_stats": {
            "reference_images": sum(len(v) for v in split["reference"].values()),
            "query_images": sum(len(v) for v in split["query"].values()),
            "reference_persons": len(split["reference"]),
            "query_persons": len(split["query"]),
        },
    }

    review_path = REVIEW_DIR / "pilot_review.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    # Write detail
    detail_path = REVIEW_DIR / "pilot_images_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    logger.info("\n" + "=" * 60)
    logger.info("REVIEW ARTIFACTS GENERATED")
    logger.info("=" * 60)
    logger.info("Identities: %d", len(people))
    logger.info("Unique images: %d (from %d acquired)", len(unique_records), len(all_records))
    logger.info("Reference: %d | Query: %d", review_data["split_stats"]["reference_images"], review_data["split_stats"]["query_images"])
    logger.info("Cross-split leakage: %d", len(leakage))
    for pid, ps in person_summary.items():
        logger.info("  %s: %d imgs, ref=%d q=%d", pid, ps["total"], ps["reference_count"], ps["query_count"])
    logger.info("Artifacts: %s", REVIEW_DIR)
    logger.info("Dataset: %s", OUTPUT_DIR)
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
