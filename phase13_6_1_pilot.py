"""Phase 13.6.1 pilot: real Wikimedia acquisition for 10-13 identities.

Runs acquisition, generates review artifacts, stops without auto-accept.
"""

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
logger = logging.getLogger("phase13_6_1_pilot")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "datasets" / "celebrity-v2-pilot"
REVIEW_DIR = ROOT / "outputs" / "phase13_6_1_review"


def main() -> int:
    from services.face_service import FaceService
    from dataset_acquisition.models import Person, CollectionStats
    from dataset_acquisition.sources.wikimedia import WikimediaSource
    from dataset_acquisition.downloader import Downloader
    from dataset_acquisition.splitter import split_reference_query, copy_split_to_disk
    from dataset_acquisition.manifest import generate_manifest, generate_quality_report

    # Select 13 identities from people.json (6 actors + 7 footballers)
    people_path = ROOT / "dataset_acquisition" / "people.json"
    with open(people_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = [Person.from_dict(p) for p in data["people"]]
    logger.info("Pilot identities: %s", [p.person_id for p in people])

    # Initialize
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    face_service = FaceService()
    face_service.get_model()  # warm up

    wikimedia = WikimediaSource(delay=2.5, max_retries=5)
    downloader = Downloader(
        output_dir=OUTPUT_DIR,
        sources=[wikimedia],
        max_images_per_person=5,
        min_image_width=200,
        min_image_height=200,
        face_service=face_service,
    )

    # Acquire
    logger.info("=== Starting acquisition ===")
    all_records = []
    for i, person in enumerate(people, 1):
        logger.info("[%d/%d] Acquiring %s ...", i, len(people), person.person_id)
        records = downloader.download_person(person)
        all_records.extend(records)
        logger.info(
            "  -> %d images downloaded (%d face_selected, %d no_face)",
            len(records),
            sum(1 for r in records if r.face_selected),
            sum(1 for r in records if r.status == "no_face"),
        )

    logger.info("=== Acquisition complete: %d total records ===", len(all_records))

    # Deduplicate across persons
    seen_hashes: set[str] = set()
    unique_records = []
    duplicates = 0
    for r in all_records:
        if r.sha256 not in seen_hashes:
            seen_hashes.add(r.sha256)
            unique_records.append(r)
        else:
            duplicates += 1
    logger.info("After cross-person dedup: %d unique, %d duplicates", len(unique_records), duplicates)

    # Split
    split = split_reference_query(unique_records, reference_ratio=0.6, min_reference=2, min_query=1, seed=42)
    logger.info(
        "Split: %d reference, %d query, %d excluded persons",
        sum(len(v) for v in split["reference"].values()),
        sum(len(v) for v in split["query"].values()),
        len(split["excluded"]),
    )

    # Cross-split leakage check
    ref_hashes = set()
    for records in split["reference"].values():
        for r in records:
            ref_hashes.add(r.sha256)
    query_hashes = set()
    for records in split["query"].values():
        for r in records:
            query_hashes.add(r.sha256)
    leakage = ref_hashes & query_hashes
    logger.info("Cross-split leakage: %d duplicate hashes", len(leakage))

    # Stats
    stats = CollectionStats(
        total_searched=len(all_records),
        total_valid=len(unique_records),
        total_duplicates=duplicates + len(leakage),
        total_no_face=sum(1 for r in unique_records if r.status == "no_face"),
        total_multi_face=0,
        total_representation=sum(1 for r in unique_records if r.status == "representation"),
        total_identity_uncertain=sum(1 for r in unique_records if r.identity_status == "uncertain"),
    )

    # Copy split to disk
    split_dir = OUTPUT_DIR / "split"
    copy_stats = copy_split_to_disk(split, split_dir)
    logger.info("Split copied to disk: %s", copy_stats)

    # Generate manifest
    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    manifest = generate_manifest(metadata_dir, "celebrity-v2-pilot", people, unique_records, split, stats)

    # Generate quality report
    report_path = generate_quality_report(metadata_dir, "celebrity-v2-pilot", unique_records, split, stats, people)

    # Review artifacts
    review_queue = downloader.get_review_queue()

    # Per-person quality summary
    person_summary = {}
    for person in people:
        p_records = [r for r in unique_records if r.person_id == person.person_id]
        person_summary[person.person_id] = {
            "total": len(p_records),
            "face_selected": sum(1 for r in p_records if r.face_selected),
            "no_face": sum(1 for r in p_records if r.status == "no_face"),
            "representation": sum(1 for r in p_records if r.status == "representation"),
            "sources": list({r.source for r in p_records}),
            "image_categories": list({r.image_category for r in p_records}),
            "reference_count": len(split["reference"].get(person.person_id, [])),
            "query_count": len(split["query"].get(person.person_id, [])),
        }

    # Write review artifacts
    review_data = {
        "pilot_run": "phase_13_6_1",
        "total_identities": len(people),
        "total_images_acquired": len(all_records),
        "total_unique_images": len(unique_records),
        "cross_split_leakage_count": len(leakage),
        "review_queue_count": len(review_queue),
        "per_person_summary": person_summary,
        "excluded_persons": split["excluded"],
        "review_queue": [item.to_dict() for item in review_queue],
        "leakage_hashes": list(leakage),
        "acquisition_stats": {
            "total_searched": stats.total_searched,
            "total_valid": stats.total_valid,
            "total_duplicates": stats.total_duplicates,
            "total_no_face": stats.total_no_face,
            "total_multi_face": stats.total_multi_face,
            "total_representation": stats.total_representation,
            "total_identity_uncertain": stats.total_identity_uncertain,
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
    logger.info("Review artifacts written to: %s", review_path)

    # Write per-image detail
    detail_path = REVIEW_DIR / "pilot_images_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Image detail written to: %s", detail_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("PILOT SUMMARY (NOT ACCEPTED - REVIEW REQUIRED)")
    logger.info("=" * 60)
    logger.info("Identities: %d", len(people))
    logger.info("Total images acquired: %d", len(all_records))
    logger.info("Unique images after dedup: %d", len(unique_records))
    logger.info("Reference images: %d", review_data["split_stats"]["reference_images"])
    logger.info("Query images: %d", review_data["split_stats"]["query_images"])
    logger.info("Cross-split leakage: %d", len(leakage))
    logger.info("Review queue (multi-face): %d", len(review_queue))
    logger.info("No-face images: %d", stats.total_no_face)
    logger.info("Representation images: %d", stats.total_representation)
    logger.info("")
    for pid, ps in person_summary.items():
        logger.info(
            "  %s: %d images, %d face_selected, %d no_face, ref=%d, q=%d",
            pid, ps["total"], ps["face_selected"], ps["no_face"],
            ps["reference_count"], ps["query_count"],
        )
    logger.info("")
    logger.info("Review artifacts: %s", REVIEW_DIR)
    logger.info("Dataset (not accepted): %s", OUTPUT_DIR)
    logger.info("To accept: run Phase 13.6.1 acceptance with generated artifacts")
    logger.info("=" * 60)

    wikimedia.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
