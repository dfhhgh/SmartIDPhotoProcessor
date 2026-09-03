"""Phase 13.6.1.1 — Corrected Pilot Script.

Fixes:
- Resume bug: only acquires missing/insufficient identities
- Stats derived from ImageRecord data (single source of truth)
- Multi-face statistics computed from records
- Proper idempotency
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
logger = logging.getLogger("phase13_6_1_1_pilot")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "datasets" / "celebrity-v2-pilot-corrected"
REVIEW_DIR = ROOT / "outputs" / "phase13_6_1_1_review"


def load_existing_records(output_dir: Path) -> list:
    """Load existing ImageRecords from download_state.json."""
    from dataset_acquisition.models import ImageRecord
    state_path = output_dir / "download_state.json"
    if not state_path.exists():
        return []
    with open(state_path, "r", encoding="utf-8") as f:
        state = json.load(f)
    records_raw = state.get("records", [])
    return [ImageRecord.from_dict(r) for r in records_raw]


def main() -> int:
    from services.face_service import FaceService
    from dataset_acquisition.models import Person, CollectionStats, compute_stats_from_records
    from dataset_acquisition.sources.wikimedia import WikimediaSource
    from dataset_acquisition.downloader import Downloader
    from dataset_acquisition.splitter import split_reference_query, copy_split_to_disk
    from dataset_acquisition.manifest import generate_manifest, generate_quality_report

    # Load people
    people_path = ROOT / "dataset_acquisition" / "people.json"
    with open(people_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    people = [Person.from_dict(p) for p in data["people"]]

    # Use 12 identities for corrected pilot (4 actors + 8 footballers)
    pilot_people = people[:12]
    logger.info("Pilot identities: %s", [p.person_id for p in pilot_people])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    face_service = FaceService()
    face_service.get_model()

    # Check resume state
    existing_records = load_existing_records(OUTPUT_DIR)
    existing_person_ids = {r.person_id for r in existing_records}
    existing_by_person = {}
    for r in existing_records:
        existing_by_person.setdefault(r.person_id, []).append(r)

    # Determine which persons need acquisition
    need_acquire = []
    already_sufficient = []
    MIN_IMAGES = 5
    for p in pilot_people:
        existing = existing_by_person.get(p.person_id, [])
        # Only count images that are not no_face or representation
        valid_count = sum(1 for r in existing if r.status == "valid" or r.face_selected)
        if valid_count >= MIN_IMAGES:
            already_sufficient.append(p.person_id)
            logger.info("  %s: already sufficient (%d valid images)", p.person_id, valid_count)
        else:
            need_acquire.append(p)
            logger.info("  %s: needs acquisition (%d valid images)", p.person_id, valid_count)

    logger.info("Already sufficient: %d, need acquisition: %d", len(already_sufficient), len(need_acquire))

    if need_acquire:
        wikimedia = WikimediaSource(delay=3.0, max_retries=5, max_rate_limit_retries=8)
        downloader = Downloader(
            output_dir=OUTPUT_DIR,
            sources=[wikimedia],
            max_images_per_person=MIN_IMAGES,
            min_image_width=200,
            min_image_height=200,
            face_service=face_service,
        )

        for i, person in enumerate(need_acquire, 1):
            logger.info("[%d/%d] Acquiring %s ...", i, len(need_acquire), person.person_id)
            records = downloader.download_person(person)
            logger.info(
                "  -> %d images (%d face_selected, %d no_face, %d multi_face, %d representation)",
                len(records),
                sum(1 for r in records if r.face_selected),
                sum(1 for r in records if r.faces_detected == 0),
                sum(1 for r in records if r.faces_detected > 1),
                sum(1 for r in records if r.image_category == "representation"),
            )

        wikimedia.close()

    # Reload all records from state after acquisition
    all_records = load_existing_records(OUTPUT_DIR)
    logger.info("Total records after acquisition: %d", len(all_records))

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

    # Stats from records (single source of truth)
    stats = compute_stats_from_records(unique_records)
    stats = CollectionStats(
        total_searched=stats.total_searched,
        total_downloaded=stats.total_downloaded,
        total_valid=stats.total_valid,
        total_duplicates=dup_count,
        total_no_face=stats.total_no_face,
        total_multi_face=stats.total_multi_face,
        total_representation=stats.total_representation,
        total_identity_uncertain=stats.total_identity_uncertain,
    )

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

    # Copy split
    split_dir = OUTPUT_DIR / "split"
    copy_stats = copy_split_to_disk(split, split_dir)

    # Generate manifest and report
    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    manifest = generate_manifest(metadata_dir, "celebrity-v2-pilot-corrected", pilot_people, unique_records, split, stats, seed=42)
    report_path = generate_quality_report(metadata_dir, "celebrity-v2-pilot-corrected", unique_records, split, stats, pilot_people)

    # Review artifacts
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # Per-person summary
    person_summary = {}
    for person in pilot_people:
        p_records = [r for r in unique_records if r.person_id == person.person_id]
        person_summary[person.person_id] = {
            "total": len(p_records),
            "face_selected": sum(1 for r in p_records if r.face_selected),
            "no_face": sum(1 for r in p_records if r.faces_detected == 0),
            "multi_face": sum(1 for r in p_records if r.faces_detected > 1),
            "representation": sum(1 for r in p_records if r.image_category == "representation"),
            "sources": list({r.source for r in p_records}),
            "reference_count": len(split["reference"].get(person.person_id, [])),
            "query_count": len(split["query"].get(person.person_id, [])),
        }

    # Source-level statistics
    source_stats: dict[str, dict[str, int]] = {}
    for r in unique_records:
        s = source_stats.setdefault(r.source, {"total": 0, "face_selected": 0, "no_face": 0, "multi_face": 0, "representation": 0})
        s["total"] += 1
        if r.face_selected:
            s["face_selected"] += 1
        if r.faces_detected == 0:
            s["no_face"] += 1
        if r.faces_detected > 1:
            s["multi_face"] += 1
        if r.image_category == "representation":
            s["representation"] += 1

    review_data = {
        "pilot_run": "phase_13_6_1_1",
        "status": "REVIEW_REQUIRED_NOT_ACCEPTED",
        "dataset_version": "celebrity-v2-pilot-corrected",
        "total_identities": len(pilot_people),
        "total_images_acquired": len(all_records),
        "total_unique_images": len(unique_records),
        "cross_person_duplicates": dup_count,
        "cross_split_leakage_count": len(leakage),
        "review_queue_count": 0,
        "per_person_summary": person_summary,
        "source_statistics": source_stats,
        "excluded_persons": split["excluded"],
        "leakage_hashes": list(leakage),
        "acquisition_stats": stats.to_dict(),
        "split_stats": {
            "reference_images": sum(len(v) for v in split["reference"].values()),
            "query_images": sum(len(v) for v in split["query"].values()),
            "reference_persons": len(split["reference"]),
            "query_persons": len(split["query"]),
        },
        "source_evaluation": {
            "wikimedia_commons": "SELECTED - person-name searchable, CC licenses, metadata-rich",
            "pexels": "REJECTED - stock photos only, not person-name searchable",
            "pixabay": "REJECTED - stock photos only, not person-name searchable",
            "unsplash": "REJECTED - stock photos only, not person-name searchable",
        },
    }

    review_path = REVIEW_DIR / "pilot_review.json"
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    detail_path = REVIEW_DIR / "pilot_images_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("CORRECTED PILOT SUMMARY (NOT ACCEPTED)")
    logger.info("=" * 60)
    logger.info("Identities: %d", len(pilot_people))
    logger.info("Unique images: %d (from %d acquired)", len(unique_records), len(all_records))
    logger.info("Reference: %d | Query: %d", review_data["split_stats"]["reference_images"], review_data["split_stats"]["query_images"])
    logger.info("Cross-split leakage: %d", len(leakage))
    logger.info("Face stats: valid=%d, no_face=%d, multi_face=%d, repr=%d, uncertain=%d",
                stats.total_valid, stats.total_no_face, stats.total_multi_face, stats.total_representation, stats.total_identity_uncertain)
    logger.info("Source stats: %s", source_stats)
    for pid, ps in person_summary.items():
        logger.info("  %s: %d imgs, face_sel=%d, no_face=%d, multi=%d, repr=%d, ref=%d q=%d",
                     pid, ps["total"], ps["face_selected"], ps["no_face"], ps["multi_face"], ps["representation"], ps["reference_count"], ps["query_count"])
    logger.info("Artifacts: %s", REVIEW_DIR)
    logger.info("Dataset: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
