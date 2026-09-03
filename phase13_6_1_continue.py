"""Continue Phase 13.6.1 pilot for remaining identities."""

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
logger = logging.getLogger("phase13_6_1_continue")

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

    people_path = ROOT / "dataset_acquisition" / "people.json"
    with open(people_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_people = [Person.from_dict(p) for p in data["people"]]

    # Check which identities already have images
    raw_dir = OUTPUT_DIR / "raw"
    already_done = set()
    if raw_dir.exists():
        for d in raw_dir.iterdir():
            if d.is_dir():
                n_files = len(list(d.glob("*.jpg")))
                if n_files >= 3:
                    already_done.add(d.name)

    remaining = [p for p in all_people if p.person_id not in already_done]
    logger.info("Already done: %s", sorted(already_done))
    logger.info("Remaining: %s", [p.person_id for p in remaining])

    if not remaining:
        logger.info("All identities already acquired.")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    face_service = FaceService()
    face_service.get_model()

    wikimedia = WikimediaSource(delay=3.0, max_retries=5, max_rate_limit_retries=8)
    downloader = Downloader(
        output_dir=OUTPUT_DIR,
        sources=[wikimedia],
        max_images_per_person=5,
        min_image_width=200,
        min_image_height=200,
        face_service=face_service,
    )

    all_records = []
    for i, person in enumerate(remaining, 1):
        logger.info("[%d/%d] Acquiring %s ...", i, len(remaining), person.person_id)
        records = downloader.download_person(person)
        all_records.extend(records)
        logger.info(
            "  -> %d images downloaded (%d face_selected, %d no_face)",
            len(records),
            sum(1 for r in records if r.face_selected),
            sum(1 for r in records if r.status == "no_face"),
        )

    logger.info("=== Continuation complete: %d new records ===", len(all_records))

    # Now regenerate full manifest from all raw images
    from dataset_acquisition.models import ImageRecord

    all_raw_records = []
    for person_dir in raw_dir.iterdir():
        if not person_dir.is_dir():
            continue
        pid = person_dir.name
        for img_path in person_dir.glob("*.jpg"):
            # Load from download_state.json
            pass

    # Read the state file to get all records
    state_path = OUTPUT_DIR / "download_state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        logger.info("State file loaded, %d downloaded persons", len(state.get("downloaded", {})))
    else:
        logger.warning("No state file found")

    # Collect all records from all persons
    all_person_records = []
    for person in all_people:
        person_records = downloader.download_person(person)
        all_person_records.extend(person_records)

    # Deduplicate
    seen_hashes: set[str] = set()
    unique_records = []
    for r in all_person_records:
        if r.sha256 not in seen_hashes:
            seen_hashes.add(r.sha256)
            unique_records.append(r)

    # Cross-split leakage check
    split = split_reference_query(unique_records, reference_ratio=0.6, min_reference=2, min_query=1, seed=42)
    ref_hashes = set()
    for records in split["reference"].values():
        for r in records:
            ref_hashes.add(r.sha256)
    query_hashes = set()
    for records in split["query"].values():
        for r in records:
            query_hashes.add(r.sha256)
    leakage = ref_hashes & query_hashes

    stats = CollectionStats(
        total_searched=len(all_person_records),
        total_valid=len(unique_records),
        total_duplicates=len(all_person_records) - len(unique_records),
        total_no_face=sum(1 for r in unique_records if r.status == "no_face"),
        total_multi_face=0,
        total_representation=sum(1 for r in unique_records if r.status == "representation"),
        total_identity_uncertain=sum(1 for r in unique_records if r.identity_status == "uncertain"),
    )

    # Copy split
    split_dir = OUTPUT_DIR / "split"
    copy_stats = copy_split_to_disk(split, split_dir)

    # Generate manifest
    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    manifest = generate_manifest(metadata_dir, "celebrity-v2-pilot", all_people, unique_records, split, stats)

    # Per-person summary
    person_summary = {}
    for person in all_people:
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

    review_queue = downloader.get_review_queue()

    review_data = {
        "pilot_run": "phase_13_6_1",
        "total_identities": len(all_people),
        "total_images_acquired": len(all_person_records),
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
    logger.info("Review written: %s", review_path)

    detail_path = REVIEW_DIR / "pilot_images_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    logger.info("\n" + "=" * 60)
    logger.info("PILOT SUMMARY (NOT ACCEPTED)")
    logger.info("=" * 60)
    logger.info("Identities: %d", len(all_people))
    logger.info("Unique images: %d", len(unique_records))
    logger.info("Reference: %d | Query: %d", review_data["split_stats"]["reference_images"], review_data["split_stats"]["query_images"])
    logger.info("Leakage: %d | Review queue: %d", len(leakage), len(review_queue))
    logger.info("No-face: %d | Representation: %d", stats.total_no_face, stats.total_representation)
    for pid, ps in person_summary.items():
        logger.info("  %s: %d imgs, %d face_sel, ref=%d q=%d", pid, ps["total"], ps["face_selected"], ps["reference_count"], ps["query_count"])
    logger.info("=" * 60)

    wikimedia.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
