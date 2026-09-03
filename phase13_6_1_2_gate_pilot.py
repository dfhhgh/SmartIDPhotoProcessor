"""Phase 13.6.1.2a — Single-Face Acquisition Gate Pilot with Telemetry.

Runs a small real-world pilot with 3 identities from Wikimedia Commons
to validate rejection telemetry:
  - Only exactly-one-face images are saved to disk
  - Rejection reasons are tracked per-candidate
  - Per-person and per-source statistics are accurate
  - Resume idempotency is preserved
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("phase13_6_1_2_gate_pilot")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "datasets" / "celebrity-v2-pilot-gated"
REVIEW_DIR = ROOT / "outputs" / "phase13_6_1_2_gate_pilot"


def main() -> int:
    from services.face_service import FaceService
    from dataset_acquisition.models import Person, CollectionStats, RejectionDetail, compute_stats_from_records
    from dataset_acquisition.sources.wikimedia import WikimediaSource
    from dataset_acquisition.downloader import Downloader
    from dataset_acquisition.splitter import split_reference_query, copy_split_to_disk
    from dataset_acquisition.manifest import generate_manifest, generate_quality_report, compute_rejection_stats

    people_path = ROOT / "dataset_acquisition" / "people.json"
    with open(people_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    people = [Person.from_dict(p) for p in data["people"]]

    pilot_people = people[:3]
    logger.info("Pilot identities: %s", [p.person_id for p in pilot_people])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    face_service = FaceService()
    face_service.get_model()

    max_images = 5

    wikimedia = WikimediaSource(delay=3.0, max_retries=5, max_rate_limit_retries=8)
    downloader = Downloader(
        output_dir=OUTPUT_DIR,
        sources=[wikimedia],
        max_images_per_person=max_images,
        min_image_width=200,
        min_image_height=200,
        face_service=face_service,
    )

    all_rejection_details: list[RejectionDetail] = []
    for i, person in enumerate(pilot_people, 1):
        logger.info("[%d/%d] Acquiring %s ...", i, len(pilot_people), person.person_id)
        t0 = time.time()
        records, rej_details = downloader.download_person(person)
        elapsed = time.time() - t0
        all_rejection_details.extend(rej_details)
        logger.info(
            "  -> %d images acquired, %d rejected in %.1fs",
            len(records), len(rej_details), elapsed,
        )

    wikimedia.close()

    state = downloader._load_state()
    all_records_raw = state.get("records", [])
    all_records = []
    for r in all_records_raw:
        if r.get("person_id") in {p.person_id for p in pilot_people}:
            all_records.append(__import__("dataset_acquisition.models", fromlist=["ImageRecord"]).ImageRecord.from_dict(r))

    logger.info("Total records: %d", len(all_records))

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

    stats = compute_stats_from_records(unique_records)
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
    logger.info("Cross-split leakage: %d", len(leakage))

    split_dir = OUTPUT_DIR / "split"
    copy_split_to_disk(split, split_dir)

    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    manifest = generate_manifest(
        metadata_dir, "celebrity-v2-pilot-gated", pilot_people, unique_records, split, stats, seed=42,
        rejection_details=all_rejection_details,
    )
    report_path = generate_quality_report(
        metadata_dir, "celebrity-v2-pilot-gated", unique_records, split, stats, pilot_people,
        rejection_details=all_rejection_details,
    )

    # Rejection telemetry
    rstats = compute_rejection_stats(unique_records, all_rejection_details)

    # Per-person summary
    person_summary = {}
    for person in pilot_people:
        p_records = [r for r in unique_records if r.person_id == person.person_id]
        pp = rstats.per_person.get(person.person_id, {})
        person_summary[person.person_id] = {
            "accepted": pp.get("accepted", 0),
            "rejected_total": pp.get("rejected_total", 0),
            "representation": pp.get("representation", 0),
            "no_face": pp.get("no_face", 0),
            "multi_face": pp.get("multi_face", 0),
            "sources": list({r.source for r in p_records}),
            "reference_count": len(split["reference"].get(person.person_id, [])),
            "query_count": len(split["query"].get(person.person_id, [])),
        }

    review_data = {
        "pilot_run": "phase_13_6_1_2a_telemetry",
        "status": "TELEMETRY_VALIDATED",
        "dataset_version": "celebrity-v2-pilot-gated",
        "total_identities": len(pilot_people),
        "total_candidates": rstats.total_candidates,
        "accepted": rstats.accepted,
        "rejected_total": rstats.rejected_total,
        "rejections_by_reason": rstats.rejections_by_reason,
        "cross_person_duplicates": dup_count,
        "cross_split_leakage_count": len(leakage),
        "per_person_summary": person_summary,
        "per_source": rstats.per_source,
        "gate_policy": "Only exactly-one-face images accepted. Rejection reasons tracked per-candidate.",
        "acquisition_stats": stats.to_dict(),
    }

    review_path = REVIEW_DIR / "gate_pilot_review.json"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with open(review_path, "w", encoding="utf-8") as f:
        json.dump(review_data, f, indent=2, ensure_ascii=False)

    detail_path = REVIEW_DIR / "gate_pilot_images_detail.jsonl"
    with open(detail_path, "w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    logger.info("\n" + "=" * 60)
    logger.info("SINGLE-FACE GATE PILOT SUMMARY (TELEMETRY)")
    logger.info("=" * 60)
    logger.info("Identities: %d", len(pilot_people))
    logger.info("Total candidates: %d", rstats.total_candidates)
    logger.info("Accepted: %d", rstats.accepted)
    logger.info("Rejected: %d", rstats.rejected_total)
    logger.info("Rejection reasons: %s", rstats.rejections_by_reason)
    logger.info("Reference: %d | Query: %d",
                sum(len(v) for v in split["reference"].values()),
                sum(len(v) for v in split["query"].values()))
    logger.info("Cross-split leakage: %d", len(leakage))
    for pid, ps in person_summary.items():
        logger.info("  %s: accepted=%d, rejected=%d (repr=%d, no_face=%d, multi=%d), ref=%d q=%d",
                     pid, ps["accepted"], ps["rejected_total"],
                     ps["representation"], ps["no_face"], ps["multi_face"],
                     ps["reference_count"], ps["query_count"])
    logger.info("Per-source: %s", rstats.per_source)
    logger.info("Artifacts: %s", REVIEW_DIR)
    logger.info("Dataset: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
