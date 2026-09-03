"""Phase 13.6.1.2 — Manual Review & Dataset Quality Gate.

Generates review artifacts for manual inspection. All images start as PENDING.
The user must manually review and update decisions before the quality gate
can pass.
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
logger = logging.getLogger("phase13_6_1_2_review")

ROOT = Path(__file__).resolve().parent
PILOT_DIR = ROOT / "datasets" / "celebrity-v2-pilot-corrected"
PILOT_JSONL = ROOT / "outputs" / "phase13_6_1_1_review" / "pilot_images_detail.jsonl"
PEOPLE_JSON = ROOT / "dataset_acquisition" / "people.json"
OUTPUT_DIR = ROOT / "outputs" / "phase13_6_1_2_review"
CONTACT_SHEET_DIR = OUTPUT_DIR / "contact_sheets"


def main() -> int:
    from dataset_acquisition.review import (
        create_review_records,
        compute_review_stats,
        classify_identity_quality,
        generate_all_contact_sheets,
        save_review_records,
        save_review_stats,
    )

    if not PILOT_JSONL.exists():
        logger.error("Pilot JSONL not found: %s", PILOT_JSONL)
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Creating review records from pilot data...")
    records = create_review_records(PILOT_JSONL, PEOPLE_JSON)
    logger.info("Created %d PENDING review records", len(records))

    json_path, jsonl_path = save_review_records(records, OUTPUT_DIR)
    logger.info("Review records: %s", json_path)
    logger.info("Review records: %s", jsonl_path)

    logger.info("Generating contact sheets...")
    sheet_paths = generate_all_contact_sheets(PILOT_JSONL, PEOPLE_JSON, CONTACT_SHEET_DIR)
    logger.info("Generated %d contact sheets", len(sheet_paths))

    stats = compute_review_stats(records)
    identity_quality = classify_identity_quality(stats.per_person)
    stats_path = save_review_stats(stats, identity_quality, OUTPUT_DIR)
    logger.info("Review stats: %s", stats_path)

    logger.info("")
    logger.info("=" * 60)
    logger.info("PHASE 13.6.1.2 — REVIEW ARTIFACTS GENERATED")
    logger.info("=" * 60)
    logger.info("Total images: %d", stats.total_images)
    logger.info("Pending: %d", stats.pending)
    logger.info("Accepted: %d", stats.accepted)
    logger.info("Rejected: %d", stats.rejected)
    logger.info("Uncertain: %d", stats.uncertain)
    logger.info("")
    logger.info("Per-person breakdown:")
    for pid, pp in sorted(stats.per_person.items()):
        quality = identity_quality.get(pid, "UNKNOWN")
        logger.info(
            "  %s: total=%d pending=%d accepted=%d rejected=%d uncertain=%d -> %s",
            pid, pp["total"], pp["pending"], pp["accepted"],
            pp["rejected"], pp["uncertain"], quality,
        )
    logger.info("")
    logger.info("Contact sheets: %s", CONTACT_SHEET_DIR)
    logger.info("Review records: %s", OUTPUT_DIR / "manual_review.json")
    logger.info("Review stats: %s", stats_path)
    logger.info("")
    logger.info("NEXT STEP: Manually review each image in the contact sheets")
    logger.info("and update manual_review.json with ACCEPT/REJECT/UNCERTAIN decisions.")
    logger.info("The quality gate verdict will be DATASET_QUALITY_GATE_INCONCLUSIVE")
    logger.info("until all PENDING reviews are resolved.")
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
