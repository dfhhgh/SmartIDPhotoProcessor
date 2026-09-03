"""Phase 13.6.2c — Openverse + Wikimedia Category Discovery Pilot

Runs a 6-person pilot comparing three approaches:
  A: Wikimedia text search (existing)
  B: Wikimedia category discovery (new)
  C: Openverse API (new)

Target: ~5 accepted images per person.
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from dataset_acquisition.downloader import Downloader
from dataset_acquisition.models import Person, RejectionDetail
from dataset_acquisition.sources.openverse import OpenverseSource
from dataset_acquisition.sources.wikimedia import WikimediaSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 6-person pilot: 3 actors + 3 footballers
PILOT_PERSONS = [
    Person(
        person_id="tom_hanks",
        display_name="Tom Hanks",
        category="actor",
        search_queries=("Tom Hanks", "Thomas Hanks actor"),
    ),
    Person(
        person_id="scarlett_johansson",
        display_name="Scarlett Johansson",
        category="actor",
        search_queries=("Scarlett Johansson", "Scarlett Johansson actress"),
    ),
    Person(
        person_id="denzel_washington",
        display_name="Denzel Washington",
        category="actor",
        search_queries=("Denzel Washington", "Denzel Washington actor"),
    ),
    Person(
        person_id="lionel_messi",
        display_name="Lionel Messi",
        category="footballer",
        search_queries=("Lionel Messi", "Lionel Messi football"),
    ),
    Person(
        person_id="cristiano_ronaldo",
        display_name="Cristiano Ronaldo",
        category="footballer",
        search_queries=("Cristiano Ronaldo", "Cristiano Ronaldo football"),
    ),
    Person(
        person_id="kylian_mbappe",
        display_name="Kylian Mbappe",
        category="footballer",
        search_queries=("Kylian Mbappe", "Kylian Mbappe football"),
    ),
]

# Wikimedia category names for approach B
WIKIMEDIA_CATEGORIES = {
    "tom_hanks": "Tom Hanks",
    "scarlett_johansson": "Scarlett Johansson",
    "denzel_washington": "Denzel Washington",
    "lionel_messi": "Lionel Messi",
    "cristiano_ronaldo": "Cristiano Ronaldo",
    "kylian_mbappe": "Kylian Mbappe",
}


def run_pilot(
    output_base: Path,
    max_images_per_person: int = 5,
    max_search_results: int = 20,
) -> dict[str, Any]:
    """Run the pilot and return comparison data."""
    results: dict[str, Any] = {}

    # ─── Approach A: Wikimedia Text Search ───────────────────────────────
    logger.info("=" * 60)
    logger.info("APPROACH A: Wikimedia Text Search")
    logger.info("=" * 60)

    output_a = output_base / "approach_a_wikimedia_text"
    output_a.mkdir(parents=True, exist_ok=True)

    wiki_source = WikimediaSource(delay=2.0, max_rate_limit_retries=10)
    try:
        dl_a = Downloader(
            output_dir=output_a,
            sources=[wiki_source],
            max_images_per_person=max_images_per_person,
            delay=2.0,
        )
        approach_a_results = {}
        for person in PILOT_PERSONS:
            logger.info("Processing %s (Approach A)...", person.display_name)
            records, rejections = dl_a.download_person(person)
            approach_a_results[person.person_id] = {
                "accepted": len(records),
                "rejected": len(rejections),
                "rejection_reasons": _count_rejections(rejections),
            }
            logger.info(
                "  %s: accepted=%d, rejected=%d",
                person.display_name, len(records), len(rejections),
            )
        results["approach_a"] = approach_a_results
    finally:
        wiki_source.close()

    # ─── Approach B: Wikimedia Category Discovery ────────────────────────
    logger.info("=" * 60)
    logger.info("APPROACH B: Wikimedia Category Discovery")
    logger.info("=" * 60)

    output_b = output_base / "approach_b_wikimedia_category"
    output_b.mkdir(parents=True, exist_ok=True)

    wiki_cat_source = WikimediaSource(delay=2.0, max_rate_limit_retries=10)
    try:
        approach_b_results = {}
        for person in PILOT_PERSONS:
            logger.info("Processing %s (Approach B)...", person.display_name)
            category = WIKIMEDIA_CATEGORIES.get(person.person_id, person.display_name)
            person_dir = output_b / "raw" / person.person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            # Use category search directly
            collected = 0
            rejections: list[RejectionDetail] = []
            for result in wiki_cat_source.search_by_category(
                category, max_results=max_search_results
            ):
                if collected >= max_images_per_person:
                    break
                # Download and validate
                image_data = wiki_cat_source.download_url(result.image_url)
                if image_data is None:
                    rejections.append(RejectionDetail(
                        person_id=person.person_id,
                        source="wikimedia_commons",
                        source_url=result.source_url,
                        rejection_reason="download_error",
                        title=result.title,
                    ))
                    continue

                # For pilot, we just count results (full gate would need face service)
                collected += 1
                logger.info(
                    "  Category result %d: %s (%s)",
                    collected, result.title, result.license,
                )

            approach_b_results[person.person_id] = {
                "accepted": collected,
                "rejected": len(rejections),
                "rejection_reasons": _count_rejections(rejections),
            }
            logger.info(
                "  %s: found=%d, download_errors=%d",
                person.display_name, collected, len(rejections),
            )
        results["approach_b"] = approach_b_results
    finally:
        wiki_cat_source.close()

    # ─── Approach C: Openverse API ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("APPROACH C: Openverse API")
    logger.info("=" * 60)

    output_c = output_base / "approach_c_openverse"
    output_c.mkdir(parents=True, exist_ok=True)

    openverse_source = OpenverseSource(delay=1.0, max_rate_limit_retries=5)
    try:
        approach_c_results = {}
        for person in PILOT_PERSONS:
            logger.info("Processing %s (Approach C)...", person.display_name)
            person_dir = output_c / "raw" / person.person_id
            person_dir.mkdir(parents=True, exist_ok=True)

            collected = 0
            rejections: list[RejectionDetail] = []
            for result in openverse_source.search(
                person.display_name, max_results=max_search_results
            ):
                if collected >= max_images_per_person:
                    break
                # Download and validate
                image_data = openverse_source.download_url(result.image_url)
                if image_data is None:
                    rejections.append(RejectionDetail(
                        person_id=person.person_id,
                        source="openverse",
                        source_url=result.source_url,
                        rejection_reason="download_error",
                        title=result.title,
                    ))
                    continue

                # For pilot, we just count results
                collected += 1
                logger.info(
                    "  Openverse result %d: %s (%s)",
                    collected, result.title, result.license,
                )

            approach_c_results[person.person_id] = {
                "accepted": collected,
                "rejected": len(rejections),
                "rejection_reasons": _count_rejections(rejections),
            }
            logger.info(
                "  %s: found=%d, download_errors=%d",
                person.display_name, collected, len(rejections),
            )
        results["approach_c"] = approach_c_results
    finally:
        openverse_source.close()

    return results


def _count_rejections(rejections: list[RejectionDetail]) -> dict[str, int]:
    """Count rejections by reason."""
    counts: dict[str, int] = {}
    for r in rejections:
        counts[r.rejection_reason] = counts.get(r.rejection_reason, 0) + 1
    return counts


def generate_comparison_table(results: dict[str, Any]) -> str:
    """Generate a markdown comparison table."""
    lines = [
        "# Phase 13.6.2c — Source Comparison Table",
        "",
        "## Pilot Results (6 persons × 3 approaches)",
        "",
        "| Person | Approach A (Wikimedia Text) | Approach B (Wikimedia Category) | Approach C (Openverse) |",
        "|--------|---------------------------|--------------------------------|----------------------|",
    ]

    for person in PILOT_PERSONS:
        pid = person.person_id
        a = results.get("approach_a", {}).get(pid, {})
        b = results.get("approach_b", {}).get(pid, {})
        c = results.get("approach_c", {}).get(pid, {})
        lines.append(
            f"| {person.display_name} "
            f"| {a.get('accepted', 0)} accepted, {a.get('rejected', 0)} rejected "
            f"| {b.get('accepted', 0)} found, {b.get('rejected', 0)} errors "
            f"| {c.get('accepted', 0)} found, {c.get('rejected', 0)} errors |"
        )

    # Totals
    totals = {"a": {"accepted": 0, "rejected": 0}, "b": {"accepted": 0, "rejected": 0}, "c": {"accepted": 0, "rejected": 0}}
    for person in PILOT_PERSONS:
        pid = person.person_id
        for approach, key in [("a", "approach_a"), ("b", "approach_b"), ("c", "approach_c")]:
            data = results.get(key, {}).get(pid, {})
            totals[approach]["accepted"] += data.get("accepted", 0)
            totals[approach]["rejected"] += data.get("rejected", 0)

    lines.append(
        f"| **TOTAL** "
        f"| **{totals['a']['accepted']}** accepted, {totals['a']['rejected']} rejected "
        f"| **{totals['b']['accepted']}** found, {totals['b']['rejected']} errors "
        f"| **{totals['c']['accepted']}** found, {totals['c']['rejected']} errors |"
    )

    lines.extend([
        "",
        "## Source Verdicts",
        "",
        "| Source | Verdict | Rationale |",
        "|--------|---------|-----------|",
        "| Wikimedia Commons (Text Search) | SELECTED | Proven in production, 86.7% Top-1 retrieval |",
        "| Wikimedia Commons (Category) | SECONDARY | Higher precision for structured categories |",
        "| Openverse API | DEFERRED | Rate-limited (5 req/day unauthenticated), license concerns |",
        "| Getty Images | BLOCKED | Section 3.11 prohibits ML/AI use |",
        "| Flickr | BLOCKED | API key required, not available |",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    output_base = Path("outputs/phase13_6_2c")
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 13.6.2c pilot...")
    results = run_pilot(output_base, max_images_per_person=5, max_search_results=15)

    # Save results
    with open(output_base / "pilot_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate comparison table
    table = generate_comparison_table(results)
    with open(output_base / "source_comparison.md", "w") as f:
        f.write(table)

    logger.info("Pilot complete. Results saved to %s", output_base)
    logger.info("\n%s", table)
