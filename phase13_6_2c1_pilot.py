"""Phase 13.6.2c.1 — Fair Source Evaluation & Pilot Correction (Telemetry Corrected)

Runs a 6-person pilot comparing three approaches with the SAME validation gate:
  A: Wikimedia text search (existing)
  B: Wikimedia category discovery (new)
  C: Openverse API (new)

All approaches pass through the shared Single-Face Acquisition Gate.
Download success != accepted. Accepted means passed the gate.

Telemetry corrections vs previous run:
  - candidates_discovered vs candidates_examined are now distinct
  - Streaming: generators passed directly, not materialized with list()
  - Per-person rate-limit = delta (after - before), not cumulative
  - Runtime = real wall-clock elapsed time
  - Metric definitions documented in AcquisitionRunResult
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from dataset_acquisition.downloader import Downloader, compute_sha256
from dataset_acquisition.models import (
    AcquisitionRunResult,
    ImageRecord,
    Person,
    RejectionDetail,
    SearchResult,
)
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
        search_queries=("Tom Hanks",),
    ),
    Person(
        person_id="scarlett_johansson",
        display_name="Scarlett Johansson",
        category="actor",
        search_queries=("Scarlett Johansson",),
    ),
    Person(
        person_id="denzel_washington",
        display_name="Denzel Washington",
        category="actor",
        search_queries=("Denzel Washington",),
    ),
    Person(
        person_id="lionel_messi",
        display_name="Lionel Messi",
        category="footballer",
        search_queries=("Lionel Messi",),
    ),
    Person(
        person_id="cristiano_ronaldo",
        display_name="Cristiano Ronaldo",
        category="footballer",
        search_queries=("Cristiano Ronaldo",),
    ),
    Person(
        person_id="kylian_mbappe",
        display_name="Kylian Mbappe",
        category="footballer",
        search_queries=("Kylian Mbappe",),
    ),
]

WIKIMEDIA_CATEGORIES = {
    "tom_hanks": "Tom Hanks",
    "scarlett_johansson": "Scarlett Johansson",
    "denzel_washington": "Denzel Washington",
    "lionel_messi": "Lionel Messi",
    "cristiano_ronaldo": "Cristiano Ronaldo",
    "kylian_mbappe": "Kylian Mbappe",
}

SEED = 42
MAX_CANDIDATES_PER_PERSON = 30
MAX_ACCEPTED_PER_PERSON = 5
MAX_SEARCH_RESULTS = 50


def _count_rejections(rejections: list[RejectionDetail]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rejections:
        counts[r.rejection_reason] = counts.get(r.rejection_reason, 0) + 1
    return counts


def _person_metrics_from_result(
    result: AcquisitionRunResult,
    runtime_seconds: float,
    rate_limit_delta: int,
) -> dict[str, Any]:
    """Build per-person metrics dict from AcquisitionRunResult with corrected telemetry."""
    return {
        "candidates_discovered": result.candidates_discovered,
        "candidates_examined": result.candidates_examined,
        "candidates_skipped_existing": result.candidates_skipped_existing,
        "candidates_skipped_rejected": result.candidates_skipped_rejected,
        "accepted": result.accepted,
        "rejected": result.rejected,
        "acceptance_rate": result.acceptance_rate,
        **_count_rejections(result.rejection_details),
        "acquisition_runtime_seconds": round(runtime_seconds, 2),
        "rate_limit_errors": rate_limit_delta,
        "accepted_images_per_minute": round(
            result.accepted / (runtime_seconds / 60), 2
        ) if runtime_seconds > 0 else 0.0,
    }


def _total_metrics(
    per_person: dict[str, dict[str, Any]],
    total_runtime: float,
    total_rate_limit: int,
) -> dict[str, Any]:
    """Aggregate per-person metrics into total with real elapsed runtime."""
    total_accepted = sum(m["accepted"] for m in per_person.values())
    total_rejected = sum(m["rejected"] for m in per_person.values())
    total_discovered = sum(m["candidates_discovered"] for m in per_person.values())
    total_examined = sum(m["candidates_examined"] for m in per_person.values())
    total_skipped_existing = sum(m["candidates_skipped_existing"] for m in per_person.values())
    total_skipped_rejected = sum(m["candidates_skipped_rejected"] for m in per_person.values())

    rejections_by_reason: dict[str, int] = {}
    for m in per_person.values():
        for reason in _REJECTION_REASONS:
            if reason in m:
                rejections_by_reason[reason] = rejections_by_reason.get(reason, 0) + m[reason]

    return {
        "candidates_discovered": total_discovered,
        "candidates_examined": total_examined,
        "candidates_skipped_existing": total_skipped_existing,
        "candidates_skipped_rejected": total_skipped_rejected,
        "accepted": total_accepted,
        "rejected": total_rejected,
        "acceptance_rate": total_accepted / total_examined if total_examined > 0 else 0.0,
        **rejections_by_reason,
        "acquisition_runtime_seconds": round(total_runtime, 2),
        "rate_limit_errors": total_rate_limit,
        "accepted_images_per_minute": round(
            total_accepted / (total_runtime / 60), 2
        ) if total_runtime > 0 else 0.0,
    }


_REJECTION_REASONS = {
    "representation", "no_face", "multi_face", "download_error",
    "decode_error", "invalid_image", "duplicate", "other",
}


def _license_distribution(records: list[ImageRecord]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for r in records:
        dist[r.license] = dist.get(r.license, 0) + 1
    return dist


def _cross_source_duplicates(all_records: dict[str, list[ImageRecord]]) -> dict[str, Any]:
    """Detect duplicates within and across sources."""
    by_hash: dict[str, list[tuple[str, str, str]]] = {}
    for source_name, records in all_records.items():
        for r in records:
            by_hash.setdefault(r.sha256, []).append((source_name, r.person_id, r.source_url))

    within_source = {}
    across_sources = []
    across_persons = []

    for sha, entries in by_hash.items():
        sources = set(e[0] for e in entries)
        persons = set(e[1] for e in entries)
        if len(entries) > 1:
            if len(sources) == 1:
                src = list(sources)[0]
                within_source[src] = within_source.get(src, 0) + (len(entries) - 1)
            else:
                across_sources.append({
                    "sha256_prefix": sha[:16],
                    "sources": list(sources),
                    "persons": list(persons),
                    "count": len(entries),
                })
            if len(persons) > 1:
                across_persons.append({
                    "sha256_prefix": sha[:16],
                    "persons": list(persons),
                    "sources": list(sources),
                })

    return {
        "within_source_duplicates": within_source,
        "across_source_duplicates": across_sources,
        "across_person_duplicates": across_persons,
        "total_hashes": len(by_hash),
    }


def _split_leakage_check(split: dict[str, dict[str, list[ImageRecord]]]) -> dict[str, Any]:
    """Verify no SHA-256 overlap between reference and query splits."""
    ref_hashes: set[str] = set()
    query_hashes: set[str] = set()
    for records in split.get("reference", {}).values():
        for r in records:
            ref_hashes.add(r.sha256)
    for records in split.get("query", {}).values():
        for r in records:
            query_hashes.add(r.sha256)
    overlap = ref_hashes & query_hashes
    return {
        "reference_count": len(ref_hashes),
        "query_count": len(query_hashes),
        "overlap_count": len(overlap),
        "leakage_detected": len(overlap) > 0,
    }


# ---------------------------------------------------------------------------
# Approach A: Wikimedia Text Search (uses download_person which searches internally)
# ---------------------------------------------------------------------------

def run_approach_a(
    output_base: Path,
    persons: list[Person],
    face_service: Any | None = None,
) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("APPROACH A: Wikimedia Text Search (via Downloader gate)")
    logger.info("=" * 60)
    output_dir = output_base / "approach_a_wikimedia_text"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = WikimediaSource(delay=2.0, max_rate_limit_retries=10)
    all_records: list[ImageRecord] = []
    all_rejections: list[RejectionDetail] = []
    per_person: dict[str, dict[str, Any]] = {}
    approach_start = time.time()

    try:
        dl = Downloader(
            output_dir=output_dir,
            sources=[source],
            max_images_per_person=MAX_ACCEPTED_PER_PERSON,
            delay=2.0,
            face_service=face_service,
        )
        for person in persons:
            rl_before = source.rate_limit_errors
            t0 = time.time()
            records, rejections = dl.download_person(person)
            elapsed = time.time() - t0
            rl_delta = source.rate_limit_errors - rl_before

            # For text search, download_person handles discovery internally.
            # candidates_examined = accepted + rejected (each went through the gate).
            # We don't have separate discovered count from the internal search loop,
            # so we use examined = accepted + rejected as the best available metric.
            candidates_examined = len(records) + len(rejections)
            per_person[person.person_id] = {
                "candidates_discovered": candidates_examined,
                "candidates_examined": candidates_examined,
                "candidates_skipped_existing": 0,
                "candidates_skipped_rejected": 0,
                "accepted": len(records),
                "rejected": len(rejections),
                "acceptance_rate": len(records) / candidates_examined if candidates_examined > 0 else 0.0,
                **_count_rejections(rejections),
                "acquisition_runtime_seconds": round(elapsed, 2),
                "rate_limit_errors": rl_delta,
                "accepted_images_per_minute": round(
                    len(records) / (elapsed / 60), 2
                ) if elapsed > 0 else 0.0,
            }
            all_records.extend(records)
            all_rejections.extend(rejections)
            logger.info(
                "  %s: accepted=%d, rejected=%d (rl_delta=%d)",
                person.display_name, len(records), len(rejections), rl_delta,
            )

        total_runtime = time.time() - approach_start
    finally:
        source.close()

    return {
        "source_name": "wikimedia_commons_text",
        "per_person": per_person,
        "total_metrics": _total_metrics(per_person, total_runtime, source.rate_limit_errors),
        "license_distribution": _license_distribution(all_records),
        "all_records": all_records,
        "all_rejections": all_rejections,
    }


# ---------------------------------------------------------------------------
# Approach B: Wikimedia Category Discovery (streaming)
# ---------------------------------------------------------------------------

def run_approach_b(
    output_base: Path,
    persons: list[Person],
    face_service: Any | None = None,
) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("APPROACH B: Wikimedia Category Discovery (streaming via shared gate)")
    logger.info("=" * 60)
    output_dir = output_base / "approach_b_wikimedia_category"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = WikimediaSource(delay=2.0, max_rate_limit_retries=10)
    all_records: list[ImageRecord] = []
    all_rejections: list[RejectionDetail] = []
    per_person: dict[str, dict[str, Any]] = {}
    approach_start = time.time()

    try:
        dl = Downloader(
            output_dir=output_dir,
            sources=[source],
            max_images_per_person=MAX_ACCEPTED_PER_PERSON,
            delay=2.0,
            face_service=face_service,
        )
        for person in persons:
            category = WIKIMEDIA_CATEGORIES.get(person.person_id, person.display_name)
            rl_before = source.rate_limit_errors
            t0 = time.time()

            # Streaming: pass generator directly, do NOT materialize with list()
            candidates_iter = source.search_by_category(category, max_results=MAX_SEARCH_RESULTS)
            result = dl.download_candidates(
                person=person,
                candidates=candidates_iter,
                source_name="wikimedia_commons",
                max_candidates=MAX_CANDIDATES_PER_PERSON,
            )

            elapsed = time.time() - t0
            rl_delta = source.rate_limit_errors - rl_before

            per_person[person.person_id] = _person_metrics_from_result(result, elapsed, rl_delta)
            all_records.extend(result.records)
            all_rejections.extend(result.rejection_details)
            logger.info(
                "  %s: discovered=%d, examined=%d, accepted=%d (rl_delta=%d)",
                person.display_name, result.candidates_discovered,
                result.candidates_examined, result.accepted, rl_delta,
            )

        total_runtime = time.time() - approach_start
    finally:
        source.close()

    return {
        "source_name": "wikimedia_commons_category",
        "per_person": per_person,
        "total_metrics": _total_metrics(per_person, total_runtime, source.rate_limit_errors),
        "license_distribution": _license_distribution(all_records),
        "all_records": all_records,
        "all_rejections": all_rejections,
    }


# ---------------------------------------------------------------------------
# Approach C: Openverse API (streaming)
# ---------------------------------------------------------------------------

def run_approach_c(
    output_base: Path,
    persons: list[Person],
    face_service: Any | None = None,
) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("APPROACH C: Openverse API (streaming via shared gate)")
    logger.info("=" * 60)
    output_dir = output_base / "approach_c_openverse"
    output_dir.mkdir(parents=True, exist_ok=True)

    source = OpenverseSource(delay=1.0, max_rate_limit_retries=5)
    all_records: list[ImageRecord] = []
    all_rejections: list[RejectionDetail] = []
    per_person: dict[str, dict[str, Any]] = {}
    approach_start = time.time()

    try:
        dl = Downloader(
            output_dir=output_dir,
            sources=[source],
            max_images_per_person=MAX_ACCEPTED_PER_PERSON,
            delay=1.0,
            face_service=face_service,
        )
        for person in persons:
            rl_before = source.rate_limit_errors
            t0 = time.time()

            # Streaming: pass generator directly, do NOT materialize with list()
            candidates_iter = source.search(person.display_name, max_results=MAX_SEARCH_RESULTS)
            result = dl.download_candidates(
                person=person,
                candidates=candidates_iter,
                source_name="openverse",
                max_candidates=MAX_CANDIDATES_PER_PERSON,
            )

            elapsed = time.time() - t0
            rl_delta = source.rate_limit_errors - rl_before

            per_person[person.person_id] = _person_metrics_from_result(result, elapsed, rl_delta)
            all_records.extend(result.records)
            all_rejections.extend(result.rejection_details)
            logger.info(
                "  %s: discovered=%d, examined=%d, accepted=%d (rl_delta=%d)",
                person.display_name, result.candidates_discovered,
                result.candidates_examined, result.accepted, rl_delta,
            )

        total_runtime = time.time() - approach_start
    finally:
        source.close()

    return {
        "source_name": "openverse",
        "per_person": per_person,
        "total_metrics": _total_metrics(per_person, total_runtime, source.rate_limit_errors),
        "license_distribution": _license_distribution(all_records),
        "all_records": all_records,
        "all_rejections": all_rejections,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pilot(output_base: Path, face_service: Any | None = None) -> dict[str, Any]:
    results_a = run_approach_a(output_base, PILOT_PERSONS, face_service)
    results_b = run_approach_b(output_base, PILOT_PERSONS, face_service)
    results_c = run_approach_c(output_base, PILOT_PERSONS, face_service)

    # Cross-source duplicate detection
    all_by_source = {
        "wikimedia_text": results_a["all_records"],
        "wikimedia_category": results_b["all_records"],
        "openverse": results_c["all_records"],
    }
    cross_dupes = _cross_source_duplicates(all_by_source)

    # Split each source's accepted records
    from dataset_acquisition.splitter import split_reference_query
    splits = {}
    leakage = {}
    for label, records in [("wikimedia_text", results_a["all_records"]),
                           ("wikimedia_category", results_b["all_records"]),
                           ("openverse", results_c["all_records"])]:
        if records:
            split = split_reference_query(records, seed=SEED)
            splits[label] = {
                "reference": {k: [r.to_dict() for r in v] for k, v in split["reference"].items()},
                "query": {k: [r.to_dict() for r in v] for k, v in split["query"].items()},
                "excluded": split["excluded"],
            }
            leakage[label] = _split_leakage_check(split)
        else:
            splits[label] = {"reference": {}, "query": {}, "excluded": {}}
            leakage[label] = {"reference_count": 0, "query_count": 0, "overlap_count": 0, "leakage_detected": False}

    return {
        "phase": "13.6.2c.1",
        "telemetry_corrected": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric_definitions": {
            "candidates_discovered": "Number yielded by source iterator",
            "candidates_examined": "Number presented to validation gate after skipping existing/rejected",
            "candidates_skipped_existing": "Skipped because source_url already downloaded",
            "candidates_skipped_rejected": "Skipped because source_url already rejected",
            "accepted": "Passed full validation gate and saved",
            "rejected": "Processed by gate and rejected",
            "acceptance_rate": "accepted / candidates_examined",
            "rate_limit_errors": "Per-person: delta (after - before). Source-level: cumulative.",
        },
        "pilot_config": {
            "persons": [p.to_dict() for p in PILOT_PERSONS],
            "max_candidates_per_person": MAX_CANDIDATES_PER_PERSON,
            "max_accepted_per_person": MAX_ACCEPTED_PER_PERSON,
            "max_search_results": MAX_SEARCH_RESULTS,
            "seed": SEED,
        },
        "approach_a": {
            "source_name": results_a["source_name"],
            "per_person": results_a["per_person"],
            "total_metrics": results_a["total_metrics"],
            "license_distribution": results_a["license_distribution"],
        },
        "approach_b": {
            "source_name": results_b["source_name"],
            "per_person": results_b["per_person"],
            "total_metrics": results_b["total_metrics"],
            "license_distribution": results_b["license_distribution"],
        },
        "approach_c": {
            "source_name": results_c["source_name"],
            "per_person": results_c["per_person"],
            "total_metrics": results_c["total_metrics"],
            "license_distribution": results_c["license_distribution"],
        },
        "cross_source_duplicates": cross_dupes,
        "splits": splits,
        "split_leakage": leakage,
        "rejection_telemetry": {
            "wikimedia_text": _count_rejections(results_a["all_rejections"]),
            "wikimedia_category": _count_rejections(results_b["all_rejections"]),
            "openverse": _count_rejections(results_c["all_rejections"]),
        },
    }


if __name__ == "__main__":
    output_base = Path("outputs/phase13_6_2c1")
    output_base.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 13.6.2c.1 Fair Source Evaluation Pilot (telemetry corrected)...")
    results = run_pilot(output_base)

    # Save results
    with open(output_base / "pilot_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info("Pilot complete. Results saved to %s", output_base)
