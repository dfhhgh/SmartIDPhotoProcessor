"""Source fallback orchestrator for scaled dataset acquisition.

Implements the priority chain:
  1. Wikimedia Text Search
  2. Wikimedia Category Discovery
  3. Openverse (only when needed)

Each source is tried in order. If a source produces enough accepted images,
later sources are skipped. This prevents unnecessary API usage.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterator

from dataset_acquisition.downloader import Downloader
from dataset_acquisition.models import (
    AcquisitionRunResult,
    ImageRecord,
    Person,
    RejectionDetail,
    SearchResult,
)
from dataset_acquisition.sources.base import ImageSource
from dataset_acquisition.sources.openverse import OpenverseSource
from dataset_acquisition.sources.wikimedia import WikimediaSource

logger = logging.getLogger(__name__)


class AcquisitionOrchestrator:
    """Orchestrates multi-source image acquisition with fallback.

    Source priority:
      1. Wikimedia text search (via Downloader.download_person)
      2. Wikimedia category discovery (via Downloader.download_candidates)
      3. Openverse API (via Downloader.download_candidates)

    Stops as soon as the target image count is reached for a person.
    """

    def __init__(
        self,
        output_dir: Path,
        max_images_per_person: int = 12,
        max_candidates_per_source: int = 50,
        wikimedia_delay: float = 2.0,
        openverse_delay: float = 1.0,
        face_service: Any | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._max_images = max_images_per_person
        self._max_candidates = max_candidates_per_source
        self._wikimedia_delay = wikimedia_delay
        self._openverse_delay = openverse_delay
        self._face_service = face_service

        self._wikimedia_source = WikimediaSource(
            delay=wikimedia_delay, max_rate_limit_retries=10
        )
        self._openverse_source = OpenverseSource(
            delay=openverse_delay, max_rate_limit_retries=5
        )

        self._per_person_metrics: dict[str, dict[str, Any]] = {}
        self._all_records: list[ImageRecord] = []
        self._all_rejections: list[RejectionDetail] = []

    def collect_person(
        self,
        person: Person,
        target_images: int | None = None,
    ) -> dict[str, Any]:
        """Acquire images for a single person using source fallback.

        Returns per-person metrics dict with:
          person_id, display_name, category
          target_images, accepted, status
          sources_tried, source_breakdown
          per_source_metrics
          rejections_by_reason
        """
        target = target_images or self._max_images
        person_dir = self._output_dir / "raw" / person.person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        # Load existing state for resume
        state_path = self._output_dir / "download_state.json"
        state: dict[str, Any] = {}
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

        # Count existing accepted images
        existing_urls = set(state.get("downloaded", {}).get(person.person_id, []))
        existing_count = len(existing_urls)

        if existing_count >= target:
            logger.info(
                "Person %s: already has %d images (target=%d), skipping.",
                person.person_id, existing_count, target,
            )
            return self._build_person_result(
                person, target, existing_count, "COMPLETED", {}, {}
            )

        sources_tried: list[str] = []
        source_breakdown: dict[str, int] = {}
        per_source_metrics: dict[str, dict[str, Any]] = {}
        all_new_rejections: list[RejectionDetail] = []

        # --- Source 1: Wikimedia Text Search ---
        logger.info("  [1/3] Trying Wikimedia text search for %s...", person.display_name)
        sources_tried.append("wikimedia_text")

        dl = Downloader(
            output_dir=self._output_dir,
            sources=[self._wikimedia_source],
            max_images_per_person=target,
            delay=self._wikimedia_delay,
            face_service=self._face_service,
        )

        t0 = time.time()
        records, rejections = dl.download_person(person)
        elapsed = time.time() - t0

        # Count NEW images (not previously downloaded)
        new_records = [r for r in records if r.source_url not in existing_urls]
        source_breakdown["wikimedia_text"] = len(new_records)
        all_new_rejections.extend(rejections)

        per_source_metrics["wikimedia_text"] = {
            "accepted": len(new_records),
            "rejected": len(rejections),
            "runtime_seconds": round(elapsed, 2),
            "rate_limit_errors": self._wikimedia_source.rate_limit_errors,
        }

        current_accepted = existing_count + len(new_records)
        logger.info(
            "  wikimedia_text: +%d accepted (total=%d/%d)",
            len(new_records), current_accepted, target,
        )

        # --- Source 2: Wikimedia Category (if needed) ---
        if current_accepted < target:
            logger.info("  [2/3] Trying Wikimedia category search for %s...", person.display_name)
            sources_tried.append("wikimedia_category")

            category = getattr(person, "wikimedia_category", None) or person.display_name
            t0 = time.time()

            candidates_iter = self._wikimedia_source.search_by_category(
                category, max_results=self._max_candidates
            )
            result = dl.download_candidates(
                person=person,
                candidates=candidates_iter,
                source_name="wikimedia_commons",
                max_candidates=self._max_candidates,
            )

            elapsed = time.time() - t0
            source_breakdown["wikimedia_category"] = result.accepted
            all_new_rejections.extend(result.rejection_details)

            per_source_metrics["wikimedia_category"] = {
                "candidates_discovered": result.candidates_discovered,
                "candidates_examined": result.candidates_examined,
                "accepted": result.accepted,
                "rejected": result.rejected,
                "acceptance_rate": result.acceptance_rate,
                "runtime_seconds": round(elapsed, 2),
                "rate_limit_errors": self._wikimedia_source.rate_limit_errors,
            }

            current_accepted += result.accepted
            logger.info(
                "  wikimedia_category: +%d accepted (total=%d/%d)",
                result.accepted, current_accepted, target,
            )

        # --- Source 3: Openverse (if still needed) ---
        if current_accepted < target:
            logger.info("  [3/3] Trying Openverse for %s...", person.display_name)
            sources_tried.append("openverse")

            t0 = time.time()

            candidates_iter = self._openverse_source.search(
                person.display_name, max_results=self._max_candidates
            )
            result = dl.download_candidates(
                person=person,
                candidates=candidates_iter,
                source_name="openverse",
                max_candidates=self._max_candidates,
            )

            elapsed = time.time() - t0
            source_breakdown["openverse"] = result.accepted
            all_new_rejections.extend(result.rejection_details)

            per_source_metrics["openverse"] = {
                "candidates_discovered": result.candidates_discovered,
                "candidates_examined": result.candidates_examined,
                "accepted": result.accepted,
                "rejected": result.rejected,
                "acceptance_rate": result.acceptance_rate,
                "runtime_seconds": round(elapsed, 2),
                "rate_limit_errors": self._openverse_source.rate_limit_errors,
            }

            current_accepted += result.accepted
            logger.info(
                "  openverse: +%d accepted (total=%d/%d)",
                result.accepted, current_accepted, target,
            )

        # Determine status
        if current_accepted >= target:
            status = "COMPLETED"
        elif current_accepted >= self._max_images * 0.5:
            status = "PARTIAL"
        elif current_accepted > 0:
            status = "INSUFFICIENT"
        else:
            status = "FAILED_ACQUISITION"

        # Collect final records from state
        final_records = self._load_person_records(person.person_id)

        self._all_records.extend(final_records)
        self._all_rejections.extend(all_new_rejections)

        return self._build_person_result(
            person, target, current_accepted, status,
            source_breakdown, per_source_metrics, all_new_rejections,
        )

    def collect_all(
        self,
        persons: list[Person],
        target_images: int | None = None,
    ) -> dict[str, Any]:
        """Collect images for all persons with source fallback.

        Returns complete collection summary.
        """
        results: dict[str, dict[str, Any]] = {}
        completed = 0
        partial = 0
        failed = 0

        for i, person in enumerate(persons):
            logger.info(
                "[%d/%d] Processing: %s (%s)",
                i + 1, len(persons), person.display_name, person.category,
            )

            person_result = self.collect_person(person, target_images)
            results[person.person_id] = person_result

            status = person_result["status"]
            if status == "COMPLETED":
                completed += 1
            elif status in ("PARTIAL", "INSUFFICIENT"):
                partial += 1
            else:
                failed += 1

            logger.info(
                "  %s: %s (accepted=%d/%d)",
                person.display_name, status,
                person_result["accepted"], person_result["target_images"],
            )

        # Close sources
        self._wikimedia_source.close()
        self._openverse_source.close()

        return {
            "total_persons": len(persons),
            "completed": completed,
            "partial": partial,
            "failed": failed,
            "total_accepted": sum(r["accepted"] for r in results.values()),
            "per_person": results,
        }

    def _load_person_records(self, person_id: str) -> list[ImageRecord]:
        """Load all records for a person from download state."""
        state_path = self._output_dir / "download_state.json"
        if not state_path.exists():
            return []

        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

        records_raw = state.get("records", [])
        return [
            ImageRecord.from_dict(r) for r in records_raw
            if r.get("person_id") == person_id
        ]

    def _build_person_result(
        self,
        person: Person,
        target: int,
        accepted: int,
        status: str,
        source_breakdown: dict[str, int],
        per_source_metrics: dict[str, dict[str, Any]],
        rejections: list[RejectionDetail] | None = None,
    ) -> dict[str, Any]:
        """Build per-person result dict."""
        rejections_by_reason: dict[str, int] = {}
        if rejections:
            for r in rejections:
                rejections_by_reason[r.rejection_reason] = (
                    rejections_by_reason.get(r.rejection_reason, 0) + 1
                )

        return {
            "person_id": person.person_id,
            "display_name": person.display_name,
            "category": person.category,
            "target_images": target,
            "accepted": accepted,
            "status": status,
            "sources_tried": list(per_source_metrics.keys()),
            "source_breakdown": source_breakdown,
            "per_source_metrics": per_source_metrics,
            "rejections_by_reason": rejections_by_reason,
        }
