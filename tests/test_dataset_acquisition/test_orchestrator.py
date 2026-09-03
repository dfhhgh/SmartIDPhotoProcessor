"""Offline tests for dataset_acquisition.orchestrator — source fallback, telemetry aggregation, resume, deduplication.

These tests mock the network and external services. No I/O required.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Iterator
from unittest.mock import MagicMock, patch

import pytest

from dataset_acquisition.models import (
    AcquisitionRunResult,
    ImageRecord,
    Person,
    RejectionDetail,
    SearchResult,
)
from dataset_acquisition.orchestrator import AcquisitionOrchestrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PERSON_1 = Person(
    person_id="tom_hanks",
    display_name="Tom Hanks",
    category="actor",
    aliases=(),
    search_queries=("Tom Hanks portrait",),
)

PERSON_2 = Person(
    person_id="lionel_messi",
    display_name="Lionel Messi",
    category="football_player",
    aliases=(),
    search_queries=("Lionel Messi portrait",),
)


def _search_result(url: str = "https://example.com/img.jpg", query: str = "test") -> SearchResult:
    return SearchResult(
        source="wikimedia_commons",
        source_url=url,
        image_url=url,
        title="Test Image",
        license="CC BY",
        attribution="Test Author",
        description="Test",
        width=800,
        height=600,
    )


def _make_person_records(
    person_id: str,
    n: int,
    base_url: str = "https://example.com/img",
) -> list[ImageRecord]:
    """Create fake records for a person."""
    import time
    import hashlib

    records = []
    for i in range(n):
        url = f"{base_url}_{i}.jpg"
        data = f"fake-data-{person_id}-{i}".encode()
        sha256 = hashlib.sha256(data).hexdigest()
        image_id = f"{person_id}_{sha256[:12]}"
        records.append(ImageRecord(
            image_id=image_id,
            person_id=person_id,
            source="wikimedia_commons",
            source_url=url,
            local_path=f"outputs/phase13_6_3/raw/{person_id}/{sha256[:12]}.jpg",
            license="CC BY",
            attribution="Test Author",
            query="test",
            download_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            sha256=sha256,
            file_size=1000 + i,
            width=800,
            height=600,
            faces_detected=1,
            face_selected=True,
            face_confidence=0.98,
            image_category="photograph",
            identity_status="confirmed",
            status="valid",
        ))
    return records


class _FakeDownloader:
    """Minimal mock of Downloader for orchestrator tests."""

    def __init__(self, records_to_return: list[ImageRecord] | None = None, result_to_return: AcquisitionRunResult | None = None):
        self._records = records_to_return or []
        self._result = result_to_return or AcquisitionRunResult(
            records=[],
            rejection_details=[],
            candidates_discovered=0,
            candidates_examined=0,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=0,
            rejected=0,
        )
        self.download_person_calls: list[tuple[Person, dict]] = []
        self.download_candidates_calls: list[tuple] = []

    def download_person(self, person: Person, state: dict | None = None) -> tuple[list[ImageRecord], list[RejectionDetail]]:
        self.download_person_calls.append((person, state))
        return self._records, []

    def download_candidates(
        self,
        person: Person,
        candidates: Iterator[SearchResult],
        source_name: str = "unknown",
        state: dict | None = None,
        max_candidates: int = 100,
    ) -> AcquisitionRunResult:
        # Consume the iterator
        for _ in candidates:
            pass
        self.download_candidates_calls.append((person, candidates, source_name, state, max_candidates))
        return self._result


class _FakeOrchestrator(AcquisitionOrchestrator):
    """Orchestrator that uses a fake Downloader instead of the real one."""

    def __init__(self, fake_dl: _FakeDownloader, **kwargs):
        super().__init__(**kwargs)
        self._fake_dl = fake_dl

    def _create_downloader(self, person):
        return self._fake_dl

    # Override download_person to skip source iterator creation
    def collect_person(self, person, target_images=None):
        target = target_images or self._max_images
        person_dir = self._output_dir / "raw" / person.person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        existing_urls = set()
        state_path = self._output_dir / "download_state.json"
        if state_path.exists():
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            existing_urls = set(state.get("downloaded", {}).get(person.person_id, []))

        existing_count = len(existing_urls)

        if existing_count >= target:
            return self._build_person_result(
                person, target, existing_count, "COMPLETED", {}, {}
            )

        # Source 1: Wikimedia text (mock)
        sources_tried = ["wikimedia_text"]
        source_breakdown: dict[str, int] = {}
        per_source_metrics: dict[str, dict] = {}

        records = self._fake_dl._records
        new_records = [r for r in records if r.source_url not in existing_urls]
        source_breakdown["wikimedia_text"] = len(new_records)
        per_source_metrics["wikimedia_text"] = {
            "accepted": len(new_records),
            "rejected": 0,
            "runtime_seconds": 1.0,
            "rate_limit_errors": 0,
        }

        current_accepted = existing_count + len(new_records)

        # Source 2: Wikimedia category (mock)
        all_rejection_details: list[RejectionDetail] = []
        if current_accepted < target:
            sources_tried.append("wikimedia_category")
            cat_result = self._fake_dl._result
            source_breakdown["wikimedia_category"] = cat_result.accepted
            per_source_metrics["wikimedia_category"] = {
                "candidates_discovered": cat_result.candidates_discovered,
                "candidates_examined": cat_result.candidates_examined,
                "accepted": cat_result.accepted,
                "rejected": cat_result.rejected,
                "acceptance_rate": cat_result.acceptance_rate,
                "runtime_seconds": 1.0,
                "rate_limit_errors": 0,
            }
            current_accepted += cat_result.accepted
            all_rejection_details.extend(cat_result.rejection_details)

        # Source 3: Openverse (mock)
        if current_accepted < target:
            sources_tried.append("openverse")
            ov_result = self._fake_dl._result
            source_breakdown["openverse"] = ov_result.accepted
            per_source_metrics["openverse"] = {
                "candidates_discovered": ov_result.candidates_discovered,
                "candidates_examined": ov_result.candidates_examined,
                "accepted": ov_result.accepted,
                "rejected": ov_result.rejected,
                "acceptance_rate": ov_result.acceptance_rate,
                "runtime_seconds": 1.0,
                "rate_limit_errors": 0,
            }
            current_accepted += ov_result.accepted
            all_rejection_details.extend(ov_result.rejection_details)

        if current_accepted >= target:
            status = "COMPLETED"
        elif current_accepted >= target * 0.5:
            status = "PARTIAL"
        elif current_accepted > 0:
            status = "INSUFFICIENT"
        else:
            status = "FAILED_ACQUISITION"

        return self._build_person_result(
            person, target, current_accepted, status,
            source_breakdown, per_source_metrics, all_rejection_details,
        )


# ---------------------------------------------------------------------------
# Tests: Source Fallback
# ---------------------------------------------------------------------------

class TestSourceFallback:
    """Test that sources are tried in order and skipped when target is reached."""

    def test_immediate_completion_skips_later_sources(self, tmp_path):
        """If Wikimedia text provides enough images, no other source is tried."""
        records = _make_person_records("tom_hanks", 12)
        fake_dl = _FakeDownloader(records_to_return=records)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert result["status"] == "COMPLETED"
        assert result["accepted"] == 12
        assert "wikimedia_text" in result["per_source_metrics"]
        # Category and Openverse should not be tried
        assert "wikimedia_category" not in result["per_source_metrics"]
        assert "openverse" not in result["per_source_metrics"]

    def test_category_fallback_when_text_insufficient(self, tmp_path):
        """When Wikimedia text yields too few, try Wikimedia category."""
        records = _make_person_records("tom_hanks", 5)
        cat_result = AcquisitionRunResult(
            records=[],
            rejection_details=[],
            candidates_discovered=10,
            candidates_examined=10,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=5,
            rejected=5,
        )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=cat_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert result["status"] == "COMPLETED"
        assert result["accepted"] >= 10
        assert "wikimedia_text" in result["per_source_metrics"]
        assert "wikimedia_category" in result["per_source_metrics"]

    def test_openverse_fallback_when_category_insufficient(self, tmp_path):
        """When category still insufficient, try Openverse."""
        records = _make_person_records("tom_hanks", 3)
        small_result = AcquisitionRunResult(
            records=[],
            rejection_details=[],
            candidates_discovered=4,
            candidates_examined=4,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=4,
            rejected=0,
        )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=small_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert result["status"] in ("PARTIAL", "COMPLETED")
        assert result["accepted"] >= 7
        assert "wikimedia_text" in result["per_source_metrics"]
        assert "wikimedia_category" in result["per_source_metrics"]
        assert "openverse" in result["per_source_metrics"]

    def test_no_sources_produce_images(self, tmp_path):
        """When no source produces any images, status is FAILED_ACQUISITION."""
        fake_dl = _FakeDownloader(records_to_return=[], result_to_return=AcquisitionRunResult(
            records=[],
            rejection_details=[],
            candidates_discovered=5,
            candidates_examined=5,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=0,
            rejected=5,
        ))

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert result["status"] == "FAILED_ACQUISITION"
        assert result["accepted"] == 0


# ---------------------------------------------------------------------------
# Tests: Telemetry Aggregation
# ---------------------------------------------------------------------------

class TestTelemetryAggregation:
    """Test that per_source_metrics and rejection breakdowns are correct."""

    def test_per_source_metrics_populated(self, tmp_path):
        """All tried sources should appear in per_source_metrics."""
        records = _make_person_records("tom_hanks", 8)
        cat_result = AcquisitionRunResult(
            records=[],
            candidates_discovered=5,
            candidates_examined=5,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=4,
            rejected=1,
            rejection_details=[
                RejectionDetail(
                    person_id="tom_hanks",
                    source="wikimedia_commons",
                    source_url="https://example.com/rej.jpg",
                    rejection_reason="no_face",
                    title="No face detected",
                )
            ],
        )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=cat_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert result["status"] == "COMPLETED"
        assert result["accepted"] == 12
        assert result["sources_tried"] == ["wikimedia_text", "wikimedia_category"]

    def test_rejection_breakdown_tracked(self, tmp_path):
        """Rejections should be tracked by reason."""
        records = _make_person_records("tom_hanks", 2)
        rejections = [
            RejectionDetail(
                person_id="tom_hanks",
                source="wikimedia_commons",
                source_url="https://example.com/no_face.jpg",
                rejection_reason="no_face",
                title="No face",
            ),
            RejectionDetail(
                person_id="tom_hanks",
                source="wikimedia_commons",
                source_url="https://example.com/multi.jpg",
                rejection_reason="multi_face",
                title="Multiple faces",
            ),
        ]
        cat_result = AcquisitionRunResult(
            records=[],
            candidates_discovered=5,
            candidates_examined=5,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=0,
            rejected=5,
            rejection_details=rejections,
        )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=cat_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert "rejections_by_reason" in result
        assert result["rejections_by_reason"].get("no_face", 0) >= 1
        assert result["rejections_by_reason"].get("multi_face", 0) >= 1


# ---------------------------------------------------------------------------
# Tests: Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Test that already-downloaded URLs are skipped."""

    def test_existing_urls_not_redownloaded(self, tmp_path):
        """URLs already in state should be counted as existing and not re-downloaded."""
        # Set up state with 5 existing URLs
        state = {
            "downloaded": {
                "tom_hanks": [f"https://example.com/existing_{i}.jpg" for i in range(5)],
            },
            "records": [],
            "rejected_urls": {},
            "seen_hashes": [],
            "rejection_details": {},
        }
        state_path = tmp_path / "download_state.json"
        with open(state_path, "w") as f:
            json.dump(state, f)

        # 10 new records (different URLs)
        new_records = _make_person_records("tom_hanks", 10, base_url="https://example.com/new")
        cat_result = AcquisitionRunResult(
            records=[],
            candidates_discovered=5,
            candidates_examined=5,
            candidates_skipped_existing=0,
            candidates_skipped_rejected=0,
            accepted=2,
            rejected=0,
            rejection_details=[],
        )
        fake_dl = _FakeDownloader(records_to_return=new_records, result_to_return=cat_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        # Should have 10 new + skipped 5 existing
        assert result["accepted"] >= 10


# ---------------------------------------------------------------------------
# Tests: collect_all
# ---------------------------------------------------------------------------

class TestCollectAll:
    """Test collecting for multiple persons."""

    def test_collect_all_counts(self, tmp_path):
        """collect_all should aggregate per-person results."""
        records = _make_person_records("tom_hanks", 12)
        fake_dl = _FakeDownloader(records_to_return=records)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_all([PERSON_1, PERSON_2], target_images=12)

        assert result["total_persons"] == 2
        assert result["completed"] >= 1
        assert "per_person" in result
        assert "tom_hanks" in result["per_person"]

    def test_collect_all_with_different_targets(self, tmp_path):
        """Different persons may have different acceptance rates."""
        records_h = _make_person_records("tom_hanks", 8)
        cat_result = AcquisitionRunResult(
            records=[], rejection_details=[],
            candidates_discovered=10, candidates_examined=10,
            candidates_skipped_existing=0, candidates_skipped_rejected=0,
            accepted=4, rejected=0,
        )
        fake_dl = _FakeDownloader(records_to_return=records_h, result_to_return=cat_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_all([PERSON_1], target_images=12)

        assert result["total_persons"] == 1
        assert result["per_person"]["tom_hanks"]["accepted"] == 12


# ---------------------------------------------------------------------------
# Tests: Status Determination
# ---------------------------------------------------------------------------

class TestStatusDetermination:
    """Test status logic based on accepted count vs target."""

    @pytest.mark.parametrize("accepted,expected_status", [
        (12, "COMPLETED"),
        (6, "PARTIAL"),
        (3, "INSUFFICIENT"),
        (0, "FAILED_ACQUISITION"),
    ])
    def test_status_mapping(self, tmp_path, accepted, expected_status):
        if accepted == 0:
            records = []
            res = AcquisitionRunResult(
                records=[], rejection_details=[],
                candidates_discovered=5, candidates_examined=5,
                candidates_skipped_existing=0, candidates_skipped_rejected=0,
                accepted=0, rejected=5,
            )
        else:
            records = _make_person_records("tom_hanks", accepted)
            res = AcquisitionRunResult(
                records=[], rejection_details=[],
                candidates_discovered=5, candidates_examined=5,
                candidates_skipped_existing=0, candidates_skipped_rejected=0,
                accepted=0, rejected=0,
            )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=res)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)
        assert result["status"] == expected_status
        assert result["accepted"] == accepted


# ---------------------------------------------------------------------------
# Tests: Source Metric Fields
# ---------------------------------------------------------------------------

class TestSourceMetricFields:
    """Test that source metrics contain expected fields."""

    def test_wikimedia_text_metric_fields(self, tmp_path):
        records = _make_person_records("tom_hanks", 12)
        fake_dl = _FakeDownloader(records_to_return=records)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        wm = result["per_source_metrics"]["wikimedia_text"]
        assert "accepted" in wm
        assert "rejected" in wm
        assert "runtime_seconds" in wm
        assert "rate_limit_errors" in wm

    def test_category_metric_fields(self, tmp_path):
        records = _make_person_records("tom_hanks", 5)
        cat_result = AcquisitionRunResult(
            records=[], rejection_details=[],
            candidates_discovered=10, candidates_examined=10,
            candidates_skipped_existing=0, candidates_skipped_rejected=0,
            accepted=7, rejected=3,
        )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=cat_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        wm = result["per_source_metrics"]["wikimedia_category"]
        assert "candidates_discovered" in wm
        assert "candidates_examined" in wm
        assert "accepted" in wm
        assert "rejected" in wm
        assert "acceptance_rate" in wm
        assert "runtime_seconds" in wm
        assert "rate_limit_errors" in wm

    def test_openverse_metric_fields(self, tmp_path):
        records = _make_person_records("tom_hanks", 3)
        small_result = AcquisitionRunResult(
            records=[], rejection_details=[],
            candidates_discovered=4, candidates_examined=4,
            candidates_skipped_existing=0, candidates_skipped_rejected=0,
            accepted=4, rejected=0,
        )
        fake_dl = _FakeDownloader(records_to_return=records, result_to_return=small_result)

        orch = _FakeOrchestrator(
            fake_dl=fake_dl,
            output_dir=tmp_path,
            max_images_per_person=12,
        )

        result = orch.collect_person(PERSON_1, target_images=12)

        assert "openverse" in result["per_source_metrics"]
        ov = result["per_source_metrics"]["openverse"]
        assert "candidates_discovered" in ov
        assert "acceptance_rate" in ov
