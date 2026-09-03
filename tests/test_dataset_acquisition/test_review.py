"""Tests for dataset_acquisition.review — manual review workflow.

Covers:
- ManualReviewRecord creation, validation, serialization
- ACCEPT/REJECT/UNCERTAIN/PENDING states
- Invalid review state
- Review reason validation
- Automated status preservation
- Manual override
- Per-person aggregation
- Quality gate (PASS/INSUFFICIENT_DATA/FAILED/PENDING)
- Pending-review behavior
- Insufficient-data behavior
- Deterministic output
- Contact sheet generation
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from PIL import Image

from dataset_acquisition.models import (
    ManualReviewRecord,
    ReviewStats,
    REVIEW_DECISIONS,
    REVIEW_REASONS,
)
from dataset_acquisition.review import (
    compute_review_stats,
    classify_identity_quality,
    create_review_records,
    update_review_record,
    generate_contact_sheet,
    generate_all_contact_sheets,
    save_review_records,
    save_review_stats,
)


def _make_review_record(
    image_id: str = "test_img_001",
    person_id: str = "test_person",
    display_name: str = "Test Person",
    automated_status: str = "valid",
    automated_image_category: str = "photograph",
    automated_identity_status: str = "confirmed",
    faces_detected: int = 1,
    face_confidence: float = 0.85,
    manual_decision: str = "PENDING",
    manual_reason: str = "",
    reviewer_notes: str = "",
) -> ManualReviewRecord:
    return ManualReviewRecord(
        person_id=person_id,
        display_name=display_name,
        image_id=image_id,
        image_path=f"fake/path/{image_id}.jpg",
        source="wikimedia_commons",
        source_url="https://example.com/image",
        automated_status=automated_status,
        automated_image_category=automated_image_category,
        automated_identity_status=automated_identity_status,
        faces_detected=faces_detected,
        face_confidence=face_confidence,
        manual_decision=manual_decision,
        manual_reason=manual_reason,
        reviewer_notes=reviewer_notes,
    )


class TestManualReviewRecord:
    def test_creation_pending(self) -> None:
        r = _make_review_record()
        assert r.manual_decision == "PENDING"
        assert r.person_id == "test_person"

    def test_creation_accept(self) -> None:
        r = _make_review_record(
            manual_decision="ACCEPT",
            manual_reason="CORRECT_IDENTITY",
        )
        assert r.manual_decision == "ACCEPT"

    def test_creation_reject(self) -> None:
        r = _make_review_record(
            manual_decision="REJECT",
            manual_reason="REPRESENTATION",
        )
        assert r.manual_decision == "REJECT"

    def test_creation_uncertain(self) -> None:
        r = _make_review_record(
            manual_decision="UNCERTAIN",
            manual_reason="INSUFFICIENT_CONTEXT",
        )
        assert r.manual_decision == "UNCERTAIN"

    def test_invalid_decision_raises(self) -> None:
        with pytest.raises(ValueError, match="manual_decision must be one of"):
            _make_review_record(manual_decision="INVALID")

    def test_invalid_reason_raises(self) -> None:
        with pytest.raises(ValueError, match="manual_reason must be one of"):
            _make_review_record(
                manual_decision="REJECT",
                manual_reason="NOT_A_REAL_REASON",
            )

    def test_empty_reason_is_valid(self) -> None:
        r = _make_review_record(manual_reason="")
        assert r.manual_reason == ""

    def test_to_dict_roundtrip(self) -> None:
        r = _make_review_record(
            manual_decision="ACCEPT",
            manual_reason="CORRECT_IDENTITY",
        )
        d = r.to_dict()
        r2 = ManualReviewRecord.from_dict(d)
        assert r == r2

    def test_to_dict_contains_all_fields(self) -> None:
        r = _make_review_record()
        d = r.to_dict()
        assert "person_id" in d
        assert "display_name" in d
        assert "image_id" in d
        assert "image_path" in d
        assert "source" in d
        assert "source_url" in d
        assert "automated_status" in d
        assert "automated_image_category" in d
        assert "automated_identity_status" in d
        assert "faces_detected" in d
        assert "face_confidence" in d
        assert "manual_decision" in d
        assert "manual_reason" in d
        assert "reviewer_notes" in d

    def test_automated_status_preserved(self) -> None:
        r = _make_review_record(
            automated_status="multi_face",
            automated_image_category="photograph",
            automated_identity_status="uncertain",
            manual_decision="ACCEPT",
            manual_reason="CORRECT_IDENTITY",
        )
        assert r.automated_status == "multi_face"
        assert r.automated_image_category == "photograph"
        assert r.automated_identity_status == "uncertain"

    def test_manual_override(self) -> None:
        r = _make_review_record(
            automated_status="valid",
            automated_identity_status="confirmed",
            manual_decision="REJECT",
            manual_reason="WRONG_IDENTITY",
            reviewer_notes="This is not the right person",
        )
        assert r.automated_status == "valid"
        assert r.manual_decision == "REJECT"
        assert r.manual_reason == "WRONG_IDENTITY"
        assert r.reviewer_notes == "This is not the right person"

    def test_all_valid_decisions(self) -> None:
        for decision in REVIEW_DECISIONS:
            r = _make_review_record(manual_decision=decision)
            assert r.manual_decision == decision

    def test_all_valid_reasons(self) -> None:
        for reason in REVIEW_REASONS:
            r = _make_review_record(
                manual_decision="REJECT",
                manual_reason=reason,
            )
            assert r.manual_reason == reason


class TestUpdateReviewRecord:
    def test_update_to_accept(self) -> None:
        records = [_make_review_record(image_id="img_001")]
        updated = update_review_record(records, "img_001", "ACCEPT", "CORRECT_IDENTITY")
        assert updated.manual_decision == "ACCEPT"
        assert updated.manual_reason == "CORRECT_IDENTITY"
        assert records[0].manual_decision == "ACCEPT"

    def test_update_to_reject(self) -> None:
        records = [_make_review_record(image_id="img_001")]
        updated = update_review_record(records, "img_001", "REJECT", "REPRESENTATION")
        assert updated.manual_decision == "REJECT"

    def test_update_to_uncertain(self) -> None:
        records = [_make_review_record(image_id="img_001")]
        updated = update_review_record(records, "img_001", "UNCERTAIN", "INSUFFICIENT_CONTEXT")
        assert updated.manual_decision == "UNCERTAIN"

    def test_update_not_found_raises(self) -> None:
        records = [_make_review_record(image_id="img_001")]
        with pytest.raises(KeyError, match="img_999"):
            update_review_record(records, "img_999", "ACCEPT")

    def test_update_invalid_decision_raises(self) -> None:
        records = [_make_review_record(image_id="img_001")]
        with pytest.raises(ValueError, match="Invalid decision"):
            update_review_record(records, "img_001", "BAD_DECISION")

    def test_update_preserves_other_fields(self) -> None:
        records = [_make_review_record(image_id="img_001", person_id="person_A")]
        updated = update_review_record(records, "img_001", "ACCEPT")
        assert updated.person_id == "person_A"
        assert updated.automated_status == "valid"

    def test_update_with_notes(self) -> None:
        records = [_make_review_record(image_id="img_001")]
        updated = update_review_record(
            records, "img_001", "REJECT", "LOW_QUALITY",
            notes="Image is too dark",
        )
        assert updated.reviewer_notes == "Image is too dark"

    def test_update_multiple_records(self) -> None:
        records = [
            _make_review_record(image_id="img_001"),
            _make_review_record(image_id="img_002"),
            _make_review_record(image_id="img_003"),
        ]
        update_review_record(records, "img_001", "ACCEPT")
        update_review_record(records, "img_002", "REJECT", "NO_FACE")
        update_review_record(records, "img_003", "UNCERTAIN")

        assert records[0].manual_decision == "ACCEPT"
        assert records[1].manual_decision == "REJECT"
        assert records[2].manual_decision == "UNCERTAIN"


class TestComputeReviewStats:
    def test_all_pending(self) -> None:
        records = [_make_review_record(image_id=f"img_{i:03d}") for i in range(5)]
        stats = compute_review_stats(records)
        assert stats.total_images == 5
        assert stats.pending == 5
        assert stats.accepted == 0
        assert stats.rejected == 0
        assert stats.uncertain == 0
        assert stats.acceptance_rate == 0.0

    def test_mixed_decisions(self) -> None:
        records = [
            _make_review_record(image_id="img_001", manual_decision="ACCEPT"),
            _make_review_record(image_id="img_002", manual_decision="ACCEPT"),
            _make_review_record(image_id="img_003", manual_decision="REJECT"),
            _make_review_record(image_id="img_004", manual_decision="UNCERTAIN"),
            _make_review_record(image_id="img_005", manual_decision="PENDING"),
        ]
        stats = compute_review_stats(records)
        assert stats.total_images == 5
        assert stats.pending == 1
        assert stats.accepted == 2
        assert stats.rejected == 1
        assert stats.uncertain == 1
        assert stats.acceptance_rate == pytest.approx(2 / 4)

    def test_empty_records(self) -> None:
        stats = compute_review_stats([])
        assert stats.total_images == 0
        assert stats.pending == 0
        assert stats.acceptance_rate == 0.0

    def test_per_person_aggregation(self) -> None:
        records = [
            _make_review_record(image_id="img_001", person_id="A", manual_decision="ACCEPT"),
            _make_review_record(image_id="img_002", person_id="A", manual_decision="ACCEPT"),
            _make_review_record(image_id="img_003", person_id="A", manual_decision="REJECT"),
            _make_review_record(image_id="img_004", person_id="B", manual_decision="UNCERTAIN"),
            _make_review_record(image_id="img_005", person_id="B", manual_decision="PENDING"),
        ]
        stats = compute_review_stats(records)
        assert "A" in stats.per_person
        assert "B" in stats.per_person
        assert stats.per_person["A"]["accepted"] == 2
        assert stats.per_person["A"]["rejected"] == 1
        assert stats.per_person["B"]["uncertain"] == 1
        assert stats.per_person["B"]["pending"] == 1

    def test_acceptance_rate_all_reviewed(self) -> None:
        records = [
            _make_review_record(image_id=f"img_{i:03d}", manual_decision="ACCEPT")
            for i in range(10)
        ]
        stats = compute_review_stats(records)
        assert stats.acceptance_rate == pytest.approx(1.0)

    def test_stats_to_dict(self) -> None:
        records = [_make_review_record(manual_decision="ACCEPT")]
        stats = compute_review_stats(records)
        d = stats.to_dict()
        assert "total_images" in d
        assert "pending" in d
        assert "accepted" in d
        assert "rejected" in d
        assert "uncertain" in d
        assert "acceptance_rate" in d
        assert "per_person" in d


class TestClassifyIdentityQuality:
    def test_pass(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 4, "rejected": 1, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "PASS"

    def test_insufficient_data(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 2, "rejected": 3, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "INSUFFICIENT_DATA"

    def test_failed(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 0, "rejected": 5, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "FAILED"

    def test_pending_overrides(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 2, "accepted": 3, "rejected": 0, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "PENDING"

    def test_exact_threshold(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 3, "rejected": 2, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "PASS"

    def test_below_threshold(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 2, "rejected": 3, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "INSUFFICIENT_DATA"

    def test_multiple_persons(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 5, "rejected": 0, "uncertain": 0},
            "B": {"total": 5, "pending": 1, "accepted": 0, "rejected": 0, "uncertain": 0},
            "C": {"total": 5, "pending": 0, "accepted": 1, "rejected": 4, "uncertain": 0},
            "D": {"total": 5, "pending": 0, "accepted": 0, "rejected": 0, "uncertain": 0},
        }
        result = classify_identity_quality(per_person, min_accepted=3)
        assert result["A"] == "PASS"
        assert result["B"] == "PENDING"
        assert result["C"] == "INSUFFICIENT_DATA"
        assert result["D"] == "FAILED"


class TestCreateReviewRecords:
    def test_creates_records_from_jsonl(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "pilot.jsonl"
        records = [
            {
                "image_id": "img_001",
                "person_id": "person_a",
                "source": "wikimedia_commons",
                "source_url": "https://example.com/1",
                "local_path": "/fake/path/1.jpg",
                "faces_detected": 1,
                "face_confidence": 0.9,
                "status": "valid",
                "image_category": "photograph",
                "identity_status": "confirmed",
            },
        ]
        with open(jsonl, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        people = tmp_path / "people.json"
        with open(people, "w") as f:
            json.dump({"people": [{"person_id": "person_a", "display_name": "Person A"}]}, f)

        result = create_review_records(jsonl, people)
        assert len(result) == 1
        assert result[0].person_id == "person_a"
        assert result[0].display_name == "Person A"
        assert result[0].manual_decision == "PENDING"

    def test_multiple_persons(self, tmp_path: Path) -> None:
        jsonl = tmp_path / "pilot.jsonl"
        lines = []
        for i in range(6):
            pid = f"person_{i % 3}"
            lines.append(json.dumps({
                "image_id": f"img_{i:03d}",
                "person_id": pid,
                "source": "wikimedia_commons",
                "source_url": f"https://example.com/{i}",
                "local_path": f"/fake/{i}.jpg",
                "faces_detected": 1,
                "face_confidence": 0.8,
                "status": "valid",
                "image_category": "photograph",
                "identity_status": "confirmed",
            }))
        with open(jsonl, "w") as f:
            f.write("\n".join(lines) + "\n")

        people = tmp_path / "people.json"
        with open(people, "w") as f:
            json.dump({"people": [
                {"person_id": "person_0", "display_name": "Person 0"},
                {"person_id": "person_1", "display_name": "Person 1"},
                {"person_id": "person_2", "display_name": "Person 2"},
            ]}, f)

        result = create_review_records(jsonl, people)
        assert len(result) == 6
        person_ids = {r.person_id for r in result}
        assert person_ids == {"person_0", "person_1", "person_2"}


class TestContactSheet:
    def test_generates_jpeg(self, tmp_path: Path) -> None:
        images = []
        for i in range(3):
            img = Image.new("RGB", (100, 100), color=(i * 80, 100, 200))
            path = tmp_path / f"img_{i}.jpg"
            img.save(str(path), "JPEG")
            images.append((f"img_{i}", str(path), f"Label {i}"))

        out = tmp_path / "sheet.jpg"
        result = generate_contact_sheet(images, out, title="Test Sheet")
        assert result.exists()
        assert result.suffix == ".jpg"

    def test_empty_images(self, tmp_path: Path) -> None:
        out = tmp_path / "empty_sheet.jpg"
        result = generate_contact_sheet([], out)
        assert result.exists()

    def test_missing_image_file(self, tmp_path: Path) -> None:
        images = [("bad_img", str(tmp_path / "nonexistent.jpg"), "Bad")]
        out = tmp_path / "sheet_with_bad.jpg"
        result = generate_contact_sheet(images, out)
        assert result.exists()


class TestSaveReviewRecords:
    def test_saves_json_and_jsonl(self, tmp_path: Path) -> None:
        records = [
            _make_review_record(image_id="img_001"),
            _make_review_record(image_id="img_002"),
        ]
        json_path, jsonl_path = save_review_records(records, tmp_path)
        assert json_path.exists()
        assert jsonl_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert len(data) == 2

        with open(jsonl_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2


class TestSaveReviewStats:
    def test_saves_stats(self, tmp_path: Path) -> None:
        records = [
            _make_review_record(image_id="img_001", manual_decision="ACCEPT"),
            _make_review_record(image_id="img_002", manual_decision="PENDING"),
        ]
        stats = compute_review_stats(records)
        identity_quality = classify_identity_quality(stats.per_person)
        path = save_review_stats(stats, identity_quality, tmp_path)
        assert path.exists()

        with open(path) as f:
            data = json.load(f)
        assert "aggregate" in data
        assert "identity_quality" in data


class TestDeterministicOutput:
    def test_same_input_same_output(self, tmp_path: Path) -> None:
        records = [
            _make_review_record(image_id="img_001", person_id="A"),
            _make_review_record(image_id="img_002", person_id="A"),
            _make_review_record(image_id="img_003", person_id="B"),
        ]
        stats1 = compute_review_stats(records)
        stats2 = compute_review_stats(records)
        assert stats1.to_dict() == stats2.to_dict()

    def test_quality_gate_deterministic(self) -> None:
        per_person = {
            "A": {"total": 5, "pending": 0, "accepted": 4, "rejected": 1, "uncertain": 0},
            "B": {"total": 5, "pending": 2, "accepted": 3, "rejected": 0, "uncertain": 0},
        }
        result1 = classify_identity_quality(per_person, min_accepted=3)
        result2 = classify_identity_quality(per_person, min_accepted=3)
        assert result1 == result2
