# Phase 13.6.1.2a — Rejection Telemetry & Reporting Fix

**Date**: 2026-09-01
**Status**: TELEMETRY_FIX_VALIDATED
**Previous Phase**: 13.6.1.2 (Single-Face Acquisition Gate)

---

## Objective

Fix misleading rejection reporting in the dataset acquisition pipeline. Phase 13.6.1.2 introduced a Single-Face Acquisition Gate that rejected images at download time, but the `no_face_rejected` field in the pilot output was inaccurate and the manifest/report lacked structured rejection reason tracking. This phase adds per-candidate rejection reason telemetry, fixes the misleading field, and produces accurate per-person and per-source rejection statistics.

## Previous Problem

1. **Misleading `no_face_rejected` field**: The pilot script tracked a `no_face_rejected` counter that did not capture all rejection reasons (representation, multi_face, download_error, etc.)
2. **No structured rejection telemetry**: Manifest and quality report had no rejection reason breakdown
3. **No per-person/per-source rejection statistics**: Only aggregate counts were available

## Implementation

### New Models (`dataset_acquisition/models.py`)

```python
REJECTION_REASONS = frozenset({
    "representation", "no_face", "multi_face",
    "download_error", "decode_error", "invalid_image",
    "duplicate", "other",
})

@dataclasses.dataclass(frozen=True)
class RejectionDetail:
    image_url: str
    source_url: str
    rejection_reason: str  # one of REJECTION_REASONS
    face_count: int | None = None
    message: str | None = None

@dataclasses.dataclass(frozen=True)
class RejectionStats:
    total_candidates: int
    accepted: int
    rejected_total: int
    rejections_by_reason: dict[str, int]
    per_person: dict[str, dict]
    per_source: dict[str, dict]
```

### Downloader Changes (`dataset_acquisition/downloader.py`)

- `download_person` now returns `tuple[list[ImageRecord], list[RejectionDetail]]`
- `new_rejection_details` contains only NEW rejections from the current run (excludes historical for resume idempotency)
- State persistence: `rejection_details` dict keyed by `person_id` alongside existing `rejected_urls`
- Backward-compatible: old state format without `rejection_details` is handled gracefully

### Manifest & Report Changes (`dataset_acquisition/manifest.py`)

- `compute_rejection_stats()` function computes `RejectionStats` from records + rejection details
- `generate_manifest()` accepts optional `rejection_details` → adds `acquisition_telemetry` section
- `generate_quality_report()` accepts optional `rejection_details` → adds "Acquisition Telemetry" section with:
  - Overall summary (total candidates, accepted, rejected)
  - Rejection reasons breakdown (all 8 reasons, even if 0)
  - Per-person rejection statistics table
  - Per-source rejection statistics table

### Pilot Script Changes (`phase13_6_1_2_gate_pilot.py`)

- Removed misleading `no_face_rejected` field
- Uses `compute_rejection_stats()` for structured telemetry
- Outputs per-person and per-source breakdowns

## Schema

### `RejectionDetail` (stored in state)

```json
{
  "image_url": "https://...",
  "source_url": "https://...",
  "rejection_reason": "representation",
  "face_count": null,
  "message": null
}
```

### State persistence

```json
{
  "rejected_urls": {"person_id": ["url1", "url2"]},
  "rejection_details": {
    "person_id": [
      {"image_url": "...", "source_url": "...", "rejection_reason": "representation"}
    ]
  }
}
```

### Manifest `acquisition_telemetry` section

```json
{
  "acquisition_telemetry": {
    "total_candidates": 31,
    "accepted": 15,
    "rejected_total": 16,
    "rejections_by_reason": {"representation": 12, "multi_face": 4},
    "per_person": {
      "tom_hanks": {"accepted": 5, "rejected_total": 4, "representation": 4},
      "scarlett_johansson": {"accepted": 5, "rejected_total": 0},
      "denzel_washington": {"accepted": 5, "rejected_total": 12, "representation": 8, "multi_face": 4}
    },
    "per_source": {
      "wikimedia_commons": {"candidates": 31, "accepted": 15, "rejected": 16, "representation": 12, "multi_face": 4}
    }
  }
}
```

## Resume Behavior

- `new_rejection_details` excludes historical rejections (only new rejections from current run)
- `state["rejection_details"][person_id]` accumulates across runs
- `state["rejected_urls"][person_id]` continues to track URLs for idempotent skipping
- On resume, loaded rejection details are merged with new ones before saving

## Pilot Results

**Pilot run**: `phase_13_6_1_2a_telemetry`
**Identities**: tom_hanks, scarlett_johansson, denzel_washington (3)
**Source**: Wikimedia Commons

| Identity | Candidates | Accepted | Rejected | Representation | Multi-Face | No-Face | Ref | Query |
|----------|-----------|----------|----------|----------------|------------|---------|-----|-------|
| tom_hanks | 9 | 5 | 4 | 4 | 0 | 0 | 3 | 2 |
| scarlett_johansson | 5 | 5 | 0 | 0 | 0 | 0 | 3 | 2 |
| denzel_washington | 17 | 5 | 12 | 8 | 4 | 0 | 3 | 2 |
| **Total** | **31** | **15** | **16** | **12** | **4** | **0** | **9** | **6** |

**Cross-person duplicates**: 0
**Cross-split leakage**: 0
**Per-source**: wikimedia_commons — 31 candidates, 15 accepted, 16 rejected

## Tests

### Dataset Acquisition Tests

**71/71 tests pass** (15 new `TestRejectionTelemetry` tests + 56 existing)

New test categories:
- `test_rejection_reason_representation` — representation rejection tracked
- `test_rejection_reason_no_face` — no_face rejection tracked
- `test_rejection_reason_multi_face` — multi_face rejection tracked
- `test_rejection_reason_download_error` — download_error rejection tracked
- `test_rejection_reason_duplicate` — duplicate rejection tracked
- `test_rejection_aggregation` — multiple reasons aggregated correctly
- `test_per_person_rejection_statistics` — per-person breakdown accurate
- `test_global_rejection_statistics` — global stats accurate
- `test_source_level_statistics` — per-source stats accurate
- `test_resume_skips_rejected_urls_with_details` — resume idempotency preserved
- `test_accepted_images_are_records` — accepted images in records list
- `test_rejected_images_are_not_records` — rejected images NOT in records list
- `test_manifest_includes_rejection_telemetry` — manifest has telemetry section
- `test_report_includes_rejection_telemetry` — report has telemetry section
- `test_no_misleading_no_face_rejected_field` — old misleading field absent
- `test_deterministic_statistics` — stats deterministic across calls

### Full Regression

**1317 passed, 27 failed** — all 27 failures are pre-existing (26 face_visibility_validator + 1 parser_qa). **Zero new failures** from telemetry changes.

## Production Isolation

Only files under `dataset_acquisition/` and `tests/test_dataset_acquisition/` were modified:
- `dataset_acquisition/models.py` — new RejectionDetail, RejectionStats, REJECTION_REASONS
- `dataset_acquisition/downloader.py` — tuple return, rejection_details persistence
- `dataset_acquisition/manifest.py` — compute_rejection_stats, telemetry sections
- `tests/test_dataset_acquisition/test_acquisition.py` — 15 new tests, existing tests adapted

No changes to `search/`, `services/`, `pipeline/`, or any production code.

## Verdict

**TELEMETRY_FIX_VALIDATED**

- Structured rejection telemetry working correctly
- Misleading `no_face_rejected` field removed
- Manifest and report include accurate per-person and per-source rejection statistics
- Resume idempotency preserved
- 71/71 dataset acquisition tests pass
- 0 new failures in full regression
- Production isolation confirmed
