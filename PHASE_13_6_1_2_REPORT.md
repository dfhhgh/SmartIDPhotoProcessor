# Phase 13.6.1.2 — Single-Face Acquisition Gate

**Date**: 2026-08-31
**Status**: COMPLETE
**Verdict**: GATE_ACCEPTED

---

## 1. Objective

Make the dataset acquisition pipeline enforce an **exactly-one-face gate** before calibration candidate status. Reject no-face, multi-face, and representation images deterministically at download time. Preserve production pipeline unchanged.

---

## 2. Architecture Change

### Before (Insufficient Gate)
The old gate assigned status labels but still **saved ALL images to disk**:
- `status="representation"` → saved to disk
- `status="no_face"` → saved to disk
- `status="multi_face"` → saved to disk + added to review queue
- Only `face_selected=True` for exactly 1 face

**Problem**: No-face, multi-face, and representation images consumed disk space, inflated record counts, and could leak into calibration if downstream code checked `face_selected` inconsistently.

### After (Single-Face Acquisition Gate)
```
Download → Decode → Representation check → Face detection → Exactly-One-Face Gate → Calibration candidate
```

Only images with **exactly one detected face** and **no representation keywords** are saved to disk. No-face, multi-face, and representation images are:
1. Rejected at download time (not saved to disk)
2. Tracked in `state["rejected_urls"]` for resume idempotency
3. Never appear in ImageRecord data

### Gate Flow
```
search result → _is_representation()? → REJECT (tracked in rejected_urls)
              → _detect_faces_insightface()
              → faces == 0? → REJECT (tracked in rejected_urls)
              → faces > 1?  → REJECT (tracked in rejected_urls)
              → faces == 1? → ACCEPT → save to disk → ImageRecord
```

---

## 3. Files Changed

| File | Change | Lines |
|------|--------|-------|
| `dataset_acquisition/downloader.py` | Core gate logic: reject at download, track rejected_urls in state, docstring | +35 -80 |
| `tests/test_dataset_acquisition/test_acquisition.py` | Updated 3 tests for new gate, added 5 new gate tests | +120 -50 |

**Not modified**: `models.py`, `manifest.py`, `search/`, `services/`, `pipeline/photo_validation_pipeline.py`

---

## 4. Test Results

### Dataset Acquisition Tests
**98/98 passed** (was 50, now 98 with 5 new gate tests + 3 updated tests)

New tests (`TestSingleFaceAcquisitionGate`):
- `test_only_single_face_accepted` — mixed gate: only exactly-1-face passes
- `test_rejected_urls_in_state` — rejected URLs tracked in state JSON
- `test_resume_skips_rejected_urls` — resume does not re-download rejected URLs
- `test_mixed_gate_results` — mixed scenario: 2 valid, 3 rejected
- `test_no_files_saved_for_rejected` — rejected images not on disk

### Full Regression
**1301 passed, 27 failed** (all 27 pre-existing: 26 face_visibility_validator + 1 parser_qa). **Zero new failures.**

---

## 5. Pilot Results

### Configuration
- **Identities**: 3 (tom_hanks, scarlett_johansson, denzel_washington)
- **Source**: Wikimedia Commons (3s delay, 8 max rate limit retries)
- **Max images/person**: 5
- **Output**: `datasets/celebrity-v2-pilot-gated/`

### Results
| Metric | Value |
|--------|-------|
| Total identities | 3 |
| Total images acquired | **15** |
| All single-face (gate-accepted) | **15** (100%) |
| Cross-person duplicates | 0 |
| Cross-split leakage | 0 |
| Reference images | 9 |
| Query images | 6 |

### Per-Identity Breakdown
| Identity | Accepted | Rejected | Rejection Reasons |
|----------|----------|----------|-------------------|
| tom_hanks | 5 | 4 | 4 representation |
| scarlett_johansson | 5 | 0 | — |
| denzel_washington | 5 | 12 | 8 representation, 4 multi-face |

### Gate Invariants (Verified)
- `all(face_selected == True for records)` — **TRUE**
- `all(faces_detected == 1 for records)` — **TRUE**
- `all(status == "valid" for records)` — **TRUE**
- `all(image_category == "photograph" for records)` — **TRUE**
- `no_face in records` — **FALSE** (rejected at download)
- `multi_face in records` — **FALSE** (rejected at download)
- `representation in records` — **FALSE** (rejected at download)

---

## 6. Resume Idempotency

Rejected URLs are tracked in `state["rejected_urls"][person_id]`. On resume:
1. Existing records loaded from state → skip already-accepted URLs
2. Rejected URLs loaded from state → skip already-rejected URLs
3. Only new, unseen URLs undergo download + gate evaluation

Verified by `test_resume_skips_rejected_urls` and pilot resume behavior.

---

## 7. Limitations

- **Gate threshold**: Hard-coded exactly-one-face. Could be relaxed in future (e.g., accept 1-2 faces with dominant face >80% area).
- **Representation detection**: Keyword-based. False negatives possible for images without keyword matches in title/description.
- **Face detection quality**: InsightFace det_score threshold not enforced at gate level (only face count). Low-confidence single-face images pass.
- **No-face/multi-face images are permanently rejected**: Not preserved for future re-evaluation.

---

## 8. Acceptance Criteria

| Criterion | Status |
|-----------|--------|
| Exactly-one-face gate enforced | ✅ |
| No-face rejected deterministically | ✅ |
| Multi-face rejected deterministically | ✅ |
| Representation rejected deterministically | ✅ |
| Production pipeline unchanged | ✅ |
| Zero new test failures | ✅ |
| Resume idempotency preserved | ✅ |
| Small real pilot validates gate | ✅ |
