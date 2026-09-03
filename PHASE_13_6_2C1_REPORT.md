# Phase 13.6.2c.1 — Fair Source Evaluation & Pilot Correction

## Verdict: PILOT_METRICS_CORRECTED

---

## Summary

This phase corrected the methodological flaws in the initial Phase 13.6.2c pilot where Approaches B and C didn't use the same validation gate as Approach A. All three image sources now pass through the identical `Single-Face Acquisition Gate` via the shared `download_candidates()` method.

**Key correction**: The `download_candidates()` method now returns `AcquisitionRunResult` with accurate, non-overlapping metric definitions:
- `candidates_discovered`: Number yielded by source iterator
- `candidates_examined`: Number presented to validation gate after skipping existing/rejected
- `candidates_skipped_existing`: Skipped because source_url already downloaded
- `candidates_skipped_rejected`: Skipped because source_url already rejected
- `accepted`: Passed full validation gate and saved
- `rejected`: Processed by gate and rejected
- `acceptance_rate = accepted / candidates_examined`

**Invariants verified**:
- `discovered >= examined + skipped_existing + skipped_rejected`
- `examined == accepted + rejected`

---

## Telemetry Corrections Applied

| Issue | Before | After |
|-------|--------|-------|
| `candidates_discovered` vs `examined` | Confused (same value) | Distinct: discovered counts iterator yields, examined counts gate entries |
| Streaming | Generators materialized with `list()` | Passed directly to `download_candidates()` |
| Per-person rate-limit | Cumulative `source.rate_limit_errors` | Delta: `after - before` per person |
| Total metrics runtime | Passed `0` as runtime | Real wall-clock elapsed time |
| Metric definitions | Implicit | Explicit in `AcquisitionRunResult` and `pilot_summary.json` |
| Shared gate | Duplicated validation logic | Extracted to `_validate_and_accept()` private helper |
| `download_candidates` return type | `tuple[list, list]` | `AcquisitionRunResult` dataclass |
| Openverse API | `MAX_PER_PAGE=50` (rejected by API) | `MAX_PER_PAGE=20` (anonymous limit) |
| Openverse fields | `identifier`/`image_url` (old API) | `id`/`url` (current API) |

---

## Pilot Results (6 persons, seed=42)

### Approach A: Wikimedia Text Search

| Metric | Value |
|--------|-------|
| Discovered | 57 |
| Examined | 57 |
| Accepted | 30 |
| Rejected | 27 |
| Acceptance Rate | **52.63%** |
| Runtime | 147.1s |
| Rate-limit Errors | 0 |

### Approach B: Wikimedia Category Discovery

| Metric | Value |
|--------|-------|
| Discovered | 51 |
| Examined | 50 |
| Accepted | 8 |
| Rejected | 42 |
| Acceptance Rate | **16.00%** |
| Runtime | 114.6s |
| Rate-limit Errors | 0 |

### Approach C: Openverse API

| Metric | Value |
|--------|-------|
| Discovered | 54 |
| Examined | 48 |
| Accepted | 30 |
| Rejected | 18 |
| Acceptance Rate | **62.50%** |
| Runtime | 755.5s |
| Rate-limit Errors | 5 |

### Comparison

| Approach | Acceptance Rate | Accepted | Runtime | Notes |
|----------|----------------|----------|---------|-------|
| A: Wikimedia Text | 52.63% | 30 | 147s | Fastest, good yield |
| B: Wikimedia Category | 16.00% | 8 | 115s | Low yield (many non-person images in categories) |
| C: Openverse | 62.50% | 30 | 756s | Highest rate, but severely rate-limited (429s) |

---

## Source Conclusions (Reaffirmed)

| Source | Verdict | Reason |
|--------|---------|--------|
| Wikimedia Commons (text) | **SUITABLE** | 52.63% acceptance, fast, reliable |
| Wikimedia Commons (category) | **MARGINAL** | 16% acceptance — categories contain many non-person images |
| Openverse | **MARGINAL** | 62.50% acceptance, but 5 req/day anonymous rate limit makes large-scale acquisition impractical |
| Getty Images | BLOCKED_BY_LICENSING | Confirmed in Phase 13.6.2a |
| Flickr | API_ACCESS_UNAVAILABLE | Confirmed in Phase 13.6.2b |

---

## Files Modified

### Core Changes
- `dataset_acquisition/models.py`: Added `AcquisitionRunResult` dataclass
- `dataset_acquisition/downloader.py`: Refactored `download_candidates()` to return `AcquisitionRunResult`, extracted `_validate_and_accept()` shared helper
- `dataset_acquisition/sources/openverse.py`: Fixed `MAX_PER_PAGE=20`, updated field names (`id`/`url`)
- `phase13_6_2c1_pilot.py`: Rewritten with corrected telemetry, streaming, rate-limit deltas

### Tests Added
- `tests/test_dataset_acquisition/test_acquisition.py`: 17 new telemetry tests (243 total, all pass)

### Artifacts
- `outputs/phase13_6_2c1/pilot_summary.json`: Full pilot results with corrected metrics

---

## Test Results

- Dataset acquisition tests: **243/243 passed**
- Full regression: **1446 passed, 27 pre-existing failures (26 face_visibility + 1 parser_qa), 0 new failures**

---

## Production Isolation

All changes are in `dataset_acquisition/` and `tests/test_dataset_acquisition/`. No modifications to `search/`, `services/`, or `pipeline/` (the `pipeline/photo_validation_pipeline.py` change is pre-existing from a prior phase).
