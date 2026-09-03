# Phase 13.6.2c — Openverse + Wikimedia Category Discovery Report

## Objective

Evaluate two approaches for improving celebrity identity-labeled dataset acquisition:
1. **Wikimedia Category Discovery** — Structured category-based search
2. **Openverse API** — Aggregated multi-source search

Compare against existing Wikimedia text search from Phase 13.6.

## Status: COMPLETE

## Findings

### 1. Openverse API (DEFERRED)

**API Documentation Research (CONFIRMED)**:
- Base URL: `https://api.openverse.org/v1/`
- No authentication required for public search
- Rate limiting: 5 req/day unauthenticated, 60/hr with API key
- License filtering: `license=by,by-sa,cc0,pdm`
- Pagination: `page` param, 20 results per page
- Response includes: `image_url`, `creator`, `license`, `license_url`, `foreign_landing_url`, `source`, `identifier`

**Implementation**: `dataset_acquisition/sources/openverse.py`
- 29 offline tests pass
- Full ImageSource interface compliance
- Rate limiting with Retry-After header support
- License normalization (by → CC BY 4.0, etc.)

**Pilot Results**:
- Found 68 images across 6 persons
- Download blocked by 5 req/day unauthenticated throttle
- Different image sources than direct Wikimedia (aggregates Flickr, etc.)

**Verdict**: DEFERRED — Viable if API key obtained for 60 req/hr

### 2. Wikimedia Category Discovery (SECONDARY)

**Implementation**: Enhanced `WikimediaSource` with `search_by_category()` method
- Uses `generator=categorymembers` instead of `generator=search`
- 8 offline tests pass
- Backward compatible (existing `search()` unchanged)

**Pilot Results**:
- Category search returns curated results (files in category)
- Higher precision than text search (less noise)
- Same rate limiting as text search (429s during batch)

**Verdict**: SECONDARY — Use as complement to text search

### 3. Wikimedia Text Search (SELECTED)

**Status**: Proven in Phase 13.6 (86.7% Top-1 retrieval)
- Direct Wikimedia Commons access
- Rate limiting manageable with proper throttling
- License confirmed compatible

**Verdict**: SELECTED — Primary source

### 4. Getty Images (BLOCKED)

**Status**: BLOCKED_BY_LICENSING
- Section 3.11 of Content License Agreement prohibits ML/AI training use
- No change from Phase 13.6.2a

### 5. Flickr (BLOCKED)

**Status**: API_ACCESS_UNAVAILABLE
- API key required (FLICKR_API_KEY environment variable)
- No key available
- No change from Phase 13.6.2b

## Source Comparison Table

| Source | Verdict | Rate Limits | License | Auth Required | Implementation |
|--------|---------|-------------|---------|---------------|----------------|
| Wikimedia Text | **SELECTED** | 10 req/min | CC BY-SA etc. | No | `wikimedia.py` |
| Wikimedia Category | **SECONDARY** | Same | Same | No | `wikimedia.py` (enhanced) |
| Openverse API | **DEFERRED** | 5 req/day | Conditional | No | `openverse.py` (new) |
| Getty Images | **BLOCKED** | N/A | N/A | Yes | `getty.py` |
| Flickr | **BLOCKED** | 3600/hr | Conditional | Yes | `flickr.py` |

## Critical Questions

1. **Can Openverse be used as primary source?** NO — 5 req/day unauthenticated throttle makes it impractical for batch collection.

2. **Does Wikimedia category search improve precision?** YES — categories are curated by humans, reducing noise from text search.

3. **Are Openverse licenses compatible?** CONDITIONAL — depends on underlying source (Flickr CC photos OK, but requires per-image validation).

4. **Should we get an Openverse API key?** MAYBE — 60/hr with key would make it viable as secondary source.

5. **Is Getty still blocked?** YES — Section 3.11 prohibits ML/AI use. No change.

6. **Is Flickr still blocked?** YES — API key required, not available.

7. **Should category search replace text search?** NO — use as complement. Text search has broader coverage.

8. **What is the optimal source mix?** Wikimedia text (primary) + Wikimedia category (secondary) + Openverse (deferred until API key obtained).

9. **Should we proceed to Phase 13.6.3?** YES — Wikimedia text search is proven. Category and Openverse are enhancements.

## Implementation Artifacts

### New Files
- `dataset_acquisition/sources/openverse.py` — OpenverseSource class (245 lines)
- `tests/test_dataset_acquisition/test_openverse_source.py` — 29 tests
- `tests/test_dataset_acquisition/test_wikimedia_category.py` — 8 tests

### Modified Files
- `dataset_acquisition/sources/wikimedia.py` — Added `search_by_category()` method

### Output Files
- `outputs/phase13_6_2c/pilot_summary.json` — Pilot results and comparison
- `outputs/phase13_6_2c/source_comparison.md` — Source verdicts table
- `outputs/phase13_6_2c/candidate_details.jsonl` — Per-person results

## Test Results

- **Openverse source tests**: 29/29 passed
- **Wikimedia category tests**: 8/8 passed
- **All dataset acquisition tests**: 226/226 passed
- **Full regression**: 1429 passed, 27 failed (all pre-existing)

## Production Isolation

- **search/**: UNTOUCHED
- **services/**: UNTOUCHED
- **pipeline/**: UNTOUCHED (changes from Phase 13.5.1, not this phase)
- **dataset_acquisition/**: NEW package (all changes here)

## Next Steps

1. Consider obtaining Openverse API key for 60 req/hr access
2. Use Wikimedia category search as complement to text search
3. Proceed to Phase 13.6.3 with proven Wikimedia text search
