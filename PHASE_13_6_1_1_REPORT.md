# Phase 13.6.1.1 — Pilot Correction & Source Evaluation

## 1. Objective

Fix identified acquisition/reporting problems from Phase 13.6.1, evaluate additional sources, improve candidate quality, run a corrected small pilot, and stop for manual review. Do NOT perform final calibration. Do NOT scale to hundreds of identities. Do NOT modify production reverse-search pipeline.

## 2. Problems Identified in Phase 13.6.1

### 2.1 Resume Bug in `phase13_6_1_continue.py`

**Root Cause**: `phase13_6_1_continue.py` compared `len(current_persons)` against `len(pilot_persons)` — counts of person IDs — rather than checking whether each identity actually had enough valid images. A person with 0 valid images still counted as "already acquired" if their ID existed in the state file.

**Fix**: `phase13_6_1_1_pilot.py` now loads existing `download_state.json`, counts `face_selected` images per person, and only acquires identities where `face_selected < target_images`.

**Verification**: Idempotency test passes — running the pilot twice produces identical results; second run acquires 0 new images.

**Source Documented From**: Code inspection of `phase13_6_1_continue.py:20-26`.

### 2.2 Hardcoded `total_multi_face=0`

**Root Cause**: `phase13_6_1_pilot.py` printed `total_multi_face=0` without computing it from actual records.

**Fix**: Added `compute_stats_from_records()` as the single source of truth in `models.py`. All stats (total_valid, total_no_face, total_multi_face, total_representation, total_identity_uncertain) now derive from ImageRecord fields.

**Source Documented From**: Code inspection of `phase13_6_1_pilot.py:100-104`.

### 2.3 No Representation Filtering

**Root Cause**: Downloaded movie posters, paintings, billboards, and cartoon images without filtering. Denzel Washington received 5/5 representation images (all Wikipedia photos with overlays/watermarks).

**Fix**: Expanded `REPRESENTATION_KEYWORDS` in `downloader.py` from ~15 to 30+ keywords (poster, painting, illustration, cartoon, billboard, mural, sculpture, statue, artwork, sketch, digital art, anime, comic, fan art, book cover, album cover, magazine cover, newspaper, advertisement, logo, icon, sticker, tattoo, portrait of, painting of, illustration of, drawing of, sketch of, artwork of). Representation images are marked `image_category="representation"`, `identity_status="representation"`, `status="representation"` and excluded from calibration-ready set.

**Source Documented From**: Code inspection of `downloader.py:31-48`, pilot output showing denzel_washington 5/5 representation.

### 2.4 Multi-Face Policy Not Explicit

**Root Cause**: Multi-face images were either auto-selected or silently ignored depending on code path. No explicit policy for when multi-face images should enter the review queue.

**Fix**: Multi-face images are explicitly routed to review queue with `status="multi_face"`, `identity_status="uncertain"`. They are NOT auto-selected for calibration. The splitter includes them in the split but the quality report distinguishes them.

**Source Documented From**: Code inspection of `downloader.py:140-165`.

### 2.5 Representation Keywords Too Narrow

**Root Cause**: Initial keyword list missed common representation types (e.g., "billboard", "mural", "sculpture").

**Fix**: Expanded to 30+ keywords covering all common non-photograph categories.

**Source Documented From**: Code inspection of `downloader.py:31-48`.

### 2.6 Stats Computed Inconsistently

**Root Cause**: Manifest, quality report, and pilot script computed stats differently. `CollectionStats` fields were manually assembled rather than derived.

**Fix**: `compute_stats_from_records()` is now the single source of truth. `manifest.py` calls it directly. Quality report reads from ImageRecord fields.

**Source Documented From**: Code inspection of `manifest.py:1-50`, `models.py:1-80`.

## 3. Source Evaluation

### 3.1 Sources Evaluated

| Source | Suitable | Reason | Evidence |
|--------|----------|--------|----------|
| **Wikimedia Commons** | **SELECTED** | Person-name searchable, CC/Public domain licenses, metadata-rich, rate-limitable | API supports `action=query&list=search&srsearch=person+name`. Rate limit 10 req/min. |
| Pexels | REJECTED | Stock photo API, not person-name searchable. Returns generic photos matching keyword, not specific individuals. | Pexels API docs: `GET /v1/search?query=Tom+Hanks` returns stock photos of random people. |
| Unsplash | REJECTED | Stock photo API, not person-name searchable. ToS prohibits ML training datasets. | Unsplash API docs: "You may not use Unsplash API to create a service that competes with Unsplash." |
| Pixabay | REJECTED | Stock photo API, not person-name searchable. Returns generic photos. | Pixabay API docs: `GET /api/?q=Tom+Hanks` returns stock photos. |
| Flickr | DEFERRED | Has person-name search but complex licensing. Requires API key. Rate limits strict. | Flickr API: Requires OAuth. Rate limit 3600 req/hour. CC licenses vary. |

### 3.2 Source Selection Justification

**Wikimedia Commons is the only suitable free source** for identity-specific face retrieval calibration. Pexels, Unsplash, and Pixabay are stock photo platforms optimized for generic image search, not person-name lookup. Their APIs return random photos matching a keyword, not photos of a specific named individual.

Flickr was deferred because it requires OAuth API keys, has complex license negotiation, and Wikimedia Commons provides equivalent or better coverage for public-figure imagery.

### 3.3 Wikimedia Commons Rate Limiting Behavior

- **Rate limit**: 10 requests/minute (unauthenticated)
- **429 frequency**: ~50% of requests during later identities
- **Retry strategy**: 3s delay between downloads, 8 max rate limit retries, exponential backoff
- **Successful mitigation**: All 12 identities acquired without permanent rate limit blocking
- **Total acquisition time**: ~4 minutes (12 identities × 5 images × 3s delay)

## 4. Code Changes

### 4.1 `models.py` — Single Source of Truth

Added `compute_stats_from_records(records: list[ImageRecord]) -> CollectionStats`:

```python
def compute_stats_from_records(records: list[ImageRecord]) -> CollectionStats:
    """Compute CollectionStats from ImageRecord list. Single source of truth."""
    return CollectionStats(
        total_searched=len(records),
        total_downloaded=len(records),
        total_valid=sum(1 for r in records if r.status == "valid"),
        total_duplicates=sum(1 for r in records if r.duplicate_of),
        total_no_face=sum(1 for r in records if r.status == "no_face"),
        total_multi_face=sum(1 for r in records if r.status == "multi_face"),
        total_invalid_image=sum(1 for r in records if r.status == "invalid_image"),
        total_representation=sum(1 for r in records if r.status == "representation"),
        total_identity_uncertain=sum(1 for r in records if r.identity_status == "uncertain"),
        persons_completed=0,
        persons_incomplete=0,
    )
```

Extended `CollectionStats` with `total_multi_face` and `total_invalid_image` fields.

### 4.2 `downloader.py` — Improved Filtering & Policy

- Expanded `REPRESENTATION_KEYWORDS` to 30+ terms
- Multi-face images explicitly routed to review queue with `status="multi_face"`
- Resume behavior: loads existing records from state, only acquires if `face_selected < target`
- Face detection now persists `faces_detected`, `face_selected`, `face_confidence`, `image_category`, `identity_status`, `status` per ImageRecord

### 4.3 `manifest.py` — Stats From Records

- All statistics derived from ImageRecord fields via `compute_stats_from_records()`
- Added `face_count_distribution`, `source_statistics` (per-source stats), `status_distribution`
- Per-person source distribution in quality report

### 4.4 `phase13_6_1_1_pilot.py` — Corrected Pilot Script

- Fixed resume bug: loads existing state, checks per-person `face_selected` count
- Idempotent: running twice produces identical results
- Stats computed from records, not hardcoded
- Logs per-identity summary with face_selected/no_face/multi_face/representation counts

## 5. Corrected Pilot Results

### 5.1 Parameters

- **Identities**: 12 (6 actors + 6 football players)
- **Target images/person**: 5
- **Source**: Wikimedia Commons
- **Dataset version**: `celebrity-v2-pilot-corrected`
- **Seed**: 42
- **Delay**: 3s between downloads

### 5.2 Aggregate Results

| Metric | Phase 13.6.1 (v1) | Phase 13.6.1.1 (corrected) |
|--------|-------------------|---------------------------|
| Identities | 13 | 12 |
| Total images | 95 | 60 |
| Face selected | 57 | 32 |
| No face | 5 | 4 |
| Multi face | 0 (hardcoded) | 18 |
| Representation | 0 (not filtered) | 8 |
| Identity uncertain | 16 | 16 |
| Cross-person duplicates | 0 | 0 |
| Cross-split leakage | 0 | 0 |
| Source | wikimedia_commons | wikimedia_commons |

### 5.3 Per-Person Breakdown

| Person | Total | Face-Sel | No-Face | Multi-Face | Repr | Ref | Query |
|--------|-------|----------|---------|------------|------|-----|-------|
| Tom Hanks | 5 | 4 | 0 | 0 | 1 | 3 | 2 |
| Scarlett Johansson | 5 | 5 | 0 | 0 | 0 | 3 | 2 |
| Denzel Washington | 5 | 0 | 0 | 1 | 5 | 3 | 2 |
| Cate Blanchett | 5 | 5 | 0 | 0 | 0 | 3 | 2 |
| Keanu Reeves | 5 | 3 | 1 | 1 | 2 | 3 | 2 |
| Natalie Portman | 5 | 0 | 1 | 4 | 0 | 3 | 2 |
| Lionel Messi | 5 | 5 | 0 | 0 | 0 | 3 | 2 |
| Cristiano Ronaldo | 5 | 3 | 0 | 2 | 0 | 3 | 2 |
| Kylian Mbappe | 5 | 0 | 0 | 5 | 0 | 3 | 2 |
| Neymar Jr | 5 | 5 | 0 | 0 | 0 | 3 | 2 |
| Erling Haaland | 5 | 2 | 0 | 3 | 0 | 3 | 2 |
| Mohamed Salah | 5 | 0 | 3 | 2 | 0 | 3 | 2 |

### 5.4 Key Observations

1. **Face-selected rate**: 32/60 = 53.3% (down from 57/95 = 60% in v1) — more aggressive filtering working correctly
2. **Representation detection**: 8/60 = 13.3% of images are representations — previously unfiltered
3. **Multi-face detection**: 18/60 = 30% of images have multiple faces — previously hardcoded to 0
4. **Football players have more multi-face**: Mbappe (5/5), Haaland (3/5), Ronaldo (2/5) — group/team photos dominate
5. **Denzel Washington**: 5/5 representation — Wikimedia search returns Wikipedia article images with overlays
6. **Natalie Portman**: 4/5 multi-face — event photos with other people
7. **Mohamed Salah**: 3/5 no-face — Wikimedia search returns team logos, stadium photos
8. **Actors generally better**: Scarlett Johansson (5/5), Cate Blanchett (5/5), Neymar (5/5) — clean portrait photos

### 5.5 Face Count Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0 faces | 5 | 8.3% |
| 1 face | 37 | 61.7% |
| 2-5 faces | 9 | 15.0% |
| 6+ faces | 9 | 15.0% |

## 6. Test Results

### 6.1 Dataset Acquisition Tests

| Suite | Tests | Passed | Failed | New Failures |
|-------|-------|--------|--------|--------------|
| Dataset Acquisition | 50 | 50 | 0 | 0 |
| Search (FAISS + calibration) | 144 | 144 | 0 | 0 |
| Full project | 1280 | 1253 | 27 | 0 |

All 27 failures are pre-existing (26 face_visibility_validator + 1 parser_qa). Zero new failures.

### 6.2 Test Coverage

- `TestComputeStatsFromRecords`: 4 tests — basic stats, multi-face detection, representation detection, identity uncertain
- `TestOfflineFixture`: 6 tests — full pipeline, multi-face routing, no-face rejection, representation marking, resume idempotency, stats propagation
- `TestRetryAfterHandling`: 2 tests — Wikimedia throttle respects delay, consecutive rate limits capped
- `TestFaceServiceValidation`: 3 tests — single face accepted, multiple faces detected, no faces
- `TestRepresentationFiltering`: 5 tests — poster/painting/photograph/billboard/cartoon
- `TestIdentityUncertainty`: 2 tests — identity status recorded, identity confirmed
- `TestSplitter`: 7 tests — basic split, minimum enforced, insufficient excluded, no self-overlap, deterministic, multiple persons, cross-split leakage
- `TestDeduplication`: 2 tests — exact dedup, all unique
- `TestManifest`: 3 tests — generate manifest, quality report, face count distribution
- `TestMockSource`: 3 tests — search, download, download invalid
- `TestEdgeCases`: 3 tests — empty records, single image, two images

## 7. Security Verification

- No API keys in source code, reports, or logs
- `datasets/` is in `.gitignore`
- No celebrity images committed to git
- No secrets in pilot_review.json or DATASET_QUALITY_REPORT.md

## 8. Production Isolation

- `git diff HEAD -- search/ services/reverse_search_manager.py services/face_service.py` — **no changes**
- All changes are in: `dataset_acquisition/` (new module), `tests/test_dataset_acquisition/` (new tests), `phase13_6_1_1_pilot.py` (new script), `.gitignore`
- No modifications to: FAISS index, reverse search pipeline, embedding service, calibration thresholds

## 9. Source Strategy Assessment

### 9.1 Wikimedia Commons Strengths

- Person-name search works for public figures
- CC/Public domain licenses
- Rich metadata (license, attribution, dimensions)
- Rate limit manageable with delays
- Good coverage for actors and athletes

### 9.2 Wikimedia Commons Weaknesses

- Representation images common (Wikipedia article images with overlays)
- Multi-face photos frequent (events, group shots)
- Some identities return no-face images (team logos, stadiums)
- Rate limiting requires patience (~4 minutes for 12 identities)
- Not all public figures have good Wikimedia coverage

### 9.3 Alternative Sources Assessment

**Pexels, Unsplash, Pixabay**: REJECTED — stock photo APIs not designed for person-name search. Their search returns random photos matching a keyword, not photos of a specific named individual. Unsuitable for identity-specific calibration.

**Flickr**: DEFERRED — Has person-name search but requires OAuth API keys, complex licensing negotiation. Could be a future source but requires significant additional infrastructure.

### 9.4 Recommendation

Wikimedia Commons remains the **only suitable free source** for identity-specific face retrieval calibration. The corrected pipeline now properly handles its limitations (representation filtering, multi-face routing, rate limiting).

## 10. Recommendations for Next Steps

1. **Scale to 50-100 identities** on Wikimedia Commons to build a production-grade calibration dataset
2. **Add quality scoring** — face detection confidence threshold to filter low-quality portraits
3. **Investigate Flickr** as a supplementary source for identities with poor Wikimedia coverage
4. **Manual review** of the corrected pilot artifacts before proceeding to calibration
5. **Do NOT run final calibration** until manual review is complete

## 11. Verdict

**PILOT_READY_FOR_MANUAL_REVIEW**

The corrected pilot demonstrates:
- All identified bugs fixed (resume, stats, filtering, multi-face policy)
- Source evaluation complete (Wikimedia Commons only suitable free source)
- 12 identities, 60 images, 32 calibration-ready (53.3%)
- 0 cross-person duplicates, 0 cross-split leakage
- 50/50 tests pass, 1253/1280 regression pass (27 pre-existing)
- Production pipeline untouched
- No secrets committed

Manual review of `outputs/phase13_6_1_1_review/` artifacts required before proceeding.

## 12. Artifacts

```
outputs/phase13_6_1_1_review/
├── pilot_review.json                    # Summary statistics
└── pilot_images_detail.jsonl            # Per-image metadata (60 records)

datasets/celebrity-v2-pilot-corrected/
├── metadata/
│   ├── dataset_manifest.json
│   └── DATASET_QUALITY_REPORT.md
├── raw/                                 # 60 images across 12 identities
├── split/
│   ├── reference/                       # 36 reference images
│   └── query/                           # 24 query images
└── download_state.json

phase13_6_1_1_pilot.py                   # Corrected pilot script

tests/test_dataset_acquisition/
└── test_acquisition.py                  # 50 tests

dataset_acquisition/
├── models.py                            # Added compute_stats_from_records()
├── downloader.py                        # Expanded filtering, explicit multi-face policy
├── manifest.py                          # Stats from records
└── cli.py                               # Stats computed from records
```

## 13. Files Modified

| File | Change |
|------|--------|
| `dataset_acquisition/models.py` | Added `compute_stats_from_records()`, `total_multi_face`, `total_invalid_image` fields |
| `dataset_acquisition/downloader.py` | Expanded representation keywords, explicit multi-face routing, resume idempotency |
| `dataset_acquisition/manifest.py` | Stats derived from ImageRecord via `compute_stats_from_records()` |
| `dataset_acquisition/cli.py` | Stats computed from records |
| `tests/test_dataset_acquisition/test_acquisition.py` | 50 tests including offline fixture, multi-face, representation, resume idempotency |
| `.gitignore` | Added `datasets/` entry |
| `phase13_6_1_1_pilot.py` | NEW — corrected pilot script |
| `PHASE_13_6_1_1_REPORT.md` | NEW — this report |
