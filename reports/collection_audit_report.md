# Dataset Collection Audit Report

## SmartIDPhotoProcessor BiSeNet Fine-Tuning Dataset

**Generated:** 2026-08-16
**Status:** Pilot-Ready

---

## 1. Existing Query Audit

### Summary

| Category | Original Queries | Optimized Queries | Issues Found |
|----------|------------------|-------------------|--------------|
| normal | 40 | 38 | Redundancy, missing age diversity |
| hijab | 40 | 32 | Demographic bias, redundancy |
| eyeglasses | 40 | 36 | Redundancy, missing age diversity |
| sunglasses | 39 | 34 | Non-face queries, redundancy |
| mask | 40 | 35 | Redundancy |
| cap | 39 | 33 | Non-face queries, redundancy |
| beard | 39 | 35 | Redundancy |
| helmet | 40 | 34 | Full-body queries, redundancy |
| scarf | 41 | 34 | Redundancy |
| hair_occlusion | 41 | 39 | **Chinese characters (critical)** |

### Critical Issues Fixed

1. **hair_occlusion.txt**: Removed 4 Chinese characters (`遮`) that would fail with English-language search APIs
2. **All categories**: Reduced redundant queries that returned similar results

---

## 2. Redundant Queries

### Identified Redundancies (Now Removed)

| Category | Removed Queries | Reason |
|----------|-----------------|--------|
| normal | "portrait photo", "face portrait", "clear face photo" | Near-identical semantics |
| hijab | "hijab portrait", "hijab headscarf portrait" | Same concept |
| eyeglasses | "eyeglasses portrait", "glasses wearing portrait" | Same concept |
| sunglasses | "sunglasses portrait", "sunglasses portrait photo" | Duplicate |
| mask | "face mask portrait", "masked face portrait" | Same concept |
| cap | "cap portrait", "cap headshot" | Very similar |
| beard | "beard portrait", "bearded face portrait" | Same concept |
| helmet | "helmet portrait", "helmet headshot" | Very similar |
| scarf | "scarf portrait", "scarf headshot" | Very similar |

---

## 3. Weak Queries

### Queries That May Return Non-Face Images

| Category | Weak Query | Issue |
|----------|------------|-------|
| sunglasses | "sunglasses beach portrait" | May return full-body beach photos |
| sunglasses | "sunglasses summer portrait" | Context too broad |
| cap | "cap street style portrait" | May return full-body fashion photos |
| helmet | "construction worker portrait" | May return full-body work photos |
| scarf | "winter fashion portrait" | May not always include scarf |

**Action**: These queries were removed or reworded to focus on face/portrait framing.

---

## 4. Recommended Queries

### Optimized Query Structure

Each category now follows this structure:

1. **Subject/type variation** (3-5 queries)
2. **Face/portrait framing** (4-6 queries)
3. **Background/lighting variation** (4-6 queries)
4. **Identity style** (2-3 queries)
5. **Style/fashion variation** (3-5 queries)

### Example: eyeglasses.txt

```
# Subject variation
person with eyeglasses
student with glasses
young person with glasses
woman with glasses portrait
man with glasses portrait
elderly person with glasses

# Frame type variation
clear frame glasses
thin frame glasses
round glasses portrait
rectangular glasses portrait
wire frame glasses portrait

# Face framing
glasses headshot
glasses face photo
glasses front facing portrait
glasses looking at camera

# Background/lighting
glasses neutral expression
glasses white background
glasses studio portrait
glasses clean background
```

---

## 5. Category Balance Proposal

### Collection Matrix

| Category | Type | Target | % | Priority | Rationale |
|----------|------|--------|---|----------|-----------|
| normal | BASELINE | 100 | 20% | HIGH | Preserves all 19 parser classes |
| eyeglasses | TARGET | 60 | 12% | HIGH | Previous parser weakness |
| sunglasses | TARGET | 60 | 12% | HIGH | Previous parser weakness |
| hijab | TARGET | 50 | 10% | MEDIUM | Head covering variation |
| mask | TARGET | 50 | 10% | MEDIUM | Lower face occlusion |
| cap | TARGET | 50 | 10% | MEDIUM | Head covering variation |
| beard | TARGET | 40 | 8% | MEDIUM | Facial hair occlusion |
| helmet | TARGET | 40 | 8% | LOW | Hard head covering |
| scarf | TARGET | 30 | 6% | LOW | Neck/face covering |
| hair_occlusion | TARGET | 20 | 4% | LOW | Hair covering features |
| **TOTAL** | | **500** | **100%** | | |

### Distribution Rationale

- **Normal (20%)**: Critical for preserving baseline behavior. Previous fine-tuning degraded BACKGROUND, SKIN, BROW, and other non-target classes. High normal count prevents catastrophic forgetting.

- **Eyeglasses/Sunglasses (12% each)**: Highest priority targets. Previous experiment showed improvement in LEFT_EYE and RIGHT_EYE classes. These are the most common occlusions in ID photos.

- **Hijab/Mask/Cap (10% each)**: Medium priority. Important for SmartIDPhotoProcessor use case (ID photos with head coverings).

- **Beard (8%)**: Medium priority. Common facial hair variation.

- **Helmet/Scarf/Hair_occlusion (4-8% each)**: Lower priority but still important for comprehensive face parsing.

---

## 6. Source Distribution Proposal

### Recommended Source Strategy

| Source | Strengths | Best For |
|--------|-----------|----------|
| Pexels | High quality, consistent | Professional portraits, studio shots |
| Pixabay | Large variety | Casual photos, diverse styles |
| Openverse | Creative Commons | Varied demographics, contexts |
| Wikimedia Commons | Educational/documentary | Cultural diversity, real-world |

### Source Balance

- **Target conditions**: Prioritize Pexels and Pixabay for consistent quality
- **Normal baseline**: Use all four sources for maximum diversity
- **Demographic diversity**: Leverage Openverse and Wikimedia for varied populations

### Cross-Source Duplicate Detection

The collection system should:
1. Track source ID + image URL for each download
2. Use perceptual hashing for cross-source duplicate detection
3. Keep first occurrence, mark subsequent as duplicates
4. Preserve metadata traceability for all decisions

---

## 7. Pilot Collection Plan

### Pilot Parameters

- **Target**: 30 images per category
- **Total**: 300 images (10 categories × 30)
- **Sources**: All four enabled sources
- **Queries**: First 5 queries per category
- **Duration**: ~30 minutes estimated

### Pilot Command

```bash
# Dry run first
python scripts/collect_pilot.py --pilot --max-per-query 3

# Then execute
python scripts/collect_pilot.py --execute --pilot --max-per-query 3
```

### Pilot Evaluation Criteria

After pilot collection, evaluate:
1. **Query quality**: Do queries return relevant face images?
2. **Source balance**: Are all sources contributing?
3. **Category balance**: Are categories receiving expected counts?
4. **Image quality**: Are images suitable for face parsing?
5. **Metadata completeness**: Is all traceability information captured?

---

## 8. Expected Dataset Size

### Full Collection Targets

| Metric | Value |
|--------|-------|
| Total images | 500 |
| Categories | 10 |
| Sources | 4 |
| Avg images/category | 50 |
| Avg images/query | ~3 |
| Total queries | ~370 |

### Storage Estimates

| Component | Size |
|-----------|------|
| Raw images (avg 200KB each) | ~100 MB |
| Metadata JSON | ~2 MB |
| Reports | ~1 MB |
| **Total** | **~103 MB** |

---

## 9. Duplicate Strategy

### Duplicate Detection Layers

1. **Source-level**: Each source returns unique IDs
2. **URL-level**: Track download URLs across sources
3. **Perceptual hashing**: Use existing DuplicateRemover
4. **Cross-source**: Compare hashes across all downloaded images

### Duplicate Handling

- **First occurrence**: Keep with full metadata
- **Subsequent occurrences**: Move to duplicates_removed/ directory
- **Metadata**: Preserve traceability for all decisions
- **Reporting**: Track duplicate counts per source and category

---

## 10. Metadata/License Strategy

### Required Metadata Fields

Every downloaded image must preserve:
- `source`: Which API (pexels, pixabay, openverse, wikimedia_commons)
- `source_id`: Provider-specific unique identifier
- `query`: Search query that produced the result
- `page_url`: Public detail page URL
- `download_url`: Direct image URL
- `photographer`: Author/creator when available
- `license_name`: License identifier
- `license_url`: Link to license terms
- `license_type`: Machine-readable license type
- `download_timestamp`: UTC timestamp of download

### License Compliance

- **Pexels License**: Free for commercial use, no attribution required
- **Pixabay License**: Free for commercial use, no attribution required
- **CC BY/Openverse**: Attribution required, track for compliance
- **Wikimedia Commons**: Various licenses, track per image

### Metadata Storage

- Individual JSON files per image in `dataset/metadata/`
- Combined CSV index for analysis
- Full traceability from query to final selection

---

## 11. Risks and Mitigations

### Risk 1: Catastrophic Forgetting

**Risk**: Fine-tuning degrades non-target classes
**Mitigation**: 20% normal baseline category, balanced distribution

### Risk 2: Source Dominance

**Risk**: One source contributes majority of images
**Mitigation**: Per-source limits, track contribution balance

### Risk 3: Query Bias

**Risk**: Queries return similar demographics/styles
**Mitigation**: Diverse query structure, multiple source strategies

### Risk 4: Quality Degradation

**Risk**: Downloaded images unsuitable for face parsing
**Mitigation**: Quality gates, face detection, resolution requirements

### Risk 5: License Violations

**Risk**: Using images without proper attribution
**Mitigation**: Metadata traceability, license tracking

### Risk 6: Network Failures

**Risk**: API rate limits or connectivity issues
**Mitigation**: Retry logic, resumable collection, progress tracking

---

## 12. Collection Commands

### Dry Run (Preview)

```bash
# Preview all categories
python scripts/collect_pilot.py

# Preview specific categories
python scripts/collect_pilot.py --categories normal eyeglasses

# Preview with limits
python scripts/collect_pilot.py --pilot --max-per-category 10
```

### Pilot Collection

```bash
# Execute pilot collection
python scripts/collect_pilot.py --execute --pilot

# Pilot with specific categories
python scripts/collect_pilot.py --execute --pilot --categories normal eyeglasses sunglasses
```

### Full Collection

```bash
# Full collection with limits
python scripts/collect_pilot.py --execute --max-per-category 50 --max-total 500

# Specific sources only
python scripts/collect_pilot.py --execute --sources pexels pixabay
```

---

## 13. Next Steps

1. **Run pilot collection**: Execute `python scripts/collect_pilot.py --execute --pilot`
2. **Inspect pilot results**: Review collected images for quality and relevance
3. **Adjust queries**: Refine queries based on pilot feedback
4. **Run full collection**: Execute full collection after pilot approval
5. **Quality filtering**: Run face detection and quality filters
6. **Dataset preparation**: Prepare final dataset for BiSeNet fine-tuning

---

## Appendix: Files Modified

| File | Change |
|------|--------|
| `dataset_builder/queries/hair_occlusion.txt` | Removed Chinese characters, improved queries |
| `dataset_builder/queries/normal.txt` | Added age diversity, reduced redundancy |
| `dataset_builder/queries/hijab.txt` | Reduced demographic bias, improved structure |
| `dataset_builder/queries/eyeglasses.txt` | Added age diversity, reduced redundancy |
| `dataset_builder/queries/sunglasses.txt` | Removed non-face queries, improved structure |
| `dataset_builder/queries/mask.txt` | Reduced redundancy, improved structure |
| `dataset_builder/queries/cap.txt` | Removed non-face queries, improved structure |
| `dataset_builder/queries/beard.txt` | Reduced redundancy, improved structure |
| `dataset_builder/queries/helmet.txt` | Removed full-body queries, improved structure |
| `dataset_builder/queries/scarf.txt` | Reduced redundancy, improved structure |

## Appendix: Files Created

| File | Purpose |
|------|---------|
| `dataset_builder/config/collection_config.py` | Collection matrix and configuration |
| `scripts/collect_pilot.py` | Controlled collection script with dry-run |
| `reports/collection_audit_report.md` | This report |
