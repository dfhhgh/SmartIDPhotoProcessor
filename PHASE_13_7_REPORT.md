# Phase 13.7 — Reverse Search Calibration & Threshold Selection

## Verdict: `CALIBRATION_SUCCESS_WITH_LIMITATIONS`

## Date
2026-09-01

## Summary
Audit and corrected re-calibration of Phase 13.6.3 reverse-search evaluation. Phase 13.6.3 had 4 methodological bugs (query→query calibration, insufficient hard negative analysis, image-level vs identity-level retrieval, threshold interpretation error). All bugs have been fixed. Corrected results show reasonable but not production-ready calibration.

## Bugs Fixed

| # | Bug | Severity | Status |
|---|-----|----------|--------|
| 1 | Query→query calibration pairs instead of query→reference | Critical | Fixed |
| 2 | Insufficient hard negative analysis (first-in-Top-K only) | Moderate | Fixed |
| 3 | Image-level vs identity-level retrieval metrics | Moderate | Fixed |
| 4 | Threshold interpretation error in report | Critical | Fixed |

## Dataset
- 22 identities (12 actors + 10 footballers)
- 176 reference images (8/person)
- 88 query images (4/person)
- 704 genuine pairs, 14,784 impostor pairs
- 264 total images, zero leakage

## Calibration Results

### Image-Level
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.8785 |
| EER | 19.60% |
| EER threshold | 0.0553 |

### Identity-Level
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.9465 |
| EER | 11.55% |
| EER threshold | 0.1214 |

### Recommended Operating Points

| Target | Threshold | FAR | FRR | TPR | F1 |
|--------|-----------|-----|-----|-----|----|
| Youden's J (image) | 0.2301 | 0.06% | 27.8% | 72.2% | 0.832 |
| FAR ≤ 1% (image) | 0.1594 | 0.91% | 27.4% | 72.6% | 0.757 |
| Youden's J (identity) | 0.2415 | 0.32% | 12.5% | 87.5% | 0.901 |
| FAR ≤ 0.1% (identity) | 0.4861 | 0.00% | 13.6% | 86.4% | 0.927 |

### Threshold Stability (5-Fold CV)
- EER threshold: mean=0.0549, std=0.0066 (stable)
- FAR≤1% threshold: mean=0.259, std=0.030

### Hard Negatives
- Global maximum impostor similarity: 0.2975 (robert_downey_jr → neymar)
- All hard negatives well below threshold at Youden's J

### Per-Identity Issues
| Identity | Genuine Mean | False Rejects | Strongest Impostor |
|----------|-------------|---------------|-------------------|
| jennifer_lawrence | 0.102 | 22 | 0.212 |
| morgan_freeman | 0.066 | 19 | 0.137 |
| leonardo_dicaprio | 0.147 | 15 | 0.243 |
| vinicius_junior | 0.290 | 16 | 0.236 |
| brad_pitt | 0.396 | 7 | 0.248 |

These identities show high intra-identity variance, likely due to image conditions (lighting, pose, age variation across reference images).

## Why `SUCCESS_WITH_LIMITATIONS`

**Strengths:**
- Identity-level ROC-AUC=0.9465 (good separability)
- Threshold stability excellent (EER std=0.0066)
- All hard negatives well below classification threshold
- Clean separation in top operating points (FAR<1% with F1>0.75)

**Limitations:**
- Image-level ROC-AUC=0.8785 (moderate overlap between genuine/impostor at image level)
- Several identities show very low genuine similarity (jennifer_lawrence mean=0.10, morgan_freeman mean=0.07)
- Dataset is small (22 identities, 88 queries) — not representative of production diversity
- High FRR at low FAR operating points (~27% at FAR≤1%)
- **Not production-approved** — this calibration is a candidate only

## Artifacts

| File | Description |
|------|-------------|
| `evaluation_audit.md` | Detailed methodology audit |
| `calibration_summary.json` | Complete calibration metrics |
| `calibration_summary.md` | Human-readable summary |
| `genuine_scores.csv` | 704 genuine pairs with similarity scores |
| `impostor_scores.csv` | 14,784 impostor pairs with similarity scores |
| `score_distribution.csv` | All scores with labels |
| `hard_negatives.csv` | Strongest impostor per query |
| `hard_negatives.md` | Top 20 hard negatives |
| `threshold_operating_points.csv` | All operating points |
| `threshold_selection.md` | Threshold recommendations |
| `identity_error_analysis.csv` | Per-identity error analysis |
| `retrieval_results.csv` | Per-query retrieval results |
| `plots/score_distributions.png` | Genuine vs impostor distributions |
| `plots/roc_curve.png` | ROC curve |
| `plots/far_frr_vs_threshold.png` | FAR/FRR vs threshold |
| `plots/impostor_tail.png` | Impostor score tail distribution |

## Regression
- 1462 passed, 27 pre-existing failures, 0 new failures
- Production isolation maintained
- No modifications to production code

## Next Steps (Future Phases)
1. Collect larger, more diverse identity-labeled dataset
2. Investigate low-genuine-similarity identities (jennifer_lawrence, morgan_freeman)
3. Consider identity-level threshold operating point (FAR≤0.1% at 0.4861)
4. Cross-validate on non-celebrity data before production deployment
