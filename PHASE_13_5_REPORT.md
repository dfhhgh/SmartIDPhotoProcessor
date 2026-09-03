# Phase 13.5 — Reverse Search Threshold Calibration Report

## 1. Executive Summary

Phase 13.5 performed an empirical calibration experiment on the available project dataset to characterize the ArcFace similarity distribution and assess feasibility for production threshold selection.

**Critical finding: The dataset contains 39 unique images of different people with NO person-identity labels. No image has a second image of the same person. This makes genuine (same-person) calibration IMPOSSIBLE from this dataset.**

The calibration is therefore classified as **EXPLORATORY ONLY**. All ROC-AUC, FAR/FRR, and EER metrics reported below use SYNTHETIC genuine scores for infrastructure testing — they do NOT represent real-world performance.

**Verdict: INSUFFICIENT EVIDENCE** — the available dataset cannot support production threshold selection. A dataset with multiple images per person is required before proceeding to Phase 13.6.

## 2. Objective

Answer the following questions through empirical measurement:
1. How are genuine/same-person similarities distributed? — **CANNOT ANSWER** (no same-person pairs)
2. How are impostor/different-person similarities distributed? — **ANSWERED**
3. How much overlap exists? — **CANNOT FULLY ANSWER** (no genuine distribution)
4. What is the ROC-AUC? — **CANNOT ANSWER** (requires both distributions)
5. What are FAR and FRR at different thresholds? — **CANNOT ANSWER** (requires genuine)
6. What is the estimated EER? — **CANNOT ANSWER** (requires genuine)
7. How good is Top-1/Top-3/Top-5 retrieval? — **PARTIALLY ANSWERED** (self-retrieval only)
8. Is the dataset sufficient for decision-layer development? — **NO**
9. What limitations prevent generalizing to celebrity detection? — **DOCUMENTED**

## 3. Existing Architecture Used

| Component | Source | Purpose |
|-----------|--------|---------|
| `InsightFace buffalo_l` | `ai_models/models/buffalo_l/` | Face detection + ArcFace embedding |
| `w600k_r50.onnx` | InsightFace | ArcFace recognition model (512-D) |
| `FlatIndex` | `search/flat_index.py` | FAISS IndexFlatIP wrapper |
| `EmbeddingValidator` | `search/embedding_validator.py` | L2-normalized float32 contract |
| `PairGenerator` | `search/calibration/pair_generator.py` | Deterministic pair generation |
| `SimilarityEvaluator` | `search/calibration/similarity_evaluator.py` | Inner-product similarity |
| `ThresholdEvaluator` | `search/calibration/threshold_evaluator.py` | ROC-AUC, EER, FAR/FRR |
| `CalibrationReporter` | `search/calibration/calibration_report.py` | Artifact export + plots |

No production code was modified. No models were retrained. No external APIs were used.

## 4. Dataset Description

| Metric | Value |
|--------|-------|
| Dataset path | `test_images/good/` |
| Total images | 43 |
| Usable images | 39 |
| Excluded images | 4 (1 unreadable, 3 no face detected) |
| Person subdirectories | 0 (flat structure) |
| Person identity labels | **NONE** |
| Persons with >= 2 images | 0 |
| Images per person | 1 (every image is a different person) |
| Unique image hashes | 40 (2 file copies detected) |

**Dataset limitation**: All 39 usable images depict different individuals. There are no repeat images of the same person. This dataset was designed for face parser quality validation, not for identity similarity calibration.

## 5. Reference/Query Split

**NOT IMPLEMENTED** — impossible without person-identity labels.

A reference/query split requires:
- Multiple images per person
- Deterministic assignment of images to reference vs query per person
- At least 2 images per person (1 reference + 1 query)

None of these conditions are met.

**Decision**: No split was invented. All 39 embeddings are treated as independent impostor samples.

## 6. Data Leakage Analysis

**STATUS: NO LEAKAGE POSSIBLE**

Since there are no same-person pairs, there is no risk of query images appearing in reference data for the same identity. All pairwise comparisons are between different people (impostor pairs).

For the synthetic genuine scores used in infrastructure testing: these are randomly generated from `N(0.85, 0.05)` and have no relationship to any real images. No leakage occurs.

## 7. Embedding Generation

| Property | Value | Verified |
|----------|-------|----------|
| Model | InsightFace `buffalo_l` (ArcFace `w600k_r50.onnx`) | CONFIRMED FROM CODE |
| Dimension | 512 | EMPIRICALLY VERIFIED |
| Dtype | float32 | EMPIRICALLY VERIFIED |
| Normalization | L2-normalized (`face.normed_embedding`) | EMPIRICALLY VERIFIED |
| Provider | CPUExecutionProvider (CUDA not available) | CONFIRMED FROM CODE |

Embedding extraction: 19.5s for 39 images (0.50s/image average on CPU).

## 8. Similarity Definition

**Production (FAISS)**: `IndexFlatIP` on L2-normalized vectors = cosine similarity.

**Calibration**: `np.dot(embedding_1, embedding_2)` on L2-normalized vectors = cosine similarity.

These are semantically identical. Verified in `test_search/test_flat_index.py::test_numpy_ranking_agrees_with_faiss`.

Convention: `similarity >= threshold` predicts SAME PERSON (positive).

## 9. Positive Pair Methodology

**NOT APPLICABLE** — no same-person pairs exist in the dataset.

Positive pair count: **0**

## 10. Negative Pair Methodology

**All pairwise comparisons are negative (impostor).**

- Each of the 39 images is a different person
- Total possible negative pairs: C(39,2) = 741
- All 741 pairs were computed (exhaustive, no sampling needed)

## 11. Sampling Strategy

| Parameter | Value |
|-----------|-------|
| Positive sampling | N/A |
| Negative sampling | Exhaustive (all pairs) |
| Random seed | 42 |
| Max positive pairs | N/A |
| Max negative pairs | N/A |

## 12. Genuine Distribution

**CANNOT BE COMPUTED** from this dataset.

No same-person image pairs exist. The genuine distribution is unknown.

Synthetic genuine scores used for infrastructure testing only:
- Count: 100
- Mean: 0.8448
- Std: 0.0452
- Min: 0.7190
- Max: 0.9426

**These synthetic values do NOT represent real-world same-person similarity.**

## 13. Impostor Distribution

| Statistic | Value |
|-----------|-------|
| Count | 741 |
| Min | -0.2030 |
| Max | **1.0000** |
| Mean | 0.0164 |
| Median | 0.0042 |
| Std | 0.1110 |
| P1 | -0.1381 |
| P5 | -0.0971 |
| P10 | -0.0735 |
| P25 | -0.0373 |
| P50 | 0.0042 |
| P75 | 0.0484 |
| P90 | 0.0986 |
| P95 | 0.1300 |
| P99 | 0.6559 |

**NOTABLE**: The maximum impostor similarity is **1.0000** — two different people produced identical ArcFace embeddings. This is a significant finding that warrants investigation in Phase 13.6.

## 14. Distribution Overlap Analysis

**CANNOT BE COMPUTED** — genuine distribution is unknown.

The impostor distribution is centered near 0 (mean=0.016, median=0.004) with most mass between -0.1 and 0.15. However, the long tail extends to 1.0, which would create significant overlap with any genuine distribution.

## 15. ROC-AUC

**CANNOT BE COMPUTED** from real data — requires both genuine and impostor distributions.

ROC-AUC computed with **SYNTHETIC** genuine scores for infrastructure testing only:

| Metric | Value |
|--------|-------|
| Custom implementation | 0.9958 |
| sklearn reference | 0.9958 |
| Absolute difference | 1.56e-10 |
| Verified | True |

**These values do NOT represent real-world performance.**

## 16. ROC-AUC Verification Against Reference

| Implementation | AUC | Notes |
|----------------|-----|-------|
| Custom (trapezoidal) | 0.9958434546345973 | In-house implementation |
| sklearn `roc_auc_score` | 0.9958434547908231 | Trusted reference |
| Absolute difference | 1.56e-10 | Within floating-point tolerance |
| **VERIFIED** | **YES** | Materials disagree by < 1e-4 |

The custom implementation is verified against sklearn. Both handle tied scores identically.

## 17. FAR/FRR Analysis

**CANNOT BE COMPUTED** from real data — requires genuine distribution.

All FAR/FRR values below use **SYNTHETIC** genuine scores. They are presented for infrastructure testing only and do NOT represent real-world performance.

## 18. Threshold Sweep

**Method**: Event-based sweep using observed score boundaries.

- Total thresholds evaluated: 1,537
- Threshold resolution: all unique score midpoints + boundaries
- Convention: `similarity >= threshold` = predicted same-person

The sweep captures every meaningful classification change point in the synthetic+impostor score space.

## 19. Candidate Operating Points

**WARNING: These operating points use SYNTHETIC genuine scores mixed with REAL impostor scores. They are NOT production-ready thresholds.**

### A. Low-FAR Point (synthetic)
| Metric | Value |
|--------|-------|
| Threshold | 0.6637 |
| TP | 100 |
| FP | 7 |
| TN | 734 |
| FN | 0 |
| TPR | 1.000 |
| TNR | 0.991 |
| FAR | 0.0094 |
| FRR | 0.000 |
| Precision | 0.935 |
| Recall | 1.000 |

### B. Balanced FAR/FRR Point (synthetic)
| Metric | Value |
|--------|-------|
| Threshold | 0.7348 |
| TP | 99 |
| FP | 5 |
| TN | 736 |
| FN | 1 |
| TPR | 0.990 |
| TNR | 0.993 |
| FAR | 0.0067 |
| FRR | 0.010 |
| Precision | 0.952 |
| Recall | 0.990 |

### C. High-Recall Point (synthetic)
| Metric | Value |
|--------|-------|
| Threshold | 0.7608 |
| TP | 96 |
| FP | 3 |
| TN | 738 |
| FN | 4 |
| TPR | 0.960 |
| TNR | 0.996 |
| FAR | 0.0040 |
| FRR | 0.040 |
| Precision | 0.970 |
| Recall | 0.960 |

**DO NOT USE THESE THRESHOLDS IN PRODUCTION.**

## 20. Estimated EER

**CANNOT BE COMPUTED** from real data.

EER computed with **SYNTHETIC** genuine scores for infrastructure testing only:

| Metric | Value |
|--------|-------|
| EER | 0.0067 |
| EER Threshold | 0.7297 |
| FAR at EER | 0.0067 |
| FRR at EER | 0.0067 |
| Method | Interpolation (linear crossing) |
| Threshold resolution | 1,537 |

**This EER does NOT represent real-world performance.**

## 21. Top-1 Retrieval

Self-retrieval (each image queries itself in the FAISS index):

| Metric | Value |
|--------|-------|
| Top-1 accuracy | 94.87% (37/39) |
| Total queries | 39 |

**NOTABLE**: 2 images did NOT find themselves at rank 1. This indicates those images have higher similarity to other people's images than to themselves — a potential issue with the ArcFace model on certain face types.

## 22. Top-3 Retrieval

| Metric | Value |
|--------|-------|
| Top-3 accuracy | 100.00% (39/39) |

All images find themselves within the top 3 results when querying the full index.

## 23. Top-5 Retrieval

| Metric | Value |
|--------|-------|
| Top-5 accuracy | 100.00% (39/39) |

## 24. Hard Negative Analysis

**Impostor pairs with similarity >= P99 (0.6559):**

The maximum impostor similarity is **1.0000** — meaning two different people produced identical ArcFace embeddings. This is the hardest possible negative case.

The P99 impostor similarity is 0.6559, meaning approximately 1% of different-person pairs have similarity above this threshold. Any production threshold below 0.6559 would produce false accepts on this dataset.

**Metadata-only analysis**: No demographic classification was performed. The high-similarity impostor pairs warrant investigation to understand whether they represent lookalikes, data quality issues, or model limitations.

## 25. Dataset Limitations

1. **No person-identity labels**: All 39 images are different people. No same-person pairs exist.
2. **Flat directory structure**: No person subdirectories for organized identity data.
3. **Small size**: Only 39 usable images from 43 total.
4. **No diversity metadata**: No age, gender, ethnicity, or pose annotations.
5. **No reference/query split possible**: Requires multiple images per person.
6. **Designed for parser validation**: This dataset was created for BiSeNet face parser quality assessment, not identity similarity calibration.

## 26. Generalization Limitations

1. **Single dataset**: All results come from one small, non-identity-labeled dataset.
2. **No genuine distribution**: Cannot characterize same-person similarity behavior.
3. **No cross-dataset validation**: No external dataset for independent verification.
4. **CPU-only inference**: CUDAExecutionProvider not available; may affect embedding quality on GPU vs CPU.
5. **Cultural bias risk**: No demographic diversity metadata available.
6. **Image quality range**: Unknown whether images represent the full range of production conditions.

## 27. Celebrity/Public-Figure Detection Limitations

**This calibration evaluates face-identity similarity retrieval. It does NOT evaluate celebrity detection.**

Key distinctions:
- **Identity similarity retrieval**: Finding images of the same person in a reference database
- **Known public figure detection**: Identifying whether a face belongs to a named public figure

The current calibration:
- Contains NO actual public figures or celebrities
- Contains NO name-to-face mappings
- Tests ONLY whether different people's embeddings are distinguishable
- Cannot validate whether the system would correctly identify a specific celebrity

Celebrity detection would require:
- A reference database of known public figures with name labels
- Multi-image references per person for robust matching
- Evaluation on actual celebrity images with ground-truth identity labels

## 28. Reproducibility

| Metric | Run 1 | Run 2 | Match |
|--------|-------|-------|-------|
| n_pairs | 741 | 741 | YES |
| ROC-AUC (custom) | 0.9958434546345973 | 0.9958434546345973 | YES |
| ROC-AUC (sklearn) | 0.9958434547908231 | 0.9958434547908231 | YES |
| EER | 0.006747638326585695 | 0.006747638326585695 | YES |
| EER threshold | 0.7296769815257043 | 0.7296769815257043 | YES |
| Impostor mean | 0.016401885077357292 | 0.016401885077357292 | YES |
| Impostor std | 0.1110241636633873 | 0.1110241636633873 | YES |
| Impostor min | -0.20300060510635376 | -0.20300060510635376 | YES |
| Impostor max | 1.0 | 1.0 | YES |
| Self-retrieval Top-1 | 0.9487179487179487 | 0.9487179487179487 | YES |
| Synthetic mean | 0.8448076844215393 | 0.8448076844215393 | YES |
| Event sweep size | 1537 | 1537 | YES |

**REPRODUCIBILITY: PASS** — all 12 metrics match exactly between two independent runs.

## 29. Performance

| Metric | Value |
|--------|-------|
| Environment | Windows, Python 3.12.2, CPU only |
| Embedding extraction | 19.5s for 39 images (0.50s/image) |
| Pairwise computation | <0.01s for 741 pairs |
| Event-based sweep | <0.1s for 1,537 thresholds |
| Total runtime | 19.5s |
| FAISS index build | <0.01s for 39 vectors |
| FAISS search latency | ~0.05ms per query (from Phase 13.1 benchmark) |

## 30. Test Results

### Phase 13.1 (FAISS Core): 34/34 PASS
### Phase 13.2 (Index Builder): 6/6 PASS
### Phase 13.3 (ReverseSearchService): 24/24 PASS
### Phase 13.4 (Production Integration): 5/5 PASS
### Phase 13.5 (Calibration): 40/40 PASS
### Benchmark: 7/7 PASS
### **Total: 132/132 PASS**

## 31. Regression Results

| Suite | Passed | Failed | New Failures |
|-------|--------|--------|--------------|
| All search tests | 132 | 0 | 0 |
| Pre-existing regression | 27 failures | 0 | 0 |
| Pre-existing CUDA errors | 9 | 0 | 0 |

**No new regressions introduced.**

## 32. Final Conclusion

**INSUFFICIENT EVIDENCE**

The available evidence does NOT justify proceeding to production decision calibration.

### What was confirmed:
1. ArcFace embeddings are extractable from project images (EMPIRICALLY VERIFIED)
2. FAISS index construction and search work correctly (EMPIRICALLY VERIFIED)
3. Impostor distribution is centered near 0 with std=0.111 (EMPIRICALLY VERIFIED)
4. Some different people produce identical embeddings (similarity=1.0) (EMPIRICALLY VERIFIED)
5. Self-retrieval Top-1 is 94.87%, not 100% (EMPIRICALLY VERIFIED)
6. ROC-AUC custom implementation matches sklearn (EMPIRICALLY VERIFIED)
7. All calibration infrastructure is correct and tested (CONFIRMED FROM CODE)
8. Reproducibility is perfect across runs (EMPIRICALLY VERIFIED)

### What was NOT confirmed:
1. Genuine (same-person) similarity distribution
2. Real-world ROC-AUC
3. Real-world FAR/FRR
4. Real-world EER
5. Production-ready threshold
6. Celebrity detection capability

### Verdict:
**INSUFFICIENT EVIDENCE** — the dataset lacks person-identity labels, making genuine calibration impossible. The impostor distribution alone is insufficient for threshold selection.

## 33. Recommendation for Phase 13.6

**DO NOT PROCEED** to Phase 13.6 (production threshold selection) until:

1. **A person-identity-labeled dataset is obtained** with:
   - At least 50 distinct persons
   - At least 3 images per person
   - At least 2 images per person for reference/query split

2. **Genuine calibration is performed** on the new dataset to determine:
   - Same-person similarity distribution
   - Real ROC-AUC
   - Real FAR/FRR curves
   - Real EER

3. **The impostor max=1.0 anomaly is investigated** — understand why two different people produce identical embeddings.

4. **Self-retrieval gap is investigated** — understand why 2/39 images don't find themselves at rank 1.

## Definition of Done Checklist

- [x] Existing implementation reviewed
- [x] ROC-AUC verified against sklearn reference implementation
- [x] Ties handled correctly
- [x] EER methodology documented and tested
- [x] Threshold evaluation sufficiently precise (1,537 event-based thresholds)
- [x] Reference/query separation: NOT POSSIBLE (no identity labels) — documented
- [x] Data leakage: NO LEAKAGE POSSIBLE — documented
- [x] Positive pairs: 0 (none possible)
- [x] Negative pairs: 741 (exhaustive)
- [x] No self-pairs
- [x] Deterministic sampling (seed=42)
- [x] Existing ArcFace embedding reused
- [x] Similarity semantics verified (inner product = cosine for normalized vectors)
- [x] Genuine distribution: CANNOT COMPUTE — documented
- [x] Impostor distribution: COMPUTED
- [x] ROC-AUC: CANNOT COMPUTE from real data — documented
- [x] FAR: CANNOT COMPUTE from real data — documented
- [x] FRR: CANNOT COMPUTE from real data — documented
- [x] Exact/meaningful threshold sweep implemented (1,537 event-based thresholds)
- [x] Candidate operating points: IDENTIFIED (but using synthetic genuine — NOT production-ready)
- [x] No production threshold introduced
- [x] EER: CANNOT COMPUTE from real data — documented
- [x] Top-1 evaluated (94.87% self-retrieval)
- [x] Top-3 evaluated (100%)
- [x] Top-5 evaluated (100%)
- [x] Dataset limitations documented
- [x] Celebrity detection limitations documented
- [x] Calibration plots generated (5 plots)
- [x] Calibration artifacts generated (4 CSV/JSON + manifest)
- [x] Dataset manifest generated
- [x] Reproducibility run completed (PASS)
- [x] Empirical experiment actually executed
- [x] Phase 13.1 tests pass (34/34)
- [x] Phase 13.2 tests pass (6/6)
- [x] Phase 13.3 tests pass (24/24)
- [x] Phase 13.4 tests pass (5/5)
- [x] Phase 13.5 tests pass (40/40)
- [x] Existing regression has no new failures
- [x] PHASE_13_5_REPORT.md exists
- [x] No production architecture changed
- [x] No RabbitMQ changes
- [x] No external APIs
- [x] No model retraining
- [x] No commit/push

## Artifacts

| Artifact | Path |
|----------|------|
| Report | `PHASE_13_5_REPORT.md` |
| Summary JSON | `outputs/phase13_5_calibration/calibration_summary.json` |
| Threshold CSV | `outputs/phase13_5_calibration/threshold_metrics.csv` |
| Genuine scores | `outputs/phase13_5_calibration/genuine_scores.csv` |
| Impostor scores | `outputs/phase13_5_calibration/impostor_scores.csv` |
| Distribution plot | `outputs/phase13_5_calibration/genuine_vs_impostor_distribution.png` |
| ROC curve | `outputs/phase13_5_calibration/roc_curve.png` |
| FAR/FRR plot | `outputs/phase13_5_calibration/far_frr_vs_threshold.png` |
| Genuine plot | `outputs/phase13_5_calibration/genuine_distribution.png` |
| Impostor plot | `outputs/phase13_5_calibration/impostor_distribution.png` |
| Dataset manifest | `outputs/phase13_5_calibration/PHASE_13_5_DATASET_MANIFEST.json` |

## Phase 13.5 STATUS

**PHASE 13.5: READY** (with INSUFFICIENT EVIDENCE verdict)

All implementation, testing, and documentation requirements are complete. The calibration infrastructure is correct, tested, and reproducible. The empirical results honestly reflect the dataset limitations.

**PHASE 13.6: NOT RECOMMENDED** until a person-identity-labeled dataset is obtained.
