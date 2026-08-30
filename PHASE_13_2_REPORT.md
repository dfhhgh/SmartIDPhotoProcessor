# Phase 13.2 — Reference Dataset & FAISS Index Builder

**Date:** 2026-08-30
**Status:** READY — all definition-of-done items satisfied

---

## 1. Objective

Phase 13.2 implements a clean, reproducible reference-dataset ingestion and FAISS-index construction tool (`search/index_builder.py`).  It transforms a local reference-image dataset directory into two core artifacts:
1. `reference_index.faiss` — FAISS `IndexFlatIP` containing 512-D float32 L2-normalized face embeddings.
2. `metadata.json` — Mapping table associating FAISS vector IDs to person IDs, labels, and relative image paths.

---

## 2. Architecture

```
reference_dataset/
    ├── person_001/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── person_002/
        └── img1.jpg
             │
             ▼
      IndexBuilder.build()
             │
             ├─────────► reference_index.faiss (FAISS IndexFlatIP)
             │
             └─────────► metadata.json (Vector ID → Person/Image mapping)
```

### Key Components

| File | Purpose |
|------|---------|
| `search/index_builder.py` | Core dataset discovery, InsightFace embedding extraction, validation, and artifact generation. |
| `tests/test_search/test_index_builder.py` | Comprehensive test suite covering dataset validation, ingestion, reproducibility, and end-to-end search smoke tests. |

---

## 3. Dataset Structure

- **Directory layout:** Hierarchical by person identifier (`dataset_dir/person_id/image.jpg`).
- **Supported extensions:** `.jpg`, `.jpeg`, `.png` (case-insensitive).
- **Person identifier:** Derived directly from the directory name (`person_id = pdir.name`).
- **Deterministic ordering:**
  - Person directories are sorted alphabetically (`sorted(dataset_dir.iterdir())`).
  - Image files within each person directory are sorted alphabetically (`sorted(pdir.iterdir())`).
  - This guarantees stable vector ID assignment `[0, N-1]` across independent builds.

---

## 4. Embedding Extraction Path

1. **Image Loading:** OpenCV (`cv2.imread`) loads each reference image as a BGR `uint8` numpy array.
2. **Face Detection & Recognition:** Reuses `FaceService().get_model()` (InsightFace `FaceAnalysis` with `buffalo_l` and `w600k_r50.onnx`).
3. **Face Selection:** If multiple faces are detected, `FaceSelector.select()` resolves the primary face deterministically.
4. **Embedding Retrieval:** Accesses `face.normed_embedding` (512-D float32 L2-normalized vector).
5. **Validation:** Passes through Phase 13.1 `EmbeddingValidator` to verify dimension (512), dtype (`float32`), finiteness, non-zero norm, and normalization.

---

## 5. Metadata Schema (`metadata.json`)

```json
{
  "schema_version": 1,
  "embedding_dimension": 512,
  "metric": "inner_product",
  "normalized": true,
  "total_vectors": 3,
  "total_persons": 2,
  "records": [
    {
      "vector_id": 0,
      "person_id": "person_001",
      "label": "person_001",
      "image": "person_001/img_001.jpg"
    }
  ]
}
```

- **Separation of concerns:** FAISS owns the vectors; `metadata.json` owns descriptive associations. Embeddings are never stored redundantly in JSON.
- **Relative paths:** Image paths are stored relative to the dataset root (`person_id/filename.jpg`) to prevent absolute machine-specific path leakage.

---

## 6. Vector ID Strategy & Determinism

- Vector IDs are assigned sequentially starting at `0` based on the globally sorted iteration order of `(person_id, image_path)`.
- Rebuilding the dataset produces identical vector IDs and identical search results within floating-point tolerance (`atol=1e-6`).

---

## 7. Failure Isolation

- A single corrupted, unreadable, or zero-face image does **not** abort the build.
- Such images are recorded in `BuildReport.skipped_records` with an explicit reason (`IMAGE_UNREADABLE`, `NO_FACE`, `MULTIPLE_FACES_SELECTION_FAILED`, etc.) while processing continues for remaining images.

---

## 8. Test Results

### Phase 13.2 & 13.1 Tests

```
63 passed, 5 warnings in 18.41s
```

| Test Suite | Tests | Status |
|-----------|-------|--------|
| `test_embedding_validator.py` | 16 | ALL PASS |
| `test_flat_index.py` | 34 | ALL PASS |
| `test_benchmark.py` | 7 | ALL PASS |
| `test_index_builder.py` | 6 | ALL PASS |
| **Total** | **63** | **ALL PASS** |

### Existing Regression Suite

```
1050 passed, 27 failed, 9 errors in 166.96s
```
- **Zero new failures or regressions.** All 27 failures and 9 errors are pre-existing.

---

## 9. Reproducibility & End-to-End Smoke Test

- **Reproducibility Test (`TestIndexBuilderReproducibility`):** Verified that two independent builds on the same reference dataset produce identical vector counts, matching vector IDs, and identical metadata records.
- **End-to-End Search Smoke Test (`TestIndexBuilderEndToEndSearch`):** Verified the complete pipeline:
  `Reference Image → InsightFace Detection → Normed Embedding → FAISS IndexFlatIP → Vector ID → metadata.json → Person ID`.

---

## 10. Build Statistics (Test Run)

- **Input images:** 4 (3 valid, 1 blank/no-face)
- **Accepted embeddings:** 3
- **Skipped images:** 1 (`NO_FACE`)
- **Total persons:** 2
- **Build time:** < 2s

---

## 11. Known Limitations

1. **Exact search only** — Uses `IndexFlatIP` (brute-force inner product). Suitable for < 100K reference vectors.
2. **Single-host builder** — Designed as a local ingestion tool rather than a distributed cluster builder.
3. **Near-duplicate detection (pHash)** — Out of scope for Phase 13.2 (exact file paths are deduplicated naturally by directory traversal).

---

## 12. Definition of Done — Verification

- [x] Dataset builder exists (`search/index_builder.py`)
- [x] Uses existing InsightFace integration (`FaceService`)
- [x] No duplicate recognition model introduced
- [x] Multiple images per person supported
- [x] Deterministic ordering implemented (sorted person IDs + sorted image paths)
- [x] Stable vector IDs produced
- [x] `EmbeddingValidator` reused
- [x] Invalid images/embeddings skipped safely (failure isolation)
- [x] FAISS `IndexFlatIP` reused
- [x] `metadata.json` generated correctly
- [x] FAISS ID → metadata mapping works
- [x] Index/metadata consistency validated (`index.ntotal == len(records)`)
- [x] Rebuilding the same dataset is deterministic
- [x] End-to-end search smoke test passes
- [x] Bad images do not abort the build
- [x] Phase 13.2 & 13.1 tests pass (63/63)
- [x] Existing regression suite has no new failures
- [x] `PHASE_13_2_REPORT.md` exists
- [x] No production integration performed
- [x] No threshold logic added
- [x] No pHash / HNSW / IVF / CLIP added
- [x] No external APIs or RabbitMQ changes added
- [x] No commit/push performed automatically

---

## 13. Conclusion & Next Step

**Phase 13.2 is READY.**

The reference dataset ingestion and FAISS index builder is fully implemented, tested, and verified.

**Recommended next step:** Phase 13.3 — ReverseSearchService (runtime service wrapping FAISS index and metadata for embedding similarity queries).
