# Phase 5: Semantic Reporting Consistency Fix

## 1. Root Cause

The forensic audit confirmed that `SemanticEvidenceEngine` had **two independent implementations** of the semantic eye-visibility logic:

| Method | Used by | Glasses fallback |
|--------|---------|-----------------|
| `compute_eye_evidence()` | `Evaluator` (reporting) | **NONE** (parser_conf stays 0.0) |
| `is_eye_visible()` | `FaceVisibilityValidator` (production) | **0.65 override** when parser=0, EYE_GLASS, landmarks valid |

For transparent glasses, this caused reports to display `parser_confidence=0.0` while production actually evaluated the evidence using `parser_confidence=0.65`. The `final_confidence` and `passed` values in the report could differ from the production decision.

## 2. Files Changed

| File | Change |
|------|--------|
| `reasoning/semantic_engine.py` | Added `_compute_effective_parser_confidence()`, updated `compute_eye_evidence()` and `is_eye_visible()` to use it |
| `tests/reasoning/test_semantic_engine.py` | Added 8 focused regression tests |

**No other files modified.**

## 3. Exact Semantic Path Before

```
compute_eye_evidence() [REPORT]:
  parser_conf = _compute_parser_confidence()  → 0.0 (raw)
  # No glasses fallback applied
  SemanticEvidence(parser_confidence=0.0, ...)
  → final_confidence = 0.5447, passed = True

is_eye_visible() [PRODUCTION]:
  parser_conf = _normalize_ratio()  → 0.0 (raw)
  if parser_conf == 0.0 and EYE_GLASS and landmark > 0.0:
      parser_conf = 0.65  ← OVERRIDE
  SemanticEvidence(parser_confidence=0.65, ...)
  → final_confidence = 0.8123, passed = True
```

**Discrepancy:** Report shows `parser_confidence=0.0, final_confidence=0.5447` but production used `parser_confidence=0.65, final_confidence=0.8123`.

## 4. Exact Semantic Path After

```
_compute_effective_parser_confidence() [SHARED]:
  parser_conf = _compute_parser_confidence()  → 0.0 (raw)
  landmark_conf = _compute_landmark_confidence()  → 1.0
  if parser_conf == 0.0 and EYE_GLASS and landmark_conf > 0.0:
      return 0.65  ← SINGLE SOURCE OF TRUTH

compute_eye_evidence() [REPORT]:
  parser_conf = _compute_effective_parser_confidence()  → 0.65
  SemanticEvidence(parser_confidence=0.65, ...)
  → final_confidence = 0.8123, passed = True

is_eye_visible() [PRODUCTION]:
  parser_conf = _compute_effective_parser_confidence()  → 0.65
  SemanticEvidence(parser_confidence=0.65, ...)
  → final_confidence = 0.8123, passed = True
```

**Both paths now produce identical results.**

## 5. How the 0.65 Fallback Is Now Represented

The 0.65 fallback lives in a **single private method**:

```python
def _compute_effective_parser_confidence(self, part, min_ratio=0.0015):
    parser_conf = self._compute_parser_confidence(part, min_ratio)
    landmark_conf = self._compute_landmark_confidence(part)
    if (
        parser_conf == 0.0
        and self._parsing.has_part(FacePart.EYE_GLASS)
        and landmark_conf > 0.0
    ):
        return 0.65
    return parser_conf
```

Both `compute_eye_evidence()` and `is_eye_visible()` call this method. There is no second implementation.

## 6. Confirmation: Production Behavior Unchanged

| Scenario | Before | After | Changed? |
|----------|--------|-------|----------|
| No glasses, eye pixels present | PASS | PASS | No |
| Transparent glasses (0.65 fallback) | PASS | PASS | No |
| Opaque sunglasses (no landmarks) | FAIL | FAIL | No |
| Semi-transparent sunglasses | PASS | PASS | No |
| Borderline pose with glasses | PASS (production) / varies (report) | PASS (both) | Report now consistent |
| No face/landmarks | FAIL | FAIL | No |

**The production `is_eye_visible()` logic is byte-for-byte identical in behavior.**

## 7. Confirmation: Weights/Thresholds Unchanged

| Constant | Value | Changed? |
|----------|-------|----------|
| `SEMANTIC_PARSER_WEIGHT` | 0.35 | No |
| `SEMANTIC_LANDMARK_WEIGHT` | 0.20 | No |
| `SEMANTIC_POSE_WEIGHT` | 0.20 | No |
| `SEMANTIC_OCCLUSION_WEIGHT` | 0.10 | No |
| `SEMANTIC_DECISION_THRESHOLD` | 0.50 | No |

## 8. Confirmation: eye_support_confidence NOT Added to Formula

`SemanticEvidence.__post_init__()` formula remains:

```python
score = (
    parser_confidence * 0.35
    + landmark_confidence * 0.20
    + pose_confidence * 0.20
    + occlusion_confidence * 0.10
) / 0.85
```

`eye_support_confidence` is **not** in this formula. It remains as a display-only field in the report.

## 9. Test Results

### Focused Regression Tests (new)

```
tests/reasoning/test_semantic_engine.py::TestReportingConsistency
  test_no_glasses_paths_agree                              PASSED
  test_transparent_glasses_065_fallback_reflected          PASSED
  test_opaque_sunglasses_no_landmarks_paths_agree          PASSED
  test_semi_transparent_sunglasses_paths_agree             PASSED
  test_borderline_pose_065_tips_balance                    PASSED
  test_compute_eye_evidence_uses_effective_parser_confidence PASSED
  test_effective_parser_confidence_no_glasses              PASSED
  test_effective_parser_confidence_glasses_no_landmarks    PASSED
```

### Full Test Suite

| Metric | Before | After |
|--------|--------|-------|
| Total collected | 1046 | 1054 |
| Passed | 1020 | 1028 |
| Failed | 26 | 26 |
| New regressions | — | **0** |

The 26 failures are all pre-existing in `test_face_visibility_validator.py` and are unrelated to this change.

## 10. Before/After Examples

### Transparent Glasses

| Field | Before (report) | After (report) | Production |
|-------|----------------|----------------|------------|
| `parser_confidence` | **0.0** | **0.65** | 0.65 |
| `final_confidence` | **0.5447** | **0.8123** | 0.8123 |
| `passed` | True | True | True |

### Sunglasses (opaque, no landmarks)

| Field | Before | After | Production |
|-------|--------|-------|------------|
| `parser_confidence` | 0.0 | 0.0 | 0.0 |
| `final_confidence` | 0.3529 | 0.3529 | 0.3529 |
| `passed` | False | False | False |

### No Glasses

| Field | Before | After | Production |
|-------|--------|-------|------------|
| `parser_confidence` | 1.0 | 1.0 | 1.0 |
| `final_confidence` | 0.9564 | 0.9564 | 0.9564 |
| `passed` | True | True | True |

### Short-Circuited Image (cb781)

FaceVisibilityValidator remains absent from the production validator list when CHEAP validation fails. The Evaluator still performs independent semantic analysis for observability, but the reported values now match the production logic.

## Acceptance Criteria

- [x] Production semantic decision behavior is unchanged
- [x] Reporting semantic evidence uses the same effective production logic
- [x] 0.65 EYE_GLASS/landmark fallback remains intact
- [x] No second independent implementation of the fallback remains
- [x] Semantic weights unchanged
- [x] Semantic threshold unchanged
- [x] eye_support_confidence NOT added to the formula
- [x] No glasses classifier introduced
- [x] EYE_GLASS segmentation unchanged
- [x] FUSED parser unchanged
- [x] CHEAP short-circuit behavior unchanged
- [x] Report does not falsely claim skipped validators executed
- [x] Transparent-glasses behavior remains correct
- [x] Sunglasses behavior remains unchanged
- [x] No new regressions
- [x] Focused regression tests added
- [x] Full test suite executed
- [x] Regression report generated
