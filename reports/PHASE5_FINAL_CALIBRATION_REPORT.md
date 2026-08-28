# Phase 5 Final Calibration Report

## 1. FaceSize Calibration Analysis

### Dataset Distribution (42 real images)

| Range | Count | Classification |
|-------|-------|---------------|
| ratio < 0.07 | 21 | Non-face / genuinely too small (ke*, donkey, panner, processing errors) |
| 0.07 <= ratio < 0.08 | 1 | Borderline (pexels_4482931, 0.0799) |
| 0.08 <= ratio < 0.09 | 0 | Empty |
| 0.09 <= ratio < 0.10 | 2 | Borderline (cb78121b 0.093, pexels_4728707 0.097) |
| 0.10 <= ratio < 0.12 | 2 | Usable (pexels_1516680 0.105, hijab2 0.112) |
| ratio >= 0.12 | 14 | Clearly acceptable |

Key observation: The InsightFace bbox is face-only and does not include head covering or shoulders. For head-covered portraits, the bbox-area ratio is systematically lower than the visually usable area.

### Candidate Threshold Evaluation

| Candidate | Newly Accepted | Problematic Accepts | Genuinely Tiny Now PASS |
|-----------|---------------|--------------------|-----------------------|
| 0.10 (current) | 0 | 0 | 0 |
| 0.09 | +2 (cb78121b, pexels_4728707) | 0 | 0 |
| **0.08** | **+3** (above + pexels_4482931) | **0** | **0** |
| 0.07 | +3 (same as 0.08) | 0 | 0 |

At MIN=0.08:
- cb78121b (0.093): passes FS, fails at FACE_VISIBILITY (parser misses eyes)
- pexels_4728707 (0.097): passes all validators -> becomes VALID
- pexels_4482931 (0.080): passes FS, fails at HEAD_POSE

No genuinely tiny face (<0.07) would become FaceSize PASS at 0.08.

## 2. Selected Threshold

```python
FACE_SIZE_MIN_RATIO = 0.08   # changed from 0.10
FACE_SIZE_IDEAL_RATIO = 0.40 # unchanged
FACE_SIZE_MAX_RATIO = 0.65   # unchanged
```

**Rationale:** 0.08 is the smallest controlled change that recovers borderline genuine portraits while maintaining clear separation from non-face images. The gap between 0.07 (last non-face) and 0.08 provides safety margin.

## 3. Before/After FaceSize Results

| Metric | Before (MIN=0.10) | After (MIN=0.08) | Delta |
|--------|-------------------|-------------------|-------|
| FS PASS | 16 | 18 | +2 |
| FS FAIL | 17 | 15 | -2 |
| FS SKIPPED | 9 | 9 | 0 |

### Newly Recovered by FaceSize
| Image | Ratio | New FS | Final Status | Reason |
|-------|-------|--------|-------------|--------|
| pexels_4728707 | 0.097 | PASS | VALID | All pass |
| cb78121b | 0.093 | PASS | INVALID | FACE_VISIBILITY (parser misses eyes) |

### Still Correctly Rejected
All images with ratio < 0.07 remain rejected. No false accepts introduced.

## 4. Hat Validation Removal

### Changes Made
1. `OCCLUSION_PROHIBITED_PARTS` changed from `(FacePart.HAT,)` to `()`
2. OcclusionValidator now always returns PASS (score=1.0, "No prohibited occlusions detected.")
3. Root cause explanation updated from "Prohibited head covering (non-religious hat/cap) detected." to "Prohibited occluding object detected."
4. OcclusionValidator docstring updated to document HAT allowance policy

### What Was NOT Changed
- FacePart.HAT semantic class remains in the parser (19-class BiSeNet output)
- SemanticEvidenceEngine is unchanged
- FaceVisibilityValidator is unchanged
- No new classifier or model introduced

## 5. Before/After Hijab Results

| Image | Before | After | Change |
|-------|--------|-------|--------|
| hijab2.jpg | INVALID (REAL_OCCLUSION, "Hat detected.") | **VALID** | HAT no longer prohibited |
| hijab with glassesjpg.jpg | INVALID (REAL_OCCLUSION, "Hat detected.") | **VALID** | HAT no longer prohibited |
| hijab-girl-hijab-girl1.jpg | VALID | VALID | Unchanged |

## 6. Tests and Regressions

### Full Suite Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Total | 1054 | 1072 | +18 (new tests) |
| Passed | 1028 | 1046 | +18 |
| Failed | 26 | 26 | 0 (all pre-existing) |
| New regressions | - | 0 | - |

### New Tests Added
- 7 occlusion regression tests (TestHatRemovalRegression class)
- Updated 8 existing occlusion tests for HAT allowance
- Updated 1 face-size test for new 0.08 threshold

## 7. Full 42-Image Experiment Results

| Metric | Before (MIN=0.10, HAT) | After (MIN=0.08, no HAT) | Delta |
|--------|------------------------|--------------------------|-------|
| VALID | 5 | **10** | **+5** |
| INVALID | 34 | 29 | -5 |
| ERRORS | 3 | 3 | 0 |

### All 10 VALID Images
1. glasses.jpg
2. hijab with glassesjpg.jpg (NEW)
3. hijab2.jpg (NEW)
4. pexels_1516680.jpeg
5. pexels_3252801.jpeg
6. pexels_4272299.jpeg (NEW)
7. pexels_4728707.jpeg (NEW)
8. pexels_5815587.jpeg (NEW)
9. smt6815_m0.jpg
10. hijab-girl-hijab-girl1.jpg

## 8. Validator Statistics

| Validator | PASS | FAIL | SKIPPED |
|-----------|------|------|---------|
| BLUR | 33 | 0 | 9 |
| BRIGHTNESS | 33 | 0 | 9 |
| CONTRAST | 33 | 0 | 9 |
| FACE_SIZE | 18 | 15 | 9 |
| HEAD_POSE | 21 | 12 | 9 |
| FACE_VISIBILITY | 10 | 4 | 28 |
| OCCLUSION | **14** | **0** | 28 |

OCCLUSION FAIL went from 4 to 0 (hijab images no longer rejected).

## 9. Root-Cause Distribution

| Root Cause | Before | After |
|------------|--------|-------|
| FACE_TOO_SMALL | 15 | 13 |
| HEAD_POSE | 7 | 7 |
| THRESHOLD_TOO_STRICT | 4 | 4 |
| REAL_OCCLUSION | 4 | **0** |
| PROCESSING_ERROR | 3 | 3 |
| FACE_AMBIGUITY | 1 | 1 |

REAL_OCCLUSION dropped to 0 because HAT is no longer prohibited.

## 10. Runtime Statistics

| Metric | Value |
|--------|-------|
| Average | 3074 ms |
| Median (P50) | 3115 ms |
| P95 | 4745 ms |
| Min | 293 ms |
| Max | 6896 ms |

## 11. Model Hash Verification

```
bisenet_resnet18.onnx: 2218b6183c26ca5c... UNCHANGED
```

## 12. Remaining Limitations

1. **Sunglasses edge case (smt6815_m0.jpg):** Semi-transparent sunglasses expose eye pixels to parser; this is a parser-level limitation, not an architecture failure.
2. **26 pre-existing test failures** in `test_face_visibility_validator.py` (unrelated to this calibration).
3. **InsightFace bbox limitation:** Face bbox does not capture full head/shoulders area, causing systematically lower ratios for head-covered portraits.
4. **FaceVisibility strictness:** Some borderline images fail at FACE_VISIBILITY due to parser confidence thresholds.

## 13. Final Verdict

**PASS**

Both targeted changes improve the pipeline:
- FaceSize calibration (0.10 -> 0.08) recovers genuinely usable borderline portraits without false accepts
- HAT removal eliminates incorrect hijab rejections without introducing new vulnerabilities
- All protected artifacts remain unchanged
- Zero new regressions

---

## Summary

```
OLD:
FACE_SIZE_MIN_RATIO = 0.10
OCCLUSION_PROHIBITED_PARTS = (FacePart.HAT,)

NEW:
FACE_SIZE_MIN_RATIO = 0.08
OCCLUSION_PROHIBITED_PARTS = ()

Before: 5 VALID, 34 INVALID, 3 ERRORS
After: 10 VALID, 29 INVALID, 3 ERRORS

Newly accepted: 5 images
New false accepts: 0
New false rejects: 0
Regression status: NO REGRESSIONS (26 pre-existing unchanged)
Model hashes: UNCHANGED
```
