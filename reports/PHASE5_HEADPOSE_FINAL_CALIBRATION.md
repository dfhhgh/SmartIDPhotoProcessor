# Phase 5 HeadPose Final Calibration Report

**Date:** 2026-08-26  
**Scope:** HeadPose validator threshold tuning only  

---

## Files Modified

| File | Change |
|------|--------|
| `config/constants.py:218-219` | `HEAD_POSE_YAW_MAX_DEGREES` 20.0 → 22.0, `HEAD_POSE_ROLL_MAX_DEGREES` 15.0 → 20.0 |

No other files modified. No validator logic, scoring formula, model, or unrelated code changed.

---

## Threshold Changes

| Constant | Old Value | New Value |
|----------|-----------|-----------|
| `HEAD_POSE_PITCH_MAX_DEGREES` | 20.0 | 20.0 (unchanged) |
| `HEAD_POSE_YAW_MAX_DEGREES` | 20.0 | **22.0** |
| `HEAD_POSE_ROLL_MAX_DEGREES` | 15.0 | **20.0** |

---

## Per-Image Impact (InsightFace raw pose data)

| Image | Pitch | Yaw | Roll | Old | New | Reason |
|-------|-------|-----|------|-----|-----|--------|
| pexels_4442025 | +1.3 | +5.6 | **+16.3** | FAIL(roll) | **PASS** | Roll was 1.3° over old 15° limit, now within 20° |
| pexels_5049702 | -3.1 | -5.2 | **-15.7** | FAIL(roll) | **PASS** | Roll was 0.7° over old 15° limit, now within 20° |
| pexels_5974073 | -3.8 | **-20.3** | -5.9 | FAIL(yaw) | **PASS** | Yaw was 0.3° over old 20° limit, now within 22° |

Images that remain FAIL under all configurations:
- pexels_3772712: yaw=37.7° (extreme)
- pexels_4057039: yaw=23.6° (extreme)
- pexels_4836878: pitch=23.1° (extreme)
- prescription_eyeglasses_lady: yaw=23.3° (extreme)

---

## Experiment Results

### HeadPose Validator (end-to-end pipeline)

| Metric | Before | After |
|--------|--------|-------|
| HEAD_POSE PASS | 22 | **29** |
| HEAD_POSE FAIL | 7 | **4** |
| HEAD_POSE SKIPPED | 13 | 9 |

### Overall Pipeline

| Metric | Before (20/20/15) | After (20/22/20) |
|--------|-------------------|-------------------|
| VALID | 10 | 11 |
| INVALID | 29 | 28 |
| PROCESSING_ERROR | 3 | 3 |

### Net Changes

- **3 images recovered** by HEAD_POSE (4442025, 5049702, 5974073 now PASS HEAD_POSE)
- **1 image (pexels_5974073)** transitions from INVALID → VALID (FACE_SIZE also passes at 0.084)
- **0 previously-VALID images became INVALID** (zero regressions)

---

## Test Suite Results

| Suite | Result |
|-------|--------|
| HeadPoseValidator tests | **101/101 passed** |
| Full test suite | **1046 passed, 26 failed (pre-existing), 0 new failures** |

---

## Model Hash Verification

| Model | Hash | Status |
|-------|------|--------|
| `ai_models/bisenet/bisenet_resnet18.onnx` | `2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9` | **UNCHANGED** |

---

## Confirmation

- ✅ Only HeadPose thresholds modified
- ✅ No validator logic changed
- ✅ No scoring formula changed
- ✅ No FaceSize changes
- ✅ No FaceVisibility changes
- ✅ No Occlusion changes
- ✅ No BiSeNet/model changes
- ✅ No InsightFace changes
- ✅ No face detection/selection changes
- ✅ No new models or classifiers introduced
- ✅ Zero regressions
- ✅ All 101 HeadPose tests pass

**This is the FINAL HeadPose calibration for the Phase 5 dataset.**
