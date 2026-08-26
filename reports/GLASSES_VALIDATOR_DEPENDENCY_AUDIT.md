# GlassesValidator Dependency Audit Report

**Date:** 2026-08-26
**Scope:** Complete repository audit for GlassesValidator removal

---

## A. GlassesValidator / Classifier Infrastructure (REMOVE)

### A1. Core Source Files

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `validators/glasses_validator.py` | 138 | GlassesValidator class definition | DELETE |
| `services/glasses_detector_classifier.py` | 364 | GlassesDetectorClassifier service (wraps `glasses-detector` library) | DELETE |
| `services/eyewear_classifier.py` | 47 | EyewearClassifier ABC (abstract interface) | DELETE |
| `models/eyewear_type.py` | 21 | EyewearType enum (NONE, CLEAR_GLASSES, PRESCRIPTION_GLASSES, SUNGLASSES) | DELETE |
| `models/eyewear_prediction.py` | 56 | EyewearPrediction dataclass | DELETE |

### A2. Test Files

| File | Lines | Purpose | Action |
|------|-------|---------|--------|
| `tests/validators/test_glasses_validator.py` | 390 | GlassesValidator unit tests | DELETE |
| `tests/services/test_glasses_detector_classifier.py` | 380 | GlassesDetectorClassifier unit tests | DELETE |
| `tests/models/test_eyewear_prediction.py` | 123 | EyewearPrediction model tests | DELETE |

### A3. Files Requiring Modification

| File | Lines | Change Required |
|------|-------|----------------|
| `pipeline/validator_factory.py` | 8, 14, 29, 37 | Remove GlassesDetectorClassifier import + GlassesValidator import + instantiation |
| `pipeline/validation_orchestrator.py` | 7, 47, 204-209, 241-244 | Remove GLASSES stage references and handling |
| `config/constants.py` | 240-252 | Remove Glasses Validation section (messages + thresholds) |
| `config/constants.py` | 228-230 | Update comment about GlassesValidator |
| `models/validation_type.py` | 20 | Remove GLASSES enum value |
| `models/validation_stage.py` | 13 | Remove GLASSES enum value |
| `evaluation/root_cause.py` | 39-43, 65 | Remove GLASSES root cause mapping + pipeline order entry |
| `scripts/check_models.py` | 18, 56-69, 82 | Remove glasses detector check function + registration |
| `tests/scripts/test_check_models.py` | 11, 65-86 | Remove glasses detector test functions |
| `tests/pipeline/test_validation_orchestrator.py` | 72-73, 81, 88 | Remove GLASSES stage mock + update validator count |
| `validators/occlusion_validator.py` | 25-28 | Update comment referencing GlassesValidator |
| `tests/validators/test_face_visibility_validator.py` | 1989 | Update comment referencing GlassesValidator |
| `requirement.txt` | 9 | Remove `glasses-detector` dependency |
| `requirements-lock.txt` | 8 | Remove `glasses-detector==1.0.4` lock entry |

### A4. Bytecode Cache

| File | Action |
|------|--------|
| `validators/__pycache__/glasses_validator.cpython-314.pyc` | DELETE |
| `validators/__pycache__/glasses_validator.cpython-312.pyc` | DELETE |

---

## B. EYE_GLASS Semantic Segmentation Infrastructure (PRESERVE)

### B1. FacePart Definition

| File | Line | Purpose |
|------|------|---------|
| `models/parsing/face_part.py` | 38-39 | `EYE_GLASS = 6` enum definition |

### B2. Parser Infrastructure

| File | Purpose |
|------|---------|
| `services/face_parser_service.py` | BiSeNet produces 19-class output including EYE_GLASS |
| `ai_models/bisenet/bisenet_resnet18.onnx` | Protected ONNX model with EYE_GLASS class |

### B3. Fusion Infrastructure

| File | Purpose |
|------|---------|
| `models/parsing/face_parsing_result.py` | FaceParsingResult contains EYE_GLASS pixels |
| Phase 3/4 fusion code | Fusion output includes EYE_GLASS class |

### B4. Validators Using EYE_GLASS (PRESERVE)

| File | Lines | Usage |
|------|-------|-------|
| `validators/face_visibility_validator.py` | 51, 74, 179 | EYE_GLASS as secondary evidence for eye visibility under glasses |
| `validators/occlusion_validator.py` | 26-28 | EYE_GLASS is intentionally NOT treated as occlusion |

### B5. Evaluation Using EYE_GLASS (PRESERVE)

| File | Lines | Usage |
|------|-------|-------|
| `evaluation/parser_qa.py` | 52, 62, 115, 122, 434-435 | EYE_GLASS pixel counts in QA metrics |
| `evaluation/overlay_renderer.py` | 23 | EYE_GLASS color mapping `(0, 255, 255)` |

### B6. Config References to EYE_GLASS (PRESERVE)

| File | Lines | Purpose |
|------|-------|---------|
| `config/constants.py` | 228-230 | Comment explaining why EYE_GLASS is excluded from prohibited parts |
| `freeze_pilot.py` | 16 | `6:'EYE_GLASS'` class ID mapping |

---

## C. Summary

| Category | Count | Action |
|----------|-------|--------|
| Source files to DELETE | 5 | Remove GlassesValidator infrastructure |
| Test files to DELETE | 3 | Remove GlassesValidator tests |
| Files to MODIFY | 14 | Remove references from production pipeline |
| Bytecode cache to DELETE | 2 | Clean compiled files |
| EYE_GLASS files to PRESERVE | 10+ | All face parsing infrastructure intact |

---

## D. Key Distinction

| Component | What It Is | Action |
|-----------|-----------|--------|
| **GlassesValidator** | Separate classifier that detects sunglasses vs clear glasses vs no glasses | REMOVE |
| **GlassesDetectorClassifier** | External `glasses-detector` library adapter | REMOVE |
| **EyewearClassifier ABC** | Abstract interface for glasses classification | REMOVE |
| **EyewearType/EyewearPrediction** | Domain models for glasses classification | REMOVE |
| **ValidationType.GLASSES** | Validation type for glasses classification check | REMOVE |
| **ValidationStage.GLASSES** | Pipeline stage for glasses classification | REMOVE |
| **FacePart.EYE_GLASS** | BiSeNet semantic class ID=6 for eyeglasses/sunglasses | PRESERVE |
| **EYE_GLASS in 19-class output** | BiSeNet's semantic segmentation output | PRESERVE |
| **EYE_GLASS in FaceParsingResult** | Parsed mask containing EYE_GLASS pixels | PRESERVE |
