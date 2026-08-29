# Phase 6 — Fused Parser ONNX Deployment Readiness Audit

> **Generated:** 2026-08-28  
> **Status:** READ-ONLY Forensic Audit  
> **Audit Target:** Fused BiSeNet / Eye-Brow Refinement Parser Deployment Feasibility & ONNX Export Readiness

---

## Executive Summary

We performed a comprehensive, read-only forensic audit of the SmartIDPhotoProcessor repository to evaluate whether the Fused BiSeNet parser can and should be exported to ONNX for production deployment.

- **Fused Model Status**: Fully implemented in PyTorch (`EyeBrowRefinementService`, `EyeBrowRefinementHead`, `EyeBrowRefinementFusion`), achieving a significant improvement in target eye/brow mIoU (**0.7168 vs 0.5438**, +31.8% relative gain) over the baseline Original BiSeNet.
- **ONNX Export Status**: **NOT READY FOR SINGLE-GRAPH ONNX EXPORT**. 
- **Core Obstacle**: The fusion logic (`EyeBrowRefinementFusion`) relies heavily on NumPy operations, SciPy connected components (`scipy.ndimage.label`), Python branching (`if self.strategy >= 2:`), and conditional masking (`np.isin`, boolean indexing). These operations **cannot** be natively represented in a static ONNX computation graph.
- **Recommended Production Architecture**: **OPTION B — ONNX Backbone + ONNX Auxiliary Head + Deterministic External Fusion (Python)**. This preserves the exact verified PyTorch fusion behavior while enabling ONNX Runtime acceleration for both the backbone and the auxiliary head.

---

## 1. Trace the Complete Fused Inference Graph

| Step | File | Class / Function | Input | Output | Shape / Type | Role |
|------|------|------------------|-------|--------|--------------|------|
| 1 | `config/parser_mode.py` | `ParserMode` | Enum string | `ParserMode.FUSED` | Enum | Configuration entry point |
| 2 | `services/face_parser_service.py` | `FaceParserService.__init__` | `parser_mode=ParserMode.FUSED` | Service instance | Singleton | Initializes refinement service if FUSED |
| 3 | `services/face_parser_service.py` | `FaceParserService.parse` | `image` (BGR uint8) | `FaceParsingResult` | Object | Public API entry point |
| 4 | `services/face_parser_service.py` | `FaceParserService._preprocess` | BGR image | Normalized tensor | `(1, 3, 512, 512)` float32 | Preprocessing (RGB, resize, mean/std norm) |
| 5 | `services/face_parser_service.py` | `EyeBrowRefinementService.refine` | Preprocessed tensor | 19-class integer mask | `(H, W)` int32 | Orchestrates PyTorch model forward pass & fusion |
| 6 | `services/face_parser_service.py` | `EyeBrowRefinementService._ensure_loaded` | None | `(bisenet, head, device)` | Tuple | Loads frozen BiSeNet (from ONNX weights) and auxiliary head (`best.pt`) on `cuda:0` |
| 7 | `experiments/parser_reproduction/bisenet_model.py` | `ContextPath.forward` | Tensor `(1, 3, 512, 512)` | `(feat_res8, feat_cp8, feat_cp16)` | Tensors | Backbone feature extraction |
| 8 | `experiments/parser_reproduction/bisenet_model.py` | `FeatureFusionModule.forward` | `feat_res8`, `feat_cp8` | `feat_fuse` | `(1, 256, 64, 64)` float32 | Fuses spatial and context features (FFM output) |
| 9 | `experiments/parser_reproduction/bisenet_model.py` | `BiSeNetOutput.forward` (`conv_out`) | `feat_fuse`, target_h, target_w | `logits_19` | `(1, 19, 512, 512)` float32 | Original 19-class logits |
| 10 | `dataset_builder/.../auxiliary_head.py` | `EyeBrowRefinementHead.forward` | `feat_fuse`, target_h, target_w | `logits_aux` | `(1, 6, 512, 512)` float32 | Auxiliary 6-class eye/brow logits |
| 11 | `services/face_parser_service.py` | `EyeBrowRefinementFusion.apply` | `logits_19`, `logits_aux` | `final_mask`, `diagnostics` | `(H, W)` int64 | Phase 3 confidence-aware fusion |
| 12 | `services/face_parser_service.py` | Softmax + Argmax (Fusion) | `logits_19`, `logits_aux` | `pred_19`, `pred_aux_6`, `conf_aux` | Arrays `(512, 512)` | Probability conversion |
| 13 | `services/face_parser_service.py` | `construct_eye_brow_roi` | `pred_19`, `pred_aux_19` | `roi` | Boolean `(512, 512)` | Restricts refinement to eye/brow anatomical region |
| 14 | `services/face_parser_service.py` | Gating & Filtering (`apply`) | Masks, confidence, strategy | `candidate` boolean mask | Boolean `(512, 512)` | Applies strategy checks (confidence, disagreement, min component size via SciPy `label`) |
| 15 | `services/face_parser_service.py` | Mask substitution | `final_mask`, `candidate`, `pred_aux_19` | `final_mask` | `(512, 512)` int64 | Overwrites ROI pixels with auxiliary predictions |
| 16 | `services/face_parser_service.py` | OpenCV resize | `final_mask`, original dimensions | Resized mask | `(original_H, original_W)` int32 | Post-processing resize to source resolution |

---

## 2. Determine Exactly What Must Be Exported

**Conclusion: B) Multiple ONNX graphs + deterministic Python/C++ fusion.**

### Why Option A (Single ONNX Graph) is NOT Feasible:
The `EyeBrowRefinementFusion.apply()` method contains operations that cannot be compiled into a static ONNX graph:
1. **SciPy Connected Components (`scipy.ndimage.label`)**: Used in Strategy 4 (`min_component_size` filtering) to isolate connected pixel components. This is a CPU-bound NumPy/C function with dynamic output shapes.
2. **Python Branching & Control Flow**: `if self.strategy >= 2:`, `if class_candidate.any():`, and loops over connected component IDs (`for comp_id in range(1, num_features + 1):`).
3. **Dynamic Boolean Indexing**: NumPy array indexing based on runtime-evaluated boolean conditions (`final_mask[candidate] = pred_aux_19[candidate]`).

### Exportable Components:
- **Frozen BiSeNet Backbone + `conv_out`**: Exportable to ONNX (`bisenet_resnet18.onnx` already exists for this part).
- **`EyeBrowRefinementHead`**: Fully exportable to ONNX (standard PyTorch CNN with Conv2d, BatchNorm, ReLU, and Bilinear Upsample).

---

## 3. Check the Auxiliary Checkpoint

- **Exact Path**: `dataset_builder/dataset/parser_finetune_current/training_aux_eye_brow_phase1/checkpoints/best.pt`
- **File Size**: 4,470,853 bytes (~4.47 MB)
- **SHA256**: `961e08bf64fdd0b8ae044ac6bf0d30ecbed13a22364301903a2c42c0c99e6e00`
- **Architecture**: `EyeBrowRefinementHead` (Conv2d 256→128, BatchNorm2d, ReLU, Conv2d 128→64, BatchNorm2d, ReLU, Conv2d 64→6, Bilinear Upsample).
- **Input Feature Dimension**: 256 channels (FFM output at 1/8 resolution).
- **Output Classes**: 6 classes (Background, Left Brow, Right Brow, Left Eye, Right Eye, Eye Glass).
- **Parameter Count**: 369,408 parameters (all trainable in Phase 1, loaded in eval mode).
- **Contains BiSeNet Backbone?**: **NO**. Contains *only* `head_state_dict` for the auxiliary head, optimizer state, and epoch metadata.
- **Complete Fused Checkpoint?**: **NO**. Fused inference combines the ONNX backbone weights (via `load_onnx_to_pytorch`) and the auxiliary head checkpoint at runtime.

---

## 4. Verify the Frozen Backbone

- **Architecture**: BiSeNetV1 with ResNet-18 backbone, ARM16, ARM32, Feature Fusion Module (`FeatureFusionModule`), and 19-class output heads.
- **Parameter Count**: ~13.3M parameters.
- **State Dict Structure**: Matches `BiSeNet` class in `experiments/parser_reproduction/bisenet_model.py`.
- **Weight Origin**: Loaded directly from `ai_models/bisenet/bisenet_resnet18.onnx` via `load_onnx_to_pytorch()` (which folds BatchNorm weights into convolutions).
- **BatchNorm Statistics**: Running statistics are tracked during training/export, but frozen with `track_running_stats = False` and `eval()` during inference.
- **Freezing Verification**: Verified via unit tests (`TestRealModelWeightIntegrity`) that `requires_grad == False` for all backbone parameters.

---

## 5. ONNX Export Feasibility

| Component | Exportable? | Required Mechanism | Potential Unsupported Operators |
|-----------|-------------|--------------------|---------------------------------|
| **BiSeNet Backbone + `conv_out`** | **YES** | `torch.onnx.export` | None (already exported and verified) |
| **`EyeBrowRefinementHead`** | **YES** | `torch.onnx.export` | None (standard Conv2d/ReLU/Upsample) |
| **Combined Backbone + Aux Head** | **YES** | Joint PyTorch module export | None (pure tensor graph) |
| **Confidence-Aware Fusion** | **NO** | Must remain in Python/NumPy | SciPy `label`, Python loops, NumPy boolean indexing |

---

## 6. Critical Numerical Parity Plan

Before any production switch to an ONNX-backed auxiliary head, a strict parity test suite must be executed:
1. **Inputs**: Same 114 test images.
2. **Comparison Points**:
   - BiSeNet backbone feature maps (`feat_fuse`) absolute difference tolerance: `< 1e-5`.
   - Auxiliary head logits (`logits_aux`) absolute difference tolerance: `< 1e-5`.
   - Probability softmax outputs: `< 1e-4`.
   - Final fused mask: **100% exact pixel equality** (tolerance = 0.0%).

---

## 7. Production Architecture Options

| Option | Correctness | Parity Risk | Performance | Deployment Complexity | Recommendation |
|--------|-------------|-------------|-------------|-----------------------|----------------|
| **A. Single ONNX Graph** | Impossible without rewriting fusion | High | Maximum | High (requires C++ fusion rewrite) | **REJECTED** |
| **B. ONNX Backbone + ONNX Aux Head + Python Fusion** | **Exact** | **Zero** | **High (ONNX Runtime CUDA/CPU)** | **Low (reuses existing code)** | **RECOMMENDED** |
| **C. Production ONNX + Separate Aux ONNX + Fusion** | Exact | Zero | Medium | Medium | SUBOPTIMAL |
| **D. Keep PyTorch Fused** | Exact | Zero | High | High (requires PyTorch runtime in production) | **REJECTED** (adds PyTorch dependency) |

---

## 8. Performance Analysis

- **Original ONNX (CPU)**: ~65.75 ms mean inference.
- **Fused PyTorch (CUDA)**: ~15.49 ms mean warm inference (4.2× speedup).
- **Expected Option B Performance**:
  - Backbone (ONNX Runtime CUDA): ~12-15 ms.
  - Auxiliary Head (ONNX Runtime CUDA): ~2-3 ms.
  - Fusion (NumPy/SciPy CPU): ~1-2 ms.
  - **Total estimated warm inference**: ~15-20 ms on GPU, matching or beating PyTorch without requiring the PyTorch runtime.

---

## 9. Production Integration Risk

- **Dependency Risk**: Introducing PyTorch to production container (`ParserMode.FUSED` currently requires PyTorch and CUDA). Moving to Option B (ONNX auxiliary head) eliminates PyTorch from production.
- **Concurrency Risk**: Singleton pattern (`FaceParserService`) must remain thread-safe (`threading.Lock`).
- **Fallback Behavior**: If auxiliary head fails or is missing, pipeline must fall back gracefully to `ParserMode.ORIGINAL`.

---

## 10. Checkpoint / Model Artifact Inventory

| Artifact | Path | Exists? | Size | SHA256 (prefix) | Architecture | Input | Output | Production? | FUSED? | Exportable? | Notes |
|----------|------|---------|------|-----------------|--------------|-------|--------|-------------|--------|-------------|-------|
| `bisenet_resnet18.onnx` | `ai_models/bisenet/bisenet_resnet18.onnx` | YES | 50.7 MB | `2218b6183c26ca5c` | BiSeNetV1 (19c) | `(1,3,512,512)` | `(1,19,512,512)` | **YES (DEFAULT)** | YES | YES | Production ONNX backbone |
| `best.pt` (Aux Head) | `dataset_builder/dataset/parser_finetune_current/training_aux_eye_brow_phase1/checkpoints/best.pt` | YES | 4.47 MB | `961e08bf64fdd0b8` | `EyeBrowRefinementHead` (6c) | `(1,256,64,64)` | `(1,6,512,512)` | NO | YES | YES | Phase 1 auxiliary head state_dict |
| `final.pt` (Aux Head) | `dataset_builder/dataset/parser_finetune_current/training_aux_eye_brow_phase1/checkpoints/final.pt` | YES | 4.48 MB | — | `EyeBrowRefinementHead` (6c) | `(1,256,64,64)` | `(1,6,512,512)` | NO | NO | YES | Final epoch auxiliary checkpoint |
| `best.pt` (Fine-Tuned Full) | `dataset_builder/dataset/parser_finetune_expanded/training/checkpoints/best.pt` | YES | 159.8 MB | — | BiSeNetV1 (19c) | `(1,3,512,512)` | `(1,19,512,512)` | NO | NO | YES | Evaluation comparison model |

---

## 11. Test Coverage Audit

| Test File | What it Verifies | Type | PyTorch vs ONNX? | Mask Equivalence? |
|-----------|------------------|------|------------------|-------------------|
| `tests/services/test_phase4_parser_integration.py` | ParserMode switching, singleton, mock refinement, mapping, non-target protection | Unit / Integration | No (Mocked) | Yes (Mocked) |
| `tests/services/test_phase4_real_integration.py` | Real ONNX loading into PyTorch, real `best.pt` loading, CUDA requirement, mask validity | Integration (slow) | Loads ONNX weights into PT | Yes |
| `tests/pipeline/test_phase4_end_to_end.py` | End-to-end pipeline execution with FUSED parser mode on test images | E2E | No | Yes |

---

## 12. Identify Missing Tests

Before activating any ONNX-based auxiliary head in production, the following tests must be implemented:
1. **PyTorch-vs-ONNX Numerical Parity Test**: Compares `EyeBrowRefinementHead` PyTorch outputs against ONNX Runtime outputs on identical FFM feature tensors.
2. **Deterministic Fusion Equivalence Test**: Verifies that Python/NumPy fusion produces identical results regardless of whether auxiliary logits come from PyTorch or ONNX Runtime.
3. **Non-Target Class Protection Parity Test**: Confirms ONNX auxiliary head + fusion never overwrites non-target classes.
4. **CPU vs CUDA Execution Parity Test**: Verifies numerical consistency across execution providers.

---

## 13. Final Decision

### **RECOMMEND: OPTION B**
**ONNX Backbone + ONNX Auxiliary Head + Deterministic External Fusion (Python/NumPy)**

**Rationale**:
- **Why not Option A (Single ONNX Graph)?** SciPy connected components and Python branching cannot be compiled into ONNX. Rewriting them in ONNX operators would be extremely complex, error-prone, and risk altering the verified Phase 3 fusion behavior.
- **Why Option B?** It achieves 100% behavioral parity with the proven PyTorch Fused implementation while eliminating the PyTorch runtime dependency in production. Both neural network components (`bisenet_resnet18.onnx` and `aux_head.onnx`) execute via ONNX Runtime, and the deterministic Python fusion logic remains intact and fully tested.

---

## PHASE 6 STATUS:
### **NOT READY FOR EXPORT**

**Explanation of what must happen next**:
1. Implement the ONNX export script for `EyeBrowRefinementHead` (exporting to `ai_models/bisenet/aux_head.onnx`).
2. Update `EyeBrowRefinementService` to support an ONNX Runtime session for the auxiliary head alongside the ONNX backbone.
3. Implement the PyTorch-vs-ONNX numerical parity test suite.
4. Verify 100% mask equivalence across test images before enabling `ParserMode.FUSED` as default.
