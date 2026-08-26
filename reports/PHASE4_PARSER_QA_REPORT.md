# Phase 4 Parser QA Final Report

## 1. Experiment Overview

| Item | Value |
|------|-------|
| Experiment | Phase 4 Parser QA - ORIGINAL vs FUSED |
| Objective | Evaluate whether FUSED mode visibly improves eye/brow parsing on production-like images |
| Date | 2026-08-26 |
| Total images | 25 |
| Successful parser runs | 23 |
| Failed parser runs | 2 (no face detected) |

**Failed images:** `pexels_3714139.jpeg`, `pexels_3899102.jpeg`

**Critical limitation:** No ground-truth masks exist for these production-like images. This is qualitative + behavioral production QA only. Quantitative IoU evidence remains in Phase 2/Phase 3 held-out evaluation.

## 2. Production Architecture

| Component | ORIGINAL | FUSED |
|-----------|----------|-------|
| Parser backbone | ONNX BiSeNet (ResNet-18) | PyTorch BiSeNet (same backbone) |
| FFM | Yes | Yes |
| Auxiliary Eye/Brow Head | No | Yes |
| Phase 3 Fusion | No | Strategy 1, threshold 0.0, min_component_size 10 |
| Non-target safety guard | No | Yes (prevents overwriting non-target classes) |

- ONNX model hash: `2218b6183c26ca5c`
- Aux checkpoint hash: `961e08bf64fdd0b8`
- Same aligned face fed to both modes

## 3. Per-Class Pixel Statistics

| Class | ORIGINAL mean | FUSED mean | Delta mean | Delta median | Direction |
|-------|--------------|-----------|-----------|-------------|-----------|
| LEFT_EYE | 27 | 47 | +20 | +23 | FUSED larger |
| RIGHT_EYE | 12 | 44 | +32 | +28 | FUSED larger |
| LEFT_BROW | 122 | 101 | -21 | +0 | Mixed |
| RIGHT_BROW | 40 | 71 | +31 | +18 | FUSED larger |
| EYE_GLASS | 1098 | 1035 | -63 | -64 | FUSED smaller |

**Interpretation:** FUSED recovers eye pixels that ORIGINAL missed (median delta +23/+28 pixels for left/right eye). Glass pixels decrease slightly (median -64), consistent with auxiliary head reclaiming eye regions from glass over-classification.

## 4. Eye Recovery Analysis

8 of 23 images show eye recovery candidates:

| Image | Eye Recovery Status |
|-------|-------------------|
| hijab with glassesjpg.jpg | L:RECOVERED R:RECOVERED |
| hijab2.jpg | L:RECOVERED R:RECOVERED |
| pexels_1516680.jpeg | L:RECOVERED R:RECOVERED |
| pexels_3252801.jpeg | L:RECOVERED R:RECOVERED |
| pexels_3772712.jpeg | L:RECOVERED R:UNCHANGED |
| pexels_4442025.jpeg | L:RECOVERED R:RECOVERED |
| pexels_4836878.jpeg | L:REDUCED R:IMPROVED |
| prescription_eyeglasses_lady.jpg | L:UNCHANGED R:RECOVERED |

## 5. Brow Recovery Analysis

9 of 23 images show brow recovery/improvement candidates:

| Image | Brow Recovery Status |
|-------|---------------------|
| hijab with glassesjpg.jpg | L:REDUCED R:RECOVERED |
| hijab2.jpg | L:REDUCED R:RECOVERED |
| pexels_3931310.jpeg | L:IMPROVED R:IMPROVED |
| pexels_4482931.jpeg | L:REDUCED R:RECOVERED |
| pexels_4728707.jpeg | L:REDUCED R:IMPROVED |
| pexels_4836878.jpeg | L:REDUCED R:IMPROVED |
| pexels_5049702.jpeg | L:REDUCED R:RECOVERED |
| pexels_5815587.jpeg | L:REDUCED R:RECOVERED |
| pexels_5974073.jpeg | L:REDUCED R:RECOVERED |

## 6. Eye/Glasses Interaction

All 23 successful images show `eye_inside_glasses_ratio = 0.0` for both ORIGINAL and FUSED, meaning the parsers correctly treat eyes and glasses as separate classes (no overlapping pixel assignments).

## 7. Possible Over-Segmentation

**Zero** possible over-segmentation cases detected. No suspicious fragmentation found (no class with >6 components or largest component <30% with >2 components).

## 8. Reduction Cases

9 images show LEFT_BROW reduction in FUSED mode:

| Image | Eye | Brow |
|-------|-----|------|
| hijab with glassesjpg.jpg | RECOVERED | REDUCED |
| hijab2.jpg | RECOVERED | REDUCED |
| pexels_4482931.jpeg | REDUCED | REDUCED |
| pexels_4728707.jpeg | REDUCED | REDUCED |
| pexels_4836878.jpeg | REDUCED | REDUCED |
| pexels_5049702.jpeg | UNCHANGED | REDUCED |
| pexels_5815587.jpeg | UNCHANGED | REDUCED |
| pexels_5945246.jpeg | UNCHANGED | REDUCED |
| pexels_5974073.jpeg | UNCHANGED | REDUCED |

## 9. Human Review Status

| Suggested Label | Count | % |
|----------------|-------|---|
| IMPROVEMENT | 14 | 61% |
| PENDING_REVIEW | 8 | 35% |
| SUSPECTED_FAILURE | 1 | 4% |

**Key finding:** 61% of processed images are suggested as IMPROVEMENT, meaning FUSED mode recovers or improves eye/brow representation that ORIGINAL missed.

## 10. Suggested Verdict Definitions

| Verdict | Definition |
|---------|-----------|
| PASS | Visible anatomical eye/brow is represented appropriately by FUSED |
| EXPECTED_CONSERVATIVE | Eye is genuinely not visible/heavily occluded; FUSED does not hallucinate |
| IMPROVEMENT | FUSED clearly represents visible eye/brow information that ORIGINAL missed |
| SUSPECTED_FAILURE | FUSED appears to miss clearly visible eye/brow information |
| SUSPECTED_OVERSEGMENTATION | FUSED creates eye/brow regions unsupported by the image |
| PENDING_REVIEW | Insufficient evidence (default) |

## 11. Recovery Heuristics

| Heuristic | Condition |
|-----------|-----------|
| RECOVERED | Original=0, FUSED<0.5% of image |
| IMPROVED | FUSED > Original + 20% |
| UNCHANGED | FUSED within ±20% of Original |
| REDUCED | FUSED < Original - 20% |
| POSSIBLE_OVERSEGMENTATION | Original=0, FUSED > 0.5% of image |

**These are QA signals, not segmentation accuracy metrics.**

## 12. Protected Artifacts (Untouched)

| Artifact | SHA256 Prefix |
|----------|---------------|
| `ai_models/bisenet/bisenet_resnet18.onnx` | `2218b6183c26ca5c` |
| `training_aux_eye_brow_phase1/checkpoints/best.pt` | `961e08bf64fdd0b8` |

## 13. Limitations

1. **No ground-truth masks** exist for these production-like images
2. Qualitative + behavioral production QA only
3. Quantitative IoU evidence remains in Phase 2/Phase 3 held-out evaluation
4. Eye visibility and occlusion type fields require manual human review
5. Recovery analysis is pixel-count + geometric based, not anatomically validated
6. Recovery heuristics are QA signals, not segmentation accuracy metrics
7. 2 images failed (no face detected) - not a parser issue

## 14. Runtime Performance

| Metric | ORIGINAL | FUSED |
|--------|----------|-------|
| Average inference time (batch) | 1142 ms | 409 ms |
| Median inference time (benchmark) | 66 ms (ONNX CPU) | 15 ms (PyTorch CUDA) |
| Peak GPU memory | - | 176 MB |

FUSED is faster in production due to PyTorch CUDA acceleration vs ONNX CPU execution.

## 15. Final Conclusion

FUSED mode demonstrates candidate eye recovery or improvement in **8 of 23 images (35%)** and brow recovery in **9 of 23 images (39%)**. The overall suggested label distribution shows **14 IMPROVEMENT (61%)**, **8 PENDING_REVIEW (35%)**, and **1 SUSPECTED_FAILURE (4%)**.

**All suggested labels are PENDING_REVIEW until human examination of the comparison PNGs.** The comparison visualizations in `reports/experiments/phase4_parser_qa/comparisons/` should be reviewed by a human to confirm these are genuine improvements.

## 16. Output Artifacts

| File | Description |
|------|-------------|
| `reports/experiments/phase4_parser_qa/per_image_results.csv` | All per-image statistics |
| `reports/experiments/phase4_parser_qa/per_image_results.json` | All per-image statistics (JSON) |
| `reports/experiments/phase4_parser_qa/parser_qa_summary.json` | Aggregate summary (JSON) |
| `reports/experiments/phase4_parser_qa/parser_qa_summary.md` | This report (auto-generated) |
| `reports/experiments/phase4_parser_qa/human_review.csv` | CSV for human review with suggested labels |
| `reports/experiments/phase4_parser_qa/comparisons/*.png` | 6-panel comparison visualizations |
