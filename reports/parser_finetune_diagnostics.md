# Parser Fine-Tuning Diagnostic Report

Generated: 2026-08-15

## Executive Summary

The fine-tuned model improved its 6 target classes (LEFT_EYE, RIGHT_EYE, HAT, LEFT_BROW, RIGHT_BROW, EYE_GLASS) but catastrophically degraded 13 non-target classes, resulting in an overall mIoU drop from 0.863 to 0.738 (-14.6%). The root cause is **catastrophic forgetting** exacerbated by a **target-only checkpoint selection criterion** that masked regression on non-target classes.

---

## 1. Split Verification

### Expanded Dataset Splits
| Split | Samples |
|-------|---------|
| Train | 122 |
| Val   | 26 |
| Test  | 27 |
| Total | 175 |

**Overlap within expanded splits**: NONE. Clean separation.

### Cross-Dataset Leakage (Expanded test vs Pilot)

| Overlap | Count | Samples |
|---------|-------|---------|
| Expanded test in Pilot **train** | 12 | sample_0001, 0015, 0017, 0045, 0050, 0073, 0077, 0084, 0085, 0088, 0098, 0014 |
| Expanded test in Pilot **val** | 1 | sample_0007 |
| Expanded test in Pilot **test** | 2 | sample_0008, 0055 |
| **Total expanded test in ANY pilot split** | **15 of 27** | |

**CRITICAL**: 15 of 27 expanded test images (55.6%) were used in the pilot fine-tuning experiment. However, the expanded fine-tuning started from ONNX weights (not the pilot checkpoint), so these images were not directly trained on in the expanded run. Still, this overlap means the expanded test set is not a fully independent holdout from the pilot experiment.

### Source Image Overlap Across Splits
- Test sources in Train: 0 images
- Test sources in Val: 0 images

No same-source-image contamination across expanded splits.

---

## 2. Ground-Truth Class Distribution (26 Test Images)

| Class | Pixels | Percentage | Images Present |
|-------|--------|-----------|----------------|
| BACKGROUND | 61,030 | 18.71% | 26 |
| SKIN | 119,531 | 36.65% | 26 |
| LEFT_BROW | 2,503 | 0.77% | 26 |
| RIGHT_BROW | 1,940 | 0.59% | 24 |
| LEFT_EYE | 1,468 | 0.45% | 23 |
| RIGHT_EYE | 1,454 | 0.45% | 24 |
| EYE_GLASS | 21,244 | 6.51% | 16 |
| LEFT_EAR | 1,344 | 0.41% | 9 |
| RIGHT_EAR | 874 | 0.27% | 8 |
| EAR_RING | 70 | 0.02% | 2 |
| NOSE | 12,363 | 3.79% | 26 |
| MOUTH | 588 | 0.18% | 11 |
| UPPER_LIP | 2,237 | 0.69% | 26 |
| LOWER_LIP | 3,417 | 1.05% | 26 |
| NECK | 3,571 | 1.09% | 19 |
| NECKLACE | 21 | 0.01% | 1 |
| CLOTH | 5,577 | 1.71% | 13 |
| HAIR | 61,019 | 18.71% | 23 |
| HAT | 25,893 | 7.94% | 7 |

Total pixels: 326,144

**Key observation**: SKIN (36.65%), BACKGROUND (18.71%), and HAIR (18.71%) together account for 74.07% of all pixels. The 6 target classes together account for only 9.62% of pixels.

---

## 3 & 4. Per-Class Metrics (Production vs Fine-Tuned)

### IoU Comparison

| Class | Prod IoU | FT IoU | Delta | Status |
|-------|----------|--------|-------|--------|
| BACKGROUND | 0.9883 | 0.8963 | -0.0920 | DEGRADED |
| SKIN | 0.9929 | 0.9269 | -0.0660 | DEGRADED |
| LEFT_BROW | 0.9007 | 0.6868 | -0.2139 | DEGRADED |
| RIGHT_BROW | 0.8350 | 0.6759 | -0.1592 | DEGRADED |
| **LEFT_EYE** | **0.3478** | **0.6137** | **+0.2659** | **IMPROVED** |
| **RIGHT_EYE** | **0.2387** | **0.6429** | **+0.4042** | **IMPROVED** |
| EYE_GLASS | 0.8874 | 0.8452 | -0.0422 | SLIGHT DEGRADATION |
| LEFT_EAR | 1.0000 | 0.6827 | -0.3173 | DEGRADED |
| RIGHT_EAR | 1.0000 | 0.6230 | -0.3770 | DEGRADED |
| EAR_RING | 1.0000 | 0.5789 | -0.4211 | DEGRADED |
| NOSE | 1.0000 | 0.9239 | -0.0761 | DEGRADED |
| MOUTH | 1.0000 | 0.6389 | -0.3611 | DEGRADED |
| UPPER_LIP | 1.0000 | 0.6730 | -0.3270 | DEGRADED |
| LOWER_LIP | 0.9959 | 0.7841 | -0.2118 | DEGRADED |
| NECK | 0.8876 | 0.6978 | -0.1898 | DEGRADED |
| NECKLACE | 1.0000 | 0.7143 | -0.2857 | DEGRADED |
| CLOTH | 0.5943 | 0.6013 | +0.0070 | ~SAME |
| HAIR | 0.9738 | 0.9206 | -0.0532 | DEGRADED |
| **HAT** | **0.7630** | **0.8920** | **+0.1290** | **IMPROVED** |

### Precision/Recall Highlights

| Class | Prod Recall | FT Recall | Delta |
|-------|------------|-----------|-------|
| LEFT_EYE | 0.3604 | 0.8106 | **+0.4503** |
| RIGHT_EYE | 0.2387 | 0.7923 | **+0.5536** |
| HAT | 0.7797 | 0.9474 | **+0.1677** |
| MOUTH | 1.0000 | 0.7942 | -0.2058 |
| UPPER_LIP | 1.0000 | 0.7903 | -0.2097 |
| LEFT_EAR | 1.0000 | 0.6868 | -0.3132 |
| CLOTH | 1.0000 | 0.6392 | -0.3608 |

---

## 5. Classes Responsible for Degradation

Ranked by absolute IoU drop:

| Rank | Class | Delta IoU | GT Pixels | Impact |
|------|-------|-----------|-----------|--------|
| 1 | EAR_RING | -0.4211 | 70 | Low (tiny class) |
| 2 | RIGHT_EAR | -0.3770 | 874 | Medium |
| 3 | MOUTH | -0.3611 | 588 | Low (tiny class) |
| 4 | UPPER_LIP | -0.3270 | 2,237 | Medium |
| 5 | LEFT_EAR | -0.3173 | 1,344 | Medium |
| 6 | NECKLACE | -0.2857 | 21 | Negligible |
| 7 | LEFT_BROW | -0.2139 | 2,503 | Medium |
| 8 | LOWER_LIP | -0.2118 | 3,417 | Medium |
| 9 | NECK | -0.1898 | 3,571 | Medium |
| 10 | RIGHT_BROW | -0.1592 | 1,940 | Medium |
| 11 | BACKGROUND | -0.0920 | 61,030 | **HIGH** (huge class) |
| 12 | NOSE | -0.0761 | 12,363 | Medium |
| 13 | SKIN | -0.0660 | 119,531 | **HIGH** (huge class) |
| 14 | HAIR | -0.0532 | 61,019 | Medium |

**The two biggest overall mIoU killers are SKIN and BACKGROUND** due to their massive pixel counts. A 6.6% IoU drop on SKIN (119,531 pixels) and 9.2% on BACKGROUND (61,030 pixels) accounts for most of the overall mIoU regression.

---

## 6. Fine-Tuning Configuration

| Parameter | Value |
|-----------|-------|
| Loss function | CrossEntropyLoss with per-pixel class weighting |
| Class weights | {4: 2.0, 5: 2.0, 6: 1.0} (LEFT_EYE, RIGHT_EYE, EYE_GLASS) |
| Non-target class weight | 1.0 (default) |
| Optimizer | AdamW |
| Learning rate | 1e-5 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR (T_max=20) |
| Epochs | 20 |
| Batch size | 4 |
| Ignore index | None (all 19 classes contribute to loss) |
| Aux loss weights | aux16=0.4, aux32=0.4 |
| Image size | 512x512 |
| Mask size | 112x112 (upscaled to 512 for loss) |
| Augmentation | H-flip(0.5), rot(+-8deg), scale(0.96-1.04), brightness/contrast(+-0.08) |

---

## 7. Checkpoint Selection Criterion

**best.pt was selected by**: `val_target_mean_iou` (epoch 17)

This metric is the mean IoU of **only the 6 target classes** (LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE, EYE_GLASS, HAT), NOT all 19 classes.

| Epoch | Target mIoU | Full mIoU | Train Loss |
|-------|------------|-----------|------------|
| 1 | 0.6363 | 0.6982 | 0.8511 |
| 4 | 0.7174 | 0.7473 | 0.5369 |
| 8 | 0.7501 | 0.7514 | 0.4571 |
| 12 | 0.7541 | 0.7583 | 0.4389 |
| 17 | **0.7546** | 0.7557 | 0.4099 |
| 20 | 0.7541 | 0.7552 | 0.4120 |

**Critical**: The model was never evaluated on all 19 classes for checkpoint selection. The best.pt at epoch 17 maximizes target-class IoU while non-target classes may have been degrading.

---

## 8. Preprocessing Differences

Both models receive the **exact same input**:
- Same BGR uint8 numpy array from `cv2.imread()`
- Same BGR-to-RGB conversion
- Same 512x512 resize with INTER_LINEAR
- Same ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- Same HWC-to-NCHW layout

**No preprocessing difference exists.** The evaluator feeds the identical image tensor to both models.

---

## 9. Class Index Mapping

All three sources use identical mappings:
- `FacePart` enum (models/parsing/face_part.py)
- `CLASS_NAMES` dict (training/config.py)
- Evaluation report class_mapping

19 classes, same order, same names. **No mapping inconsistency.**

---

## 10. Root Cause Analysis

### Primary cause: Catastrophic forgetting with target-only checkpoint selection

1. **Class weights only boost target classes**: LEFT_EYE and RIGHT_EYE get 2x weight, EYE_GLASS gets 1x, all others get 1x. This biases the loss toward improving eyes.

2. **Checkpoint selection ignores non-target classes**: `best.pt` is selected by `target_mean_iou` (6 classes only). The model was never penalized for degrading SKIN, BACKGROUND, MOUTH, etc.

3. **Fine-tuning unfreezes all layers**: The entire BiSeNet is fine-tuned (no frozen backbone), so improving small eye regions requires the model to shift its feature representations, which degrades large stable regions like SKIN and BACKGROUND.

4. **Small dataset, high learning rate**: 107 training samples with lr=1e-5 and 20 epochs may be sufficient to destabilize the pretrained backbone's representations of non-target classes.

### Secondary cause: Pretrained model was already near-perfect on non-target classes

The production ONNX model achieved IoU=1.0 on 8 classes (LEFT_EAR, RIGHT_EAR, EAR_RING, NOSE, MOUTH, UPPER_LIP, NECKLACE). Any fine-tuning pressure risks moving these away from their optimal weights.

---

## Recommendations

1. **Freeze the backbone** and only fine-tune the classification head
2. **Use all-19-class mIoU** as checkpoint selection criterion (or a weighted combination)
3. **Add regularization**: increase weight decay, add EWC/MAS to protect non-target classes
4. **Reduce learning rate** (e.g., 1e-6) and/or use differential LR (lower for backbone, higher for head)
5. **Evaluate on ALL classes** during training, not just targets
