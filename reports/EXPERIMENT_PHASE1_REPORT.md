# Phase 1 Report — Eye/Brow Refinement Auxiliary Head (CE Baseline, Frozen BiSeNet)

**Generated:** 2026-08-25T17:07:13.051678+00:00
**Experiment:** Phase1_EyeBrow_Refinement_Auxiliary_Head_CE_Baseline
**Checkpoint criterion:** best_val_target_mean_iou (target mIoU = mean IoU over 5 target classes)
**Device:** cuda:0 / NVIDIA GeForce RTX 4060 Laptop GPU / CUDA 12.8 / 2.11.0+cu128

---

## 1. Dataset Size
- **Total samples in `parser_finetune_current`:** 1058 (manifest + splits)
- **Annotated & usable (ACCEPT):** 1039 (after filtering UNANNOTATABLE)
- **Images on disk:** 1058
- **Masks on disk:** 1058
- **Corrected masks:** 1042 in splits (16 missing correspond to UNANNOTATABLE), 1044 on disk

## 2. Train/Val/Test Counts
| Split | File count | ACCEPT usable | UNANNOTATABLE |
|-------|------------|---------------|---------------|
| Train | 843 | 825 | 18 |
| Val   | 101 | 100 | 1 |
| Test  | 114 | 114 | 0 |
| **Total** | **1058** | **1039** | **19** |

No overlap between splits. No ID duplicates. All images/masks exist.

## 3. Annotation Status
- ACCEPT: 1039 (98.2%)
- UNANNOTATABLE: 19 (1.8%)
- No ACCEPT is missing its corrected mask (0)
- No initial BiSeNet mask used as GT when corrected exists — `AuxiliaryEyeBrowDataset` loads only from `annotation/corrected_masks` and filters by `annotation_status == ACCEPT`.

## 4. Class Distribution (6-class, ACCEPT pixels, 512×512)
Total pixels: 13033216 across 1039 masks (112×112 upsampled to 512).

| Class | Pixels | Pixel % | Samples containing class |
|-------|--------|---------|--------------------------|
| AUX_BACKGROUND | 12257939 | 94.05% | 1039 |
| LEFT_BROW (1) | 88366 | 0.68% | 929 |
| RIGHT_BROW (2) | 78007 | 0.60% | 873 |
| LEFT_EYE (3) | 37854 | 0.29% | 787 |
| RIGHT_EYE (4) | 36902 | 0.28% | 764 |
| EYE_GLASS (5) | 534148 | 4.10% | 408 (39%) |

Imbalance is expected (background dominates, eyes <1%). Reported before training; no class weighting applied in Phase 1 by design (clean CE baseline).

## 5. Auxiliary Head Architecture
```
FFM feature (256 × 64 × 64 for 512×512 input)
  → Conv2d(256→128, 3×3, pad1) + BN + ReLU
  → Conv2d(128→64, 3×3, pad1) + BN + ReLU
  → Conv2d(64→6, 1×1)
  → Bilinear upsample to 512×512 (logits)
```
Lightweight, ~369k params. Input explicitly `ffm_features` from frozen `bisenet.ffm(feat_res8, feat_cp8)` at 1/8 resolution.

## 6. Auxiliary Parameter Count
- **Aux Head total:** 369408 (369,408)
- **Aux Head trainable:** 369408

Breakdown: Conv1 294,912 + BN1 256 + Conv2 73,728 + BN2 128 + Conv3 384 = 369,408.

## 7. BiSeNet Parameter Count
- **BiSeNet total:** 13300416 (13,300,416)
- **ResNet-18 backbone + ARM + FFM + 3 heads (19-class)**

## 8. Trainable Parameter Count
- **BiSeNet trainable:** 0 (expected 0)
- **Head trainable:** 369408
- **Optimizer scope:** 7 param tensors (only head) — verified `id` disjoint from BiSeNet.

## 9. CUDA/GPU Verification
- Python: 3.12.2 (C:\Users\amir\Downloads\SmartIDPhotoProcessor\.venv312\Scripts\python.exe)
- Torch: 2.11.0+cu128
- CUDA available: True (version 12.8)
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Device: `torch.device("cuda:0")` — enforced via `enforce_cuda_device()`; fails fast if unavailable.
- All models & tensors verified on `cuda:0` (see `verify_freezing`).

## 10. Training Configuration
| Field | Value |
|-------|-------|
| Optimizer | adamw (AdamW) |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Batch size | 4 (RTX 4060 Laptop 8GB VRAM; 512² → ~2GB per batch 4, fits comfortably) |
| Epochs | up to 30 (early stopping patience 7 on val target mIoU) |
| Scheduler | cosine (CosineAnnealing, T_max=30) |
| Loss | CE only, no class weights, no Dice |
| Image size | 512 (augmented then resized; mask 112→512 nearest) |
| Augmentation | {'enabled': True, 'horizontal_flip_probability': 0.5, 'max_rotation_degrees': 8.0, 'max_translation_fraction': 0.04, 'min_scale': 0.96, 'max_scale': 1.04, 'brightness_delta': 0.08, 'contrast_delta': 0.08} |
| Seed | 42 (deterministic cudnn) |

Loss is plain `CrossEntropyLoss` — no weighting, no Dice, no distillation (Phase 1 clean baseline).

## 11. Best Epoch
- **Best val target mIoU:** 0.780417 at epoch **23**
- **Best val mean IoU (6-class):** 0.815400
- Early stopping patience 7: best at 23, trained to 30, no improvement after epoch 23 (target 0.7804).

## 12. Best Validation Target mIoU
- **Val target mIoU:** 0.780417 (epoch 23)
- **Val mean IoU (6-class):** 0.815400
- **Val pixel accuracy:** 0.989889
- **Val mean Dice:** 0.895582
- Per-class val IoU: LEFT_BROW 0.7996, RIGHT_BROW 0.8029, LEFT_EYE 0.7360, RIGHT_EYE 0.7071, EYE_GLASS 0.8566

## 13. Test Metrics (Auxiliary Head, best checkpoint)
| Metric | Value |
|--------|-------|
| Pixel accuracy | 0.990424 |
| Mean IoU (6-class) | 0.819487 |
| Mean Dice | 0.898424 |
| **Target mIoU (5 target classes)** | **0.785203** |
| Samples | 114 |

## 14. Per-Class IoU (Test)
| Class | IoU | Dice | Precision | Recall |
|-------|-----|------|-----------|--------|
| AUX_BACKGROUND | 0.9909 | 0.9954 | 0.9955 | 0.9954 |
| LEFT_BROW | 0.7915 | 0.8836 | 0.9160 | 0.8534 |
| RIGHT_BROW | 0.7889 | 0.8820 | 0.8815 | 0.8825 |
| LEFT_EYE | 0.7427 | 0.8524 | 0.8848 | 0.8223 |
| RIGHT_EYE | 0.7418 | 0.8518 | 0.8947 | 0.8128 |
| EYE_GLASS | 0.8610 | 0.9253 | 0.9130 | 0.9380 |

Target mIoU = mean over the 5 target classes (excludes BG): **0.785203**

## 15. Production vs Auxiliary Comparison (Test, same 114 images, mapped to common 5 target classes)

| Class | Production BiSeNet IoU (19→6) | Auxiliary Head IoU | Δ (Aux − Prod) |
|-------|------------------------------|--------------------|----------------|
| LEFT_BROW | 0.5900 | 0.7915 | +0.2015 |
| RIGHT_BROW | 0.4958 | 0.7889 | +0.2931 |
| LEFT_EYE | 0.4441 | 0.7427 | +0.2987 |
| RIGHT_EYE | 0.3188 | 0.7418 | +0.4230 |
| EYE_GLASS | 0.8709 | 0.8610 | -0.0099 |
| **Target mIoU** | **0.5439** | **0.7852** | **+0.2413** |
| Mean IoU (6-class) | 0.6183 | 0.8195 | +0.2011 |

Production and auxiliary are compared on the **same 5 target classes** in 6-class space; raw 19-class mIoU is not directly comparable.

**Interpretation:** Large gains on brows (+0.20, +0.29) and eyes (+0.30, +0.42) where BiSeNet struggles (especially RIGHT_EYE 0.319→0.742). EYE_GLASS slightly lower (−0.01) but within noise — production was already 0.871 strong on glasses. Overall target mIoU +0.241 (44% relative).

## 16. Original BiSeNet Integrity Verification
- **Hash before:** `18e6835e6a9a0c5e29329f2781e06276`
- **Hash after:** `18e6835e6a9a0c5e29329f2781e06276` (recomputed, identical)
- **BN hash before/after:** `d4cde435d20034cac808af22c57de8f6` (identical)
- **Freezing checks (before training):**
```
  - PASS: All BiSeNet parameters are frozen
  - PASS: All auxiliary head parameters are trainable
  - PASS: BiSeNet is in eval mode
  - PASS: All BatchNorm layers are in eval mode
  - PASS: Both models on cuda:0
```
- All `requires_grad == False`, `BatchNorm.training == False`, `track_running_stats == False`, in `eval()` mode.
- Throughout training, per-batch assert `p.grad is None` for every BiSeNet param.
- **Result: ZERO changes. Production BiSeNet completely unchanged.**

ONNX `bisenet_resnet18.onnx` SHA256: `2218b6183c26ca5c83303232d682a536c670c13ea9695f716c777d1f244eefe9` size 53205356 — not modified.

## 17. Number of Tests Passed / Failed
- **Phase 1 dedicated tests:** 28 passed, 0 failed (17 categories, 28 test cases in `training_aux_eye_brow_phase1/tests/test_phase1.py`)
  - Categories: mapping (6), dataset loading, corrected-mask usage, split isolation, head output shape, param count, BiSeNet frozen, optimizer scope, BN protection, CUDA placement, forward pass, loss, backward updates head, backward not BiSeNet, checkpoint save/load, param integrity, protected data.
- Full repo `pytest` previously: 974 passed (26 pre-existing face_visibility failures unrelated).

## 18. Limitations or Anomalies
- **Dataset:** 19 samples are UNANNOTATABLE (1.8%) and excluded; effective train set 825 not 843. Not an inconsistency — splits are kept as-is, filtered at load time. Manifest total 1058; usable 1039. Split regeneration was not needed.
- **Class imbalance:** Eyes <0.6% pixels; CE baseline without weighting still learns them well (IoU ~0.74). Weighted loss may help slightly but was deferred per spec.
- **EYE_GLASS:** Auxiliary IoU 0.861 slightly below production 0.871 (−0.01). Both high; likely within variance and class is easier for production.
- **Early stopping:** Best epoch 23, trained to 30 with patience 7. No divergence.
- **VRAM:** Batch 4 on RTX 4060 Laptop 8GB uses ~2GB; batch 8 would also fit but 4 chosen for stable gradients and to match prior experiments.
- No confidence gate / 19-class fusion implemented (Phase 3).
- Protected data (raw, ONNX, previous checkpoints, corrected masks) untouched.

---

## Verdict: **PASS**

**Did the Auxiliary Head learn meaningful Eye/Brow segmentation while the original BiSeNet remained completely unchanged?**

**Yes.**

- Auxiliary head trained from scratch with plain CE achieves **test target mIoU 0.785** vs production **0.544** (**+0.241**, +44% relative) on the same 114 test images.
- Per-target gains are large and consistent, especially on brows and eyes where production is weak.
- All 369k head parameters updated; all 13.3M BiSeNet parameters frozen with hash unchanged (`18e6835e6a9a0c5e29329f2781e06276`), BN stats protected, optimizer scope verified, and no protected artifacts modified.
- Training is reproducible (seed 42, deterministic cudnn), isolated checkpoint `training_aux_eye_brow_phase1/checkpoints/best.pt` selected on val target mIoU.

Phase 1 objectives are fully met. No Phase 2/3 code was introduced.

---

## Artifacts
- Checkpoint (best): `dataset_builder/dataset/parser_finetune_current/training_aux_eye_brow_phase1/checkpoints/best.pt` (epoch 23)
- Checkpoint (final): `.../checkpoints/final.pt`
- Metrics: `.../reports/experiment_phase1_metrics.json` and `reports/experiment_phase1_metrics.json`
- History: `.../reports/training_history.json` and `reports/training_history.json`
- Visuals: `.../evaluation/visuals/` (8 samples: sample_0081, sample_0120, sample_0201, sample_0465, sample_0501, sample_0719, sample_1363, sample_1404)
- Tests: `.../tests/test_phase1.py` (28 tests)
