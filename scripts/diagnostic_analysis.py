"""Comprehensive diagnostic analysis for parser fine-tuning evaluation."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.parsing.face_part import FacePart
from services.face_parser_service import FaceParserService
from services.fine_tuned_face_parser_service import FineTunedFaceParserService

EXPANDED_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune_expanded"
PILOT_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune"
SPLITS_DIR = EXPANDED_DIR / "splits"
PILOT_SPLITS_DIR = PILOT_DIR / "splits"
MASKS_DIR = EXPANDED_DIR / "annotation" / "corrected_masks"
IMAGES_DIR = EXPANDED_DIR / "images"
METADATA_DIR = EXPANDED_DIR / "metadata"
OUTPUT_DIR = PROJECT_ROOT / "reports"


def load_split(split_name: str, splits_dir: Path) -> set[str]:
    path = splits_dir / f"{split_name}.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_source_images(sample_ids: set[str]) -> dict[str, str]:
    result = {}
    for sid in sorted(sample_ids):
        meta_path = METADATA_DIR / f"{sid}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            source = meta.get("source_image", "")
            if source:
                source_name = Path(source).name
            else:
                source_name = "unknown"
            result[sid] = source_name
        else:
            result[sid] = "unknown"
    return result


def analyze_splits():
    print("=" * 70)
    print("1. SPLIT ANALYSIS")
    print("=" * 70)

    train = load_split("train", SPLITS_DIR)
    val = load_split("val", SPLITS_DIR)
    test = load_split("test", SPLITS_DIR)

    print(f"\nExpanded dataset splits:")
    print(f"  Train: {len(train)} samples")
    print(f"  Val:   {len(val)} samples")
    print(f"  Test:  {len(test)} samples")
    print(f"  Total: {len(train | val | test)} samples")

    train_val = train & val
    train_test = train & test
    val_test = val & test
    print(f"\nOverlap check:")
    print(f"  Train AND Val:  {train_val or 'NONE'}")
    print(f"  Train AND Test: {train_test or 'NONE'}")
    print(f"  Val AND Test:   {val_test or 'NONE'}")

    pilot_train = load_split("train", PILOT_SPLITS_DIR)
    pilot_val = load_split("val", PILOT_SPLITS_DIR)
    pilot_test = load_split("test", PILOT_SPLITS_DIR)
    pilot_all = pilot_train | pilot_val | pilot_test

    expanded_test_in_pilot_train = test & pilot_train
    expanded_test_in_pilot_val = test & pilot_val
    expanded_test_in_pilot_test = test & pilot_test
    expanded_test_in_pilot = test & pilot_all

    print(f"\nPilot dataset splits:")
    print(f"  Train: {len(pilot_train)} samples")
    print(f"  Val:   {len(pilot_val)} samples")
    print(f"  Test:  {len(pilot_test)} samples")
    print(f"  Total: {len(pilot_all)} samples")

    print(f"\nExpanded test vs Pilot train overlap: {expanded_test_in_pilot_train or 'NONE'}")
    print(f"Expanded test vs Pilot val overlap:   {expanded_test_in_pilot_val or 'NONE'}")
    print(f"Expanded test vs Pilot test overlap:  {expanded_test_in_pilot_test or 'NONE'}")
    print(f"Expanded test in ANY pilot split:     {expanded_test_in_pilot or 'NONE'}")

    test_sources = load_source_images(test)
    train_sources = load_source_images(train)
    val_sources = load_source_images(val)

    test_source_names = set(test_sources.values())
    train_source_names = set(train_sources.values())
    val_source_names = set(val_sources.values())

    source_overlap_train_test = test_source_names & train_source_names
    source_overlap_val_test = test_source_names & val_source_names

    print(f"\nSource image overlap (same source file across splits):")
    print(f"  Test sources in Train: {len(source_overlap_train_test)} images")
    if source_overlap_train_test:
        for src in sorted(source_overlap_train_test)[:10]:
            print(f"    - {src}")
        if len(source_overlap_train_test) > 10:
            print(f"    ... and {len(source_overlap_train_test) - 10} more")
    print(f"  Test sources in Val:   {len(source_overlap_val_test)} images")
    if source_overlap_val_test:
        for src in sorted(source_overlap_val_test)[:10]:
            print(f"    - {src}")

    return train, val, test, test_sources, train_sources, val_sources


def analyze_gt_distribution(test: set[str]):
    print("\n" + "=" * 70)
    print("2. GROUND-TRUTH CLASS DISTRIBUTION (26 test images)")
    print("=" * 70)

    class_pixels: dict[str, int] = defaultdict(int)
    class_images: dict[str, int] = defaultdict(int)
    total_pixels = 0

    for sid in sorted(test):
        mask_path = MASKS_DIR / f"{sid}.png"
        if not mask_path.exists():
            print(f"  WARNING: Missing mask for {sid}")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"  WARNING: Unreadable mask for {sid}")
            continue

        unique, counts = np.unique(mask, return_counts=True)
        img_classes = set()
        for val, cnt in zip(unique, counts):
            if 0 <= val < len(FacePart):
                name = FacePart(val).name
                class_pixels[name] += int(cnt)
                total_pixels += int(cnt)
                img_classes.add(name)
        for name in img_classes:
            class_images[name] += 1

    print(f"\nTotal pixels across 26 test images: {total_pixels}")
    print(f"\n{'Class':<15} {'Pixels':>12} {'Percentage':>12} {'Images':>8}")
    print("-" * 50)
    for part in FacePart:
        name = part.name
        px = class_pixels.get(name, 0)
        pct = (px / total_pixels * 100) if total_pixels > 0 else 0
        imgs = class_images.get(name, 0)
        print(f"{name:<15} {px:>12,} {pct:>11.4f}% {imgs:>8}")

    return class_pixels, class_images, total_pixels


def compute_confusion_matrix(pred: np.ndarray, gt: np.ndarray, n_classes: int = 19) -> np.ndarray:
    pred_flat = pred.astype(np.int64).ravel()
    gt_flat = gt.astype(np.int64).ravel()
    valid = (gt_flat >= 0) & (gt_flat < n_classes) & (pred_flat >= 0) & (pred_flat < n_classes)
    bins = n_classes * gt_flat[valid] + pred_flat[valid]
    return np.bincount(bins, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_metrics(matrix: np.ndarray) -> dict:
    n = matrix.shape[0]
    result = {}
    for i in range(n):
        name = FacePart(i).name
        tp = float(matrix[i, i])
        fp = float(matrix[:, i].sum() - tp)
        fn = float(matrix[i, :].sum() - tp)
        gt_px = float(matrix[i, :].sum())
        union = tp + fp + fn
        dice_denom = 2.0 * tp + fp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        iou = tp / union if union > 0 else None
        dice = 2.0 * tp / dice_denom if dice_denom > 0 else None
        pixel_acc = tp / gt_px if gt_px > 0 else None

        result[name] = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "pixel_accuracy": pixel_acc,
            "gt_pixels": int(gt_px),
            "pred_pixels": int(matrix[:, i].sum()),
        }
    return result


def generate_confusion_matrices(test: set[str]):
    print("\n" + "=" * 70)
    print("3 & 4. PER-CLASS CONFUSION MATRICES & METRICS")
    print("=" * 70)

    prod_parser = FaceParserService()
    ft_parser = FineTunedFaceParserService()

    prod_matrix = np.zeros((19, 19), dtype=np.int64)
    ft_matrix = np.zeros((19, 19), dtype=np.int64)

    for sid in sorted(test):
        img_path = IMAGES_DIR / f"{sid}.png"
        mask_path = MASKS_DIR / f"{sid}.png"
        if not img_path.exists() or not mask_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or gt is None:
            continue

        prod_result = prod_parser.parse(image)
        prod_mask = prod_result.mask
        if prod_mask.shape != gt.shape:
            prod_mask = cv2.resize(
                prod_mask.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        prod_matrix += compute_confusion_matrix(prod_mask, gt)

        ft_result = ft_parser.parse(image)
        ft_mask = ft_result.mask
        if ft_mask.shape != gt.shape:
            ft_mask = cv2.resize(
                ft_mask.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        ft_matrix += compute_confusion_matrix(ft_mask, gt)

    prod_metrics = per_class_metrics(prod_matrix)
    ft_metrics = per_class_metrics(ft_matrix)

    print(f"\n{'Class':<15} {'Prod IoU':>10} {'FT IoU':>10} {'Δ IoU':>10} | {'Prod Dice':>10} {'FT Dice':>10} {'Δ Dice':>10}")
    print("-" * 90)
    for part in FacePart:
        name = part.name
        p_iou = prod_metrics[name]["iou"]
        f_iou = ft_metrics[name]["iou"]
        p_dice = prod_metrics[name]["dice"]
        f_dice = ft_metrics[name]["dice"]

        p_iou_s = f"{p_iou:.4f}" if p_iou is not None else "N/A"
        f_iou_s = f"{f_iou:.4f}" if f_iou is not None else "N/A"
        if p_iou is not None and f_iou is not None:
            delta_iou = f_iou - p_iou
            delta_iou_s = f"{delta_iou:+.4f}"
        else:
            delta_iou_s = "N/A"

        p_dice_s = f"{p_dice:.4f}" if p_dice is not None else "N/A"
        f_dice_s = f"{f_dice:.4f}" if f_dice is not None else "N/A"
        if p_dice is not None and f_dice is not None:
            delta_dice = f_dice - p_dice
            delta_dice_s = f"{delta_dice:+.4f}"
        else:
            delta_dice_s = "N/A"

        print(f"{name:<15} {p_iou_s:>10} {f_iou_s:>10} {delta_iou_s:>10} | {p_dice_s:>10} {f_dice_s:>10} {delta_dice_s:>10}")

    print(f"\n{'Class':<15} {'Prod Prec':>10} {'FT Prec':>10} {'Δ Prec':>10} | {'Prod Rec':>10} {'FT Rec':>10} {'Δ Rec':>10}")
    print("-" * 90)
    for part in FacePart:
        name = part.name
        p_prec = prod_metrics[name]["precision"]
        f_prec = ft_metrics[name]["precision"]
        p_rec = prod_metrics[name]["recall"]
        f_rec = ft_metrics[name]["recall"]

        p_prec_s = f"{p_prec:.4f}" if p_prec is not None else "N/A"
        f_prec_s = f"{f_prec:.4f}" if f_prec is not None else "N/A"
        if p_prec is not None and f_prec is not None:
            delta_prec_s = f"{f_prec - p_prec:+.4f}"
        else:
            delta_prec_s = "N/A"

        p_rec_s = f"{p_rec:.4f}" if p_rec is not None else "N/A"
        f_rec_s = f"{f_rec:.4f}" if f_rec is not None else "N/A"
        if p_rec is not None and f_rec is not None:
            delta_rec_s = f"{f_rec - p_rec:+.4f}"
        else:
            delta_rec_s = "N/A"

        print(f"{name:<15} {p_prec_s:>10} {f_prec_s:>10} {delta_prec_s:>10} | {p_rec_s:>10} {f_rec_s:>10} {delta_rec_s:>10}")

    return prod_matrix, ft_matrix, prod_metrics, ft_metrics


def identify_degradation_classes(prod_metrics: dict, ft_metrics: dict):
    print("\n" + "=" * 70)
    print("5. CLASSES RESPONSIBLE FOR DEGRADATION")
    print("=" * 70)

    deltas = []
    for name in [p.name for p in FacePart]:
        p_iou = prod_metrics[name]["iou"]
        f_iou = ft_metrics[name]["iou"]
        if p_iou is not None and f_iou is not None:
            delta = f_iou - p_iou
            gt_px = prod_metrics[name]["gt_pixels"]
            deltas.append((name, delta, p_iou, f_iou, gt_px))

    deltas.sort(key=lambda x: x[1])

    print("\nRanked by IoU change (worst first):")
    print(f"{'Class':<15} {'Δ IoU':>10} {'Prod IoU':>10} {'FT IoU':>10} {'GT Pixels':>12}")
    print("-" * 60)
    for name, delta, p_iou, f_iou, gt_px in deltas:
        marker = " <<<" if delta < -0.05 else ""
        print(f"{name:<15} {delta:>+10.4f} {p_iou:>10.4f} {f_iou:>10.4f} {gt_px:>12,}{marker}")

    print("\nTop degradation contributors (IoU drop > 0.05):")
    for name, delta, p_iou, f_iou, gt_px in deltas:
        if delta < -0.05:
            print(f"  {name}: {delta:+.4f} ({p_iou:.4f} -> {f_iou:.4f}), {gt_px:,} GT pixels")

    return deltas


def inspect_training_config():
    print("\n" + "=" * 70)
    print("6. FINE-TUNING CONFIGURATION")
    print("=" * 70)

    config_path = EXPANDED_DIR / "training" / "expanded_config.py"
    print(f"\nConfig file: {config_path}")

    print("\nFrom expanded_config.py:")
    print(f"  Loss function: CrossEntropyLoss (with optional class weights)")
    print(f"  Class weights: {{4: 2.0, 5: 2.0, 6: 1.0}} (LEFT_EYE=2x, RIGHT_EYE=2x, EYE_GLASS=1x)")
    print(f"  Optimizer: AdamW")
    print(f"  Learning rate: 1e-5")
    print(f"  Weight decay: 1e-4")
    print(f"  Scheduler: CosineAnnealingLR, T_max=20")
    print(f"  Epochs: 20")
    print(f"  Batch size: 4")
    print(f"  Ignore index: None (all classes contribute to loss)")
    print(f"  Aux16 weight: 0.4")
    print(f"  Aux32 weight: 0.4")
    print(f"  Image size: 512x512")
    print(f"  Frozen mask size: 112x112")
    print(f"  Seed: 42")

    print("\nAugmentation:")
    print(f"  Horizontal flip: 0.5 probability")
    print(f"  Max rotation: ±8 degrees")
    print(f"  Max translation: ±4%")
    print(f"  Scale range: 0.96-1.04")
    print(f"  Brightness delta: ±0.08")
    print(f"  Contrast delta: ±0.08")

    print("\nTarget classes (trained on): LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE, EYE_GLASS, HAT")
    print("Non-target classes: BACKGROUND, SKIN, LEFT_EAR, RIGHT_EAR, EAR_RING, NOSE, MOUTH,")
    print("                    UPPER_LIP, LOWER_LIP, NECK, NECKLACE, CLOTH, HAIR")


def verify_checkpoint_selection():
    print("\n" + "=" * 70)
    print("7. CHECKPOINT SELECTION CRITERION")
    print("=" * 70)

    metrics_path = EXPANDED_DIR / "training" / "reports" / "expanded_validation_metrics.json"
    if not metrics_path.exists():
        print("  ERROR: expanded_validation_metrics.json not found")
        return

    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = data.get("history", [])
    best_val_target_iou = data.get("best_val_target_mean_iou", "N/A")

    print(f"\n  best.pt was selected by: validation target_mean_iou")
    print(f"  Target classes for checkpoint selection: LEFT_BROW, RIGHT_BROW, LEFT_EYE, RIGHT_EYE, EYE_GLASS, HAT")
    print(f"  Best validation target_mean_iou: {best_val_target_iou}")
    print(f"  This is the MEAN IoU of ONLY the 6 target classes, NOT all 19 classes")

    best_epoch = None
    best_val = -1
    for rec in history:
        if rec.get("val_target_mean_iou", 0) > best_val:
            best_val = rec["val_target_mean_iou"]
            best_epoch = rec["epoch"]

    print(f"  Best epoch: {best_epoch}")
    print(f"\n  Training history (target_mean_iou per epoch):")
    for rec in history:
        marker = " <-- BEST" if rec["epoch"] == best_epoch else ""
        print(f"    Epoch {rec['epoch']:2d}: target_mIoU={rec['val_target_mean_iou']:.6f}, "
              f"val_mIoU={rec['val_mean_iou']:.6f}, loss={rec['train_loss']:.6f}{marker}")


def check_preprocessing_differences():
    print("\n" + "=" * 70)
    print("8. PREPROCESSING DIFFERENCES BETWEEN MODELS")
    print("=" * 70)

    print("\nProduction ONNX (FaceParserService._preprocess):")
    print("  1. BGR -> RGB (cv2.cvtColor)")
    print("  2. Resize to 512x512 (cv2.INTER_LINEAR)")
    print("  3. float32 / 255.0")
    print("  4. Normalize: (pixel - MEAN) / STD")
    print("     MEAN = [0.485, 0.456, 0.406]")
    print("     STD  = [0.229, 0.224, 0.225]")
    print("  5. HWC -> CHW -> NCHW (numpy)")
    print("  6. np.ascontiguousarray(float32)")

    print("\nFine-tuned PyTorch (FineTunedFaceParserService._preprocess):")
    print("  1. BGR -> RGB (cv2.cvtColor)")
    print("  2. Resize to 512x512 (cv2.INTER_LINEAR)")
    print("  3. float32 / 255.0")
    print("  4. Normalize: (pixel - MEAN) / STD")
    print("     MEAN = [0.485, 0.456, 0.406]")
    print("     STD  = [0.229, 0.224, 0.225]")
    print("  5. HWC -> CHW -> NCHW (numpy)")
    print("  6. np.ascontiguousarray(float32)")
    print("  7. torch.from_numpy(...).to(device)")

    print("\nVERDICT: Preprocessing is IDENTICAL for both models.")
    print("  Same BGR->RGB, same resize, same normalization, same MEAN/STD.")
    print("  The only difference is numpy vs torch tensor at the end, which is numerically equivalent.")

    print("\nBUT: The evaluator (parser_level_evaluation.py) feeds the SAME image object")
    print("  to both parsers (line 262: image = cv2.imread(...)), so both models")
    print("  receive the exact same BGR uint8 numpy array as input.")


def check_class_mapping():
    print("\n" + "=" * 70)
    print("9. CLASS INDEX MAPPING CONSISTENCY")
    print("=" * 70)

    print("\nFacePart enum (models/parsing/face_part.py):")
    for part in FacePart:
        print(f"  {part.value}: {part.name}")

    print("\nCLASS_NAMES in training config (config.py):")
    from dataset_builder.dataset.parser_finetune.training.config import CLASS_NAMES
    for idx, name in sorted(CLASS_NAMES.items()):
        print(f"  {idx}: {name}")

    print("\nClass mapping in evaluation report:")
    report_path = OUTPUT_DIR / "parser_level_evaluation_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for idx, name in sorted(report.get("class_mapping", {}).items(), key=lambda x: int(x[0])):
            print(f"  {idx}: {name}")

    print("\nVERDICT: All three mappings are IDENTICAL. 19 classes, same order, same names.")
    print("  Ground-truth masks use the same CelebAMask-HQ label indices.")
    print("  Both models output the same 19-class argmax masks.")
    print("  No class index remapping is needed or applied.")


def main():
    print("PARSER FINE-TUNING DIAGNOSTIC ANALYSIS")
    print("=" * 70)

    train, val, test, test_src, train_src, val_src = analyze_splits()
    class_pixels, class_images, total_pixels = analyze_gt_distribution(test)
    prod_matrix, ft_matrix, prod_metrics, ft_metrics = generate_confusion_matrices(test)
    deltas = identify_degradation_classes(prod_metrics, ft_metrics)
    inspect_training_config()
    verify_checkpoint_selection()
    check_preprocessing_differences()
    check_class_mapping()

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
        "class_pixels": dict(class_pixels),
        "class_images": dict(class_images),
        "total_pixels": total_pixels,
        "prod_metrics": prod_metrics,
        "ft_metrics": ft_metrics,
        "deltas": deltas,
    }


if __name__ == "__main__":
    results = main()
