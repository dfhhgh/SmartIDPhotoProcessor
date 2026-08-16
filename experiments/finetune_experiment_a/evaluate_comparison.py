"""Evaluate Experiment A against production and previous fine-tuned models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.parsing.face_part import FacePart
from services.face_parser_service import FaceParserService
from services.fine_tuned_face_parser_service import FineTunedFaceParserService
from experiments.parser_reproduction.bisenet_model import BiSeNet
from experiments.parser_reproduction.weight_mapping import load_onnx_to_pytorch
from dataset_builder.dataset.parser_finetune.training.metrics import SegmentationMetrics
from dataset_builder.dataset.parser_finetune.training.config import CLASS_NAMES, TARGET_CLASS_IDS

EXPANDED_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune_expanded"
IMAGES_DIR = EXPANDED_DIR / "images"
MASKS_DIR = EXPANDED_DIR / "annotation" / "corrected_masks"
SPLITS_DIR = EXPANDED_DIR / "splits"
EXPERIMENT_A_CHECKPOINT = PROJECT_ROOT / "experiments" / "finetune_experiment_a" / "checkpoints" / "best.pt"
PREVIOUS_FT_CHECKPOINT = EXPANDED_DIR / "training" / "checkpoints" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "finetune_experiment_a" / "reports"


def load_split(split_name: str) -> list[str]:
    path = SPLITS_DIR / f"{split_name}.txt"
    return sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


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

        result[name] = {
            "iou": iou,
            "dice": dice,
            "precision": precision,
            "recall": recall,
            "gt_pixels": int(gt_px),
        }
    return result


class ExperimentAParser:
    """Wrapper for Experiment A checkpoint."""

    def __init__(self, checkpoint_path: Path, device: torch.device):
        self.device = device
        self.model = BiSeNet(n_classes=19)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device)
        self.model.eval()

        self.MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def parse(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
        normalized = (resized.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        tensor = torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()

        class_map = preds[0]
        h, w = image.shape[:2]
        if class_map.shape != (h, w):
            mask = cv2.resize(class_map.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        else:
            mask = class_map.astype(np.int32)

        class Result:
            def __init__(self, mask):
                self.mask = mask
        return Result(mask)


def evaluate_model_on_test(model_name: str, parse_fn, test_ids: list[str]) -> dict:
    """Evaluate a model on the test set and return per-class metrics."""
    n_classes = 19
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    per_image = []

    for sid in test_ids:
        img_path = IMAGES_DIR / f"{sid}.png"
        mask_path = MASKS_DIR / f"{sid}.png"
        if not img_path.exists() or not mask_path.exists():
            continue

        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or gt is None:
            continue

        result = parse_fn(image)
        pred_mask = result.mask
        if pred_mask.shape != gt.shape:
            pred_mask = cv2.resize(pred_mask.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        cm = compute_confusion_matrix(pred_mask, gt, n_classes)
        matrix += cm
        per_image.append({"sample_id": sid, "confusion_matrix": cm.tolist()})

    metrics = per_class_metrics(matrix)

    # Compute summary stats
    ious = [v["iou"] for v in metrics.values() if v["iou"] is not None]
    target_ious = [metrics[CLASS_NAMES[i]]["iou"] for i in TARGET_CLASS_IDS if metrics[CLASS_NAMES[i]]["iou"] is not None]
    non_target_ids = [i for i in range(19) if i not in set(TARGET_CLASS_IDS)]
    non_target_ious = [metrics[CLASS_NAMES[i]]["iou"] for i in non_target_ids if metrics[CLASS_NAMES[i]]["iou"] is not None]

    total = int(matrix.sum())
    correct = int(np.trace(matrix))

    return {
        "model": model_name,
        "pixel_accuracy": float(correct / total) if total else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "target_mean_iou": float(np.mean(target_ious)) if target_ious else 0.0,
        "non_target_mean_iou": float(np.mean(non_target_ious)) if non_target_ious else 0.0,
        "per_class": metrics,
        "per_image": per_image,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_ids = load_split("test")
    print(f"Evaluating {len(test_ids)} test images...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Production ONNX
    print("\n--- Production ONNX ---")
    prod_parser = FaceParserService()
    prod_result = evaluate_model_on_test("production_onnx", prod_parser.parse, test_ids)
    print(f"  mIoU: {prod_result['mean_iou']:.6f}")
    print(f"  Target mIoU: {prod_result['target_mean_iou']:.6f}")
    print(f"  Non-target mIoU: {prod_result['non_target_mean_iou']:.6f}")

    # 2. Previous fine-tuned (full network)
    print("\n--- Previous Full-Network Fine-Tuned ---")
    prev_parser = FineTunedFaceParserService(checkpoint_path=PREVIOUS_FT_CHECKPOINT, device=device)
    prev_result = evaluate_model_on_test("previous_full_finetuned", prev_parser.parse, test_ids)
    print(f"  mIoU: {prev_result['mean_iou']:.6f}")
    print(f"  Target mIoU: {prev_result['target_mean_iou']:.6f}")
    print(f"  Non-target mIoU: {prev_result['non_target_mean_iou']:.6f}")

    # 3. Experiment A (head-only)
    print("\n--- Experiment A (Head-Only) ---")
    exp_a_parser = ExperimentAParser(checkpoint_path=EXPERIMENT_A_CHECKPOINT, device=device)
    exp_a_result = evaluate_model_on_test("experiment_a_head_only", exp_a_parser.parse, test_ids)
    print(f"  mIoU: {exp_a_result['mean_iou']:.6f}")
    print(f"  Target mIoU: {exp_a_result['target_mean_iou']:.6f}")
    print(f"  Non-target mIoU: {exp_a_result['non_target_mean_iou']:.6f}")

    # Build comparison table
    print("\n" + "=" * 100)
    print("PER-CLASS IoU COMPARISON")
    print("=" * 100)
    print(f"{'Class':<15} {'Production':>12} {'Prev FT':>12} {'Exp A':>12} {'ExpA vs Prod':>14} {'ExpA vs PrevFT':>14}")
    print("-" * 80)

    for i in range(19):
        name = CLASS_NAMES[i]
        p_iou = prod_result["per_class"][name]["iou"]
        f_iou = prev_result["per_class"][name]["iou"]
        a_iou = exp_a_result["per_class"][name]["iou"]

        p_s = f"{p_iou:.4f}" if p_iou is not None else "N/A"
        f_s = f"{f_iou:.4f}" if f_iou is not None else "N/A"
        a_s = f"{a_iou:.4f}" if a_iou is not None else "N/A"

        if p_iou is not None and a_iou is not None:
            d_prod = f"{a_iou - p_iou:+.4f}"
        else:
            d_prod = "N/A"

        if f_iou is not None and a_iou is not None:
            d_prev = f"{a_iou - f_iou:+.4f}"
        else:
            d_prev = "N/A"

        marker = ""
        if i in TARGET_CLASS_IDS:
            marker = " *"
        print(f"{name:<15} {p_s:>12} {f_s:>12} {a_s:>12} {d_prod:>14} {d_prev:>14}{marker}")

    print("-" * 80)
    print(f"{'MEAN IoU':<15} {prod_result['mean_iou']:>12.6f} {prev_result['mean_iou']:>12.6f} {exp_a_result['mean_iou']:>12.6f}")
    print(f"{'TARGET IoU':<15} {prod_result['target_mean_iou']:>12.6f} {prev_result['target_mean_iou']:>12.6f} {exp_a_result['target_mean_iou']:>12.6f}")
    print(f"{'NON-TARGET IoU':<15} {prod_result['non_target_mean_iou']:>12.6f} {prev_result['non_target_mean_iou']:>12.6f} {exp_a_result['non_target_mean_iou']:>12.6f}")
    print(f"{'PIXEL ACC':<15} {prod_result['pixel_accuracy']:>12.6f} {prev_result['pixel_accuracy']:>12.6f} {exp_a_result['pixel_accuracy']:>12.6f}")
    print("\n* = target class")

    # Save results
    comparison = {
        "test_images": len(test_ids),
        "production": {k: v for k, v in prod_result.items() if k != "per_image"},
        "previous_full_finetuned": {k: v for k, v in prev_result.items() if k != "per_image"},
        "experiment_a": {k: v for k, v in exp_a_result.items() if k != "per_image"},
    }
    (OUTPUT_DIR / "three_way_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8"
    )
    print(f"\nResults saved to {OUTPUT_DIR / 'three_way_comparison.json'}")


if __name__ == "__main__":
    main()
