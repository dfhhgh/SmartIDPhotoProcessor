"""Parser-level evaluation for production ONNX and fine-tuned BiSeNet.

This module evaluates segmentation masks only. It does not run the photo
validation pipeline and does not use production parser outputs as ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from models.parsing.face_part import FacePart
from services.face_parser_service import FaceParserService
from services.fine_tuned_face_parser_service import FineTunedFaceParserService

SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
VALID_SPLITS = {"train", "val", "test", "all"}


@dataclass(frozen=True)
class ParserEvaluationSample:
    sample_id: str
    split: str
    source_category: str
    source_image: Path
    image_path: Path
    ground_truth_mask_path: Path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_path(value: str | None, project_root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _load_split_map(dataset_dir: Path) -> dict[str, str]:
    split_map: dict[str, str] = {}
    for split in ("train", "val", "test"):
        split_path = dataset_dir / "splits" / f"{split}.txt"
        if not split_path.exists():
            continue
        for line in split_path.read_text(encoding="utf-8").splitlines():
            sample_id = line.strip()
            if sample_id:
                split_map[sample_id] = split
    return split_map


def _load_samples(
    dataset_dir: Path,
    project_root: Path,
    split: str,
    evaluation_image_dir: Path | None = None,
) -> tuple[list[ParserEvaluationSample], list[str]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    manifest_path = dataset_dir / "reports" / "expanded_manifest.json"
    metadata_dir = dataset_dir / "metadata"
    annotation_metadata_dir = dataset_dir / "annotation" / "metadata"
    corrected_masks_dir = dataset_dir / "annotation" / "corrected_masks"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

    wanted_names: set[str] | None = None
    missing_evaluation_images: list[str] = []
    if evaluation_image_dir is not None:
        if not evaluation_image_dir.exists():
            raise FileNotFoundError(f"Evaluation image directory not found: {evaluation_image_dir}")
        wanted_names = {
            path.name
            for path in evaluation_image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        }

    manifest = _read_json(manifest_path)
    split_map = _load_split_map(dataset_dir)
    samples: list[ParserEvaluationSample] = []
    matched_names: set[str] = set()

    for entry in manifest.get("samples", []):
        sample_id = str(entry["sample_id"])
        sample_split = split_map.get(sample_id, "")
        if split != "all" and sample_split != split:
            continue

        metadata_path = metadata_dir / f"{sample_id}.json"
        annotation_metadata_path = annotation_metadata_dir / f"{sample_id}.json"
        if not metadata_path.exists() or not annotation_metadata_path.exists():
            continue

        metadata = _read_json(metadata_path)
        annotation_metadata = _read_json(annotation_metadata_path)
        if annotation_metadata.get("annotation_status") != "ACCEPT":
            continue

        source_image = _resolve_path(metadata.get("source_image"), project_root)
        image_path = _resolve_path(metadata.get("aligned_image"), project_root)
        gt_path = corrected_masks_dir / f"{sample_id}.png"
        if source_image is None or image_path is None:
            continue
        if wanted_names is not None and source_image.name not in wanted_names:
            continue
        if wanted_names is not None:
            matched_names.add(source_image.name)
        if not image_path.exists() or not gt_path.exists():
            continue

        samples.append(
            ParserEvaluationSample(
                sample_id=sample_id,
                split=sample_split,
                source_category=str(metadata.get("source_category", "")),
                source_image=source_image,
                image_path=image_path,
                ground_truth_mask_path=gt_path,
            )
        )

    if wanted_names is not None:
        missing_evaluation_images = sorted(wanted_names - matched_names)

    return sorted(samples, key=lambda item: item.sample_id), missing_evaluation_images


def _validate_mask(mask: np.ndarray | None, path: Path) -> np.ndarray:
    if mask is None:
        raise ValueError(f"Ground-truth mask is unreadable: {path}")
    if mask.ndim != 2:
        raise ValueError(f"Ground-truth mask must be single-channel: {path}")
    if mask.dtype != np.uint8:
        raise ValueError(f"Ground-truth mask must be uint8, got {mask.dtype}: {path}")
    if int(mask.min()) < 0 or int(mask.max()) > max(int(part) for part in FacePart):
        raise ValueError(f"Ground-truth mask has labels outside FacePart range: {path}")
    return mask


def _resize_prediction_to_gt(prediction: np.ndarray, gt_shape: tuple[int, int]) -> np.ndarray:
    if prediction.shape == gt_shape:
        return prediction.astype(np.int64)
    return cv2.resize(
        prediction.astype(np.uint8),
        (gt_shape[1], gt_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def _confusion_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    n_classes = len(FacePart)
    pred = prediction.astype(np.int64).ravel()
    gt = target.astype(np.int64).ravel()
    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
    bins = n_classes * gt[valid] + pred[valid]
    return np.bincount(bins, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _metrics_from_matrix(matrix: np.ndarray) -> dict[str, Any]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    per_class_iou: dict[str, float | None] = {}
    per_class_dice: dict[str, float | None] = {}
    per_class_pixel_accuracy: dict[str, float | None] = {}
    ious: list[float] = []
    dice_scores: list[float] = []

    for part in FacePart:
        idx = int(part)
        name = part.name
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - tp)
        fn = float(matrix[idx, :].sum() - tp)
        gt_pixels = float(matrix[idx, :].sum())
        union = tp + fp + fn
        dice_denom = 2.0 * tp + fp + fn

        iou = tp / union if union else None
        dice = 2.0 * tp / dice_denom if dice_denom else None
        class_acc = tp / gt_pixels if gt_pixels else None

        per_class_iou[name] = iou
        per_class_dice[name] = dice
        per_class_pixel_accuracy[name] = class_acc
        if iou is not None:
            ious.append(iou)
        if dice is not None:
            dice_scores.append(dice)

    return {
        "pixel_accuracy": float(correct / total) if total else 0.0,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "mean_dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
        "per_class_iou": per_class_iou,
        "per_class_dice": per_class_dice,
        "per_class_pixel_accuracy": per_class_pixel_accuracy,
    }


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _improvement(new_value: float | None, old_value: float | None) -> dict[str, float | None]:
    if new_value is None or old_value is None:
        return {"absolute": None, "relative_pct": None}
    relative = None if math.isclose(old_value, 0.0) else ((new_value - old_value) / old_value) * 100.0
    return {"absolute": new_value - old_value, "relative_pct": relative}


def evaluate_parsers(
    *,
    dataset_dir: Path,
    output_dir: Path,
    split: str = "test",
    evaluation_image_dir: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    samples, missing_evaluation_images = _load_samples(
        dataset_dir=dataset_dir,
        project_root=project_root,
        split=split,
        evaluation_image_dir=evaluation_image_dir,
    )
    if not samples:
        raise RuntimeError("No ACCEPT samples with real corrected masks were found for evaluation.")

    production_parser = FaceParserService()
    fine_tuned_parser = FineTunedFaceParserService(
        checkpoint_path=dataset_dir / "training" / "checkpoints" / "best.pt",
        device=device,
    )

    production_matrix = np.zeros((len(FacePart), len(FacePart)), dtype=np.int64)
    fine_tuned_matrix = np.zeros_like(production_matrix)
    per_image: list[dict[str, Any]] = []
    production_failures = 0
    fine_tuned_failures = 0

    for sample in samples:
        image = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        gt_mask = _validate_mask(
            cv2.imread(str(sample.ground_truth_mask_path), cv2.IMREAD_UNCHANGED),
            sample.ground_truth_mask_path,
        )
        if image is None:
            production_failures += 1
            fine_tuned_failures += 1
            per_image.append(
                {
                    "sample_id": sample.sample_id,
                    "image_name": sample.source_image.name,
                    "production_inference_succeeded": False,
                    "fine_tuned_inference_succeeded": False,
                    "error": f"Image is unreadable: {sample.image_path}",
                }
            )
            continue

        record: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "image_name": sample.source_image.name,
            "split": sample.split,
            "source_category": sample.source_category,
            "image_path": _safe_relative(sample.image_path, project_root),
            "ground_truth_mask_path": _safe_relative(sample.ground_truth_mask_path, project_root),
            "ground_truth_shape": list(gt_mask.shape),
        }

        try:
            prod_mask = _resize_prediction_to_gt(production_parser.parse(image).mask, gt_mask.shape)
            prod_matrix = _confusion_matrix(prod_mask, gt_mask)
            production_matrix += prod_matrix
            record["production_inference_succeeded"] = True
            record["production_metrics"] = _metrics_from_matrix(prod_matrix)
        except Exception as exc:  # noqa: BLE001 - report per-image inference failure.
            production_failures += 1
            record["production_inference_succeeded"] = False
            record["production_error"] = str(exc)
            record["production_metrics"] = None

        try:
            ft_mask = _resize_prediction_to_gt(fine_tuned_parser.parse(image).mask, gt_mask.shape)
            ft_matrix = _confusion_matrix(ft_mask, gt_mask)
            fine_tuned_matrix += ft_matrix
            record["fine_tuned_inference_succeeded"] = True
            record["fine_tuned_metrics"] = _metrics_from_matrix(ft_matrix)
        except Exception as exc:  # noqa: BLE001 - report per-image inference failure.
            fine_tuned_failures += 1
            record["fine_tuned_inference_succeeded"] = False
            record["fine_tuned_error"] = str(exc)
            record["fine_tuned_metrics"] = None

        per_image.append(record)

    production_metrics = _metrics_from_matrix(production_matrix)
    fine_tuned_metrics = _metrics_from_matrix(fine_tuned_matrix)
    prod_dice = production_metrics["mean_dice"]
    ft_dice = fine_tuned_metrics["mean_dice"]
    prod_iou = production_metrics["mean_iou"]
    ft_iou = fine_tuned_metrics["mean_iou"]
    prod_acc = production_metrics["pixel_accuracy"]
    ft_acc = fine_tuned_metrics["pixel_accuracy"]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "parser_level_evaluation",
        "dataset": {
            "dataset_dir": str(dataset_dir),
            "split": split,
            "evaluation_image_dir": str(evaluation_image_dir) if evaluation_image_dir else None,
            "missing_evaluation_images_without_accepted_ground_truth": missing_evaluation_images,
            "mask_resizing": (
                "Predicted masks are resized to the 112x112 corrected-mask shape with "
                "cv2.INTER_NEAREST only when needed; ground-truth masks are not resized."
            ),
        },
        "class_mapping": {str(int(part)): part.name for part in FacePart},
        "summary": {
            "evaluated_images": len(per_image),
            "production_failed_inference_cases": production_failures,
            "fine_tuned_failed_inference_cases": fine_tuned_failures,
            "mean_production_miou": prod_iou,
            "mean_fine_tuned_miou": ft_iou,
            "miou_improvement": _improvement(ft_iou, prod_iou),
            "mean_production_dice": prod_dice,
            "mean_fine_tuned_dice": ft_dice,
            "mean_dice_improvement": _improvement(ft_dice, prod_dice),
            "production_pixel_accuracy": prod_acc,
            "fine_tuned_pixel_accuracy": ft_acc,
            "pixel_accuracy_improvement": _improvement(ft_acc, prod_acc),
        },
        "production_metrics": production_metrics,
        "fine_tuned_metrics": fine_tuned_metrics,
        "per_image": per_image,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parser_level_evaluation_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_image_metrics.json").write_text(
        json.dumps(per_image, indent=2),
        encoding="utf-8",
    )
    _write_summary(output_dir / "parser_level_evaluation_summary.txt", report)
    return report


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "PARSER-LEVEL EVALUATION SUMMARY",
        "",
        f"Dataset: {report['dataset']['dataset_dir']}",
        f"Split/filter: {report['dataset']['split']}",
        f"Evaluated images: {summary['evaluated_images']}",
        f"Production failed inference cases: {summary['production_failed_inference_cases']}",
        f"Fine-tuned failed inference cases: {summary['fine_tuned_failed_inference_cases']}",
        "",
        f"Production mIoU: {summary['mean_production_miou']:.6f}",
        f"Fine-tuned mIoU: {summary['mean_fine_tuned_miou']:.6f}",
        f"Absolute mIoU improvement: {summary['miou_improvement']['absolute']:.6f}",
        f"Relative mIoU improvement (%): {summary['miou_improvement']['relative_pct']:.6f}",
        "",
        f"Production mean Dice/F1: {summary['mean_production_dice']:.6f}",
        f"Fine-tuned mean Dice/F1: {summary['mean_fine_tuned_dice']:.6f}",
        f"Absolute mean Dice improvement: {summary['mean_dice_improvement']['absolute']:.6f}",
        "",
        f"Production pixel accuracy: {summary['production_pixel_accuracy']:.6f}",
        f"Fine-tuned pixel accuracy: {summary['fine_tuned_pixel_accuracy']:.6f}",
        f"Absolute pixel accuracy improvement: {summary['pixel_accuracy_improvement']['absolute']:.6f}",
        "",
        "Ground-truth handling:",
        str(report["dataset"]["mask_resizing"]),
    ]
    missing = report["dataset"]["missing_evaluation_images_without_accepted_ground_truth"]
    if missing:
        lines.extend(["", "Evaluation images without accepted ground-truth masks:", *missing])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate BiSeNet parsers against real masks.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=project_root / "dataset_builder" / "dataset" / "parser_finetune_expanded",
    )
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument(
        "--evaluation-image-dir",
        type=Path,
        default=None,
        help="Optional directory of evaluation images; only annotated matching source filenames are evaluated.",
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "reports")
    parser.add_argument("--device", default=None, help="Optional PyTorch device for the fine-tuned model.")
    args = parser.parse_args()

    report = evaluate_parsers(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        split=args.split,
        evaluation_image_dir=args.evaluation_image_dir,
        device=args.device,
    )
    summary = report["summary"]
    print("Parser-level evaluation complete")
    print(f"Evaluated images: {summary['evaluated_images']}")
    print(f"Production mIoU: {summary['mean_production_miou']:.6f}")
    print(f"Fine-tuned mIoU: {summary['mean_fine_tuned_miou']:.6f}")
    print(f"Absolute mIoU improvement: {summary['miou_improvement']['absolute']:.6f}")


if __name__ == "__main__":
    main()
