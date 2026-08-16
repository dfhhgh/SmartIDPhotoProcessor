"""Evaluation script comparing production ONNX BiSeNet and fine-tuned PyTorch best.pt on good test images."""

from __future__ import annotations

import argparse
import json
import logging
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

from config.settings import Settings
from pipeline.photo_validation_pipeline import PhotoValidationPipeline
from models.validation_execution_mode import ValidationExecutionMode
from services.face_parser_service import FaceParserService
from services.fine_tuned_face_parser_service import FineTunedFaceParserService

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def compute_file_hash(path: Path) -> str:
    if not path.exists():
        return "not_found"
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def evaluate_pipelines(image_paths: list[Path], output_dir: Path) -> dict:
    settings = Settings()
    onnx_model_path = settings.MODEL_ROOT / "bisenet" / "bisenet_resnet18.onnx"
    onnx_hash_before = compute_file_hash(onnx_model_path)

    project_root = Path(__file__).resolve().parents[1]
    best_pt_path = project_root / "dataset_builder" / "dataset" / "parser_finetune_expanded" / "training" / "checkpoints" / "best.pt"
    best_pt_hash = compute_file_hash(best_pt_path)

    logger.info("Initializing production pipeline (ONNX parser)...")
    prod_pipeline = PhotoValidationPipeline(execution_mode=ValidationExecutionMode.DEVELOPMENT)
    prod_parser = FaceParserService()

    logger.info("Initializing evaluation pipeline (Fine-tuned best.pt parser)...")
    finetuned_parser = FineTunedFaceParserService(checkpoint_path=best_pt_path)
    finetuned_pipeline = PhotoValidationPipeline(
        parser_service=finetuned_parser,
        execution_mode=ValidationExecutionMode.DEVELOPMENT,
    )

    onnx_hash_after = compute_file_hash(onnx_model_path)
    assert onnx_hash_before == onnx_hash_after, "CRITICAL: Production ONNX model hash changed!"

    results = []
    reached_parsing_both = 0
    prod_face_fail = 0
    ft_face_fail = 0
    prod_valid_count = 0
    ft_valid_count = 0
    both_valid_count = 0
    prod_only_valid = 0
    ft_only_valid = 0
    both_invalid = 0
    agreement_scores = []

    for img_path in image_paths:
        logger.info("Processing image: %s", img_path.name)
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Could not read image: %s", img_path)
            continue

        prod_valid = False
        prod_metrics = []
        prod_aligned = None
        prod_error = None
        try:
            prod_result = prod_pipeline.validate(img)
            prod_valid = prod_result.validation_result.is_valid
            prod_metrics = [
                {
                    "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                    "passed": m.passed,
                    "score": m.score,
                    "message": m.message,
                }
                for m in prod_result.validation_result.metrics
            ]
            prod_aligned = prod_result.aligned_image
        except Exception as e:
            prod_error = str(e)
            prod_face_fail += 1

        ft_valid = False
        ft_metrics = []
        ft_aligned = None
        ft_error = None
        try:
            ft_result = finetuned_pipeline.validate(img)
            ft_valid = ft_result.validation_result.is_valid
            ft_metrics = [
                {
                    "type": m.type.value if hasattr(m.type, "value") else str(m.type),
                    "passed": m.passed,
                    "score": m.score,
                    "message": m.message,
                }
                for m in ft_result.validation_result.metrics
            ]
            ft_aligned = ft_result.aligned_image
        except Exception as e:
            ft_error = str(e)
            ft_face_fail += 1

        prod_mask = None
        if prod_aligned is not None:
            try:
                prod_mask = prod_parser.parse(prod_aligned).mask
            except Exception:
                pass

        ft_mask = None
        if ft_aligned is not None:
            try:
                ft_mask = finetuned_parser.parse(ft_aligned).mask
            except Exception:
                pass

        reached_parsing = (prod_mask is not None) and (ft_mask is not None)
        if reached_parsing:
            reached_parsing_both += 1

        mask_agreement = None
        if prod_mask is not None and ft_mask is not None and prod_mask.shape == ft_mask.shape:
            mask_agreement = float(np.mean(prod_mask == ft_mask) * 100.0)
            agreement_scores.append(mask_agreement)

        if prod_valid:
            prod_valid_count += 1
        if ft_valid:
            ft_valid_count += 1

        if prod_valid and ft_valid:
            both_valid_count += 1
        elif prod_valid and not ft_valid:
            prod_only_valid += 1
        elif not prod_valid and ft_valid:
            ft_only_valid += 1
        else:
            both_invalid += 1

        record = {
            "image_path": str(img_path),
            "production_valid": prod_valid,
            "finetuned_valid": ft_valid,
            "production_error": prod_error,
            "finetuned_error": ft_error,
            "reached_parsing": reached_parsing,
            "mask_agreement_pct": mask_agreement,
            "production_metrics": prod_metrics,
            "finetuned_metrics": ft_metrics,
            "aligned_shape": list(prod_aligned.shape) if prod_aligned is not None else None,
        }
        results.append(record)

    total_images = len(results)
    mean_agreement = float(np.mean(agreement_scores)) if agreement_scores else 0.0

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "finetune_evaluation_report.json"
    report_data = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_model": {
            "name": "bisenet_resnet18.onnx",
            "path": str(onnx_model_path),
            "sha256": onnx_hash_after,
        },
        "fine_tuned_model": {
            "name": "best.pt",
            "path": str(best_pt_path),
            "sha256": best_pt_hash,
        },
        "statistics": {
            "total_images": total_images,
            "successfully_processed_by_both": total_images - max(prod_face_fail, ft_face_fail),
            "reached_parsing_both": reached_parsing_both,
            "production_face_detection_failures": prod_face_fail,
            "finetuned_face_detection_failures": ft_face_fail,
            "production_valid_count": prod_valid_count,
            "finetuned_valid_count": ft_valid_count,
            "both_valid_count": both_valid_count,
            "production_only_valid_count": prod_only_valid,
            "finetuned_only_valid_count": ft_only_valid,
            "both_invalid_count": both_invalid,
            "mean_mask_agreement_pct": mean_agreement,
        },
        "results": results,
    }
    report_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    logger.info("Evaluation report saved to %s", report_path)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total images: {total_images}")
    print(f"Both pipelines reached parsing: {reached_parsing_both}")
    print(f"Production valid: {prod_valid_count}")
    print(f"Fine-tuned valid: {ft_valid_count}")
    print(f"Both valid: {both_valid_count}")
    print(f"Mean mask agreement: {mean_agreement:.2f}%")
    print("=" * 60)

    return report_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BiSeNet vs Production ONNX")
    project_root = Path(__file__).resolve().parents[1]
    default_image_dir = project_root / "test_images" / "good"

    parser.add_argument(
        "--image-dir",
        type=Path,
        default=default_image_dir,
        help="Directory containing test images (default: test_images/good)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of images to evaluate",
    )
    args = parser.parse_args()

    image_dir = args.image_dir
    if not image_dir.exists():
        logger.error("Image directory not found: %s", image_dir)
        return

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if args.limit is not None:
        image_paths = image_paths[:args.limit]

    if not image_paths:
        logger.error("No supported test images found in %s", image_dir)
        return

    logger.info("Evaluating %d images from %s", len(image_paths), image_dir)
    output_dir = project_root / "dataset_builder" / "dataset" / "parser_finetune_expanded" / "training" / "reports"

    evaluate_pipelines(image_paths, output_dir)


if __name__ == "__main__":
    main()
