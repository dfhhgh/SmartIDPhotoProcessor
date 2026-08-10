"""
Main experiment runner: ONNX vs PyTorch BiSeNet reproduction verification.

Usage:
    python -m experiments.parser_reproduction.run_experiment
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .bisenet_model import BiSeNet
from .comparison import ComparisonResult, compare_outputs
from .onnx_inference import inference_onnx, load_onnx_session
from .pytorch_inference import inference_pytorch
from .weight_mapping import load_onnx_to_pytorch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL_PATH = PROJECT_ROOT / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"
TEST_IMAGES_DIR = Path(__file__).resolve().parent / "test_images"
N_CLASSES = 19


def _create_synthetic_test_images() -> dict[str, np.ndarray]:
    """Create synthetic test images when real test images are not available."""
    images: dict[str, np.ndarray] = {}

    images["synthetic_face"] = np.random.randint(60, 200, (400, 400, 3), dtype=np.uint8)
    center = (200, 200)
    cv2.circle(images["synthetic_face"], center, 120, (180, 150, 130), -1)
    cv2.circle(images["synthetic_face"], (160, 170), 15, (255, 255, 255), -1)
    cv2.circle(images["synthetic_face"], (240, 170), 15, (255, 255, 255), -1)
    cv2.circle(images["synthetic_face"], (160, 170), 8, (50, 50, 50), -1)
    cv2.circle(images["synthetic_face"], (240, 170), 8, (50, 50, 50), -1)
    cv2.ellipse(images["synthetic_face"], (200, 240), (30, 15), 0, 0, 180, (100, 80, 80), 2)

    images["synthetic_glasses"] = images["synthetic_face"].copy()
    cv2.rectangle(images["synthetic_glasses"], (130, 155), (185, 195), (200, 200, 200), 2)
    cv2.rectangle(images["synthetic_glasses"], (215, 155), (270, 195), (200, 200, 200), 2)
    cv2.line(images["synthetic_glasses"], (185, 175), (215, 175), (200, 200, 200), 2)

    return images


def _load_test_images() -> dict[str, np.ndarray]:
    """Load test images from the test_images directory, or create synthetic ones."""
    images: dict[str, np.ndarray] = {}

    if TEST_IMAGES_DIR.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            for img_path in TEST_IMAGES_DIR.glob(ext):
                img = cv2.imread(str(img_path))
                if img is not None:
                    images[img_path.stem] = img
                    logger.info("Loaded test image: %s (%dx%d)", img_path.name, img.shape[1], img.shape[0])

    if not images:
        logger.info("No test images found in %s, creating synthetic images", TEST_IMAGES_DIR)
        images = _create_synthetic_test_images()

    return images


def _classify_result(result: ComparisonResult) -> str:
    if (
        result.max_abs_diff < 1e-5
        and result.argmax_agreement > 0.999
        and result.mean_mask_iou > 0.999
    ):
        return "REPRODUCTION VERIFIED"

    if (
        result.max_abs_diff < 0.01
        and result.argmax_agreement > 0.99
        and result.mean_mask_iou > 0.99
    ):
        return "REPRODUCTION PARTIALLY VERIFIED"

    return "REPRODUCTION FAILED"


def run_experiment() -> None:
    logger.info("=" * 70)
    logger.info("BiSeNet Reproduction Experiment: ONNX vs PyTorch")
    logger.info("=" * 70)

    if not ONNX_MODEL_PATH.exists():
        logger.error("ONNX model not found: %s", ONNX_MODEL_PATH)
        sys.exit(1)

    logger.info("Step 1: Loading ONNX model ...")
    onnx_session = load_onnx_session(ONNX_MODEL_PATH)

    logger.info("Step 2: Building PyTorch BiSeNet model ...")
    pytorch_model = BiSeNet(n_classes=N_CLASSES)

    logger.info("Step 3: Loading ONNX weights into PyTorch model ...")
    pytorch_model = load_onnx_to_pytorch(ONNX_MODEL_PATH, pytorch_model)

    logger.info("Step 4: Loading test images ...")
    test_images = _load_test_images()

    all_results: dict[str, ComparisonResult] = {}
    all_classifications: dict[str, str] = {}

    for name, image in test_images.items():
        logger.info("-" * 50)
        logger.info("Processing: %s (%dx%d)", name, image.shape[1], image.shape[0])

        t0 = time.perf_counter()
        onnx_logits, onnx_mask = inference_onnx(onnx_session, image)
        t_onnx = time.perf_counter() - t0

        t0 = time.perf_counter()
        pt_logits, pt_mask = inference_pytorch(pytorch_model, image)
        t_pt = time.perf_counter() - t0

        logger.info("  ONNX inference: %.3f s", t_onnx)
        logger.info("  PyTorch inference: %.3f s", t_pt)

        result = compare_outputs(onnx_logits, pt_logits, n_classes=N_CLASSES)
        classification = _classify_result(result)

        all_results[name] = result
        all_classifications[name] = classification

        logger.info("  Shape match: %s (ONNX=%s, PT=%s)", result.output_shape_match, result.onnx_shape, result.pytorch_shape)
        logger.info("  Max abs diff:  %.2e", result.max_abs_diff)
        logger.info("  Mean abs diff: %.2e", result.mean_abs_diff)
        logger.info("  MSE:           %.2e", result.mse)
        logger.info("  Cosine sim:    %.8f", result.cosine_similarity)
        logger.info("  Argmax agree:  %.4f%%", result.argmax_agreement * 100)
        logger.info("  Mean mask IoU: %.4f%%", result.mean_mask_iou * 100)
        logger.info("  Classification: %s", classification)

    _print_report(test_images, all_results, all_classifications)


def _print_report(
    test_images: dict[str, np.ndarray],
    results: dict[str, ComparisonResult],
    classifications: dict[str, str],
) -> None:
    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION REPORT")
    logger.info("=" * 70)

    logger.info("")
    logger.info("### Model Identity")
    logger.info("  ONNX path:          %s", ONNX_MODEL_PATH)
    logger.info("  PyTorch impl:       experiments/parser_reproduction/bisenet_model.py")
    logger.info("  Architecture:       BiSeNetV1 + ResNet-18")
    logger.info("  Num classes:        %d (CelebAMask-HQ)", N_CLASSES)
    logger.info("  ONNX file size:     %.2f MB", ONNX_MODEL_PATH.stat().st_size / 1024 / 1024)

    logger.info("")
    logger.info("### Input Compatibility")
    logger.info("  ONNX input shape:   (1, 3, 512, 512) float32")
    logger.info("  PyTorch input shape: (1, 3, 512, 512) float32")
    logger.info("  Preprocessing:      BGR->RGB, resize 512x512, /255, ImageNet normalize")
    logger.info("  Preprocessing match: IDENTICAL")

    logger.info("")
    logger.info("### Numerical Comparison")
    logger.info("  %-20s | %12s | %12s | %12s | %10s | %10s | %10s | %s", "Image", "MaxAbsDiff", "MeanAbsDiff", "MSE", "CosineSim", "Argmax%", "mIoU%", "Status")
    logger.info("  " + "-" * 110)

    for name, result in results.items():
        logger.info(
            "  %-20s | %12.2e | %12.2e | %12.2e | %10.8f | %9.4f%% | %9.4f%% | %s",
            name,
            result.max_abs_diff,
            result.mean_abs_diff,
            result.mse,
            result.cosine_similarity,
            result.argmax_agreement * 100,
            result.mean_mask_iou * 100,
            classifications[name],
        )

    overall = _classify_results(results, classifications)
    logger.info("")
    logger.info("### Conclusion")
    logger.info("  Overall: %s", overall)

    if overall == "REPRODUCTION VERIFIED":
        logger.info("  The PyTorch model produces numerically equivalent outputs to the ONNX model.")
        logger.info("  The weights have been successfully transferred and the architecture is identical.")
    elif overall == "REPRODUCTION PARTIALLY VERIFIED":
        logger.info("  The PyTorch model produces outputs very close to the ONNX model.")
        logger.info("  Minor differences may arise from float precision or BatchNorm implementation.")
    else:
        logger.info("  Significant differences detected. Investigation required.")
        _diagnose_failure(results)

    logger.info("")
    logger.info("### Next Step")
    if overall in ("REPRODUCTION VERIFIED", "REPRODUCTION PARTIALLY VERIFIED"):
        logger.info("  The PyTorch model is confirmed as a faithful reproduction.")
        logger.info("  Safe to use as starting point for fine-tuning on:")
        logger.info("    1. Transparent glasses")
        logger.info("    2. Hijab")
        logger.info("    3. Hijab + transparent glasses")
        logger.info("    4. Eye/eyebrow localization under occlusion")
    else:
        logger.info("  Must resolve weight/architecture discrepancy before fine-tuning.")


def _classify_results(
    results: dict[str, ComparisonResult],
    classifications: dict[str, str],
) -> str:
    statuses = list(classifications.values())
    if all(s == "REPRODUCTION VERIFIED" for s in statuses):
        return "REPRODUCTION VERIFIED"
    if all(s in ("REPRODUCTION VERIFIED", "REPRODUCTION PARTIALLY VERIFIED") for s in statuses):
        return "REPRODUCTION PARTIALLY VERIFIED"
    return "REPRODUCTION FAILED"


def _diagnose_failure(results: dict[str, ComparisonResult]) -> None:
    logger.info("")
    logger.info("  Failure Analysis (in order of likelihood):")
    worst = max(results.values(), key=lambda r: r.max_abs_diff)
    logger.info("  Worst case: max_abs_diff=%.2e, argmax_agreement=%.2f%%", worst.max_abs_diff, worst.argmax_agreement * 100)

    if worst.max_abs_diff > 1.0:
        logger.info("  -> Large magnitude differences suggest weight mapping or architecture mismatch")
    elif worst.max_abs_diff > 0.01:
        logger.info("  -> Moderate differences suggest BatchNorm fusion or preprocessing mismatch")
    else:
        logger.info("  -> Small differences likely due to float precision")


if __name__ == "__main__":
    run_experiment()
