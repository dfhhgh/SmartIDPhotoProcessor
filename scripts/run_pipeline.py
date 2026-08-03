#!/usr/bin/env python3
"""
Integration testing script for the Photo Validation Pipeline.

Executes the pipeline steps on real images (single image path or directory),
saves cropped images immediately after FaceCropper succeeds and aligned images
immediately after FaceAligner succeeds (even if FaceParserService or subsequent
validators throw an exception), produces text reports, prints console reports,
and outputs a final execution summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pipeline.detector import FaceDetector
from pipeline.selector import FaceSelector
from pipeline.cropper import FaceCropper
from pipeline.face_coordinate_transformer import FaceCoordinateTransformer
from pipeline.aligner import FaceAligner
from pipeline.photo_exporter import PhotoExporter
from services.face_parser_service import FaceParserService
from pipeline.validation_orchestrator import ValidationOrchestrator
from validators.face_ambiguity_validator import FaceAmbiguityValidator
from models.validation_result import ValidationResult

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageLoadError(Exception):
    """Raised when OpenCV fails to load an image."""
    pass


class PipelineExecutionError(Exception):
    """Raised when the pipeline encounters an error during execution."""
    pass


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Photo Validation Pipeline on images or directories."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single image file or a directory containing images.",
    )
    return parser.parse_args()


def collect_image_paths(target_path: Path) -> list[Path]:
    """Collect all supported image paths from a file or directory."""
    image_paths: list[Path] = []
    if target_path.is_file():
        if target_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            image_paths.append(target_path)
        else:
            print(
                f"Error: Unsupported file format '{target_path.suffix}'. "
                f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}",
                file=sys.stderr,
            )
    elif target_path.is_dir():
        for p in sorted(target_path.rglob("*")):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                image_paths.append(p)
    else:
        print(f"Error: Path '{target_path}' does not exist or is not valid.", file=sys.stderr)
    return image_paths


def process_image(
    img_path: Path,
    aligned_dir: Path,
    cropped_dir: Path,
    reports_dir: Path,
    exported_dir: Path | None = None,
) -> tuple[str, int]:
    """Process a single image through pipeline components with immediate debugging saves.

    Returns:
        A tuple of (outcome: str, elapsed_ms: int) where outcome is
        'valid', 'invalid', or 'processing_error'.
    """
    if exported_dir is None:
        exported_dir = reports_dir.parent / "exported"
    exported_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    overall_status = "PROCESSING_ERROR"
    error_category = None
    error_message = None
    metrics_data = []
    outcome = "processing_error"

    try:
        # Load image using OpenCV
        image = cv2.imread(str(img_path))
        if image is None:
            raise ImageLoadError(f"OpenCV failed to load image from path: {img_path}")

        try:
            detector = FaceDetector()
            selector = FaceSelector()
            cropper = FaceCropper()
            transformer = FaceCoordinateTransformer()
            aligner = FaceAligner()
            ambiguity_validator = FaceAmbiguityValidator()
            parser_service = FaceParserService()
            exporter = PhotoExporter()
            from models.validation_execution_mode import ValidationExecutionMode

            orchestrator = ValidationOrchestrator(
                parser_service=parser_service,
                execution_mode=ValidationExecutionMode.DEVELOPMENT,
            )
            faces = detector.detect(image)
            selection_result = selector.select(faces, image.shape)
            ambiguity_metric = ambiguity_validator.validate(selection_result)

            if not ambiguity_metric.passed:
                validation_result = ValidationResult(metrics=[ambiguity_metric])
            else:
                selected_face = selection_result.selected_face
                crop_result = cropper.crop(image, selected_face)

                # Save cropped image immediately after FaceCropper succeeds
                if crop_result.image is not None and isinstance(crop_result.image, np.ndarray) and crop_result.image.size > 0:
                    cropped_output_path = cropped_dir / img_path.name
                    cv2.imwrite(str(cropped_output_path), crop_result.image)

                transformed_face = transformer.transform(
                    selected_face,
                    crop_result.crop_x,
                    crop_result.crop_y,
                )
                alignment_result = aligner.align(crop_result.image, transformed_face)

                # Save aligned image immediately after FaceAligner succeeds
                if alignment_result.aligned_image is not None and isinstance(alignment_result.aligned_image, np.ndarray) and alignment_result.aligned_image.size > 0:
                    aligned_output_path = aligned_dir / img_path.name
                    cv2.imwrite(str(aligned_output_path), alignment_result.aligned_image)

                parsing_result = parser_service.parse(alignment_result.aligned_image)
                validation_result = orchestrator.validate(
                    image=alignment_result.aligned_image,
                    face=alignment_result.aligned_face,
                    parsing_result=parsing_result,
                    original_image=image,
                    original_face=selected_face,
                )

                if True:  # Always export if validation passes
                    export_result = exporter.export(crop_result.image)
                    exported_output_path = exported_dir / img_path.name
                    cv2.imwrite(str(exported_output_path), export_result.exported_image)

        except Exception as e:
            raise PipelineExecutionError(f"Pipeline validation failed: {e}") from e

        is_valid = validation_result.is_valid
        overall_status = "VALID" if is_valid else "INVALID"
        outcome = "valid" if is_valid else "invalid"

        # Collect metrics info
        for metric in validation_result.metrics:
            metrics_data.append(
                {
                    "name": metric.type.name,
                    "passed": metric.passed,
                    "score": metric.score,
                    "message": metric.message,
                }
            )

    except ImageLoadError as e:
        error_category = "Image Loading Failure"
        error_message = str(e)
        overall_status = "PROCESSING_ERROR"
        outcome = "processing_error"
    except PipelineExecutionError as e:
        error_category = "Pipeline Execution Failure"
        error_message = str(e)
        overall_status = "PROCESSING_ERROR"
        outcome = "processing_error"
    except Exception as e:
        error_category = "Unexpected Error"
        error_message = str(e)
        overall_status = "PROCESSING_ERROR"
        outcome = "processing_error"

    end_time = time.perf_counter()
    elapsed_ms = int((end_time - start_time) * 1000)

    # Print console report
    print("=" * 52)
    print(f"Image: {img_path.name}")
    print("=" * 52)
    print()
    print(f"Overall Result : {overall_status}")
    print()

    if error_category and error_message:
        print(f"Error Category : {error_category}")
        print(f"Error Details  : {error_message}")
        print()
    else:
        for m in metrics_data:
            status_str = "PASS" if m["passed"] else "FAIL"
            score_str = f"Score: {m['score']:.2f}"
            line = f"{m['name']:<18} {status_str:<6} {score_str}"
            if not m["passed"] and m["message"]:
                line += f"   Message: {m['message']}"
            print(line)
        print()

    print("-" * 52)
    print()
    print(f"Processing Time: {elapsed_ms} ms")
    print()

    # Save text report
    report_path = reports_dir / f"{img_path.stem}.txt"
    report_lines = [
        f"Image Name    : {img_path.name}",
        f"Overall Result: {overall_status}",
    ]
    if error_category and error_message:
        report_lines.append(f"Error Category: {error_category}")
        report_lines.append(f"Error Details : {error_message}")
    else:
        report_lines.append("")
        report_lines.append("Validators:")
        for m in metrics_data:
            status_str = "PASS" if m["passed"] else "FAIL"
            msg_part = f" - Message: {m['message']}" if m["message"] else ""
            report_lines.append(
                f"  {m['name']:<18} : {status_str}   Score: {m['score']:.2f}{msg_part}"
            )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return outcome, elapsed_ms


def main() -> None:
    """Main entry point for the integration testing script."""
    args = parse_arguments()
    target_path = Path(args.path)

    image_paths = collect_image_paths(target_path)
    if not image_paths:
        print(f"No valid images found at: {target_path}", file=sys.stderr)
        sys.exit(1)

    outputs_dir = Path("outputs")
    aligned_dir = outputs_dir / "aligned"
    cropped_dir = outputs_dir / "cropped"
    exported_dir = outputs_dir / "exported"
    reports_dir = outputs_dir / "reports"

    aligned_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    exported_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    valid_count = 0
    invalid_count = 0
    error_count = 0
    total_time = 0

    print(f"Starting pipeline execution for {len(image_paths)} image(s)...\n")

    for img_path in image_paths:
        total_processed += 1
        outcome, elapsed_ms = process_image(
            img_path=img_path,
            aligned_dir=aligned_dir,
            cropped_dir=cropped_dir,
            reports_dir=reports_dir,
            exported_dir=exported_dir,
        )
        total_time += elapsed_ms
        if outcome == "valid":
            valid_count += 1
        elif outcome == "invalid":
            invalid_count += 1
        else:
            error_count += 1

    avg_time = int(total_time / total_processed) if total_processed > 0 else 0

    print("======================================")
    print("SUMMARY")
    print("======================================")
    print()
    print(f"Images Processed : {total_processed}")
    print(f"Valid Images     : {valid_count}")
    print(f"Invalid Images   : {invalid_count}")
    print(f"Processing Errors: {error_count}")
    print(f"Average Time     : {avg_time} ms")
    print()


if __name__ == "__main__":
    main()
