"""
Evaluator orchestrator reusing the production PhotoValidationPipeline.
"""

from __future__ import annotations

import os
import time
from typing import Any
import cv2

from evaluation.models import (
    FaceMetrics,
    FacePartEvaluation,
    ImageEvaluationResult,
    ValidatorEvaluation,
)
from evaluation.json_exporter import JsonExporter
from evaluation.report_generator import ReportGenerator
from evaluation.statistics import StatisticsAggregator
from evaluation.summary import SummaryGenerator
from evaluation.visualization import Visualization
from evaluation.root_cause import RootCauseAnalyzer
from models.parsing.face_part import FacePart
from models.validation_type import ValidationType
from pipeline.photo_validation_pipeline import PhotoValidationPipeline
from reasoning.semantic_engine import SemanticEvidenceEngine
from config.constants import (
    FACE_VISIBILITY_REQUIRED_PART_THRESHOLDS,
    FACE_VISIBILITY_EYEBROW_THRESHOLDS,
)


class Evaluator:
    """Orchestrates the evaluation pipeline by reusing PhotoValidationPipeline

    and production components exclusively, ensuring zero logic duplication.
    """

    def __init__(self, output_dir: str = "evaluation_results") -> None:
        self._output_dir = output_dir
        self._pipeline = PhotoValidationPipeline()
        
        self._json_exporter = JsonExporter(output_dir)
        self._report_generator = ReportGenerator(output_dir)
        self._summary_generator = SummaryGenerator(output_dir)
        self._visualization = Visualization(output_dir)

    def evaluate_path(self, path: str) -> list[ImageEvaluationResult]:
        """Evaluate a single image path or directory of images."""
        if os.path.isfile(path):
            return [self.evaluate_image(path)]
        elif os.path.isdir(path):
            results: list[ImageEvaluationResult] = []
            valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            for root, _, files in os.walk(path):
                for file in files:
                    if file.lower().endswith(valid_exts):
                        img_path = os.path.join(root, file)
                        try:
                            results.append(self.evaluate_image(img_path))
                        except Exception as exc:
                            print(f"Error evaluating {img_path}: {exc}")
            
            if results:
                stats = StatisticsAggregator.aggregate(results)
                self._summary_generator.generate_dataset_summary(results, stats)
            return results
        else:
            raise ValueError(f"Path does not exist: {path}")

    def evaluate_image(self, image_path: str) -> ImageEvaluationResult:
        """Execute full evaluation on a single image file via PhotoValidationPipeline."""
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        h, w = image.shape[:2]
        start_time = time.perf_counter()

        # Execute production pipeline
        processing_result = self._pipeline.validate(image)
        exec_time_ms = (time.perf_counter() - start_time) * 1000.0

        validation_result = processing_result.validation_result
        primary_face = processing_result.selected_face

        detected_faces = 1 if primary_face is not None else 0
        primary_face_conf = float(primary_face.det_score) if primary_face is not None else 0.0
        selection_confidence = primary_face_conf

        # Extract validator evaluations from production ValidationMetrics
        validators_eval: list[ValidatorEvaluation] = []
        for metric in validation_result.metrics:
            validators_eval.append(
                ValidatorEvaluation(
                    validator_type=metric.type,
                    validator_name=metric.type.name.title().replace("_", "") + "Validator",
                    passed=metric.passed,
                    score=metric.score,
                    message=metric.message,
                )
            )

        # Extract face metrics
        face_metrics = None
        if primary_face is not None:
            bbox = tuple(float(x) for x in primary_face.bbox[:4])
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            area_ratio = area / max(w * h, 1)
            cx, cy = float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)
            pose = getattr(primary_face, "pose", (0.0, 0.0, 0.0))
            pitch, yaw, roll = pose if pose is not None else (0.0, 0.0, 0.0)
            face_metrics = FaceMetrics(
                bbox=bbox,
                area_ratio=area_ratio,
                center=(cx, cy),
                crop_size=(w, h),
                alignment_size=(112, 112),
                yaw=float(yaw),
                pitch=float(pitch),
                roll=float(roll),
                detection_confidence=primary_face_conf,
            )

        # Re-parse aligned image (unavoidable — production does not expose FaceParsingResult)
        from services.face_parser_service import FaceParserService
        parser_service = FaceParserService()
        target_eval_image = processing_result.aligned_image if processing_result.aligned_image is not None else image
        parsing_result = parser_service.parse(target_eval_image)

        # Semantic analysis — consume SemanticEvidenceEngine public APIs only
        engine = SemanticEvidenceEngine(parsing_result=parsing_result, face=primary_face)
        semantic_parts: dict[str, FacePartEvaluation] = {}

        target_parts = [
            (FacePart.LEFT_BROW, FacePart.RIGHT_BROW),
            (FacePart.LEFT_EYE, FacePart.RIGHT_EYE),
            FacePart.NOSE,
            (FacePart.MOUTH, FacePart.UPPER_LIP, FacePart.LOWER_LIP),
        ]

        for part_group in target_parts:
            if isinstance(part_group, tuple):
                parts = part_group
            else:
                parts = (part_group,)

            primary_part = parts[0]

            if primary_part in (FacePart.LEFT_EYE, FacePart.RIGHT_EYE):
                evidence = engine.compute_eye_evidence(primary_part)
            elif primary_part in (FacePart.LEFT_BROW, FacePart.RIGHT_BROW):
                eye_part = FacePart.LEFT_EYE if primary_part == FacePart.LEFT_BROW else FacePart.RIGHT_EYE
                evidence = engine.compute_eyebrow_evidence(primary_part, eye_part)
            elif primary_part == FacePart.NOSE:
                evidence = engine.compute_part_evidence(
                    FacePart.NOSE,
                    FACE_VISIBILITY_REQUIRED_PART_THRESHOLDS[FacePart.NOSE],
                )
            elif primary_part == FacePart.MOUTH:
                evidence = engine.compute_mouth_evidence()
            else:
                min_ratio = FACE_VISIBILITY_EYEBROW_THRESHOLDS.get(
                    primary_part, FACE_VISIBILITY_EYEBROW_THRESHOLDS[FacePart.LEFT_BROW]
                )
                evidence = engine.compute_part_evidence(primary_part, min_ratio)

            final_conf = evidence.final_confidence
            passed = evidence.passed

            for part in parts:
                pixels = parsing_result.part_area(part)
                ratio = parsing_result.part_ratio(part)

                semantic_parts[part.name] = FacePartEvaluation(
                    part_name=part.name,
                    pixels=pixels,
                    area_ratio=ratio,
                    parser_confidence=evidence.parser_confidence,
                    landmark_confidence=evidence.landmark_confidence,
                    pose_confidence=evidence.pose_confidence,
                    occlusion_confidence=evidence.occlusion_confidence,
                    eye_support_confidence=evidence.eye_support_confidence,
                    final_confidence=final_conf,
                    passed=passed,
                )

        # Parser statistics for all classes
        total_mask_pixels = parsing_result.total_pixels()
        parser_stats: dict[str, dict[str, Any]] = {}
        for part in FacePart:
            px = parsing_result.part_area(part)
            pct = (px / total_mask_pixels) * 100.0 if total_mask_pixels > 0 else 0.0
            parser_stats[part.name] = {
                "pixels": px,
                "percentage": pct,
            }

        overall_passed = validation_result.is_valid
        root_cause = RootCauseAnalyzer.analyze(validators_eval, semantic_parts) if not overall_passed else None

        output_paths = self._visualization.save_visualizations(image_name, image, parsing_result, primary_face)

        result = ImageEvaluationResult(
            image_name=image_name,
            resolution=(w, h),
            execution_time_ms=exec_time_ms,
            detected_faces=detected_faces,
            primary_face_confidence=primary_face_conf,
            selection_confidence=selection_confidence,
            face_metrics=face_metrics,
            validators=validators_eval,
            semantic_parts=semantic_parts,
            parser_statistics=parser_stats,
            root_cause=root_cause,
            overall_passed=overall_passed,
            output_paths=output_paths,
        )

        self._json_exporter.export(result)
        self._report_generator.generate_image_report(result)

        return result
