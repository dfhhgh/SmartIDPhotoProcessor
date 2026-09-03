"""
Photo validation pipeline orchestration module.

Connects the detection, selection, cropping, coordinate transformation,
alignment, parsing, validation, and export components into a single sequential workflow.
"""

from __future__ import annotations

import logging

import numpy as np

from models.photo_processing_result import PhotoProcessingResult
from models.validation_execution_mode import ValidationExecutionMode
from models.validation_result import ValidationResult
from pipeline.aligner import FaceAligner
from pipeline.cropper import FaceCropper
from pipeline.detector import FaceDetector
from pipeline.face_coordinate_transformer import FaceCoordinateTransformer
from pipeline.photo_exporter import PhotoExporter
from pipeline.selector import FaceSelector
from pipeline.validation_orchestrator import ValidationOrchestrator
from services.face_parser_service import FaceParserService
from services.reverse_search_manager import ReverseSearchServiceManager
from search.reverse_search_service import ReverseSearchService, ReverseSearchResult
from validators.base_selection_validator import BaseSelectionValidator
from validators.face_ambiguity_validator import FaceAmbiguityValidator

logger = logging.getLogger(__name__)


class PhotoValidationPipeline:
    """Orchestrates the complete photo validation pipeline.

    Execution steps:
    1. Detect all faces in the image.
    2. Select the primary face, producing a SelectionResult with confidence
       metadata (score margin, ambiguity ratio, candidate count).
    3. Validate selection reliability via FaceAmbiguityValidator. If the
       selection is ambiguous (e.g. two competing faces), short-circuit
       and return early without cropping, aligning, or running the
       ValidationOrchestrator.
    4. Crop the selected face (producing CropResult with image and crop offset).
    5. Transform face coordinates using FaceCoordinateTransformer.
    6. Align the cropped image and face into aligned coordinate space.
    7. Run face parsing on the aligned face.
    8. Execute validation orchestrator on the aligned image, aligned face, and parsing result
       (with original image passed to FaceSizeValidator).
    9. Export validated photo via PhotoExporter if valid.
    10. Return PhotoProcessingResult.
    """

    def __init__(
        self,
        detector: FaceDetector | None = None,
        selector: FaceSelector | None = None,
        cropper: FaceCropper | None = None,
        coordinate_transformer: FaceCoordinateTransformer | None = None,
        aligner: FaceAligner | None = None,
        parser_service: FaceParserService | None = None,
        orchestrator: ValidationOrchestrator | None = None,
        ambiguity_validator: BaseSelectionValidator | None = None,
        exporter: PhotoExporter | None = None,
        execution_mode: ValidationExecutionMode = ValidationExecutionMode.PRODUCTION,
        parser_mode=None,
        reverse_search_service: ReverseSearchService | None = None,
        reverse_search_enabled: bool | None = None,
    ) -> None:
        """Initialise pipeline components with optional dependency injection.

        Args:
            execution_mode: Only used when `orchestrator` is not injected --
                controls whether the default ValidationOrchestrator short-circuits
                (PRODUCTION) or runs every validator unconditionally (DEVELOPMENT).
            parser_mode: ParserMode for FaceParserService. Only used when
                `parser_service` is not injected.
            reverse_search_service: Optional injected ReverseSearchService.
            reverse_search_enabled: Optional override for reverse search activation.
        """
        self._detector = detector if detector is not None else FaceDetector()
        self._selector = selector if selector is not None else FaceSelector()
        self._cropper = cropper if cropper is not None else FaceCropper()
        self._coordinate_transformer = (
            coordinate_transformer
            if coordinate_transformer is not None
            else FaceCoordinateTransformer()
        )
        self._aligner = aligner if aligner is not None else FaceAligner()
        self._ambiguity_validator = (
            ambiguity_validator if ambiguity_validator is not None else FaceAmbiguityValidator()
        )
        self._parser_service = (
            parser_service if parser_service is not None else FaceParserService(parser_mode=parser_mode)
        )
        self._orchestrator = (
            orchestrator
            if orchestrator is not None
            else ValidationOrchestrator(
                parser_service=self._parser_service,
                execution_mode=execution_mode,
            )
        )
        self._exporter = exporter if exporter is not None else PhotoExporter()

        rs_manager = ReverseSearchServiceManager()
        self._reverse_search_enabled = (
            reverse_search_enabled if reverse_search_enabled is not None else rs_manager.enabled
        )
        self._reverse_search_service = (
            reverse_search_service if reverse_search_service is not None else rs_manager.get_service()
        )

    def validate(self, image: np.ndarray) -> PhotoProcessingResult:
        """Execute the complete validation workflow on an input image.

        Args:
            image: BGR uint8 NumPy image array.

        Returns:
            A PhotoProcessingResult containing validation results and output images.

        Raises:
            ValueError, TypeError, FaceDetectionError, FaceSelectionError,
            FaceCroppingError, FaceCoordinateTransformationError,
            FaceAlignmentError, FaceParserError, etc. from underlying components.
        """
        logger.info("Starting photo validation pipeline.")

        # 1. Detect all faces
        faces = self._detector.detect(image)

        # 2. Select the best face, with confidence metadata about that choice
        selection_result = self._selector.select(faces, image.shape)

        # 3. Validate that the selection itself is reliable (e.g. reject when
        #    a second face competes strongly for "primary subject"), without
        #    re-scoring faces or duplicating selector logic.
        ambiguity_metric = self._ambiguity_validator.validate(selection_result)
        if not ambiguity_metric.passed:
            logger.info(
                "Face selection is ambiguous. Short-circuiting before crop/align/validation."
            )
            return PhotoProcessingResult(
                validation_result=ValidationResult(metrics=[ambiguity_metric]),
                selected_face=selection_result.selected_face,
                aligned_image=None,
                cropped_image=None,
                export_result=None,
            )

        selected_face = selection_result.selected_face

        # 4. Crop the selected face
        crop_result = self._cropper.crop(image, selected_face)

        # 5. Transform face coordinates into cropped image coordinate space
        transformed_face = self._coordinate_transformer.transform(
            selected_face,
            crop_result.crop_x,
            crop_result.crop_y,
        )

        # 6. Align the cropped image and face using transformed landmarks
        alignment_result = self._aligner.align(crop_result.image, transformed_face)

        # 7. Execute ValidationOrchestrator with optimized staging / lazy parsing
        if isinstance(self._orchestrator, ValidationOrchestrator):
            validation_result = self._orchestrator.validate(
                image=alignment_result.aligned_image,
                face=alignment_result.aligned_face,
                parsing_result=None,
                original_image=image,
                original_face=selected_face,
                crop_image=crop_result.image,
                crop_face=transformed_face,
            )
        else:
            parsing_result = self._parser_service.parse(alignment_result.aligned_image)
            validation_result = self._orchestrator.validate(
                image=alignment_result.aligned_image,
                face=alignment_result.aligned_face,
                parsing_result=parsing_result,
                original_image=image,
                original_face=selected_face,
                crop_image=crop_result.image,
                crop_face=transformed_face,
            )

        cropped_image = crop_result.image
        export_result = None
        if validation_result.is_valid:
            export_result = self._exporter.export(cropped_image)

        # Execute Reverse Search screening if enabled and embedding is available
        reverse_search_result: ReverseSearchResult | None = None
        if self._reverse_search_enabled and self._reverse_search_service is not None:
            try:
                embedding = getattr(selected_face, "normed_embedding", None)
                if embedding is not None:
                    reverse_search_result = self._reverse_search_service.search(embedding, k=5)
                else:
                    logger.warning("Reverse search enabled, but selected face has no normed_embedding.")
            except Exception as exc:
                logger.exception("Reverse search execution failed; continuing without search evidence.")
                reverse_search_result = None

        logger.info("Photo validation pipeline completed successfully.")
        return PhotoProcessingResult(
            validation_result=validation_result,
            selected_face=selected_face,
            aligned_image=alignment_result.aligned_image,
            cropped_image=cropped_image,
            export_result=export_result,
            reverse_search_result=reverse_search_result,
        )
