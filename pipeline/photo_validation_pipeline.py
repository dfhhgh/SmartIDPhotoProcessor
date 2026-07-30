"""
Photo validation pipeline orchestration module.

Connects the detection, selection, cropping, coordinate transformation,
alignment, parsing, and validation components into a single sequential workflow.
"""

from __future__ import annotations

import logging

import numpy as np

from models.photo_processing_result import PhotoProcessingResult
from pipeline.aligner import FaceAligner
from pipeline.face_coordinate_transformer import FaceCoordinateTransformer
from pipeline.cropper import FaceCropper
from pipeline.detector import FaceDetector
from pipeline.selector import FaceSelector
from pipeline.validation_orchestrator import ValidationOrchestrator
from services.face_parser_service import FaceParserService

logger = logging.getLogger(__name__)


class PhotoValidationPipeline:
    """Orchestrates the complete photo validation pipeline.

    Execution steps:
    1. Detect all faces in the image.
    2. Select the primary face.
    3. Crop the selected face (producing CropResult with image and crop offset).
    4. Transform face coordinates using FaceCoordinateTransformer.
    5. Align the cropped image and face into aligned coordinate space.
    6. Run face parsing on the aligned face.
    7. Execute validation orchestrator on the aligned image, aligned face, and parsing result.
    8. Return PhotoProcessingResult (containing crop_result.image if validation succeeds, else None).
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
    ) -> None:
        """Initialise pipeline components with optional dependency injection."""
        self._detector = detector if detector is not None else FaceDetector()
        self._selector = selector if selector is not None else FaceSelector()
        self._cropper = cropper if cropper is not None else FaceCropper()
        self._coordinate_transformer = (
            coordinate_transformer
            if coordinate_transformer is not None
            else FaceCoordinateTransformer()
        )
        self._aligner = aligner if aligner is not None else FaceAligner()
        self._parser_service = (
            parser_service if parser_service is not None else FaceParserService()
        )
        self._orchestrator = (
            orchestrator if orchestrator is not None else ValidationOrchestrator()
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

        # 2. Select the best face
        selected_face = self._selector.select(faces, image.shape)

        # 3. Crop the selected face
        crop_result = self._cropper.crop(image, selected_face)

        # 4. Transform face coordinates into cropped image coordinate space
        transformed_face = self._coordinate_transformer.transform(
            selected_face,
            crop_result.crop_x,
            crop_result.crop_y,
        )

        # 5. Align the cropped image and face using transformed landmarks
        alignment_result = self._aligner.align(crop_result.image, transformed_face)

        # 6. Run FaceParserService on the aligned face
        parsing_result = self._parser_service.parse(alignment_result.aligned_image)

        # 7. Execute ValidationOrchestrator
        validation_result = self._orchestrator.validate(
            image=alignment_result.aligned_image,
            face=alignment_result.aligned_face,
            parsing_result=parsing_result,
        )

        # 8. Set cropped image based on validation success
        cropped_image = crop_result.image if validation_result.is_valid else None

        logger.info("Photo validation pipeline completed successfully.")
        return PhotoProcessingResult(
            validation_result=validation_result,
            selected_face=selected_face,
            aligned_image=alignment_result.aligned_image,
            cropped_image=cropped_image,
        )
