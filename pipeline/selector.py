"""
Face selection module.

This module is responsible for selecting the primary face
from multiple detected faces, and exposing how confident that
selection is via SelectionResult.
"""

from __future__ import annotations

import logging
import math
from insightface.app.common import Face

from exceptions.face_exceptions import FaceSelectionError
from models.ranked_face import RankedFace
from models.selection_result import SelectionResult

logger = logging.getLogger(__name__)


class FaceSelector:
    """
    Select the primary face from multiple detected faces.

    The selection is based on a weighted scoring algorithm
    that considers:

    - Face size
    - Distance from image center
    - Detection confidence
    """

    def select(
        self,
        faces: list[Face],
        image_shape: tuple[int, ...],
    ) -> SelectionResult:
        """
        Select the most suitable face from the detected faces.

        Args:
            faces:
                List of detected faces.

            image_shape:
                Shape of the original image.

        Returns:
            A SelectionResult containing the selected primary face along
            with confidence metadata (score margin, ambiguity ratio, and
            candidate count) describing how reliable that selection is.

        Raises:
            FaceSelectionError:
                If no valid face can be selected.
        """
        if not faces:
            logger.error(
                "No faces were provided for selection."
            )

            raise FaceSelectionError(
                "No faces were provided for selection."
            )

        scored_faces: list[tuple[Face, float]] = []
        for face in faces:
            area_score = self._calculate_area_score(
                face,
                image_shape,
            )

            center_score = self._calculate_center_score(
                face,
                image_shape,
            )

            confidence_score = self._calculate_confidence_score(
                face,
            )
            final_score = self._calculate_final_score(
                                area_score,
                                center_score,
                                confidence_score,
                            )
            scored_faces.append((face, final_score))

        if not scored_faces:
            logger.error(
                "Failed to select a primary face."
            )

            raise FaceSelectionError(
                "Failed to select a primary face."
            )

        return self._build_selection_result(scored_faces)

    def _build_selection_result(
        self,
        scored_faces: list[tuple[Face, float]],
    ) -> SelectionResult:
        """
        Rank scored candidates and build the resulting SelectionResult.

        Args:
            scored_faces:
                Every detected face paired with its final weighted score.

        Returns:
            A SelectionResult describing the winning face and how
            confidently it beat the runner-up, if any.
        """
        ranked_faces = tuple(
            RankedFace(face=face, score=score)
            for face, score in sorted(
                scored_faces,
                key=lambda scored: scored[1],
                reverse=True,
            )
        )

        selected_face, selected_score = ranked_faces[0].face, ranked_faces[0].score

        second_best_score: float | None = (
            ranked_faces[1].score if len(ranked_faces) > 1 else None
        )

        if second_best_score is None:
            score_margin = selected_score
            ambiguity_ratio = 0.0
        else:
            score_margin = selected_score - second_best_score
            ambiguity_ratio = (
                second_best_score / selected_score if selected_score > 0 else 0.0
            )

        return SelectionResult(
            selected_face=selected_face,
            selected_score=selected_score,
            second_best_score=second_best_score,
            score_margin=score_margin,
            ambiguity_ratio=ambiguity_ratio,
            detected_faces_count=len(ranked_faces),
            ranked_faces=ranked_faces,
        )

    def _get_valid_bbox(
        self,
        face: Face,
        ) -> tuple[float, float, float, float]:
        """
        Validate and return the face bounding box coordinates.

        Args:
            face:
                Detected face.

        Returns:
            A tuple containing:
            (x1, y1, x2, y2)

        Raises:
            FaceSelectionError:
                If the bounding box is invalid.
        """

        bbox = face.bbox

        if bbox is None:
            logger.error("Face bounding box is missing.")

            raise FaceSelectionError(
                "Face bounding box is missing."
            )

        if len(bbox) != 4:
            logger.error(
                "Face bounding box must contain exactly four coordinates."
            )

            raise FaceSelectionError(
                "Face bounding box must contain exactly four coordinates."
            )

        x1, y1, x2, y2 = bbox

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            logger.error(
                "Face bounding box has invalid dimensions."
            )

            raise FaceSelectionError(
                "Face bounding box has invalid dimensions."
            )

        return x1, y1, x2, y2

    def _calculate_area_score(
        self,
        face: Face,
        image_shape: tuple[int, ...],
        ) -> float:
        """
        Calculate the normalized face area score.
        """

        x1, y1, x2, y2 = self._get_valid_bbox(face)
        width = x2 - x1
        height = y2 - y1

        
        face_area = width * height
        image_height, image_width = image_shape[:2]

        if image_height <= 0 or image_width <= 0:
            logger.error(
                "Image dimensions must be positive."
            )
            raise FaceSelectionError(
                "Image dimensions must be positive."
            )

        image_area = image_width * image_height
        area_score = face_area / image_area

        return max(0.0, min(area_score, 1.0))


    def _calculate_center_score(
        self,
        face: Face,
        image_shape: tuple[int, ...],
    ) -> float:
        """
        Calculate the score based on the distance from
        the image center.
        """
        x1, y1, x2, y2 = self._get_valid_bbox(face)

        image_height, image_width = image_shape[:2]

        if image_height <= 0 or image_width <= 0:
            logger.error(
            "Image dimensions must be positive."
            )

            raise FaceSelectionError(
            "Image dimensions must be positive."
            )
        face_center_x = (x1 + x2) / 2
        face_center_y = (y1 + y2) / 2

        image_center_x = image_width / 2
        image_center_y = image_height / 2
        distance = math.hypot(
        face_center_x - image_center_x,
        face_center_y - image_center_y,
        )

        max_distance = math.hypot(
            image_center_x,
            image_center_y,
        )

        if max_distance == 0:
            logger.error(
                "Maximum image center distance is zero."
            )

            raise FaceSelectionError(
                "Maximum image center distance is zero."
            )

        normalized_distance = distance / max_distance

        center_score = 1.0 - normalized_distance

        return max(0.0, min(center_score, 1.0))



    def _calculate_confidence_score(
        self,
        face: Face,
    ) -> float:
        """
        Calculate the detection confidence score.
        """
        confidence = face.det_score
        if confidence is None:
            logger.error(
                "Face detection confidence is missing."
            )

            raise FaceSelectionError(
                "Face detection confidence is missing."
            )
        if not 0.0 <= confidence <= 1.0:
            logger.error(
                "Face detection confidence must be between 0.0 and 1.0."
            )

            raise FaceSelectionError(
                "Face detection confidence must be between 0.0 and 1.0."
            )
        return confidence


    def _calculate_final_score(
        self,
        area_score: float,
        center_score: float,
        confidence_score: float,
    ) -> float:
        """
        Combine all scores into a final weighted score.
        """
        final_score = (
                    0.5 * area_score
                    + 0.3 * center_score
                    + 0.2 * confidence_score
                )
        if not 0.0 <= final_score <= 1.0:
            logger.error(
                "Final face selection score must be between 0.0 and 1.0."
            )

            raise FaceSelectionError(
                "Final face selection score must be between 0.0 and 1.0."
            ) 
        return final_score
