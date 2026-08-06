"""
Visualization orchestrator for saving debug and evaluation images.
"""

from __future__ import annotations

import os
import cv2
import numpy as np
from insightface.app.common import Face

from evaluation.overlay_renderer import OverlayRenderer
from models.parsing.face_parsing_result import FaceParsingResult


class Visualization:
    """Orchestrates saving all required visual debug artifacts for evaluated images."""

    def __init__(self, output_dir: str) -> None:
        self._output_dir = output_dir
        self._masks_dir = os.path.join(output_dir, "masks")
        self._overlays_dir = os.path.join(output_dir, "overlays")
        os.makedirs(self._masks_dir, exist_ok=True)
        os.makedirs(self._overlays_dir, exist_ok=True)

    def save_visualizations(
        self,
        image_name: str,
        image: np.ndarray,
        parsing_result: FaceParsingResult | None,
        face: Face | None,
    ) -> dict[str, str]:
        """Save all visual output artifacts and return their relative paths."""
        base_name = os.path.splitext(image_name)[0]
        paths: dict[str, str] = {}

        # 1. Original Image
        orig_path = os.path.join(self._overlays_dir, f"{base_name}_original.jpg")
        cv2.imwrite(orig_path, image)
        paths["original"] = orig_path

        if parsing_result is not None:
            # 2. Colored segmentation mask
            colored_mask = OverlayRenderer.render_colored_mask(parsing_result.mask)
            mask_path = os.path.join(self._masks_dir, f"{base_name}_mask.png")
            cv2.imwrite(mask_path, colored_mask)
            paths["colored_mask"] = mask_path

            # 3. Transparent overlay
            overlay = OverlayRenderer.render_transparent_overlay(image, parsing_result.mask)
            overlay_path = os.path.join(self._overlays_dir, f"{base_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay)
            paths["transparent_overlay"] = overlay_path

        # 4. Landmarks, Bbox, Pose
        annotated = OverlayRenderer.render_bbox(image, face)
        annotated = OverlayRenderer.render_landmarks(annotated, face)
        annotated = OverlayRenderer.render_pose_axes(annotated, face)
        annotated_path = os.path.join(self._overlays_dir, f"{base_name}_annotated.jpg")
        cv2.imwrite(annotated_path, annotated)
        paths["annotated"] = annotated_path

        return paths
