"""
Overlay renderer for producing visual debug outputs.
"""

from __future__ import annotations

import cv2
import numpy as np
from insightface.app.common import Face

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult


# Distinct BGR color palette for CelebAMask-HQ 19 classes
_CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    FacePart.BACKGROUND: (0, 0, 0),
    FacePart.SKIN: (180, 130, 70),
    FacePart.LEFT_BROW: (40, 40, 200),
    FacePart.RIGHT_BROW: (200, 40, 40),
    FacePart.LEFT_EYE: (255, 0, 0),
    FacePart.RIGHT_EYE: (0, 0, 255),
    FacePart.EYE_GLASS: (0, 255, 255),
    FacePart.LEFT_EAR: (128, 0, 128),
    FacePart.RIGHT_EAR: (128, 128, 0),
    FacePart.EAR_RING: (0, 128, 128),
    FacePart.NOSE: (0, 255, 0),
    FacePart.MOUTH: (255, 0, 255),
    FacePart.UPPER_LIP: (200, 100, 200),
    FacePart.LOWER_LIP: (100, 200, 200),
    FacePart.NECK: (90, 90, 90),
    FacePart.NECKLACE: (0, 165, 255),
    FacePart.CLOTH: (128, 128, 128),
    FacePart.HAIR: (0, 64, 128),
    FacePart.HAT: (0, 128, 255),
}


class OverlayRenderer:
    """Renders visual overlays, segmentation masks, landmarks, and pose axes."""

    @staticmethod
    def render_colored_mask(mask: np.ndarray) -> np.ndarray:
        """Render a full-color BGR image from a 2D integer segmentation mask."""
        h, w = mask.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in _CLASS_COLORS.items():
            colored[mask == class_id] = color
        return colored

    @staticmethod
    def render_transparent_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Blend a colored segmentation mask over the original BGR image."""
        if image.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        colored_mask = OverlayRenderer.render_colored_mask(mask)
        return cv2.addWeighted(image, 1.0 - alpha, colored_mask, alpha, 0)

    @staticmethod
    def render_landmarks(image: np.ndarray, face: Face | None) -> np.ndarray:
        """Draw facial keypoints (landmarks) on the image."""
        output = image.copy()
        if face is None or not hasattr(face, "kps") or face.kps is None:
            return output
        kps = face.kps
        for pt in kps:
            if np.isfinite(pt).all():
                cv2.circle(output, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
                cv2.circle(output, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), 1)
        return output

    @staticmethod
    def render_bbox(image: np.ndarray, face: Face | None) -> np.ndarray:
        """Draw face bounding box on the image."""
        output = image.copy()
        if face is None or not hasattr(face, "bbox") or face.bbox is None:
            return output
        bbox = face.bbox
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, "Face", (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return output

    @staticmethod
    def render_pose_axes(image: np.ndarray, face: Face | None) -> np.ndarray:
        """Draw head pose indicators / bounding box center on the image."""
        output = image.copy()
        if face is None or not hasattr(face, "bbox") or face.bbox is None:
            return output
        bbox = face.bbox
        x1, y1, x2, y2 = map(int, bbox[:4])
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        pose = getattr(face, "pose", (0.0, 0.0, 0.0))
        pitch, yaw, roll = pose if pose is not None else (0.0, 0.0, 0.0)
        
        text = f"Yaw:{yaw:.1f} Pitch:{pitch:.1f} Roll:{roll:.1f}"
        cv2.putText(output, text, (x1, min(y2 + 25, output.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        cv2.drawMarker(output, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        return output
