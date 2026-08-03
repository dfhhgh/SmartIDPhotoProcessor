"""Deterministic export stage for final ID-photo generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from config.constants import (
    EXPORT_INTERPOLATION_METHOD,
    MAX_ALLOWED_UPSCALE,
    MAX_EXPORT_HEIGHT,
    MAX_EXPORT_WIDTH,
    MIN_EXPORT_HEIGHT,
    MIN_EXPORT_WIDTH,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SAFE_UPSCALE_FACTOR,
)
from models.export_result import ExportQuality, ExportResult

logger = logging.getLogger(__name__)


class PhotoExporter:
    """Produce a deterministic, print-ready ID photo from an already validated crop.

    This stage does not detect faces, crop faces, align faces, or validate images.
    It only resizes an existing validated crop to a fixed target size while preserving
    aspect ratio and avoiding distortion.
    """

    def __init__(self, settings: dict[str, Any] | None = None) -> None:
        self._settings = self._merge_settings(settings)

    def export(
        self,
        cropped_image: np.ndarray | None,
        settings: dict[str, Any] | None = None,
    ) -> ExportResult:
        """Resize a validated crop to a fixed ID-photo size safely."""
        if settings is not None:
            self._settings = self._merge_settings(settings)

        if cropped_image is None:
            raise ValueError("Cropped image must be a non-empty numpy array.")

        if not isinstance(cropped_image, np.ndarray):
            raise TypeError("Cropped image must be a numpy array.")

        if cropped_image.size == 0:
            raise ValueError("Cropped image must be a non-empty numpy array.")

        if cropped_image.ndim != 3 or cropped_image.shape[2] != 3:
            raise ValueError("Cropped image must be a 3-channel BGR image.")

        original_height, original_width = cropped_image.shape[:2]
        target_width = int(self._settings["target_width"])
        target_height = int(self._settings["target_height"])
        self._validate_target_size(target_width, target_height)
        interpolation = self._settings["interpolation_method"]

        content_size = self._compute_content_size(original_height, original_width, target_width, target_height)
        resize_factor = self._compute_upscale_factor(original_width, original_height, content_size[0], content_size[1])
        interpolation_method = self._select_interpolation(resize_factor, interpolation)
        resized = self._resize_image(cropped_image, content_size, interpolation_method)

        export_warnings: list[str] = []
        upscale_factor = resize_factor
        was_upscaled = upscale_factor > 1.0
        quality = ExportQuality.SAFE

        if was_upscaled:
            if upscale_factor > self._settings["safe_upscale_factor"]:
                quality = ExportQuality.WARNING
                export_warnings.append(
                    f"excessive upscaling detected: factor {upscale_factor:.2f} exceeds safe threshold {self._settings['safe_upscale_factor']:.2f}."
                )

            if self._settings["reject_on_excessive_upscale"] and upscale_factor > self._settings["max_allowed_upscale"]:
                raise ValueError(
                    f"Export image upscale factor {upscale_factor:.2f} exceeds the maximum allowed upscale {self._settings['max_allowed_upscale']:.2f}."
                )

        export_image = self._compose_export_image(
            resized,
            target_width,
            target_height,
            self._settings["background_color"],
        )

        return ExportResult(
            exported_image=export_image,
            original_size=(original_width, original_height),
            exported_size=(target_width, target_height),
            content_size=content_size,
            offset_x=max(0, (target_width - content_size[0]) // 2),
            offset_y=max(0, (target_height - content_size[1]) // 2),
            padding_color=self._settings["background_color"],
            interpolation_used=interpolation_method,
            was_upscaled=was_upscaled,
            upscale_factor=upscale_factor,
            quality=quality,
            warnings=tuple(export_warnings),
        )

    def _merge_settings(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        defaults = {
            "target_width": OUTPUT_WIDTH,
            "target_height": OUTPUT_HEIGHT,
            "safe_upscale_factor": SAFE_UPSCALE_FACTOR,
            "max_allowed_upscale": MAX_ALLOWED_UPSCALE,
            "reject_on_excessive_upscale": False,
            "interpolation_method": EXPORT_INTERPOLATION_METHOD,
            "background_color": (255, 255, 255),
        }
        if settings is None:
            return defaults
        merged = defaults.copy()
        merged.update(settings)
        return merged

    def _compute_content_size(
        self,
        original_height: int,
        original_width: int,
        target_width: int,
        target_height: int,
    ) -> tuple[int, int]:
        scale = min(target_width / original_width, target_height / original_height)
        content_width = max(1, int(round(original_width * scale)))
        content_height = max(1, int(round(original_height * scale)))
        return (content_width, content_height)

    def _resize_image(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        interpolation: int,
    ) -> np.ndarray:
        return cv2.resize(image, size, interpolation=interpolation)

    def _validate_target_size(self, target_width: int, target_height: int) -> None:
        if target_width < MIN_EXPORT_WIDTH or target_width > MAX_EXPORT_WIDTH:
            raise ValueError(
                f"Target width {target_width} must be between {MIN_EXPORT_WIDTH} and {MAX_EXPORT_WIDTH}."
            )
        if target_height < MIN_EXPORT_HEIGHT or target_height > MAX_EXPORT_HEIGHT:
            raise ValueError(
                f"Target height {target_height} must be between {MIN_EXPORT_HEIGHT} and {MAX_EXPORT_HEIGHT}."
            )

    def _select_interpolation(self, upscale_factor: float, fallback: int) -> int:
        if upscale_factor > 1.0:
            return cv2.INTER_CUBIC
        return fallback if isinstance(fallback, int) else cv2.INTER_AREA

    def _compose_export_image(
        self,
        content_image: np.ndarray,
        target_width: int,
        target_height: int,
        background_color: tuple[int, int, int],
    ) -> np.ndarray:
        if content_image.shape[1] == target_width and content_image.shape[0] == target_height:
            return content_image

        canvas = np.full((target_height, target_width, 3), background_color, dtype=np.uint8)
        offset_x = max(0, (target_width - content_image.shape[1]) // 2)
        offset_y = max(0, (target_height - content_image.shape[0]) // 2)
        canvas[offset_y : offset_y + content_image.shape[0], offset_x : offset_x + content_image.shape[1]] = content_image
        return canvas

    def _compute_upscale_factor(
        self,
        original_width: int,
        original_height: int,
        content_width: int,
        content_height: int,
    ) -> float:
        if content_width <= 0 or content_height <= 0:
            return 1.0
        scale_x = content_width / original_width
        scale_y = content_height / original_height
        return max(scale_x, scale_y) if max(scale_x, scale_y) > 1.0 else 1.0
