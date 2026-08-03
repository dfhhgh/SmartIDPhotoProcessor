"""Export result domain model."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np


class ExportQuality(Enum):
    """Quality levels for the exported image."""

    SAFE = auto()
    WARNING = auto()


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Represents the output of the deterministic export stage."""

    exported_image: np.ndarray
    original_size: tuple[int, int]
    exported_size: tuple[int, int]
    content_size: tuple[int, int]
    offset_x: int
    offset_y: int
    padding_color: tuple[int, int, int]
    interpolation_used: int
    was_upscaled: bool
    upscale_factor: float
    quality: ExportQuality
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate export result fields."""
        if self.exported_image is None or not isinstance(self.exported_image, np.ndarray):
            raise TypeError("Exported image must be a numpy array.")
        if self.exported_image.size == 0:
            raise ValueError("Exported image must not be empty.")
        if self.exported_image.ndim != 3:
            raise ValueError("Exported image must be a 3-dimensional image.")
        if not isinstance(self.quality, ExportQuality):
            raise TypeError("Quality must be an ExportQuality instance.")
        if not isinstance(self.padding_color, tuple) or len(self.padding_color) != 3:
            raise TypeError("Padding color must be a 3-item tuple.")
        if not all(isinstance(value, int) for value in self.padding_color):
            raise TypeError("Padding color values must be integers.")
        if not all(0 <= value <= 255 for value in self.padding_color):
            raise ValueError("Padding color values must be between 0 and 255.")
        if not isinstance(self.warnings, tuple):
            raise TypeError("Warnings must be a tuple.")

    @property
    def export_quality(self) -> str:
        """Return a string-compatible quality value for the legacy exporter API."""
        return self.quality.name.lower()

    @property
    def export_warnings(self) -> list[str]:
        """Return the warnings as a mutable list for compatibility with tests."""
        return list(self.warnings)
