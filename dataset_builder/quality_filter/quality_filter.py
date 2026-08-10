"""
Image quality filtering for dataset collection.

Reuses the production SmartIDPhotoProcessor validators (BlurValidator,
BrightnessValidator, ContrastValidator) to classify images based on
technical quality.  No quality algorithms are duplicated; all
thresholds and formulas come from production code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from dataset_builder.config.settings import Settings

# Reuse production validators directly.
from validators.blur_validator import BlurValidator
from validators.brightness_validator import BrightnessValidator
from validators.contrast_validator import ContrastValidator

from models.validation_type import ValidationType


# ------------------------------------------------------------------
# Rejection reasons
# ------------------------------------------------------------------


class QualityRejectionReason(str, Enum):
    """Categorical reason why an image was rejected by quality filtering."""

    LOAD_FAILED = "load_failed"
    """Image could not be loaded from disk."""

    BLUR = "blur"
    """Image sharpness below threshold."""

    BRIGHTNESS = "brightness"
    """Image brightness outside acceptable range."""

    CONTRAST = "contrast"
    """Image contrast below threshold."""

    MULTIPLE_FAILURES = "multiple_failures"
    """Image failed more than one quality check."""


# ------------------------------------------------------------------
# Immutable dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptedQualityImage:
    """An image that passed quality filtering.

    Contains scores from all three production validators.
    """

    path: Path
    """Absolute path to the image file."""

    blur_score: float
    """Normalized blur score from BlurValidator (0.0 - 1.0)."""

    brightness_score: float
    """Normalized brightness score from BrightnessValidator (0.0 - 1.0)."""

    contrast_score: float
    """Normalized contrast score from ContrastValidator (0.0 - 1.0)."""


@dataclass(frozen=True)
class RejectedQualityImage:
    """An image that failed quality filtering."""

    path: Path
    """Absolute path to the image file."""

    reason: QualityRejectionReason
    """Categorical rejection reason."""

    blur_score: float
    """Normalized blur score from BlurValidator (0.0 - 1.0)."""

    brightness_score: float
    """Normalized brightness score from BrightnessValidator (0.0 - 1.0)."""

    contrast_score: float
    """Normalized contrast score from ContrastValidator (0.0 - 1.0)."""


@dataclass(frozen=True)
class QualityFilterStatistics:
    """Summary statistics from a quality filtering scan.

    The ``rejection_reason_distribution`` is computed inside
    :meth:`QualityFilter.statistics` so that downstream consumers
    (e.g. :class:`StatisticsAggregator`) can read it directly
    without recomputation.
    """

    total_images: int
    """Total number of images processed."""

    accepted_images: int
    """Number of images that passed filtering."""

    rejected_images: int
    """Number of images that failed filtering."""

    acceptance_ratio: float
    """Fraction of images accepted (0.0 - 1.0)."""

    rejection_reason_distribution: dict[str, int]
    """Mapping of rejection reason value to its occurrence count."""


# ------------------------------------------------------------------
# Type alias for progress callback
# ------------------------------------------------------------------

ProgressCallback = Callable[[int, int], None]
"""Signature: ``(processed: int, total: int) -> None``."""


# ------------------------------------------------------------------
# Quality filter
# ------------------------------------------------------------------


class QualityFilter:
    """Classify images based on technical quality metrics.

    Reuses the production :class:`BlurValidator`,
    :class:`BrightnessValidator`, and :class:`ContrastValidator`
    from the SmartIDPhotoProcessor pipeline.  No quality algorithms
    are duplicated; all thresholds come from production code.

    Responsibilities:

    - Assess image blur using Laplacian variance.
    - Assess image brightness using mean grayscale intensity.
    - Assess image contrast using standard deviation of intensity.
    - Classify images as accepted or rejected based on validator results.

    This filter is NOT responsible for:

    - face detection
    - face counting
    - duplicate detection
    - metadata management

    Parameters
    ----------
    settings:
        Application settings (currently unused; thresholds come from
        production validators).

    Examples
    --------
    ::

        qf = QualityFilter(settings)
        qf.scan(Path("dataset/raw"))
        stats = qf.statistics()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._blur_validator = BlurValidator()
        self._brightness_validator = BrightnessValidator()
        self._contrast_validator = ContrastValidator()
        self._accepted: list[AcceptedQualityImage] = []
        self._rejected: list[RejectedQualityImage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def accepted(self) -> list[AcceptedQualityImage]:
        """Return all images that passed quality filtering.

        Returns
        -------
        list[AcceptedQualityImage]
            A copy of the internal accepted list.
        """
        return list(self._accepted)

    @property
    def rejected(self) -> list[RejectedQualityImage]:
        """Return all images that failed quality filtering.

        Returns
        -------
        list[RejectedQualityImage]
            A copy of the internal rejected list.
        """
        return list(self._rejected)

    def scan(
        self,
        directory: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Scan a directory and classify every image by quality.

        Previous results are cleared before scanning.

        Parameters
        ----------
        directory:
            Root directory to scan recursively.
        progress_callback:
            Optional function called with ``(processed, total)``
            after each image is processed.

        Raises
        ------
        FileNotFoundError
            When *directory* does not exist.
        """
        self.clear()

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        files = sorted(
            p for p in directory.rglob("*")
            if p.is_file() and p.suffix.lower() in self._settings.SUPPORTED_IMAGE_EXTENSIONS
        )

        total = len(files)
        for idx, file_path in enumerate(files):
            result = self.filter_image(file_path)
            if isinstance(result, AcceptedQualityImage):
                self._accepted.append(result)
            else:
                self._rejected.append(result)

            if progress_callback is not None:
                progress_callback(idx + 1, total)

    def filter_image(self, path: Path) -> AcceptedQualityImage | RejectedQualityImage:
        """Classify a single image based on quality metrics.

        Runs all three production validators and collects their
        scores.  If every validator passes, the image is accepted.

        Parameters
        ----------
        path:
            Path to the image file.

        Returns
        -------
        AcceptedQualityImage or RejectedQualityImage
            Classification result.
        """
        image = self._load_image(path)
        if image is None:
            return RejectedQualityImage(
                path=path,
                reason=QualityRejectionReason.LOAD_FAILED,
                blur_score=0.0,
                brightness_score=0.0,
                contrast_score=0.0,
            )

        blur_metric = self._blur_validator.validate(image)
        brightness_metric = self._brightness_validator.validate(image)
        contrast_metric = self._contrast_validator.validate(image)

        blur_score = blur_metric.score
        brightness_score = brightness_metric.score
        contrast_score = contrast_metric.score

        failed: list[QualityRejectionReason] = []

        if not blur_metric.passed:
            failed.append(QualityRejectionReason.BLUR)

        if not brightness_metric.passed:
            failed.append(QualityRejectionReason.BRIGHTNESS)

        if not contrast_metric.passed:
            failed.append(QualityRejectionReason.CONTRAST)

        if not failed:
            return AcceptedQualityImage(
                path=path,
                blur_score=blur_score,
                brightness_score=brightness_score,
                contrast_score=contrast_score,
            )

        if len(failed) > 1:
            reason = QualityRejectionReason.MULTIPLE_FAILURES
        else:
            reason = failed[0]

        return RejectedQualityImage(
            path=path,
            reason=reason,
            blur_score=blur_score,
            brightness_score=brightness_score,
            contrast_score=contrast_score,
        )

    def statistics(self) -> QualityFilterStatistics:
        """Return summary statistics for the last scan.

        The ``rejection_reason_distribution`` is computed here so
        that downstream consumers never need to iterate the rejected
        list themselves.

        Returns
        -------
        QualityFilterStatistics
            Snapshot of filtering results.
        """
        total = len(self._accepted) + len(self._rejected)
        accepted = len(self._accepted)
        ratio = accepted / total if total > 0 else 0.0

        distribution: dict[str, int] = {}
        for item in self._rejected:
            key = item.reason.value
            distribution[key] = distribution.get(key, 0) + 1

        return QualityFilterStatistics(
            total_images=total,
            accepted_images=accepted,
            rejected_images=len(self._rejected),
            acceptance_ratio=ratio,
            rejection_reason_distribution=distribution,
        )

    def clear(self) -> None:
        """Clear all internal caches from the last scan."""
        self._accepted.clear()
        self._rejected.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, path: Path) -> np.ndarray | None:
        """Load an image from disk as an OpenCV BGR array.

        Returns ``None`` when the image cannot be read.
        """
        try:
            image = cv2.imread(str(path))
            return image
        except Exception:
            return None
