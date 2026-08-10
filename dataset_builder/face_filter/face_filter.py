"""
Face detection and filtering for dataset collection.

Reuses the production SmartIDPhotoProcessor face detection pipeline
to classify images based on face suitability for dataset collection.

This module intentionally performs only:

- face detection
- face counting
- basic geometry filtering (area ratio, pose)

It is NOT responsible for:

- blur assessment
- brightness assessment
- contrast assessment
- duplicate detection
- image quality scoring
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from dataset_builder.config.settings import Settings

# Reuse the production face detector directly.
from pipeline.detector import FaceDetector


# ------------------------------------------------------------------
# Rejection reasons
# ------------------------------------------------------------------


class RejectionReason(str, Enum):
    """Categorical reason why an image was rejected."""

    NO_FACE = "no_face"
    """No face detected in the image."""

    MULTIPLE_FACES = "multiple_faces"
    """More than one face detected."""

    FACE_TOO_SMALL = "face_too_small"
    """Largest face area ratio below threshold."""

    PROFILE_FACE = "profile_face"
    """Profile face detected when disabled."""

    LOAD_FAILED = "load_failed"
    """Image could not be loaded from disk."""


# ------------------------------------------------------------------
# Immutable dataclasses
# ------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptedImage:
    """An image that passed face filtering.

    Contains only geometry and detection metadata.  Quality metrics
    (blur, brightness, contrast) are evaluated separately by
    :class:`QualityFilter`.
    """

    path: Path
    """Absolute path to the image file."""

    face_count: int
    """Number of detected faces (always 1 for accepted images)."""

    largest_face_ratio: float
    """Area ratio of the largest face to the image area."""

    detection_score: float
    """InsightFace detection confidence for the primary face."""


@dataclass(frozen=True)
class RejectedImage:
    """An image that failed face filtering."""

    path: Path
    """Absolute path to the image file."""

    reason: RejectionReason
    """Categorical rejection reason."""

    detected_faces: int
    """Number of faces detected (may be 0)."""

    largest_face_ratio: float
    """Area ratio of the largest face, or 0.0 if none detected."""


@dataclass(frozen=True)
class FaceFilterStatistics:
    """Summary statistics from a face filtering scan.

    The ``rejection_reason_distribution`` is computed inside
    :meth:`FaceFilter.statistics` so that downstream consumers
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
# Face filter
# ------------------------------------------------------------------


class FaceFilter:
    """Classify images based on face detection and suitability.

    Reuses the production :class:`FaceDetector` from the
    SmartIDPhotoProcessor pipeline.  No files are moved or
    modified; this class only classifies.

    Responsibilities:

    - Detect faces using the production InsightFace pipeline.
    - Count faces per image.
    - Compute face area ratio relative to image area.
    - Reject images with zero, multiple, too-small, or profile faces.

    This filter is NOT responsible for blur, brightness, contrast,
    duplicate detection, or any other quality assessment.

    Parameters
    ----------
    settings:
        Application settings controlling face filtering thresholds.

    Examples
    --------
    ::

        ff = FaceFilter(settings)
        ff.scan(Path("dataset/raw"))
        stats = ff.statistics()
    """

    def __init__(self, settings: Settings) -> None:
        self._settings: Settings = settings
        self._detector: FaceDetector = FaceDetector()
        self._accepted: list[AcceptedImage] = []
        self._rejected: list[RejectedImage] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def accepted(self) -> list[AcceptedImage]:
        """Return all images that passed filtering.

        Returns
        -------
        list[AcceptedImage]
            A copy of the internal accepted list.
        """
        return list(self._accepted)

    @property
    def rejected(self) -> list[RejectedImage]:
        """Return all images that failed filtering.

        Returns
        -------
        list[RejectedImage]
            A copy of the internal rejected list.
        """
        return list(self._rejected)

    def scan(
        self,
        directory: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Scan a directory and classify every image.

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
            if isinstance(result, AcceptedImage):
                self._accepted.append(result)
            else:
                self._rejected.append(result)

            if progress_callback is not None:
                progress_callback(idx + 1, total)

    def filter_image(self, path: Path) -> AcceptedImage | RejectedImage:
        """Classify a single image based on face detection.

        Parameters
        ----------
        path:
            Path to the image file.

        Returns
        -------
        AcceptedImage or RejectedImage
            Classification result.
        """
        image = self._load_image(path)
        if image is None:
            return RejectedImage(
                path=path,
                reason=RejectionReason.LOAD_FAILED,
                detected_faces=0,
                largest_face_ratio=0.0,
            )

        try:
            faces = self._detector.detect(image)
        except Exception:
            faces = []

        img_h, img_w = image.shape[:2]
        img_area = img_h * img_w

        if len(faces) == 0:
            return RejectedImage(
                path=path,
                reason=RejectionReason.NO_FACE,
                detected_faces=0,
                largest_face_ratio=0.0,
            )

        largest_face = self._find_largest_face(faces)
        largest_ratio = self._compute_face_ratio(largest_face, img_area)

        if len(faces) > self._settings.MAX_FACES_PER_IMAGE:
            return RejectedImage(
                path=path,
                reason=RejectionReason.MULTIPLE_FACES,
                detected_faces=len(faces),
                largest_face_ratio=largest_ratio,
            )

        if largest_ratio < self._settings.MIN_FACE_AREA_RATIO:
            return RejectedImage(
                path=path,
                reason=RejectionReason.FACE_TOO_SMALL,
                detected_faces=1,
                largest_face_ratio=largest_ratio,
            )

        if not self._settings.ALLOW_PROFILE_FACES:
            if self._is_profile_face(largest_face):
                return RejectedImage(
                    path=path,
                    reason=RejectionReason.PROFILE_FACE,
                    detected_faces=1,
                    largest_face_ratio=largest_ratio,
                )

        return AcceptedImage(
            path=path,
            face_count=1,
            largest_face_ratio=largest_ratio,
            detection_score=float(largest_face.det_score),
        )

    def statistics(self) -> FaceFilterStatistics:
        """Return summary statistics for the last scan.

        The ``rejection_reason_distribution`` is computed here so
        that downstream consumers never need to iterate the rejected
        list themselves.

        Returns
        -------
        FaceFilterStatistics
            Snapshot of filtering results.
        """
        total = len(self._accepted) + len(self._rejected)
        accepted = len(self._accepted)
        ratio = accepted / total if total > 0 else 0.0

        distribution: dict[str, int] = {}
        for item in self._rejected:
            key = item.reason.value
            distribution[key] = distribution.get(key, 0) + 1

        return FaceFilterStatistics(
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

    @staticmethod
    def _find_largest_face(faces: list[object]) -> object:
        """Return the face with the largest bounding box area.

        Parameters
        ----------
        faces:
            Non-empty list of detected face objects.

        Returns
        -------
        object
            The face with the maximum bbox area.
        """

        def _bbox_area(face: object) -> float:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                return 0.0
            try:
                x1, y1, x2, y2 = bbox[:4]
                return max(0.0, x2 - x1) * max(0.0, y2 - y1)
            except (TypeError, IndexError):
                return 0.0

        return max(faces, key=_bbox_area)

    @staticmethod
    def _compute_face_ratio(face: object, img_area: int) -> float:
        """Compute the ratio of face bounding box area to image area.

        Parameters
        ----------
        face:
            Detected face with a ``bbox`` attribute.
        img_area:
            Total image area in pixels.

        Returns
        -------
        float
            Area ratio in the range [0.0, 1.0].
        """
        if img_area <= 0:
            return 0.0

        bbox = getattr(face, "bbox", None)
        if bbox is None:
            return 0.0

        try:
            x1, y1, x2, y2 = bbox[:4]
            face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            return face_area / img_area
        except (TypeError, IndexError):
            return 0.0

    def _is_profile_face(self, face: object) -> bool:
        """Check whether a face is a profile (non-frontal) pose.

        Uses the yaw component of the head pose.  A yaw exceeding
        ``Settings.MAX_PROFILE_YAW_DEGREES`` is considered a profile
        face.
        """
        pose = getattr(face, "pose", None)
        if pose is None:
            return False

        try:
            _pitch, yaw, _roll = pose
            return abs(float(yaw)) > self._settings.MAX_PROFILE_YAW_DEGREES
        except (TypeError, ValueError):
            return False
