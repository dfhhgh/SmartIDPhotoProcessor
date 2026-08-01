"""
Glasses-detector classifier service.

Adapter that bridges the external ``glasses-detector`` library into the
project domain by implementing :class:`EyewearClassifier`.  The library
provides independent binary classifiers for eyeglasses and sunglasses;
this service combines their outputs into a single
:class:`EyewearPrediction`.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np
try:
    from glasses_detector import GlassesClassifier
except ImportError:
    GlassesClassifier = None  # type: ignore[assignment, misc]
from insightface.app.common import Face

from config.constants import (
    GLASSES_EYEGLASSES_PROBABILITY_THRESHOLD,
    GLASSES_SUNGLASSES_PROBABILITY_THRESHOLD,
)
from config.settings import Settings
from models.eyewear_prediction import EyewearPrediction
from models.eyewear_type import EyewearType
from services.eyewear_classifier import EyewearClassifier

logger = logging.getLogger(__name__)


class GlassesDetectorError(Exception):
    """Raised when glasses-detector inference fails."""


class GlassesDetectorClassifier(EyewearClassifier):
    """Inference adapter wrapping the ``glasses-detector`` library.

    Two independent binary classifiers are maintained internally:

    * **eyeglasses** – detects transparent prescription frames.
    * **sunglasses** – detects opaque / tinted lenses.

    The sunglasses classifier is consulted first; if it fires the
    eyewear is classified as :attr:`EyewearType.SUNGLASSES`.  Otherwise
    the eyeglasses classifier decides between
    :attr:`EyewearType.CLEAR_GLASSES` and :attr:`EyewearType.NONE`.

    The two underlying models are loaded lazily on the first call to
    :meth:`classify` and cached for the lifetime of this instance.

    Unlike :class:`FaceService` and :class:`FaceParserService`, this
    class is intentionally NOT a process-wide singleton. Its single
    shared instance is already guaranteed by dependency injection: the
    composition root constructs one :class:`GlassesDetectorClassifier`
    and passes it into :class:`GlassesValidator`'s constructor, so a
    ``__new__``-based singleton would only duplicate a guarantee the
    wiring already provides. A lock still guards the lazy-load step,
    since the one shared, injected instance may be invoked concurrently
    from multiple threads.
    """

    def __init__(self) -> None:
        """Initialise the adapter with project hardware configuration.

        Model loading is deferred until the first call to
        :meth:`classify`; see :meth:`_ensure_loaded`.
        """
        settings = Settings()
        self._device = self._resolve_device(
            use_gpu=settings.USE_GPU,
            gpu_id=settings.GPU_ID,
        )

        self._eyeglasses_classifier: GlassesClassifier | None = None
        self._sunglasses_classifier: GlassesClassifier | None = None
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Device resolution
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_device(
        use_gpu: bool,
        gpu_id: int,
    ) -> str:
        """Return the ``glasses-detector`` device string for the current config.

        Mirrors the CPU/GPU decision already made by :class:`FaceService`
        and :class:`FaceParserService`, so this classifier automatically
        follows the project's hardware settings instead of a hardcoded
        device.

        Args:
            use_gpu: Whether GPU execution is enabled project-wide.
            gpu_id: Index of the GPU device to use when enabled.

        Returns:
            ``"cuda:<gpu_id>"`` when GPU execution is enabled, otherwise
            ``"cpu"``.
        """
        if use_gpu:
            return f"cuda:{gpu_id}"

        return "cpu"

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def _load_classifiers(self) -> None:
        """Create the two binary classifiers (once)."""
        logger.info(
            "Loading glasses-detector classifiers (device=%s) …",
            self._device,
        )

        try:
            self._eyeglasses_classifier = GlassesClassifier(
                kind="eyeglasses",
                size="small",
                device=self._device,
            )
            self._sunglasses_classifier = GlassesClassifier(
                kind="sunglasses",
                size="small",
                device=self._device,
            )
            logger.info("Glasses-detector classifiers loaded successfully.")
        except Exception as exc:
            logger.exception("Failed to load glasses-detector classifiers.")
            raise GlassesDetectorError(
                "Could not initialize glasses-detector classifiers.",
            ) from exc

    def _ensure_loaded(self) -> tuple[GlassesClassifier, GlassesClassifier]:
        """Return both classifiers, loading them on first access (thread-safe)."""
        if self._eyeglasses_classifier is None or self._sunglasses_classifier is None:
            with self._load_lock:
                if self._eyeglasses_classifier is None or self._sunglasses_classifier is None:
                    self._load_classifiers()

        if self._eyeglasses_classifier is None or self._sunglasses_classifier is None:
            raise GlassesDetectorError(
                "Glasses-detector classifiers failed to initialize.",
            )

        return self._eyeglasses_classifier, self._sunglasses_classifier

    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_inputs(
        image: np.ndarray,
        face: Face,
    ) -> None:
        """Reject inputs that cannot be processed."""
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a numpy.ndarray, got {type(image).__name__}.",
            )

        if image.size == 0:
            raise ValueError("image must not be empty.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape (H, W, 3), got {image.shape}.",
            )

        if image.dtype != np.uint8:
            raise TypeError(
                f"image dtype must be uint8, got {image.dtype}.",
            )

        if face is None:
            raise ValueError("face must not be None.")

        if not isinstance(face, Face):
            raise TypeError(
                f"face must be a Face instance, got {type(face).__name__}.",
            )

    # ------------------------------------------------------------------ #
    # ROI extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_face_roi(
        image: np.ndarray,
        face: Face,
    ) -> np.ndarray:
        """Crop the face region defined by ``face.bbox``.

        Performs only the minimal slicing required to feed the external
        library; no alignment or resizing is applied here.

        Args:
            image: Full BGR image.
            face: Detected face whose ``bbox`` attribute contains
                ``[x1, y1, x2, y2]`` coordinates.

        Returns:
            BGR sub-image containing the face region.

        Raises:
            GlassesDetectorError: If the bounding box is invalid or
                produces an empty crop.
        """
        bbox = getattr(face, "bbox", None)

        if bbox is None:
            raise GlassesDetectorError(
                "Face object has no bounding box information.",
            )

        try:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
        except (TypeError, ValueError) as exc:
            raise GlassesDetectorError(
                "Face bounding box values are not valid integers.",
            ) from exc

        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(x1, min(x2, w))
        y2 = max(y1, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            raise GlassesDetectorError(
                f"Face bounding box produces an empty crop: "
                f"[{x1}, {y1}, {x2}, {y2}].",
            )

        return image[y1:y2, x1:x2]

    # ------------------------------------------------------------------ #
    # Prediction mapping
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_prediction(
        eyeglasses_prob: float,
        sunglasses_prob: float,
    ) -> EyewearPrediction:
        """Combine both classifier outputs into a single domain prediction.

        Decision priority: sunglasses is checked before eyeglasses.

        WHY: sunglasses frames visually resemble eyeglass frames, so the
        eyeglasses classifier frequently also fires (a false positive)
        on a face wearing sunglasses. Checking sunglasses first ensures
        opaque/tinted lenses are always classified as
        :attr:`EyewearType.SUNGLASSES`, rather than being masked by a
        simultaneous eyeglasses positive.

        The library cannot distinguish prescription glasses, so
        :attr:`EyewearType.PRESCRIPTION_GLASSES` is never returned here.

        Args:
            eyeglasses_prob: Probability from the eyeglasses classifier.
            sunglasses_prob: Probability from the sunglasses classifier.

        Returns:
            An :class:`EyewearPrediction` with the predicted
            :class:`EyewearType` and the confidence associated with
            that specific prediction.
        """
        if sunglasses_prob > GLASSES_SUNGLASSES_PROBABILITY_THRESHOLD:
            return EyewearPrediction(
                eyewear_type=EyewearType.SUNGLASSES,
                confidence=sunglasses_prob,
            )

        if eyeglasses_prob > GLASSES_EYEGLASSES_PROBABILITY_THRESHOLD:
            return EyewearPrediction(
                eyewear_type=EyewearType.CLEAR_GLASSES,
                confidence=eyeglasses_prob,
            )

        return EyewearPrediction(
            eyewear_type=EyewearType.NONE,
            confidence=eyeglasses_prob,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        image: np.ndarray,
        face: Face,
    ) -> EyewearPrediction:
        """Classify eyewear on the detected face region.

        Extracts the minimal face ROI from the bounding box, runs
        the eyeglasses and sunglasses classifiers, and combines
        their outputs into a single :class:`EyewearPrediction`.

        Args:
            image: Full BGR ``uint8`` image of shape ``(H, W, 3)`.
            face: Detected face with bounding box metadata.

        Returns:
            An :class:`EyewearPrediction` with the predicted
            :class:`EyewearType` and associated confidence score.

        Raises:
            TypeError:  If *image* is not a ``numpy.ndarray`` or
                *face* is not a ``Face`` instance.
            ValueError: If *image* is empty or *face* is ``None``.
            GlassesDetectorError: If ROI extraction or inference fails.
        """
        self._validate_inputs(image, face)

        roi_bgr = self._extract_face_roi(image, face)
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        roi_bgr = self._extract_face_roi(image, face)

        # للتجربة فقط
        roi_rgb = cv2.resize(roi_rgb, (224, 224))

        print("=" * 60)
        print("ROI shape after resize:", roi_rgb.shape)
        print("=" * 60)



        eyeglasses_classifier, sunglasses_classifier = self._ensure_loaded()

        try:
            eyeglasses_prob: float = eyeglasses_classifier.predict(
                roi_rgb,
                format="proba",
                input_size=None,
            )
        except Exception as exc:
            logger.exception("Eyeglasses classifier inference failed.")
            raise GlassesDetectorError(
                "Eyeglasses classification failed.",
            ) from exc

        try:
            sunglasses_prob: float = sunglasses_classifier.predict(
                roi_rgb,
                format="proba",
                input_size=None,
            )
        except Exception as exc:
            logger.exception("Sunglasses classifier inference failed.")
            raise GlassesDetectorError(
                "Sunglasses classification failed.",
            ) from exc

        return self._build_prediction(eyeglasses_prob, sunglasses_prob)