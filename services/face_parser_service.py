"""
Service responsible for running BiSeNet face-parsing inference.

Follows the same singleton + lazy-loading architecture as FaceService
so that both AI services behave consistently across the project.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from config.settings import Settings
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult

logger = logging.getLogger(__name__)

# onnxruntime accepts either a bare provider name or a (name, options) pair.
ExecutionProvider = str | tuple[str, dict[str, int]]


class FaceParserError(Exception):
    """Raised when face-parsing inference fails."""


class FaceParserService:
    """
    Thin wrapper around a BiSeNet ONNX model that produces a
    :class:`FaceParsingResult` for any BGR ``uint8`` face image.

    The model is loaded lazily on the first call to :meth:`parse` and
    cached for the lifetime of the process (singleton pattern).
    """

    _instance: FaceParserService | None = None
    _initialized: bool = False
    _instance_lock: threading.Lock = threading.Lock()

    # BiSeNet (ResNet-18) trained on CelebAMask-HQ expects 512×512 RGB.
    _MODEL_INPUT_SIZE: tuple[int, int] = (512, 512)
    _MEAN: npt.NDArray[np.float32] = np.array(
        [0.485, 0.456, 0.406], dtype=np.float32
    )
    _STD: npt.NDArray[np.float32] = np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    # ------------------------------------------------------------------ #
    # Singleton lifecycle
    # ------------------------------------------------------------------ #

    def __new__(cls) -> FaceParserService:
        # Double-checked locking: avoids taking the lock on the (hot) path
        # where the instance already exists, while still being safe if
        # two threads race to create it the first time.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return

        settings = Settings()

        self._model_path: Path = (
            settings.MODEL_ROOT / "bisenet" / "bisenet_resnet18.onnx"
        )
        self._use_gpu: bool = settings.USE_GPU
        self._gpu_id: int = settings.GPU_ID

        self._session: ort.InferenceSession | None = None
        self._load_lock = threading.Lock()

        self._initialized = True

    # ------------------------------------------------------------------ #
    # Model management
    # ------------------------------------------------------------------ #

    def _load_model(self) -> None:
        """Create the ONNX Runtime inference session (once)."""
        logger.info("Loading BiSeNet face-parsing model from %s ...", self._model_path)

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"BiSeNet ONNX model not found at: {self._model_path}"
            )

        providers = self._resolve_providers()

        try:
            self._session = ort.InferenceSession(
                str(self._model_path),
                providers=providers,
            )
            logger.info(
                "BiSeNet model loaded successfully (providers=%s).",
                self._session.get_providers(),
            )
        except Exception as exc:
            logger.exception("Failed to load BiSeNet ONNX model.")
            raise FaceParserError(
                "Could not initialize BiSeNet face-parsing model."
            ) from exc

    def _resolve_providers(self) -> list[ExecutionProvider]:
        """Return the ordered list of execution providers to request."""
        available = ort.get_available_providers()

        if self._use_gpu and "CUDAExecutionProvider" in available:
            return [
                ("CUDAExecutionProvider", {"device_id": self._gpu_id}),
                "CPUExecutionProvider",
            ]

        return ["CPUExecutionProvider"]

    def _ensure_loaded(self) -> ort.InferenceSession:
        """Return the session, loading the model on first access (thread-safe)."""
        if self._session is None:
            with self._load_lock:
                # Re-check inside the lock: another thread may have
                # finished loading while we were waiting for it.
                if self._session is None:
                    self._load_model()

        if self._session is None:
            # _load_model() either sets self._session or raises; this is
            # a defensive guard rather than the expected path.
            raise FaceParserError("Face-parsing model failed to initialize.")

        return self._session

    # ------------------------------------------------------------------ #
    # Input validation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        """Reject inputs that cannot be processed."""
        if not isinstance(image, np.ndarray):
            raise TypeError(
                f"image must be a numpy.ndarray, got {type(image).__name__}."
            )

        if image.size == 0:
            raise ValueError("image must not be empty.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"image must have shape (H, W, 3), got {image.shape}."
            )

        if image.dtype != np.uint8:
            raise TypeError(
                f"image dtype must be uint8, got {image.dtype}."
            )

    # ------------------------------------------------------------------ #
    # Preprocessing
    # ------------------------------------------------------------------ #

    def _preprocess(self, image: np.ndarray) -> npt.NDArray[np.float32]:
        """
        Prepare *image* for the BiSeNet ONNX graph.

        Steps: BGR→RGB → resize → float32 [0,1] → normalize → NCHW.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            self._MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        normalized = (resized.astype(np.float32) / 255.0 - self._MEAN) / self._STD

        # HWC → CHW → NCHW
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(tensor, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def _run_inference(
        self,
        session: ort.InferenceSession,
        tensor: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Execute the ONNX graph and return raw logits, validating shape."""
        input_name = session.get_inputs()[0].name
        output_metas = session.get_outputs()
        outputs = session.run(None, {input_name: tensor})

        if not outputs or not output_metas:
            raise FaceParserError("Model returned no outputs.")

        valid_outputs = []
        for meta, arr in zip(output_metas, outputs):
            if (
                isinstance(arr, np.ndarray)
                and arr.ndim == 4
                and arr.shape[0] == 1
                and arr.shape[1] == len(FacePart)
            ):
                valid_outputs.append((meta.name, arr))

        if not valid_outputs:
            raise FaceParserError(
                f"No valid segmentation output found matching expected shape (1, {len(FacePart)}, H, W)."
            )

        if len(valid_outputs) > 1:
            logger.warning(
                "Multiple valid segmentation outputs detected (%d). Selecting based on rules.",
                len(valid_outputs),
            )

        # Prefer output whose metadata name is "output"
        selected_output = None
        for name, arr in valid_outputs:
            if name == "output":
                selected_output = arr
                break

        # If no output named "output" exists, select the first valid output
        if selected_output is None:
            selected_output = valid_outputs[0][1]

        if selected_output.ndim != 4 or selected_output.shape[0] != 1:
            raise FaceParserError(
                f"Unexpected BiSeNet output shape {selected_output.shape}; "
                "expected (1, C, H, W)."
            )

        if selected_output.shape[1] != len(FacePart):
            raise FaceParserError(
                f"Model output has {selected_output.shape[1]} class channels, "
                f"but {len(FacePart)} FacePart classes are defined."
            )

        return selected_output

    # ------------------------------------------------------------------ #
    # Post-processing
    # ------------------------------------------------------------------ #

    def _postprocess(
        self,
        raw_output: npt.NDArray[np.float32],
        original_height: int,
        original_width: int,
    ) -> FaceParsingResult:
        """
        Convert raw network output to a :class:`FaceParsingResult`.

        * argmax over the class axis → 2-D integer mask
        * resize back to the original image resolution (nearest-neighbour
          to preserve label integrity)
        """
        # raw_output shape: (1, C, H, W) → squeeze batch
        class_map: npt.NDArray[np.intp] = np.argmax(raw_output[0], axis=0)

        # Resize to original resolution if necessary.
        model_h, model_w = class_map.shape
        if (model_h, model_w) != (original_height, original_width):
            segmentation_mask = cv2.resize(
                class_map.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        else:
            segmentation_mask = class_map.astype(np.int32)

        return FaceParsingResult(
            mask=segmentation_mask,
            image_height=original_height,
            image_width=original_width,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def parse(self, image: np.ndarray) -> FaceParsingResult:
        """
        Run face-parsing inference on *image* and return the
        per-pixel segmentation result.

        Args:
            image: BGR ``uint8`` image of shape ``(H, W, 3)``.

        Returns:
            An immutable :class:`FaceParsingResult` whose mask covers
            every pixel of *image* with a :class:`FacePart` label.

        Raises:
            TypeError:  If *image* is not a ``numpy.ndarray`` or not ``uint8``.
            ValueError: If *image* is empty or has an incompatible shape.
            FaceParserError: If preprocessing, inference, or post-processing fails.
        """
        self._validate_image(image)

        original_height, original_width = image.shape[:2]

        session = self._ensure_loaded()

        try:
            tensor = self._preprocess(image)
        except Exception as exc:
            logger.exception("Face-parsing preprocessing failed.")
            raise FaceParserError("Face-parsing preprocessing failed.") from exc

        try:
            raw_output = self._run_inference(session, tensor)
        except FaceParserError:
            raise
        except Exception as exc:
            logger.exception("BiSeNet inference failed.")
            raise FaceParserError("Face-parsing inference failed.") from exc

        try:
            return self._postprocess(raw_output, original_height, original_width)
        except Exception as exc:
            logger.exception("Face-parsing post-processing failed.")
            raise FaceParserError("Face-parsing post-processing failed.") from exc