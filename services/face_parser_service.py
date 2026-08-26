"""
Service responsible for running BiSeNet face-parsing inference.

Follows the same singleton + lazy-loading architecture as FaceService
so that both AI services behave consistently across the project.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
import torch.nn.functional as F
from scipy.ndimage import label

from config.parser_mode import ParserMode
from config.settings import Settings
from dataset_builder.dataset.parser_finetune_current.training_aux_eye_brow_phase1.auxiliary_head import (
    EyeBrowRefinementHead,
)
from experiments.parser_reproduction.bisenet_model import BiSeNet
from experiments.parser_reproduction.weight_mapping import load_onnx_to_pytorch
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult

logger = logging.getLogger(__name__)

# onnxruntime accepts either a bare provider name or a (name, options) pair.
ExecutionProvider = str | tuple[str, dict[str, int]]


class FaceParserError(Exception):
    """Raised when face-parsing inference fails."""



CLASS_MAP_19_TO_6: dict[int, int] = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
}
CLASS_MAP_6_TO_19: dict[int, int] = {
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
}
TARGET_CLASSES_6: frozenset[int] = frozenset(CLASS_MAP_6_TO_19)
TARGET_CLASSES_19: frozenset[int] = frozenset(CLASS_MAP_19_TO_6)


def map_19_to_6_numpy(mask_19: np.ndarray) -> np.ndarray:
    """Map a production 19-class mask into the auxiliary 6-class label space."""
    out = np.zeros_like(mask_19, dtype=np.int64)
    for src, dst in CLASS_MAP_19_TO_6.items():
        out[mask_19 == src] = dst
    return out


def construct_eye_brow_roi(pred_19: np.ndarray, pred_aux_19: np.ndarray) -> np.ndarray:
    """Construct the Phase 3 anatomical ROI from original and auxiliary targets."""
    roi = np.zeros_like(pred_19, dtype=bool)
    for c in TARGET_CLASSES_19:
        roi |= pred_19 == c
    for c in TARGET_CLASSES_6:
        roi |= pred_aux_19 == CLASS_MAP_6_TO_19[c]
    return roi


@dataclass(frozen=True, slots=True)
class FusionDiagnostics:
    """Diagnostics from one Phase 3 fusion operation."""

    corrections_attempted: int
    corrections_accepted: int
    corrections_rejected: int
    roi_pixels: int

    def to_dict(self) -> dict[str, int]:
        return {
            "corrections_attempted": self.corrections_attempted,
            "corrections_accepted": self.corrections_accepted,
            "corrections_rejected": self.corrections_rejected,
            "roi_pixels": self.roi_pixels,
        }


class EyeBrowRefinementFusion:
    """Phase 3 fusion engine for BiSeNet 19-class and auxiliary 6-class logits."""

    def __init__(
        self,
        strategy: int = 1,
        threshold: float = 0.0,
        min_component_size: int = 10,
    ) -> None:
        self.strategy = strategy
        self.threshold = threshold
        self.min_component_size = min_component_size

    def apply(
        self,
        logits_19: torch.Tensor,
        logits_aux: torch.Tensor,
    ) -> tuple[npt.NDArray[np.int64], FusionDiagnostics]:
        """Apply the validated Phase 3 fusion gates."""
        if logits_19.ndim == 4:
            logits_19 = logits_19.squeeze(0)
        if logits_aux.ndim == 4:
            logits_aux = logits_aux.squeeze(0)

        prob_19 = F.softmax(logits_19, dim=0)
        _, pred_19 = prob_19.max(dim=0)
        pred_19_np = pred_19.detach().cpu().numpy()

        prob_aux = F.softmax(logits_aux, dim=0)
        conf_aux, pred_aux_6 = prob_aux.max(dim=0)
        pred_aux_6_np = pred_aux_6.detach().cpu().numpy()
        conf_aux_np = conf_aux.detach().cpu().numpy()

        final_mask = pred_19_np.copy()
        pred_aux_19 = np.zeros_like(pred_aux_6_np, dtype=np.int64)
        for c6, c19 in CLASS_MAP_6_TO_19.items():
            pred_aux_19[pred_aux_6_np == c6] = c19

        roi = construct_eye_brow_roi(pred_19_np, pred_aux_19)
        corrections_attempted = 0
        corrections_accepted = 0
        corrections_rejected = 0

        if self.strategy != 0:
            is_aux_target = np.isin(pred_aux_6_np, list(TARGET_CLASSES_6))
            is_original_target = np.isin(pred_19_np, list(TARGET_CLASSES_19))
            disagreement = pred_19_np != pred_aux_19
            high_conf = conf_aux_np >= self.threshold
            candidate = roi & is_aux_target & is_original_target

            if self.strategy >= 2:
                candidate &= high_conf
            if self.strategy >= 3:
                candidate &= disagreement
            if self.strategy >= 4:
                spatial_mask = np.zeros_like(candidate)
                for c6 in TARGET_CLASSES_6:
                    class_candidate = candidate & (pred_aux_6_np == c6)
                    if class_candidate.any():
                        labeled, num_features = label(class_candidate)
                        for comp_id in range(1, num_features + 1):
                            component_pixels = labeled == comp_id
                            if component_pixels.sum() >= self.min_component_size:
                                spatial_mask |= component_pixels
                candidate = spatial_mask

            corrections_attempted = int((roi & is_aux_target).sum())
            corrections_accepted = int(candidate.sum())
            corrections_rejected = corrections_attempted - corrections_accepted
            final_mask[candidate] = pred_aux_19[candidate]

        diagnostics = FusionDiagnostics(
            corrections_attempted=corrections_attempted,
            corrections_accepted=corrections_accepted,
            corrections_rejected=corrections_rejected,
            roi_pixels=int(roi.sum()),
        )
        return final_mask.astype(np.int64), diagnostics


class EyeBrowRefinementService:
    """Lazy frozen PyTorch BiSeNet + auxiliary head for fused parser mode."""

    def __init__(
        self,
        onnx_model_path: Path,
        checkpoint_path: Path,
        fusion: EyeBrowRefinementFusion,
        device: torch.device | None = None,
        ffm_channels: int = 256,
        aux_mid_channels: tuple[int, ...] = (128, 64),
    ) -> None:
        self._onnx_model_path = Path(onnx_model_path)
        self._checkpoint_path = Path(checkpoint_path)
        self._fusion = fusion
        self._device = device
        self._ffm_channels = ffm_channels
        self._aux_mid_channels = aux_mid_channels
        self._bisenet: BiSeNet | None = None
        self._head: EyeBrowRefinementHead | None = None
        self._load_lock = threading.Lock()
        self.last_diagnostics: FusionDiagnostics | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self._checkpoint_path

    @property
    def bisenet(self) -> BiSeNet | None:
        return self._bisenet

    @property
    def head(self) -> EyeBrowRefinementHead | None:
        return self._head

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            if self._device.type != "cuda" or self._device.index not in (0, None):
                raise RuntimeError("Fused parser auxiliary inference must run on cuda:0.")
            return torch.device("cuda:0")

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Fused parser mode requires CUDA for auxiliary inference. "
                "torch.cuda.is_available() is False; no CPU fallback is allowed."
            )
        return torch.device("cuda:0")

    def _ensure_loaded(self) -> tuple[BiSeNet, EyeBrowRefinementHead, torch.device]:
        device = self._resolve_device()
        if self._bisenet is None or self._head is None:
            with self._load_lock:
                if self._bisenet is None:
                    self._bisenet = self._load_frozen_bisenet(device)
                if self._head is None:
                    self._head = self._load_frozen_head(device)
        return self._bisenet, self._head, device

    def _load_frozen_bisenet(self, device: torch.device) -> BiSeNet:
        if not self._onnx_model_path.exists():
            raise FileNotFoundError(f"BiSeNet ONNX model not found at: {self._onnx_model_path}")
        model = BiSeNet(n_classes=len(FacePart))
        model = load_onnx_to_pytorch(self._onnx_model_path, model).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
                module.track_running_stats = False
        return model

    def _load_frozen_head(self, device: torch.device) -> EyeBrowRefinementHead:
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(f"Auxiliary Eye/Brow checkpoint not found at: {self._checkpoint_path}")
        head = EyeBrowRefinementHead(
            self._ffm_channels,
            self._aux_mid_channels,
            n_classes=6,
        ).to(device)
        checkpoint: dict[str, Any] = torch.load(
            self._checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        head.load_state_dict(checkpoint["head_state_dict"])
        head.eval()
        for param in head.parameters():
            param.requires_grad = False
        return head

    def refine(
        self,
        input_tensor: npt.NDArray[np.float32],
        original_height: int,
        original_width: int,
    ) -> npt.NDArray[np.int32]:
        """Run Phase 3 fused inference and return a resized 19-class mask."""
        bisenet, head, device = self._ensure_loaded()
        tensor = torch.from_numpy(input_tensor).to(device=device, dtype=torch.float32)
        _, _, target_h, target_w = tensor.shape

        with torch.no_grad():
            feat_res8, feat_cp8, feat_cp16 = bisenet.cp(tensor)
            fused_features = bisenet.ffm(feat_res8, feat_cp8)
            logits_19 = bisenet.conv_out(fused_features, target_h, target_w)
            logits_aux = head(fused_features, target_h, target_w)
            final_mask, diagnostics = self._fusion.apply(logits_19, logits_aux)

        self.last_diagnostics = diagnostics

        if final_mask.shape != (original_height, original_width):
            final_mask = cv2.resize(
                final_mask.astype(np.uint8),
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

        return final_mask.astype(np.int32)

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

    # BiSeNet (ResNet-18) trained on CelebAMask-HQ expects 512Ãƒâ€”512 RGB.
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

    def __new__(cls, *args, **kwargs) -> FaceParserService:
        # Double-checked locking: avoids taking the lock on the (hot) path
        # where the instance already exists, while still being safe if
        # two threads race to create it the first time.
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        parser_mode: ParserMode | str | None = None,
        refinement_service: EyeBrowRefinementService | None = None,
    ) -> None:
        if self._initialized:
            return

        settings = Settings()

        self._model_path: Path = (
            settings.MODEL_ROOT / "bisenet" / "bisenet_resnet18.onnx"
        )
        self._use_gpu: bool = settings.USE_GPU
        self._gpu_id: int = settings.GPU_ID
        configured_mode = parser_mode if parser_mode is not None else getattr(settings, "PARSER_MODE", ParserMode.ORIGINAL)
        if not isinstance(configured_mode, (ParserMode, str)):
            configured_mode = ParserMode.ORIGINAL
        self._parser_mode = ParserMode(configured_mode)
        self._refinement_service = refinement_service
        if self._parser_mode is ParserMode.FUSED and self._refinement_service is None:
            self._refinement_service = EyeBrowRefinementService(
                onnx_model_path=self._model_path,
                checkpoint_path=getattr(settings, "AUX_EYE_BROW_CHECKPOINT_PATH", Settings.AUX_EYE_BROW_CHECKPOINT_PATH),
                fusion=EyeBrowRefinementFusion(
                    strategy=getattr(settings, "EYE_BROW_FUSION_STRATEGY", Settings.EYE_BROW_FUSION_STRATEGY),
                    threshold=getattr(settings, "EYE_BROW_FUSION_THRESHOLD", Settings.EYE_BROW_FUSION_THRESHOLD),
                    min_component_size=getattr(settings, "EYE_BROW_FUSION_MIN_COMPONENT_SIZE", Settings.EYE_BROW_FUSION_MIN_COMPONENT_SIZE),
                ),
            )

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

        Steps: BGRÃ¢â€ â€™RGB Ã¢â€ â€™ resize Ã¢â€ â€™ float32 [0,1] Ã¢â€ â€™ normalize Ã¢â€ â€™ NCHW.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            self._MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )

        normalized = (resized.astype(np.float32) / 255.0 - self._MEAN) / self._STD

        # HWC Ã¢â€ â€™ CHW Ã¢â€ â€™ NCHW
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

        * argmax over the class axis Ã¢â€ â€™ 2-D integer mask
        * resize back to the original image resolution (nearest-neighbour
          to preserve label integrity)
        """
        # raw_output shape: (1, C, H, W) Ã¢â€ â€™ squeeze batch
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

        try:
            tensor = self._preprocess(image)
        except Exception as exc:
            logger.exception("Face-parsing preprocessing failed.")
            raise FaceParserError("Face-parsing preprocessing failed.") from exc

        if self._parser_mode is ParserMode.FUSED:
            try:
                if self._refinement_service is None:
                    raise FaceParserError("Fused parser mode has no refinement service configured.")
                fused_mask = self._refinement_service.refine(
                    tensor,
                    original_height,
                    original_width,
                )
                return FaceParsingResult(
                    mask=fused_mask,
                    image_height=original_height,
                    image_width=original_width,
                )
            except FaceParserError:
                raise
            except Exception as exc:
                logger.exception("Fused face-parsing inference failed.")
                raise FaceParserError("Fused face-parsing inference failed.") from exc

        session = self._ensure_loaded()

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

