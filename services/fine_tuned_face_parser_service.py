"""Evaluation-only FaceParserService backed by PyTorch best.pt checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import torch

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from experiments.parser_reproduction.bisenet_model import BiSeNet
from services.face_parser_service import FaceParserError

logger = logging.getLogger(__name__)


class FineTunedFaceParserService:
    """
    Evaluation-only face parser service that loads the fine-tuned PyTorch
    BiSeNet checkpoint (best.pt) and implements the exact same public parse()
    interface as FaceParserService.
    """

    _MODEL_INPUT_SIZE: tuple[int, int] = (512, 512)
    _MEAN: npt.NDArray[np.float32] = np.array(
        [0.485, 0.456, 0.406], dtype=np.float32
    )
    _STD: npt.NDArray[np.float32] = np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        if checkpoint_path is None:
            project_root = Path(__file__).resolve().parents[1]
            checkpoint_path = (
                project_root
                / "dataset_builder"
                / "dataset"
                / "parser_finetune_expanded"
                / "training"
                / "checkpoints"
                / "best.pt"
            )
        self._checkpoint_path = Path(checkpoint_path)

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model: BiSeNet | None = None
        self._load_model()

    def _load_model(self) -> None:
        logger.info("Loading fine-tuned BiSeNet checkpoint from %s ...", self._checkpoint_path)
        if not self._checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned BiSeNet checkpoint not found at: {self._checkpoint_path}"
            )

        try:
            model = BiSeNet(n_classes=len(FacePart))
            checkpoint = torch.load(self._checkpoint_path, map_location=self._device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            self._model = model
            logger.info(
                "Fine-tuned BiSeNet model loaded successfully on device %s (best val target mIoU: %s).",
                self._device,
                checkpoint.get("best_val_target_mean_iou", "N/A"),
            )
        except Exception as exc:
            logger.exception("Failed to load fine-tuned BiSeNet model.")
            raise FaceParserError(
                "Could not initialize fine-tuned BiSeNet face-parsing model."
            ) from exc

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
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

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            self._MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_LINEAR,
        )
        normalized = (resized.astype(np.float32) / 255.0 - self._MEAN) / self._STD
        tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return torch.from_numpy(np.ascontiguousarray(tensor, dtype=np.float32)).to(self._device)

    def parse(self, image: np.ndarray) -> FaceParsingResult:
        self._validate_image(image)
        original_height, original_width = image.shape[:2]

        if self._model is None:
            raise FaceParserError("Fine-tuned BiSeNet model is not loaded.")

        try:
            tensor = self._preprocess(image)
        except Exception as exc:
            logger.exception("Fine-tuned face-parsing preprocessing failed.")
            raise FaceParserError("Face-parsing preprocessing failed.") from exc

        try:
            with torch.no_grad():
                outputs = self._model(tensor)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        except Exception as exc:
            logger.exception("Fine-tuned BiSeNet inference failed.")
            raise FaceParserError("Face-parsing inference failed.") from exc

        try:
            class_map = preds[0]
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
        except Exception as exc:
            logger.exception("Fine-tuned face-parsing post-processing failed.")
            raise FaceParserError("Face-parsing post-processing failed.") from exc
