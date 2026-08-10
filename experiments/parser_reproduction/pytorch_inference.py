"""
PyTorch inference module for BiSeNet face parsing.

Uses identical preprocessing to the ONNX path for fair comparison.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt
import torch

from .bisenet_model import BiSeNet

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE: tuple[int, int] = (512, 512)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(tensor).unsqueeze(0).float()


def run_pytorch_inference(
    model: BiSeNet,
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        out, out16, out32 = model(tensor)
    return out, out16, out32


def postprocess(
    raw_output: torch.Tensor,
    original_h: int,
    original_w: int,
) -> npt.NDArray[np.int32]:
    logits = raw_output.squeeze(0).cpu().numpy()
    class_map = np.argmax(logits, axis=0)
    model_h, model_w = class_map.shape
    if (model_h, model_w) != (original_h, original_w):
        mask = cv2.resize(
            class_map.astype(np.uint8),
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    else:
        mask = class_map.astype(np.int32)
    return mask


def inference_pytorch(
    model: BiSeNet,
    image_bgr: np.ndarray,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    h, w = image_bgr.shape[:2]
    tensor = preprocess(image_bgr)
    out, _, _ = run_pytorch_inference(model, tensor)
    logits_np = out.cpu().numpy()
    mask = postprocess(out, h, w)
    return logits_np, mask
