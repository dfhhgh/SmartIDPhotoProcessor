"""
ONNX inference module reproducing production FaceParserService.

Identical preprocessing and inference pipeline to services/face_parser_service.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE: tuple[int, int] = (512, 512)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_onnx_session(
    onnx_path: Path,
    use_gpu: bool = False,
) -> ort.InferenceSession:
    providers: list[str | tuple[str, dict[str, int]]] = ["CPUExecutionProvider"]
    if use_gpu:
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers = [
                ("CUDAExecutionProvider", {"device_id": 0}),
                "CPUExecutionProvider",
            ]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    logger.info("ONNX session loaded: providers=%s", session.get_providers())
    return session


def preprocess(image_bgr: np.ndarray) -> npt.NDArray[np.float32]:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(tensor, dtype=np.float32)


def run_onnx_inference(
    session: ort.InferenceSession,
    tensor: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    input_name = session.get_inputs()[0].name
    output_metas = session.get_outputs()
    outputs = session.run(None, {input_name: tensor})

    if not outputs or not output_metas:
        raise RuntimeError("ONNX model returned no outputs")

    selected = None
    for meta, arr in zip(output_metas, outputs):
        if isinstance(arr, np.ndarray) and meta.name == "output":
            selected = arr
            break

    if selected is None:
        selected = outputs[0]

    return selected


def postprocess(
    raw_output: npt.NDArray[np.float32],
    original_h: int,
    original_w: int,
) -> npt.NDArray[np.int32]:
    class_map = np.argmax(raw_output[0], axis=0)
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


def inference_onnx(
    session: ort.InferenceSession,
    image_bgr: np.ndarray,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int32]]:
    h, w = image_bgr.shape[:2]
    tensor = preprocess(image_bgr)
    raw_output = run_onnx_inference(session, tensor)
    mask = postprocess(raw_output, h, w)
    return raw_output, mask
