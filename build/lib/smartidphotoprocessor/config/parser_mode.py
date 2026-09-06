"""Parser mode enum for selecting between ORIGINAL (ONNX) and FUSED (ONNX + Auxiliary Refinement) parsing."""
from __future__ import annotations

from enum import Enum


class ParserMode(str, Enum):
    """Execution mode for FaceParserService."""

    ORIGINAL = "ORIGINAL"
    """Use the existing production BiSeNet ONNX model without auxiliary refinement."""

    FUSED = "FUSED"
    """Use bisenet_resnet18_with_ffm.onnx + aux_head.onnx + deterministic Python fusion (production default)."""
