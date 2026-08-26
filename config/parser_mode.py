"""Parser mode enum for selecting between ORIGINAL (ONNX) and FUSED (PyTorch + Auxiliary Refinement) parsing."""
from __future__ import annotations

from enum import Enum


class ParserMode(str, Enum):
    """Execution mode for FaceParserService."""

    ORIGINAL = "ORIGINAL"
    """Use the existing production BiSeNet ONNX model without auxiliary refinement."""

    FUSED = "FUSED"
    """Use frozen BiSeNet + Phase 1 Auxiliary Head + Phase 3 confidence-aware fusion."""
