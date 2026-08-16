"""Unit tests for FineTunedFaceParserService."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from models.parsing.face_parsing_result import FaceParsingResult
from services.fine_tuned_face_parser_service import FineTunedFaceParserService
from experiments.parser_reproduction.bisenet_model import BiSeNet


class TestFineTunedFaceParserService:
    def test_loads_checkpoint_and_parses(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            model = BiSeNet(n_classes=19)
            payload = {
                "model_state_dict": model.state_dict(),
                "best_val_target_mean_iou": 0.75,
            }
            torch.save(payload, ckpt_path)

            parser = FineTunedFaceParserService(checkpoint_path=ckpt_path, device="cpu")
            img = np.zeros((112, 112, 3), dtype=np.uint8)
            result = parser.parse(img)

            assert isinstance(result, FaceParsingResult)
            assert result.mask.shape == (112, 112)
            assert result.mask.dtype == np.int32
            assert result.image_height == 112
            assert result.image_width == 112

    def test_rejects_invalid_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            model = BiSeNet(n_classes=19)
            torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
            parser = FineTunedFaceParserService(checkpoint_path=ckpt_path, device="cpu")

            with pytest.raises(TypeError):
                parser.parse("not an array")
            with pytest.raises(ValueError):
                parser.parse(np.array([], dtype=np.uint8))
