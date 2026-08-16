"""Unit and integration tests for FineTunedFaceParserService and real best.pt checkpoint loading."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from models.parsing.face_parsing_result import FaceParsingResult
from services.fine_tuned_face_parser_service import FineTunedFaceParserService
from models.parsing.face_part import FacePart


class TestFineTunedFaceParserIntegration:
    def test_loads_real_best_pt_checkpoint_successfully(self):
        project_root = Path(__file__).resolve().parents[2]
        best_pt_path = (
            project_root
            / "dataset_builder"
            / "dataset"
            / "parser_finetune_expanded"
            / "training"
            / "checkpoints"
            / "best.pt"
        )
        if not best_pt_path.exists():
            pytest.skip("Real best.pt checkpoint not found on disk.")

        parser = FineTunedFaceParserService(checkpoint_path=best_pt_path, device="cpu")
        img = np.zeros((112, 112, 3), dtype=np.uint8)
        result = parser.parse(img)

        assert isinstance(result, FaceParsingResult)
        assert result.mask.shape == (112, 112)
        assert result.mask.dtype == np.int32
        assert result.image_height == 112
        assert result.image_width == 112

        unique_classes = set(np.unique(result.mask))
        assert unique_classes.issubset(set(range(len(FacePart))))

    def test_synthetic_checkpoint_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "best.pt"
            from experiments.parser_reproduction.bisenet_model import BiSeNet
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
