"""Unit tests for the evaluation runner (evaluate_finetuned_parser.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
import cv2
import numpy as np

from evaluation.evaluate_finetuned_parser import evaluate_pipelines, SUPPORTED_EXTENSIONS


class TestEvaluationRunner:
    def test_supported_extensions(self):
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS

    def test_evaluate_pipelines_handles_errors_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            img_path = tmp_path / "dummy.png"
            # Create a blank dummy image
            dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), dummy_img)

            report = evaluate_pipelines([img_path], tmp_path / "reports")
            assert report["statistics"]["total_images"] == 1
            assert len(report["results"]) == 1
            assert report["production_model"]["sha256"] != "not_found"
            assert report["fine_tuned_model"]["sha256"] != "not_found"
