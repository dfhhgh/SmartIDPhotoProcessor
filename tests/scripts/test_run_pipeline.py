"""Tests for scripts/run_pipeline.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.ranked_face import RankedFace
from models.selection_result import SelectionResult
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from scripts.run_pipeline import collect_image_paths, process_image
from tests.factories import create_face


def test_process_image_saves_exported_images_for_valid_results(tmp_path):
    """Valid results should be exported to outputs/exported using the original filename."""
    img_file = tmp_path / "person03.jpg"
    img_file.write_bytes(b"dummy")

    aligned_dir = tmp_path / "outputs" / "aligned"
    cropped_dir = tmp_path / "outputs" / "cropped"
    exported_dir = tmp_path / "outputs" / "exported"
    reports_dir = tmp_path / "outputs" / "reports"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    exported_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    exported_image = np.zeros((100, 100, 3), dtype=np.uint8)

    with patch("scripts.run_pipeline.FaceDetector") as mock_det_cls, \
         patch("scripts.run_pipeline.FaceSelector") as mock_sel_cls, \
         patch("scripts.run_pipeline.FaceCropper") as mock_crop_cls, \
         patch("scripts.run_pipeline.FaceCoordinateTransformer") as mock_trans_cls, \
         patch("scripts.run_pipeline.FaceAligner") as mock_align_cls, \
         patch("scripts.run_pipeline.FaceParserService") as mock_parse_cls, \
         patch("scripts.run_pipeline.ValidationOrchestrator") as mock_orch_cls, \
         patch("scripts.run_pipeline.FaceAmbiguityValidator") as mock_amb_cls, \
         patch("scripts.run_pipeline.PhotoExporter") as mock_export_cls, \
         patch("cv2.imread", return_value=image), \
         patch("cv2.imwrite", return_value=True) as mock_imwrite:

        selected_face = create_face()
        selection_result = _make_selection_result(selected_face)
        mock_det_cls.return_value.detect.return_value = [MagicMock()]
        mock_sel_cls.return_value.select.return_value = selection_result
        mock_amb_cls.return_value.validate.return_value = _make_ambiguity_metric()
        mock_crop_cls.return_value.crop.return_value = MagicMock(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            crop_x=10,
            crop_y=20,
        )
        mock_trans_cls.return_value.transform.return_value = MagicMock()
        mock_align_cls.return_value.align.return_value = MagicMock(
            aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
            aligned_face=MagicMock(),
        )
        mock_parse_cls.return_value.parse.return_value = MagicMock()

        mock_validation_result = MagicMock()
        mock_validation_result.is_valid = True
        metric = MagicMock()
        metric.type.name = "BLUR"
        metric.passed = True
        metric.score = 0.95
        metric.message = ""
        mock_validation_result.metrics = [metric]
        mock_orch_cls.return_value.validate.return_value = mock_validation_result
        mock_export_cls.return_value.export.return_value = MagicMock(exported_image=exported_image)

        process_image(
            img_path=img_file,
            aligned_dir=aligned_dir,
            cropped_dir=cropped_dir,
            exported_dir=exported_dir,
            reports_dir=reports_dir,
        )

    mock_export_cls.return_value.export.assert_called_once()
    mock_imwrite.assert_any_call(str(exported_dir / img_file.name), exported_image)


def _make_selection_result(selected_face=None) -> SelectionResult:
    face = selected_face if selected_face is not None else create_face()
    ranked_face = RankedFace(face=face, score=0.9)
    return SelectionResult(
        selected_face=face,
        selected_score=0.9,
        second_best_score=None,
        score_margin=0.9,
        ambiguity_ratio=0.0,
        detected_faces_count=1,
        ranked_faces=(ranked_face,),
    )


def _make_ambiguity_metric(passed: bool = True) -> ValidationMetric:
    return ValidationMetric(
        type=ValidationType.FACE_AMBIGUITY,
        passed=passed,
        score=1.0 if passed else 0.1,
        message="" if passed else "Ambiguous face selection.",
    )


def test_collect_image_paths_single_file(tmp_path):
    """Verify collecting a single valid image file."""
    img_file = tmp_path / "test.jpg"
    img_file.write_bytes(b"dummy")

    paths = collect_image_paths(img_file)
    assert paths == [img_file]


def test_collect_image_paths_unsupported_file(tmp_path):
    """Verify collecting an unsupported file returns empty list."""
    txt_file = tmp_path / "test.txt"
    txt_file.write_bytes(b"dummy")

    paths = collect_image_paths(txt_file)
    assert paths == []


def test_collect_image_paths_directory(tmp_path):
    """Verify collecting images from a directory."""
    jpg_file = tmp_path / "a.jpg"
    png_file = tmp_path / "b.png"
    txt_file = tmp_path / "c.txt"
    jpg_file.write_bytes(b"dummy")
    png_file.write_bytes(b"dummy")
    txt_file.write_bytes(b"dummy")

    paths = collect_image_paths(tmp_path)
    assert sorted(paths) == sorted([jpg_file, png_file])


def test_collect_image_paths_empty_directory(tmp_path):
    """Verify processing an empty directory returns empty list."""
    paths = collect_image_paths(tmp_path)
    assert paths == []


def test_process_image_success_valid(tmp_path):
    """Verify process_image correctly handles valid validation results and saves outputs."""
    img_file = tmp_path / "person01.jpg"
    img_file.write_bytes(b"dummy")

    aligned_dir = tmp_path / "outputs" / "aligned"
    cropped_dir = tmp_path / "outputs" / "cropped"
    reports_dir = tmp_path / "outputs" / "reports"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((200, 200, 3), dtype=np.uint8)

    with patch("scripts.run_pipeline.FaceDetector") as mock_det_cls, \
         patch("scripts.run_pipeline.FaceSelector") as mock_sel_cls, \
         patch("scripts.run_pipeline.FaceCropper") as mock_crop_cls, \
         patch("scripts.run_pipeline.FaceCoordinateTransformer") as mock_trans_cls, \
         patch("scripts.run_pipeline.FaceAligner") as mock_align_cls, \
         patch("scripts.run_pipeline.FaceParserService") as mock_parse_cls, \
         patch("scripts.run_pipeline.ValidationOrchestrator") as mock_orch_cls, \
         patch("scripts.run_pipeline.FaceAmbiguityValidator") as mock_amb_cls, \
         patch("cv2.imread", return_value=image), \
         patch("cv2.imwrite", return_value=True) as mock_imwrite:

        selected_face = create_face()
        selection_result = _make_selection_result(selected_face)
        mock_det_cls.return_value.detect.return_value = [MagicMock()]
        mock_sel_cls.return_value.select.return_value = selection_result
        mock_amb_cls.return_value.validate.return_value = _make_ambiguity_metric()
        mock_crop_cls.return_value.crop.return_value = MagicMock(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            crop_x=10,
            crop_y=20,
        )
        mock_trans_cls.return_value.transform.return_value = MagicMock()
        mock_align_cls.return_value.align.return_value = MagicMock(
            aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
            aligned_face=MagicMock(),
        )
        mock_parse_cls.return_value.parse.return_value = MagicMock()

        mock_validation_result = MagicMock()
        mock_validation_result.is_valid = True
        metric = MagicMock()
        metric.type.name = "BLUR"
        metric.passed = True
        metric.score = 0.95
        metric.message = ""
        mock_validation_result.metrics = [metric]
        mock_orch_cls.return_value.validate.return_value = mock_validation_result

        outcome, elapsed = process_image(
            img_path=img_file,
            aligned_dir=aligned_dir,
            cropped_dir=cropped_dir,
            reports_dir=reports_dir,
        )

    assert outcome == "valid"
    assert isinstance(elapsed, int)
    assert (reports_dir / "person01.txt").exists()
    report_content = (reports_dir / "person01.txt").read_text(encoding="utf-8")
    assert "Overall Result: VALID" in report_content
    assert "BLUR" in report_content
    assert mock_imwrite.call_count >= 2
    mock_amb_cls.return_value.validate.assert_called_once_with(selection_result)
    mock_crop_cls.return_value.crop.assert_called_once_with(image, selected_face)


def test_process_image_saving_happens_even_if_parser_fails(tmp_path):
    """Verify cropped and aligned images are saved even if FaceParserService raises an exception."""
    img_file = tmp_path / "person02.jpg"
    img_file.write_bytes(b"dummy")

    aligned_dir = tmp_path / "outputs" / "aligned"
    cropped_dir = tmp_path / "outputs" / "cropped"
    reports_dir = tmp_path / "outputs" / "reports"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with patch("scripts.run_pipeline.FaceDetector") as mock_det_cls, \
         patch("scripts.run_pipeline.FaceSelector") as mock_sel_cls, \
         patch("scripts.run_pipeline.FaceCropper") as mock_crop_cls, \
         patch("scripts.run_pipeline.FaceCoordinateTransformer") as mock_trans_cls, \
         patch("scripts.run_pipeline.FaceAligner") as mock_align_cls, \
         patch("scripts.run_pipeline.FaceParserService") as mock_parse_cls, \
         patch("scripts.run_pipeline.FaceAmbiguityValidator") as mock_amb_cls, \
         patch("cv2.imread", return_value=np.zeros((200, 200, 3), dtype=np.uint8)), \
         patch("cv2.imwrite", return_value=True) as mock_imwrite:

        mock_det_cls.return_value.detect.return_value = [MagicMock()]
        mock_sel_cls.return_value.select.return_value = _make_selection_result()
        mock_amb_cls.return_value.validate.return_value = _make_ambiguity_metric()
        mock_crop_cls.return_value.crop.return_value = MagicMock(
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            crop_x=10,
            crop_y=20,
        )
        mock_trans_cls.return_value.transform.return_value = MagicMock()
        mock_align_cls.return_value.align.return_value = MagicMock(
            aligned_image=np.zeros((112, 112, 3), dtype=np.uint8),
            aligned_face=MagicMock(),
        )
        mock_parse_cls.return_value.parse.side_effect = RuntimeError("Parser error")

        outcome, elapsed = process_image(
            img_path=img_file,
            aligned_dir=aligned_dir,
            cropped_dir=cropped_dir,
            reports_dir=reports_dir,
        )

    assert outcome == "processing_error"
    assert mock_imwrite.call_count >= 2
    report_path = reports_dir / "person02.txt"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Overall Result: PROCESSING_ERROR" in content
    assert "Pipeline Execution Failure" in content


def test_process_image_stops_before_crop_when_face_selection_is_ambiguous(tmp_path):
    """Verify ambiguous selection is reported as invalid without downstream Face usage."""
    img_file = tmp_path / "ambiguous.jpg"
    img_file.write_bytes(b"dummy")

    aligned_dir = tmp_path / "outputs" / "aligned"
    cropped_dir = tmp_path / "outputs" / "cropped"
    reports_dir = tmp_path / "outputs" / "reports"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with patch("scripts.run_pipeline.FaceDetector") as mock_det_cls, \
         patch("scripts.run_pipeline.FaceSelector") as mock_sel_cls, \
         patch("scripts.run_pipeline.FaceCropper") as mock_crop_cls, \
         patch("scripts.run_pipeline.FaceCoordinateTransformer") as mock_trans_cls, \
         patch("scripts.run_pipeline.FaceAligner") as mock_align_cls, \
         patch("scripts.run_pipeline.FaceParserService") as mock_parse_cls, \
         patch("scripts.run_pipeline.ValidationOrchestrator") as mock_orch_cls, \
         patch("scripts.run_pipeline.FaceAmbiguityValidator") as mock_amb_cls, \
         patch("cv2.imread", return_value=np.zeros((200, 200, 3), dtype=np.uint8)), \
         patch("cv2.imwrite", return_value=True) as mock_imwrite:

        selection_result = _make_selection_result()
        mock_det_cls.return_value.detect.return_value = [MagicMock()]
        mock_sel_cls.return_value.select.return_value = selection_result
        mock_amb_cls.return_value.validate.return_value = _make_ambiguity_metric(
            passed=False
        )

        outcome, elapsed = process_image(
            img_path=img_file,
            aligned_dir=aligned_dir,
            cropped_dir=cropped_dir,
            reports_dir=reports_dir,
        )

    assert outcome == "invalid"
    assert isinstance(elapsed, int)
    mock_amb_cls.return_value.validate.assert_called_once_with(selection_result)
    mock_crop_cls.return_value.crop.assert_not_called()
    mock_trans_cls.return_value.transform.assert_not_called()
    mock_align_cls.return_value.align.assert_not_called()
    mock_parse_cls.return_value.parse.assert_not_called()
    mock_orch_cls.return_value.validate.assert_not_called()
    mock_imwrite.assert_not_called()
    content = (reports_dir / "ambiguous.txt").read_text(encoding="utf-8")
    assert "Overall Result: INVALID" in content
    assert "FACE_AMBIGUITY" in content


def test_process_image_image_load_failure(tmp_path):
    """Verify cv2.imread returning None results in PROCESSING_ERROR."""
    img_file = tmp_path / "bad.jpg"
    img_file.write_bytes(b"dummy")

    aligned_dir = tmp_path / "outputs" / "aligned"
    cropped_dir = tmp_path / "outputs" / "cropped"
    reports_dir = tmp_path / "outputs" / "reports"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with patch("cv2.imread", return_value=None):
        outcome, elapsed = process_image(
            img_path=img_file,
            aligned_dir=aligned_dir,
            cropped_dir=cropped_dir,
            reports_dir=reports_dir,
        )

    assert outcome == "processing_error"
    report_path = reports_dir / "bad.txt"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Overall Result: PROCESSING_ERROR" in content
    assert "Image Loading Failure" in content
