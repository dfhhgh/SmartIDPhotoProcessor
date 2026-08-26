"""Phase 4 production parser integration tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from config.parser_mode import ParserMode
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
from models.validation_execution_mode import ValidationExecutionMode
from models.validation_metric import ValidationMetric
from models.validation_stage import ValidationStage
from models.validation_type import ValidationType
from pipeline.validation_orchestrator import ValidationOrchestrator
from services.face_parser_service import (
    CLASS_MAP_19_TO_6,
    CLASS_MAP_6_TO_19,
    TARGET_CLASSES_19,
    EyeBrowRefinementFusion,
    EyeBrowRefinementService,
    FaceParserError,
    FaceParserService,
    construct_eye_brow_roi,
    map_19_to_6_numpy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONNX_PATH = PROJECT_ROOT / "ai_models" / "bisenet" / "bisenet_resnet18.onnx"
AUX_CHECKPOINT = (
    PROJECT_ROOT
    / "dataset_builder"
    / "dataset"
    / "parser_finetune_current"
    / "training_aux_eye_brow_phase1"
    / "checkpoints"
    / "best.pt"
)


@pytest.fixture(autouse=True)
def reset_parser_singleton():
    FaceParserService._instance = None
    FaceParserService._initialized = False
    yield
    FaceParserService._instance = None
    FaceParserService._initialized = False


def _hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _logits(classes: np.ndarray, n_classes: int) -> torch.Tensor:
    logits = torch.zeros((n_classes, *classes.shape), dtype=torch.float32)
    for cls in range(n_classes):
        logits[cls][classes == cls] = 10.0
    return logits


def test_1_original_parser_mode_still_uses_onnx_path():
    service = FaceParserService(parser_mode=ParserMode.ORIGINAL)
    session = MagicMock()
    raw = np.zeros((1, len(FacePart), 8, 8), dtype=np.float32)

    with (
        patch.object(service, "_ensure_loaded", return_value=session) as ensure_loaded,
        patch.object(service, "_run_inference", return_value=raw) as run_inference,
    ):
        result = service.parse(np.zeros((8, 8, 3), dtype=np.uint8))

    ensure_loaded.assert_called_once()
    run_inference.assert_called_once()
    assert isinstance(result, FaceParsingResult)


def test_2_fused_parser_mode_uses_configured_checkpoint_without_onnx_session(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.touch()
    refinement = MagicMock()
    refinement.refine.return_value = np.zeros((8, 8), dtype=np.int32)

    service = FaceParserService(
        parser_mode=ParserMode.FUSED,
        refinement_service=refinement,
    )
    with patch.object(service, "_ensure_loaded") as ensure_loaded:
        result = service.parse(np.zeros((8, 8, 3), dtype=np.uint8))

    ensure_loaded.assert_not_called()
    refinement.refine.assert_called_once()
    assert isinstance(result, FaceParsingResult)


def test_3_auxiliary_checkpoint_hash_is_phase1_best():
    assert AUX_CHECKPOINT.exists()
    assert _hash(AUX_CHECKPOINT).startswith("961e08bf64fdd0b8")


def test_4_production_onnx_hash_is_unchanged():
    assert ONNX_PATH.exists()
    assert _hash(ONNX_PATH).startswith("2218b6183c26ca5c")


def test_5_auxiliary_service_rejects_cpu_device():
    service = EyeBrowRefinementService(
        onnx_model_path=ONNX_PATH,
        checkpoint_path=AUX_CHECKPOINT,
        fusion=EyeBrowRefinementFusion(),
        device=torch.device("cpu"),
    )
    with pytest.raises(RuntimeError, match="cuda:0"):
        service._resolve_device()


def test_6_correct_19_to_6_mapping():
    arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 18], dtype=np.int64)
    mapped = map_19_to_6_numpy(arr)
    assert np.array_equal(mapped, np.array([0, 0, 1, 2, 3, 4, 5, 0, 0]))
    assert CLASS_MAP_19_TO_6 == {2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    assert CLASS_MAP_6_TO_19 == {1: 2, 2: 3, 3: 4, 4: 5, 5: 6}


def test_7_aux_background_cannot_overwrite_original_target_class():
    original = np.full((6, 6), int(FacePart.LEFT_EYE), dtype=np.int64)
    aux = np.zeros((6, 6), dtype=np.int64)
    final, stats = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    assert np.array_equal(final, original)
    assert stats.corrections_accepted == 0


def test_8_only_target_classes_can_be_modified():
    """Non-target classes (SKIN=1, NOSE=10, HAIR=17) must NOT be overwritten.
    Only original target class pixels (LEFT_EYE=4) may be refined."""
    original = np.array([[1, 10], [17, 4]], dtype=np.int64)
    aux = np.array([[3, 3], [3, 3]], dtype=np.int64)  # aux class 3 → LEFT_EYE (19-class 4)
    final, _ = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    # Non-target classes preserved
    assert final[0, 0] == 1   # SKIN preserved
    assert final[0, 1] == 10  # NOSE preserved
    assert final[1, 0] == 17  # HAIR preserved
    # Target class may be refined (aux agrees → stays LEFT_EYE=4)
    assert final[1, 1] == 4   # LEFT_EYE stays LEFT_EYE
    assert set(np.unique(final)).issubset({1, 4, 10, 17})


def test_9_roi_restriction_returns_false_when_no_target_present():
    pred_19 = np.full((4, 4), int(FacePart.SKIN), dtype=np.int64)
    pred_aux_19 = np.zeros((4, 4), dtype=np.int64)
    assert not construct_eye_brow_roi(pred_19, pred_aux_19).any()


def test_10_non_target_classes_remain_identical_when_aux_is_background():
    original = np.array([[1, 10, 17], [14, 16, 18]], dtype=np.int64)
    aux = np.zeros_like(original)
    final, _ = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    assert np.array_equal(final, original)


def test_11_final_mask_contains_only_valid_face_part_ids():
    final, _ = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        torch.randn(len(FacePart), 12, 12),
        torch.randn(6, 12, 12),
    )
    assert int(final.min()) >= 0
    assert int(final.max()) <= int(FacePart.HAT)


def test_12_final_mask_has_correct_dimensions_after_parse():
    refinement = MagicMock()
    refinement.refine.return_value = np.zeros((7, 9), dtype=np.int32)
    service = FaceParserService(
        parser_mode=ParserMode.FUSED,
        refinement_service=refinement,
    )
    result = service.parse(np.zeros((7, 9, 3), dtype=np.uint8))
    assert result.mask.shape == (7, 9)
    assert result.image_size() == (7, 9)


def test_13_face_parsing_result_accepts_fused_mask():
    mask = np.full((5, 5), int(FacePart.RIGHT_BROW), dtype=np.int32)
    result = FaceParsingResult(mask=mask, image_height=5, image_width=5)
    assert result.has_part(FacePart.RIGHT_BROW)


def test_14_existing_validator_contract_accepts_fused_result():
    validator = MagicMock()
    validator.stage = ValidationStage.PARSING
    validator.validate.return_value = ValidationMetric(
        type=ValidationType.FACE_VISIBILITY,
        passed=True,
        score=1.0,
    )
    parsing = FaceParsingResult(np.zeros((5, 5), dtype=np.int32), 5, 5)
    orchestrator = ValidationOrchestrator(validators=(validator,))
    orchestrator.validate(np.zeros((5, 5, 3), dtype=np.uint8), parsing_result=parsing)
    validator.validate.assert_called_once()


def test_15_validation_orchestrator_lazy_behavior_is_unchanged():
    parser = MagicMock()
    validator = MagicMock()
    validator.stage = ValidationStage.CHEAP
    validator.validate.return_value = ValidationMetric(
        type=ValidationType.BLUR,
        passed=False,
        score=0.0,
    )
    orchestrator = ValidationOrchestrator(
        validators=(validator,),
        parser_service=parser,
        execution_mode=ValidationExecutionMode.PRODUCTION,
    )
    orchestrator.validate(np.zeros((5, 5, 3), dtype=np.uint8))
    parser.parse.assert_not_called()


def test_16_development_mode_remains_functional_with_injected_parser():
    parser = MagicMock()
    parsing = FaceParsingResult(np.zeros((5, 5), dtype=np.int32), 5, 5)
    parser.parse.return_value = parsing
    validator = MagicMock()
    validator.stage = ValidationStage.PARSING
    validator.validate.return_value = ValidationMetric(
        type=ValidationType.OCCLUSION,
        passed=True,
        score=1.0,
    )
    orchestrator = ValidationOrchestrator(
        validators=(validator,),
        parser_service=parser,
        execution_mode=ValidationExecutionMode.DEVELOPMENT,
    )
    orchestrator.validate(np.zeros((5, 5, 3), dtype=np.uint8))
    parser.parse.assert_not_called()
    validator.validate.assert_called_once()


def test_17_dependency_injection_selects_fused_refinement_service():
    refinement = MagicMock()
    refinement.refine.return_value = np.zeros((5, 5), dtype=np.int32)
    service = FaceParserService(ParserMode.FUSED, refinement_service=refinement)
    service.parse(np.zeros((5, 5, 3), dtype=np.uint8))
    refinement.refine.assert_called_once()


def test_18_original_mode_is_deterministic_with_same_logits():
    service = FaceParserService(parser_mode=ParserMode.ORIGINAL)
    raw = np.random.default_rng(42).normal(size=(1, len(FacePart), 6, 6)).astype(np.float32)
    first = service._postprocess(raw, 6, 6).mask
    second = service._postprocess(raw, 6, 6).mask
    assert np.array_equal(first, second)


def test_19_fused_mode_is_deterministic_with_same_logits():
    log19 = torch.randn(len(FacePart), 10, 10)
    logaux = torch.randn(6, 10, 10)
    fusion = EyeBrowRefinementFusion(strategy=1, threshold=0.0)
    first, _ = fusion.apply(log19, logaux)
    second, _ = fusion.apply(log19, logaux)
    assert np.array_equal(first, second)


def test_20_repeated_fused_parse_produces_equivalent_masks():
    refinement = MagicMock()
    mask = np.full((4, 4), int(FacePart.LEFT_BROW), dtype=np.int32)
    refinement.refine.return_value = mask
    service = FaceParserService(ParserMode.FUSED, refinement_service=refinement)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    first = service.parse(image).mask
    second = service.parse(image).mask
    assert np.array_equal(first, second)


def test_21_fused_failures_are_reported_as_face_parser_error():
    refinement = MagicMock()
    refinement.refine.side_effect = RuntimeError("CUDA unavailable")
    service = FaceParserService(ParserMode.FUSED, refinement_service=refinement)
    with pytest.raises(FaceParserError, match="Fused face-parsing inference failed"):
        service.parse(np.zeros((4, 4, 3), dtype=np.uint8))


def test_22_non_target_skin_not_overwritten_when_aux_predicts_target():
    """Regression: aux predicting LEFT_EYE on a SKIN pixel must not overwrite SKIN."""
    original = np.full((4, 4), int(FacePart.SKIN), dtype=np.int64)
    aux = np.full((4, 4), 3, dtype=np.int64)  # aux class 3 = LEFT_EYE
    final, stats = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    assert np.array_equal(final, original), (
        "Non-target SKIN class was overwritten by auxiliary LEFT_EYE prediction"
    )
    assert stats.corrections_accepted == 0


def test_23_non_target_hair_not_overwritten_when_aux_predicts_brow():
    """Regression: aux predicting LEFT_BROW on a HAIR pixel must not overwrite HAIR."""
    original = np.full((4, 4), int(FacePart.HAIR), dtype=np.int64)
    aux = np.full((4, 4), 1, dtype=np.int64)  # aux class 1 = LEFT_BROW
    final, stats = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    assert np.array_equal(final, original), (
        "Non-target HAIR class was overwritten by auxiliary LEFT_BROW prediction"
    )
    assert stats.corrections_accepted == 0


def test_24_non_target_hat_not_overwritten_when_aux_predicts_eyeglass():
    """Regression: aux predicting EYE_GLASS on a HAT pixel must not overwrite HAT."""
    original = np.full((4, 4), int(FacePart.HAT), dtype=np.int64)
    aux = np.full((4, 4), 5, dtype=np.int64)  # aux class 5 = EYE_GLASS
    final, stats = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    assert np.array_equal(final, original), (
        "Non-target HAT class was overwritten by auxiliary EYE_GLASS prediction"
    )
    assert stats.corrections_accepted == 0


def test_25_original_target_refined_when_aux_disagrees():
    """Aux predicting LEFT_BROW where original has LEFT_EYE should refine within target classes."""
    original = np.full((4, 4), int(FacePart.LEFT_EYE), dtype=np.int64)
    aux = np.full((4, 4), 1, dtype=np.int64)  # aux class 1 = LEFT_BROW
    final, _ = EyeBrowRefinementFusion(strategy=1, threshold=0.0).apply(
        _logits(original, len(FacePart)),
        _logits(aux, 6),
    )
    assert (final == int(FacePart.LEFT_BROW)).all()
    assert set(np.unique(final)).issubset(TARGET_CLASSES_19)
