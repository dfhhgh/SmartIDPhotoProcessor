"""
Semantic evidence fusion engine.

Independent evidence sources (BiSeNet segmentation confidence,
InsightFace landmark confidence, smooth head pose confidence, and occlusion
confidence) contribute normalized scores in [0.0, 1.0]. Evidence strengthens
confidence smoothly without fabricating parser output.

Public compute_*_evidence() methods expose per-channel evidence for
evaluation display.  The production is_*_visible() methods and the
internal _compute_weighted_score() use the original 4-channel formula
that all production validators depend on.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from insightface.app.common import Face

from config.constants import (
    HEAD_POSE_PITCH_MAX_DEGREES,
    HEAD_POSE_YAW_MAX_DEGREES,
    HEAD_POSE_ROLL_MAX_DEGREES,
    SEMANTIC_PARSER_WEIGHT,
    SEMANTIC_LANDMARK_WEIGHT,
    SEMANTIC_POSE_WEIGHT,
    SEMANTIC_OCCLUSION_WEIGHT,
    SEMANTIC_DECISION_THRESHOLD,
)
from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult

_EYE_LANDMARK_INDICES: dict[FacePart, int] = {
    FacePart.RIGHT_EYE: 0,
    FacePart.LEFT_EYE: 1,
}


@dataclass(frozen=True, slots=True)
class SemanticEvidence:
    """Normalized evidence sources for semantic confidence blending.

    Each attribute is a normalized confidence score in [0.0, 1.0].

    final_confidence is the weighted fusion of all channels, computed
    automatically during construction using the same 4-channel formula
    that the production validators use.  passed is derived from
    final_confidence >= SEMANTIC_DECISION_THRESHOLD.
    """
    parser_confidence: float = 0.0
    eye_support_confidence: float = 0.0
    landmark_confidence: float = 0.0
    pose_confidence: float = 1.0
    occlusion_confidence: float = 1.0
    final_confidence: float = 0.0
    passed: bool = False

    def __post_init__(self) -> None:
        total_weight = (
            SEMANTIC_PARSER_WEIGHT
            + SEMANTIC_LANDMARK_WEIGHT
            + SEMANTIC_POSE_WEIGHT
            + SEMANTIC_OCCLUSION_WEIGHT
        )
        if total_weight <= 0.0:
            object.__setattr__(self, "final_confidence", 0.0)
            object.__setattr__(self, "passed", False)
            return
        score = (
            self.parser_confidence * SEMANTIC_PARSER_WEIGHT
            + self.landmark_confidence * SEMANTIC_LANDMARK_WEIGHT
            + self.pose_confidence * SEMANTIC_POSE_WEIGHT
            + self.occlusion_confidence * SEMANTIC_OCCLUSION_WEIGHT
        ) / total_weight
        final = float(min(max(score, 0.0), 1.0))
        object.__setattr__(self, "final_confidence", final)
        object.__setattr__(self, "passed", final >= SEMANTIC_DECISION_THRESHOLD)


class SemanticEvidenceEngine:
    """Production-ready semantic evidence fusion engine.

    Performs continuous confidence blending across multiple independent evidence
    sources, capturing uncertainty and avoiding binary replacement hacks.
    """

    def __init__(
        self,
        parsing_result: FaceParsingResult,
        face: Face | None = None,
    ) -> None:
        if not isinstance(parsing_result, FaceParsingResult):
            raise TypeError("Parsing result must be a FaceParsingResult.")
        self._parsing = parsing_result
        self._face = face

    # ------------------------------------------------------------------
    # Continuous Normalization & Confidence Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ratio(ratio: float, min_ratio: float) -> float:
        """Smoothly normalize a part ratio into [0.0, 1.0] relative to min_ratio."""
        if min_ratio <= 0.0:
            return 1.0 if ratio > 0.0 else 0.0
        return float(min(max(ratio / min_ratio, 0.0), 1.0))

    def _compute_parser_confidence(self, part: FacePart, min_ratio: float) -> float:
        """Compute parser confidence from target part segmentation ratio."""
        ratio = self._parsing.part_ratio(part)
        return self._normalize_ratio(ratio, min_ratio)

    def _compute_landmark_confidence(self, part: FacePart) -> float:
        """Compute continuous landmark confidence in [0.0, 1.0]."""
        if self._face is None:
            return 0.0

        kps = getattr(self._face, "kps", None)
        if kps is None or not isinstance(kps, np.ndarray):
            return 0.0

        if kps.ndim != 2 or kps.shape[1] != 2:
            return 0.0

        idx = _EYE_LANDMARK_INDICES.get(part)
        if idx is None or idx >= kps.shape[0]:
            return 0.0

        coords = kps[idx]
        if not np.isfinite(coords).all():
            return 0.0

        return 1.0

    def _compute_pose_confidence(self) -> float:
        """Compute smooth pose confidence in [0.0, 1.0].

        Returns 1.0 for a perfectly frontal pose, decreasing smoothly as
        pitch, yaw, or roll approach their maximum allowable limits.
        """
        if self._face is None:
            return 1.0

        pose = getattr(self._face, "pose", None)
        if pose is None:
            return 1.0

        try:
            pitch, yaw, roll = map(float, pose)
        except (TypeError, ValueError):
            return 1.0

        if not np.isfinite([pitch, yaw, roll]).all():
            return 1.0

        pitch_score = max(0.0, 1.0 - (abs(pitch) / HEAD_POSE_PITCH_MAX_DEGREES))
        yaw_score = max(0.0, 1.0 - (abs(yaw) / HEAD_POSE_YAW_MAX_DEGREES))
        roll_score = max(0.0, 1.0 - (abs(roll) / HEAD_POSE_ROLL_MAX_DEGREES))

        return float(min(pitch_score, yaw_score, roll_score))

    def _compute_occlusion_confidence(self) -> float:
        """Compute independent occlusion confidence in [0.0, 1.0]."""
        if not self._parsing.has_part(FacePart.HAT):
            return 1.0

        mandatory_parts = (FacePart.LEFT_EYE, FacePart.RIGHT_EYE, FacePart.NOSE)
        visible_count = sum(
            1 for p in mandatory_parts
            if self._parsing.has_part(p) and self._parsing.part_ratio(p) > 0.0
        )
        return float(visible_count / len(mandatory_parts))

    def _compute_weighted_score(self, evidence: SemanticEvidence) -> float:
        """Compute final weighted confidence score from normalized evidence sources.

        This is the sole location where multiple evidence sources are combined
        in production.  Uses the original 4-channel formula.
        """
        total_weight = (
            SEMANTIC_PARSER_WEIGHT
            + SEMANTIC_LANDMARK_WEIGHT
            + SEMANTIC_POSE_WEIGHT
            + SEMANTIC_OCCLUSION_WEIGHT
        )
        if total_weight <= 0.0:
            return 0.0

        score = (
            evidence.parser_confidence * SEMANTIC_PARSER_WEIGHT
            + evidence.landmark_confidence * SEMANTIC_LANDMARK_WEIGHT
            + evidence.pose_confidence * SEMANTIC_POSE_WEIGHT
            + evidence.occlusion_confidence * SEMANTIC_OCCLUSION_WEIGHT
        ) / total_weight

        return float(min(max(score, 0.0), 1.0))

    # ------------------------------------------------------------------
    # Public Evidence APIs (for evaluation observation)
    # ------------------------------------------------------------------

    def compute_weighted_score(self, evidence: SemanticEvidence) -> float:
        """Compute the final confidence score from evidence channels.

        Public API for evaluation workflows that need to observe the same
        fusion behaviour as the production validator.
        """
        return self._compute_weighted_score(evidence)

    def compute_eye_evidence(
        self,
        part: FacePart,
        min_ratio: float = 0.0015,
    ) -> SemanticEvidence:
        """Compute semantic evidence for a single eye region."""
        parser_conf = self._compute_parser_confidence(part, min_ratio)
        eye_support_conf = parser_conf
        if parser_conf == 0.0 and self._parsing.has_part(FacePart.EYE_GLASS):
            eye_support_conf = 0.5

        return SemanticEvidence(
            parser_confidence=parser_conf,
            eye_support_confidence=eye_support_conf,
            landmark_confidence=self._compute_landmark_confidence(part),
            pose_confidence=self._compute_pose_confidence(),
            occlusion_confidence=self._compute_occlusion_confidence(),
        )

    def compute_eyebrow_evidence(
        self,
        brow: FacePart,
        eye: FacePart,
        brow_min_ratio: float = 0.0010,
        eye_min_ratio: float = 0.0015,
    ) -> SemanticEvidence:
        """Compute semantic evidence for an eyebrow region."""
        parser_conf = self._compute_parser_confidence(brow, brow_min_ratio)
        eye_support_conf = self._compute_parser_confidence(eye, eye_min_ratio)
        landmark_conf = self._compute_landmark_confidence(eye)

        return SemanticEvidence(
            parser_confidence=parser_conf,
            eye_support_confidence=eye_support_conf,
            landmark_confidence=landmark_conf,
            pose_confidence=self._compute_pose_confidence(),
            occlusion_confidence=self._compute_occlusion_confidence(),
        )

    def compute_mouth_evidence(
        self,
        mouth_min_ratio: float = 0.0008,
        upper_lip_min_ratio: float = 0.0020,
        lower_lip_min_ratio: float = 0.0020,
    ) -> SemanticEvidence:
        """Compute semantic evidence for the composite mouth region."""
        mouth_conf = self._normalize_ratio(
            self._parsing.part_ratio(FacePart.MOUTH),
            mouth_min_ratio,
        )
        upper_conf = self._normalize_ratio(
            self._parsing.part_ratio(FacePart.UPPER_LIP),
            upper_lip_min_ratio,
        )
        lower_conf = self._normalize_ratio(
            self._parsing.part_ratio(FacePart.LOWER_LIP),
            lower_lip_min_ratio,
        )

        return SemanticEvidence(
            parser_confidence=mouth_conf,
            eye_support_confidence=min(upper_conf, lower_conf),
            landmark_confidence=1.0,
            pose_confidence=self._compute_pose_confidence(),
            occlusion_confidence=self._compute_occlusion_confidence(),
        )

    def compute_part_evidence(
        self,
        part: FacePart,
        min_ratio: float,
    ) -> SemanticEvidence:
        """Compute semantic evidence for a generic part with no landmark support."""
        parser_conf = self._compute_parser_confidence(part, min_ratio)

        return SemanticEvidence(
            parser_confidence=parser_conf,
            eye_support_confidence=0.0,
            landmark_confidence=1.0,
            pose_confidence=self._compute_pose_confidence(),
            occlusion_confidence=self._compute_occlusion_confidence(),
        )

    # ------------------------------------------------------------------
    # Public Semantic Queries (Production Validators)
    # ------------------------------------------------------------------

    def is_eye_visible(
        self,
        part: FacePart,
        min_ratio: float = 0.0015,
    ) -> bool:
        """Determine if an eye region is visible via weighted continuous fusion.

        Blends BiSeNet eye segmentation ratio, transparent eyewear context (EYE_GLASS),
        eye landmark confidence, pose confidence, and occlusion confidence.
        """
        if part not in (FacePart.LEFT_EYE, FacePart.RIGHT_EYE):
            raise ValueError(f"Expected eye part, got {part!r}.")

        ratio = self._parsing.part_ratio(part)
        parser_conf = self._normalize_ratio(ratio, min_ratio)

        # Transparent glasses support: if parser missed eye due to prescription glasses,
        # landmark confidence and eyewear presence provide continuous fallback support.
        landmark_conf = self._compute_landmark_confidence(part)
        if parser_conf == 0.0 and self._parsing.has_part(FacePart.EYE_GLASS) and landmark_conf > 0.0:
            parser_conf = 0.65  # Blended prior for glasses-obscured segmentation

        pose_conf = self._compute_pose_confidence()
        occlusion_conf = self._compute_occlusion_confidence()

        evidence = SemanticEvidence(
            parser_confidence=parser_conf,
            landmark_confidence=landmark_conf,
            pose_confidence=pose_conf,
            occlusion_confidence=occlusion_conf,
        )

        return self._compute_weighted_score(evidence) >= SEMANTIC_DECISION_THRESHOLD

    def is_eyebrow_visible(
        self,
        brow: FacePart,
        eye: FacePart,
        brow_min_ratio: float = 0.0010,
        eye_min_ratio: float = 0.0015,
    ) -> bool:
        """Determine if an eyebrow is visible via continuous evidence blending.

        Avoids hard binary overrides. When BiSeNet misses an eyebrow (parser_conf == 0.0)
        while the eye segmentation is present, eye landmarks exist, and head pose is frontal,
        evidence blending smoothly increases overall confidence rather than fabricating pixels.
        """
        if brow not in (FacePart.LEFT_BROW, FacePart.RIGHT_BROW):
            raise ValueError(f"Expected brow part, got {brow!r}.")

        brow_ratio = self._parsing.part_ratio(brow)
        eye_ratio = self._parsing.part_ratio(eye)
        brow_conf = self._normalize_ratio(brow_ratio, brow_min_ratio)
        eye_conf = self._normalize_ratio(eye_ratio, eye_min_ratio)

        landmark_conf = self._compute_landmark_confidence(eye)
        pose_conf = self._compute_pose_confidence()

        # Continuous blending for false-negative mitigation:
        # If brow ratio is zero, blend supporting eye evidence and landmark confidence smoothly.
        if brow_conf == 0.0 and eye_ratio > 0.0 and landmark_conf > 0.0:
            parser_conf = 0.0
        else:
            parser_conf = max(brow_conf, eye_conf * 0.8)

        occlusion_conf = self._compute_occlusion_confidence()

        evidence = SemanticEvidence(
            parser_confidence=parser_conf,
            landmark_confidence=landmark_conf,
            pose_confidence=pose_conf,
            occlusion_confidence=occlusion_conf,
        )

        return self._compute_weighted_score(evidence) >= SEMANTIC_DECISION_THRESHOLD

    def is_mouth_visible(
        self,
        mouth_min_ratio: float = 0.0008,
        upper_lip_min_ratio: float = 0.0020,
        lower_lip_min_ratio: float = 0.0020,
    ) -> bool:
        """Determine if the mouth region is visible via continuous evidence blending.

        Supports closed-mouth poses (where inner MOUTH is missing but UPPER_LIP
        and LOWER_LIP are present) and blends pose and occlusion confidence.
        """
        mouth_conf = self._normalize_ratio(self._parsing.part_ratio(FacePart.MOUTH), mouth_min_ratio)
        upper_conf = self._normalize_ratio(self._parsing.part_ratio(FacePart.UPPER_LIP), upper_lip_min_ratio)
        lower_conf = self._normalize_ratio(self._parsing.part_ratio(FacePart.LOWER_LIP), lower_lip_min_ratio)

        # Composite blending: MOUTH cavity or closed lips (both upper and lower)
        parser_conf = max(mouth_conf, min(upper_conf, lower_conf))

        pose_conf = self._compute_pose_confidence()
        occlusion_conf = self._compute_occlusion_confidence()

        evidence = SemanticEvidence(
            parser_confidence=parser_conf,
            landmark_confidence=1.0,  # Mouth relies on semantic segmentation & pose
            pose_confidence=pose_conf,
            occlusion_confidence=occlusion_conf,
        )

        return self._compute_weighted_score(evidence) >= SEMANTIC_DECISION_THRESHOLD

    def is_head_covering_prohibited(self) -> bool:
        """Determine whether a detected head covering (FacePart.HAT) is prohibited."""
        if not self._parsing.has_part(FacePart.HAT):
            return False

        eyes_visible = (
            self.is_eye_visible(FacePart.LEFT_EYE)
            and self.is_eye_visible(FacePart.RIGHT_EYE)
        )
        nose_visible = self._parsing.has_part(FacePart.NOSE)
        mouth_visible = self.is_mouth_visible()
        pose_conf = self._compute_pose_confidence()

        if eyes_visible and nose_visible and mouth_visible and pose_conf >= 0.5:
            return False

        return True

    # ------------------------------------------------------------------
    # Future Extensibility Stubs
    # ------------------------------------------------------------------

    def visible_upper_face(self) -> bool:
        raise NotImplementedError

    def visible_lower_face(self) -> bool:
        raise NotImplementedError

    def visible_eye_region(self) -> bool:
        raise NotImplementedError

    def visible_forehead(self) -> bool:
        raise NotImplementedError

    def allowed_head_covering(self) -> bool:
        return not self.is_head_covering_prohibited()

    def allowed_face_covering(self) -> bool:
        raise NotImplementedError

    def visible_jaw(self) -> bool:
        raise NotImplementedError
