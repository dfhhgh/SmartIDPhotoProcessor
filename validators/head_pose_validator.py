"""
Head pose validator.
"""

import numpy as np
from insightface.app.common import Face

from config.constants import (
    HEAD_POSE_PITCH_MAX_DEGREES,
    HEAD_POSE_ROLL_MAX_DEGREES,
    HEAD_POSE_YAW_MAX_DEGREES,
)
from models.validation_metric import ValidationMetric
from models.validation_type import ValidationType
from validators.base_validator import BaseValidator


class HeadPoseValidator(BaseValidator):
    """Validates whether the detected head pose is acceptable for ID processing.

    Pose is read directly from ``face.pose`` as provided by InsightFace and
    is expected to contain ``(pitch, yaw, roll)`` expressed in degrees. This
    validator does not estimate pose itself; it only interprets the values
    already produced by the upstream face detection model.
    """

    def validate(
        self,
        image: np.ndarray,
        face: Face | None = None,
    ) -> ValidationMetric:
        """Validate head pose using pitch, yaw, and roll angles.

        Args:
            image: Image data to validate.
            face: Detected face with pose information.

        Returns:
            A ValidationMetric containing a quality score clamped to the
            range [0.0, 1.0], where 1.0 indicates a perfectly frontal pose.

        Raises:
            TypeError: If image is not a NumPy array, or face.pose does not
                contain three finite numeric values.
            ValueError: If image is None, empty, face is None, or face has
                no usable pose attribute.
        """
        if image is None:
            raise ValueError(
                "Image must not be None."
            )

        if not isinstance(
            image,
            np.ndarray,
        ):
            raise TypeError(
                "Image must be a numpy array."
            )

        if image.size == 0:
            raise ValueError(
                "Image must not be empty."
            )

        if face is None:
            raise ValueError(
                "Face must not be None."
            )

        pitch, yaw, roll = self._extract_pose(
            face=face,
        )

        pitch_valid = abs(pitch) <= HEAD_POSE_PITCH_MAX_DEGREES
        yaw_valid = abs(yaw) <= HEAD_POSE_YAW_MAX_DEGREES
        roll_valid = abs(roll) <= HEAD_POSE_ROLL_MAX_DEGREES

        passed = pitch_valid and yaw_valid and roll_valid
        score = self._compute_score(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
        )
        message = self._build_message(
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            pitch_valid=pitch_valid,
            yaw_valid=yaw_valid,
            roll_valid=roll_valid,
        )

        return ValidationMetric(
            type=ValidationType.HEAD_POSE,
            passed=passed,
            score=score,
            message=message,
        )

    def _extract_pose(
        self,
        face: Face,
    ) -> tuple[float, float, float]:
        """Extract and validate the (pitch, yaw, roll) pose from *face*.

        Args:
            face: Detected face expected to expose a ``pose`` attribute.

        Returns:
            A tuple of ``(pitch, yaw, roll)`` as native Python floats,
            expressed in degrees.

        Raises:
            ValueError: If face has no ``pose`` attribute, or pose does not
                contain exactly three values.
            TypeError: If any pose value is not numeric, or is not finite.
        """
        pose = getattr(
            face,
            "pose",
            None,
        )

        if pose is None:
            raise ValueError(
                "Face pose must not be None."
            )

        try:
            pitch, yaw, roll = pose
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Face pose must contain exactly three values (pitch, yaw, roll)."
            ) from exc

        try:
            pitch, yaw, roll = float(pitch), float(yaw), float(roll)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Face pose values must be numeric."
            ) from exc

        angles = np.array(
            [pitch, yaw, roll],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(angles)):
            raise TypeError(
                "Face pose values must be finite numbers."
            )

        return pitch, yaw, roll

    def _compute_score(
        self,
        pitch: float,
        yaw: float,
        roll: float,
    ) -> float:
        """Compute an overall pose quality score from the three axes.

        Each axis is scored independently: 1.0 at zero degrees, decreasing
        linearly to 0.5 at its configured threshold and continuing toward
        0.0 for more extreme rotations. The overall score is the average
        of the three per-axis scores, so a single badly rotated axis
        reduces — but does not by itself zero out — the final score.

        Args:
            pitch: Pitch angle in degrees.
            yaw: Yaw angle in degrees.
            roll: Roll angle in degrees.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        pitch_score = self._normalize_angle(
            angle=pitch,
            max_degrees=HEAD_POSE_PITCH_MAX_DEGREES,
        )
        yaw_score = self._normalize_angle(
            angle=yaw,
            max_degrees=HEAD_POSE_YAW_MAX_DEGREES,
        )
        roll_score = self._normalize_angle(
            angle=roll,
            max_degrees=HEAD_POSE_ROLL_MAX_DEGREES,
        )

        return float(
            (pitch_score + yaw_score + roll_score) / 3.0
        )

    def _normalize_angle(
        self,
        angle: float,
        max_degrees: float,
    ) -> float:
        """Normalize a single pose angle into a bounded quality score.

        The score equals 1.0 at zero degrees and decreases linearly to
        0.5 at the configured threshold, continuing toward 0.0 for more
        extreme rotations.

        Args:
            angle: Rotation angle in degrees, measured from frontal (0.0).
            max_degrees: Acceptable threshold for this axis, in degrees.

        Returns:
            Quality score between 0.0 and 1.0.
        """
        score = 1.0 - 0.5 * (abs(angle) / max_degrees)

        return float(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            )
        )

    def _build_message(
        self,
        pitch: float,
        yaw: float,
        roll: float,
        pitch_valid: bool,
        yaw_valid: bool,
        roll_valid: bool,
    ) -> str:
        """Build a human-readable message describing head pose failures.

        Args:
            pitch: Pitch angle in degrees.
            yaw: Yaw angle in degrees.
            roll: Roll angle in degrees.
            pitch_valid: Whether pitch is within its configured threshold.
            yaw_valid: Whether yaw is within its configured threshold.
            roll_valid: Whether roll is within its configured threshold.

        Returns:
            A descriptive message string.
        """
        if pitch_valid and yaw_valid and roll_valid:
            return "Head pose is acceptable."

        issues: list[str] = []

        if not yaw_valid:
            if yaw > 0:
                issues.append("Head is turned too far right.")
            else:
                issues.append("Head is turned too far left.")

        if not pitch_valid:
            if pitch > 0:
                issues.append("Head is tilted too far downward.")
            else:
                issues.append("Head is tilted too far upward.")

        if not roll_valid:
            issues.append("Head roll exceeds the acceptable limit.")

        return " ".join(issues)