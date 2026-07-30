"""Tests for GlassesDetectorClassifier service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from insightface.app.common import Face

from config.constants import (
    GLASSES_EYEGLASSES_PROBABILITY_THRESHOLD,
    GLASSES_SUNGLASSES_PROBABILITY_THRESHOLD,
)
from models.eyewear_prediction import EyewearPrediction
from models.eyewear_type import EyewearType
from services.glasses_detector_classifier import (
    GlassesDetectorClassifier,
    GlassesDetectorError,
)


@pytest.fixture
def classifier() -> GlassesDetectorClassifier:
    """Create a GlassesDetectorClassifier instance."""
    return GlassesDetectorClassifier()


@pytest.fixture
def mock_face() -> Face:
    """Create a mock Face instance with valid bbox."""
    face = Face()
    face.bbox = np.array([10.0, 20.0, 90.0, 110.0], dtype=np.float32)
    return face


@pytest.fixture
def valid_image() -> np.ndarray:
    """Create a valid BGR uint8 test image."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


# ================================================================== #
# 1. Constructor                                                     #
# ================================================================== #


class TestConstructor:
    """Verify initialization and lazy model loading behavior."""

    def test_initializes_without_loading_models(self, classifier):
        """Constructing GlassesDetectorClassifier must not load underlying models immediately."""
        assert classifier._eyeglasses_classifier is None
        assert classifier._sunglasses_classifier is None


# ================================================================== #
# 2. Input validation                                                #
# ================================================================== #


class TestInputValidation:
    """Verify input validation rules in _validate_inputs()."""

    def test_valid_inputs_pass(self, classifier, valid_image, mock_face):
        """Valid uint8 BGR image and Face instance pass validation."""
        classifier._validate_inputs(valid_image, mock_face)

    def test_non_numpy_image_raises_type_error(self, classifier, mock_face):
        """Non-numpy image input raises TypeError."""
        with pytest.raises(TypeError, match="image must be a numpy.ndarray"):
            classifier._validate_inputs("not_an_array", mock_face)

    def test_empty_image_raises_value_error(self, classifier, mock_face):
        """Empty numpy image raises ValueError."""
        empty_img = np.empty((0, 0, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="image must not be empty"):
            classifier._validate_inputs(empty_img, mock_face)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((100, 100), id="2d_grayscale"),
            pytest.param((100, 100, 4), id="4d_bgra"),
        ],
    )
    def test_invalid_image_shape_raises_value_error(self, classifier, mock_face, shape):
        """Images with ndim != 3 or channels != 3 raise ValueError."""
        img = np.zeros(shape, dtype=np.uint8)
        with pytest.raises(ValueError, match="image must have shape"):
            classifier._validate_inputs(img, mock_face)

    def test_invalid_image_dtype_raises_type_error(self, classifier, mock_face):
        """Non-uint8 image dtype raises TypeError."""
        img = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(TypeError, match="image dtype must be uint8"):
            classifier._validate_inputs(img, mock_face)

    def test_none_face_raises_value_error(self, classifier, valid_image):
        """None face input raises ValueError."""
        with pytest.raises(ValueError, match="face must not be None"):
            classifier._validate_inputs(valid_image, None)

    def test_invalid_face_type_raises_type_error(self, classifier, valid_image):
        """Non-Face instance input raises TypeError."""
        with pytest.raises(TypeError, match="face must be a Face instance"):
            classifier._validate_inputs(valid_image, "not_a_face")


# ================================================================== #
# 3. ROI extraction                                                  #
# ================================================================== #


class TestRoiExtraction:
    """Verify face ROI extraction and bounding box handling."""

    def test_valid_crop(self, classifier, valid_image, mock_face):
        """Extracts sub-image matching valid bbox coordinates."""
        mock_face.bbox = np.array([10, 20, 50, 60], dtype=np.float32)
        roi = classifier._extract_face_roi(valid_image, mock_face)
        assert roi.shape == (40, 40, 3)

    def test_extracts_roi_from_matching_aligned_coordinates(self, classifier):
        """Aligned-image coordinates must slice the corresponding aligned image region."""
        aligned_image = np.arange(
            112 * 112 * 3,
            dtype=np.uint16,
        ).reshape(112, 112, 3)
        face = Face()
        face.bbox = np.array([20.0, 30.0, 80.0, 90.0], dtype=np.float32)

        roi = classifier._extract_face_roi(
            aligned_image,
            face,
        )

        expected = aligned_image[30:90, 20:80]
        assert np.array_equal(roi, expected)

    def test_bbox_clamping(self, classifier, valid_image, mock_face):
        """Bounding box coordinates extending beyond image bounds are clamped."""
        mock_face.bbox = np.array([-50, -50, 300, 300], dtype=np.float32)
        roi = classifier._extract_face_roi(valid_image, mock_face)
        assert roi.shape == (200, 200, 3)

    def test_invalid_bbox_coordinates_raise_error(self, classifier, valid_image, mock_face):
        """Non-numeric or malformed bbox coordinates raise GlassesDetectorError."""
        mock_face.bbox = ["invalid", 0, 50, 50]
        with pytest.raises(GlassesDetectorError, match="Face bounding box values are not valid integers"):
            classifier._extract_face_roi(valid_image, mock_face)

    def test_empty_crop_raises_error(self, classifier, valid_image, mock_face):
        """Bbox resulting in zero or negative area raises GlassesDetectorError."""
        mock_face.bbox = np.array([50, 50, 50, 50], dtype=np.float32)
        with pytest.raises(GlassesDetectorError, match="Face bounding box produces an empty crop"):
            classifier._extract_face_roi(valid_image, mock_face)

    def test_missing_bbox_raises_error(self, classifier, valid_image):
        """Face object lacking bbox attribute raises GlassesDetectorError."""
        class DummyFace:
            pass
        face_without_bbox = DummyFace()
        with pytest.raises(GlassesDetectorError, match="Face object has no bounding box information"):
            classifier._extract_face_roi(valid_image, face_without_bbox)  # type: ignore[arg-type]


# ================================================================== #
# 4. Lazy loading & Concurrency                                     #
# ================================================================== #


class TestLazyLoading:
    """Verify lazy model loading, caching, and thread safety."""

    def test_models_loaded_only_once(self, classifier):
        """First call to _ensure_loaded initializes classifiers; subsequent calls reuse them."""
        mock_cls_instance = MagicMock()
        with patch(
            "services.glasses_detector_classifier.GlassesClassifier",
            return_value=mock_cls_instance,
        ) as mock_ctor:
            eyeglasses, sunglasses = classifier._ensure_loaded()
            assert eyeglasses is mock_cls_instance
            assert sunglasses is mock_cls_instance
            assert mock_ctor.call_count == 2  # one for eyeglasses, one for sunglasses

            # Call again
            eyeglasses_2, sunglasses_2 = classifier._ensure_loaded()
            assert eyeglasses_2 is eyeglasses
            assert sunglasses_2 is sunglasses
            assert mock_ctor.call_count == 2  # no additional calls

    def test_concurrent_lazy_loading(self, classifier):
        """Multiple concurrent calls to classify() trigger model loading exactly once."""
        mock_cls_instance = MagicMock()
        mock_cls_instance.predict.return_value = 0.1

        with patch(
            "services.glasses_detector_classifier.GlassesClassifier",
            return_value=mock_cls_instance,
        ) as mock_ctor:
            valid_image = np.zeros((100, 100, 3), dtype=np.uint8)
            face = Face()
            face.bbox = np.array([10, 10, 50, 50], dtype=np.float32)

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [
                    executor.submit(classifier.classify, valid_image, face)
                    for _ in range(10)
                ]
                for f in futures:
                    f.result()

            # Exactly two constructor calls (one for eyeglasses, one for sunglasses)
            assert mock_ctor.call_count == 2


# ================================================================== #
# 5. Prediction mapping & Threshold Boundary Tests                   #
# ================================================================== #


class TestPredictionMapping:
    """Verify raw probability mapping, threshold behavior, and exact boundaries."""

    @pytest.mark.parametrize(
        "eyeglasses_prob, sunglasses_prob, expected_type",
        [
            pytest.param(0.2, 0.8, EyewearType.SUNGLASSES, id="sunglasses_above_threshold"),
            pytest.param(0.9, 0.2, EyewearType.CLEAR_GLASSES, id="eyeglasses_above_threshold"),
            pytest.param(0.1, 0.1, EyewearType.NONE, id="both_below_threshold"),
            pytest.param(0.8, 0.9, EyewearType.SUNGLASSES, id="sunglasses_takes_priority"),
        ],
    )
    def test_build_prediction_combinations(
        self, eyeglasses_prob, sunglasses_prob, expected_type
    ):
        """Probability combinations correctly map to EyewearType adhering to priority."""
        prediction = GlassesDetectorClassifier._build_prediction(
            eyeglasses_prob, sunglasses_prob
        )
        assert prediction.eyewear_type == expected_type
        if expected_type is EyewearType.SUNGLASSES:
            assert prediction.confidence == sunglasses_prob
        elif expected_type is EyewearType.CLEAR_GLASSES:
            assert prediction.confidence == eyeglasses_prob
        else:
            assert prediction.confidence == eyeglasses_prob

    @pytest.mark.parametrize(
        "delta, expected_type",
        [
            pytest.param(-1e-5, EyewearType.NONE, id="below_sunglasses_threshold"),
            pytest.param(0.0, EyewearType.NONE, id="exact_sunglasses_threshold_strict_greater"),
            pytest.param(1e-5, EyewearType.SUNGLASSES, id="above_sunglasses_threshold"),
        ],
    )
    def test_sunglasses_threshold_boundary(self, delta, expected_type):
        """Verify strict inequality (> threshold) behavior for sunglasses probability."""
        prob = GLASSES_SUNGLASSES_PROBABILITY_THRESHOLD + delta
        prediction = GlassesDetectorClassifier._build_prediction(0.0, prob)
        assert prediction.eyewear_type == expected_type

    @pytest.mark.parametrize(
        "delta, expected_type",
        [
            pytest.param(-1e-5, EyewearType.NONE, id="below_eyeglasses_threshold"),
            pytest.param(0.0, EyewearType.NONE, id="exact_eyeglasses_threshold_strict_greater"),
            pytest.param(1e-5, EyewearType.CLEAR_GLASSES, id="above_eyeglasses_threshold"),
        ],
    )
    def test_eyeglasses_threshold_boundary(self, delta, expected_type):
        """Verify strict inequality (> threshold) behavior for eyeglasses probability."""
        prob = GLASSES_EYEGLASSES_PROBABILITY_THRESHOLD + delta
        prediction = GlassesDetectorClassifier._build_prediction(prob, 0.0)
        assert prediction.eyewear_type == expected_type


# ================================================================== #
# 6. Error handling                                                  #
# ================================================================== #


class TestErrorHandling:
    """Verify exception handling and wrapping into GlassesDetectorError."""

    def test_eyeglasses_classifier_exception_wrapped(self, classifier, valid_image, mock_face):
        """Exceptions during eyeglasses prediction raise GlassesDetectorError."""
        mock_eyeglasses = MagicMock()
        mock_eyeglasses.predict.side_effect = RuntimeError("inference timeout")
        mock_sunglasses = MagicMock()

        with patch.object(
            classifier,
            "_ensure_loaded",
            return_value=(mock_eyeglasses, mock_sunglasses),
        ):
            with pytest.raises(GlassesDetectorError, match="Eyeglasses classification failed"):
                classifier.classify(valid_image, mock_face)

    def test_sunglasses_classifier_exception_wrapped(self, classifier, valid_image, mock_face):
        """Exceptions during sunglasses prediction raise GlassesDetectorError."""
        mock_eyeglasses = MagicMock()
        mock_eyeglasses.predict.return_value = 0.1
        mock_sunglasses = MagicMock()
        mock_sunglasses.predict.side_effect = RuntimeError("CUDA OOM")

        with patch.object(
            classifier,
            "_ensure_loaded",
            return_value=(mock_eyeglasses, mock_sunglasses),
        ):
            with pytest.raises(GlassesDetectorError, match="Sunglasses classification failed"):
                classifier.classify(valid_image, mock_face)

    def test_initialization_failure_wrapped(self, classifier):
        """Failures while instantiating GlassesClassifier raise GlassesDetectorError."""
        with patch(
            "services.glasses_detector_classifier.GlassesClassifier",
            side_effect=Exception("Model weights missing"),
        ):
            with pytest.raises(GlassesDetectorError, match="Could not initialize glasses-detector classifiers"):
                classifier._load_classifiers()


# ================================================================== #
# 7. RGB conversion                                                  #
# ================================================================== #


class TestRgbConversion:
    """Verify BGR to RGB image color space conversion."""

    def test_rgb_conversion_performed_once(self, classifier, valid_image, mock_face):
        """cv2.cvtColor is called exactly once with BGR2RGB before classification."""
        mock_eyeglasses = MagicMock()
        mock_eyeglasses.predict.return_value = 0.1
        mock_sunglasses = MagicMock()
        mock_sunglasses.predict.return_value = 0.1

        with patch.object(
            classifier,
            "_ensure_loaded",
            return_value=(mock_eyeglasses, mock_sunglasses),
        ), patch("cv2.cvtColor", wraps=cv2.cvtColor) as mock_cvt:
            classifier.classify(valid_image, mock_face)
            mock_cvt.assert_called_once()
            call_args = mock_cvt.call_args[0]
            conversion_code = call_args[1]
            assert conversion_code == cv2.COLOR_BGR2RGB


# ================================================================== #
# 8. classify() integration                                          #
# ================================================================== #


class TestClassifyIntegration:
    """Verify complete classify() execution flow and returned model fields."""

    def test_classify_returns_expected_prediction(self, classifier, valid_image, mock_face):
        """classify() returns a properly populated EyewearPrediction object."""
        mock_eyeglasses = MagicMock()
        mock_eyeglasses.predict.return_value = 0.92
        mock_sunglasses = MagicMock()
        mock_sunglasses.predict.return_value = 0.05

        with patch.object(
            classifier,
            "_ensure_loaded",
            return_value=(mock_eyeglasses, mock_sunglasses),
        ):
            prediction = classifier.classify(valid_image, mock_face)

        assert isinstance(prediction, EyewearPrediction)
        assert prediction.eyewear_type == EyewearType.CLEAR_GLASSES
        assert prediction.confidence == 0.92
