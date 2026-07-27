"""Tests for FaceParserService — Stage 1 (Lifecycle & Config) and Stage 2 (Validation & Preprocessing)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.face_parser_service import (
    FaceParserError,
    FaceParserService,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset FaceParserService singleton state before and after every test."""
    FaceParserService._instance = None
    FaceParserService._initialized = False
    yield
    FaceParserService._instance = None
    FaceParserService._initialized = False


@pytest.fixture
def settings_factory(tmp_path):
    """Factory for MagicMock Settings backed by a real temp directory tree."""

    def _make(
        use_gpu: bool = False,
        gpu_id: int = 0,
        model_exists: bool = True,
    ) -> MagicMock:
        model_root = tmp_path / "models"
        bisenet_dir = model_root / "bisenet"
        bisenet_dir.mkdir(parents=True, exist_ok=True)

        if model_exists:
            (bisenet_dir / "bisenet_resnet18.onnx").touch()

        return MagicMock(
            MODEL_ROOT=model_root,
            USE_GPU=use_gpu,
            GPU_ID=gpu_id,
        )

    return _make


@pytest.fixture
def service(settings_factory) -> FaceParserService:
    """A FaceParserService constructed against a valid, existing model file."""
    with patch(
        "services.face_parser_service.Settings",
        return_value=settings_factory(),
    ):
        return FaceParserService()


@pytest.fixture
def service_with_missing_model(settings_factory) -> FaceParserService:
    """A FaceParserService whose configured model file does not exist on disk."""
    with patch(
        "services.face_parser_service.Settings",
        return_value=settings_factory(model_exists=False),
    ):
        return FaceParserService()


@pytest.fixture
def fake_onnx_session() -> MagicMock:
    """A stand-in for onnxruntime.InferenceSession."""
    session = MagicMock()
    session.get_providers.return_value = ["CPUExecutionProvider"]
    return session


# ================================================================== #
# 1. Singleton behaviour                                              #
# ================================================================== #


class TestSingleton:
    """Verify that FaceParserService follows the singleton pattern."""

    def test_multiple_constructions_return_same_instance(self, settings_factory):
        """Two calls to FaceParserService() must return the identical object."""
        # Arrange
        with patch(
            "services.face_parser_service.Settings",
            return_value=settings_factory(),
        ):
            # Act
            first = FaceParserService()
            second = FaceParserService()

        # Assert
        assert first is second

    def test_configuration_is_preserved_across_repeated_construction(
        self, settings_factory
    ):
        """Provider resolution behaviour set up on first construction must
        still hold after subsequent, no-op constructions."""
        # Arrange
        with patch(
            "services.face_parser_service.Settings",
            return_value=settings_factory(use_gpu=True, gpu_id=5),
        ):
            FaceParserService()
            second = FaceParserService()

        # Act
        with patch(
            "services.face_parser_service.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            providers = second._resolve_providers()

        # Assert
        assert providers == [
            ("CUDAExecutionProvider", {"device_id": 5}),
            "CPUExecutionProvider",
        ]


# ================================================================== #
# 2. Lazy model loading                                               #
# ================================================================== #


class TestLazyLoading:
    """Verify that ONNX model loading is deferred until first access."""

    def test_model_not_loaded_during_construction(self, service):
        """No inference session should exist immediately after construction."""
        # Assert
        assert service._session is None

    def test_ensure_loaded_creates_session_on_first_call(
        self, service, fake_onnx_session
    ):
        """First access must create and return a real inference session."""
        # Arrange
        with patch(
            "services.face_parser_service.ort.InferenceSession",
            return_value=fake_onnx_session,
        ) as mock_ctor:
            # Act
            result = service._ensure_loaded()

        # Assert
        assert result is fake_onnx_session
        mock_ctor.assert_called_once()

    def test_ensure_loaded_reuses_session_on_subsequent_calls(
        self, service, fake_onnx_session
    ):
        """Repeated access must not create additional inference sessions."""
        # Arrange
        with patch(
            "services.face_parser_service.ort.InferenceSession",
            return_value=fake_onnx_session,
        ) as mock_ctor:
            # Act
            first = service._ensure_loaded()
            second = service._ensure_loaded()

        # Assert
        assert first is second is fake_onnx_session
        mock_ctor.assert_called_once()

    def test_missing_model_file_raises_file_not_found_error(
        self, service_with_missing_model
    ):
        """A configured but absent model file must raise FileNotFoundError."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            service_with_missing_model._ensure_loaded()

    def test_session_creation_failure_raises_face_parser_error(self, service):
        """Any failure while constructing the ONNX session must be wrapped."""
        # Arrange
        with patch(
            "services.face_parser_service.ort.InferenceSession",
            side_effect=RuntimeError("corrupt model file"),
        ):
            # Act & Assert
            with pytest.raises(FaceParserError, match="Could not initialize"):
                service._ensure_loaded()


# ================================================================== #
# 3. Execution provider resolution                                     #
# ================================================================== #


class TestResolveProviders:
    """Verify _resolve_providers under various hardware configurations."""

    @pytest.mark.parametrize(
        "available_providers, use_gpu, expected",
        [
            pytest.param(
                ["CPUExecutionProvider"],
                False,
                ["CPUExecutionProvider"],
                id="cpu_only_gpu_disabled",
            ),
            pytest.param(
                ["CPUExecutionProvider"],
                True,
                ["CPUExecutionProvider"],
                id="cuda_unavailable_gpu_enabled",
            ),
            pytest.param(
                ["CUDAExecutionProvider", "CPUExecutionProvider"],
                False,
                ["CPUExecutionProvider"],
                id="cuda_available_gpu_disabled",
            ),
            pytest.param(
                [],
                True,
                ["CPUExecutionProvider"],
                id="no_providers_available_gpu_enabled",
            ),
            pytest.param(
                [],
                False,
                ["CPUExecutionProvider"],
                id="no_providers_available_gpu_disabled",
            ),
            pytest.param(
                ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
                False,
                ["CPUExecutionProvider"],
                id="unrelated_providers_present_gpu_disabled",
            ),
        ],
    )
    def test_cpu_fallback_matrix(
        self, service, available_providers, use_gpu, expected
    ):
        """CPU-only fallback must hold whenever GPU is disabled or CUDA is absent."""
        # Arrange
        service._use_gpu = use_gpu

        with patch(
            "services.face_parser_service.ort.get_available_providers",
            return_value=available_providers,
        ):
            # Act
            providers = service._resolve_providers()

        # Assert
        assert providers == expected

    @pytest.mark.parametrize("gpu_id", [0, 2, 7])
    def test_cuda_available_and_enabled_prioritises_cuda(self, service, gpu_id):
        """CUDA + USE_GPU=True must yield CUDA first, then CPU fallback."""
        # Arrange
        service._use_gpu = True
        service._gpu_id = gpu_id

        with patch(
            "services.face_parser_service.ort.get_available_providers",
            return_value=["CUDAExecutionProvider", "CPUExecutionProvider"],
        ):
            # Act
            providers = service._resolve_providers()

        # Assert
        assert providers == [
            ("CUDAExecutionProvider", {"device_id": gpu_id}),
            "CPUExecutionProvider",
        ]

    def test_cuda_prioritised_regardless_of_position_in_available_list(
        self, service
    ):
        """Provider selection must not depend on the order returned by onnxruntime."""
        # Arrange
        service._use_gpu = True
        service._gpu_id = 0

        with patch(
            "services.face_parser_service.ort.get_available_providers",
            return_value=["CPUExecutionProvider", "CUDAExecutionProvider"],
        ):
            # Act
            providers = service._resolve_providers()

        # Assert
        assert providers == [
            ("CUDAExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider",
        ]

    def test_unrelated_providers_ignored_when_cuda_enabled(self, service):
        """Providers other than CUDA/CPU must never appear in the result."""
        # Arrange
        service._use_gpu = True
        service._gpu_id = 3

        with patch(
            "services.face_parser_service.ort.get_available_providers",
            return_value=[
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ):
            # Act
            providers = service._resolve_providers()

        # Assert
        assert providers == [
            ("CUDAExecutionProvider", {"device_id": 3}),
            "CPUExecutionProvider",
        ]


# ================================================================== #
# 4. Image Validation (_validate_image)                              #
# ================================================================== #


class TestValidateImage:
    """Verify input image validation rules in _validate_image()."""

    def test_valid_uint8_bgr_image_passes_validation(self, service):
        """A valid 3-channel uint8 NumPy image must pass without exception."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act & Assert
        service._validate_image(image)

    def test_none_input_raises_type_error(self, service):
        """None input must raise TypeError with an informative message."""
        # Act & Assert
        with pytest.raises(TypeError, match="image must be a numpy.ndarray, got NoneType"):
            service._validate_image(None)

    @pytest.mark.parametrize(
        "invalid_input, type_name",
        [
            pytest.param([[0, 0, 0]], "list", id="list"),
            pytest.param((0, 0, 0), "tuple", id="tuple"),
            pytest.param("image.jpg", "str", id="str"),
            pytest.param(123, "int", id="int"),
        ],
    )
    def test_non_numpy_input_raises_type_error(self, service, invalid_input, type_name):
        """Non-numpy.ndarray objects must raise TypeError."""
        # Act & Assert
        with pytest.raises(TypeError, match=f"image must be a numpy.ndarray, got {type_name}"):
            service._validate_image(invalid_input)

    def test_empty_numpy_array_raises_value_error(self, service):
        """Empty NumPy arrays (size 0) must raise ValueError."""
        # Arrange
        image = np.empty((0, 100, 3), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match="image must not be empty"):
            service._validate_image(image)

    def test_grayscale_image_raises_value_error(self, service):
        """2-D grayscale images (H, W) must raise ValueError."""
        # Arrange
        image = np.zeros((100, 100), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match=r"image must have shape \(H, W, 3\), got \(100, 100\)"):
            service._validate_image(image)

    def test_four_channel_image_raises_value_error(self, service):
        """4-channel images (e.g. BGRA) must raise ValueError."""
        # Arrange
        image = np.zeros((100, 100, 4), dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match=r"image must have shape \(H, W, 3\), got \(100, 100, 4\)"):
            service._validate_image(image)

    @pytest.mark.parametrize(
        "shape",
        [
            pytest.param((100,), id="1D"),
            pytest.param((1, 100, 100, 3), id="4D"),
        ],
    )
    def test_invalid_dimensions_raise_value_error(self, service, shape):
        """Arrays with ndim != 3 must raise ValueError."""
        # Arrange
        image = np.zeros(shape, dtype=np.uint8)

        # Act & Assert
        with pytest.raises(ValueError, match=r"image must have shape \(H, W, 3\)"):
            service._validate_image(image)

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(np.float32, id="float32"),
            pytest.param(np.int32, id="int32"),
            pytest.param(np.bool_, id="bool"),
        ],
    )
    def test_wrong_dtype_raises_type_error(self, service, dtype):
        """Non-uint8 arrays must raise TypeError mentioning the actual dtype."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=dtype)

        # Act & Assert
        with pytest.raises(TypeError, match="image dtype must be uint8"):
            service._validate_image(image)


# ================================================================== #
# 5. Preprocessing (_preprocess)                                     #
# ================================================================== #


class TestPreprocess:
    """Verify image tensor transformation in _preprocess()."""

    def test_returned_tensor_dtype_is_float32(self, service):
        """The preprocessed tensor must have float32 dtype."""
        # Arrange
        image = np.zeros((200, 300, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        assert tensor.dtype == np.float32

    def test_returned_tensor_shape_is_nchw_512(self, service):
        """Regardless of input resolution, the output tensor must be (1, 3, 512, 512)."""
        # Arrange
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        assert tensor.shape == (1, 3, 512, 512)

    def test_returned_tensor_is_c_contiguous(self, service):
        """The tensor must be contiguous in C memory layout for ONNX Runtime."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        assert tensor.flags.c_contiguous is True

    def test_original_image_is_not_modified(self, service):
        """_preprocess must operate on copies and leave the input array untouched."""
        # Arrange
        image = np.full((100, 100, 3), fill_value=128, dtype=np.uint8)
        original_copy = image.copy()

        # Act
        service._preprocess(image)

        # Assert
        assert np.array_equal(image, original_copy)

    def test_bgr_to_rgb_channel_swapping(self, service):
        """Input BGR color channels must map to RGB order in the output tensor."""
        # Arrange
        # Solid BGR image with B=100, G=150, R=200
        image = np.full((512, 512, 3), fill_value=(100, 150, 200), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        # In RGB: Red=200, Green=150, Blue=100
        # Normalization: (value / 255.0 - mean) / std
        expected_r = (200 / 255.0 - 0.485) / 0.229
        expected_g = (150 / 255.0 - 0.456) / 0.224
        expected_b = (100 / 255.0 - 0.406) / 0.225

        # Tensor shape (1, 3, 512, 512) -> Channel 0 = Red, Channel 1 = Green, Channel 2 = Blue
        assert tensor[0, 0, 0, 0] == pytest.approx(expected_r, rel=1e-4)
        assert tensor[0, 1, 0, 0] == pytest.approx(expected_g, rel=1e-4)
        assert tensor[0, 2, 0, 0] == pytest.approx(expected_b, rel=1e-4)

    def test_resize_applied_to_target_resolution(self, service):
        """A non-square image must be resized to (512, 512)."""
        # Arrange
        image = np.zeros((300, 600, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        assert tensor.shape[2:] == (512, 512)

    def test_normalization_with_imagenet_stats(self, service):
        """Zero pixel values (BGR=0,0,0) must map to -mean/std for ImageNet."""
        # Arrange
        image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        expected_r = -0.485 / 0.229  # ≈ -2.1179
        expected_g = -0.456 / 0.224  # ≈ -2.0357
        expected_b = -0.406 / 0.225  # ≈ -1.8044

        assert tensor[0, 0, 0, 0] == pytest.approx(expected_r, rel=1e-4)
        assert tensor[0, 1, 0, 0] == pytest.approx(expected_g, rel=1e-4)
        assert tensor[0, 2, 0, 0] == pytest.approx(expected_b, rel=1e-4)

    def test_chw_ordering_transposition(self, service):
        """The spatial HWC channels must be transposed to CHW layout."""
        # Arrange
        image = np.zeros((512, 512, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        # Shape is (1, C, H, W) where C=3, H=512, W=512
        assert tensor.shape[1] == 3
        assert tensor.shape[2] == 512
        assert tensor.shape[3] == 512

    def test_batch_dimension_is_added(self, service):
        """Output tensor must have a batch dimension of size 1 at axis 0."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        assert tensor.shape[0] == 1

    def test_output_values_are_finite(self, service):
        """Preprocessed tensor must contain no NaN or Infinite values."""
        # Arrange
        image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

        # Act
        tensor = service._preprocess(image)

        # Assert
        assert np.isfinite(tensor).all()
