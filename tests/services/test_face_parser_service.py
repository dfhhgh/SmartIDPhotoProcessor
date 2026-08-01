"""Tests for FaceParserService — Stage 1 (Lifecycle & Config), Stage 2 (Validation & Preprocessing), Stage 3 (Inference), Stage 4 (Postprocessing), and Stage 5 (Parse Pipeline)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.parsing.face_part import FacePart
from models.parsing.face_parsing_result import FaceParsingResult
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

    input_meta = MagicMock()
    input_meta.name = "input_image"
    session.get_inputs.return_value = [input_meta]

    output_meta = MagicMock()
    output_meta.name = "output"
    session.get_outputs.return_value = [output_meta]

    # Default mock output logits of shape (1, 19, 512, 512)
    fake_logits = np.zeros((1, 19, 512, 512), dtype=np.float32)
    session.run.return_value = [fake_logits]

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


# ================================================================== #
# 6. Stage 3: Inference Execution (_run_inference)                   #
# ================================================================== #


class TestRunInference:
    """Verify ONNX Runtime model execution in _run_inference()."""

    def test_session_run_called_exactly_once(self, service, fake_onnx_session):
        """session.run() must be invoked exactly once per inference call."""
        # Arrange
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act
        service._run_inference(fake_onnx_session, input_tensor)

        # Assert
        fake_onnx_session.run.assert_called_once()

    def test_correct_input_tensor_passed_to_onnx(self, service, fake_onnx_session):
        """The exact input tensor array object must be passed into session.run()."""
        # Arrange
        input_tensor = np.ones((1, 3, 512, 512), dtype=np.float32)

        # Act
        service._run_inference(fake_onnx_session, input_tensor)

        # Assert
        _, args = fake_onnx_session.run.call_args
        feed_dict = args if isinstance(args, dict) and args else fake_onnx_session.run.call_args[0][1]
        assert feed_dict["input_image"] is input_tensor

    def test_correct_input_tensor_name_used(self, service, fake_onnx_session):
        """The input node name queried from session.get_inputs() must be used as the feed key."""
        # Arrange
        fake_onnx_session.get_inputs()[0].name = "custom_tensor_node_name"
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act
        service._run_inference(fake_onnx_session, input_tensor)

        # Assert
        feed_dict = fake_onnx_session.run.call_args[0][1]
        assert "custom_tensor_node_name" in feed_dict

    def test_returned_logits_propagated_correctly(self, service, fake_onnx_session):
        """The raw logits output array returned by session.run() must be returned by _run_inference()."""
        # Arrange
        expected_logits = np.random.randn(1, 19, 512, 512).astype(np.float32)
        fake_onnx_session.run.return_value = [expected_logits]
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act
        result_logits = service._run_inference(fake_onnx_session, input_tensor)

        # Assert
        assert np.array_equal(result_logits, expected_logits)

    def test_session_run_raises_runtime_error_propagates(self, service, fake_onnx_session):
        """RuntimeError raised directly by ONNX Session propagates from _run_inference."""
        # Arrange
        fake_onnx_session.run.side_effect = RuntimeError("ONNX execution error")
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act & Assert
        with pytest.raises(RuntimeError, match="ONNX execution error"):
            service._run_inference(fake_onnx_session, input_tensor)

    @pytest.mark.parametrize(
        "outputs, expected_msg",
        [
            pytest.param([], "Model returned no outputs", id="empty_output_list"),
            pytest.param(
                [np.zeros((19, 512, 512), dtype=np.float32)],
                "No valid segmentation output found",
                id="missing_batch_dim_3d",
            ),
            pytest.param(
                [np.zeros((2, 19, 512, 512), dtype=np.float32)],
                "No valid segmentation output found",
                id="batch_size_greater_than_1",
            ),
            pytest.param(
                [np.zeros((1, 10, 512, 512), dtype=np.float32)],
                "No valid segmentation output found",
                id="wrong_class_channels",
            ),
        ],
    )
    def test_invalid_output_shapes_raise_face_parser_error(
        self, service, fake_onnx_session, outputs, expected_msg
    ):
        """Unexpected model output shapes or counts must raise FaceParserError."""
        # Arrange
        fake_onnx_session.run.return_value = outputs
        fake_onnx_session.get_outputs.return_value = [MagicMock(name=str(i)) for i in range(len(outputs))]
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act & Assert
        with pytest.raises(FaceParserError, match=expected_msg):
            service._run_inference(fake_onnx_session, input_tensor)

    def test_multiple_outputs_prefers_output_named_output(self, service, fake_onnx_session):
        """When multiple valid outputs exist, prefer the one named 'output'."""
        out1 = MagicMock()
        out1.name = "aux_output"
        out2 = MagicMock()
        out2.name = "output"
        fake_onnx_session.get_outputs.return_value = [out1, out2]

        arr1 = np.ones((1, 19, 512, 512), dtype=np.float32) * 1.0
        arr2 = np.ones((1, 19, 512, 512), dtype=np.float32) * 2.0
        fake_onnx_session.run.return_value = [arr1, arr2]

        tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)
        result = service._run_inference(fake_onnx_session, tensor)
        assert np.array_equal(result, arr2)

    def test_multiple_outputs_selects_first_valid_if_no_output_name(self, service, fake_onnx_session):
        """When multiple valid outputs exist and none is named 'output', select the first valid output."""
        out1 = MagicMock()
        out1.name = "first"
        out2 = MagicMock()
        out2.name = "second"
        fake_onnx_session.get_outputs.return_value = [out1, out2]

        arr1 = np.ones((1, 19, 512, 512), dtype=np.float32) * 5.0
        arr2 = np.ones((1, 19, 512, 512), dtype=np.float32) * 6.0
        fake_onnx_session.run.return_value = [arr1, arr2]

        tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)
        result = service._run_inference(fake_onnx_session, tensor)
        assert np.array_equal(result, arr1)

    def test_large_output_tensors_handled_correctly(self, service, fake_onnx_session):
        """Large output logits tensors (e.g. 1024x1024 spatial) must be processed without error."""
        # Arrange
        large_logits = np.zeros((1, 19, 1024, 1024), dtype=np.float32)
        fake_onnx_session.run.return_value = [large_logits]
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act
        result = service._run_inference(fake_onnx_session, input_tensor)

        # Assert
        assert result.shape == (1, 19, 1024, 1024)

    def test_output_dtype_preserved(self, service, fake_onnx_session):
        """The float32 data type of returned logits must be preserved."""
        # Arrange
        logits = np.zeros((1, 19, 512, 512), dtype=np.float32)
        fake_onnx_session.run.return_value = [logits]
        input_tensor = np.zeros((1, 3, 512, 512), dtype=np.float32)

        # Act
        result = service._run_inference(fake_onnx_session, input_tensor)

        # Assert
        assert result.dtype == np.float32


# ================================================================== #
# 7. Stage 4: Post-processing (_postprocess)                        #
# ================================================================== #


class TestPostprocess:
    """Verify conversion of raw network logits to FaceParsingResult in _postprocess()."""

    def test_argmax_produces_expected_labels(self, service):
        """Pixel label map must equal the class channel index with maximum logit value."""
        # Arrange
        raw_output = np.zeros((1, 19, 4, 4), dtype=np.float32)
        # Set class index 4 (LEFT_EYE) to highest value at (0, 0)
        raw_output[0, 4, 0, 0] = 100.0
        # Set class index 10 (NOSE) to highest value at (1, 1)
        raw_output[0, 10, 1, 1] = 100.0

        # Act
        result = service._postprocess(raw_output, original_height=4, original_width=4)

        # Assert
        assert result.mask[0, 0] == int(FacePart.LEFT_EYE)
        assert result.mask[1, 1] == int(FacePart.NOSE)

    def test_output_mask_dtype_is_integer(self, service):
        """Segmentation mask in FaceParsingResult must have integer dtype."""
        # Arrange
        raw_output = np.zeros((1, 19, 512, 512), dtype=np.float32)

        # Act
        result = service._postprocess(raw_output, original_height=512, original_width=512)

        # Assert
        assert np.issubdtype(result.mask.dtype, np.integer)

    def test_output_shape_matches_original_image_dimensions(self, service):
        """When original height/width differ from model output, mask must be resized back."""
        # Arrange
        raw_output = np.zeros((1, 19, 512, 512), dtype=np.float32)
        orig_h, orig_w = 300, 400

        # Act
        result = service._postprocess(raw_output, original_height=orig_h, original_width=orig_w)

        # Assert
        assert result.mask.shape == (300, 400)
        assert result.image_height == 300
        assert result.image_width == 400

    def test_single_class_output(self, service):
        """When a single class dominates all logits, the mask must contain only that class."""
        # Arrange
        raw_output = np.zeros((1, 19, 64, 64), dtype=np.float32)
        raw_output[0, 1, :, :] = 50.0  # Class 1 (SKIN) highest everywhere

        # Act
        result = service._postprocess(raw_output, original_height=64, original_width=64)

        # Assert
        assert (result.mask == int(FacePart.SKIN)).all()

    def test_multiple_class_output(self, service):
        """Logits predicting distinct regions must yield corresponding multi-part mask."""
        # Arrange
        raw_output = np.zeros((1, 19, 2, 2), dtype=np.float32)
        raw_output[0, 0, 0, 0] = 10.0  # BACKGROUND
        raw_output[0, 1, 0, 1] = 10.0  # SKIN
        raw_output[0, 2, 1, 0] = 10.0  # LEFT_BROW
        raw_output[0, 10, 1, 1] = 10.0 # NOSE

        # Act
        result = service._postprocess(raw_output, original_height=2, original_width=2)

        # Assert
        expected_mask = np.array(
            [[0, 1], [2, 10]], dtype=np.int32
        )
        assert np.array_equal(result.mask, expected_mask)

    def test_uniform_logits_defaults_to_first_class(self, service):
        """Equal logits across all classes argmax to class 0 (BACKGROUND)."""
        # Arrange
        raw_output = np.ones((1, 19, 16, 16), dtype=np.float32)

        # Act
        result = service._postprocess(raw_output, original_height=16, original_width=16)

        # Assert
        assert (result.mask == int(FacePart.BACKGROUND)).all()

    def test_negative_logits_handled_correctly(self, service):
        """Argmax must correctly identify the maximum index even when all logits are negative."""
        # Arrange
        raw_output = np.full((1, 19, 4, 4), fill_value=-100.0, dtype=np.float32)
        raw_output[0, 17, 2, 2] = -5.0  # Class 17 (HAIR) is highest (-5 > -100)

        # Act
        result = service._postprocess(raw_output, original_height=4, original_width=4)

        # Assert
        assert result.mask[2, 2] == int(FacePart.HAIR)

    @pytest.mark.parametrize("scale", [1e6, 1e-6])
    def test_extreme_magnitude_logits(self, service, scale):
        """Argmax must perform correctly with very large or very small numeric scales."""
        # Arrange
        raw_output = np.zeros((1, 19, 4, 4), dtype=np.float32)
        raw_output[0, 5, 0, 0] = scale  # Class 5 (RIGHT_EYE)

        # Act
        result = service._postprocess(raw_output, original_height=4, original_width=4)

        # Assert
        assert result.mask[0, 0] == int(FacePart.RIGHT_EYE)

    def test_output_values_contain_no_nan_or_inf(self, service):
        """The output mask must be completely finite without NaN or Inf values."""
        # Arrange
        raw_output = np.random.randn(1, 19, 32, 32).astype(np.float32)

        # Act
        result = service._postprocess(raw_output, original_height=32, original_width=32)

        # Assert
        assert np.isfinite(result.mask).all()

    def test_postprocess_is_deterministic(self, service):
        """Calling _postprocess twice with identical input produces identical FaceParsingResult masks."""
        # Arrange
        raw_output = np.random.randn(1, 19, 64, 64).astype(np.float32)

        # Act
        res1 = service._postprocess(raw_output, original_height=64, original_width=64)
        res2 = service._postprocess(raw_output, original_height=64, original_width=64)

        # Assert
        assert np.array_equal(res1.mask, res2.mask)


# ================================================================== #
# 8. Stage 5: Pipeline Orchestration (parse)                       #
# ================================================================== #


class TestParsePipeline:
    """Verify complete orchestration and data flow in parse()."""

    def test_happy_path_returns_face_parsing_result(self, service, fake_onnx_session):
        """parse() given a valid uint8 image returns a FaceParsingResult."""
        # Arrange
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        with patch.object(service, "_ensure_loaded", return_value=fake_onnx_session):
            # Act
            result = service.parse(image)

        # Assert
        assert isinstance(result, FaceParsingResult)
        assert result.image_height == 100
        assert result.image_width == 200
        assert result.mask.shape == (100, 200)

    def test_pipeline_method_execution_order(self, service, fake_onnx_session):
        """validate -> preprocess -> run_inference -> postprocess order must be maintained."""
        # Arrange
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        call_order: list[str] = []

        def spy_validate(img):
            call_order.append("validate")

        def spy_preprocess(img):
            call_order.append("preprocess")
            return np.zeros((1, 3, 512, 512), dtype=np.float32)

        def spy_inference(sess, tensor):
            call_order.append("inference")
            return np.zeros((1, 19, 512, 512), dtype=np.float32)

        def spy_postprocess(raw, h, w):
            call_order.append("postprocess")
            return FaceParsingResult(mask=np.zeros((h, w), dtype=np.int32), image_height=h, image_width=w)

        with (
            patch.object(service, "_ensure_loaded", return_value=fake_onnx_session),
            patch.object(service, "_validate_image", side_effect=spy_validate),
            patch.object(service, "_preprocess", side_effect=spy_preprocess),
            patch.object(service, "_run_inference", side_effect=spy_inference),
            patch.object(service, "_postprocess", side_effect=spy_postprocess),
        ):
            # Act
            service.parse(image)

        # Assert
        assert call_order == ["validate", "preprocess", "inference", "postprocess"]

    def test_invalid_image_propagates_validation_exception(self, service):
        """Invalid images (e.g. empty or non-uint8) must raise validation error from parse()."""
        # Arrange
        invalid_image = np.zeros((100, 100, 3), dtype=np.float32)

        # Act & Assert
        with pytest.raises(TypeError, match="image dtype must be uint8"):
            service.parse(invalid_image)

    def test_inference_failure_becomes_face_parser_error(self, service, fake_onnx_session):
        """Exceptions raised during model execution must be wrapped in FaceParserError."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        fake_onnx_session.run.side_effect = RuntimeError("GPU memory exhausted")

        with patch.object(service, "_ensure_loaded", return_value=fake_onnx_session):
            # Act & Assert
            with pytest.raises(FaceParserError, match="Face-parsing inference failed"):
                service.parse(image)

    def test_postprocessing_failure_becomes_face_parser_error(self, service, fake_onnx_session):
        """Exceptions raised during post-processing must be wrapped in FaceParserError."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with (
            patch.object(service, "_ensure_loaded", return_value=fake_onnx_session),
            patch.object(service, "_postprocess", side_effect=RuntimeError("Resizing failed")),
        ):
            # Act & Assert
            with pytest.raises(FaceParserError, match="Face-parsing post-processing failed"):
                service.parse(image)

    def test_repeated_parse_calls_reuse_loaded_session(self, service, fake_onnx_session):
        """Subsequent parse() calls must reuse the cached session and not reload."""
        # Arrange
        image = np.zeros((100, 100, 3), dtype=np.uint8)

        with patch.object(service, "_load_model") as mock_load:
            service._session = fake_onnx_session

            # Act
            service.parse(image)
            service.parse(image)

        # Assert
        mock_load.assert_not_called()

    @pytest.mark.parametrize(
        "height, width",
        [
            pytest.param(120, 120, id="small_square"),
            pytest.param(1080, 1920, id="large_hd_landscape"),
            pytest.param(800, 600, id="portrait_aspect_ratio"),
        ],
    )
    def test_varying_image_sizes_produce_correctly_sized_results(
        self, service, fake_onnx_session, height, width
    ):
        """parse() must produce a result mask with dimensions matching any input image size."""
        # Arrange
        image = np.zeros((height, width, 3), dtype=np.uint8)

        with patch.object(service, "_ensure_loaded", return_value=fake_onnx_session):
            # Act
            result = service.parse(image)

        # Assert
        assert result.image_height == height
        assert result.image_width == width
        assert result.mask.shape == (height, width)
