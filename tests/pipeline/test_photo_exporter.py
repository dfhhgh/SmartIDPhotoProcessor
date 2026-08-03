import numpy as np
import pytest

from config.constants import (
    MAX_ALLOWED_UPSCALE,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    SAFE_UPSCALE_FACTOR,
)
from pipeline.photo_exporter import PhotoExporter


@pytest.fixture
def exporter() -> PhotoExporter:
    return PhotoExporter()


def test_exporter_handles_crop_already_larger_than_target(exporter: PhotoExporter) -> None:
    image = np.zeros((1200, 1000, 3), dtype=np.uint8)

    result = exporter.export(image)

    assert result.exported_size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    assert result.original_size == (1000, 1200)
    assert result.was_upscaled is False
    assert result.upscale_factor == 1.0
    assert result.exported_image.shape[:2] == (OUTPUT_HEIGHT, OUTPUT_WIDTH)


def test_exporter_handles_crop_exactly_target_size(exporter: PhotoExporter) -> None:
    image = np.zeros((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), dtype=np.uint8)

    result = exporter.export(image)

    assert result.exported_size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    assert result.was_upscaled is False
    assert result.upscale_factor == 1.0
    assert result.exported_image.shape[:2] == (OUTPUT_HEIGHT, OUTPUT_WIDTH)


def test_exporter_performs_safe_upscale_with_cubic_interpolation(exporter: PhotoExporter) -> None:
    image = np.zeros((400, 400, 3), dtype=np.uint8)

    result = exporter.export(image)

    assert result.was_upscaled is True
    assert result.upscale_factor > 1.0
    assert result.upscale_factor <= SAFE_UPSCALE_FACTOR
    assert result.export_quality == "safe"
    assert result.exported_size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)


def test_exporter_warns_for_excessive_upscale(exporter: PhotoExporter) -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = exporter.export(image)

    assert result.was_upscaled is True
    assert result.upscale_factor > SAFE_UPSCALE_FACTOR
    assert result.export_quality == "warning"
    assert any("excessive upscaling" in warning.lower() for warning in result.export_warnings)


def test_exporter_rejects_excessive_upscale_when_reconfigured(exporter: PhotoExporter) -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="exceeds the maximum allowed upscale"):
        exporter.export(image, settings={"reject_on_excessive_upscale": True, "max_allowed_upscale": 1.0})


def test_exporter_rejects_invalid_image(exporter: PhotoExporter) -> None:
    with pytest.raises(ValueError, match="must be a non-empty numpy array"):
        exporter.export(None)  # type: ignore[arg-type]


def test_exporter_preserves_aspect_ratio(exporter: PhotoExporter) -> None:
    image = np.zeros((300, 600, 3), dtype=np.uint8)

    result = exporter.export(image)

    assert result.content_size[0] / result.content_size[1] == pytest.approx(600 / 300)


def test_exporter_reports_exported_dimensions(exporter: PhotoExporter) -> None:
    image = np.zeros((400, 300, 3), dtype=np.uint8)

    result = exporter.export(image)

    assert result.exported_size == (OUTPUT_WIDTH, OUTPUT_HEIGHT)
    assert result.content_size[0] > 0
    assert result.content_size[1] > 0


def test_exporter_uses_configuration_thresholds(exporter: PhotoExporter) -> None:
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    result = exporter.export(
        image,
        settings={
            "safe_upscale_factor": 1.1,
            "max_allowed_upscale": 1.3,
            "reject_on_excessive_upscale": False,
        },
    )

    assert result.export_quality == "warning"
    assert result.upscale_factor > 1.1


def test_exporter_uses_custom_target_size(exporter: PhotoExporter) -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    result = exporter.export(image, settings={"target_width": 400, "target_height": 500})

    assert result.exported_size == (400, 500)
    assert result.exported_image.shape[:2] == (500, 400)
