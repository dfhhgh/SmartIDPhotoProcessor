"""Unit tests for OpenverseSource."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from dataset_builder.config.settings import Settings
from dataset_builder.sources.base_source import SearchResult
from dataset_builder.sources.openverse import OpenverseSource


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        DATASET_DIR=tmp_path / "dataset",
        MIN_IMAGE_WIDTH=640,
        MIN_IMAGE_HEIGHT=480,
    )


@pytest.fixture
def source(settings: Settings) -> OpenverseSource:
    return OpenverseSource(settings)


def test_source_name(source: OpenverseSource) -> None:
    assert source.name == "openverse"


def test_validate_configuration_success(source: OpenverseSource) -> None:
    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert source.validate_configuration() is True


def test_validate_configuration_failure(source: OpenverseSource) -> None:
    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_get.side_effect = requests.RequestException("API error")

        with pytest.raises(requests.RequestException):
            source.validate_configuration()


def test_search_success(source: OpenverseSource) -> None:
    mock_payload = {
        "result_count": 1,
        "results": [
            {
                "id": "abc-123",
                "title": "Test Image",
                "url": "https://example.com/image.jpg",
                "foreign_landing_url": "https://example.com/page",
                "width": 800,
                "height": 600,
                "creator": "John Doe",
                "license": "by",
                "license_version": "4.0",
                "license_url": "https://creativecommons.org/licenses/by/4.0/",
                "thumbnail": "https://example.com/thumb.jpg",
            }
        ],
    }

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_get.return_value = mock_response

        results = source.search("face", page=1, per_page=10)

        assert len(results) == 1
        res = results[0]
        assert res.id == "abc-123"
        assert res.download_url == "https://example.com/image.jpg"
        assert res.width == 800
        assert res.height == 600
        assert res.photographer == "John Doe"
        assert res.license_name == "CC BY 4.0"
        assert res.license_url == "https://creativecommons.org/licenses/by/4.0/"
        assert res.license_type == "by"
        assert res.source == "openverse"


def test_search_filters_small_images(source: OpenverseSource) -> None:
    mock_payload = {
        "results": [
            {
                "id": "small-1",
                "url": "https://example.com/small.jpg",
                "width": 200,  # Below MIN_IMAGE_WIDTH (640)
                "height": 150,  # Below MIN_IMAGE_HEIGHT (480)
                "creator": "Jane Doe",
            }
        ]
    }

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_get.return_value = mock_response

        results = source.search("face", page=1, per_page=10)
        assert len(results) == 0


def test_search_empty_results(source: OpenverseSource) -> None:
    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}
        mock_get.return_value = mock_response

        results = source.search("nonexistent", page=1, per_page=10)
        assert len(results) == 0


def test_download_success(source: OpenverseSource, tmp_path: Path) -> None:
    result = SearchResult(
        id="test-id-1",
        download_url="https://example.com/photo.jpg",
        preview_url="https://example.com/thumb.jpg",
        page_url="https://example.com/page",
        width=800,
        height=600,
        photographer="Photographer",
        license_name="CC BY 4.0",
        query="face",
        source="openverse",
    )

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"fake-image-bytes"
        mock_get.return_value = mock_response

        download_res = source.download(result, tmp_path)

        assert download_res.success is True
        assert download_res.local_path is not None
        assert download_res.local_path.exists()
        assert download_res.local_path.read_bytes() == b"fake-image-bytes"
        assert download_res.local_path.name == "openverse_test-id-1.jpg"


def test_download_failure(source: OpenverseSource, tmp_path: Path) -> None:
    result = SearchResult(
        id="fail-id",
        download_url="https://example.com/fail.jpg",
        preview_url="",
        page_url="",
        width=800,
        height=600,
        photographer="",
        license_name="",
        query="",
        source="openverse",
    )

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_get.side_effect = requests.RequestException("Download failed")

        download_res = source.download(result, tmp_path)

        assert download_res.success is False
        assert download_res.local_path is None
        assert "Download failed" in download_res.error_message


def test_build_metadata(source: OpenverseSource, tmp_path: Path) -> None:
    result = SearchResult(
        id="meta-id",
        download_url="https://example.com/img.jpg",
        preview_url="",
        page_url="https://example.com/page",
        width=1024,
        height=768,
        photographer="Author",
        license_name="CC0",
        query="face",
        source="openverse",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
        license_type="pdm",
    )
    local_file = tmp_path / "openverse_meta_id.jpg"

    meta = source.build_metadata(result, local_file)
    assert meta.id == "meta-id"
    assert meta.source == "openverse"
    assert meta.local_path == local_file
    assert meta.download_url == result.download_url
    assert meta.width == 1024
    assert meta.height == 768
    assert meta.photographer == "Author"
    assert meta.license_name == "CC0"
    assert meta.license_url == "https://creativecommons.org/publicdomain/zero/1.0/"
    assert meta.license_type == "pdm"
