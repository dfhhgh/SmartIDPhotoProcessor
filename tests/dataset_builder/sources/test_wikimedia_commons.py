"""Unit tests for WikimediaCommonsSource."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from dataset_builder.config.settings import Settings
from dataset_builder.sources.base_source import SearchResult
from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        DATASET_DIR=tmp_path / "dataset",
        MIN_IMAGE_WIDTH=640,
        MIN_IMAGE_HEIGHT=480,
    )


@pytest.fixture
def source(settings: Settings) -> WikimediaCommonsSource:
    return WikimediaCommonsSource(settings)


def test_source_name(source: WikimediaCommonsSource) -> None:
    assert source.name == "wikimedia_commons"


def test_validate_configuration_success(source: WikimediaCommonsSource) -> None:
    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        assert source.validate_configuration() is True


def test_validate_configuration_failure(source: WikimediaCommonsSource) -> None:
    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_get.side_effect = requests.RequestException("API error")

        with pytest.raises(requests.RequestException):
            source.validate_configuration()


def test_search_success(source: WikimediaCommonsSource) -> None:
    mock_payload = {
        "query": {
            "pages": {
                "12345": {
                    "pageid": 12345,
                    "ns": 6,
                    "title": "File:Test Face Portrait.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/e/e7/Test_Face_Portrait.jpg",
                            "width": 1200,
                            "height": 900,
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Test_Face_Portrait.jpg",
                            "extmetadata": {
                                "Artist": {"value": "<a href=\"#\">John Smith</a>"},
                                "LicenseShortName": {"value": "CC BY-SA 4.0"},
                                "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
                                "License": {"value": "cc-by-sa-4.0"},
                            },
                        }
                    ],
                }
            }
        }
    }

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_get.return_value = mock_response

        results = source.search("face", page=1, per_page=10)

        assert len(results) == 1
        res = results[0]
        assert res.id == "12345"
        assert res.download_url == "https://upload.wikimedia.org/wikipedia/commons/e/e7/Test_Face_Portrait.jpg"
        assert res.width == 1200
        assert res.height == 900
        assert res.photographer == "John Smith"
        assert res.license_name == "CC BY-SA 4.0"
        assert res.license_url == "https://creativecommons.org/licenses/by-sa/4.0"
        assert res.license_type == "cc-by-sa-4.0"
        assert res.source == "wikimedia_commons"


def test_search_filters_small_images(source: WikimediaCommonsSource) -> None:
    mock_payload = {
        "query": {
            "pages": {
                "54321": {
                    "pageid": 54321,
                    "title": "File:Small.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/small.jpg",
                            "width": 300,  # Below MIN_IMAGE_WIDTH (640)
                            "height": 200,  # Below MIN_IMAGE_HEIGHT (480)
                            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Small.jpg",
                            "extmetadata": {},
                        }
                    ],
                }
            }
        }
    }

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_payload
        mock_get.return_value = mock_response

        results = source.search("face", page=1, per_page=10)
        assert len(results) == 0


def test_search_empty_results(source: WikimediaCommonsSource) -> None:
    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"query": {"pages": {}}}
        mock_get.return_value = mock_response

        results = source.search("nonexistent", page=1, per_page=10)
        assert len(results) == 0


def test_download_success(source: WikimediaCommonsSource, tmp_path: Path) -> None:
    result = SearchResult(
        id="9999",
        download_url="https://upload.wikimedia.org/wikipedia/commons/test.jpg",
        preview_url="",
        page_url="",
        width=1000,
        height=800,
        photographer="Artist",
        license_name="CC BY 3.0",
        query="face",
        source="wikimedia_commons",
    )

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"wikimedia-image-bytes"
        mock_get.return_value = mock_response

        download_res = source.download(result, tmp_path)

        assert download_res.success is True
        assert download_res.local_path is not None
        assert download_res.local_path.exists()
        assert download_res.local_path.read_bytes() == b"wikimedia-image-bytes"
        assert download_res.local_path.name == "wikimedia_9999.jpg"


def test_download_failure(source: WikimediaCommonsSource, tmp_path: Path) -> None:
    result = SearchResult(
        id="fail-id",
        download_url="https://upload.wikimedia.org/wikipedia/commons/fail.jpg",
        preview_url="",
        page_url="",
        width=800,
        height=600,
        photographer="",
        license_name="",
        query="",
        source="wikimedia_commons",
    )

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_get.side_effect = requests.RequestException("Wikimedia download failed")

        download_res = source.download(result, tmp_path)

        assert download_res.success is False
        assert download_res.local_path is None
        assert "Wikimedia download failed" in download_res.error_message


def test_build_metadata(source: WikimediaCommonsSource, tmp_path: Path) -> None:
    result = SearchResult(
        id="meta-wm",
        download_url="https://upload.wikimedia.org/wikipedia/commons/img.jpg",
        preview_url="",
        page_url="https://commons.wikimedia.org/wiki/File:Img.jpg",
        width=1920,
        height=1080,
        photographer="Wiki Author",
        license_name="CC BY-SA 3.0",
        query="face",
        source="wikimedia_commons",
        license_url="https://creativecommons.org/licenses/by-sa/3.0",
        license_type="cc-by-sa-3.0",
    )
    local_file = tmp_path / "wikimedia_meta_wm.jpg"

    meta = source.build_metadata(result, local_file)
    assert meta.id == "meta-wm"
    assert meta.source == "wikimedia_commons"
    assert meta.local_path == local_file
    assert meta.download_url == result.download_url
    assert meta.width == 1920
    assert meta.height == 1080
    assert meta.photographer == "Wiki Author"
    assert meta.license_name == "CC BY-SA 3.0"
    assert meta.license_url == "https://creativecommons.org/licenses/by-sa/3.0"
    assert meta.license_type == "cc-by-sa-3.0"


def test_user_agent_header_is_descriptive(source: WikimediaCommonsSource) -> None:
    """Verify the custom User-Agent follows Wikimedia's policy.

    Wikimedia requires: <client name>/<version> (<contact information>)
    See: https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
    """
    ua = source._custom_headers.get("User-Agent", "")
    # Must not be a generic/default agent
    assert "python-requests" not in ua.lower()
    assert "DatasetBuilder/1.0 (+Research)" not in ua
    # Must contain contact info (URL or email)
    has_url = "http" in ua.lower()
    has_email = "@" in ua
    assert has_url or has_email, f"User-Agent must contain contact info: {ua}"
    # Must identify the client
    assert "SmartIDPhotoProcessor" in ua or "DatasetBuilder" in ua


def test_user_agent_sent_on_download(
    source: WikimediaCommonsSource, tmp_path: Path
) -> None:
    """Verify the download request sends the Wikimedia-specific User-Agent."""
    result = SearchResult(
        id="ua-test",
        download_url="https://upload.wikimedia.org/wikipedia/commons/test.jpg",
        preview_url="",
        page_url="",
        width=800,
        height=600,
        photographer="Test",
        license_name="CC BY",
        query="face",
        source="wikimedia_commons",
    )

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"image-bytes"
        mock_get.return_value = mock_response

        source.download(result, tmp_path)

        # Verify headers were passed to the HTTP client
        call_kwargs = mock_get.call_args
        headers = call_kwargs.kwargs.get("headers", call_kwargs[1].get("headers", {}))
        assert "User-Agent" in headers
        ua = headers["User-Agent"]
        assert "SmartIDPhotoProcessor" in ua
        assert "http" in ua.lower() or "@" in ua


def test_download_handles_403_forbidden(
    source: WikimediaCommonsSource, tmp_path: Path
) -> None:
    """Verify 403 is handled gracefully and classified as access_forbidden."""
    result = SearchResult(
        id="forbidden-id",
        download_url="https://upload.wikimedia.org/wikipedia/commons/forbidden.jpg",
        preview_url="",
        page_url="",
        width=800,
        height=600,
        photographer="",
        license_name="",
        query="",
        source="wikimedia_commons",
    )

    with patch("dataset_builder.utils.http_client.HTTPClient.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 403
        http_error = requests.HTTPError(
            "403 Client Error: Forbidden",
            response=mock_response,
        )
        mock_get.side_effect = http_error

        download_res = source.download(result, tmp_path)

        assert download_res.success is False
        assert download_res.local_path is None
        assert "403" in download_res.error_message or "Forbidden" in download_res.error_message
