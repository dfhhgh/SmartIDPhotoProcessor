"""Tests for OpenverseSource implementation.

All tests use mocks/fixtures — no Openverse API access required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataset_acquisition.sources.openverse import (
    OpenverseSource,
    COMPATIBLE_LICENSES,
    OPENVERSE_API_BASE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_search_response(
    results: list[dict[str, Any]] | None = None,
    next_url: str | None = None,
) -> dict[str, Any]:
    """Build a fake Openverse search response."""
    if results is None:
        results = [
            {
                "identifier": "abc-123-def",
                "title": "Test Person Portrait",
                "creator": "Test Photographer",
                "creator_url": "https://example.com/photographer",
                "image_url": "https://images.openverse.org/abc-123-def.jpg",
                "foreign_landing_url": "https://openverse.org/image/abc-123-def",
                "license": "by",
                "source": "flickr",
                "width": 4000,
                "height": 3000,
                "description": "A test portrait photo",
            }
        ]
    response: dict[str, Any] = {"results": results}
    if next_url:
        response["next"] = next_url
    return response


def _make_error_response(status_code: int = 400, detail: str = "Bad Request") -> dict[str, Any]:
    """Build a fake Openverse error response."""
    return {"detail": detail}


def _mock_openverse_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {"Content-Type": "application/json"}
    return resp


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestOpenverseConfiguration:
    """Test OpenverseSource configuration and initialization."""

    def test_instantiates_without_env_vars(self) -> None:
        """Creates OpenverseSource when no env vars are required."""
        source = OpenverseSource()
        assert source.name == "openverse"

    def test_custom_license_filter(self) -> None:
        """Creates OpenverseSource with custom license filter."""
        custom = {"cc0", "pdm"}
        source = OpenverseSource(license_filter=custom)
        assert source.name == "openverse"

    def test_default_license_filter(self) -> None:
        """Uses COMPATIBLE_LICENSES as default."""
        source = OpenverseSource()
        assert source._license_filter == COMPATIBLE_LICENSES

    def test_rate_limit_errors_zero_initially(self) -> None:
        """Rate limit errors start at zero."""
        source = OpenverseSource()
        assert source.rate_limit_errors == 0

    def test_custom_throttle_settings(self) -> None:
        """Accepts custom throttle settings."""
        source = OpenverseSource(delay=2.0, max_retries=5, request_timeout=60)
        assert source._delay == 2.0
        assert source._max_retries == 5
        assert source._request_timeout == 60


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------


class TestOpenverseSearch:
    """Test OpenverseSource.search() behavior."""

    def test_search_yields_results(self) -> None:
        """search() yields SearchResult objects."""
        source = OpenverseSource()
        mock_resp = _mock_openverse_response(200, _make_search_response())

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search("test person", max_results=5))

        assert len(results) == 1
        assert results[0].source == "openverse"
        assert results[0].image_url == "https://images.openverse.org/abc-123-def.jpg"
        assert results[0].source_url == "https://openverse.org/image/abc-123-def"

    def test_search_respects_max_results(self) -> None:
        """search() respects max_results parameter."""
        source = OpenverseSource()
        resp = _make_search_response(
            results=[
                {
                    "identifier": f"img-{i}",
                    "title": f"Image {i}",
                    "image_url": f"https://images.openverse.org/img-{i}.jpg",
                    "license": "cc0",
                }
                for i in range(10)
            ]
        )
        mock_resp = _mock_openverse_response(200, resp)

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search("test", max_results=3))

        assert len(results) == 3

    def test_search_empty_results(self) -> None:
        """search() handles empty results."""
        source = OpenverseSource()
        mock_resp = _mock_openverse_response(200, _make_search_response(results=[]))

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search("nonexistent"))

        assert len(results) == 0

    def test_search_api_failure(self) -> None:
        """search() handles API failure gracefully."""
        source = OpenverseSource()
        mock_resp = _mock_openverse_response(500)

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search("test"))

        assert len(results) == 0

    def test_search_pagination(self) -> None:
        """search() paginates through multiple pages."""
        source = OpenverseSource()

        page1 = _make_search_response(
            results=[
                {
                    "identifier": f"img-{i}",
                    "title": f"Image {i}",
                    "image_url": f"https://images.openverse.org/img-{i}.jpg",
                    "license": "by",
                }
                for i in range(2)
            ],
            next_url=f"{OPENVERSE_API_BASE}/images/?page=2",
        )
        page2 = _make_search_response(
            results=[
                {
                    "identifier": f"img-{i+2}",
                    "title": f"Image {i+2}",
                    "image_url": f"https://images.openverse.org/img-{i+2}.jpg",
                    "license": "by",
                }
                for i in range(2)
            ],
            next_url=None,
        )

        call_count = 0

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_openverse_response(200, page1)
            return _mock_openverse_response(200, page2)

        with patch.object(source._session, "get", side_effect=mock_get):
            results = list(source.search("test", max_results=10))

        assert len(results) == 4
        assert call_count == 2

    def test_search_license_filter_applied(self) -> None:
        """search() includes license filter in API params."""
        source = OpenverseSource(license_filter={"cc0", "by"})
        mock_resp = _mock_openverse_response(200, _make_search_response())

        with patch.object(source._session, "get", return_value=mock_resp) as mock_get:
            list(source.search("test", max_results=1))

        call_args = mock_get.call_args
        params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
        assert "license" in params
        assert set(params["license"].split(",")) == {"cc0", "by"}


# ---------------------------------------------------------------------------
# Result Normalization Tests
# ---------------------------------------------------------------------------


class TestOpenverseNormalization:
    """Test OpenverseSource result normalization."""

    def test_normalize_result_with_all_fields(self) -> None:
        """Normalizes a complete result item."""
        source = OpenverseSource()
        item = {
            "identifier": "abc-123",
            "title": "Test Photo",
            "creator": "Jane Doe",
            "creator_url": "https://example.com/jane",
            "image_url": "https://images.openverse.org/abc-123.jpg",
            "foreign_landing_url": "https://openverse.org/image/abc-123",
            "license": "by",
            "source": "flickr",
            "width": 2000,
            "height": 1500,
            "description": "A test photo",
        }

        result = source._normalize_result(item, "test query")
        assert result is not None
        assert result.source == "openverse"
        assert result.title == "Test Photo"
        assert result.description == "A test photo"
        assert result.license == "CC BY 4.0"
        assert result.attribution == "Photo by Jane Doe"
        assert result.width == 2000
        assert result.height == 1500
        assert result.metadata["identifier"] == "abc-123"
        assert result.metadata["creator"] == "Jane Doe"
        assert result.metadata["license_slug"] == "by"

    def test_normalize_result_missing_identifier(self) -> None:
        """Returns None for items without identifier."""
        source = OpenverseSource()
        item = {"title": "No ID", "image_url": "https://example.com/img.jpg"}
        result = source._normalize_result(item, "query")
        assert result is None

    def test_normalize_result_missing_image_url(self) -> None:
        """Returns None for items without image_url."""
        source = OpenverseSource()
        item = {"identifier": "abc-123", "title": "No Image"}
        result = source._normalize_result(item, "query")
        assert result is None

    def test_normalize_result_fallback_source_url(self) -> None:
        """Builds fallback source URL when foreign_landing_url and creator_url are missing."""
        source = OpenverseSource()
        item = {
            "identifier": "abc-123",
            "image_url": "https://images.openverse.org/abc-123.jpg",
        }
        result = source._normalize_result(item, "query")
        assert result is not None
        assert result.source_url == "https://openverse.org/image/abc-123"


# ---------------------------------------------------------------------------
# License Normalization Tests
# ---------------------------------------------------------------------------


class TestOpenverseLicenseNormalization:
    """Test license slug normalization."""

    def test_normalize_license_cc_by(self) -> None:
        assert OpenverseSource._normalize_license("by") == "CC BY 4.0"

    def test_normalize_license_cc_by_sa(self) -> None:
        assert OpenverseSource._normalize_license("by-sa") == "CC BY-SA 4.0"

    def test_normalize_license_cc0(self) -> None:
        assert OpenverseSource._normalize_license("cc0") == "CC0 1.0"

    def test_normalize_license_pdm(self) -> None:
        assert OpenverseSource._normalize_license("pdm") == "Public Domain Mark"

    def test_normalize_license_unknown(self) -> None:
        assert OpenverseSource._normalize_license("unknown-license") == "unknown-license"

    def test_normalize_license_empty(self) -> None:
        assert OpenverseSource._normalize_license("") == "unknown"


# ---------------------------------------------------------------------------
# Download Tests
# ---------------------------------------------------------------------------


class TestOpenverseDownload:
    """Test OpenverseSource.download_url() behavior."""

    def test_download_success(self) -> None:
        """download_url() returns bytes on success."""
        source = OpenverseSource()
        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        mock_resp = _mock_openverse_response(200)
        mock_resp.content = fake_image
        mock_resp.headers = {"Content-Type": "image/jpeg"}

        def mock_iter(chunk_size: int = 8192) -> list[bytes]:
            return [fake_image]

        mock_resp.iter_content = mock_iter

        with patch.object(source._session, "get", return_value=mock_resp):
            result = source.download_url("https://images.openverse.org/test.jpg")

        assert result == fake_image

    def test_download_non_image_content_type(self) -> None:
        """download_url() returns None for non-image content."""
        source = OpenverseSource()
        mock_resp = _mock_openverse_response(200)
        mock_resp.headers = {"Content-Type": "text/html"}

        with patch.object(source._session, "get", return_value=mock_resp):
            result = source.download_url("https://example.com/page")

        assert result is None

    def test_download_http_error(self) -> None:
        """download_url() returns None on HTTP error."""
        source = OpenverseSource()
        mock_resp = _mock_openverse_response(404)

        with patch.object(source._session, "get", return_value=mock_resp):
            result = source.download_url("https://images.openverse.org/missing.jpg")

        assert result is None

    def test_download_rate_limit(self) -> None:
        """download_url() handles rate limiting."""
        source = OpenverseSource(max_rate_limit_retries=2)
        rate_limit_resp = _mock_openverse_response(429)
        rate_limit_resp.headers = {"Content-Type": "application/json", "Retry-After": "0.1"}
        success_resp = _mock_openverse_response(200)
        success_resp.headers = {"Content-Type": "image/jpeg"}

        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100

        def mock_iter(chunk_size: int = 8192) -> list[bytes]:
            return [fake_image]

        success_resp.iter_content = mock_iter

        with patch.object(source._session, "get", side_effect=[rate_limit_resp, success_resp]):
            result = source.download_url("https://images.openverse.org/test.jpg")

        assert result == fake_image
        assert source.rate_limit_errors == 1


# ---------------------------------------------------------------------------
# Throttle Tests
# ---------------------------------------------------------------------------


class TestOpenverseThrottle:
    """Test OpenverseSource throttle behavior."""

    def test_throttle_enforces_delay(self) -> None:
        """_throttle() sleeps when requests come too fast."""
        source = OpenverseSource(delay=0.1)
        source._last_request_time = time.time()

        with patch("dataset_acquisition.sources.openverse.time.sleep") as mock_sleep:
            source._throttle()
            mock_sleep.assert_called_once()

    def test_throttle_no_sleep_when_enough_time(self) -> None:
        """_throttle() does not sleep when enough time has passed."""
        source = OpenverseSource(delay=0.01)
        source._last_request_time = time.time() - 1.0  # 1 second ago

        with patch("dataset_acquisition.sources.openverse.time.sleep") as mock_sleep:
            source._throttle()
            mock_sleep.assert_not_called()


# ---------------------------------------------------------------------------
# Close Tests
# ---------------------------------------------------------------------------


class TestOpenverseClose:
    """Test OpenverseSource.close() behavior."""

    def test_close_session(self) -> None:
        """close() closes the HTTP session."""
        source = OpenverseSource()
        with patch.object(source._session, "close") as mock_close:
            source.close()
            mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestOpenverseIntegration:
    """Integration tests combining search and download."""

    def test_search_and_download_flow(self) -> None:
        """Full search → download flow works end-to-end."""
        source = OpenverseSource()

        search_resp = _make_search_response(
            results=[
                {
                    "identifier": "test-123",
                    "title": "Test Photo",
                    "image_url": "https://images.openverse.org/test-123.jpg",
                    "foreign_landing_url": "https://openverse.org/image/test-123",
                    "license": "cc0",
                    "width": 1000,
                    "height": 800,
                }
            ]
        )
        mock_search = _mock_openverse_response(200, search_resp)

        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        mock_download = _mock_openverse_response(200)
        mock_download.headers = {"Content-Type": "image/jpeg"}

        def mock_iter(chunk_size: int = 8192) -> list[bytes]:
            return [fake_image]

        mock_download.iter_content = mock_iter

        with patch.object(source._session, "get", side_effect=[mock_search, mock_download]):
            results = list(source.search("test person", max_results=1))
            assert len(results) == 1

            image_data = source.download_url(results[0].image_url)
            assert image_data == fake_image

        source.close()


import time  # noqa: E402
