"""Tests for FlickrSource implementation.

All tests use mocks/fixtures — no Flickr API access required.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataset_acquisition.sources.flickr import (
    FlickrSource,
    LICENSE_IDS_COMPATIBLE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure FLICKR_API_KEY is removed for each test."""
    monkeypatch.delenv("FLICKR_API_KEY", raising=False)


@pytest.fixture()
def _env_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake env var so FlickrSource can be instantiated."""
    monkeypatch.setenv("FLICKR_API_KEY", "test-flickr-key")


def _make_search_response(
    photos: list[dict[str, Any]] | None = None,
    total: int = 1,
    pages: int = 1,
) -> dict[str, Any]:
    """Build a fake Flickr search response."""
    if photos is None:
        photos = [
            {
                "id": "12345",
                "owner": "owner123",
                "ownername": "Test Photographer",
                "secret": "abc123",
                "server": "1234",
                "farm": 1,
                "title": "Test Person Portrait",
                "ispublic": 1,
                "license": 4,
                "description": {"_content": "A test portrait photo"},
                "dateupload": "1609459200",
                "datetaken": "2020-12-31 12:00:00",
                "tags": "portrait test celebrity",
                "width_z": "640",
                "height_z": "480",
                "url_z": "https://live.staticflickr.com/1234/12345_abc123_z.jpg",
            }
        ]
    return {
        "stat": "ok",
        "photos": {
            "page": 1,
            "pages": pages,
            "perpage": 100,
            "total": str(total),
            "photo": photos,
        },
    }


def _make_error_response(code: int = 100, message: str = "Invalid API Key") -> dict[str, Any]:
    """Build a fake Flickr error response."""
    return {"stat": "fail", "code": code, "message": message}


def _mock_flickr_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {"Content-Type": "application/json"}
    return resp


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestFlickrConfiguration:
    """Test FlickrSource configuration and initialization."""

    def test_requires_api_key(self) -> None:
        """Raises ValueError when FLICKR_API_KEY is missing."""
        with pytest.raises(ValueError, match="FLICKR_API_KEY"):
            FlickrSource()

    @pytest.mark.usefixtures("_env_ready")
    def test_instantiates_with_env_var(self) -> None:
        """Creates FlickrSource when env var is set."""
        src = FlickrSource()
        assert src.name == "flickr"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_name_property(self) -> None:
        """Name is 'flickr'."""
        src = FlickrSource()
        assert src.name == "flickr"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_custom_parameters(self) -> None:
        """Accepts custom delay, retry, timeout parameters."""
        src = FlickrSource(delay=1.0, max_retries=5, request_timeout=60)
        assert src._delay == 1.0
        assert src._max_retries == 5
        assert src._request_timeout == 60
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_default_license_filter(self) -> None:
        """Default license filter includes compatible licenses."""
        src = FlickrSource()
        assert 4 in src._license_filter  # CC BY 2.0
        assert 9 in src._license_filter  # CC0 1.0
        assert 0 not in src._license_filter  # All Rights Reserved
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_custom_license_filter(self) -> None:
        """Custom license filter overrides default."""
        custom = {9, 10}  # CC0 and Public Domain Mark only
        src = FlickrSource(license_filter=custom)
        assert src._license_filter == custom
        src.close()


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------


class TestFlickrSearch:
    """Test search API interaction and pagination."""

    @pytest.mark.usefixtures("_env_ready")
    def test_search_yields_results(self) -> None:
        """search() yields SearchResult objects for valid API responses."""
        src = FlickrSource()
        search_resp = _make_search_response()
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("Tom Hanks", max_results=5))

        assert len(results) == 1
        r = results[0]
        assert r.source == "flickr"
        assert r.metadata["photo_id"] == "12345"
        assert r.title == "Test Person Portrait"
        assert r.license == "CC BY 2.0"
        assert r.image_url == "https://live.staticflickr.com/1234/12345_abc123_z.jpg"
        assert r.metadata["owner"] == "owner123"
        assert r.metadata["owner_name"] == "Test Photographer"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_returns_nothing_on_api_error(self) -> None:
        """search() yields nothing on Flickr API error."""
        src = FlickrSource()
        error_resp = _make_error_response(100, "Invalid API Key")
        mock_resp = _mock_flickr_response(200, error_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test"))

        assert results == []
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_returns_nothing_on_http_error(self) -> None:
        """search() yields nothing on HTTP 500."""
        src = FlickrSource()
        mock_resp = _mock_flickr_response(500)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test"))

        assert results == []
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_skips_photos_without_id(self) -> None:
        """search() skips photos that lack an id."""
        src = FlickrSource()
        search_resp = _make_search_response(
            photos=[
                {"title": "no id"},  # missing id
                {
                    "id": "valid",
                    "owner": "owner1",
                    "ownername": "Valid Owner",
                    "secret": "abc",
                    "server": "1",
                    "farm": 1,
                    "title": "Valid",
                    "license": 4,
                    "description": {"_content": ""},
                    "dateupload": "",
                    "datetaken": "",
                    "tags": "",
                    "url_z": "https://live.staticflickr.com/1/valid_abc_z.jpg",
                },
            ],
            total=2,
        )
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=10))

        assert len(results) == 1
        assert results[0].metadata["photo_id"] == "valid"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_skips_photos_without_url(self) -> None:
        """search() skips photos with no image URL (missing id, secret, server)."""
        src = FlickrSource()
        search_resp = _make_search_response(
            photos=[
                {
                    "id": "nourl",
                    "owner": "owner1",
                    "ownername": "No URL",
                    # Missing secret and server — cannot build static URL
                    "title": "No URL",
                    "license": 4,
                    "description": {"_content": ""},
                },
                {
                    "id": "ok",
                    "owner": "owner2",
                    "ownername": "OK Owner",
                    "secret": "def",
                    "server": "2",
                    "farm": 2,
                    "title": "OK",
                    "license": 4,
                    "description": {"_content": ""},
                    "url_z": "https://live.staticflickr.com/2/ok_def_z.jpg",
                },
            ],
            total=2,
        )
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=10))

        assert len(results) == 1
        assert results[0].metadata["photo_id"] == "ok"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_respects_max_results(self) -> None:
        """search() stops yielding after max_results is reached."""
        src = FlickrSource()
        photos = [
            {
                "id": str(i),
                "owner": f"owner{i}",
                "ownername": f"Owner {i}",
                "secret": "abc",
                "server": "1",
                "farm": 1,
                "title": f"Image {i}",
                "license": 4,
                "description": {"_content": ""},
                "url_z": f"https://live.staticflickr.com/1/{i}_abc_z.jpg",
            }
            for i in range(5)
        ]
        search_resp = _make_search_response(photos=photos, total=5)
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=2))

        assert len(results) == 2
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_pagination(self) -> None:
        """search() paginates through multiple pages."""
        src = FlickrSource()
        page1 = _make_search_response(
            photos=[
                {
                    "id": "p1-1",
                    "owner": "owner1",
                    "ownername": "Owner1",
                    "secret": "a",
                    "server": "1",
                    "farm": 1,
                    "title": "Page 1",
                    "license": 4,
                    "description": {"_content": ""},
                    "url_z": "https://live.staticflickr.com/1/p1-1_a_z.jpg",
                }
            ],
            total=2,
            pages=2,
        )
        page2 = _make_search_response(
            photos=[
                {
                    "id": "p2-1",
                    "owner": "owner2",
                    "ownername": "Owner2",
                    "secret": "b",
                    "server": "2",
                    "farm": 2,
                    "title": "Page 2",
                    "license": 9,
                    "description": {"_content": ""},
                    "url_z": "https://live.staticflickr.com/2/p2-1_b_z.jpg",
                }
            ],
            total=2,
            pages=2,
        )

        resp1 = _mock_flickr_response(200, page1)
        resp2 = _mock_flickr_response(200, page2)

        with patch.object(src._session, "get", side_effect=[resp1, resp2]) as mock_get:
            results = list(src.search("test", max_results=10))

        assert len(results) == 2
        assert results[0].metadata["photo_id"] == "p1-1"
        assert results[1].metadata["photo_id"] == "p2-1"
        assert mock_get.call_count == 2
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_stops_on_empty_page(self) -> None:
        """search() stops when API returns empty photo list."""
        src = FlickrSource()
        empty_resp = _make_search_response(photos=[], total=0, pages=0)
        mock_resp = _mock_flickr_response(200, empty_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test"))

        assert results == []
        src.close()


# ---------------------------------------------------------------------------
# License Tests
# ---------------------------------------------------------------------------


class TestFlickrLicense:
    """Test license filtering and metadata."""

    @pytest.mark.usefixtures("_env_ready")
    def test_license_filter_applied_to_search(self) -> None:
        """Search request includes license parameter."""
        src = FlickrSource()
        search_resp = _make_search_response()
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp) as mock_get:
            list(src.search("test", max_results=5))

        # Check that license parameter was included
        call_args = mock_get.call_args
        params = call_args.kwargs.get("params") or call_args[1].get("params", {})
        assert "license" in params
        license_param = params["license"]
        assert "4" in license_param  # CC BY 2.0
        assert "9" in license_param  # CC0
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_license_id_to_name_mapping(self) -> None:
        """License IDs map to correct names."""
        assert FlickrSource._license_id_to_name(0) == "All Rights Reserved"
        assert FlickrSource._license_id_to_name(4) == "CC BY 2.0"
        assert FlickrSource._license_id_to_name(9) == "CC0 1.0"
        assert FlickrSource._license_id_to_name(10) == "Public Domain Mark"
        assert FlickrSource._license_id_to_name(11) == "CC BY 4.0"

    @pytest.mark.usefixtures("_env_ready")
    def test_search_result_includes_license_info(self) -> None:
        """SearchResult includes license name and attribution."""
        src = FlickrSource()
        search_resp = _make_search_response()
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test"))

        assert len(results) == 1
        r = results[0]
        assert r.license == "CC BY 2.0"
        assert "Flickr" in r.attribution
        assert r.metadata["license_id"] == 4
        src.close()


# ---------------------------------------------------------------------------
# Rate Limit Tests
# ---------------------------------------------------------------------------


class TestFlickrRateLimit:
    """Test rate-limit handling with retries."""

    @pytest.mark.usefixtures("_env_ready")
    def test_rate_limit_retries_with_backoff(self) -> None:
        """search() retries on 429 and returns results after backoff."""
        src = FlickrSource(delay=0.01, max_rate_limit_retries=3)
        rate_resp = _mock_flickr_response(429)
        success_resp = _mock_flickr_response(200, _make_search_response())

        with patch.object(src._session, "get", side_effect=[rate_resp, success_resp]):
            results = list(src.search("test", max_results=5))

        assert len(results) == 1
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_rate_limit_exhausted(self) -> None:
        """search() gives up after max_rate_limit_retries 429s."""
        src = FlickrSource(delay=0.01, max_rate_limit_retries=2)
        rate_resp = _mock_flickr_response(429)

        with patch.object(src._session, "get", return_value=rate_resp):
            results = list(src.search("test", max_results=5))

        assert results == []
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_rate_limit_retries(self) -> None:
        """download_url() retries on 429 and succeeds."""
        src = FlickrSource(delay=0.01, max_retries=3)
        rate_resp = _mock_flickr_response(429)
        success_resp = _mock_flickr_response(200)
        success_resp.headers = {"Content-Type": "image/jpeg"}
        success_resp.iter_content = MagicMock(return_value=[b"fake-image-bytes"])

        with patch.object(src._session, "get", side_effect=[rate_resp, success_resp]):
            result = src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert result == b"fake-image-bytes"
        src.close()


# ---------------------------------------------------------------------------
# Download Tests
# ---------------------------------------------------------------------------


class TestFlickrDownload:
    """Test image download functionality."""

    @pytest.mark.usefixtures("_env_ready")
    def test_download_success(self) -> None:
        """download_url() returns bytes on HTTP 200 with image content-type."""
        src = FlickrSource()
        resp = _mock_flickr_response(200)
        resp.headers = {"Content-Type": "image/jpeg"}
        resp.iter_content = MagicMock(return_value=[b"\x89PNG", b"more-data"])

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert result == b"\x89PNGmore-data"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_returns_none_on_non_image_content_type(self) -> None:
        """download_url() returns None for non-image content-type."""
        src = FlickrSource()
        resp = _mock_flickr_response(200)
        resp.headers = {"Content-Type": "text/html"}

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_returns_none_on_404(self) -> None:
        """download_url() returns None on HTTP 404."""
        src = FlickrSource()
        resp = _mock_flickr_response(404)

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_returns_none_on_network_error(self) -> None:
        """download_url() returns None after exhausting retries."""
        src = FlickrSource(max_retries=1, delay=0.01)
        import requests as _requests

        with patch.object(src._session, "get", side_effect=_requests.ConnectionError("fail")):
            result = src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_respects_max_size(self) -> None:
        """download_url() returns None when download exceeds 50 MB."""
        src = FlickrSource()
        resp = _mock_flickr_response(200)
        resp.headers = {"Content-Type": "image/jpeg"}
        big_chunk = b"x" * (10 * 1024 * 1024)  # 10 MB chunk
        resp.iter_content = MagicMock(return_value=[big_chunk] * 7)  # 70 MB total

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert result is None
        src.close()


# ---------------------------------------------------------------------------
# Photo Normalization Tests
# ---------------------------------------------------------------------------


class TestFlickrNormalization:
    """Test photo normalization and URL building."""

    @pytest.mark.usefixtures("_env_ready")
    def test_normalize_photo_metadata(self) -> None:
        """Normalized result includes expected metadata fields."""
        src = FlickrSource()
        photo = {
            "id": "55555",
            "owner": "owner555",
            "ownername": "Photo User",
            "secret": "xyz",
            "server": "555",
            "farm": 5,
            "title": "Celebrity Portrait",
            "license": 9,
            "description": {"_content": "CC0 portrait"},
            "dateupload": "1609459200",
            "datetaken": "2020-12-31 12:00:00",
            "tags": "portrait celebrity",
            "width_z": "640",
            "height_z": "480",
            "url_z": "https://live.staticflickr.com/555/55555_xyz_z.jpg",
        }

        result = src._normalize_photo(photo, "celebrity")

        assert result is not None
        assert result.metadata["photo_id"] == "55555"
        assert result.metadata["owner"] == "owner555"
        assert result.metadata["owner_name"] == "Photo User"
        assert result.metadata["license_id"] == 9
        assert result.license == "CC0 1.0"
        assert result.source_url == "https://www.flickr.com/photos/owner555/55555"
        assert result.width == 640
        assert result.height == 480
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_normalize_returns_none_for_no_id(self) -> None:
        """_normalize_photo returns None for photo without id."""
        src = FlickrSource()
        photo = {"title": "no id"}
        result = src._normalize_photo(photo, "query")
        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_build_static_url(self) -> None:
        """_build_static_url constructs URL from photo fields."""
        src = FlickrSource()
        photo = {
            "id": "123",
            "secret": "abc",
            "server": "456",
            "farm": 7,
        }
        url = src._build_static_url(photo)
        assert url == "https://live.staticflickr.com/456/123_abc_b.jpg"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_build_static_url_incomplete(self) -> None:
        """_build_static_url returns empty string for incomplete photo."""
        src = FlickrSource()
        photo = {"id": "123"}  # missing secret and server
        url = src._build_static_url(photo)
        assert url == ""
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_description_dict_handling(self) -> None:
        """search() handles Flickr's dict-format description correctly."""
        src = FlickrSource()
        search_resp = _make_search_response(
            photos=[
                {
                    "id": "desc-test",
                    "owner": "owner1",
                    "ownername": "Owner1",
                    "secret": "a",
                    "server": "1",
                    "farm": 1,
                    "title": "Desc Test",
                    "license": 4,
                    "description": {"_content": "This is a description"},
                    "url_z": "https://live.staticflickr.com/1/desc-test_a_z.jpg",
                }
            ]
        )
        mock_resp = _mock_flickr_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test"))

        assert len(results) == 1
        assert results[0].description == "This is a description"
        src.close()


# ---------------------------------------------------------------------------
# ImageSource Interface Compatibility
# ---------------------------------------------------------------------------


class TestFlickrInterfaceCompatibility:
    """Test that FlickrSource satisfies the ImageSource interface."""

    @pytest.mark.usefixtures("_env_ready")
    def test_has_required_methods(self) -> None:
        """FlickrSource has search, download_url, close methods."""
        src = FlickrSource()
        assert hasattr(src, "search")
        assert hasattr(src, "download_url")
        assert hasattr(src, "close")
        assert callable(src.search)
        assert callable(src.download_url)
        assert callable(src.close)
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_name_is_string(self) -> None:
        """name property returns a non-empty string."""
        src = FlickrSource()
        assert isinstance(src.name, str)
        assert len(src.name) > 0
        src.close()


# ---------------------------------------------------------------------------
# Secret-Safe Logging Tests
# ---------------------------------------------------------------------------


class TestFlickrSecretSafety:
    """Verify secrets are not exposed in logs or repr."""

    @pytest.mark.usefixtures("_env_ready")
    def test_api_key_not_in_repr(self) -> None:
        """API key is not present in object repr."""
        src = FlickrSource()
        r = repr(src)
        assert "test-flickr-key" not in r
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_api_key_not_in_search_error_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """API key is not logged in search failure messages."""
        src = FlickrSource()
        error_resp = _make_error_response(100, "Invalid API Key")
        mock_resp = _mock_flickr_response(200, error_resp)

        with caplog.at_level("WARNING"):
            with patch.object(src._session, "get", return_value=mock_resp):
                list(src.search("test"))

        assert "test-flickr-key" not in caplog.text
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_api_key_not_in_download_error_logs(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """API key is not logged in download failure messages."""
        src = FlickrSource(max_retries=0, delay=0.01)
        resp = _mock_flickr_response(404)

        with caplog.at_level("WARNING"):
            with patch.object(src._session, "get", return_value=resp):
                src.download_url("https://live.staticflickr.com/1/123_abc_z.jpg")

        assert "test-flickr-key" not in caplog.text
        src.close()


# ---------------------------------------------------------------------------
# Context Manager / Close Tests
# ---------------------------------------------------------------------------


class TestFlickrCleanup:
    """Test cleanup and close behavior."""

    @pytest.mark.usefixtures("_env_ready")
    def test_close_does_not_raise(self) -> None:
        """close() completes without error."""
        src = FlickrSource()
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_session_closed_on_close(self) -> None:
        """close() closes the underlying requests session."""
        src = FlickrSource()
        with patch.object(src._session, "close") as mock_close:
            src.close()
        mock_close.assert_called_once()


# ---------------------------------------------------------------------------
# License Compatibility Constants
# ---------------------------------------------------------------------------


class TestLicenseConstants:
    """Test that license constants are correctly defined."""

    def test_compatible_licenses_defined(self) -> None:
        """LICENSE_IDS_COMPATIBLE contains expected license IDs."""
        assert 4 in LICENSE_IDS_COMPATIBLE  # CC BY 2.0
        assert 5 in LICENSE_IDS_COMPATIBLE  # CC BY-SA 2.0
        assert 9 in LICENSE_IDS_COMPATIBLE  # CC0 1.0
        assert 10 in LICENSE_IDS_COMPATIBLE  # Public Domain Mark
        assert 11 in LICENSE_IDS_COMPATIBLE  # CC BY 4.0
        assert 12 in LICENSE_IDS_COMPATIBLE  # CC BY-SA 4.0

    def test_incompatible_licenses_excluded(self) -> None:
        """LICENSE_IDS_COMPATIBLE excludes restrictive licenses."""
        assert 0 not in LICENSE_IDS_COMPATIBLE  # All Rights Reserved
        assert 1 not in LICENSE_IDS_COMPATIBLE  # CC BY-NC-SA 2.0
        assert 2 not in LICENSE_IDS_COMPATIBLE  # CC BY-NC 2.0
        assert 3 not in LICENSE_IDS_COMPATIBLE  # CC BY-NC-ND 2.0
        assert 6 not in LICENSE_IDS_COMPATIBLE  # CC BY-ND 2.0
        assert 7 not in LICENSE_IDS_COMPATIBLE  # No known copyright
        assert 13 not in LICENSE_IDS_COMPATIBLE  # CC BY-ND 4.0
        assert 14 not in LICENSE_IDS_COMPATIBLE  # CC BY-NC 4.0
        assert 15 not in LICENSE_IDS_COMPATIBLE  # CC BY-NC-SA 4.0
        assert 16 not in LICENSE_IDS_COMPATIBLE  # CC BY-NC-ND 4.0
