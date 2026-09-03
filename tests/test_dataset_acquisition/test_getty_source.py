"""Tests for GettySource implementation.

All tests use mocks/fixtures — no Getty API access required.
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataset_acquisition.sources.getty import GettySource, GETTY_API_BASE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure GETTY_API_KEY and GETTY_API_SECRET are removed for each test."""
    monkeypatch.delenv("GETTY_API_KEY", raising=False)
    monkeypatch.delenv("GETTY_API_SECRET", raising=False)


@pytest.fixture()
def _env_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set fake env vars so GettySource can be instantiated."""
    monkeypatch.setenv("GETTY_API_KEY", "test-api-key")
    monkeypatch.setenv("GETTY_API_SECRET", "test-api-secret")


def _make_search_response(
    images: list[dict[str, Any]] | None = None,
    result_count: int = 1,
) -> dict[str, Any]:
    """Build a fake Getty search response."""
    if images is None:
        images = [
            {
                "id": "12345",
                "title": "Test Person Portrait",
                "asset_family": "creative",
                "caption": "A test portrait photo",
                "license_model": "royaltyfree",
                "max_dimensions": {"width": 4000, "height": 3000},
                "display_sizes": [
                    {"name": "thumb", "uri": "https://media.gettyimages.com/thumb/12345.jpg"},
                    {"name": "comp", "uri": "https://media.gettyimages.com/comp/12345.jpg"},
                ],
                "collection_name": "TestCollection",
            }
        ]
    return {"result_count": result_count, "images": images}


def _make_auth_response(access_token: str = "fake-token") -> dict[str, Any]:
    """Build a fake Getty auth response."""
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 1800,
    }


def _mock_getty_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {"content-type": "application/json"}
    return resp


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestGettyConfiguration:
    """Test GettySource configuration and initialization."""

    def test_requires_api_key(self) -> None:
        """Raises ValueError when GETTY_API_KEY is missing."""
        with pytest.raises(ValueError, match="GETTY_API_KEY"):
            GettySource()

    def test_requires_api_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raises ValueError when GETTY_API_SECRET is missing."""
        monkeypatch.setenv("GETTY_API_KEY", "key-only")
        with pytest.raises(ValueError, match="GETTY_API_SECRET"):
            GettySource()

    @pytest.mark.usefixtures("_env_ready")
    def test_instantiates_with_env_vars(self) -> None:
        """Creates GettySource when both env vars are set."""
        src = GettySource()
        assert src.name == "getty_images"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_name_property(self) -> None:
        """Name is 'getty_images'."""
        src = GettySource()
        assert src.name == "getty_images"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_custom_parameters(self) -> None:
        """Accepts custom delay, retry, timeout parameters."""
        src = GettySource(delay=1.0, max_retries=5, request_timeout=60)
        assert src._delay == 1.0
        assert src._max_retries == 5
        assert src._request_timeout == 60
        src.close()


# ---------------------------------------------------------------------------
# Authentication Tests
# ---------------------------------------------------------------------------


class TestGettyAuthentication:
    """Test OAuth2 authentication flow."""

    @pytest.mark.usefixtures("_env_ready")
    def test_authenticate_success(self) -> None:
        """Successful auth sets access token and Authorization header."""
        src = GettySource()
        mock_resp = _mock_getty_response(200, _make_auth_response())

        with patch.object(src._session, "post", return_value=mock_resp) as mock_post:
            result = src._authenticate()

        assert result is True
        assert src._access_token == "fake-token"
        assert "Authorization" in src._session.headers
        assert src._session.headers["Authorization"] == "Bearer fake-token"
        mock_post.assert_called_once()
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_authenticate_already_has_token(self) -> None:
        """Returns True without re-authenticating if token exists."""
        src = GettySource()
        src._access_token = "existing-token"

        with patch.object(src._session, "post") as mock_post:
            result = src._authenticate()

        assert result is True
        mock_post.assert_not_called()
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_authenticate_failure_401(self) -> None:
        """Returns False on HTTP 401."""
        src = GettySource()
        mock_resp = _mock_getty_response(401)

        with patch.object(src._session, "post", return_value=mock_resp):
            result = src._authenticate()

        assert result is False
        assert src._access_token is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_authenticate_missing_token_in_response(self) -> None:
        """Returns False when response JSON lacks access_token."""
        src = GettySource()
        mock_resp = _mock_getty_response(200, {"token_type": "Bearer"})

        with patch.object(src._session, "post", return_value=mock_resp):
            result = src._authenticate()

        assert result is False
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_authenticate_network_error(self) -> None:
        """Returns False on network exception."""
        src = GettySource()

        import requests as _requests

        with patch.object(
            src._session, "post", side_effect=_requests.ConnectionError("fail")
        ):
            result = src._authenticate()

        assert result is False
        src.close()


# ---------------------------------------------------------------------------
# Search Tests
# ---------------------------------------------------------------------------


class TestGettySearch:
    """Test search API interaction and pagination."""

    @pytest.mark.usefixtures("_env_ready")
    def test_search_yields_results(self) -> None:
        """search() yields SearchResult objects for valid API responses."""
        src = GettySource()
        src._access_token = "fake-token"

        search_resp = _make_search_response()
        mock_resp = _mock_getty_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("tom hanks", max_results=5))

        assert len(results) == 1
        r = results[0]
        assert r.source == "getty_images"
        assert r.metadata["image_id"] == "12345"
        assert r.title == "Test Person Portrait"
        assert r.width == 4000
        assert r.height == 3000
        assert r.license == "royaltyfree"
        assert r.image_url == "https://media.gettyimages.com/comp/12345.jpg"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_prefers_comp_over_thumb(self) -> None:
        """search() picks comp URI over thumb when available."""
        src = GettySource()
        src._access_token = "fake-token"

        search_resp = _make_search_response(
            images=[
                {
                    "id": "99999",
                    "title": "Comp preferred",
                    "asset_family": "creative",
                    "caption": "test",
                    "license_model": "royaltyfree",
                    "max_dimensions": {"width": 2000, "height": 1500},
                    "display_sizes": [
                        {"name": "thumb", "uri": "https://media.gettyimages.com/thumb/99999.jpg"},
                        {"name": "comp", "uri": "https://media.gettyimages.com/comp/99999.jpg"},
                    ],
                }
            ]
        )
        mock_resp = _mock_getty_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=5))

        assert len(results) == 1
        assert results[0].image_url == "https://media.gettyimages.com/comp/99999.jpg"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_returns_nothing_on_auth_failure(self) -> None:
        """search() yields nothing when authentication fails."""
        src = GettySource()
        mock_resp = _mock_getty_response(401)

        with patch.object(src._session, "post", return_value=mock_resp):
            results = list(src.search("test"))

        assert results == []
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_returns_nothing_on_api_error(self) -> None:
        """search() yields nothing on HTTP 500."""
        src = GettySource()
        src._access_token = "fake-token"

        mock_resp = _mock_getty_response(500)
        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test"))

        assert results == []
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_skips_images_without_id(self) -> None:
        """search() skips images that lack an id field."""
        src = GettySource()
        src._access_token = "fake-token"

        search_resp = {
            "result_count": 2,
            "images": [
                {"title": "no id image"},  # missing id
                {
                    "id": "valid",
                    "title": "Valid image",
                    "asset_family": "creative",
                    "caption": "valid",
                    "license_model": "royaltyfree",
                    "max_dimensions": {"width": 1000, "height": 800},
                    "display_sizes": [
                        {"name": "comp", "uri": "https://media.gettyimages.com/comp/valid.jpg"}
                    ],
                },
            ],
        }
        mock_resp = _mock_getty_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=10))

        assert len(results) == 1
        assert results[0].metadata["image_id"] == "valid"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_skips_images_without_display_uri(self) -> None:
        """search() skips images with no display_sizes."""
        src = GettySource()
        src._access_token = "fake-token"

        search_resp = {
            "result_count": 2,
            "images": [
                {"id": "nodisplay", "title": "No display", "display_sizes": []},
                {
                    "id": "ok",
                    "title": "OK image",
                    "asset_family": "creative",
                    "caption": "ok",
                    "license_model": "royaltyfree",
                    "max_dimensions": {"width": 1000, "height": 800},
                    "display_sizes": [
                        {"name": "comp", "uri": "https://media.gettyimages.com/comp/ok.jpg"}
                    ],
                },
            ],
        }
        mock_resp = _mock_getty_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=10))

        assert len(results) == 1
        assert results[0].metadata["image_id"] == "ok"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_respects_max_results(self) -> None:
        """search() stops yielding after max_results is reached."""
        src = GettySource()
        src._access_token = "fake-token"

        images = [
            {
                "id": str(i),
                "title": f"Image {i}",
                "asset_family": "creative",
                "caption": f"caption {i}",
                "license_model": "royaltyfree",
                "max_dimensions": {"width": 1000, "height": 800},
                "display_sizes": [
                    {"name": "comp", "uri": f"https://media.gettyimages.com/comp/{i}.jpg"}
                ],
            }
            for i in range(5)
        ]
        search_resp = _make_search_response(images=images, result_count=5)
        mock_resp = _mock_getty_response(200, search_resp)

        with patch.object(src._session, "get", return_value=mock_resp):
            results = list(src.search("test", max_results=2))

        assert len(results) == 2
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_search_pagination(self) -> None:
        """search() paginates through multiple pages."""
        src = GettySource()
        src._access_token = "fake-token"

        # result_count=25 forces 2 pages (page_size=20)
        page1 = {
            "result_count": 25,
            "images": [
                {
                    "id": f"p1-{i}",
                    "title": f"Page 1 Image {i}",
                    "asset_family": "creative",
                    "caption": f"p1-{i}",
                    "license_model": "royaltyfree",
                    "max_dimensions": {"width": 1000, "height": 800},
                    "display_sizes": [
                        {"name": "comp", "uri": f"https://media.gettyimages.com/comp/p1-{i}.jpg"}
                    ],
                }
                for i in range(20)
            ],
        }
        page2 = {
            "result_count": 25,
            "images": [
                {
                    "id": f"p2-{i}",
                    "title": f"Page 2 Image {i}",
                    "asset_family": "creative",
                    "caption": f"p2-{i}",
                    "license_model": "royaltyfree",
                    "max_dimensions": {"width": 1000, "height": 800},
                    "display_sizes": [
                        {"name": "comp", "uri": f"https://media.gettyimages.com/comp/p2-{i}.jpg"}
                    ],
                }
                for i in range(5)
            ],
        }

        resp1 = _mock_getty_response(200, page1)
        resp2 = _mock_getty_response(200, page2)

        with patch.object(src._session, "get", side_effect=[resp1, resp2]) as mock_get:
            results = list(src.search("test", max_results=10))

        # Should stop after 10 results (20 on page 1, but max_results=10)
        assert len(results) == 10
        assert results[0].metadata["image_id"] == "p1-0"
        assert results[9].metadata["image_id"] == "p1-9"
        assert mock_get.call_count == 1  # Only page 1 needed
        src.close()


# ---------------------------------------------------------------------------
# Rate Limit Tests
# ---------------------------------------------------------------------------


class TestGettyRateLimit:
    """Test rate-limit handling with retries."""

    @pytest.mark.usefixtures("_env_ready")
    def test_rate_limit_retries_with_backoff(self) -> None:
        """search() retries on 429 and returns results after backoff."""
        src = GettySource(delay=0.01, max_rate_limit_retries=3)
        src._access_token = "fake-token"

        rate_limit_resp = _mock_getty_response(429)
        success_resp = _mock_getty_response(200, _make_search_response())

        with patch.object(src._session, "get", side_effect=[rate_limit_resp, success_resp]):
            results = list(src.search("test", max_results=5))

        assert len(results) == 1
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_rate_limit_exhausted(self) -> None:
        """search() gives up after max_rate_limit_retries 429s."""
        src = GettySource(delay=0.01, max_rate_limit_retries=2)
        src._access_token = "fake-token"

        rate_limit_resp = _mock_getty_response(429)

        with patch.object(src._session, "get", return_value=rate_limit_resp):
            results = list(src.search("test", max_results=5))

        assert results == []
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_rate_limit_retries(self) -> None:
        """download_url() retries on 429 and succeeds."""
        src = GettySource(delay=0.01, max_retries=3)
        src._access_token = "fake-token"

        rate_limit_resp = _mock_getty_response(429)
        success_resp = _mock_getty_response(200)
        success_resp.headers = {"content-type": "image/jpeg"}
        success_resp.iter_content = MagicMock(return_value=[b"fake-image-bytes"])

        with patch.object(
            src._session, "get", side_effect=[rate_limit_resp, success_resp]
        ):
            result = src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert result == b"fake-image-bytes"
        src.close()


# ---------------------------------------------------------------------------
# Download Tests
# ---------------------------------------------------------------------------


class TestGettyDownload:
    """Test image download functionality."""

    @pytest.mark.usefixtures("_env_ready")
    def test_download_success(self) -> None:
        """download_url() returns bytes on HTTP 200 with image content-type."""
        src = GettySource()
        src._access_token = "fake-token"

        resp = _mock_getty_response(200)
        resp.headers = {"content-type": "image/jpeg"}
        resp.iter_content = MagicMock(return_value=[b"\x89PNG", b"more-data"])

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert result == b"\x89PNGmore-data"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_returns_none_on_non_image_content_type(self) -> None:
        """download_url() returns None for non-image content-type."""
        src = GettySource()
        src._access_token = "fake-token"

        resp = _mock_getty_response(200)
        resp.headers = {"content-type": "text/html"}

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_returns_none_on_404(self) -> None:
        """download_url() returns None on HTTP 404."""
        src = GettySource()
        src._access_token = "fake-token"

        resp = _mock_getty_response(404)

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_returns_none_on_network_error(self) -> None:
        """download_url() returns None after exhausting retries."""
        src = GettySource(max_retries=1, delay=0.01)
        src._access_token = "fake-token"

        import requests as _requests

        with patch.object(
            src._session, "get", side_effect=_requests.ConnectionError("fail")
        ):
            result = src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_download_respects_max_size(self) -> None:
        """download_url() returns None when download exceeds 50 MB."""
        src = GettySource()
        src._access_token = "fake-token"

        resp = _mock_getty_response(200)
        resp.headers = {"content-type": "image/jpeg"}
        # Simulate 60 MB response
        big_chunk = b"x" * (10 * 1024 * 1024)  # 10 MB chunk
        resp.iter_content = MagicMock(return_value=[big_chunk] * 7)  # 70 MB total

        with patch.object(src._session, "get", return_value=resp):
            result = src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert result is None
        src.close()


# ---------------------------------------------------------------------------
# Search Result Normalization Tests
# ---------------------------------------------------------------------------


class TestGettyNormalization:
    """Test search result normalization."""

    @pytest.mark.usefixtures("_env_ready")
    def test_normalize_search_result_metadata(self) -> None:
        """Normalized result includes expected metadata fields."""
        src = GettySource()
        img = {
            "id": "55555",
            "title": "Celebrity Portrait",
            "asset_family": "editorial",
            "caption": "Red carpet photo",
            "license_model": "rightsmanaged",
            "max_dimensions": {"width": 6000, "height": 4000},
            "display_sizes": [
                {"name": "comp", "uri": "https://media.gettyimages.com/comp/55555.jpg"}
            ],
            "collection_name": "GettyImages",
        }

        result = src._normalize_search_result(img, "celebrity")

        assert result is not None
        assert result.metadata["image_id"] == "55555"
        assert result.metadata["asset_family"] == "editorial"
        assert result.metadata["collection_name"] == "GettyImages"
        assert result.metadata["query"] == "celebrity"
        assert result.license == "rightsmanaged"
        assert result.source_url == "https://www.gettyimages.com/photos/55555"
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_normalize_returns_none_for_no_id(self) -> None:
        """_normalize_search_result returns None for image without id."""
        src = GettySource()
        img = {"title": "no id"}
        result = src._normalize_search_result(img, "query")
        assert result is None
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_display_uri_priority(self) -> None:
        """_extract_best_display_uri follows preferred order."""
        src = GettySource()

        # Thumb only
        img1 = {"display_sizes": [{"name": "thumb", "uri": "https://thumb.jpg"}]}
        assert src._extract_best_display_uri(img1) == "https://thumb.jpg"

        # Comp preferred over thumb
        img2 = {
            "display_sizes": [
                {"name": "thumb", "uri": "https://thumb.jpg"},
                {"name": "comp", "uri": "https://comp.jpg"},
            ]
        }
        assert src._extract_best_display_uri(img2) == "https://comp.jpg"

        # Empty display_sizes
        img3 = {"display_sizes": []}
        assert src._extract_best_display_uri(img3) is None

        # No display_sizes key
        img4 = {}
        assert src._extract_best_display_uri(img4) is None

        src.close()


# ---------------------------------------------------------------------------
# ImageSource Interface Compatibility
# ---------------------------------------------------------------------------


class TestGettyInterfaceCompatibility:
    """Test that GettySource satisfies the ImageSource interface."""

    @pytest.mark.usefixtures("_env_ready")
    def test_has_required_methods(self) -> None:
        """GettySource has search, download_url, close methods."""
        src = GettySource()
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
        src = GettySource()
        assert isinstance(src.name, str)
        assert len(src.name) > 0
        src.close()


# ---------------------------------------------------------------------------
# Secret-Safe Logging Tests
# ---------------------------------------------------------------------------


class TestGettySecretSafety:
    """Verify secrets are not exposed in logs or repr."""

    @pytest.mark.usefixtures("_env_ready")
    def test_api_key_not_in_repr(self) -> None:
        """API key is not present in object repr."""
        src = GettySource()
        r = repr(src)
        assert "test-api-key" not in r
        assert "test-api-secret" not in r
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_api_key_not_in_search_error_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """API key is not logged in search failure messages."""
        src = GettySource()
        mock_resp = _mock_getty_response(401)

        with caplog.at_level("WARNING"):
            with patch.object(src._session, "post", return_value=mock_resp):
                list(src.search("test"))

        assert "test-api-key" not in caplog.text
        src.close()

    @pytest.mark.usefixtures("_env_ready")
    def test_api_key_not_in_download_error_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """API key is not logged in download failure messages."""
        src = GettySource(max_retries=0, delay=0.01)
        resp = _mock_getty_response(404)

        with caplog.at_level("WARNING"):
            with patch.object(src._session, "get", return_value=resp):
                src.download_url("https://media.gettyimages.com/comp/123.jpg")

        assert "test-api-key" not in caplog.text
        src.close()


# ---------------------------------------------------------------------------
# Context Manager / Close Tests
# ---------------------------------------------------------------------------


class TestGettyCleanup:
    """Test cleanup and close behavior."""

    @pytest.mark.usefixtures("_env_ready")
    def test_close_does_not_raise(self) -> None:
        """close() completes without error."""
        src = GettySource()
        src.close()  # Should not raise

    @pytest.mark.usefixtures("_env_ready")
    def test_session_closed_on_close(self) -> None:
        """close() closes the underlying requests session."""
        src = GettySource()
        with patch.object(src._session, "close") as mock_close:
            src.close()
        mock_close.assert_called_once()
