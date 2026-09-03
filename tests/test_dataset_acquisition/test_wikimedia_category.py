"""Tests for Wikimedia category-based discovery.

All tests use mocks/fixtures — no Wikimedia API access required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from dataset_acquisition.sources.wikimedia import WikimediaSource


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_category_response(
    pages: list[dict[str, Any]] | None = None,
    continue_key: str | None = None,
) -> dict[str, Any]:
    """Build a fake Wikimedia categorymembers response."""
    if pages is None:
        pages = [
            {
                "pageid": 12345,
                "title": "File:Test_Person_001.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/Test_Person_001.jpg",
                        "thumburl": "https://upload.wikimedia.org/thumb/Test_Person_001.jpg/800px-Test_Person_001.jpg",
                        "width": 4000,
                        "height": 3000,
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY-SA 4.0"},
                            "Attribution": {"value": "Test Photographer"},
                            "ImageDescription": {"value": "A test photo"},
                        },
                    }
                ],
            }
        ]

    response: dict[str, Any] = {
        "query": {
            "pages": {str(p["pageid"]): p for p in pages},
        },
    }

    if continue_key:
        response["continue"] = {"gcmcontinue": continue_key, "continue": "-||"}

    return response


def _mock_wikimedia_response(status_code: int = 200, json_data: Any = None) -> MagicMock:
    """Build a mock requests.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or {}
    resp.headers = {"Content-Type": "application/json"}
    return resp


# ---------------------------------------------------------------------------
# Category Search Tests
# ---------------------------------------------------------------------------


class TestWikimediaCategorySearch:
    """Test WikimediaSource.search_by_category() behavior."""

    def test_search_by_category_yields_results(self) -> None:
        """search_by_category() yields SearchResult objects."""
        source = WikimediaSource()
        mock_resp = _mock_wikimedia_response(200, _make_category_response())

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search_by_category("Category:Tom Hanks", max_results=5))

        assert len(results) == 1
        assert results[0].source == "wikimedia_commons"
        assert results[0].metadata["search_mode"] == "category"
        assert results[0].metadata["category"] == "Category:Tom Hanks"

    def test_search_by_category_normalizes_name(self) -> None:
        """search_by_category() adds Category: prefix if missing."""
        source = WikimediaSource()
        mock_resp = _mock_wikimedia_response(200, _make_category_response())

        with patch.object(source._session, "get", return_value=mock_resp) as mock_get:
            list(source.search_by_category("Tom Hanks", max_results=1))

        call_args = mock_get.call_args
        params = call_args[1].get("params", call_args[0][1] if len(call_args[0]) > 1 else {})
        assert params.get("gcmtitle") == "Category:Tom Hanks"

    def test_search_by_category_empty_results(self) -> None:
        """search_by_category() handles empty category."""
        source = WikimediaSource()
        mock_resp = _mock_wikimedia_response(200, _make_category_response(pages=[]))

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search_by_category("Category:Empty"))

        assert len(results) == 0

    def test_search_by_category_api_failure(self) -> None:
        """search_by_category() handles API failure."""
        source = WikimediaSource()
        mock_resp = _mock_wikimedia_response(500)

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search_by_category("Category:Test"))

        assert len(results) == 0

    def test_search_by_category_pagination(self) -> None:
        """search_by_category() paginates through results."""
        source = WikimediaSource()

        page1 = _make_category_response(
            pages=[
                {"pageid": 1, "title": "File:Img1.jpg", "imageinfo": [{"url": "http://img1.jpg", "thumburl": "http://thumb1.jpg", "width": 100, "height": 100, "extmetadata": {}}]},
                {"pageid": 2, "title": "File:Img2.jpg", "imageinfo": [{"url": "http://img2.jpg", "thumburl": "http://thumb2.jpg", "width": 100, "height": 100, "extmetadata": {}}]},
            ],
            continue_key="continue-token-123",
        )
        page2 = _make_category_response(
            pages=[
                {"pageid": 3, "title": "File:Img3.jpg", "imageinfo": [{"url": "http://img3.jpg", "thumburl": "http://thumb3.jpg", "width": 100, "height": 100, "extmetadata": {}}]},
            ],
            continue_key=None,
        )

        call_count = 0

        def mock_get(url: str, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_wikimedia_response(200, page1)
            return _mock_wikimedia_response(200, page2)

        with patch.object(source._session, "get", side_effect=mock_get):
            results = list(source.search_by_category("Category:Test", max_results=10))

        assert len(results) == 3
        assert call_count == 2

    def test_search_by_category_respects_max_results(self) -> None:
        """search_by_category() respects max_results parameter."""
        source = WikimediaSource()
        resp = _make_category_response(
            pages=[
                {"pageid": i, "title": f"File:Img{i}.jpg", "imageinfo": [{"url": f"http://img{i}.jpg", "thumburl": f"http://thumb{i}.jpg", "width": 100, "height": 100, "extmetadata": {}}]}
                for i in range(10)
            ]
        )
        mock_resp = _mock_wikimedia_response(200, resp)

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search_by_category("Category:Test", max_results=3))

        assert len(results) == 3


# ---------------------------------------------------------------------------
# Text Search Backward Compatibility
# ---------------------------------------------------------------------------


class TestWikimediaTextSearchBackwardCompatibility:
    """Verify text search still works after category enhancement."""

    def test_search_still_works(self) -> None:
        """search() still uses text search by default."""
        source = WikimediaSource()

        text_resp = {
            "query": {
                "pages": {
                    "99999": {
                        "pageid": 99999,
                        "title": "File:TextResult.jpg",
                        "imageinfo": [
                            {
                                "url": "http://text.jpg",
                                "thumburl": "http://text_thumb.jpg",
                                "width": 800,
                                "height": 600,
                                "extmetadata": {},
                            }
                        ],
                    }
                }
            }
        }

        mock_resp = _mock_wikimedia_response(200, text_resp)

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search("test query", max_results=1))

        assert len(results) == 1
        assert results[0].metadata["search_mode"] == "text"


# ---------------------------------------------------------------------------
# Result Normalization Tests
# ---------------------------------------------------------------------------


class TestWikimediaCategoryNormalization:
    """Test category search result normalization."""

    def test_normalize_result_with_all_fields(self) -> None:
        """Normalizes a complete category result."""
        source = WikimediaSource()
        pages = {
            "12345": {
                "pageid": 12345,
                "title": "File:Test_Person.jpg",
                "imageinfo": [
                    {
                        "url": "https://upload.wikimedia.org/Test_Person.jpg",
                        "thumburl": "https://upload.wikimedia.org/thumb/Test_Person.jpg/800px-Test_Person.jpg",
                        "width": 4000,
                        "height": 3000,
                        "extmetadata": {
                            "LicenseShortName": {"value": "CC BY 4.0"},
                            "Attribution": {"value": "Jane Doe"},
                            "ImageDescription": {"value": "Portrait of test person"},
                        },
                    }
                ],
            }
        }

        mock_resp = _mock_wikimedia_response(200, {"query": {"pages": pages}})

        with patch.object(source._session, "get", return_value=mock_resp):
            results = list(source.search_by_category("Category:Test", max_results=1))

        assert len(results) == 1
        r = results[0]
        assert r.source == "wikimedia_commons"
        assert r.title == "File:Test_Person.jpg"
        assert r.description == "Portrait of test person"
        assert r.license == "CC BY 4.0"
        assert r.attribution == "Jane Doe"
        assert r.width == 4000
        assert r.height == 3000
        assert r.metadata["category"] == "Category:Test"
        assert r.metadata["search_mode"] == "category"
