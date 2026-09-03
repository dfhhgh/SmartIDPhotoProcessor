"""Wikimedia Commons image source implementation.

Uses the Wikimedia Commons API (no authentication required for read-only queries).
Searches for images by person name and returns structured results with license metadata.

API docs: https://commons.wikimedia.org/w/api.php
Rate limits: 10 req/min unauthenticated, 200 req/min with proper User-Agent.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests

from dataset_acquisition.models import SearchResult
from dataset_acquisition.sources.base import ImageSource

logger = logging.getLogger(__name__)

API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "SmartIDPhotoProcessor/1.0 (research dataset acquisition; contact: github.com/anomalyco/opencode)"
REQUEST_TIMEOUT = 30
DEFAULT_DELAY = 1.0


class WikimediaSource(ImageSource):
    """Wikimedia Commons image source."""

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        max_retries: int = 3,
        max_rate_limit_retries: int = 5,
        request_timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self._delay = delay
        self._max_retries = max_retries
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        self._last_request_time = 0.0
        self._rate_limit_errors = 0
        self._consecutive_rate_limits = 0

    @property
    def name(self) -> str:
        return "wikimedia_commons"

    @property
    def rate_limit_errors(self) -> int:
        return self._rate_limit_errors

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request_time = time.time()

    def _api_get(self, params: dict[str, Any]) -> dict | None:
        """Make a throttled GET request to the Wikimedia API."""
        self._throttle()

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(API_URL, params=params, timeout=self._request_timeout)

                if resp.status_code == 429:
                    self._rate_limit_errors += 1
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits > self._max_rate_limit_retries:
                        logger.warning("Too many consecutive rate limits, aborting request")
                        return None
                    wait = min(3.0 * self._consecutive_rate_limits, 15.0)
                    logger.warning("Rate limited (429), waiting %.1fs (consecutive=%d)", wait, self._consecutive_rate_limits)
                    time.sleep(wait)
                    continue

                self._consecutive_rate_limits = 0
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning(
                    "Wikimedia API request failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(2 ** attempt, 10.0))
        return None

    def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> Iterator[SearchResult]:
        """Search Wikimedia Commons for images matching *query*."""
        return self._search_text(query, max_results)

    def search_by_category(
        self,
        category: str,
        max_results: int = 20,
    ) -> Iterator[SearchResult]:
        """Search Wikimedia Commons by category name.

        Uses the categorymembers generator to find all files in a category.
        Category should be the full category name (e.g., "Category:Tom Hanks").
        """
        return self._search_category(category, max_results)

    def _search_text(
        self,
        query: str,
        max_results: int,
    ) -> Iterator[SearchResult]:
        """Text search implementation."""
        limit = min(max_results, 50)
        offset = 0
        collected = 0

        while collected < max_results:
            page_size = min(limit - collected, 50)
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": "6",
                "gsrlimit": str(page_size),
                "gsroffset": str(offset),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size",
                "iiurlwidth": "800",
                "format": "json",
            }

            data = self._api_get(params)
            if data is None:
                break

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break

            for page_id, page_data in pages.items():
                image_info = page_data.get("imageinfo", [{}])
                if not image_info:
                    continue

                info = image_info[0]
                ext_meta = info.get("extmetadata", {})

                image_url = info.get("thumburl") or info.get("url", "")
                source_url = info.get("url", "")
                title = page_data.get("title", "")
                width = info.get("width", 0)
                height = info.get("height", 0)

                license_info = ext_meta.get("LicenseShortName", {}).get("value", "unknown")
                attribution = ext_meta.get("Attribution", {}).get("value", "")

                if not image_url:
                    continue

                yield SearchResult(
                    source=self.name,
                    source_url=source_url,
                    image_url=image_url,
                    title=title,
                    description=ext_meta.get("ImageDescription", {}).get("value", ""),
                    license=license_info,
                    attribution=attribution,
                    width=width,
                    height=height,
                    metadata={
                        "page_id": page_id,
                        "page_title": title,
                        "search_mode": "text",
                    },
                )
                collected += 1
                if collected >= max_results:
                    break

            if not data.get("continue"):
                break
            offset = data["continue"].get("gsroffset", offset + page_size)

    def _search_category(
        self,
        category: str,
        max_results: int,
    ) -> Iterator[SearchResult]:
        """Category-based search implementation.

        Pagination fix: continuation token is now correctly carried across
        iterations by updating the continue_params dict, which is merged
        into each API request.
        """
        # Normalize category name
        if not category.startswith("Category:"):
            category = f"Category:{category}"

        collected = 0
        continue_params: dict[str, str] = {}

        while collected < max_results:
            page_size = min(max_results - collected, 50)
            params: dict[str, Any] = {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "file",
                "gcmlimit": str(page_size),
                "prop": "imageinfo",
                "iiprop": "url|extmetadata|size",
                "iiurlwidth": "800",
                "format": "json",
                **continue_params,
            }

            data = self._api_get(params)
            if data is None:
                break

            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break

            for page_id, page_data in pages.items():
                image_info = page_data.get("imageinfo", [{}])
                if not image_info:
                    continue

                info = image_info[0]
                ext_meta = info.get("extmetadata", {})

                image_url = info.get("thumburl") or info.get("url", "")
                source_url = info.get("url", "")
                title = page_data.get("title", "")
                width = info.get("width", 0)
                height = info.get("height", 0)

                license_info = ext_meta.get("LicenseShortName", {}).get("value", "unknown")
                attribution = ext_meta.get("Attribution", {}).get("value", "")

                if not image_url:
                    continue

                yield SearchResult(
                    source=self.name,
                    source_url=source_url,
                    image_url=image_url,
                    title=title,
                    description=ext_meta.get("ImageDescription", {}).get("value", ""),
                    license=license_info,
                    attribution=attribution,
                    width=width,
                    height=height,
                    metadata={
                        "page_id": page_id,
                        "page_title": title,
                        "category": category,
                        "search_mode": "category",
                    },
                )
                collected += 1
                if collected >= max_results:
                    break

            # Handle continuation
            cont_data = data.get("continue")
            if not cont_data:
                break
            continue_key = cont_data.get("gcmcontinue")
            if not continue_key:
                break
            continue_params = {"gcmcontinue": continue_key}

    def download_url(self, url: str) -> bytes | None:
        """Download raw image bytes from a Wikimedia URL."""
        self._throttle()

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(url, timeout=self._request_timeout, stream=True)

                if resp.status_code == 429:
                    self._rate_limit_errors += 1
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits > self._max_rate_limit_retries:
                        logger.warning("Too many consecutive rate limits on download, aborting")
                        return None
                    wait = min(3.0 * self._consecutive_rate_limits, 15.0)
                    logger.warning("Rate limited on download (429), waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                self._consecutive_rate_limits = 0
                resp.raise_for_status()

                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and "octet" not in content_type:
                    logger.warning("Non-image content type: %s", content_type)
                    return None

                chunks = []
                total_size = 0
                max_size = 50 * 1024 * 1024

                for chunk in resp.iter_content(chunk_size=8192):
                    chunks.append(chunk)
                    total_size += len(chunk)
                    if total_size > max_size:
                        logger.warning("Image too large, aborting download: %s", url)
                        return None

                return b"".join(chunks)

            except requests.RequestException as exc:
                logger.warning(
                    "Download failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(2 ** attempt, 10.0))

        return None

    def close(self) -> None:
        self._session.close()
