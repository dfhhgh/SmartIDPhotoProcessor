"""Openverse API image source implementation.

Uses the Openverse REST API (no authentication required for public search).
Searches for images by person name and returns structured results with license metadata.

API docs: https://docs.openverse.org/
Rate limits: 5 req/day unauthenticated, 60/hr with Authorization header.
License filtering: by, by-sa, cc0, pdm (compatible licenses).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests

from dataset_acquisition.models import SearchResult
from dataset_acquisition.sources.base import ImageSource

logger = logging.getLogger(__name__)

OPENVERSE_API_BASE = "https://api.openverse.org/v1"
REQUEST_TIMEOUT = 30
DEFAULT_DELAY = 1.0  # Conservative: 5 req/day unauthenticated
MAX_PER_PAGE = 20  # Openverse max per page for anonymous requests

# Compatible license slugs (same as other sources)
COMPATIBLE_LICENSES = {"by", "by-sa", "cc0", "pdm"}


class OpenverseSource(ImageSource):
    """Openverse API image source.

    Implements the ImageSource interface for Openverse public API.
    No authentication required for basic usage.
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        max_retries: int = 3,
        max_rate_limit_retries: int = 5,
        request_timeout: int = REQUEST_TIMEOUT,
        license_filter: set[str] | None = None,
    ) -> None:
        self._delay = delay
        self._max_retries = max_retries
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout
        self._license_filter = license_filter or COMPATIBLE_LICENSES
        self._last_request_time: float = 0.0
        self._consecutive_rate_limits: int = 0
        self._rate_limit_errors: int = 0
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "SmartIDPhotoProcessor/1.0 (research dataset acquisition)"}
        )

    @property
    def name(self) -> str:
        return "openverse"

    @property
    def rate_limit_errors(self) -> int:
        return self._rate_limit_errors

    def _throttle(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request_time = time.time()

    def _api_get(self, path: str, params: dict[str, Any] | None = None) -> dict | None:
        """Make a throttled GET request to the Openverse API.

        Returns parsed JSON dict or None on failure.
        """
        self._throttle()

        url = f"{OPENVERSE_API_BASE}{path}"
        rate_limit_retries = 0

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(
                    url, params=params or {}, timeout=self._request_timeout
                )

                if resp.status_code == 429:
                    self._rate_limit_errors += 1
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits > self._max_rate_limit_retries:
                        logger.warning("Too many consecutive Openverse rate limits, aborting")
                        return None
                    # Use Retry-After header if available, otherwise exponential backoff
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        wait = min(2 ** rate_limit_retries, 15)
                    logger.warning(
                        "Openverse rate limited (429), waiting %.1fs (consecutive=%d)",
                        wait,
                        self._consecutive_rate_limits,
                    )
                    time.sleep(wait)
                    rate_limit_retries += 1
                    continue

                self._consecutive_rate_limits = 0

                if resp.status_code != 200:
                    logger.warning("Openverse API HTTP %d on %s", resp.status_code, path)
                    return None

                return resp.json()

            except requests.RequestException as exc:
                logger.warning(
                    "Openverse API request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(2 ** attempt, 10.0))

        return None

    def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> Iterator[SearchResult]:
        """Search Openverse for images matching *query*.

        Uses the /images/ endpoint with license filtering.
        Yields SearchResult objects.
        """
        per_page = min(MAX_PER_PAGE, max_results)
        collected = 0
        page = 1

        while collected < max_results:
            params: dict[str, Any] = {
                "q": query,
                "page": str(page),
                "page_size": str(per_page),
            }

            # Add license filter if set
            if self._license_filter:
                params["license"] = ",".join(sorted(self._license_filter))

            data = self._api_get("/images/", params)
            if data is None:
                break

            results = data.get("results", [])
            if not results:
                break

            for item in results:
                if collected >= max_results:
                    break

                search_result = self._normalize_result(item, query)
                if search_result is not None:
                    collected += 1
                    yield search_result

            # Check if more pages exist
            next_url = data.get("next")
            if not next_url or collected >= max_results:
                break

            page += 1

    def _normalize_result(self, item: dict[str, Any], query: str) -> SearchResult | None:
        """Normalize an Openverse API result item into a SearchResult.

        Returns None if the item cannot be normalized.
        """
        identifier = item.get("id") or item.get("identifier")
        if not identifier:
            return None

        # Get image URL (url field in current API, image_url in older versions)
        image_url = item.get("url") or item.get("image_url", "")
        if not image_url:
            return None

        # Source URL (foreign landing URL or creator page)
        source_url = item.get("foreign_landing_url") or item.get("creator_url", "")
        if not source_url:
            # Build a fallback URL
            source_url = f"https://openverse.org/image/{identifier}"

        # License
        license_slug = item.get("license", "")
        license_name = self._normalize_license(license_slug)

        # Attribution
        creator = item.get("creator", "")
        attribution = f"Photo by {creator}" if creator else ""

        # Dimensions
        width = int(item.get("width", 0))
        height = int(item.get("height", 0))

        return SearchResult(
            source=self.name,
            source_url=source_url,
            image_url=image_url,
            title=item.get("title", ""),
            description=item.get("description", "") or "",
            license=license_name,
            attribution=attribution,
            width=width,
            height=height,
            metadata={
                "identifier": identifier,
                "creator": creator,
                "creator_url": item.get("creator_url", ""),
                "license_slug": license_slug,
                "source": item.get("source", ""),
                "query": query,
            },
        )

    @staticmethod
    def _normalize_license(license_slug: str) -> str:
        """Normalize Openverse license slug to human-readable name."""
        mapping = {
            "by": "CC BY 4.0",
            "by-sa": "CC BY-SA 4.0",
            "by-nc": "CC BY-NC 4.0",
            "by-nc-sa": "CC BY-NC-SA 4.0",
            "by-nc-nd": "CC BY-NC-ND 4.0",
            "by-nd": "CC BY-ND 4.0",
            "cc0": "CC0 1.0",
            "pdm": "Public Domain Mark",
            "sampling+": "Sampling Plus",
            "fry": "Free Art License",
            "nc-sampling+": "NC Sampling Plus",
        }
        return mapping.get(license_slug, license_slug or "unknown")

    def download_url(self, url: str) -> bytes | None:
        """Download raw image bytes from a URL."""
        self._throttle()

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(
                    url, timeout=self._request_timeout, stream=True
                )

                if resp.status_code == 429:
                    self._rate_limit_errors += 1
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits > self._max_rate_limit_retries:
                        logger.warning("Too many consecutive Openverse rate limits on download")
                        return None
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after:
                        wait = float(retry_after)
                    else:
                        wait = min(2 ** attempt, 15)
                    logger.warning("Openverse download rate limited (429), waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                self._consecutive_rate_limits = 0

                if resp.status_code != 200:
                    logger.warning("Openverse download HTTP %d", resp.status_code)
                    return None

                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and "octet" not in content_type:
                    logger.warning("Openverse download: non-image content-type '%s'", content_type)
                    return None

                chunks = []
                total_size = 0
                max_size = 50 * 1024 * 1024

                for chunk in resp.iter_content(chunk_size=8192):
                    chunks.append(chunk)
                    total_size += len(chunk)
                    if total_size > max_size:
                        logger.warning("Openverse download exceeded max size (50 MB)")
                        return None

                return b"".join(chunks)

            except requests.RequestException as exc:
                logger.warning(
                    "Openverse download failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(2 ** attempt, 10.0))

        return None

    def close(self) -> None:
        """Release HTTP session resources."""
        self._session.close()
