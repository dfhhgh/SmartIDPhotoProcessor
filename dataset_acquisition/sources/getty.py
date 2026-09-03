"""Getty Images API source implementation.

Requires:
- GETTY_API_KEY environment variable
- GETTY_API_SECRET environment variable (for OAuth2 access token)

Usage restrictions: See Getty Images Content License Agreement Section 3.11.
This source is provided for future use IF licensing is explicitly authorized
by Getty Images for ML/dataset purposes.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Iterator, Optional

import requests

from .base import ImageSource
from ..models import SearchResult

logger = logging.getLogger(__name__)

GETTY_API_BASE = "https://api.gettyimages.com/v3"
GETTY_AUTH_URL = "https://authentication.gettyimages.com/oauth2/token"

DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_RESULTS = 20
DEFAULT_DELAY = 0.5
DEFAULT_REQUEST_TIMEOUT = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_MAX_RATE_LIMIT_RETRIES = 5

# Search fields that include display sizes with download URIs
SEARCH_FIELDS_WITH_SIZES = "id,title,asset_family,caption,license_model,max_dimensions,display_sizes"
# Detail fields for individual image lookup
DETAIL_FIELDS = "id,title,asset_family,caption,license_model,max_dimensions,display_sizes"


class GettySource(ImageSource):
    """Getty Images API source.

    Implements the ImageSource interface for Getty Images v3 API.
    Requires GETTY_API_KEY and GETTY_API_SECRET environment variables.
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_rate_limit_retries: int = DEFAULT_MAX_RATE_LIMIT_RETRIES,
        request_timeout: int = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        self._delay = delay
        self._max_retries = max_retries
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout
        self._last_request_time: float = 0.0
        self._consecutive_rate_limits: int = 0
        self._session = requests.Session()
        self._access_token: Optional[str] = None

        api_key = os.environ.get("GETTY_API_KEY", "")
        api_secret = os.environ.get("GETTY_API_SECRET", "")

        if not api_key:
            raise ValueError(
                "GETTY_API_KEY environment variable is required. "
                "Contact Getty Images sales representative for API access."
            )
        if not api_secret:
            raise ValueError(
                "GETTY_API_SECRET environment variable is required. "
                "Contact Getty Images sales representative for API access."
            )

        self._api_key = api_key
        self._api_secret = api_secret
        self._session.headers.update(
            {
                "Api-Key": self._api_key,
                "Accept": "application/json",
            }
        )

    @property
    def name(self) -> str:
        return "getty_images"

    def _throttle(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request_time = time.time()

    def _authenticate(self) -> bool:
        """Obtain OAuth2 access token using client credentials grant.

        Returns True if authentication succeeded.
        """
        if self._access_token:
            return True

        try:
            self._throttle()
            response = self._session.post(
                GETTY_AUTH_URL,
                data={
                    "client_id": self._api_key,
                    "client_secret": self._api_secret,
                    "grant_type": "client_credentials",
                },
                timeout=self._request_timeout,
            )
            if response.status_code == 200:
                data = response.json()
                self._access_token = data.get("access_token")
                if self._access_token:
                    self._session.headers["Authorization"] = (
                        f"Bearer {self._access_token}"
                    )
                    logger.info("Getty Images authentication successful")
                    return True
                logger.warning("Getty Images auth response missing access_token")
                return False
            logger.warning(
                "Getty Images authentication failed: HTTP %d", response.status_code
            )
            return False
        except requests.RequestException as exc:
            logger.warning("Getty Images authentication error: %s", exc)
            return False

    def _api_get(
        self, path: str, params: Optional[dict] = None
    ) -> Optional[dict]:
        """Make a throttled GET request to the Getty API.

        Returns parsed JSON dict or None on failure.
        """
        self._throttle()

        url = f"{GETTY_API_BASE}{path}"
        rate_limit_retries = 0

        while True:
            try:
                response = self._session.get(
                    url, params=params or {}, timeout=self._request_timeout
                )

                if response.status_code == 200:
                    self._consecutive_rate_limits = 0
                    return response.json()

                if response.status_code == 429:
                    rate_limit_retries += 1
                    if rate_limit_retries > self._max_rate_limit_retries:
                        logger.warning(
                            "Getty Images rate limit exceeded after %d retries",
                            self._max_rate_limit_retries,
                        )
                        return None
                    # Exponential backoff capped at 15s
                    backoff = min(2 ** rate_limit_retries, 15)
                    logger.info(
                        "Getty Images rate limited (attempt %d/%d), waiting %ds",
                        rate_limit_retries,
                        self._max_rate_limit_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue

                if response.status_code == 401:
                    logger.warning("Getty Images unauthorized (HTTP 401)")
                    return None
                if response.status_code == 403:
                    logger.warning("Getty Images forbidden (HTTP 403)")
                    return None
                if response.status_code == 404:
                    logger.debug("Getty Images not found (HTTP 404): %s", path)
                    return None

                logger.warning(
                    "Getty Images API error: HTTP %d on %s",
                    response.status_code,
                    path,
                )
                return None

            except requests.RequestException as exc:
                logger.warning("Getty Images request failed: %s", exc)
                return None

    def search(
        self, query: str, max_results: int = DEFAULT_MAX_RESULTS
    ) -> Iterator[SearchResult]:
        """Search Getty Images for the given query.

        Uses the creative search endpoint. Paginates internally.
        Yields SearchResult objects for discovered images.
        """
        if not self._authenticate():
            logger.warning(
                "Getty Images authentication failed, cannot search for '%s'",
                query,
            )
            return

        page = 1
        page_size = min(DEFAULT_PAGE_SIZE, max_results)
        total_yielded = 0

        while total_yielded < max_results:
            params = {
                "phrase": query,
                "page": page,
                "page_size": page_size,
                "fields": SEARCH_FIELDS_WITH_SIZES,
            }

            data = self._api_get("/search/images/creative", params)
            if data is None:
                break

            result_count = data.get("result_count", 0)
            images = data.get("images", [])

            if not images:
                break

            for img in images:
                if total_yielded >= max_results:
                    break

                search_result = self._normalize_search_result(img, query)
                if search_result is not None:
                    total_yielded += 1
                    yield search_result

            # Check if we've exhausted all pages
            total_pages = (result_count + page_size - 1) // page_size
            if page >= total_pages:
                break

            page += 1

    def _normalize_search_result(
        self, img: dict, query: str
    ) -> Optional[SearchResult]:
        """Normalize a Getty API image dict into a SearchResult.

        Returns None if the image cannot be normalized.
        """
        image_id = img.get("id")
        if not image_id:
            return None

        # Get the best available display URI (comp size preferred)
        image_url = self._extract_best_display_uri(img)
        if not image_url:
            logger.debug("Getty image %s has no display URI", image_id)
            return None

        # Build source URL for deduplication
        source_url = f"https://www.gettyimages.com/photos/{image_id}"

        max_dims = img.get("max_dimensions", {})
        width = max_dims.get("width", 0)
        height = max_dims.get("height", 0)

        asset_family = img.get("asset_family", "creative")
        license_model = img.get("license_model", "royaltyfree")

        return SearchResult(
            source=self.name,
            source_url=source_url,
            image_url=image_url,
            title=img.get("title", ""),
            description=img.get("caption", ""),
            license=license_model,
            attribution="",
            width=width,
            height=height,
            metadata={
                "image_id": image_id,
                "asset_family": asset_family,
                "collection_name": img.get("collection_name", ""),
                "query": query,
            },
        )

    def _extract_best_display_uri(self, img: dict) -> Optional[str]:
        """Extract the best available display URI from a Getty image dict.

        Prefers comp > high_res_comp > mid_res_comp > thumb.
        """
        display_sizes = img.get("display_sizes", [])
        if not display_sizes:
            return None

        # Preferred order: comp sizes first (larger), then thumb
        preferred_order = [
            "high_res_comp",
            "comp",
            "preview",
            "mid_res_comp",
            "thumb",
        ]

        size_map = {s.get("name", ""): s.get("uri", "") for s in display_sizes}

        for name in preferred_order:
            uri = size_map.get(name)
            if uri:
                return uri

        # Fall back to first available
        first = display_sizes[0]
        return first.get("uri") if isinstance(first, dict) else None

    def download_url(self, url: str) -> Optional[bytes]:
        """Download image bytes from a Getty display URL.

        Returns raw image bytes or None on failure.
        """
        self._throttle()

        retries = 0
        while retries <= self._max_retries:
            try:
                response = self._session.get(
                    url, timeout=self._request_timeout, stream=True
                )

                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type or "octet" in content_type:
                        # Stream with 50 MB max
                        chunks = []
                        total_size = 0
                        max_size = 50 * 1024 * 1024
                        for chunk in response.iter_content(chunk_size=8192):
                            total_size += len(chunk)
                            if total_size > max_size:
                                logger.warning(
                                    "Getty download exceeded max size (%d MB)",
                                    max_size // (1024 * 1024),
                                )
                                return None
                            chunks.append(chunk)
                        return b"".join(chunks)
                    logger.debug(
                        "Getty download: unexpected content-type '%s'", content_type
                    )
                    return None

                if response.status_code == 429:
                    retries += 1
                    if retries > self._max_retries:
                        logger.warning(
                            "Getty download rate limited after %d retries",
                            self._max_retries,
                        )
                        return None
                    backoff = min(2 ** retries, 15)
                    time.sleep(backoff)
                    continue

                logger.warning(
                    "Getty download failed: HTTP %d", response.status_code
                )
                return None

            except requests.RequestException as exc:
                retries += 1
                if retries > self._max_retries:
                    logger.warning("Getty download error after retries: %s", exc)
                    return None
                time.sleep(min(2 ** retries, 15))

        return None

    def close(self) -> None:
        """Release HTTP session resources."""
        self._session.close()
