"""Flickr API image source implementation.

Uses the Flickr REST API (api_key required, no OAuth for public search).
Searches for images by person name and returns structured results with license metadata.

API docs: https://www.flickr.com/services/api/
Rate limits: 3600 requests/hour for non-commercial keys.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Iterator

import requests

from dataset_acquisition.models import SearchResult
from dataset_acquisition.sources.base import ImageSource

logger = logging.getLogger(__name__)

FLICKR_API_URL = "https://api.flickr.com/services/rest/"
REQUEST_TIMEOUT = 30
DEFAULT_DELAY = 0.34  # ~3600 requests/hour
MAX_PER_PAGE = 100
MAX_TOTAL_RESULTS = 4000  # Flickr hard cap

# License IDs that permit the intended research/dataset use.
# CC BY 2.0 (4), CC BY-SA 2.0 (5), CC0 1.0 (9), Public Domain Mark (10),
# CC BY 4.0 (11), CC BY-SA 4.0 (12).
# We include these because they permit reproduction, distribution,
# derivative works (with attribution/share-alike where required).
# We EXCLUDE: All Rights Reserved (0), CC BY-NC* (1,2,3,14,15,16),
# CC BY-ND* (6,13), No known copyright restrictions (7).
LICENSE_IDS_COMPATIBLE = {4, 5, 9, 10, 11, 12}


class FlickrSource(ImageSource):
    """Flickr REST API image source.

    Implements the ImageSource interface for Flickr.
    Requires FLICKR_API_KEY environment variable.
    """

    def __init__(
        self,
        delay: float = DEFAULT_DELAY,
        max_retries: int = 3,
        max_rate_limit_retries: int = 5,
        request_timeout: int = REQUEST_TIMEOUT,
        license_filter: set[int] | None = None,
    ) -> None:
        self._delay = delay
        self._max_retries = max_retries
        self._max_rate_limit_retries = max_rate_limit_retries
        self._request_timeout = request_timeout
        self._license_filter = license_filter or LICENSE_IDS_COMPATIBLE
        self._last_request_time = 0.0
        self._consecutive_rate_limits = 0
        self._rate_limit_errors = 0
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "SmartIDPhotoProcessor/1.0 (research dataset acquisition)"}
        )

        api_key = os.environ.get("FLICKR_API_KEY", "")
        if not api_key:
            raise ValueError(
                "FLICKR_API_KEY environment variable is required. "
                "Obtain a free key at https://www.flickr.com/services/apps/create/"
            )
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "flickr"

    @property
    def rate_limit_errors(self) -> int:
        return self._rate_limit_errors

    def _throttle(self) -> None:
        """Enforce minimum delay between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_request_time = time.time()

    def _api_get(self, method: str, params: dict[str, Any] | None = None) -> dict | None:
        """Make a throttled GET request to the Flickr REST API.

        Returns parsed JSON dict or None on failure.
        """
        self._throttle()

        request_params: dict[str, Any] = {
            "method": method,
            "api_key": self._api_key,
            "format": "json",
            "nojsoncallback": "1",
        }
        if params:
            request_params.update(params)

        for attempt in range(self._max_retries):
            try:
                resp = self._session.get(
                    FLICKR_API_URL, params=request_params, timeout=self._request_timeout
                )

                if resp.status_code == 429:
                    self._rate_limit_errors += 1
                    self._consecutive_rate_limits += 1
                    if self._consecutive_rate_limits > self._max_rate_limit_retries:
                        logger.warning("Too many consecutive Flickr rate limits, aborting")
                        return None
                    wait = min(3.0 * self._consecutive_rate_limits, 15.0)
                    logger.warning(
                        "Flickr rate limited (429), waiting %.1fs (consecutive=%d)",
                        wait,
                        self._consecutive_rate_limits,
                    )
                    time.sleep(wait)
                    continue

                self._consecutive_rate_limits = 0

                if resp.status_code != 200:
                    logger.warning("Flickr API HTTP %d on %s", resp.status_code, method)
                    return None

                data = resp.json()

                # Flickr wraps errors in stat field
                stat = data.get("stat", "")
                if stat == "fail":
                    error_code = data.get("code", "unknown")
                    error_msg = data.get("message", "unknown")
                    logger.warning(
                        "Flickr API error %s: %s (method=%s)", error_code, error_msg, method
                    )
                    return None

                return data

            except requests.RequestException as exc:
                logger.warning(
                    "Flickr API request failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(2**attempt, 10.0))

        return None

    def search(
        self,
        query: str,
        max_results: int = 20,
    ) -> Iterator[SearchResult]:
        """Search Flickr for images matching *query*.

        Uses flickr.photos.search with text search and license filtering.
        """
        per_page = min(MAX_PER_PAGE, max_results)
        collected = 0
        page = 1

        extras = (
            "description,license,date_upload,date_taken,owner_name,"
            "tags,machine_tags,o_dims,url_l,url_o,url_z,url_c"
        )

        # Build license filter string
        license_str = ",".join(str(lid) for lid in self._license_filter)

        while collected < max_results:
            params: dict[str, Any] = {
                "text": query,
                "per_page": str(per_page),
                "page": str(page),
                "extras": extras,
                "sort": "relevance",
                "content_types": "0",  # photos only
                "media": "photos",
                "safe_search": "1",  # safe content only (unauthenticated)
            }
            if license_str:
                params["license"] = license_str

            data = self._api_get("flickr.photos.search", params)
            if data is None:
                break

            photos = data.get("photos", {}).get("photo", [])
            if not photos:
                break

            for photo in photos:
                if collected >= max_results:
                    break

                search_result = self._normalize_photo(photo, query)
                if search_result is not None:
                    collected += 1
                    yield search_result

            # Check if more pages exist
            total = int(data.get("photos", {}).get("total", 0))
            pages = int(data.get("photos", {}).get("pages", 0))
            if page >= pages or collected >= max_results:
                break

            page += 1

    def _normalize_photo(self, photo: dict[str, Any], query: str) -> SearchResult | None:
        """Normalize a Flickr photo dict into a SearchResult.

        Returns None if the photo cannot be normalized.
        """
        photo_id = photo.get("id")
        if not photo_id:
            return None

        # Build source URL
        owner = photo.get("owner", "")
        source_url = f"https://www.flickr.com/photos/{owner}/{photo_id}"

        # Get image URL: prefer url_l (large), then url_c, url_z, then build static URL
        image_url = (
            photo.get("url_l")
            or photo.get("url_c")
            or photo.get("url_z")
            or self._build_static_url(photo)
        )
        if not image_url:
            return None

        # Extract dimensions
        width = int(photo.get("width_l") or photo.get("width_c") or photo.get("width_z") or 0)
        height = int(photo.get("height_l") or photo.get("height_c") or photo.get("height_z") or 0)

        # License
        license_id = int(photo.get("license", 0))
        license_name = self._license_id_to_name(license_id)

        # Description: Flickr returns dict with _content key
        desc_raw = photo.get("description", {})
        if isinstance(desc_raw, dict):
            description = desc_raw.get("_content", "")
        else:
            description = str(desc_raw)

        return SearchResult(
            source=self.name,
            source_url=source_url,
            image_url=image_url,
            title=photo.get("title", ""),
            description=description,
            license=license_name,
            attribution=f"Photo by {photo.get('ownername', 'unknown')} on Flickr",
            width=width,
            height=height,
            metadata={
                "photo_id": photo_id,
                "owner": owner,
                "owner_name": photo.get("ownername", ""),
                "license_id": license_id,
                "date_upload": photo.get("dateupload", ""),
                "date_taken": photo.get("datetaken", ""),
                "tags": photo.get("tags", ""),
                "query": query,
            },
        )

    def _build_static_url(self, photo: dict[str, Any]) -> str:
        """Build a static Flickr image URL from photo dict fields.

        Falls back to server/farm-based URL if url_* not available.
        """
        photo_id = photo.get("id", "")
        secret = photo.get("secret", "")
        server = photo.get("server", "")
        farm = photo.get("farm", "")

        if not all([photo_id, secret, server]):
            return ""

        # Static URL format: https://live.staticflickr.com/{server}/{id}_{secret}_{size}.jpg
        # Use 'b' suffix for large (1024 on longest side)
        return f"https://live.staticflickr.com/{server}/{photo_id}_{secret}_b.jpg"

    @staticmethod
    def _license_id_to_name(license_id: int) -> str:
        """Map Flickr license ID to human-readable name."""
        mapping = {
            0: "All Rights Reserved",
            1: "CC BY-NC-SA 2.0",
            2: "CC BY-NC 2.0",
            3: "CC BY-NC-ND 2.0",
            4: "CC BY 2.0",
            5: "CC BY-SA 2.0",
            6: "CC BY-ND 2.0",
            7: "No known copyright restrictions",
            8: "United States Government Work",
            9: "CC0 1.0",
            10: "Public Domain Mark",
            11: "CC BY 4.0",
            12: "CC BY-SA 4.0",
            13: "CC BY-ND 4.0",
            14: "CC BY-NC 4.0",
            15: "CC BY-NC-SA 4.0",
            16: "CC BY-NC-ND 4.0",
        }
        return mapping.get(license_id, f"Unknown ({license_id})")

    def download_url(self, url: str) -> bytes | None:
        """Download raw image bytes from a Flickr URL."""
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
                        logger.warning("Too many consecutive Flickr rate limits on download")
                        return None
                    wait = min(3.0 * self._consecutive_rate_limits, 15.0)
                    logger.warning("Flickr download rate limited (429), waiting %.1fs", wait)
                    time.sleep(wait)
                    continue

                self._consecutive_rate_limits = 0

                if resp.status_code != 200:
                    logger.warning("Flickr download HTTP %d", resp.status_code)
                    return None

                content_type = resp.headers.get("Content-Type", "")
                if "image" not in content_type and "octet" not in content_type:
                    logger.warning("Flickr download: non-image content-type '%s'", content_type)
                    return None

                chunks = []
                total_size = 0
                max_size = 50 * 1024 * 1024

                for chunk in resp.iter_content(chunk_size=8192):
                    chunks.append(chunk)
                    total_size += len(chunk)
                    if total_size > max_size:
                        logger.warning("Flickr download exceeded max size (50 MB)")
                        return None

                return b"".join(chunks)

            except requests.RequestException as exc:
                logger.warning(
                    "Flickr download failed (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries,
                    exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(min(2**attempt, 10.0))

        return None

    def close(self) -> None:
        """Release HTTP session resources."""
        self._session.close()
