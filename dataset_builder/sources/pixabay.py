"""
Pixabay image source provider.

Adapter between the Pixabay API and the Dataset Builder's
:class:`BaseSource` contract.  All Pixabay-specific URL construction,
field mapping, and response parsing is encapsulated here.
"""

from __future__ import annotations

import time
from pathlib import Path

from dataset_builder.config.settings import Settings
from dataset_builder.sources.base_source import (
    BaseSource,
    DownloadResult,
    ImageMetadata,
    SearchResult,
)
from dataset_builder.utils.http_client import HTTPClient


class PixabaySource(BaseSource):
    """Concrete :class:`BaseSource` implementation for the Pixabay API.

    Uses :class:`HTTPClient` for all network communication.  The API
    key is read exclusively from ``Settings.PIXABAY_API_KEY``.

    Parameters
    ----------
    settings:
        Application settings containing the Pixabay API key and
        downloader configuration.
    """

    _SEARCH_URL: str = "https://pixabay.com/api/"
    _SOURCE_NAME: str = "pixabay"
    _LICENSE_NAME: str = "Pixabay License"
    _MAX_RESULTS_PER_PAGE: int = 200

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key: str = settings.PIXABAY_API_KEY
        self._http: HTTPClient = HTTPClient(settings)

    # ------------------------------------------------------------------
    # BaseSource properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the source identifier."""
        return self._SOURCE_NAME

    # ------------------------------------------------------------------
    # BaseSource abstract methods
    # ------------------------------------------------------------------

    def validate_configuration(self) -> bool:
        """Validate the Pixabay API key by making a test request.

        Returns
        -------
        bool
            ``True`` when the configuration is valid.

        Raises
        ------
        ValueError
            When ``PIXABAY_API_KEY`` is missing, empty, or rejected
            by the Pixabay API (401/403).
        """
        if not self._api_key:
            raise ValueError(
                "Pixabay API key is missing.  "
                "Set the PIXABAY_API_KEY environment variable."
            )

        try:
            response = self._http.get(
                url=self._SEARCH_URL,
                params={"key": self._api_key, "q": "test", "per_page": 3},
            )
            return response.status_code == 200
        except Exception as exc:
            status = getattr(exc, "response", None)
            if status is not None and status.status_code in (401, 403):
                raise ValueError(
                    "Pixabay API key is invalid.  "
                    "The server returned HTTP "
                    f"{status.status_code}."
                ) from exc
            raise

    def search(self, query: str, page: int, per_page: int) -> list[SearchResult]:
        """Query the Pixabay Search API and return matching images.

        Images smaller than ``Settings.MIN_IMAGE_WIDTH`` or
        ``Settings.MIN_IMAGE_HEIGHT`` are automatically discarded.

        Parameters
        ----------
        query:
            Free-text search string.
        page:
            1-indexed page number.
        per_page:
            Number of results per page (clamped to 200).

        Returns
        -------
        list[SearchResult]
            Matching images.  Empty list when no results are found.

        Raises
        ------
        requests.HTTPError
            When the API returns a non-2xx status code.
        """
        clamped_per_page = min(per_page, self._MAX_RESULTS_PER_PAGE)

        params: dict[str, str | int] = {
            "key": self._api_key,
            "q": query,
            "page": page,
            "per_page": clamped_per_page,
            "image_type": "photo",
            "safesearch": "true",
        }

        response = self._http.get(
            url=self._SEARCH_URL,
            params=params,
        )

        data = response.json()
        hits: list[dict[str, object]] = data.get("hits", [])

        results: list[SearchResult] = []
        for hit in hits:
            search_result = self._parse_hit(hit, query)
            if self._is_large_enough(search_result):
                results.append(search_result)

        return results

    def download(
        self, result: SearchResult, destination_directory: Path
    ) -> DownloadResult:
        """Download a single Pixabay image to the given directory.

        The file is saved as ``pixabay_<id>.<ext>`` to avoid
        collisions with other providers.

        Parameters
        ----------
        result:
            The search result to download.
        destination_directory:
            Target folder for the saved file.

        Returns
        -------
        DownloadResult
            Outcome of the download attempt.
        """
        start_time = time.monotonic()

        try:
            response = self._http.get(url=result.download_url)

            extension = self._extract_extension(result.download_url)
            filename = f"{self._SOURCE_NAME}_{result.id}{extension}"
            file_path = destination_directory / filename

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_bytes(response.content)

            elapsed = time.monotonic() - start_time

            return DownloadResult(
                success=True,
                local_path=file_path,
                error_message="",
                download_time_seconds=elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start_time

            return DownloadResult(
                success=False,
                local_path=None,
                error_message=str(exc),
                download_time_seconds=elapsed,
            )

    def build_metadata(self, result: SearchResult, local_file: Path) -> ImageMetadata:
        """Construct an :class:`ImageMetadata` record for a downloaded image.

        Parameters
        ----------
        result:
            The original search result.
        local_file:
            Path to the saved image on disk.

        Returns
        -------
        ImageMetadata
            Complete metadata record.
        """
        return ImageMetadata(
            id=result.id,
            source=self._SOURCE_NAME,
            local_path=local_file,
            download_url=result.download_url,
            page_url=result.page_url,
            width=result.width,
            height=result.height,
            photographer=result.photographer,
            license_name=result.license_name,
            query=result.query,
        )

    def close(self) -> None:
        """Close the underlying HTTP client and release resources."""
        self._http.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_hit(self, hit: dict[str, object], query: str) -> SearchResult:
        """Convert a raw Pixabay hit JSON object into a SearchResult."""
        hit_id = str(hit.get("id", ""))
        user = str(hit.get("user", ""))
        image_width = int(hit.get("imageWidth", 0))
        image_height = int(hit.get("imageHeight", 0))

        large_url = str(hit.get("largeImageURL", ""))
        preview_url = str(hit.get("webformatURL", ""))
        page_url = str(hit.get("pageURL", ""))

        return SearchResult(
            id=hit_id,
            download_url=large_url,
            preview_url=preview_url,
            page_url=page_url,
            width=image_width,
            height=image_height,
            photographer=user,
            license_name=self._LICENSE_NAME,
            query=query,
            source=self._SOURCE_NAME,
        )

    def _is_large_enough(self, result: SearchResult) -> bool:
        """Check whether an image meets minimum dimension requirements."""
        return (
            result.width >= self._settings.MIN_IMAGE_WIDTH
            and result.height >= self._settings.MIN_IMAGE_HEIGHT
        )

    @staticmethod
    def _extract_extension(url: str) -> str:
        """Extract a supported file extension from a URL.

        Falls back to ``.jpg`` when the URL has no recognizable
        extension or the extension is not in the supported set.
        """
        supported: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")

        try:
            path_part = url.split("?")[0]
            filename = path_part.split("/")[-1]
            if "." in filename:
                ext = "." + filename.rsplit(".", 1)[-1].lower()
                if ext in supported:
                    return ext
        except (IndexError, AttributeError):
            pass

        return ".jpg"
