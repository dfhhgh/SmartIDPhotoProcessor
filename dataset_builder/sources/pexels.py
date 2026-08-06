"""
Pexels image source provider.

Adapter between the Pexels API and the Dataset Builder's
:class:`BaseSource` contract.  All Pexels-specific URL construction,
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


class PexelsSource(BaseSource):
    """Concrete :class:`BaseSource` implementation for the Pexels API.

    Uses :class:`HTTPClient` for all network communication.  The API
    key is read exclusively from ``Settings.PEXELS_API_KEY``.

    Parameters
    ----------
    settings:
        Application settings containing the Pexels API key and
        downloader configuration.
    """

    _SEARCH_URL: str = "https://api.pexels.com/v1/search"
    _SOURCE_NAME: str = "pexels"
    _LICENSE_NAME: str = "Pexels License"
    _PAGE_URL_TEMPLATE: str = "https://www.pexels.com/photo/{photo_id}/"
    _MAX_RESULTS_PER_PAGE: int = 80

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._api_key: str = settings.PEXELS_API_KEY
        self._http: HTTPClient = HTTPClient(settings)
        self._authenticated_headers: dict[str, str] = {
            "Authorization": self._api_key,
        }

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
        """Validate the Pexels API key by making a test request.

        Returns
        -------
        bool
            ``True`` when the configuration is valid.

        Raises
        ------
        ValueError
            When ``PEXELS_API_KEY`` is missing, empty, or rejected
            by the Pexels API (401/403).
        """
        if not self._api_key:
            raise ValueError(
                "Pexels API key is missing.  "
                "Set the PEXELS_API_KEY environment variable."
            )

        try:
            response = self._http.get(
                url=self._SEARCH_URL,
                params={"query": "test", "per_page": 1},
                headers=self._authenticated_headers,
            )
            return response.status_code == 200
        except Exception as exc:
            status = getattr(exc, "response", None)
            if status is not None and status.status_code in (401, 403):
                raise ValueError(
                    "Pexels API key is invalid.  "
                    "The server returned HTTP "
                    f"{status.status_code}."
                ) from exc
            raise

    def search(self, query: str, page: int, per_page: int) -> list[SearchResult]:
        """Query the Pexels Search API and return matching images.

        Images smaller than ``Settings.MIN_IMAGE_WIDTH`` or
        ``Settings.MIN_IMAGE_HEIGHT`` are automatically discarded.

        Parameters
        ----------
        query:
            Free-text search string.
        page:
            1-indexed page number.
        per_page:
            Number of results per page (clamped to 80).

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
            "query": query,
            "page": page,
            "per_page": clamped_per_page,
        }

        response = self._http.get(
            url=self._SEARCH_URL,
            params=params,
            headers=self._authenticated_headers,
        )

        data = response.json()
        photos: list[dict[str, object]] = data.get("photos", [])

        results: list[SearchResult] = []
        for photo in photos:
            search_result = self._parse_photo(photo, query)
            if self._is_large_enough(search_result):
                results.append(search_result)

        return results

    def download(
        self, result: SearchResult, destination_directory: Path
    ) -> DownloadResult:
        """Download a single Pexels image to the given directory.

        The file is saved as ``pexels_<id>.<ext>`` to avoid
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
            response = self._http.get(
                url=result.download_url,
                headers=self._authenticated_headers,
            )

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

    def _parse_photo(self, photo: dict[str, object], query: str) -> SearchResult:
        """Convert a raw Pexels photo JSON object into a SearchResult."""
        photo_id = str(photo.get("id", ""))
        photographer = str(photo.get("photographer", ""))
        width = int(photo.get("width", 0))
        height = int(photo.get("height", 0))

        src: dict[str, str] = photo.get("src", {})  # type: ignore[assignment]
        original_url = str(src.get("original", ""))

        page_url = self._PAGE_URL_TEMPLATE.format(photo_id=photo_id)

        return SearchResult(
            id=photo_id,
            download_url=original_url,
            preview_url=str(src.get("medium", original_url)),
            page_url=page_url,
            width=width,
            height=height,
            photographer=photographer,
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
