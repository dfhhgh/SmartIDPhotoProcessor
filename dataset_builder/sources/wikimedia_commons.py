"""
Wikimedia Commons image source provider.

Adapter between the Wikimedia Commons MediaWiki API and the Dataset Builder's
:class:`BaseSource` contract.
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


class WikimediaCommonsSource(BaseSource):
    """Concrete :class:`BaseSource` implementation for the Wikimedia Commons API."""

    _API_URL: str = "https://commons.wikimedia.org/w/api.php"
    _SOURCE_NAME: str = "wikimedia_commons"
    _MAX_RESULTS_PER_PAGE: int = 50

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        # Wikimedia requires a descriptive User-Agent with real contact info.
        # See: https://foundation.wikimedia.org/wiki/Policy:User-Agent_policy
        headers = dict(settings.DEFAULT_HEADERS)
        headers["User-Agent"] = settings.WIKIMEDIA_USER_AGENT
        self._http: HTTPClient = HTTPClient(settings)
        self._custom_headers = headers

    @property
    def name(self) -> str:
        return self._SOURCE_NAME

    def validate_configuration(self) -> bool:
        """Validate Wikimedia Commons API configuration by making a test request."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "meta": "siteinfo",
            }
            response = self._http.get(
                url=self._API_URL,
                params=params,
                headers=self._custom_headers,
            )
            return response.status_code == 200
        except Exception:
            raise

    def search(self, query: str, page: int, per_page: int) -> list[SearchResult]:
        """Search Wikimedia Commons using MediaWiki generator=search with namespace=6 (File)."""
        clamped_per_page = min(per_page, self._MAX_RESULTS_PER_PAGE)

        # Pagination in MediaWiki query search uses gsroffset.
        # page=1 -> offset=0, page=2 -> offset=clamped_per_page, etc.
        offset = (page - 1) * clamped_per_page

        params: dict[str, str | int] = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,  # Namespace 6 = File:
            "gsrlimit": clamped_per_page,
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|size|extmetadata",
        }

        response = self._http.get(
            url=self._API_URL,
            params=params,
            headers=self._custom_headers,
        )

        data = response.json()
        query_data = data.get("query", {})
        pages = query_data.get("pages", {})

        results: list[SearchResult] = []
        for page_id, page_info in pages.items():
            if not isinstance(page_info, dict):
                continue
            search_result = self._parse_page_info(page_info, query)
            if search_result and self._is_large_enough(search_result):
                results.append(search_result)

        return results

    def download(
        self, result: SearchResult, destination_directory: Path
    ) -> DownloadResult:
        start_time = time.monotonic()

        try:
            response = self._http.get(
                url=result.download_url,
                headers=self._custom_headers,
            )

            extension = self._extract_extension(result.download_url)
            safe_id = self._sanitize_id(result.id)
            filename = f"wikimedia_{safe_id}{extension}"
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
            license_url=result.license_url,
            license_type=result.license_type,
        )

    def close(self) -> None:
        self._http.close()

    def _parse_page_info(self, page_info: dict[str, object], query: str) -> SearchResult | None:
        page_id = str(page_info.get("pageid", ""))
        title = str(page_info.get("title", ""))
        imageinfo = page_info.get("imageinfo", [])
        if not imageinfo or not isinstance(imageinfo, list):
            return None

        info = imageinfo[0]
        if not isinstance(info, dict):
            return None

        download_url = str(info.get("url", ""))
        if not download_url:
            return None

        width = int(info.get("width", 0))
        height = int(info.get("height", 0))
        descriptionurl = str(info.get("descriptionurl", ""))

        extmetadata = info.get("extmetadata", {})
        if not isinstance(extmetadata, dict):
            extmetadata = {}

        # Extract artist/creator
        artist_dict = extmetadata.get("Artist", {})
        artist = str(artist_dict.get("value", "Unknown")) if isinstance(artist_dict, dict) else "Unknown"
        # Clean HTML tags if present in artist string (MediaWiki often returns HTML)
        photographer = self._strip_html(artist) or "Unknown"

        # Extract license info
        license_short = extmetadata.get("LicenseShortName", {})
        license_name = str(license_short.get("value", "Unknown")) if isinstance(license_short, dict) else "Unknown"

        license_url_dict = extmetadata.get("LicenseUrl", {})
        license_url = str(license_url_dict.get("value", "")) if isinstance(license_url_dict, dict) else None

        license_dict = extmetadata.get("License", {})
        license_type = str(license_dict.get("value", "")) if isinstance(license_dict, dict) else None

        return SearchResult(
            id=page_id or title,
            download_url=download_url,
            preview_url=download_url,
            page_url=descriptionurl,
            width=width,
            height=height,
            photographer=photographer,
            license_name=license_name,
            query=query,
            source=self._SOURCE_NAME,
            license_url=license_url,
            license_type=license_type,
        )

    def _is_large_enough(self, result: SearchResult) -> bool:
        return (
            result.width >= self._settings.MIN_IMAGE_WIDTH
            and result.height >= self._settings.MIN_IMAGE_HEIGHT
        )

    @staticmethod
    def _sanitize_id(item_id: str) -> str:
        """Make ID/title safe for filenames on Windows and Unix."""
        return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in item_id)

    @staticmethod
    def _extract_extension(url: str) -> str:
        supported: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp", ".svg")

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

    @staticmethod
    def _strip_html(text: str) -> str:
        """Basic HTML tag stripper for MediaWiki attribution fields."""
        import re
        clean = re.sub(r"<.*?>", "", text)
        return clean.strip()
