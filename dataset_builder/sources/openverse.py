"""
Openverse image source provider.

Adapter between the Openverse API and the Dataset Builder's
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


class OpenverseSource(BaseSource):
    """Concrete :class:`BaseSource` implementation for the Openverse API."""

    _SEARCH_URL: str = "https://api.openverse.org/v1/images/"
    _SOURCE_NAME: str = "openverse"
    _MAX_RESULTS_PER_PAGE: int = 50

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._http: HTTPClient = HTTPClient(settings)

    @property
    def name(self) -> str:
        return self._SOURCE_NAME

    def validate_configuration(self) -> bool:
        """Validate Openverse API configuration by making a test request."""
        try:
            response = self._http.get(
                url=self._SEARCH_URL,
                params={"q": "test", "page_size": 1},
            )
            return response.status_code == 200
        except Exception:
            # If network or API is down during validation, we can raise or return False
            # but per Pexels/Pixabay convention, raising HTTPError/ValueError or returning True/False
            raise

    def search(self, query: str, page: int, per_page: int) -> list[SearchResult]:
        clamped_per_page = min(per_page, self._MAX_RESULTS_PER_PAGE)

        params: dict[str, str | int] = {
            "q": query,
            "page": page,
            "page_size": clamped_per_page,
            "mature": "false",
        }

        response = self._http.get(
            url=self._SEARCH_URL,
            params=params,
        )

        data = response.json()
        results_list: list[dict[str, object]] = data.get("results", [])

        results: list[SearchResult] = []
        for item in results_list:
            search_result = self._parse_item(item, query)
            if search_result and self._is_large_enough(search_result):
                results.append(search_result)

        return results

    def download(
        self, result: SearchResult, destination_directory: Path
    ) -> DownloadResult:
        start_time = time.monotonic()

        try:
            response = self._http.get(url=result.download_url)

            extension = self._extract_extension(result.download_url)
            safe_id = self._sanitize_id(result.id)
            filename = f"{self._SOURCE_NAME}_{safe_id}{extension}"
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

    def _parse_item(self, item: dict[str, object], query: str) -> SearchResult | None:
        item_id = str(item.get("id", ""))
        download_url = str(item.get("url", ""))
        if not item_id or not download_url:
            return None

        width = int(item.get("width", 0))
        height = int(item.get("height", 0))
        creator = str(item.get("creator", "Unknown"))
        if not creator or creator == "None":
            creator = "Unknown"

        page_url = str(item.get("foreign_landing_url", "")) or str(item.get("detail_url", ""))
        license_type = str(item.get("license", ""))
        license_version = str(item.get("license_version", ""))
        license_name = f"CC {license_type.upper()} {license_version}".strip() if license_type else "Unknown"
        license_url = str(item.get("license_url", "")) or None

        thumbnail = str(item.get("thumbnail", download_url))

        return SearchResult(
            id=item_id,
            download_url=download_url,
            preview_url=thumbnail,
            page_url=page_url,
            width=width,
            height=height,
            photographer=creator,
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
        """Make ID safe for filenames on Windows and Unix."""
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
