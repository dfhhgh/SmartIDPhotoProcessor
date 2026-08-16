"""
Smoke test for all image sources in the Dataset Builder.

Verifies that each configured source (Pexels, Pixabay, Openverse,
Wikimedia Commons) can successfully participate in the source
architecture: validation -> search -> download -> metadata.

This is an integration test only — no dataset collection,
no retraining, no pipeline modifications.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so dataset_builder is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

from dataset_builder.config.settings import Settings
from dataset_builder.sources.base_source import (
    BaseSource,
    DownloadResult,
    ImageMetadata,
    SearchResult,
)
from dataset_builder.sources.pexels import PexelsSource
from dataset_builder.sources.pixabay import PixabaySource
from dataset_builder.sources.openverse import OpenverseSource
from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

_STATUS_PASS = "PASS"
_STATUS_FAIL = "FAIL"
_STATUS_SKIP = "SKIPPED"
_STATUS_NOT_TESTED = "NOT TESTED"


@dataclass
class SourceResult:
    """Aggregated test result for a single source."""

    name: str
    configuration: str = _STATUS_SKIP
    search: str = _STATUS_SKIP
    search_count: int = 0
    download: str = _STATUS_SKIP
    image_decode: str = _STATUS_SKIP
    metadata: str = _STATUS_SKIP
    pipeline_compat: str = _STATUS_SKIP
    failure_category: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "configuration": self.configuration,
            "search": self.search,
            "search_results": self.search_count,
            "download": self.download,
            "image_decode": self.image_decode,
            "metadata": self.metadata,
            "pipeline_compat": self.pipeline_compat,
            "failure_category": self.failure_category,
            "error_message": self.error_message,
        }


# ---------------------------------------------------------------------------
# Source registry — mirrors main.py _create_sources()
# ---------------------------------------------------------------------------

SOURCE_REGISTRY: dict[str, type[BaseSource]] = {
    "pexels": PexelsSource,
    "pixabay": PixabaySource,
    "openverse": OpenverseSource,
    "wikimedia_commons": WikimediaCommonsSource,
}

# Generic query suitable for face/person searches across all sources
DEFAULT_QUERY = "portrait"

# Small result count for smoke testing
SEARCH_RESULTS_COUNT = 5


# ---------------------------------------------------------------------------
# Network error classification
# ---------------------------------------------------------------------------

def _classify_error(exc: Exception) -> tuple[str, str]:
    """Classify an exception into a human-readable failure category.

    Returns (category, message) tuple.
    """
    exc_str = str(exc).lower()

    # Network connectivity issues
    if any(kw in exc_str for kw in ("connection", "connect", "econnreset", "timeout", "network")):
        if "econnreset" in exc_str:
            return "network_connectivity", f"Network reset (ECONNRESET): {exc}"
        if "timeout" in exc_str:
            return "network_timeout", f"Connection timed out: {exc}"
        return "network_connectivity", f"Network connectivity failure: {exc}"

    # Authentication / configuration issues
    if any(kw in exc_str for kw in ("401", "403", "unauthorized", "forbidden", "api key")):
        return "authentication", f"Authentication/configuration failure: {exc}"

    # HTTP/API errors
    if any(kw in exc_str for kw in ("http", "status", "response", "api")):
        return "http_api", f"HTTP/API failure: {exc}"

    # General request errors
    if "request" in exc_str:
        return "http_api", f"Request failure: {exc}"

    return "code_failure", f"Source code failure: {exc}"


# ---------------------------------------------------------------------------
# Image decoding helper
# ---------------------------------------------------------------------------

def _verify_image_decode(file_path: Path) -> tuple[bool, str]:
    """Verify that an image file can be decoded by PIL/OpenCV.

    Returns (success, message).
    """
    try:
        from PIL import Image

        with Image.open(file_path) as img:
            width, height = img.size
            if width <= 0 or height <= 0:
                return False, f"Invalid image dimensions: {width}x{height}"
            return True, f"Decoded {width}x{height}"
    except ImportError:
        pass

    # Fallback: try OpenCV
    try:
        import cv2

        img = cv2.imread(str(file_path))
        if img is None:
            return False, "OpenCV returned None — file may not be a valid image"
        height, width = img.shape[:2]
        if width <= 0 or height <= 0:
            return False, f"Invalid image dimensions: {width}x{height}"
        return True, f"Decoded {width}x{height}"
    except Exception as exc:
        return False, f"Image decode failed: {exc}"


# ---------------------------------------------------------------------------
# Optional pipeline compatibility check
# ---------------------------------------------------------------------------

def _check_pipeline_compat(
    source: BaseSource,
    result: SearchResult,
    local_file: Path,
) -> str:
    """Lightweight check that a downloaded image can enter downstream processing.

    Only checks that the image is readable and metadata is constructible.
    Does NOT run face detection, quality filters, or the full pipeline.
    """
    try:
        # Verify metadata can be built
        metadata = source.build_metadata(result, local_file)
        if not metadata.local_path.exists():
            return _STATUS_FAIL

        # Verify the image is decodable
        ok, _ = _verify_image_decode(local_file)
        if not ok:
            return _STATUS_FAIL

        return _STATUS_PASS
    except Exception:
        return _STATUS_FAIL


# ---------------------------------------------------------------------------
# Core test functions
# ---------------------------------------------------------------------------

def _test_source(
    source_name: str,
    source_cls: type[BaseSource],
    settings: Settings,
) -> SourceResult:
    """Run the full smoke test for a single source."""
    result = SourceResult(name=source_name)

    # ------------------------------------------------------------------
    # Step 3: Configuration / Validation
    # ------------------------------------------------------------------
    try:
        source = source_cls(settings)
        source.validate_configuration()
        result.configuration = _STATUS_PASS
    except ValueError as exc:
        # Missing API credentials — not a code failure
        msg = str(exc)
        if "missing" in msg.lower() or "key" in msg.lower():
            result.configuration = _STATUS_SKIP
            result.failure_category = "missing_credentials"
            result.error_message = msg
            source.close()
            return result
        # Invalid key is still a configuration issue
        result.configuration = _STATUS_FAIL
        result.failure_category = "authentication"
        result.error_message = msg
        source.close()
        return result
    except Exception as exc:
        cat, msg = _classify_error(exc)
        result.configuration = _STATUS_FAIL
        result.failure_category = cat
        result.error_message = msg
        try:
            source.close()
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Step 4: Search
    # ------------------------------------------------------------------
    try:
        search_results = source.search(
            query=DEFAULT_QUERY,
            page=1,
            per_page=SEARCH_RESULTS_COUNT,
        )
        result.search_count = len(search_results)
        result.search = _STATUS_PASS
    except Exception as exc:
        cat, msg = _classify_error(exc)
        result.search = _STATUS_FAIL
        result.failure_category = cat
        result.error_message = msg
        source.close()
        return result

    if len(search_results) == 0:
        result.search = _STATUS_PASS  # empty results are valid
        result.search_count = 0
        result.download = _STATUS_SKIP
        result.image_decode = _STATUS_SKIP
        result.metadata = _STATUS_SKIP
        result.pipeline_compat = _STATUS_NOT_TESTED
        source.close()
        return result

    # Validate SearchResult fields
    for sr in search_results:
        assert isinstance(sr.id, str) and len(sr.id) > 0, "SearchResult.id must be non-empty string"
        assert isinstance(sr.source, str) and sr.source == source_name, f"SearchResult.source must be '{source_name}'"
        assert isinstance(sr.download_url, str) and sr.download_url.startswith("http"), "SearchResult.download_url must be an HTTP URL"
        assert isinstance(sr.page_url, str), "SearchResult.page_url must be a string"
        assert isinstance(sr.width, int) and sr.width > 0, "SearchResult.width must be positive int"
        assert isinstance(sr.height, int) and sr.height > 0, "SearchResult.height must be positive int"
        assert isinstance(sr.photographer, str), "SearchResult.photographer must be a string"
        assert isinstance(sr.license_name, str), "SearchResult.license_name must be a string"
        assert isinstance(sr.query, str) and sr.query == DEFAULT_QUERY, f"SearchResult.query must be '{DEFAULT_QUERY}'"

    # ------------------------------------------------------------------
    # Step 5 & 6: Download + Metadata (using first result)
    # ------------------------------------------------------------------
    test_result = search_results[0]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        try:
            download_result = source.download(test_result, tmp_path)

            if not download_result.success:
                result.download = _STATUS_FAIL
                cat, msg = _classify_error(Exception(download_result.error_message))
                result.failure_category = cat
                result.error_message = msg
                source.close()
                return result

            result.download = _STATUS_PASS

            # Verify file exists and is not empty
            local_path = download_result.local_path
            assert local_path is not None, "DownloadResult.local_path must not be None on success"
            assert local_path.exists(), f"Downloaded file does not exist: {local_path}"
            assert local_path.stat().st_size > 0, f"Downloaded file is empty: {local_path}"

            # Step 5: Image decode
            decode_ok, decode_msg = _verify_image_decode(local_path)
            if decode_ok:
                result.image_decode = _STATUS_PASS
            else:
                result.image_decode = _STATUS_FAIL
                result.failure_category = "image_decode"
                result.error_message = decode_msg
                source.close()
                return result

            # Step 6: Metadata
            try:
                metadata = source.build_metadata(test_result, local_path)
                assert metadata.source == source_name
                assert metadata.id == test_result.id
                assert metadata.local_path == local_path
                assert metadata.download_url == test_result.download_url
                assert metadata.page_url == test_result.page_url
                assert metadata.width == test_result.width
                assert metadata.height == test_result.height
                assert metadata.photographer == test_result.photographer
                assert metadata.license_name == test_result.license_name
                result.metadata = _STATUS_PASS
            except Exception as exc:
                result.metadata = _STATUS_FAIL
                result.failure_category = "metadata"
                result.error_message = str(exc)

            # Step 7: Pipeline compatibility (lightweight)
            result.pipeline_compat = _check_pipeline_compat(source, test_result, local_path)

        except Exception as exc:
            cat, msg = _classify_error(exc)
            result.download = _STATUS_FAIL
            result.failure_category = cat
            result.error_message = msg

    source.close()
    return result


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _print_report(results: list[SourceResult]) -> None:
    """Print a human-readable console report."""
    print("\n" + "=" * 60)
    print("DATASET BUILDER SOURCE SMOKE TEST")
    print("=" * 60)

    for r in results:
        print(f"\n{r.name}")
        print(f"  Configuration: {r.configuration}")
        if r.search == _STATUS_SKIP and r.failure_category == "missing_credentials":
            print(f"  Search:        SKIPPED — {r.error_message}")
        else:
            print(f"  Search:        {r.search} ({r.search_count} results)")
        print(f"  Download:      {r.download}")
        print(f"  Image decode:  {r.image_decode}")
        print(f"  Metadata:      {r.metadata}")
        print(f"  Pipeline:      {r.pipeline_compat}")
        if r.failure_category:
            print(f"  Failure cat.:  {r.failure_category}")
        if r.error_message:
            print(f"  Error:         {r.error_message[:120]}")

    # Summary
    total = len(results)
    passed = sum(1 for r in results if all(
        s == _STATUS_PASS
        for s in [r.configuration, r.search, r.download, r.image_decode, r.metadata]
    ))
    failed = sum(1 for r in results if any(
        s == _STATUS_FAIL
        for s in [r.configuration, r.search, r.download, r.image_decode, r.metadata]
    ))
    skipped = total - passed - failed

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sources tested: {total}")
    print(f"Passed:         {passed}")
    print(f"Failed:         {failed}")
    print(f"Skipped:        {skipped}")
    print("=" * 60 + "\n")


def _write_json_report(results: list[SourceResult], output_path: Path) -> None:
    """Write a machine-readable JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "sources": {r.name: r.to_dict() for r in results},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"JSON report written to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Run the source smoke test and return exit code."""
    settings = Settings()

    # Determine which sources to test (use ENABLED_SOURCES from settings,
    # but also include all sources in registry for comprehensive testing)
    sources_to_test = list(SOURCE_REGISTRY.keys())

    results: list[SourceResult] = []

    for source_name in sources_to_test:
        source_cls = SOURCE_REGISTRY.get(source_name)
        if source_cls is None:
            results.append(SourceResult(
                name=source_name,
                configuration=_STATUS_FAIL,
                failure_category = "code_failure",
                error_message=f"Source class not found in registry: {source_name}",
            ))
            continue

        print(f"\nTesting {source_name}...")
        try:
            r = _test_source(source_name, source_cls, settings)
            results.append(r)
        except Exception as exc:
            cat, msg = _classify_error(exc)
            results.append(SourceResult(
                name=source_name,
                configuration=_STATUS_FAIL,
                failure_category=cat,
                error_message=msg,
            ))

    # Output
    _print_report(results)

    # JSON report
    report_path = _PROJECT_ROOT / "reports" / "source_smoke_test_report.json"
    _write_json_report(results, report_path)

    # Exit code: 0 if all passed or skipped, 1 if any failed
    has_failure = any(
        any(s == _STATUS_FAIL for s in [r.configuration, r.search, r.download, r.image_decode, r.metadata])
        for r in results
    )
    return 1 if has_failure else 0


if __name__ == "__main__":
    sys.exit(main())
