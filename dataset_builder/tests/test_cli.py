"""Unit tests for CLI argument parsing and category selection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dataset_builder.main import (
    _validate_categories,
    _validate_max_per_query,
    parse_args,
)


# ------------------------------------------------------------------
# parse_args tests
# ------------------------------------------------------------------


class TestParseArgs:
    """Tests for the parse_args function."""

    def test_no_arguments(self) -> None:
        """No CLI arguments produces defaults."""
        args = parse_args([])
        assert args.categories is None
        assert args.max_per_query is None

    def test_single_category(self) -> None:
        """Single --categories value."""
        args = parse_args(["--categories", "hijab"])
        assert args.categories == ["hijab"]
        assert args.max_per_query is None

    def test_multiple_categories(self) -> None:
        """Multiple --categories values."""
        args = parse_args(["--categories", "hijab", "beard"])
        assert args.categories == ["hijab", "beard"]

    def test_max_per_query(self) -> None:
        """--max-per-query with positive integer."""
        args = parse_args(["--max-per-query", "5"])
        assert args.max_per_query == 5
        assert args.categories is None

    def test_categories_and_max_per_query(self) -> None:
        """Both --categories and --max-per-query together."""
        args = parse_args(["--categories", "hijab", "--max-per-query", "5"])
        assert args.categories == ["hijab"]
        assert args.max_per_query == 5

    def test_max_per_query_zero_rejected(self) -> None:
        """--max-per-query 0 is rejected by argparse type=int (valid int)
        but caught by validation later."""
        args = parse_args(["--max-per-query", "0"])
        assert args.max_per_query == 0

    def test_max_per_query_negative_rejected(self) -> None:
        """--max-per-query -1 is a valid int but caught by validation."""
        args = parse_args(["--max-per-query", "-1"])
        assert args.max_per_query == -1

    def test_max_per_query_invalid_string(self) -> None:
        """--max-per-query with non-integer string is rejected by argparse."""
        with pytest.raises(SystemExit):
            parse_args(["--max-per-query", "abc"])

    def test_max_per_query_float_rejected(self) -> None:
        """--max-per-query with float string is rejected by argparse."""
        with pytest.raises(SystemExit):
            parse_args(["--max-per-query", "3.5"])


# ------------------------------------------------------------------
# _validate_categories tests
# ------------------------------------------------------------------


class TestValidateCategories:
    """Tests for the _validate_categories function."""

    AVAILABLE = ["beard", "cap", "eyeglasses", "hijab", "normal"]

    def test_none_returns_none(self) -> None:
        """None input returns None (process all)."""
        assert _validate_categories(None, self.AVAILABLE) is None

    def test_valid_single_category(self) -> None:
        """Single valid category passes."""
        result = _validate_categories(["hijab"], self.AVAILABLE)
        assert result == ["hijab"]

    def test_valid_multiple_categories(self) -> None:
        """Multiple valid categories pass."""
        result = _validate_categories(["hijab", "beard"], self.AVAILABLE)
        assert result == ["hijab", "beard"]

    def test_unknown_category_exits(self) -> None:
        """Unknown category triggers sys.exit."""
        with pytest.raises(SystemExit) as exc_info:
            _validate_categories(["nonexistent"], self.AVAILABLE)
        assert exc_info.value.code == 1

    def test_one_invalid_among_valid(self) -> None:
        """One invalid category among valid ones triggers sys.exit."""
        with pytest.raises(SystemExit):
            _validate_categories(["hijab", "nonexistent"], self.AVAILABLE)


# ------------------------------------------------------------------
# _validate_max_per_query tests
# ------------------------------------------------------------------


class TestValidateMaxPerQuery:
    """Tests for the _validate_max_per_query function."""

    def test_none_returns_none(self) -> None:
        """None input returns None."""
        assert _validate_max_per_query(None) is None

    def test_positive_value_passes(self) -> None:
        """Positive integer passes."""
        assert _validate_max_per_query(5) == 5

    def test_one_passes(self) -> None:
        """Value of 1 passes (minimum valid)."""
        assert _validate_max_per_query(1) == 1

    def test_zero_exits(self) -> None:
        """Zero triggers sys.exit."""
        with pytest.raises(SystemExit) as exc_info:
            _validate_max_per_query(0)
        assert exc_info.value.code == 1

    def test_negative_exits(self) -> None:
        """Negative value triggers sys.exit."""
        with pytest.raises(SystemExit) as exc_info:
            _validate_max_per_query(-1)
        assert exc_info.value.code == 1


# ------------------------------------------------------------------
# Downloader max_per_query override tests
# ------------------------------------------------------------------


class TestDownloaderMaxPerQuery:
    """Tests for Downloader max_per_query override."""

    def test_default_uses_settings_value(self) -> None:
        """Without override, _max_per_query_override is None."""
        from dataset_builder.downloader.downloader import Downloader

        settings = MagicMock()
        settings.MAX_IMAGES_PER_QUERY = 200
        query_loader = MagicMock()
        sources: list = []
        dl = Downloader(settings, query_loader, sources)
        assert dl._max_per_query_override is None

    def test_override_stored(self) -> None:
        """Override value is stored on the Downloader."""
        from dataset_builder.downloader.downloader import Downloader

        settings = MagicMock()
        settings.MAX_IMAGES_PER_QUERY = 200
        query_loader = MagicMock()
        sources: list = []
        dl = Downloader(settings, query_loader, sources, max_per_query=5)
        assert dl._max_per_query_override == 5


# ------------------------------------------------------------------
# Downloader.download_categories tests
# ------------------------------------------------------------------


class TestDownloaderDownloadCategories:
    """Tests for Downloader.download_categories method."""

    def test_calls_download_category_for_each(self) -> None:
        """download_categories calls download_category for each item."""
        from dataset_builder.downloader.downloader import Downloader

        settings = MagicMock()
        settings.MAX_IMAGES_PER_QUERY = 200
        query_loader = MagicMock()
        sources: list = []
        dl = Downloader(settings, query_loader, sources)

        with patch.object(dl, "download_category") as mock_dl:
            dl.download_categories(["hijab", "beard"])
            assert mock_dl.call_count == 2
            mock_dl.assert_any_call("hijab")
            mock_dl.assert_any_call("beard")

    def test_empty_list_no_calls(self) -> None:
        """Empty category list results in no download calls."""
        from dataset_builder.downloader.downloader import Downloader

        settings = MagicMock()
        settings.MAX_IMAGES_PER_QUERY = 200
        query_loader = MagicMock()
        sources: list = []
        dl = Downloader(settings, query_loader, sources)

        with patch.object(dl, "download_category") as mock_dl:
            dl.download_categories([])
            mock_dl.assert_not_called()


# ------------------------------------------------------------------
# DatasetBuilder.build categories parameter tests
# ------------------------------------------------------------------


class TestDatasetBuilderBuild:
    """Tests for DatasetBuilder.build categories parameter."""

    def _make_builder(self) -> object:
        """Create a DatasetBuilder with mock dependencies."""
        from dataset_builder.dataset_pipeline import DatasetBuilder

        return DatasetBuilder(
            settings=MagicMock(),
            query_loader=MagicMock(),
            downloader=MagicMock(),
            metadata_manager=MagicMock(),
            duplicate_remover=MagicMock(),
            face_filter=MagicMock(),
            quality_filter=MagicMock(),
            statistics_aggregator=MagicMock(),
            report_generator=MagicMock(),
        )

    def test_build_no_categories_calls_download_all(self) -> None:
        """build() without categories calls download_all."""
        builder = self._make_builder()
        builder.build()
        builder._downloader.download_all.assert_called_once()

    def test_build_with_categories_calls_download_categories(self) -> None:
        """build(categories=[...]) calls download_categories."""
        builder = self._make_builder()
        builder.build(categories=["hijab"])
        builder._downloader.download_categories.assert_called_once_with(
            ["hijab"]
        )

    def test_build_with_categories_not_called_download_all(self) -> None:
        """build(categories=[...]) does NOT call download_all."""
        builder = self._make_builder()
        builder.build(categories=["hijab"])
        builder._downloader.download_all.assert_not_called()
