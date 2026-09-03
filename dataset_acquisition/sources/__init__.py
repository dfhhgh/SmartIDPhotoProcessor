"""Source abstraction for image providers."""

from dataset_acquisition.sources.base import ImageSource
from dataset_acquisition.sources.wikimedia import WikimediaSource

__all__ = ["ImageSource", "WikimediaSource"]
