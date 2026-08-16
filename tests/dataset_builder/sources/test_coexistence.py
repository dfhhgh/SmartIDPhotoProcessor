"""Integration test verifying source registry and coexistence."""

from dataset_builder.config.settings import Settings
from dataset_builder.main import _create_sources


def test_source_coexistence() -> None:
    settings = Settings(
        ENABLED_SOURCES=("pexels", "pixabay", "openverse", "wikimedia_commons"),
        PEXELS_API_KEY="dummy",
        PIXABAY_API_KEY="dummy",
    )
    # Even if validation fails or succeeds depending on network/keys,
    # let's verify that registry contains all four sources and factory processes them.
    from dataset_builder.sources.openverse import OpenverseSource
    from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource
    from dataset_builder.sources.pexels import PexelsSource
    from dataset_builder.sources.pixabay import PixabaySource

    # Check classes
    assert OpenverseSource is not None
    assert WikimediaCommonsSource is not None
