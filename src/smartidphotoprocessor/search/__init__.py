"""Vector search core abstraction.

Provides a minimal, domain-agnostic FAISS-based vector search layer.
This module knows nothing about faces, students, celebrities, or
validation rules — it only works with NumPy float32 vectors.

Note: faiss is an optional dependency (pip install smartidphotoprocessor[reverse-search]).
FlatIndex is imported lazily to avoid ImportError when faiss is not installed.
"""


def __getattr__(name: str):
    if name == "FlatIndex":
        from smartidphotoprocessor.search.flat_index import FlatIndex
        return FlatIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FlatIndex"]
