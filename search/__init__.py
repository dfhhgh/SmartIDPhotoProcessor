"""Vector search core abstraction.

Provides a minimal, domain-agnostic FAISS-based vector search layer.
This module knows nothing about faces, students, celebrities, or
validation rules — it only works with NumPy float32 vectors.
"""

from search.flat_index import FlatIndex

__all__ = ["FlatIndex"]
