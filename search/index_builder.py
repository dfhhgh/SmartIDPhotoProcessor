"""Reference dataset ingestion and FAISS index builder.

Transforms a local reference image directory into:
- reference_index.faiss (FAISS IndexFlatIP)
- metadata.json (Mapping from FAISS vector ID to person and image path)

Uses the existing InsightFace integration via FaceService and the
Phase 13.1 FlatIndex & EmbeddingValidator.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from pipeline.selector import FaceSelector
from search.embedding_validator import EmbeddingError, EmbeddingValidator
from search.flat_index import FlatIndex
from services.face_service import FaceService

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# Result & Report models
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MetadataRecord:
    vector_id: int
    person_id: str
    label: str
    image: str  # Relative path from dataset root

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector_id": self.vector_id,
            "person_id": self.person_id,
            "label": self.label,
            "image": self.image,
        }


@dataclass(frozen=True, slots=True)
class SkippedImage:
    image_path: str
    person_id: str
    reason: str


@dataclass(frozen=True, slots=True)
class BuildReport:
    total_input_images: int
    accepted_embeddings: int
    skipped_images: int
    total_persons: int
    skipped_records: tuple[SkippedImage, ...]
    index_path: str
    metadata_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_input_images": self.total_input_images,
            "accepted_embeddings": self.accepted_embeddings,
            "skipped_images": self.skipped_images,
            "total_persons": self.total_persons,
            "skipped_records": [
                {
                    "image_path": s.image_path,
                    "person_id": s.person_id,
                    "reason": s.reason,
                }
                for s in self.skipped_records
            ],
            "index_path": self.index_path,
            "metadata_path": self.metadata_path,
        }


# ---------------------------------------------------------------------------
# IndexBuilder
# ---------------------------------------------------------------------------

class IndexBuilder:
    """Ingests a reference dataset directory and builds FAISS index + metadata."""

    def __init__(self, dimension: int = 512) -> None:
        self._dimension = dimension
        self._validator = EmbeddingValidator(dimension=dimension, normalize=False)
        self._face_service = FaceService()
        self._selector = FaceSelector()

    def build(
        self,
        dataset_dir: Path | str,
        output_dir: Path | str,
        index_filename: str = "reference_index.faiss",
        metadata_filename: str = "metadata.json",
    ) -> tuple[FlatIndex, BuildReport]:
        """Build FAISS index and metadata from *dataset_dir*.

        Dataset Structure Expected:
            dataset_dir/
                person_001/
                    img1.jpg
                    img2.jpg
                person_002/
                    img1.jpg

        Parameters
        ----------
        dataset_dir:
            Path to reference dataset root directory.
        output_dir:
            Directory where ``reference_index.faiss`` and ``metadata.json``
            will be saved.
        index_filename:
            Filename for the FAISS index.
        metadata_filename:
            Filename for the JSON metadata.

        Returns
        -------
        tuple[FlatIndex, BuildReport]
            The populated FAISS index and the build report.

        Raises
        ------
        ValueError
            If the dataset directory does not exist or contains no valid images.
        """
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)

        if not dataset_dir.exists() or not dataset_dir.is_dir():
            raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

        # 1. Discover images deterministically
        person_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
        if not person_dirs:
            raise ValueError(f"No person subdirectories found in {dataset_dir}")

        # Collect all valid (person_id, image_path) pairs in deterministic order
        tasks: list[tuple[str, Path]] = []
        for pdir in person_dirs:
            person_id = pdir.name
            img_paths = sorted([
                p for p in pdir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            ])
            for ipath in img_paths:
                tasks.append((person_id, ipath))

        total_input = len(tasks)
        if total_input == 0:
            raise ValueError(f"No supported images found in dataset: {dataset_dir}")

        # 2. Process images with failure isolation
        model = self._face_service.get_model()
        valid_embeddings: list[np.ndarray] = []
        metadata_records: list[MetadataRecord] = []
        skipped_records: list[SkippedImage] = []
        persons_seen: set[str] = set()

        current_vector_id = 0

        for person_id, img_path in tasks:
            rel_image_str = f"{person_id}/{img_path.name}"
            try:
                # Load image
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    skipped_records.append(
                        SkippedImage(str(img_path), person_id, "IMAGE_UNREADABLE")
                    )
                    continue

                # Detect faces using InsightFace
                faces = model.get(img)
                if not faces:
                    skipped_records.append(
                        SkippedImage(str(img_path), person_id, "NO_FACE")
                    )
                    continue

                # Select primary face if multiple
                if len(faces) == 1:
                    selected_face = faces[0]
                else:
                    try:
                        selection_result = self._selector.select(faces, img.shape)
                        selected_face = selection_result.selected_face
                    except Exception:
                        skipped_records.append(
                            SkippedImage(str(img_path), person_id, "MULTIPLE_FACES_SELECTION_FAILED")
                        )
                        continue

                # Extract embedding
                embedding = getattr(selected_face, "normed_embedding", None)
                if embedding is None:
                    # Fallback to normalizing raw embedding
                    raw_emb = getattr(selected_face, "embedding", None)
                    if raw_emb is None:
                        skipped_records.append(
                            SkippedImage(str(img_path), person_id, "NO_EMBEDDING_PRODUCED")
                        )
                        continue
                    norm = np.linalg.norm(raw_emb)
                    if norm == 0:
                        skipped_records.append(
                            SkippedImage(str(img_path), person_id, "ZERO_NORM_EMBEDDING")
                        )
                        continue
                    embedding = (raw_emb / norm).astype(np.float32)

                # Validate embedding contract
                try:
                    validated = self._validator.validate(embedding)
                except EmbeddingError as exc:
                    skipped_records.append(
                        SkippedImage(str(img_path), person_id, f"INVALID_EMBEDDING: {exc}")
                    )
                    continue

                # Accept embedding
                vec = validated[0]  # Take 1-D vector from (1, 512)
                valid_embeddings.append(vec)
                metadata_records.append(
                    MetadataRecord(
                        vector_id=current_vector_id,
                        person_id=person_id,
                        label=person_id,
                        image=rel_image_str,
                    )
                )
                persons_seen.add(person_id)
                current_vector_id += 1

            except Exception as exc:
                logger.exception("Error processing reference image %s", img_path)
                skipped_records.append(
                    SkippedImage(str(img_path), person_id, f"EXCEPTION: {exc}")
                )

        if not valid_embeddings:
            raise ValueError(
                f"Dataset build failed: 0 of {total_input} images produced valid embeddings."
            )

        # 3. Build FAISS index
        index = FlatIndex(dimension=self._dimension, normalize=False)
        matrix = np.vstack(valid_embeddings).astype(np.float32)
        index.add(matrix)

        # 4. Save artifacts
        output_dir.mkdir(parents=True, exist_ok=True)
        idx_path = output_dir / index_filename
        meta_path = output_dir / metadata_filename

        index.save(idx_path)

        metadata_dict = {
            "schema_version": 1,
            "embedding_dimension": self._dimension,
            "metric": "inner_product",
            "normalized": True,
            "total_vectors": index.size,
            "total_persons": len(persons_seen),
            "records": [r.to_dict() for r in metadata_records],
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        report = BuildReport(
            total_input_images=total_input,
            accepted_embeddings=index.size,
            skipped_images=len(skipped_records),
            total_persons=len(persons_seen),
            skipped_records=tuple(skipped_records),
            index_path=str(idx_path),
            metadata_path=str(meta_path),
        )

        logger.info(
            "Reference dataset built successfully: %d vectors, %d persons, %d skipped.",
            report.accepted_embeddings,
            report.total_persons,
            report.skipped_images,
        )

        return index, report
