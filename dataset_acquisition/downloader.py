"""Image downloader with FaceService validation, representation filtering, and resume support.

Uses InsightFace/FaceService for face detection instead of OpenCV Haar Cascade.
Handles representation detection, identity uncertainty, and multi-face policy.

Single-Face Acquisition Gate (Phase 13.6.1.2):
  Download → Decode → Representation check → Face detection → Exactly-One-Face Gate → Calibration candidate

Only images with exactly one detected face and no representation keywords are saved.
No-face, multi-face, and representation images are rejected at download time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from dataset_acquisition.models import (
    AcquisitionRunResult,
    ImageRecord,
    DownloadResult,
    Person,
    RejectionDetail,
    ReviewItem,
    SearchResult,
)
from dataset_acquisition.sources.base import ImageSource

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

REPRESENTATION_KEYWORDS = {
    "poster", "painting", "portrait", "illustration", "cartoon", "statue",
    "sculpture", "drawing", "sketch", "artwork", "mural", "graffiti",
    "billboard", "magazine cover", "movie poster", "album cover",
    "film poster", "promotional", "promotional photo", "headshot",
    "stock photo", "stock image", "watermark", "fotor", "canva",
    "wedding photo", "family portrait", "baby photo", "pet photo",
    "clip art", "clipart", "icon", "logo", "badge", "stamp",
}


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Downloader:
    """Orchestrates image download, validation, and storage.

    Face validation policy:
    - Uses InsightFace/FaceService for face detection (aligned with production pipeline)
    - Single face: accepted, face_selected=True
    - Multiple faces: placed in review queue (identity ambiguous), NOT auto-selected
    - No face: rejected
    - Representation (poster/painting): marked and excluded from calibration

    Rejection policy:
    - Only rejects genuinely unusable cases (corrupt, too small, no face)
    - Does NOT reject for glasses, lighting, pose, expression, compression
    """

    def __init__(
        self,
        output_dir: Path,
        sources: list[ImageSource],
        max_images_per_person: int = 15,
        min_image_width: int = 200,
        min_image_height: int = 200,
        delay: float = 1.0,
        face_service: Any | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._sources = {s.name: s for s in sources}
        self._max_images = max_images_per_person
        self._min_width = min_image_width
        self._min_height = min_image_height
        self._delay = delay
        self._face_service = face_service

        self._seen_hashes: set[str] = set()
        self._records: list[ImageRecord] = []
        self._review_queue: list[ReviewItem] = []
        self._state_path = self._output_dir / "download_state.json"

    def _load_state(self) -> dict[str, Any]:
        if self._state_path.exists():
            with open(self._state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "downloaded": {},
            "seen_hashes": [],
            "review_queue": [],
            "records": [],
            "rejected_urls": {},
            "rejection_details": {},
        }

    def _save_state(self, state: dict[str, Any]) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def _validate_image(self, data: bytes) -> tuple[bool, str, int, int]:
        """Validate downloaded image data.

        Returns (is_valid, error_msg, width, height).
        """
        if len(data) < 100:
            return False, "File too small", 0, 0

        try:
            img = Image.open(BytesIO(data))
            img.verify()
        except Exception as exc:
            return False, f"Invalid image: {exc}", 0, 0

        try:
            img = Image.open(BytesIO(data))
            width, height = img.size
        except Exception:
            return False, "Cannot read dimensions", 0, 0

        if width < self._min_width or height < self._min_height:
            return False, f"Too small: {width}x{height}", width, height

        if width * height > 100_000_000:
            return False, f"Too large: {width}x{height}", width, height

        return True, "", width, height

    def _detect_faces_insightface(self, data: bytes) -> tuple[int, float, Any | None]:
        """Detect faces using InsightFace/FaceService.

        Returns (num_faces, best_confidence, best_face_or_none).
        """
        if self._face_service is None:
            return 0, 0.0, None

        try:
            nparr = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return 0, 0.0, None

            model = self._face_service.get_model()
            faces = model.get(img)

            if not faces:
                return 0, 0.0, None

            best_face = max(faces, key=lambda f: getattr(f, "det_score", 0.0))
            best_conf = float(getattr(best_face, "det_score", 0.0))

            return len(faces), best_conf, best_face

        except Exception as exc:
            logger.warning("InsightFace detection failed: %s", exc)
            return 0, 0.0, None

    def _is_representation(self, search_result: SearchResult) -> bool:
        """Heuristic check if image might be a representation rather than photograph."""
        title_lower = (search_result.title or "").lower()
        desc_lower = (search_result.description or "").lower()
        combined = f"{title_lower} {desc_lower}"

        for keyword in REPRESENTATION_KEYWORDS:
            if keyword in combined:
                return True
        return False

    def _validate_and_accept(
        self,
        result: SearchResult,
        image_data: bytes,
        person_id: str,
        source_name: str,
        query: str,
        now_ts: str,
        person_dir: Path,
        downloaded_ids: set[str],
        rejected_urls: set[str],
        new_rejection_details: list[RejectionDetail],
    ) -> ImageRecord | None:
        """Shared Single-Face Acquisition Gate for both download_person() and download_candidates().

        Returns an ImageRecord if accepted, or None if rejected (and appends RejectionDetail).
        """
        # Download error check (image_data is None)
        if image_data is None:
            new_rejection_details.append(RejectionDetail(
                person_id=person_id,
                source=source_name,
                source_url=result.source_url,
                rejection_reason="download_error",
                title=result.title,
                timestamp=now_ts,
            ))
            rejected_urls.add(result.source_url)
            return None

        sha = compute_sha256(image_data)
        if sha in self._seen_hashes:
            new_rejection_details.append(RejectionDetail(
                person_id=person_id,
                source=source_name,
                source_url=result.source_url,
                rejection_reason="duplicate",
                title=result.title,
                timestamp=now_ts,
            ))
            rejected_urls.add(result.source_url)
            logger.debug("Duplicate hash, skipping: %s", sha[:16])
            return None

        is_valid, error, width, height = self._validate_image(image_data)
        if not is_valid:
            new_rejection_details.append(RejectionDetail(
                person_id=person_id,
                source=source_name,
                source_url=result.source_url,
                rejection_reason="invalid_image",
                title=result.title,
                timestamp=now_ts,
            ))
            rejected_urls.add(result.source_url)
            logger.debug("Invalid image: %s", error)
            return None

        is_repr = self._is_representation(result)
        num_faces, face_conf, best_face = self._detect_faces_insightface(image_data)

        # Representation: reject (not a real photograph)
        if is_repr:
            new_rejection_details.append(RejectionDetail(
                person_id=person_id,
                source=source_name,
                source_url=result.source_url,
                rejection_reason="representation",
                title=result.title,
                timestamp=now_ts,
            ))
            rejected_urls.add(result.source_url)
            logger.info("Rejected (representation): %s — %s", result.source_url, result.title)
            return None

        # No face
        if num_faces == 0:
            new_rejection_details.append(RejectionDetail(
                person_id=person_id,
                source=source_name,
                source_url=result.source_url,
                rejection_reason="no_face",
                title=result.title,
                timestamp=now_ts,
            ))
            rejected_urls.add(result.source_url)
            logger.info("Rejected (no_face): %s", result.source_url)
            return None

        # Multiple faces
        if num_faces > 1:
            new_rejection_details.append(RejectionDetail(
                person_id=person_id,
                source=source_name,
                source_url=result.source_url,
                rejection_reason="multi_face",
                title=result.title,
                timestamp=now_ts,
            ))
            rejected_urls.add(result.source_url)
            logger.info("Rejected (multi_face, %d faces): %s", num_faces, result.source_url)
            return None

        # Exactly one face: accept as calibration candidate
        ext = Path(result.image_url).suffix.lower()
        if ext not in SUPPORTED_FORMATS:
            ext = ".jpg"
        image_id = f"{person_id}_{sha[:12]}"
        filename = f"{image_id}{ext}"
        filepath = person_dir / filename
        filepath.write_bytes(image_data)

        record = ImageRecord(
            image_id=image_id,
            person_id=person_id,
            source=source_name,
            source_url=result.source_url,
            local_path=str(filepath),
            license=result.license,
            attribution=result.attribution,
            query=query,
            download_timestamp=now_ts,
            sha256=sha,
            file_size=len(image_data),
            width=width,
            height=height,
            faces_detected=num_faces,
            face_selected=True,
            face_confidence=face_conf,
            image_category="photograph",
            identity_status="confirmed",
            status="valid",
        )

        self._seen_hashes.add(sha)
        downloaded_ids.add(result.source_url)
        logger.info(
            "Accepted: %s (%dx%d, %s, faces=%d, conf=%.3f)",
            filename, width, height, result.license, num_faces, face_conf,
        )
        return record

    def download_person(
        self,
        person: Person,
        state: dict[str, Any] | None = None,
    ) -> tuple[list[ImageRecord], list[RejectionDetail]]:
        """Download images for a single person.

        Single-Face Acquisition Gate:
          Only saves images with exactly one detected face and no representation keywords.
          No-face, multi-face, and representation images are rejected at download time
          and tracked via rejected_urls to avoid re-download on resume.

        Returns (accepted_records, new_rejection_details).
        """
        if state is None:
            state = self._load_state()

        person_dir = self._output_dir / "raw" / person.person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        self._seen_hashes = set(state.get("seen_hashes", []))
        records: list[ImageRecord] = []

        existing_source_urls = set(state.get("downloaded", {}).get(person.person_id, []))
        existing_records_raw = state.get("records", [])
        existing_records = [
            ImageRecord.from_dict(r) for r in existing_records_raw
            if r.get("person_id") == person.person_id
        ]
        if existing_records:
            records.extend(existing_records)
            logger.info(
                "Person %s: loaded %d existing records from state.",
                person.person_id, len(existing_records),
            )

        if len(records) >= self._max_images:
            logger.info("Person %s already has %d images, skipping.", person.person_id, len(records))
            return records, []

        queries = person.search_queries or [person.display_name]
        downloaded_ids = existing_source_urls.copy()

        rejected_urls: set[str] = set(
            state.get("rejected_urls", {}).get(person.person_id, [])
        )

        existing_rejection_details_raw = state.get("rejection_details", {}).get(person.person_id, [])
        all_rejection_details: list[RejectionDetail] = [
            RejectionDetail.from_dict(r) for r in existing_rejection_details_raw
        ]
        new_rejection_details: list[RejectionDetail] = []

        now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        for query in queries:
            if len(records) >= self._max_images:
                break

            for source_name, source in self._sources.items():
                if len(records) >= self._max_images:
                    break

                try:
                    for result in source.search(query, max_results=20):
                        if len(records) >= self._max_images:
                            break

                        if result.source_url in downloaded_ids:
                            continue
                        if result.source_url in rejected_urls:
                            continue

                        image_data = source.download_url(result.image_url)

                        record = self._validate_and_accept(
                            result=result,
                            image_data=image_data,
                            person_id=person.person_id,
                            source_name=source_name,
                            query=query,
                            now_ts=now_ts,
                            person_dir=person_dir,
                            downloaded_ids=downloaded_ids,
                            rejected_urls=rejected_urls,
                            new_rejection_details=new_rejection_details,
                        )
                        if record is not None:
                            records.append(record)
                            self._records.append(record)

                except Exception as exc:
                    logger.error("Error searching %s for '%s': %s", source_name, query, exc)

        # Persist state
        if person.person_id not in state.get("downloaded", {}):
            state.setdefault("downloaded", {})[person.person_id] = []
        new_urls = [r.source_url for r in records if r.source_url not in existing_source_urls]
        state["downloaded"][person.person_id].extend(new_urls)
        state["seen_hashes"] = list(self._seen_hashes)

        state.setdefault("rejected_urls", {})[person.person_id] = list(rejected_urls)

        state.setdefault("rejection_details", {})[person.person_id] = [
            r.to_dict() for r in all_rejection_details + new_rejection_details
        ]

        all_records_dict = state.get("records", [])
        existing_ids = {r.get("image_id") for r in all_records_dict}
        for r in records:
            if r.image_id not in existing_ids:
                all_records_dict.append(r.to_dict())
        state["records"] = all_records_dict

        self._save_state(state)
        return records, new_rejection_details

    def download_candidates(
        self,
        person: Person,
        candidates: Iterator[SearchResult],
        source_name: str = "unknown",
        state: dict[str, Any] | None = None,
        max_candidates: int = 100,
    ) -> AcquisitionRunResult:
        """Process pre-searched candidates through the shared validation gate.

        This is the fair evaluation entry point: all sources must pass through
        the same gate logic as download_person(). The candidates are already
        discovered SearchResult objects from any source.

        Streaming: the candidates iterator is consumed lazily. candidates_discovered
        counts how many the iterator yielded; if the iterator is terminated early
        (e.g. by max_candidates), candidates_discovered equals the number actually
        consumed, and this is documented in the result.

        Returns AcquisitionRunResult with structured metrics.
        """
        if state is None:
            state = self._load_state()

        person_dir = self._output_dir / "raw" / person.person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        self._seen_hashes = set(state.get("seen_hashes", []))
        records: list[ImageRecord] = []

        existing_source_urls = set(state.get("downloaded", {}).get(person.person_id, []))
        existing_records_raw = state.get("records", [])
        existing_records = [
            ImageRecord.from_dict(r) for r in existing_records_raw
            if r.get("person_id") == person.person_id
        ]
        if existing_records:
            records.extend(existing_records)

        downloaded_ids = existing_source_urls.copy()
        rejected_urls: set[str] = set(
            state.get("rejected_urls", {}).get(person.person_id, [])
        )
        existing_rejection_details_raw = state.get("rejection_details", {}).get(person.person_id, [])
        all_rejection_details: list[RejectionDetail] = [
            RejectionDetail.from_dict(r) for r in existing_rejection_details_raw
        ]
        new_rejection_details: list[RejectionDetail] = []

        now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        candidates_discovered = 0
        candidates_examined = 0
        skipped_existing = 0
        skipped_rejected = 0

        source = self._sources.get(source_name, list(self._sources.values())[0] if self._sources else None)

        for result in candidates:
            candidates_discovered += 1

            if len(records) >= self._max_images:
                break
            if candidates_examined >= max_candidates:
                break

            # Skip already-downloaded URLs (not counted as examined)
            if result.source_url in downloaded_ids:
                skipped_existing += 1
                continue
            # Skip already-rejected URLs (not counted as examined)
            if result.source_url in rejected_urls:
                skipped_rejected += 1
                continue

            candidates_examined += 1

            image_data = source.download_url(result.image_url)

            record = self._validate_and_accept(
                result=result,
                image_data=image_data,
                person_id=person.person_id,
                source_name=source_name,
                query="",
                now_ts=now_ts,
                person_dir=person_dir,
                downloaded_ids=downloaded_ids,
                rejected_urls=rejected_urls,
                new_rejection_details=new_rejection_details,
            )
            if record is not None:
                records.append(record)
                self._records.append(record)

        # Persist state
        if person.person_id not in state.get("downloaded", {}):
            state.setdefault("downloaded", {})[person.person_id] = []
        new_urls = [r.source_url for r in records if r.source_url not in existing_source_urls]
        state["downloaded"][person.person_id].extend(new_urls)
        state["seen_hashes"] = list(self._seen_hashes)
        state.setdefault("rejected_urls", {})[person.person_id] = list(rejected_urls)
        state.setdefault("rejection_details", {})[person.person_id] = [
            r.to_dict() for r in all_rejection_details + new_rejection_details
        ]
        all_records_dict = state.get("records", [])
        existing_ids = {r.get("image_id") for r in all_records_dict}
        for r in records:
            if r.image_id not in existing_ids:
                all_records_dict.append(r.to_dict())
        state["records"] = all_records_dict
        self._save_state(state)

        accepted = len(records) - len(existing_records)
        rejected = len(new_rejection_details)

        return AcquisitionRunResult(
            records=records,
            rejection_details=new_rejection_details,
            candidates_discovered=candidates_discovered,
            candidates_examined=candidates_examined,
            candidates_skipped_existing=skipped_existing,
            candidates_skipped_rejected=skipped_rejected,
            accepted=accepted,
            rejected=rejected,
        )

    def get_review_queue(self) -> list[ReviewItem]:
        return list(self._review_queue)
