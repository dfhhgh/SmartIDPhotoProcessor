"""
Replacement Collection for parser_finetune_current.

Downloads replacement images to restore dataset to ~1000 unique samples.
Uses online pHash dedup gate against ALL existing images + raw dataset.

Usage:
  python scripts/collect_replacements.py
"""
import hashlib
import json
import logging
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune_current"
IMAGES_DIR = CURRENT_DIR / "images"
INITIAL_MASKS_DIR = CURRENT_DIR / "initial_masks"
MASKS_DIR = CURRENT_DIR / "masks"
METADATA_DIR = CURRENT_DIR / "metadata"
ANNOTATION_DIR = CURRENT_DIR / "annotation"
CORRECTED_MASKS_DIR = ANNOTATION_DIR / "corrected_masks"
SPLITS_DIR = CURRENT_DIR / "splits"
MANIFEST_PATH = CURRENT_DIR / "annotation_manifest.json"
RAW_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "raw"
STAGING_DIR = CURRENT_DIR / "tmp_download"
REPORTS_DIR = CURRENT_DIR / "reports"
BASELINES_PATH = CURRENT_DIR / "dedup_baselines.json"

HASH_SIZE = 16
THRESHOLD = 5
PROTECTED_COUNT = 302
TARGET_TOTAL = 1050
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class OnlineDedupIndex:
    def __init__(self, hash_size=HASH_SIZE, threshold=THRESHOLD):
        self.hash_size = hash_size
        self.threshold = threshold
        self.hashes = []

    def compute_hash(self, img_path):
        try:
            img = Image.open(img_path)
            return imagehash.phash(img, hash_size=self.hash_size)
        except Exception:
            return None

    def load_from_directory(self, directory, glob_pattern="*.png"):
        count = 0
        for f in directory.glob(glob_pattern):
            h = self.compute_hash(f)
            if h is not None:
                self.hashes.append((f, h))
                count += 1
        return count

    def is_duplicate(self, img_path):
        h = self.compute_hash(img_path)
        if h is None:
            return True
        for _, existing_hash in self.hashes:
            if h - existing_hash <= self.threshold:
                return True
        return False

    def add(self, img_path):
        h = self.compute_hash(img_path)
        if h is not None:
            self.hashes.append((img_path, h))
            return True
        return False

    def __len__(self):
        return len(self.hashes)


def verify_baselines():
    if not BASELINES_PATH.exists():
        log.warning("No baselines found, skipping verification")
        return True
    baselines = json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    violations = []
    for name, expected_hash in baselines.get("protected_images", {}).items():
        fp = IMAGES_DIR / name
        if not fp.exists():
            violations.append("MISSING image: " + name)
        elif hashlib.sha256(fp.read_bytes()).hexdigest() != expected_hash:
            violations.append("CHANGED image: " + name)
    for name, expected_hash in baselines.get("corrected_masks", {}).items():
        fp = CORRECTED_MASKS_DIR / name
        if not fp.exists():
            violations.append("MISSING corrected mask: " + name)
        elif hashlib.sha256(fp.read_bytes()).hexdigest() != expected_hash:
            violations.append("CHANGED corrected mask: " + name)
    for name, expected_hash in baselines.get("protected_initial_masks", {}).items():
        fp = INITIAL_MASKS_DIR / name
        if not fp.exists():
            violations.append("MISSING initial mask: " + name)
        elif hashlib.sha256(fp.read_bytes()).hexdigest() != expected_hash:
            violations.append("CHANGED initial mask: " + name)
    for name, expected_hash in baselines.get("protected_metadata", {}).items():
        fp = METADATA_DIR / name
        if not fp.exists():
            violations.append("MISSING metadata: " + name)
        elif hashlib.sha256(fp.read_bytes()).hexdigest() != expected_hash:
            violations.append("CHANGED metadata: " + name)
    if violations:
        log.error("BASELINE VERIFICATION FAILED:")
        for v in violations:
            log.error("  %s", v)
        return False
    log.info("Baseline verification: PASS (%d protected images)", len(baselines.get("protected_images", {})))
    return True


_detector = None


def get_detector():
    global _detector
    if _detector is None:
        sys.path.insert(0, str(PROJECT_ROOT))
        from pipeline.detector import FaceDetector
        _detector = FaceDetector()
    return _detector


def validate_face(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return False, "decode_failed"
        h, w = img.shape[:2]
        if h < 100 or w < 100:
            return False, "too_small"
        detector = get_detector()
        faces = detector.detect(img)
        if not faces or len(faces) == 0:
            return False, "no_face"
        if len(faces) > 1:
            return False, "multiple_faces"
        face = faces[0]
        bbox = face.bbox
        area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (h * w)
        if area_ratio < 0.02:
            return False, "face_too_small"
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if blur < 60.0:
            return False, "blurry"
        brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        if brightness < 30 or brightness > 230:
            return False, "bad_brightness"
        return True, "ok"
    except Exception as e:
        return False, "error: " + str(e)


def create_sources():
    sys.path.insert(0, str(PROJECT_ROOT))
    from dataset_builder.config.settings import Settings
    settings = Settings()
    sources = []
    for SourceClsName in ["PexelsSource", "PixabaySource", "OpenverseSource", "WikimediaCommonsSource"]:
        try:
            if SourceClsName == "PexelsSource":
                from dataset_builder.sources.pexels import PexelsSource
                s = PexelsSource(settings)
            elif SourceClsName == "PixabaySource":
                from dataset_builder.sources.pixabay import PixabaySource
                s = PixabaySource(settings)
            elif SourceClsName == "OpenverseSource":
                from dataset_builder.sources.openverse import OpenverseSource
                s = OpenverseSource(settings)
            elif SourceClsName == "WikimediaCommonsSource":
                from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource
                s = WikimediaCommonsSource(settings)
            s.validate_configuration()
            sources.append(s)
            log.info("Source %s: available", s.name)
        except Exception as e:
            log.info("Source %s: not available (%s)", SourceClsName, e)
    return sources, settings


def load_queries():
    queries_dir = PROJECT_ROOT / "dataset_builder" / "queries"
    category_queries = {}
    for qf in queries_dir.glob("*.txt"):
        if qf.stem.startswith("__"):
            continue
        lines = [l.strip() for l in qf.read_text().splitlines() if l.strip() and not l.startswith("#")]
        if lines:
            category_queries[qf.stem] = lines
    return category_queries


def main():
    if not verify_baselines():
        sys.exit(1)

    random.seed(SEED)

    log.info("Building online dedup index...")
    dedup = OnlineDedupIndex()
    n1 = dedup.load_from_directory(IMAGES_DIR)
    log.info("Indexed %d current images", n1)
    n2 = 0
    for cat_dir in RAW_DIR.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.iterdir():
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    h = dedup.compute_hash(f)
                    if h is not None:
                        dedup.hashes.append((f, h))
                        n2 += 1
    log.info("Indexed %d raw dataset images (read-only)", n2)
    log.info("Total index: %d images", len(dedup))

    existing_ids = [s.stem for s in IMAGES_DIR.glob("*.png")]
    max_id = max(int(s.split("_")[1]) for s in existing_ids) if existing_ids else -1
    next_id = max_id + 1

    category_queries = load_queries()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_cats = {}
    for s in manifest["samples"]:
        cat = s.get("source_category", "unknown")
        current_cats[cat] = current_cats.get(cat, 0) + 1
    total_current = len(manifest["samples"])
    total_needed = max(0, TARGET_TOTAL - total_current)
    log.info("Current: %d, Target: %d, Need: %d", total_current, TARGET_TOTAL, total_needed)

    targets = {}
    for cat in category_queries:
        current = current_cats.get(cat, 0)
        deficit = max(0, TARGET_TOTAL // len(category_queries) - current)
        targets[cat] = deficit
    total_targets = sum(targets.values())
    if total_targets > 0 and total_targets != total_needed:
        scale = total_needed / total_targets
        for cat in targets:
            targets[cat] = max(0, int(targets[cat] * scale))
    # Ensure at least 5 per category that needs any
    for cat in targets:
        if targets[cat] > 0 and targets[cat] < 5:
            targets[cat] = 5
    log.info("Targets: %s (total: %d)", targets, sum(targets.values()))

    sources, settings = create_sources()
    if not sources:
        log.error("No sources available!")
        sys.exit(1)

    STAGING_DIR.mkdir(exist_ok=True)

    sys.path.insert(0, str(PROJECT_ROOT))
    from dataset_builder.dataset.parser_finetune.face_pipeline import align_face
    from dataset_builder.dataset.parser_finetune.mask_generation import generate_mask, save_mask
    from dataset_builder.dataset.parser_finetune.config import PilotConfig
    cfg = PilotConfig()

    accepted = 0
    downloaded = 0
    dup_rejected = 0
    face_rejected = 0
    quality_rejected = 0
    per_source = {}
    per_category = {}
    t_start = time.time()

    for cat, target in targets.items():
        if target <= 0:
            continue
        if cat not in category_queries:
            continue
        log.info("=== %s (target: %d) ===", cat, target)
        cat_accepted = 0
        queries = category_queries[cat]
        random.shuffle(queries)

        for query in queries:
            if cat_accepted >= target:
                break
            for source in sources:
                if cat_accepted >= target:
                    break
                page = 1
                max_pages = 8
                while cat_accepted < target and page <= max_pages:
                    try:
                        results = source.search(query, page=page, per_page=15)
                    except Exception as e:
                        log.warning("Search error (%s, %s, p%d): %s", source.name, query, page, e)
                        break
                    if not results:
                        break
                    for result in results:
                        if cat_accepted >= target:
                            break
                        downloaded += 1
                        try:
                            dl = source.download(result, STAGING_DIR)
                        except Exception:
                            continue
                        if not dl.success or dl.local_path is None:
                            continue
                        local_path = dl.local_path
                        if not local_path.exists():
                            continue
                        if dedup.is_duplicate(local_path):
                            dup_rejected += 1
                            local_path.unlink(missing_ok=True)
                            continue
                        ok, reason = validate_face(local_path)
                        if not ok:
                            if "face" in reason or "no_face" in reason or "multiple" in reason:
                                face_rejected += 1
                            else:
                                quality_rejected += 1
                            local_path.unlink(missing_ok=True)
                            continue
                        sample_id = "sample_%04d" % next_id
                        next_id += 1
                        cat_accepted += 1
                        accepted += 1
                        per_source[source.name] = per_source.get(source.name, 0) + 1
                        per_category[cat] = per_category.get(cat, 0) + 1
                        try:
                            aligned = align_face(local_path, sample_id, cat, cfg)
                            if aligned is None:
                                cat_accepted -= 1
                                accepted -= 1
                                local_path.unlink(missing_ok=True)
                                continue
                            aligned_path = IMAGES_DIR / (sample_id + ".png")
                            cv2.imwrite(str(aligned_path), aligned.aligned_image)
                            mask = generate_mask(aligned.aligned_image)
                            if mask is not None:
                                save_mask(mask, INITIAL_MASKS_DIR / (sample_id + ".png"))
                                save_mask(mask, MASKS_DIR / (sample_id + ".png"))
                            meta = {
                                "sample_id": sample_id,
                                "source_category": cat,
                                "source_filename": local_path.name,
                                "source": source.name,
                                "annotation_status": "annotation_pending",
                            }
                            (METADATA_DIR / (sample_id + ".json")).write_text(
                                json.dumps(meta, indent=2), encoding="utf-8"
                            )
                            dedup.add(aligned_path)
                        except Exception as e:
                            log.warning("Processing error for %s: %s", sample_id, e)
                            cat_accepted -= 1
                            accepted -= 1
                            (IMAGES_DIR / (sample_id + ".png")).unlink(missing_ok=True)
                            continue
                        local_path.unlink(missing_ok=True)
                    page += 1
                if cat_accepted >= target:
                    break
        log.info("  %s: %d/%d accepted (%.0fs)", cat, cat_accepted, target, time.time() - t_start)

    if STAGING_DIR.exists():
        shutil.rmtree(str(STAGING_DIR), ignore_errors=True)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    existing_ids_in_manifest = set(s["sample_id"] for s in manifest["samples"])
    new_ids = []
    for sid in sorted(p.stem for p in IMAGES_DIR.glob("*.png")):
        if sid not in existing_ids_in_manifest:
            mf = METADATA_DIR / (sid + ".json")
            if mf.exists():
                m = json.loads(mf.read_text())
                new_sample = {
                    "sample_id": sid,
                    "source_category": m.get("source_category", "unknown"),
                    "aligned_image": str((IMAGES_DIR / (sid + ".png")).resolve()),
                    "initial_mask": str((INITIAL_MASKS_DIR / (sid + ".png")).resolve()),
                    "ground_truth_mask": str((MASKS_DIR / (sid + ".png")).resolve()),
                    "split": "train",
                    "annotation_status": "annotation_pending",
                }
                manifest["samples"].append(new_sample)
                new_ids.append(sid)
    manifest["total_samples"] = len(manifest["samples"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for sid in new_ids:
        r = random.random()
        if r < 0.8:
            split_file = "train.txt"
        elif r < 0.9:
            split_file = "val.txt"
        else:
            split_file = "test.txt"
        fp = SPLITS_DIR / split_file
        lines = [l.strip() for l in fp.read_text().splitlines() if l.strip()]
        lines.append(sid)
        fp.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "accepted": accepted,
        "downloaded": downloaded,
        "dup_rejected": dup_rejected,
        "face_rejected": face_rejected,
        "quality_rejected": quality_rejected,
        "per_source": per_source,
        "per_category": per_category,
        "final_total": len(list(IMAGES_DIR.glob("*.png"))),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "replacement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    violations = verify_baselines()
    if not violations:
        log.error("POST-REPLACEMENT VERIFICATION FAILED")
        sys.exit(1)

    log.info("Replacement complete: %d accepted, %d downloaded, %d dup-rejected", accepted, downloaded, dup_rejected)
    log.info("Final dataset: %d images", report["final_total"])
    return True


if __name__ == "__main__":
    main()
