"""Dataset expansion: 302 -> ~1000 protected images.
Usage: .venv312\Scripts\python.exe scripts\expand_dataset.py
"""
from __future__ import annotations
import hashlib, json, logging, os, shutil, sys, tempfile, time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import cv2, numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv; load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("expand_dataset")

PROTECTED_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "parser_finetune_current"
PROTECTED_IMAGES = PROTECTED_DIR / "images"
PROTECTED_CORRECTED = PROTECTED_DIR / "annotation" / "corrected_masks"
RAW_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "raw"
EXPANSION_RAW = PROJECT_ROOT / "dataset_builder" / "dataset" / "expansion_staging"
PRE_EXPANSION_HASHES = PROTECTED_DIR / "pre_expansion_corrected_hashes.json"

TARGET_TOTAL = 1000
CATEGORY_DOWNLOAD_TARGETS = {
    "normal": 200, "eyeglasses": 150, "sunglasses": 120,
    "hijab": 100, "mask": 100, "cap": 80, "beard": 80,
    "helmet": 60, "scarf": 60, "hair_occlusion": 60,
}
MAX_PAGES = 8
PER_PAGE = 20

@dataclass
class ExpansionStats:
    total_protected_images: int = 0
    total_protected_corrected: int = 0
    candidates_downloaded: int = 0
    candidates_failed: int = 0
    dup_existing: int = 0
    dup_among_new: int = 0
    face_rejected: int = 0
    quality_rejected: int = 0
    decode_rejected: int = 0
    new_accepted: int = 0
    new_aligned: int = 0
    new_masked: int = 0
    final_total: int = 0
    per_source: dict = field(default_factory=dict)
    per_category: dict = field(default_factory=dict)
    pages_searched: int = 0

def create_sources():
    from dataset_builder.config.settings import Settings
    from dataset_builder.sources.pexels import PexelsSource
    from dataset_builder.sources.pixabay import PixabaySource
    from dataset_builder.sources.openverse import OpenverseSource
    from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource
    settings = Settings()
    registry = {
        "pexels": PexelsSource, "pixabay": PixabaySource,
        "openverse": OpenverseSource, "wikimedia_commons": WikimediaCommonsSource,
    }
    sources = []
    for name in settings.ENABLED_SOURCES:
        cls = registry.get(name)
        if cls is None: continue
        try:
            s = cls(settings); s.validate_configuration()
            sources.append(s); logger.info("  Source: %s", name)
        except (ValueError, Exception) as e:
            logger.warning("  Source disabled: %s -- %s", name, e)
    return sources, settings

def compute_phash(img_path, hash_size=16):
    try:
        import imagehash; from PIL import Image
        with Image.open(img_path) as img:
            return imagehash.phash(img, hash_size=hash_size)
    except Exception: return None

class DuplicateChecker:
    def __init__(self):
        self.hashes = []; self.threshold = 5
    def load_existing(self):
        logger.info("Loading existing image hashes for dedup...")
        count = 0
        for f in PROTECTED_IMAGES.iterdir():
            if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                h = compute_phash(f)
                if h is not None: self.hashes.append((f, h)); count += 1
        if RAW_DIR.exists():
            for cat_dir in RAW_DIR.iterdir():
                if cat_dir.is_dir():
                    for f in cat_dir.iterdir():
                        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                            h = compute_phash(f)
                            if h is not None: self.hashes.append((f, h)); count += 1
        logger.info("  Loaded %d hashes", count)
    def is_duplicate(self, img_path):
        h = compute_phash(img_path)
        if h is None: return True
        for _, eh in self.hashes:
            if h - eh <= self.threshold: return True
        return False
    def add(self, img_path):
        h = compute_phash(img_path)
        if h is not None: self.hashes.append((img_path, h))

_detector_singleton = None
def get_detector():
    global _detector_singleton
    if _detector_singleton is None:
        from pipeline.detector import FaceDetector
        _detector_singleton = FaceDetector()
    return _detector_singleton

def validate_face(img_path):
    try:
        img = cv2.imread(str(img_path))
        if img is None: return False, "decode_failed"
        h, w = img.shape[:2]
        if w < 100 or h < 100: return False, "image_too_small"
        detector = get_detector()
        faces = detector.detect(img)
        if not faces: return False, "no_face"
        if len(faces) > 1: return False, "multiple_faces"
        face = faces[0]; bbox = face.bbox
        if bbox is not None:
            fa = float((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
            ia = float(h*w)
            if fa/ia < 0.02: return False, "face_too_small"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur < 60.0: return False, "blurry"
        bright = float(np.mean(gray))
        if bright < 30 or bright > 230: return False, "bad_brightness"
        return True, "ok"
    except Exception as e: return False, str(e)

def load_queries():
    queries_dir = PROJECT_ROOT / "dataset_builder" / "queries"
    cat_queries = {}
    for cat in CATEGORY_DOWNLOAD_TARGETS:
        qfile = queries_dir / f"{cat}.txt"
        if qfile.exists():
            lines = qfile.read_text(encoding="utf-8").splitlines()
            qs = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
            cat_queries[cat] = qs
    return cat_queries

def download_candidates(sources, dup_checker, stats):
    """Download new image candidates from all sources."""
    EXPANSION_RAW.mkdir(parents=True, exist_ok=True)
    cat_queries = load_queries()
    seen_source_ids = set()
    for cat, target in CATEGORY_DOWNLOAD_TARGETS.items():
        cat_dir = EXPANSION_RAW / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        existing_count = len([f for f in cat_dir.iterdir() if f.is_file()]) if cat_dir.exists() else 0
        if existing_count >= target:
            logger.info("%s: already have %d candidates, skipping", cat, existing_count)
            stats.per_category[cat] = existing_count
            continue
        queries = cat_queries.get(cat, ["portrait face"])
        cat_downloaded = existing_count
        for qi, query in enumerate(queries):
            if cat_downloaded >= target: break
            for source in sources:
                if cat_downloaded >= target: break
                for page in range(1, MAX_PAGES + 1):
                    if cat_downloaded >= target: break
                    stats.pages_searched += 1
                    try: results = source.search(query=query, page=page, per_page=PER_PAGE)
                    except Exception: break
                    if not results: break
                    for sr in results:
                        if cat_downloaded >= target: break
                        source_key = f"{source.name}::{sr.id}"
                        if source_key in seen_source_ids: continue
                        seen_source_ids.add(source_key)
                        with tempfile.TemporaryDirectory() as tmp:
                            try: dl = source.download(sr, Path(tmp))
                            except Exception: stats.candidates_failed += 1; continue
                            if not dl.success or dl.local_path is None:
                                stats.candidates_failed += 1; continue
                            local = dl.local_path
                            if dup_checker.is_duplicate(local):
                                stats.dup_existing += 1; continue
                            ext = local.suffix or ".jpg"
                            dest = cat_dir / f"{source.name}_{sr.id}{ext}"
                            counter = 1
                            while dest.exists():
                                dest = cat_dir / f"{source.name}_{sr.id}_{counter}{ext}"; counter += 1
                            shutil.copy2(str(local), str(dest))
                            dup_checker.add(dest)
                            stats.candidates_downloaded += 1
                            stats.per_source[source.name] = stats.per_source.get(source.name, 0) + 1
                            cat_downloaded += 1
                        time.sleep(0.3)
        stats.per_category[cat] = cat_downloaded
        logger.info("  %s: downloaded %d candidates", cat, cat_downloaded)
    return stats

def filter_candidates(dup_checker, stats):
    """Filter downloaded candidates: dedup + face validation."""
    logger.info("Filtering downloaded candidates...")
    for cat_dir in EXPANSION_RAW.iterdir():
        if not cat_dir.is_dir(): continue
        cat = cat_dir.name
        for f in list(cat_dir.iterdir()):
            if not f.is_file(): continue
            ok, reason = validate_face(f)
            if not ok:
                if reason in ("decode_failed", "image_too_small"):
                    stats.decode_rejected += 1
                elif reason in ("no_face", "multiple_faces", "face_too_small"):
                    stats.face_rejected += 1
                elif reason in ("blurry", "bad_brightness"):
                    stats.quality_rejected += 1
                else:
                    stats.face_rejected += 1
                f.unlink(missing_ok=True)
                continue
            stats.new_accepted += 1
    logger.info("  Accepted: %d, Face rejected: %d, Quality rejected: %d, Decode rejected: %d",
                stats.new_accepted, stats.face_rejected, stats.quality_rejected, stats.decode_rejected)
    return stats

def process_images(stats):
    """Run face alignment + mask generation on accepted candidates."""
    logger.info("Running face alignment + mask generation...")
    from dataset_builder.dataset.parser_finetune.face_pipeline import align_face
    from dataset_builder.dataset.parser_finetune.mask_generation import generate_mask, save_mask
    from dataset_builder.dataset.parser_finetune.metadata import SampleMetadata, save_sample_metadata, save_class_mapping

    existing_images = sorted(PROTECTED_IMAGES.glob("*.png"))
    next_id = max(int(f.stem.split("_")[-1]) for f in existing_images) + 1 if existing_images else 0

    new_samples = []
    for cat_dir in sorted(EXPANSION_RAW.iterdir()):
        if not cat_dir.is_dir(): continue
        cat = cat_dir.name
        for f in sorted(cat_dir.iterdir()):
            if not f.is_file(): continue
            sample_id = f"sample_{next_id:04d}"
            alignment = align_face(f, sample_id, cat)
            if alignment is None:
                logger.debug("Alignment failed: %s", f.name)
                continue
            aligned_path = PROTECTED_IMAGES / f"{sample_id}.png"
            cv2.imwrite(str(aligned_path), alignment.aligned_image)
            stats.new_aligned += 1

            mask = generate_mask(alignment.aligned_image)
            if mask is None:
                logger.debug("Mask failed: %s", f.name)
                continue
            from dataset_builder.dataset.parser_finetune.config import PilotConfig
            cfg = PilotConfig()
            initial_mask_path = cfg.INITIAL_MASKS_DIR / f"{sample_id}.png"
            initial_mask_path.parent.mkdir(parents=True, exist_ok=True)
            save_mask(mask, initial_mask_path)
            gt_mask_path = PROTECTED_DIR / "masks" / f"{sample_id}.png"
            gt_mask_path.parent.mkdir(parents=True, exist_ok=True)
            save_mask(mask, gt_mask_path)
            stats.new_masked += 1

            metadata = SampleMetadata(
                sample_id=sample_id,
                source_image=str(f),
                source_category=cat,
                aligned_image=str(aligned_path),
                initial_mask=str(initial_mask_path),
                ground_truth_mask=str(gt_mask_path),
                face_bbox=alignment.face_bbox,
                face_kps=alignment.face_kps,
                image_width=alignment.aligned_image.shape[1],
                image_height=alignment.aligned_image.shape[0],
                detection_score=alignment.detection_score,
                face_area_ratio=alignment.face_area_ratio,
                selection_reason="expansion_download",
                quality_status="annotation_pending",
            )
            meta_dir = PROTECTED_DIR / "metadata"
            meta_dir.mkdir(parents=True, exist_ok=True)
            save_sample_metadata(metadata, meta_dir)

            new_samples.append({
                "sample_id": sample_id,
                "source_category": cat,
                "source_filename": f.name,
                "aligned_image": str(aligned_path),
                "initial_mask": str(initial_mask_path),
                "ground_truth_mask": str(gt_mask_path),
                "annotation_status": "annotation_pending",
                "ground_truth_status": "initial_model_mask_pending_correction",
            })
            next_id += 1

            if stats.new_masked % 25 == 0:
                logger.info("  Processed %d images", stats.new_masked)

    return new_samples

def update_manifest(new_samples):
    """Update annotation_manifest.json with new samples."""
    manifest_path = PROTECTED_DIR / "annotation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"version": "current_v1", "annotation_status": "annotation_pending",
                     "total_samples": 0, "splits": {"train": 0, "val": 0, "test": 0},
                     "category_distribution": {}, "samples": []}
    for s in new_samples:
        manifest["samples"].append(s)
        cat = s["source_category"]
        manifest["category_distribution"][cat] = manifest["category_distribution"].get(cat, 0) + 1
    manifest["total_samples"] = len(manifest["samples"])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Updated manifest: %d total samples", manifest["total_samples"])
    return manifest

def generate_splits(new_samples):
    """Assign new samples to train/val/test (80/10/10)."""
    import random; random.seed(42)
    new_ids = [s["sample_id"] for s in new_samples]
    cats = [s["source_category"] for s in new_samples]
    train_ids, val_ids, test_ids = [], [], []
    for cat in set(cats):
        cat_ids = [sid for sid, c in zip(new_ids, cats) if c == cat]
        random.shuffle(cat_ids)
        n = len(cat_ids)
        n_train = int(n * 0.8); n_val = int(n * 0.1)
        train_ids.extend(cat_ids[:n_train])
        val_ids.extend(cat_ids[n_train:n_train+n_val])
        test_ids.extend(cat_ids[n_train+n_val:])
    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    splits_dir = PROTECTED_DIR / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        p = splits_dir / f"{name}.txt"
        existing = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
        all_ids = list(dict.fromkeys(existing + ids))
        p.write_text("\n".join(all_ids) + "\n", encoding="utf-8")
        logger.info("  Split %s: +%d -> %d total", name, len(ids), len(all_ids))
    return splits

def verify_protection():
    """Verify all protected data is untouched."""
    logger.info("Verifying protected data integrity...")
    pre_hashes = json.loads(PRE_EXPANSION_HASHES.read_text(encoding="utf-8"))
    post_hashes = {}
    for f in PROTECTED_CORRECTED.iterdir():
        if f.suffix.lower() == ".png":
            post_hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    mismatches = []
    for name, h in pre_hashes.items():
        if name not in post_hashes:
            mismatches.append(f"MISSING: {name}")
        elif post_hashes[name] != h:
            mismatches.append(f"CHANGED: {name}")
    if mismatches:
        logger.error("PROTECTION VIOLATION: %s", mismatches)
        return False
    logger.info("  All %d corrected masks verified unchanged", len(pre_hashes))
    post_images = sorted(PROTECTED_IMAGES.glob("*.png"))
    logger.info("  Protected images: %d", len(post_images))
    logger.info("  Protected corrected masks: %d", len(post_hashes))
    return True

def generate_report(stats):
    """Generate expansion report."""
    report_dir = PROTECTED_DIR / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "expansion_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protected_dataset": {
            "images": stats.total_protected_images,
            "corrected_masks": stats.total_protected_corrected,
        },
        "collection": {
            "candidates_downloaded": stats.candidates_downloaded,
            "candidates_failed": stats.candidates_failed,
            "per_source": stats.per_source,
            "pages_searched": stats.pages_searched,
        },
        "filtering": {
            "duplicates_against_existing": stats.dup_existing,
            "face_rejected": stats.face_rejected,
            "quality_rejected": stats.quality_rejected,
            "decode_rejected": stats.decode_rejected,
        },
        "processing": {
            "new_images_accepted": stats.new_accepted,
            "new_images_aligned": stats.new_aligned,
            "new_images_masked": stats.new_masked,
        },
        "final_total": stats.final_total,
        "per_category": stats.per_category,
        "limitation": "Person identity metadata unavailable. Person-disjoint splitting not guaranteed.",
    }
    report_path = report_dir / "expansion_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Expansion report: %s", report_path)
    return report

def main():
    logger.info("=" * 70)
    logger.info("DATASET EXPANSION: 302 -> ~1000")
    logger.info("=" * 70)
    stats = ExpansionStats()

    stats.total_protected_images = len(list(PROTECTED_IMAGES.glob("*.png")))
    stats.total_protected_corrected = len(list(PROTECTED_CORRECTED.glob("*.png")))
    logger.info("Protected: %d images, %d corrected masks",
                stats.total_protected_images, stats.total_protected_corrected)

    dup_checker = DuplicateChecker()
    dup_checker.load_existing()

    sources, settings = create_sources()
    if not sources:
        logger.error("No valid sources available. Check API keys.")
        sys.exit(1)

    download_candidates(sources, dup_checker, stats)
    filter_candidates(dup_checker, stats)

    new_samples = process_images(stats)
    if not new_samples:
        logger.error("No new images processed.")
        sys.exit(1)

    update_manifest(new_samples)
    generate_splits(new_samples)

    stats.final_total = stats.total_protected_images + stats.new_masked
    logger.info("Final total: %d (protected: %d + new: %d)",
                stats.final_total, stats.total_protected_images, stats.new_masked)

    if not verify_protection():
        sys.exit(1)

    generate_report(stats)

    logger.info("=" * 70)
    logger.info("EXPANSION COMPLETE")
    logger.info("  Existing protected: %d", stats.total_protected_images)
    logger.info("  New valid images: %d", stats.new_masked)
    logger.info("  Final total: %d", stats.final_total)
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
