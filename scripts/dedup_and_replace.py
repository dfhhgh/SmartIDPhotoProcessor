"""
Deduplication + Replacement Collection for parser_finetune_current.

Usage:
  python scripts/dedup_and_replace.py --dry-run
  python scripts/dedup_and_replace.py --quarantine
  python scripts/dedup_and_replace.py --replace
  python scripts/dedup_and_replace.py --verify
"""
import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

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
ANNOTATION_METADATA_DIR = ANNOTATION_DIR / "metadata"
SPLITS_DIR = CURRENT_DIR / "splits"
MANIFEST_PATH = CURRENT_DIR / "annotation_manifest.json"
QUARANTINE_DIR = CURRENT_DIR / "dedup_quarantine"
RAW_DIR = PROJECT_ROOT / "dataset_builder" / "dataset" / "raw"
EXPANSION_STAGING = PROJECT_ROOT / "dataset_builder" / "dataset" / "expansion_staging"
BASELINES_PATH = CURRENT_DIR / "dedup_baselines.json"
REPORTS_DIR = CURRENT_DIR / "reports"

HASH_SIZE = 16
THRESHOLD = 5
PROTECTED_COUNT = 302
TARGET_TOTAL = 1050

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def compute_phash(img_path):
    try:
        img = Image.open(img_path)
        return imagehash.phash(img, hash_size=HASH_SIZE)
    except Exception:
        return None


def compute_sha256(data):
    return hashlib.sha256(data).hexdigest()


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def find_duplicate_groups(hashes, threshold=THRESHOLD):
    items = list(hashes.keys())
    n = len(items)
    uf = UnionFind(n)
    hash_list = [hashes[k] for k in items]
    for i in range(n):
        for j in range(i + 1, n):
            if hash_list[i] - hash_list[j] <= threshold:
                uf.union(i, j)
    groups = {}
    for i in range(n):
        root = uf.find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(items[i])
    return groups


def load_baselines():
    if BASELINES_PATH.exists():
        return json.loads(BASELINES_PATH.read_text(encoding="utf-8"))
    return {}


def verify_baselines(baselines):
    violations = []
    for name, expected_hash in baselines.get("protected_images", {}).items():
        fp = IMAGES_DIR / name
        if not fp.exists():
            violations.append("MISSING image: " + name)
        elif compute_sha256(fp.read_bytes()) != expected_hash:
            violations.append("CHANGED image: " + name)
    for name, expected_hash in baselines.get("corrected_masks", {}).items():
        fp = CORRECTED_MASKS_DIR / name
        if not fp.exists():
            violations.append("MISSING corrected mask: " + name)
        elif compute_sha256(fp.read_bytes()) != expected_hash:
            violations.append("CHANGED corrected mask: " + name)
    for name, expected_hash in baselines.get("protected_initial_masks", {}).items():
        fp = INITIAL_MASKS_DIR / name
        if not fp.exists():
            violations.append("MISSING initial mask: " + name)
        elif compute_sha256(fp.read_bytes()) != expected_hash:
            violations.append("CHANGED initial mask: " + name)
    for name, expected_hash in baselines.get("protected_metadata", {}).items():
        fp = METADATA_DIR / name
        if not fp.exists():
            violations.append("MISSING metadata: " + name)
        elif compute_sha256(fp.read_bytes()) != expected_hash:
            violations.append("CHANGED metadata: " + name)
    return violations


def get_sample_category(sample_id):
    mf = METADATA_DIR / (sample_id + ".json")
    if mf.exists():
        return json.loads(mf.read_text()).get("source_category", "unknown")
    return "unknown"


def analyze_and_classify():
    """Compute hashes, find duplicate groups, classify into quarantine/keep."""
    log.info("Computing pHash for all images...")
    t0 = time.time()
    all_images = sorted(IMAGES_DIR.glob("*.png"))
    hashes = {}
    for f in all_images:
        h = compute_phash(f)
        if h is not None:
            hashes[f.stem] = h
    log.info("Hashed %d images in %.1fs", len(hashes), time.time() - t0)

    log.info("Finding duplicate groups (threshold=%d)...", THRESHOLD)
    groups = find_duplicate_groups(hashes, THRESHOLD)
    dup_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    log.info("Found %d duplicate groups", len(dup_groups))

    protected = set("sample_%04d" % i for i in range(PROTECTED_COUNT))
    quarantine_list = []
    keep_new = []

    for g_members in dup_groups.values():
        g_prot = [s for s in g_members if s in protected]
        g_new = [s for s in g_members if s not in protected]

        if g_prot:
            for s in g_new:
                quarantine_list.append({
                    "sample_id": s,
                    "reason": "duplicate_of_protected",
                    "group_size": len(g_members),
                    "protected_in_group": g_prot,
                    "representative": g_prot[0],
                })
            for s in g_prot:
                keep_new.append(s)
        else:
            best = max(g_new, key=lambda s: (IMAGES_DIR / (s + ".png")).stat().st_size)
            for s in g_new:
                if s != best:
                    quarantine_list.append({
                        "sample_id": s,
                        "reason": "duplicate_new",
                        "group_size": len(g_members),
                        "representative": best,
                    })
            keep_new.append(best)

    unique_new = [s for s in hashes if len(groups[find_group_root(groups, s)]) == 1 and s not in protected]
    keep_new.extend(unique_new)

    protected_all = [s for s in hashes if s in protected]
    return hashes, groups, dup_groups, quarantine_list, keep_new, protected_all


def find_group_root(groups, sample_id):
    for root, members in groups.items():
        if sample_id in members:
            return root
    return None


def generate_dry_run_report():
    """Generate dry run report without modifying anything."""
    log.info("=== DRY RUN MODE ===")
    hashes, groups, dup_groups, quarantine_list, keep_new, protected_all = analyze_and_classify()

    report = {
        "mode": "dry_run",
        "threshold": THRESHOLD,
        "hash_size": HASH_SIZE,
        "total_images": len(hashes),
        "protected_images": len(protected_all),
        "new_images": len(hashes) - len(protected_all),
        "duplicate_groups": len(dup_groups),
        "quarantine_count": len(quarantine_list),
        "keep_count": len(keep_new),
        "protected_count": len(protected_all),
        "estimated_final": len(protected_all) + len(keep_new),
        "estimated_replacements_needed": max(0, TARGET_TOTAL - len(protected_all) - len(keep_new)),
    }

    q_cats = {}
    for q in quarantine_list:
        c = get_sample_category(q["sample_id"])
        q_cats[c] = q_cats.get(c, 0) + 1
    report["quarantine_by_category"] = q_cats

    k_cats = {}
    for s in keep_new:
        c = get_sample_category(s)
        k_cats[c] = k_cats.get(c, 0) + 1
    report["keep_by_category"] = k_cats

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "dedup_dry_run_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = "# Deduplication Dry Run Report\n\n"
    md += "| Metric | Value |\n|---|---|\n"
    md += "| Total images | %d |\n" % report["total_images"]
    md += "| Protected originals | %d |\n" % report["protected_images"]
    md += "| New images | %d |\n" % report["new_images"]
    md += "| Duplicate groups | %d |\n" % report["duplicate_groups"]
    md += "| Quarantine (removable) | %d |\n" % report["quarantine_count"]
    md += "| Keep (retained) | %d |\n" % report["keep_count"]
    md += "| Estimated final | %d |\n" % report["estimated_final"]
    md += "| Replacements needed | ~%d |\n" % report["estimated_replacements_needed"]
    md += "\n## Quarantine by Category\n\n"
    md += "| Category | Removed |\n|---|---|\n"
    for cat, cnt in sorted(q_cats.items(), key=lambda x: -x[1]):
        md += "| %s | %d |\n" % (cat, cnt)
    md += "\n## Keep by Category\n\n"
    md += "| Category | Retained |\n|---|---|\n"
    for cat, cnt in sorted(k_cats.items(), key=lambda x: -x[1]):
        md += "| %s | %d |\n" % (cat, cnt)
    (REPORTS_DIR / "dedup_dry_run_report.md").write_text(md, encoding="utf-8")

    log.info("Dry run report saved to %s", REPORTS_DIR)
    log.info("Quarantine: %d images", len(quarantine_list))
    log.info("Keep: %d images", len(keep_new))
    log.info("Estimated final: %d", report["estimated_final"])
    log.info("Replacements needed: ~%d", report["estimated_replacements_needed"])
    return report


def run_quarantine():
    """Move removable duplicates to quarantine directory."""
    log.info("=== QUARANTINE MODE ===")

    baselines = load_baselines()
    violations = verify_baselines(baselines)
    if violations:
        log.error("PRE-QUARANTINE VERIFICATION FAILED:")
        for v in violations:
            log.error("  %s", v)
        return False
    log.info("Pre-quarantine baselines: PASS (%d protected images verified)", len(baselines.get("protected_images", {})))

    hashes, groups, dup_groups, quarantine_list, keep_new, protected_all = analyze_and_classify()

    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    (QUARANTINE_DIR / "images").mkdir(exist_ok=True)
    (QUARANTINE_DIR / "initial_masks").mkdir(exist_ok=True)
    (QUARANTINE_DIR / "masks").mkdir(exist_ok=True)
    (QUARANTINE_DIR / "metadata").mkdir(exist_ok=True)

    moved = 0
    for q in quarantine_list:
        sid = q["sample_id"]
        for subdir in ["images", "initial_masks", "masks", "metadata"]:
            src = CURRENT_DIR / subdir / (sid + ".png") if subdir != "metadata" else CURRENT_DIR / subdir / (sid + ".json")
            dst = QUARANTINE_DIR / subdir / src.name
            if src.exists() and not dst.exists():
                shutil.move(str(src), str(dst))
                moved += 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    q_ids = set(q["sample_id"] for q in quarantine_list)
    manifest["samples"] = [s for s in manifest["samples"] if s["sample_id"] not in q_ids]
    manifest["total_samples"] = len(manifest["samples"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Update splits
    for split_file in ["train.txt", "val.txt", "test.txt"]:
        fp = SPLITS_DIR / split_file
        if fp.exists():
            lines = [l.strip() for l in fp.read_text().splitlines() if l.strip()]
            remaining = [l for l in lines if l not in q_ids]
            fp.write_text("\n".join(remaining) + "\n", encoding="utf-8")

    quarantine_manifest = {
        "quarantine_count": len(quarantine_list),
        "moved_files": moved,
        "samples": quarantine_list,
    }
    (QUARANTINE_DIR / "dedup_quarantine_manifest.json").write_text(
        json.dumps(quarantine_manifest, indent=2), encoding="utf-8"
    )

    violations = verify_baselines(baselines)
    if violations:
        log.error("POST-QUARANTINE VERIFICATION FAILED:")
        for v in violations:
            log.error("  %s", v)
        return False

    log.info("Quarantine complete: %d samples moved, %d files relocated", len(quarantine_list), moved)
    log.info("Post-quarantine baselines: PASS")
    log.info("Remaining images: %d", len(list(IMAGES_DIR.glob("*.png"))))
    return True


def run_replacement():
    """Download replacement images with online dedup gate."""
    log.info("=== REPLACEMENT MODE ===")

    baselines = load_baselines()
    violations = verify_baselines(baselines)
    if violations:
        log.error("PRE-REPLACEMENT VERIFICATION FAILED:")
        for v in violations:
            log.error("  %s", v)
        return False

    sys.path.insert(0, str(PROJECT_ROOT))
    from dataset_builder.config.settings import Settings
    from dataset_builder.dataset.parser_finetune.face_pipeline import align_face
    from dataset_builder.dataset.parser_finetune.mask_generation import generate_mask, save_mask
    from dataset_builder.dataset.parser_finetune.config import PilotConfig
    from dataset_builder.collection.duplicate_index import DuplicateIndex

    settings = Settings()
    dup_index = DuplicateIndex(settings)

    # Build index from ALL current images + raw + protected
    log.info("Building duplicate index from existing images...")
    existing = sorted(IMAGES_DIR.glob("*.png"))
    for f in existing:
        dup_index.add_image(f)
    log.info("Indexed %d existing images", dup_index.size)

    # Also index raw dataset (read-only)
    raw_count = 0
    for cat_dir in RAW_DIR.iterdir():
        if cat_dir.is_dir():
            for f in cat_dir.glob("*"):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                    dup_index.add_image(f)
                    raw_count += 1
    log.info("Indexed %d raw dataset images (read-only)", raw_count)

    # Find current sample IDs to determine next ID
    current_ids = [s.stem for s in IMAGES_DIR.glob("*.png")]
    max_id = max(int(s.split("_")[1]) for s in current_ids) if current_ids else -1
    next_id = max_id + 1
    log.info("Next available sample ID: sample_%04d", next_id)

    # Load queries
    queries_dir = PROJECT_ROOT / "dataset_builder" / "queries"
    category_queries = {}
    for qf in queries_dir.glob("*.txt"):
        if qf.stem.startswith("__"):
            continue
        lines = [l.strip() for l in qf.read_text().splitlines() if l.strip() and not l.startswith("#")]
        if lines:
            category_queries[qf.stem] = lines

    # Determine replacement targets per category
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current_cats = {}
    for s in manifest["samples"]:
        cat = s.get("source_category", "unknown")
        current_cats[cat] = current_cats.get(cat, 0) + 1
    total_current = len(manifest["samples"])
    total_needed = max(0, TARGET_TOTAL - total_current)
    log.info("Current dataset: %d, Target: %d, Need: %d", total_current, TARGET_TOTAL, total_needed)

    # Proportional allocation: fill underrepresented categories
    ideal_per_cat = TARGET_TOTAL // len(category_queries)
    targets = {}
    for cat in category_queries:
        current = current_cats.get(cat, 0)
        deficit = ideal_per_cat - current
        targets[cat] = max(0, deficit)
    # Scale targets to match total_needed
    total_targets = sum(targets.values())
    if total_targets > 0:
        scale = total_needed / total_targets
        for cat in targets:
            targets[cat] = int(targets[cat] * scale)
    log.info("Replacement targets: %s", targets)

    # Create sources
    from dataset_builder.sources.pexels import PexelsSource
    from dataset_builder.sources.pixabay import PixabaySource
    from dataset_builder.sources.openverse import OpenverseSource
    from dataset_builder.sources.wikimedia_commons import WikimediaCommonsSource

    sources = []
    for SourceCls in [PexelsSource, PixabaySource, OpenverseSource, WikimediaCommonsSource]:
        try:
            s = SourceCls(settings)
            s.validate_configuration()
            sources.append(s)
        except Exception as e:
            log.warning("Source %s not available: %s", SourceCls.__name__, e)
    log.info("Available sources: %s", [s.name for s in sources])

    cfg = PilotConfig()
    accepted = 0
    total_downloaded = 0
    total_face_rejected = 0
    total_quality_rejected = 0
    total_dup_rejected = 0
    per_source = {}
    per_category = {}

    for cat, target in targets.items():
        if target <= 0:
            continue
        if cat not in category_queries:
            continue
        log.info("Downloading replacements for '%s' (target: %d)", cat, target)
        cat_accepted = 0
        queries = category_queries[cat]

        for query in queries:
            if cat_accepted >= target:
                break
            for source in sources:
                if cat_accepted >= target:
                    break
                try:
                    page = 1
                    while cat_accepted < target and page <= 8:
                        results = source.search(query, page=page, per_page=15)
                        if not results:
                            break
                        for result in results:
                            if cat_accepted >= target:
                                break
                            total_downloaded += 1
                            # Dedup check
                            if dup_index.is_duplicate(result.get("url_local") or result.get("path", "")):
                                total_dup_rejected += 1
                                continue
                            # Download to temp
                            tmp_dir = CURRENT_DIR / "tmp_download"
                            tmp_dir.mkdir(exist_ok=True)
                            try:
                                local_path = source.download(result, tmp_dir)
                            except Exception:
                                continue
                            if local_path is None or not local_path.exists():
                                continue
                            # Dedup check on downloaded file
                            if dup_index.is_duplicate(local_path):
                                total_dup_rejected += 1
                                local_path.unlink(missing_ok=True)
                                continue
                            # Face filter
                            try:
                                from pipeline.detector import FaceDetector
                                detector = FaceDetector()
                                import cv2
                                img = cv2.imread(str(local_path))
                                if img is None or img.shape[0] < 100 or img.shape[1] < 100:
                                    total_quality_rejected += 1
                                    local_path.unlink(missing_ok=True)
                                    continue
                                faces = detector.detect(img)
                                if not faces or len(faces) != 1:
                                    total_face_rejected += 1
                                    local_path.unlink(missing_ok=True)
                                    continue
                                face = faces[0]
                                bbox = face.bbox
                                area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / (img.shape[0] * img.shape[1])
                                if area_ratio < 0.02:
                                    total_face_rejected += 1
                                    local_path.unlink(missing_ok=True)
                                    continue
                                import cv2
                                blur = cv2.Laplacian(img, cv2.CV_64F).var()
                                brightness = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
                                if blur < 60.0 or brightness < 30 or brightness > 230:
                                    total_quality_rejected += 1
                                    local_path.unlink(missing_ok=True)
                                    continue
                            except Exception:
                                total_face_rejected += 1
                                local_path.unlink(missing_ok=True)
                                continue

                            # ACCEPT
                            sample_id = "sample_%04d" % next_id
                            next_id += 1
                            cat_accepted += 1
                            accepted += 1
                            per_source[source.name] = per_source.get(source.name, 0) + 1
                            per_category[cat] = per_category.get(cat, 0) + 1

                            # Align
                            try:
                                aligned = align_face(local_path, sample_id, cat, cfg)
                                if aligned is None:
                                    cat_accepted -= 1
                                    accepted -= 1
                                    local_path.unlink(missing_ok=True)
                                    continue
                                import cv2
                                aligned_path = IMAGES_DIR / (sample_id + ".png")
                                cv2.imwrite(str(aligned_path), aligned.aligned_image)
                            except Exception:
                                cat_accepted -= 1
                                accepted -= 1
                                local_path.unlink(missing_ok=True)
                                continue

                            # Generate mask
                            try:
                                mask = generate_mask(aligned.aligned_image)
                                if mask is not None:
                                    save_mask(mask, INITIAL_MASKS_DIR / (sample_id + ".png"))
                                    save_mask(mask, MASKS_DIR / (sample_id + ".png"))
                            except Exception:
                                pass

                            # Metadata
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

                            # Add to hash index immediately
                            dup_index.add_image(aligned_path)
                            local_path.unlink(missing_ok=True)

                            if cat_accepted % 10 == 0:
                                log.info("  %s: %d/%d accepted", cat, cat_accepted, target)
                except Exception as e:
                    log.warning("Error with source %s query '%s': %s", source.name, query, e)
                    continue

        log.info("  %s: done (%d accepted)", cat, cat_accepted)

    # Update manifest
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    new_sample_ids = []
    for sid in sorted(p.stem for p in IMAGES_DIR.glob("*.png")):
        if not any(s["sample_id"] == sid for s in manifest["samples"]):
            mf = METADATA_DIR / (sid + ".json")
            if mf.exists():
                m = json.loads(mf.read_text())
                manifest["samples"].append({
                    "sample_id": sid,
                    "source_category": m.get("source_category", "unknown"),
                    "aligned_image": str(IMAGES_DIR / (sid + ".png")),
                    "initial_mask": str(INITIAL_MASKS_DIR / (sid + ".png")),
                    "ground_truth_mask": str(MASKS_DIR / (sid + ".png")),
                    "split": "train",
                    "annotation_status": "annotation_pending",
                })
                new_sample_ids.append(sid)
    manifest["total_samples"] = len(manifest["samples"])
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Update splits
    for sid in new_sample_ids:
        import random
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

    # Cleanup temp
    tmp_dir = CURRENT_DIR / "tmp_download"
    if tmp_dir.exists():
        shutil.rmtree(str(tmp_dir), ignore_errors=True)

    report = {
        "accepted": accepted,
        "total_downloaded": total_downloaded,
        "dup_rejected": total_dup_rejected,
        "face_rejected": total_face_rejected,
        "quality_rejected": total_quality_rejected,
        "per_source": per_source,
        "per_category": per_category,
    }
    (REPORTS_DIR / "replacement_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    violations = verify_baselines(baselines)
    if violations:
        log.error("POST-REPLACEMENT VERIFICATION FAILED:")
        for v in violations:
            log.error("  %s", v)
        return False

    log.info("Replacement complete: %d accepted, %d downloaded, %d dup-rejected, %d face-rejected, %d quality-rejected",
             accepted, total_downloaded, total_dup_rejected, total_face_rejected, total_quality_rejected)
    log.info("Post-replacement baselines: PASS")
    log.info("Final dataset: %d images", len(list(IMAGES_DIR.glob("*.png"))))
    return True


def run_verify():
    """Full integrity verification."""
    log.info("=== VERIFICATION MODE ===")
    baselines = load_baselines()
    if not baselines:
        log.error("No baselines found at %s", BASELINES_PATH)
        return False

    violations = verify_baselines(baselines)
    if violations:
        log.error("VERIFICATION FAILED:")
        for v in violations:
            log.error("  %s", v)
        return False

    images = list(IMAGES_DIR.glob("*.png"))
    init_masks = list(INITIAL_MASKS_DIR.glob("*.png"))
    masks = list(MASKS_DIR.glob("*.png"))
    corrected = list(CORRECTED_MASKS_DIR.glob("*.png"))

    log.info("Images: %d", len(images))
    log.info("Initial masks: %d", len(init_masks))
    log.info("Ground truth masks: %d", len(masks))
    log.info("Corrected masks: %d", len(corrected))

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    log.info("Manifest samples: %d", manifest["total_samples"])

    for split in ["train.txt", "val.txt", "test.txt"]:
        fp = SPLITS_DIR / split
        if fp.exists():
            lines = [l.strip() for l in fp.read_text().splitlines() if l.strip()]
            log.info("Split %s: %d", split.replace(".txt", ""), len(lines))

    protected = set("sample_%04d" % i for i in range(PROTECTED_COUNT))
    active_ids = set(s.stem for s in images)
    quarantine_ids = set()
    if QUARANTINE_DIR.exists():
        quarantine_ids = set(s.stem for s in (QUARANTINE_DIR / "images").glob("*.png")) if (QUARANTINE_DIR / "images").exists() else set()

    prot_in_dataset = protected & active_ids
    prot_missing = protected - active_ids
    log.info("Protected in dataset: %d", len(prot_in_dataset))
    if prot_missing:
        log.error("MISSING PROTECTED: %s", sorted(prot_missing))
    log.info("Quarantined: %d", len(quarantine_ids))

    has_init = sum(1 for s in active_ids if (INITIAL_MASKS_DIR / (s + ".png")).exists())
    missing_init = len(active_ids) - has_init
    log.info("Active with initial mask: %d/%d", has_init, len(active_ids))
    if missing_init:
        log.warning("Missing initial masks: %d", missing_init)

    return len(violations) == 0


def main():
    parser = argparse.ArgumentParser(description="Dedup + Replacement for parser_finetune_current")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Generate report without modifying")
    group.add_argument("--quarantine", action="store_true", help="Move duplicates to quarantine")
    group.add_argument("--replace", action="store_true", help="Download replacements with online dedup")
    group.add_argument("--verify", action="store_true", help="Verify integrity")
    args = parser.parse_args()

    if args.dry_run:
        generate_dry_run_report()
    elif args.quarantine:
        ok = run_quarantine()
        sys.exit(0 if ok else 1)
    elif args.replace:
        ok = run_replacement()
        sys.exit(0 if ok else 1)
    elif args.verify:
        ok = run_verify()
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
