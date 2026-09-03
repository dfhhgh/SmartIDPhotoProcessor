"""STEP 5 & 6: Investigate hard-negative data integrity."""
import json
import hashlib
import os
from pathlib import Path


def main():
    v4 = Path("datasets/celebrity-v4")

    # 1. Check for cross-identity file copies (same filename across identities)
    print("=" * 60)
    print("CHECK 1: Cross-identity filename collisions")
    print("=" * 60)
    filename_map = {}
    for root, dirs, files in os.walk(v4 / "reference"):
        for f in files:
            if f.endswith(".jpg"):
                fp = os.path.join(root, f)
                identity = Path(root).name
                if f not in filename_map:
                    filename_map[f] = []
                filename_map[f].append((identity, fp))

    collisions = {k: v for k, v in filename_map.items() if len(v) > 1}
    if collisions:
        print("FOUND %d filename collisions:" % len(collisions))
        for fname, identities in list(collisions.items())[:10]:
            print("  %s: %s" % (fname, [i[0] for i in identities]))
    else:
        print("No filename collisions found.")

    # 2. Check for content duplicates across identities
    print("\n" + "=" * 60)
    print("CHECK 2: Content duplicates across identities")
    print("=" * 60)
    hash_map = {}
    for root, dirs, files in os.walk(v4 / "reference"):
        for f in files:
            if f.endswith(".jpg"):
                fp = os.path.join(root, f)
                identity = Path(root).name
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                if h not in hash_map:
                    hash_map[h] = []
                hash_map[h].append((identity, fp))

    content_dups = {k: v for k, v in hash_map.items() if len(v) > 1}
    if content_dups:
        print("FOUND %d content duplicates:" % len(content_dups))
        for h, entries in list(content_dups.items())[:10]:
            print("  %s: %s" % (h[:16], [(e[0], os.path.basename(e[1])) for e in entries]))
    else:
        print("No content duplicates found.")

    # 3. Check for cross-split leakage (same content in ref and query)
    print("\n" + "=" * 60)
    print("CHECK 3: Cross-split content leakage")
    print("=" * 60)
    ref_hashes = {}
    for root, dirs, files in os.walk(v4 / "reference"):
        for f in files:
            if f.endswith(".jpg"):
                fp = os.path.join(root, f)
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                ref_hashes[h] = (Path(root).name, fp)

    cal_hashes = {}
    for root, dirs, files in os.walk(v4 / "calibration"):
        for f in files:
            if f.endswith(".jpg"):
                fp = os.path.join(root, f)
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                cal_hashes[h] = (Path(root).name, fp)

    held_hashes = {}
    for root, dirs, files in os.walk(v4 / "held_out"):
        for f in files:
            if f.endswith(".jpg"):
                fp = os.path.join(root, f)
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                held_hashes[h] = (Path(root).name, fp)

    ref_cal = set(ref_hashes.keys()) & set(cal_hashes.keys())
    ref_held = set(ref_hashes.keys()) & set(held_hashes.keys())
    cal_held = set(cal_hashes.keys()) & set(held_hashes.keys())

    print("ref vs cal leakage: %d" % len(ref_cal))
    print("ref vs held leakage: %d" % len(ref_held))
    print("cal vs held leakage: %d" % len(cal_held))

    # 4. Check Morgan Freeman reference specifically
    print("\n" + "=" * 60)
    print("CHECK 4: Morgan Freeman reference images")
    print("=" * 60)
    mf_dir = v4 / "reference" / "morgan_freeman"
    if mf_dir.exists():
        for fp in sorted(mf_dir.iterdir()):
            if fp.suffix == ".jpg":
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                print("  %s (hash: %s)" % (fp.name, h[:16]))

    # 5. Check Lebron James calibration images
    print("\n" + "=" * 60)
    print("CHECK 5: Lebron James calibration images")
    print("=" * 60)
    lj_dir = v4 / "calibration" / "lebron_james"
    if lj_dir.exists():
        for fp in sorted(lj_dir.iterdir()):
            if fp.suffix == ".jpg":
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                print("  %s (hash: %s)" % (fp.name, h[:16]))

    # 6. Check for duplicate images within the same identity
    print("\n" + "=" * 60)
    print("CHECK 6: Duplicate images within identities")
    print("=" * 60)
    for split_dir in ["reference", "calibration", "held_out"]:
        split_path = v4 / split_dir
        if not split_path.exists():
            continue
        for person_dir in sorted(split_path.iterdir()):
            if not person_dir.is_dir():
                continue
            person_hashes = {}
            for fp in person_dir.glob("*.jpg"):
                with open(fp, "rb") as fh:
                    h = hashlib.sha256(fh.read()).hexdigest()
                if h in person_hashes:
                    print("  %s/%s: duplicate content %s == %s" %
                          (split_dir, person_dir.name, fp.name, person_hashes[h]))
                person_hashes[h] = fp.name

    # 7. Identify the exact v3 source file for Morgan Freeman's reference
    print("\n" + "=" * 60)
    print("CHECK 7: Morgan Freeman v3 source")
    print("=" * 60)
    mf_v3_ref = Path("datasets/celebrity-v3/reference/morgan_freeman")
    if mf_v3_ref.exists():
        for fp in sorted(mf_v3_ref.iterdir()):
            if fp.suffix == ".jpg":
                print("  %s" % fp.name)
                # Check if this matches the v4 reference
                with open(fp, "rb") as fh:
                    v3_hash = hashlib.sha256(fh.read()).hexdigest()
                v4_match = v4 / "reference" / "morgan_freeman" / fp.name
                if v4_match.exists():
                    with open(v4_match, "rb") as fh:
                        v4_hash = hashlib.sha256(fh.read()).hexdigest()
                    print("    v3 hash: %s, v4 hash: %s, match: %s" %
                          (v3_hash[:16], v4_hash[:16], v3_hash == v4_hash))


if __name__ == "__main__":
    main()
