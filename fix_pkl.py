"""
fix_pkl.py
----------
Fixes the PKL structure and merges all JSON files into one clean database.

The correct structure is:
    {
        "F10": {
            "video_name": "F10.mp4",
            "video_path": "D:\\video-4\\F10.mp4",
            "hashes": [...],
            "frames": [...],
            ...
        },
        "F11": { ... },
        ...
    }

Run: python fix_pkl.py
"""
import pickle
import json
import glob
import os

PKL_FILE      = "merged_phash_database.pkl"
OUTPUT_PKL    = "fixed_phash_database.pkl"
JSON_PATTERNS = ["F*_hashes.json", "SB24*.json"]

flat_metadata_keys = {"video_name", "video_path", "duration_seconds", "duration_human",
                      "resolution", "hash_algorithm", "hash_size", "extraction_mode",
                      "extraction_fps", "total_frames", "processed_at", "processing_time",
                      "cores_used", "frames", "hashes", "labels", "arena_crop",
                      "overlay_masks", "moving_logo_removal"}


def get_video_id(data):
    """Derive a clean video ID from a record dict."""
    name = data.get("video_name") or data.get("video_path") or ""
    base = os.path.splitext(os.path.basename(name))[0]
    return base if base else None


def is_flat_metadata(data):
    """Returns True if a dict looks like flat single-video metadata fields."""
    if not isinstance(data, dict):
        return False
    overlap = set(str(k) for k in data.keys()) & flat_metadata_keys
    return len(overlap) >= 5


def wrap_flat_record(data):
    """Wrap a flat metadata dict into a proper {video_id: record} dict."""
    video_id = get_video_id(data)
    if not video_id:
        video_id = "unknown_video"
    return {video_id: dict(data)}


def load_json(path):
    """Load a JSON file and return as a {video_id: record} dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Check if it's already a multi-record dict
        is_multi = all(isinstance(v, dict) for v in list(data.values())[:3])
        if is_multi and not is_flat_metadata(data):
            return data  # already well-formed

        # Flat single-video record
        if is_flat_metadata(data):
            return wrap_flat_record(data)

        # Generic dict — return as-is
        return data

    elif isinstance(data, list):
        # List of records — index by video_name or position
        result = {}
        for i, record in enumerate(data):
            if isinstance(record, dict):
                vid_id = get_video_id(record) or f"record_{i}"
                result[vid_id] = record
            else:
                result[f"record_{i}"] = record
        return result

    return {}


# ── Step 1: Load and fix PKL ──────────────────────────────────────────
print("=" * 60)
print("  PKL Fixer & Merger")
print("=" * 60)

print(f"\n[1/4] Loading PKL: {PKL_FILE}")
with open(PKL_FILE, "rb") as f:
    raw_pkl = pickle.load(f)
print(f"      Raw keys: {len(raw_pkl)}")

fixed_db = {}

if is_flat_metadata(raw_pkl):
    print(f"      ⚠️  Detected flat metadata structure — wrapping into proper record")
    wrapped = wrap_flat_record(raw_pkl)
    fixed_db.update(wrapped)
    vid_id = list(wrapped.keys())[0]
    print(f"      Wrapped as video_id: '{vid_id}'")
else:
    # Already multi-record — validate each entry
    good, bad = 0, 0
    for k, v in raw_pkl.items():
        if isinstance(v, dict):
            fixed_db[k] = v
            good += 1
        else:
            print(f"      ⚠️  Skipping abnormal entry: key='{k}' type={type(v)}")
            bad += 1
    print(f"      Good records: {good}  |  Skipped: {bad}")

print(f"      PKL records after fix: {len(fixed_db)}")

# ── Step 2: Load JSON files ───────────────────────────────────────────
print(f"\n[2/4] Loading JSON files...")

json_files = []
for pattern in JSON_PATTERNS:
    json_files.extend(glob.glob(pattern))
json_files = sorted(set(json_files))
print(f"      Found {len(json_files)} JSON files")

added, skipped, errors = 0, 0, 0

for path in json_files:
    try:
        records = load_json(path)
        file_added = 0
        for vid_id, record in records.items():
            if vid_id in fixed_db:
                print(f"      ⚠️  Duplicate '{vid_id}' from {path} — keeping existing")
                skipped += 1
            else:
                fixed_db[vid_id] = record
                added += 1
                file_added += 1
        print(f"      ✅ {path}: +{file_added} records")
    except Exception as e:
        print(f"      ❌ ERROR reading {path}: {e}")
        errors += 1

print(f"\n      Added  : {added}")
print(f"      Skipped: {skipped} (duplicates)")
print(f"      Errors : {errors}")

# ── Step 3: Validate final DB ─────────────────────────────────────────
print(f"\n[3/4] Validating final database...")

no_hashes, no_video_name = [], []
total_frames = 0

for vid_id, record in fixed_db.items():
    if not isinstance(record, dict):
        continue
    if not record.get("hashes"):
        no_hashes.append(vid_id)
    if not record.get("video_name") and not record.get("video_path"):
        no_video_name.append(vid_id)
    h = record.get("hashes") or []
    total_frames += len(h)

print(f"      Total video records : {len(fixed_db)}")
print(f"      Total frame hashes  : {total_frames}")
if no_hashes:
    print(f"      ⚠️  Records with no hashes ({len(no_hashes)}): {no_hashes[:5]}")
if no_video_name:
    print(f"      ⚠️  Records with no video name ({len(no_video_name)}): {no_video_name[:5]}")

# ── Step 4: Save ──────────────────────────────────────────────────────
print(f"\n[4/4] Saving fixed database...")
with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(fixed_db, f, protocol=pickle.HIGHEST_PROTOCOL)

size_mb = os.path.getsize(OUTPUT_PKL) / (1024 * 1024)
print(f"\n✅  Done! Saved: {OUTPUT_PKL}  ({size_mb:.1f} MB)")
print(f"\n   Next step: update rebuild_vptree.py to use")
print(f"   INPUT_PKL = \"{OUTPUT_PKL}\" then run:")
print(f"   python rebuild_vptree.py")
