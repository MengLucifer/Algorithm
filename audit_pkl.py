"""
audit_pkl.py
------------
Audits the merged PKL and JSON files to:
1. Show the full structure of every record
2. Identify which records are well-formed video entries
3. Identify which records are abnormal (flat metadata fields)
4. Suggest which to keep or delete

Run: python audit_pkl.py
"""
import pickle
import json
import glob
import imagehash

PKL_FILE      = "merged_phash_database.pkl"
JSON_PATTERNS = ["F*_hashes.json", "SB24*.json"]

# ── Load PKL ──────────────────────────────────────────────────────────
print("=" * 65)
print("  PKL AUDIT")
print("=" * 65)

with open(PKL_FILE, "rb") as f:
    db = pickle.load(f)

print(f"\nTotal keys in PKL: {len(db)}\n")
print(f"{'Key':<30} {'Type':<20} {'Preview'}")
print("-" * 80)

for key, val in db.items():
    if isinstance(val, list):
        first = str(val[0])[:50] if val else "(empty)"
        print(f"{str(key):<30} list[{len(val)}]{'':>12} {first}")
    elif isinstance(val, dict):
        print(f"{str(key):<30} dict({len(val)} keys){'':>8} keys={list(val.keys())[:3]}")
    else:
        print(f"{str(key):<30} {type(val).__name__:<20} {str(val)[:50]}")

# ── Detect structure type ─────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  STRUCTURE DIAGNOSIS")
print("=" * 65)

# Check if PKL looks like flat metadata (single video fields as top-level keys)
flat_metadata_keys = {"video_name", "video_path", "duration_seconds", "duration_human",
                      "resolution", "hash_algorithm", "hash_size", "extraction_mode",
                      "extraction_fps", "total_frames", "processed_at", "processing_time",
                      "cores_used", "frames", "hashes", "labels", "arena_crop",
                      "overlay_masks", "moving_logo_removal"}

db_keys = set(str(k) for k in db.keys())
flat_overlap = db_keys & flat_metadata_keys
is_flat = len(flat_overlap) >= 5  # if 5+ known metadata fields exist as top-level keys

if is_flat:
    print("\n⚠️  PROBLEM DETECTED:")
    print("   Your PKL is structured as a FLAT DICT of metadata fields")
    print("   (one video's fields spread as top-level keys).")
    print("   It should be a dict of {video_id: {all fields}}.")
    print(f"\n   Detected flat metadata fields: {sorted(flat_overlap)}")

    hashes_val = db.get("hashes")
    if hashes_val and isinstance(hashes_val, list):
        print(f"\n   ✅ 'hashes' key found with {len(hashes_val)} frame hashes")
        print(f"      First hash: {str(hashes_val[0])[:60]}")
    else:
        print(f"\n   ❌ No usable 'hashes' list found")
else:
    print("\n   ✅ PKL looks like a proper multi-record dict")

# ── Inspect JSON files ────────────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  JSON FILE AUDIT")
print("=" * 65)

json_files = []
for pattern in JSON_PATTERNS:
    json_files.extend(glob.glob(pattern))
json_files = sorted(set(json_files))

print(f"\nFound {len(json_files)} JSON files\n")

well_formed   = []
abnormal      = []

for path in json_files:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            keys = list(data.keys())
            has_hashes = "hashes" in data or "frames_hashes" in data
            has_video  = "video_name" in data or "video_path" in data
            is_multi   = all(isinstance(v, dict) for v in list(data.values())[:3])

            if is_multi:
                print(f"  ✅ GOOD   {path:<45} multi-record dict ({len(data)} videos)")
                well_formed.append(path)
            elif has_video and has_hashes:
                print(f"  ⚠️  FLAT   {path:<45} single-video flat metadata")
                abnormal.append(path)
            else:
                overlap = set(keys) & flat_metadata_keys
                if len(overlap) >= 3:
                    print(f"  ⚠️  FLAT   {path:<45} looks like flat metadata ({len(keys)} keys)")
                    abnormal.append(path)
                else:
                    print(f"  ✅ GOOD   {path:<45} dict ({len(keys)} keys)")
                    well_formed.append(path)

        elif isinstance(data, list):
            print(f"  ✅ GOOD   {path:<45} list of {len(data)} records")
            well_formed.append(path)
        else:
            print(f"  ❌ UNKNOWN {path:<45} type={type(data)}")
            abnormal.append(path)

    except Exception as e:
        print(f"  ❌ ERROR  {path}: {e}")
        abnormal.append(path)

# ── Summary & Recommendation ──────────────────────────────────────────
print("\n\n" + "=" * 65)
print("  RECOMMENDATION")
print("=" * 65)

print(f"\n  Well-formed files : {len(well_formed)}")
print(f"  Abnormal files    : {len(abnormal)}")

if abnormal:
    print(f"\n  Abnormal files to review/delete:")
    for f in abnormal:
        print(f"    - {f}")

print(f"""
  SUGGESTED FIX:
  Run  python fix_pkl.py  to:
    1. Re-structure the PKL into {{video_id: {{all fields}}}} format
    2. Merge all well-formed JSON files into it
    3. Skip/report abnormal files
    4. Save as  fixed_phash_database.pkl
""")
