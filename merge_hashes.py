"""
merge_hashes.py
---------------
Merges all JSON hash files (F-series + SB24-series) into the existing
phash_database.pkl file, producing a new merged_phash_database.pkl.

Run this script from the folder that contains your files.

Requirements:
    pip install numpy
"""

import os
import json
import pickle
import glob

# ── Configuration ────────────────────────────────────────────────────
INPUT_PKL        = "phash_database.pkl"   # your existing PKL database
OUTPUT_PKL       = "merged_phash_database.pkl"
JSON_PATTERNS    = ["F*_hashes.json", "SB24*.json"]   # patterns to match your JSON files
# ─────────────────────────────────────────────────────────────────────


def load_pkl(path):
    """Load and return the existing PKL database."""
    print(f"[1/4] Loading existing PKL: {path}")
    with open(path, "rb") as f:
        db = pickle.load(f)
    print(f"      Type: {type(db)}")

    # Show a preview of the structure so you can confirm it loaded correctly
    if isinstance(db, dict):
        keys = list(db.keys())[:5]
        print(f"      Keys preview (first 5): {keys}")
        print(f"      Total entries: {len(db)}")
    elif isinstance(db, list):
        print(f"      Total entries: {len(db)}")
        if db:
            print(f"      First entry type: {type(db[0])}")
    else:
        print(f"      Value: {db}")
    return db


def collect_json_files(patterns):
    """Find all JSON files matching the given glob patterns."""
    found = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        found.extend(matches)
    found = sorted(set(found))  # deduplicate and sort
    print(f"\n[2/4] Found {len(found)} JSON file(s) to merge:")
    for f in found:
        size_kb = os.path.getsize(f) / 1024
        print(f"      {f}  ({size_kb:.0f} KB)")
    return found


def load_json_files(json_files):
    """Load all JSON files and merge their contents into a single dict."""
    print(f"\n[3/4] Loading JSON files...")
    merged = {}
    skipped = 0

    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict):
                overlap = set(merged.keys()) & set(data.keys())
                if overlap:
                    print(f"      WARNING: {len(overlap)} duplicate key(s) in {path} — keeping existing values")
                    skipped += len(overlap)
                for k, v in data.items():
                    if k not in merged:
                        merged[k] = v
            elif isinstance(data, list):
                # If JSON is a list of records, try to convert to dict using
                # common key fields; adjust 'hash' / 'path' to match your schema
                for record in data:
                    if isinstance(record, dict):
                        # Use 'path', 'filename', or 'id' as the key — adjust as needed
                        key = record.get("path") or record.get("filename") or record.get("id")
                        if key and key not in merged:
                            merged[key] = record
                        elif key:
                            skipped += 1
                    else:
                        # Fallback: store by sequential index
                        merged[len(merged)] = record
            else:
                print(f"      WARNING: Unexpected JSON structure in {path} (type={type(data)}), skipping")

        except json.JSONDecodeError as e:
            print(f"      ERROR reading {path}: {e}")
        except Exception as e:
            print(f"      ERROR: {path}: {e}")

    print(f"      Loaded {len(merged)} unique entries from JSON files ({skipped} duplicates skipped)")
    return merged


def merge_and_save(pkl_db, json_data, output_path):
    """Merge JSON data into the PKL database and save."""
    print(f"\n[4/4] Merging and saving to: {output_path}")

    if isinstance(pkl_db, dict):
        before = len(pkl_db)
        overlap = set(pkl_db.keys()) & set(json_data.keys())
        if overlap:
            print(f"      WARNING: {len(overlap)} key(s) already exist in PKL — JSON entries will NOT overwrite them")

        # Merge: PKL takes priority for existing keys
        merged_db = {**json_data, **pkl_db}  # pkl_db overwrites duplicates
        after = len(merged_db)
        print(f"      PKL entries:   {before}")
        print(f"      JSON entries:  {len(json_data)}")
        print(f"      New entries added: {after - before}")
        print(f"      Total entries in merged DB: {after}")

    elif isinstance(pkl_db, list):
        # If PKL is a list, convert JSON dict values to list and extend
        json_list = list(json_data.values()) if isinstance(json_data, dict) else json_data
        merged_db = pkl_db + json_list
        print(f"      PKL entries:   {len(pkl_db)}")
        print(f"      JSON entries:  {len(json_list)}")
        print(f"      Total entries: {len(merged_db)}")

    else:
        print("      WARNING: PKL database has an unexpected type.")
        print("      Wrapping both PKL and JSON into a dict with keys 'pkl_data' and 'json_data'.")
        merged_db = {"pkl_data": pkl_db, "json_data": json_data}

    with open(output_path, "wb") as f:
        pickle.dump(merged_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅  Done! Saved: {output_path}  ({size_mb:.1f} MB)")
    return merged_db


def main():
    print("=" * 55)
    print("  pHash Database Merger")
    print("=" * 55)

    # Check that the PKL file exists
    if not os.path.exists(INPUT_PKL):
        print(f"ERROR: Cannot find '{INPUT_PKL}' in the current directory.")
        print(f"       Please run this script from the folder containing your files.")
        return

    pkl_db    = load_pkl(INPUT_PKL)
    json_files = collect_json_files(JSON_PATTERNS)

    if not json_files:
        print("No JSON files found. Check that JSON_PATTERNS in the script matches your filenames.")
        return

    json_data  = load_json_files(json_files)
    merge_and_save(pkl_db, json_data, OUTPUT_PKL)


if __name__ == "__main__":
    main()
