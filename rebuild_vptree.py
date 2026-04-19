"""
rebuild_vptree.py  (fixed for video metadata PKL structure)
------------------------------------------------------------
Your PKL is a dict where each value is a video metadata record like:
{
    "video_name": "F10.mp4",
    "video_path": "D:\\video-4\\F10.mp4",
    "hashes": [...],      <-- list of per-frame pHashes
    "frames": [...],
    "labels": [...],
    ...
}
This script extracts every frame hash from every video and builds one
unified VPTree across all of them.

Run AFTER merge_hashes.py:
    python rebuild_vptree.py

Requirements:
    pip install imagehash vptree Pillow
"""

import pickle
import imagehash
import vptree

# ── Configuration ─────────────────────────────────────────────────────
INPUT_PKL     = "merged_phash_database.pkl"
OUTPUT_VPTREE = "merged_phash_database.vptree"
OUTPUT_KEYS   = "merged_phash_database_keyindex.pkl"
# ──────────────────────────────────────────────────────────────────────


def hamming_distance(hash1, hash2):
    return hash1 - hash2  # imagehash overloads '-' as Hamming distance


def load_db(path):
    print(f"[1/4] Loading PKL: {path}")
    with open(path, "rb") as f:
        db = pickle.load(f)
    print(f"      Type         : {type(db)}")
    print(f"      Total records: {len(db)}")

    # Print first record structure so we can confirm layout
    if isinstance(db, dict):
        first_key = next(iter(db))
        first_val = db[first_key]
        print(f"\n      First record key : {first_key}")
        if isinstance(first_val, dict):
            print(f"      First record fields: {list(first_val.keys())}")
            hashes_field = first_val.get("hashes")
            if hashes_field is not None:
                print(f"      'hashes' field type : {type(hashes_field)}")
                print(f"      'hashes' field length: {len(hashes_field)}")
                if hashes_field:
                    print(f"      First hash sample  : {str(hashes_field[0])[:60]}")
    return db


def extract_hashes(db):
    """
    Extract per-frame hashes from video metadata records.
    Each record has a 'hashes' list — we flatten all frames from all videos.
    Keys are stored as (video_name, frame_index) tuples.
    """
    print(f"\n[2/4] Extracting frame hashes from all video records...")
    hashes = []
    keys   = []
    errors = 0

    items = db.items() if isinstance(db, dict) else enumerate(db)

    for record_key, record in items:
        if not isinstance(record, dict):
            continue

        video_name = record.get("video_name") or record.get("video_path") or str(record_key)
        raw_hashes = record.get("hashes") or record.get("frames_hashes") or []

        if not raw_hashes:
            print(f"      WARNING: No 'hashes' list found for '{video_name}', skipping")
            continue

        for frame_idx, raw_h in enumerate(raw_hashes):
            try:
                if isinstance(raw_h, imagehash.ImageHash):
                    h = raw_h
                elif isinstance(raw_h, str):
                    h = imagehash.hex_to_hash(raw_h)
                elif isinstance(raw_h, dict):
                    # Sometimes hashes are stored as {"hash": "...", "frame": N}
                    raw_str = raw_h.get("hash") or raw_h.get("phash") or raw_h.get("value")
                    if raw_str is None:
                        errors += 1
                        continue
                    h = imagehash.hex_to_hash(raw_str) if isinstance(raw_str, str) else raw_str
                else:
                    errors += 1
                    continue

                hashes.append(h)
                keys.append((video_name, frame_idx))

            except Exception as e:
                errors += 1

        print(f"      {video_name}: {len(raw_hashes)} frames extracted")

    print(f"\n      Total frame hashes extracted: {len(hashes)}")
    if errors:
        print(f"      Skipped (unparseable)       : {errors}")
    return hashes, keys


def build_vptree(hashes):
    print(f"\n[3/4] Building VPTree from {len(hashes)} frame hashes...")
    print(f"      (This may take a minute for large datasets)")
    tree = vptree.VPTree(hashes, hamming_distance)
    print(f"      VPTree built successfully")
    return tree


def save_outputs(tree, keys):
    print(f"\n[4/4] Saving outputs...")

    with open(OUTPUT_VPTREE, "wb") as f:
        pickle.dump(tree, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      Saved: {OUTPUT_VPTREE}")

    with open(OUTPUT_KEYS, "wb") as f:
        pickle.dump(keys, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      Saved: {OUTPUT_KEYS}")


def demo_query(tree, hashes, keys):
    print(f"\n── Sanity check query ──")
    if not hashes:
        return
    test_hash = hashes[0]
    results = tree.get_all_in_range(test_hash, 10)
    print(f"Query: {keys[0]}")
    print(f"Matches within Hamming distance 10: {len(results)}")
    for dist, h in sorted(results, key=lambda x: x[0])[:5]:
        try:
            idx = hashes.index(h)
            print(f"  distance={dist}  ->  {keys[idx]}")
        except ValueError:
            print(f"  distance={dist}")

    print(f"\n✅  All done!")
    print(f"    {OUTPUT_VPTREE}")
    print(f"    {OUTPUT_KEYS}")


def main():
    print("=" * 55)
    print("  VPTree Rebuilder for pHash Database (video)")
    print("=" * 55)

    db             = load_db(INPUT_PKL)
    hashes, keys   = extract_hashes(db)

    if not hashes:
        print("\nERROR: No hashes extracted.")
        print("Tip: Open inspect_pkl.py to see your exact PKL structure:")
        print("     python inspect_pkl.py")
        return

    tree = build_vptree(hashes)
    save_outputs(tree, keys)
    demo_query(tree, hashes, keys)


if __name__ == "__main__":
    main()
