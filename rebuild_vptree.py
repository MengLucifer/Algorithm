"""
rebuild_vptree.py  (imagehash edition)
---------------------------------------
Your PKL stores per-frame hashes as raw bytes.
This version converts bytes -> imagehash.ImageHash, then builds the VPTree
using imagehash's built-in Hamming distance (the '-' operator).

Run:
    python rebuild_vptree.py

Requirements:
    pip install imagehash vptree Pillow numpy
"""

import pickle
import numpy as np
import imagehash
import vptree

# ── Configuration ─────────────────────────────────────────────────────
INPUT_PKL     = "fixed_phash_database.pkl"
OUTPUT_VPTREE = "merged_phash_database.vptree"
OUTPUT_KEYS   = "merged_phash_database_keyindex.pkl"
# ──────────────────────────────────────────────────────────────────────


def hamming_distance(hash1, hash2):
    """imagehash overloads '-' as Hamming distance."""
    return hash1 - hash2


def bytes_to_imagehash(raw_bytes):
    """
    Convert raw bytes to an imagehash.ImageHash object.
    Unpacks bytes to bits, then reshapes into a square bool array.
    Supports 8-byte (64-bit), 16-byte (128-bit), 4-byte (32-bit) hashes.
    """
    bits = np.unpackbits(np.frombuffer(raw_bytes, dtype=np.uint8)).astype(bool)
    total_bits = len(bits)
    side = int(total_bits ** 0.5)
    if side * side != total_bits:
        # Not a perfect square — pad to next square
        next_sq = side + 1
        bits = np.pad(bits, (0, next_sq * next_sq - total_bits))
        side = next_sq
    return imagehash.ImageHash(bits.reshape(side, side))


def load_db(path):
    print(f"[1/4] Loading PKL: {path}")
    with open(path, "rb") as f:
        db = pickle.load(f)
    print(f"      Type         : {type(db)}")
    print(f"      Total records: {len(db)}")

    if isinstance(db, dict):
        first_key = next(iter(db))
        first_val = db[first_key]
        print(f"\n      First record key   : {first_key}")
        if isinstance(first_val, dict):
            print(f"      First record fields: {list(first_val.keys())}")
            hashes_field = first_val.get("hashes")
            if hashes_field is not None:
                print(f"      'hashes' field type  : {type(hashes_field)}")
                print(f"      'hashes' field length: {len(hashes_field)}")
                if hashes_field:
                    sample = hashes_field[0]
                    print(f"      First hash type      : {type(sample)}")
                    if isinstance(sample, bytes):
                        print(f"      First hash length    : {len(sample)} bytes "
                              f"({len(sample) * 8}-bit hash)")
    return db


def extract_hashes(db):
    """
    Extract per-frame hashes from video metadata records.
    Converts raw bytes -> imagehash.ImageHash for each frame.
    Keys are (video_name, frame_index) tuples.
    """
    print(f"\n[2/4] Extracting and converting frame hashes...")
    hashes = []
    keys   = []
    errors = 0

    items = db.items() if isinstance(db, dict) else enumerate(db)

    HASH_FIELDS = ("hashes", "frames_hashes", "phashes", "frame_hashes")

    for record_key, record in items:
        if not isinstance(record, dict):
            continue

        video_name = (record.get("video_name")
                      or record.get("video_path")
                      or str(record_key))

        raw_hashes = None
        for field in HASH_FIELDS:
            raw_hashes = record.get(field)
            if raw_hashes:
                break

        if not raw_hashes:
            print(f"      WARNING: No hash list found for '{video_name}', skipping")
            continue

        frame_count = 0
        for frame_idx, raw_h in enumerate(raw_hashes):
            try:
                if isinstance(raw_h, imagehash.ImageHash):
                    h = raw_h                                   # already ImageHash

                elif isinstance(raw_h, bytes):
                    h = bytes_to_imagehash(raw_h)               # bytes -> ImageHash

                elif isinstance(raw_h, str):
                    h = imagehash.hex_to_hash(raw_h)            # hex string -> ImageHash

                elif isinstance(raw_h, (list, tuple)) and all(isinstance(x, int) for x in raw_h):
                    h = bytes_to_imagehash(bytes(raw_h))        # list of ints -> ImageHash

                elif isinstance(raw_h, dict):
                    raw_val = (raw_h.get("hash")
                               or raw_h.get("phash")
                               or raw_h.get("value"))
                    if raw_val is None:
                        errors += 1
                        continue
                    if isinstance(raw_val, bytes):
                        h = bytes_to_imagehash(raw_val)
                    elif isinstance(raw_val, str):
                        h = imagehash.hex_to_hash(raw_val)
                    else:
                        h = raw_val

                else:
                    print(f"      Unknown hash type at frame {frame_idx}: {type(raw_h)}")
                    errors += 1
                    continue

                hashes.append(h)
                keys.append((video_name, frame_idx))
                frame_count += 1

            except Exception as e:
                print(f"      Parse error [{video_name}] frame {frame_idx}: {e}")
                errors += 1

        print(f"      {video_name}: {frame_count} frames extracted")

    print(f"\n      Total frame hashes extracted: {len(hashes)}")
    if errors:
        print(f"      Skipped (unparseable)       : {errors}")
    return hashes, keys


def build_vptree(hashes):
    print(f"\n[3/4] Building VPTree from {len(hashes)} hashes...")
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
    print(f"Query : {keys[0]}")
    print(f"Matches within Hamming distance 10: {len(results)}")
    for dist, h in sorted(results, key=lambda x: x[0])[:5]:
        try:
            idx = hashes.index(h)
            print(f"  distance={dist}  ->  {keys[idx]}")
        except ValueError:
            print(f"  distance={dist}  ->  (index lookup failed)")

    print(f"\n✅  All done!")
    print(f"    {OUTPUT_VPTREE}")
    print(f"    {OUTPUT_KEYS}")


def main():
    print("=" * 55)
    print("  VPTree Rebuilder for pHash Database (video)")
    print("  Mode: imagehash (bytes -> ImageHash conversion)")
    print("=" * 55)

    db           = load_db(INPUT_PKL)
    hashes, keys = extract_hashes(db)

    if not hashes:
        print("\nERROR: No hashes extracted.")
        print("Tip: Check your PKL structure with:")
        print("     python -c \"import pickle; db=pickle.load(open('fixed_phash_database.pkl','rb')); "
              "k=next(iter(db)); print(type(db[k]['hashes'][0]))\"")
        return

    tree = build_vptree(hashes)
    save_outputs(tree, keys)
    demo_query(tree, hashes, keys)


if __name__ == "__main__":
    main()
