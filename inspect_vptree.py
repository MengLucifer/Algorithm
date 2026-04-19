"""
inspect_vptree.py
------------------
Inspect the contents of your saved VPTree and key index files.
Run:
    python inspect_vptree.py
"""

import pickle
import numpy as np
import imagehash

# ── Configuration ─────────────────────────────────────────────────────
VPTREE_FILE = "merged_phash_database.vptree"
KEYS_FILE   = "merged_phash_database_keyindex.pkl"
# ──────────────────────────────────────────────────────────────────────


def load_files():
    print("=" * 55)
    print("  VPTree Inspector")
    print("=" * 55)

    print(f"\n[1/3] Loading key index: {KEYS_FILE}")
    with open(KEYS_FILE, "rb") as f:
        keys = pickle.load(f)
    print(f"      Total keys : {len(keys)}")
    print(f"      Key type   : {type(keys[0])}")
    print(f"      Sample keys (first 5):")
    for k in keys[:5]:
        print(f"        {k}")

    print(f"\n[2/3] Loading VPTree: {VPTREE_FILE}")
    with open(VPTREE_FILE, "rb") as f:
        tree = pickle.load(f)
    print(f"      VPTree type: {type(tree)}")
    print(f"      VPTree loaded successfully")

    return tree, keys


def summarize_keys(keys):
    print(f"\n[3/3] Summary")
    print(f"      Total frames indexed : {len(keys)}")

    # Group by video name
    videos = {}
    for video_name, frame_idx in keys:
        videos.setdefault(video_name, []).append(frame_idx)

    print(f"      Total videos indexed : {len(videos)}")
    print(f"\n      Per-video frame counts:")
    for video, frames in sorted(videos.items()):
        print(f"        {video:<30} {len(frames):>7} frames")


def query_test(tree, keys):
    print(f"\n── Query Test ──")
    print(f"Searching for near-duplicates of the first frame...")

    # Reload hashes to get the first one for querying
    # We just use tree.get_all_in_range with a sample
    try:
        # Get the vantage point hash from the tree root as test query
        root_hash = tree.vp
        results = tree.get_all_in_range(root_hash, 10)
        print(f"Test query hash   : {root_hash}")
        print(f"Matches (distance ≤ 10): {len(results)}")
        print(f"\nTop 10 closest matches:")
        for dist, h in sorted(results, key=lambda x: x[0])[:10]:
            print(f"  distance={dist}  hash={str(h)[:20]}...")
    except Exception as e:
        print(f"Query test failed: {e}")


def query_by_video(tree, keys, video_name, frame_idx=0, threshold=10):
    """
    Query using a specific frame from a specific video.
    Useful to test if near-duplicate detection works.
    """
    print(f"\n── Custom Query ──")
    print(f"Looking up frame {frame_idx} of '{video_name}'...")

    # Find the hash for this frame
    try:
        target_idx = keys.index((video_name, frame_idx))
    except ValueError:
        print(f"  ERROR: ({video_name}, {frame_idx}) not found in key index.")
        return

    # Reload all hashes from the PKL to get the actual hash value
    # (VPTree doesn't store hashes in a list directly)
    print(f"  Found at index {target_idx} in key list.")
    print(f"  To do a full query, run rebuild_vptree.py and use demo_query().")


def main():
    tree, keys = load_files()
    summarize_keys(keys)
    query_test(tree, keys)

    print(f"\n✅  Inspection complete.")
    print(f"    Use query_by_video(tree, keys, 'F10.mp4', frame_idx=0) "
          f"to test specific frames.")


if __name__ == "__main__":
    main()
