"""
check_missing_videos.py
------------------------
Checks your PKL database for:
  - Videos with no hashes at all
  - Videos with empty hash lists
  - Videos where hashes could not be parsed
  - Videos missing expected fields
  - Summary of healthy vs broken records

Run:
    python check_missing_videos.py
"""

import pickle

# ── Configuration ─────────────────────────────────────────────────────
INPUT_PKL   = "fixed_phash_database.pkl"
HASH_FIELDS = ("hashes", "frames_hashes", "phashes", "frame_hashes")
# ──────────────────────────────────────────────────────────────────────


def load_db(path):
    print(f"Loading PKL: {path}")
    with open(path, "rb") as f:
        db = pickle.load(f)
    print(f"Total records: {len(db)}\n")
    return db


def check_records(db):
    ok          = []   # healthy records
    empty       = []   # record exists but hashes list is empty
    missing     = []   # no hash field found at all
    bad_type    = []   # hash field exists but wrong type
    parse_errors = []  # hashes exist but individual entries are unparseable

    items = db.items() if isinstance(db, dict) else enumerate(db)

    for record_key, record in items:
        if not isinstance(record, dict):
            bad_type.append((record_key, f"record itself is {type(record)}, expected dict"))
            continue

        video_name = (record.get("video_name")
                      or record.get("video_path")
                      or str(record_key))

        # Find which hash field exists
        found_field = None
        raw_hashes  = None
        for field in HASH_FIELDS:
            val = record.get(field)
            if val is not None:
                found_field = field
                raw_hashes  = val
                break

        # No hash field at all
        if found_field is None:
            missing.append((video_name, f"none of {HASH_FIELDS} found — "
                                        f"available fields: {list(record.keys())}"))
            continue

        # Hash field exists but is not a list
        if not isinstance(raw_hashes, (list, tuple)):
            bad_type.append((video_name, f"'{found_field}' is {type(raw_hashes)}, expected list"))
            continue

        # Hash field is an empty list
        if len(raw_hashes) == 0:
            empty.append((video_name, f"'{found_field}' list is empty"))
            continue

        # Check individual hash entries
        frame_errors = []
        for i, h in enumerate(raw_hashes[:10]):   # sample first 10 frames
            if isinstance(h, (bytes, str)):
                continue
            elif isinstance(h, dict):
                if not any(h.get(k) for k in ("hash", "phash", "value")):
                    frame_errors.append(f"frame {i}: dict missing hash/phash/value keys")
            else:
                frame_errors.append(f"frame {i}: unexpected type {type(h)}")

        if frame_errors:
            parse_errors.append((video_name, found_field, len(raw_hashes), frame_errors))
        else:
            ok.append((video_name, found_field, len(raw_hashes)))

    return ok, empty, missing, bad_type, parse_errors


def print_report(ok, empty, missing, bad_type, parse_errors):
    total = len(ok) + len(empty) + len(missing) + len(bad_type) + len(parse_errors)

    print("=" * 60)
    print("  VIDEO HASH DATABASE — HEALTH CHECK REPORT")
    print("=" * 60)

    # ── Healthy ──
    print(f"\n✅  HEALTHY ({len(ok)} / {total} videos)")
    print(f"    {'Video':<35} {'Field':<15} Frames")
    print(f"    {'-'*35} {'-'*15} ------")
    for video_name, field, count in sorted(ok):
        print(f"    {video_name:<35} {field:<15} {count:>7}")

    # ── Empty hash lists ──
    if empty:
        print(f"\n⚠️   EMPTY HASH LIST ({len(empty)} videos) — field found but no frames stored")
        for video_name, reason in sorted(empty):
            print(f"    {video_name}")
            print(f"      → {reason}")

    # ── Missing hash field ──
    if missing:
        print(f"\n❌  MISSING HASH FIELD ({len(missing)} videos) — no recognised hash field in record")
        for video_name, reason in sorted(missing):
            print(f"    {video_name}")
            print(f"      → {reason}")

    # ── Wrong type ──
    if bad_type:
        print(f"\n❌  WRONG TYPE ({len(bad_type)} records)")
        for video_name, reason in sorted(bad_type):
            print(f"    {video_name}")
            print(f"      → {reason}")

    # ── Parse errors ──
    if parse_errors:
        print(f"\n⚠️   PARSE ERRORS ({len(parse_errors)} videos) — hashes exist but some frames unreadable")
        for video_name, field, count, errors in sorted(parse_errors):
            print(f"    {video_name}  (field='{field}', total frames={count})")
            for e in errors[:3]:
                print(f"      → {e}")

    # ── Final summary ──
    print(f"\n{'=' * 60}")
    print(f"  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Total records   : {total}")
    print(f"  Healthy         : {len(ok)}")
    print(f"  Empty lists     : {len(empty)}")
    print(f"  Missing field   : {len(missing)}")
    print(f"  Wrong type      : {len(bad_type)}")
    print(f"  Parse errors    : {len(parse_errors)}")

    broken = len(empty) + len(missing) + len(bad_type) + len(parse_errors)
    if broken == 0:
        print(f"\n  ✅  All records are healthy — safe to build VPTree.")
    else:
        print(f"\n  ⚠️   {broken} records have issues and will be skipped by rebuild_vptree.py.")
        print(f"  These videos will NOT be searchable in the VPTree.")


def main():
    db = load_db(INPUT_PKL)
    ok, empty, missing, bad_type, parse_errors = check_records(db)
    print_report(ok, empty, missing, bad_type, parse_errors)


if __name__ == "__main__":
    main()
