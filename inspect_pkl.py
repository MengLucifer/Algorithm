"""
inspect_pkl.py
--------------
Prints the full structure of your PKL so you can confirm
what fields are available before running rebuild_vptree.py
"""
import pickle
import json

PKL_FILE = "merged_phash_database.pkl"

with open(PKL_FILE, "rb") as f:
    db = pickle.load(f)

print(f"Type  : {type(db)}")
print(f"Length: {len(db)}")
print()

items = list(db.items()) if isinstance(db, dict) else list(enumerate(db))

for i, (key, val) in enumerate(items[:3]):  # show first 3 records
    print(f"── Record {i+1} ──────────────────────────")
    print(f"  Key: {key}")
    if isinstance(val, dict):
        for field, v in val.items():
            if isinstance(v, list):
                print(f"  {field}: list of {len(v)}  →  first item: {str(v[0])[:80] if v else '(empty)'}")
            else:
                print(f"  {field}: {str(v)[:80]}")
    else:
        print(f"  Value: {str(val)[:120]}")
    print()
