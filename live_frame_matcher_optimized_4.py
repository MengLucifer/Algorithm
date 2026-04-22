"""
Live Video Frame Matcher — Windows 24-Core Pipeline (OPTIMIZED v4)
===================================================================
Captures frames from a live-streaming website, masks the watermark area,
computes perceptual hashes, and matches against a pre-built hash database.

VERIFICATION PIPELINE (v4 — fully integrated):
    TIER 1 — Foundation (must all pass)
        1. pHash frame match
        2. Consecutive sequence order
        3. Scene cut timing pattern
    TIER 2 — Strengthening (raises confidence)
        4. Motion energy pattern
        5. Spatial zone brightness
        6. Cycle-aware timestamp tracking

Tuned for:
    CPU:  Intel Core Ultra 9 285K — 24 cores / 24 threads @ 3700 MHz
    RAM:  32 GB DDR5-5600
    GPU:  NVIDIA RTX 4080
    OS:   Windows 10 Pro
    MON:  1920×784 (browser fullscreen)

Requirements:
    pip install opencv-python-headless numpy Pillow mss pygetwindow pywin32

Usage:
    # Build scene-cut + motion + zone databases from your video files
    python live_frame_matcher_optimized_4.py build-verify-db --video-dir "C:\\videos"

    # Import JSON hashes from video_preprocess.py output
    python live_frame_matcher_optimized_4.py import-json "C:\\hash_db"

    # Start live matching with full verification
    python live_frame_matcher_optimized_4.py run

    # Benchmark
    python live_frame_matcher_optimized_4.py benchmark --db "C:\\hash_db\\phash_database.pkl"

    # Test match against DB
    python live_frame_matcher_optimized_4.py test-match --db "C:\\hash_db\\phash_database.pkl"
"""

import os
import sys
import time
import json
import pickle
import logging
import argparse
import multiprocessing as mp
from collections import deque, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Event
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Precompute popcount lookup table
# ---------------------------------------------------------------------------
_POPCOUNT_TABLE = bytes([bin(i).count('1') for i in range(256)])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # --- Source window ---
    window_title: str = "__pick__"
    capture_fps:  int = 5

    source_w: int = 1920
    source_h: int = 784
    proc_w:   int = 1907
    proc_h:   int = 720

    # --- Focus / crop (source space) ---
    focus_x: int = 5
    focus_y: int = 2
    focus_w: int = 1907
    focus_h: int = 720

    # --- Static masks (post-crop coords) ---
    mask_x:     int   = 650
    mask_y:     int   = 286
    mask_w:     int   = 495
    mask_h:     int   = 191
    mask_color: tuple = (0, 0, 0)

    mask2_x:       int  = 1178
    mask2_y:       int  = 520
    mask2_w:       int  = 82
    mask2_h:       int  = 61
    mask2_enabled: bool = True

    # --- Moving logo ---
    moving_logo_enabled:  bool  = True
    logo_min_area:        int   = 500
    logo_max_area_ratio:  float = 0.05
    logo_sat_threshold:   int   = 80
    logo_motion_threshold:int   = 15
    logo_inpaint_radius:  int   = 5
    logo_history_size:    int   = 15
    logo_pad:             int   = 10

    # --- pHash ---
    phash_size:       int = 16
    highfreq_factor:  int = 4

    # --- Hash database ---
    hash_db_path: str = r"C:\hash_db\phash_database.pkl"

    # --- Matching ---
    hamming_threshold: int   = 12
    match_cores:       int   = 16
    phash_cores:       int   = 4
    result_timeout:    float = 5.0

    # --- Verification databases ---
    verify_db_dir: str = r"C:\hash_db\verify"

    # --- Logging / output ---
    log_level:  str = "INFO"
    result_dir: str = r"C:\hash_db\results"


# ===========================================================================
# CORE HASH UTILITIES
# ===========================================================================

def compute_phash_bytes(gray_frame: np.ndarray,
                        hash_size: int = 16,
                        highfreq_factor: int = 4) -> bytes:
    img_size = hash_size * highfreq_factor
    resized  = cv2.resize(gray_frame, (img_size, img_size),
                          interpolation=cv2.INTER_AREA)
    dct       = cv2.dct(np.float32(resized))
    dctlowfreq = dct[:hash_size, :hash_size]
    med  = np.median(dctlowfreq)
    bits = (dctlowfreq > med).flatten()
    return np.packbits(bits.astype(np.uint8)).tobytes()


def hamming_distance(a_bytes: bytes, b_bytes: bytes) -> int:
    xor   = int.from_bytes(a_bytes, 'big') ^ int.from_bytes(b_bytes, 'big')
    xor_b = xor.to_bytes(len(a_bytes), 'big')
    return sum(_POPCOUNT_TABLE[v] for v in xor_b)


def hamming_distance_int(a_int: int, b_int: int, n_bytes: int = 32) -> int:
    xor          = a_int ^ b_int
    actual_bytes = max(n_bytes, (xor.bit_length() + 7) // 8)
    xor_bytes    = xor.to_bytes(actual_bytes, 'big')
    return sum(_POPCOUNT_TABLE[b] for b in xor_bytes)


def bytes_to_int(h: bytes) -> int:
    return int.from_bytes(h, 'big')


def parse_label(label: str):
    """Returns (video_name, video_sec_float) from 'name|HH:MM:SS|t<sec>'."""
    parts = label.split("|")
    if len(parts) != 3:
        return None, None
    video_name = parts[0]
    try:
        video_sec = float(parts[2].replace("t", ""))
    except ValueError:
        return video_name, None
    return video_name, video_sec


# ===========================================================================
# VP-TREE
# ===========================================================================

class VPNode:
    __slots__ = ('point', 'point_int', 'label', 'threshold', 'left', 'right')
    def __init__(self, point, point_int, label,
                 threshold=0, left=None, right=None):
        self.point     = point
        self.point_int = point_int
        self.label     = label
        self.threshold = threshold
        self.left      = left
        self.right     = right


def _build_vptree(points, point_ints, labels):
    if not points:
        return None

    import collections
    WorkItem = collections.namedtuple('WorkItem', ['idxs', 'set_on_parent'])

    root = None

    def _make_setter(parent_node, left):
        if left:
            def setter(n): parent_node.left = n
        else:
            def setter(n): parent_node.right = n
        return setter

    stack = [WorkItem(list(range(len(points))), None)]

    while stack:
        item          = stack.pop()
        idxs          = item.idxs
        set_on_parent = item.set_on_parent

        if not idxs:
            if set_on_parent:
                set_on_parent(None)
            continue

        if len(idxs) == 1:
            i    = idxs[0]
            node = VPNode(points[i], point_ints[i], labels[i])
            if set_on_parent:
                set_on_parent(node)
            elif root is None:
                root = node
            continue

        pick    = np.random.randint(len(idxs))
        vp_idx  = idxs[pick]
        rest    = idxs[:pick] + idxs[pick+1:]
        vp_int  = point_ints[vp_idx]
        distances = [hamming_distance_int(vp_int, point_ints[i]) for i in rest]
        med       = int(np.median(distances)) if distances else 0

        left_idx, right_idx = [], []
        for d, i in zip(distances, rest):
            if d < med:
                left_idx.append(i)
            else:
                right_idx.append(i)

        if not left_idx and right_idx:
            mid      = len(right_idx) // 2
            left_idx = right_idx[:mid]
            right_idx = right_idx[mid:]

        node = VPNode(points[vp_idx], vp_int, labels[vp_idx], med)
        if set_on_parent:
            set_on_parent(node)
        elif root is None:
            root = node

        stack.append(WorkItem(right_idx, _make_setter(node, False)))
        stack.append(WorkItem(left_idx,  _make_setter(node, True)))

    return root


def _search_vptree(node, target_int, threshold, best, n_bytes=32):
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue
        d = hamming_distance_int(current.point_int, target_int, n_bytes)
        if d < best[0]:
            best[0] = d
            best[1] = current.label
        if d - threshold <= current.threshold:
            stack.append(current.left)
        if d + threshold >= current.threshold:
            stack.append(current.right)


class VPTree:
    def __init__(self, hashes: list, labels: list):
        point_ints  = [bytes_to_int(h) for h in hashes]
        self.n_bytes = len(hashes[0]) if hashes else 32
        t0           = time.perf_counter()
        self.root    = _build_vptree(list(hashes), point_ints, list(labels))
        logging.info("VP-tree built in %.1f ms",
                     (time.perf_counter() - t0) * 1000)

    def query(self, target: bytes, threshold: int):
        target_int = bytes_to_int(target)
        best       = [threshold + 1, None]
        _search_vptree(self.root, target_int, threshold, best, self.n_bytes)
        if best[1] is not None:
            return best[0], best[1]
        return None


# ===========================================================================
# HASH DATABASE
# ===========================================================================

class HashDB:
    _POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)],
                             dtype=np.int32)

    def __init__(self, path: str, threshold: int = 12):
        self.path      = path
        self.threshold = threshold
        self.hashes:   list = []
        self.labels:   list = []
        self.vptree:   Optional[VPTree] = None
        self._hash_array  = None
        self._labels_list = None

    def load(self):
        p = Path(self.path)
        if not p.exists():
            logging.warning("hash_db not found at %s — starting empty", self.path)
            return
        with open(p, "rb") as f:
            data = pickle.load(f)
        self.hashes = data["hashes"]
        self.labels = data["labels"]
        logging.info("Loaded %d hashes from %s", len(self.hashes), self.path)
        if self.hashes:
            hash_len = len(self.hashes[0])
            self._hash_array = np.frombuffer(
                b''.join(self.hashes), dtype=np.uint8
            ).reshape(len(self.hashes), hash_len)
            self._labels_list = self.labels

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"hashes": self.hashes, "labels": self.labels}, f)
        logging.info("Saved %d hashes to %s", len(self.hashes), self.path)

    def add(self, h: bytes, label: str):
        self.hashes.append(h)
        self.labels.append(label)

    def _vptree_cache_path(self):
        return str(Path(self.path).with_suffix('.vptree'))

    def build_index(self):
        if not self.hashes:
            return
        cache_path = self._vptree_cache_path()
        if os.path.exists(cache_path):
            try:
                if os.path.getmtime(cache_path) >= os.path.getmtime(self.path):
                    t0 = time.perf_counter()
                    with open(cache_path, "rb") as f:
                        self.vptree = pickle.load(f)
                    logging.info("VP-tree loaded from cache in %.0f ms",
                                 (time.perf_counter() - t0) * 1000)
                    return
            except Exception as e:
                logging.warning("VP-tree cache load failed (%s) — rebuilding", e)

        logging.info("Building VP-tree over %d entries …", len(self.hashes))
        self.vptree = VPTree(self.hashes, self.labels)
        try:
            old = sys.getrecursionlimit()
            sys.setrecursionlimit(max(old, len(self.hashes) + 1000))
            with open(cache_path, "wb") as f:
                pickle.dump(self.vptree, f, protocol=pickle.HIGHEST_PROTOCOL)
            sys.setrecursionlimit(old)
            logging.info("VP-tree cached (%.1f MB)",
                         os.path.getsize(cache_path) / (1024*1024))
        except Exception as e:
            logging.warning("Could not cache VP-tree: %s", e)

    def query(self, target: bytes):
        if self.vptree:
            return self.vptree.query(target, self.threshold)
        if self._hash_array is not None:
            target_bytes = np.frombuffer(target, dtype=np.uint8)
            if target_bytes.shape[0] == self._hash_array.shape[1]:
                xor       = np.bitwise_xor(self._hash_array, target_bytes)
                distances = self._POPCOUNT_LUT[xor].sum(axis=1)
                best_idx  = int(np.argmin(distances))
                best_dist = int(distances[best_idx])
                if best_dist <= self.threshold:
                    return (best_dist, self._labels_list[best_idx])
                return None
        target_int = bytes_to_int(target)
        best_d, best_l = self.threshold + 1, None
        n_bytes = len(target)
        for h, l in zip(self.hashes, self.labels):
            d = hamming_distance_int(target_int, bytes_to_int(h), n_bytes)
            if d < best_d:
                best_d, best_l = d, l
        return (best_d, best_l) if best_l else None

    def import_json(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        video_name = data.get("video_name", Path(json_path).stem)
        frames     = data.get("frames", [])
        imported   = 0
        for frame in frames:
            hex_hash  = frame.get("h")
            timestamp = frame.get("t", 0)
            if hex_hash is None:
                continue
            try:
                h_bytes = bytes.fromhex(hex_hash)
            except ValueError as e:
                logging.warning("Skipping frame t=%s: invalid hex (%s)", timestamp, e)
                continue
            total_sec = timestamp
            ts_str = "{:02d}:{:02d}:{:02d}".format(
                int(total_sec // 3600),
                int((total_sec % 3600) // 60),
                int(total_sec % 60),
            )
            self.add(h_bytes, f"{video_name}|{ts_str}|t{timestamp}")
            imported += 1
        logging.info("Imported %d hashes from %s", imported, json_path)
        return imported


# ===========================================================================
# FAST HASH STORE  — vectorised numpy, replaces list-of-bytes in HashDB
# ===========================================================================

class FastHashStore:
    """
    Stores hashes as a contiguous uint8 numpy array for vectorised Hamming.
    ~4-8x faster than list-of-bytes when VP-tree cache is cold.
    """
    _LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

    def __init__(self, threshold: int = 12):
        self.threshold  = threshold
        self.hash_array : Optional[np.ndarray] = None  # (N, hash_len) uint8
        self.labels     : list = []

    def from_lists(self, hashes: list, labels: list):
        if not hashes:
            return
        hash_len        = len(hashes[0])
        self.hash_array = np.frombuffer(
            b''.join(hashes), dtype=np.uint8
        ).reshape(len(hashes), hash_len)
        self.labels = list(labels)

    def query_numpy(self, target: bytes) -> Optional[tuple]:
        if self.hash_array is None:
            return None
        t   = np.frombuffer(target, dtype=np.uint8)
        xor = np.bitwise_xor(self.hash_array, t)
        d   = self._LUT[xor].sum(axis=1)
        idx = int(np.argmin(d))
        if d[idx] <= self.threshold:
            return int(d[idx]), self.labels[idx]
        return None

    def __len__(self):
        return len(self.labels)


# ===========================================================================
# PARALLEL STARTUP LOADER
# ===========================================================================

def _load_json_file(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _parallel_load_verify_dbs(verify_dir: str) -> dict:
    """
    Load all 3 verify DBs simultaneously using threads (I/O bound).
    Returns dict with keys: scene_cuts, motion, zones
    """
    vdir   = Path(verify_dir)
    db_map = {
        "scene_cuts": str(vdir / "scene_cuts.json"),
        "motion":     str(vdir / "motion.json"),
        "zones":      str(vdir / "zones.json"),
    }
    results = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_load_json_file, path): key
            for key, path in db_map.items()
            if Path(path).exists()
        }
        for key, path in db_map.items():
            if not Path(path).exists():
                logging.warning("[LOADER] Verify DB missing: %s", path)
                results[key] = None
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                results[key] = fut.result()
                logging.info("[LOADER] ✓ %s.json loaded", key)
            except Exception as e:
                logging.warning("[LOADER] ✗ %s failed: %s", key, e)
                results[key] = None
    return results


def _inject_verify_data(verify_data: dict, verify_dir: str) -> dict:
    """
    Instantiates verifier objects and injects pre-loaded JSON — no disk re-read.
    Returns dict: {scene, motion, zones}
    """
    verifiers = {}
    vdir      = str(Path(verify_dir))

    if verify_data.get("scene_cuts"):
        sc        = SceneCutTimeline(str(Path(vdir) / "scene_cuts.json"))
        sc.cut_db = verify_data["scene_cuts"].get("cuts", {})
        sc.gap_db = verify_data["scene_cuts"].get("gaps", {})
        verifiers["scene"] = sc
        logging.info("[LOADER] scene_verifier ready (%d videos)", len(sc.cut_db))

    if verify_data.get("motion"):
        mv           = MotionPatternVerifier(str(Path(vdir) / "motion.json"))
        mv.motion_db = verify_data["motion"]
        verifiers["motion"] = mv
        logging.info("[LOADER] motion_verifier ready (%d videos)", len(mv.motion_db))

    if verify_data.get("zones"):
        zv         = SpatialZoneVerifier(str(Path(vdir) / "zones.json"))
        zv.zone_db = verify_data["zones"]
        verifiers["zones"] = zv
        logging.info("[LOADER] zone_verifier ready (%d videos)", len(zv.zone_db))

    return verifiers


def parallel_load_all(cfg: "Config") -> tuple:
    """
    Loads PKL hash DB + all 3 verify DBs simultaneously.
    PKL load  → thread A  (I/O)
    Verify DBs → thread B  (3 files in parallel, I/O)
    Both run concurrently, then VP-tree is built/cached after.

    Returns: (db, store, vptree, verifiers)
    """
    t_total = time.perf_counter()
    logging.info("[LOADER] ═══ PARALLEL STARTUP BEGIN ═══")

    # ── Thread A: load PKL ────────────────────────────────────────────────────
    def _load_pkl():
        db = HashDB(cfg.hash_db_path, cfg.hamming_threshold)
        db.load()
        return db

    # ── Thread B: load verify DBs ─────────────────────────────────────────────
    def _load_verify():
        return _parallel_load_verify_dbs(cfg.verify_db_dir)

    db          = None
    verify_data = None

    with ThreadPoolExecutor(max_workers=2) as outer:
        fut_db     = outer.submit(_load_pkl)
        fut_verify = outer.submit(_load_verify)
        db          = fut_db.result()
        verify_data = fut_verify.result()

    logging.info("[LOADER] PKL + verify DBs loaded in %.2f s",
                 time.perf_counter() - t_total)

    # ── Build FastHashStore (numpy array) ─────────────────────────────────────
    t0    = time.perf_counter()
    store = FastHashStore(threshold=cfg.hamming_threshold)
    store.from_lists(db.hashes, db.labels)
    logging.info("[LOADER] FastHashStore: %d entries in %.1f ms",
                 len(store), (time.perf_counter() - t0) * 1000)

    # ── Build / load VP-tree (cache-aware) ────────────────────────────────────
    db.build_index()                          # uses existing cache logic

    # ── Inject verify data into verifier objects ──────────────────────────────
    # (must happen after verifier classes are defined — they are, at this point)
    verifiers = _inject_verify_data(verify_data, cfg.verify_db_dir)

    elapsed = time.perf_counter() - t_total
    logging.info("[LOADER] ═══ PARALLEL STARTUP DONE in %.2f s — %d hashes ═══",
                 elapsed, len(store))

    return db, store, verifiers


# ===========================================================================
# VERIFICATION — TIER 1a: pHash SEQUENCE ORDER
# ===========================================================================

class SequenceVerifier:
    """
    TIER 1 — Step 2
    Verifies that N consecutive live frames match N consecutive DB frames
    IN THE CORRECT ORDER with consistent spacing.
    """

    def __init__(self, sequence_len: int = 10, max_gap_sec: float = 2.0):
        self.sequence_len = sequence_len
        self.max_gap      = max_gap_sec
        self.buffer       = []

    def add_match(self, video_name: str, video_sec: float,
                  hamming_dist: int) -> dict:
        self.buffer.append((video_name, video_sec, hamming_dist))
        if len(self.buffer) > self.sequence_len * 2:
            self.buffer.pop(0)
        return self._verify()

    def _verify(self) -> dict:
        if len(self.buffer) < self.sequence_len:
            return {"status": "BUILDING",
                    "have": len(self.buffer),
                    "need": self.sequence_len}

        recent = self.buffer[-self.sequence_len:]
        names  = [r[0] for r in recent]
        if len(set(names)) > 1:
            return {"status": "MIXED_VIDEOS"}

        timestamps = [r[1] for r in recent]
        diffs      = [timestamps[i+1] - timestamps[i]
                      for i in range(len(timestamps) - 1)]

        all_positive   = all(d > 0 for d in diffs)
        all_consistent = all(0 < d <= self.max_gap for d in diffs)

        if all_positive and all_consistent:
            return {
                "status":   "SEQUENCE_VERIFIED",
                "video":    recent[0][0],
                "from_t":   recent[0][1],
                "to_t":     recent[-1][1],
                "n_frames": self.sequence_len,
            }
        return {"status": "SEQUENCE_BROKEN",
                "diffs":  diffs}

    def reset(self):
        self.buffer.clear()


# ===========================================================================
# VERIFICATION — TIER 1b: SCENE CUT TIMING PATTERN
# ===========================================================================

class SceneCutTimeline:
    """
    TIER 1 — Step 3
    Pre-build: records WHEN scene cuts occur in each video file.
    Runtime:   detects cuts in live stream and matches GAP PATTERN.
    """

    CUT_THRESHOLD = 30
    MIN_CUT_GAP   = 2.0

    def __init__(self, db_path: str):
        self.db_path  = db_path
        self.cut_db   = {}
        self.gap_db   = {}

        self.live_cuts      = []
        self.live_gaps      = []
        self.last_gray      = None
        self.last_cut_time  = -999.0
        self.last_real_time = None

    def build(self, video_dir: str, sample_fps: int = 5):
        exts  = {".mp4", ".mkv", ".avi"}
        files = [f for f in Path(video_dir).rglob("*")
                 if f.suffix.lower() in exts]

        for vf in files:
            logging.info("[SCENE-CUT] Scanning: %s", vf.name)
            cuts = self._extract_cuts(str(vf), sample_fps)
            gaps = self._to_gaps(cuts)
            self.cut_db[vf.name] = cuts
            self.gap_db[vf.name] = gaps
            logging.info("[SCENE-CUT]   %d cuts, gaps: %s ...",
                         len(cuts), gaps[:5])

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump({"cuts": self.cut_db, "gaps": self.gap_db}, f, indent=2)
        logging.info("[SCENE-CUT] DB saved: %s", self.db_path)

    def _extract_cuts(self, video_path: str, sample_fps: int) -> list:
        cap      = cv2.VideoCapture(video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / sample_fps))
        prev     = None
        cuts     = []
        last_cut = -999.0
        idx      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (64, 64))
                sec   = idx / fps
                if prev is not None:
                    diff = float(np.mean(np.abs(
                        small.astype(np.float32) - prev.astype(np.float32)
                    )))
                    if diff > self.CUT_THRESHOLD and (sec - last_cut) > self.MIN_CUT_GAP:
                        cuts.append(round(sec, 2))
                        last_cut = sec
                prev = small
            idx += 1
        cap.release()
        return cuts

    def _to_gaps(self, cuts: list) -> list:
        if len(cuts) < 2:
            return []
        return [round(cuts[i+1] - cuts[i], 2) for i in range(len(cuts) - 1)]

    def load(self):
        with open(self.db_path) as f:
            data = json.load(f)
        self.cut_db = data["cuts"]
        self.gap_db = data["gaps"]
        logging.info("[SCENE-CUT] Loaded %d video cut profiles", len(self.cut_db))

    def add_live_frame(self, gray_frame: np.ndarray,
                       real_time: float) -> Optional[dict]:
        small = cv2.resize(gray_frame, (64, 64))

        if self.last_gray is not None:
            diff = float(np.mean(np.abs(
                small.astype(np.float32) - self.last_gray.astype(np.float32)
            )))
            elapsed_since_cut = real_time - self.last_cut_time

            if diff > self.CUT_THRESHOLD and elapsed_since_cut > self.MIN_CUT_GAP:
                if self.last_cut_time > 0:
                    gap = round(real_time - self.last_cut_time, 2)
                    self.live_gaps.append(gap)
                    if len(self.live_gaps) > 50:
                        self.live_gaps.pop(0)
                self.live_cuts.append(real_time)
                self.last_cut_time = real_time

        self.last_gray      = small
        self.last_real_time = real_time

        if len(self.live_gaps) >= 3:
            return self.match_gap_pattern(self.live_gaps)
        return {"status": "ACCUMULATING_CUTS",
                "cuts_so_far": len(self.live_cuts)}

    def match_gap_pattern(self, live_gaps: list,
                          tolerance_sec: float = 1.5,
                          min_match: int = 3) -> dict:
        if len(live_gaps) < min_match:
            return {"status": "INSUFFICIENT_GAPS",
                    "have": len(live_gaps)}

        best_video = None
        best_score = 0

        for video_name, db_gaps in self.gap_db.items():
            score = self._sliding_match(live_gaps, db_gaps, tolerance_sec)
            if score > best_score:
                best_score = score
                best_video = video_name

        if best_score >= min_match:
            return {
                "status":     "CUT_PATTERN_VERIFIED",
                "video":      best_video,
                "score":      best_score,
                "confidence": f"{best_score} consecutive gap(s) matched",
            }
        return {"status": "CUT_PATTERN_UNMATCHED",
                "best_score": best_score}

    def _sliding_match(self, live_gaps: list, db_gaps: list,
                       tolerance: float) -> int:
        best = 0
        n    = len(live_gaps)
        for start in range(max(1, len(db_gaps) - n + 1)):
            window  = db_gaps[start: start + n]
            matched = sum(1 for lg, dg in zip(live_gaps, window)
                          if abs(lg - dg) <= tolerance)
            best = max(best, matched)
        return best

    def reset_live(self):
        self.live_cuts     = []
        self.live_gaps     = []
        self.last_cut_time = -999.0


# ===========================================================================
# VERIFICATION — TIER 2a: MOTION ENERGY PATTERN
# ===========================================================================

class MotionPatternVerifier:
    """
    TIER 2 — Step 4
    Records per-frame motion energy (mean pixel diff between consecutive frames).
    """

    SAMPLE_FPS  = 2
    WINDOW_SEC  = 30
    TOLERANCE   = 0.05

    def __init__(self, db_path: str):
        self.db_path   = db_path
        self.motion_db = {}

        self.live_motion = deque(maxlen=self.WINDOW_SEC * self.SAMPLE_FPS)
        self.prev_gray   = None

    def build(self, video_dir: str):
        exts  = {".mp4", ".mkv", ".avi"}
        files = [f for f in Path(video_dir).rglob("*")
                 if f.suffix.lower() in exts]

        for vf in files:
            logging.info("[MOTION] Building profile: %s", vf.name)
            profile = self._extract_profile(str(vf))
            self.motion_db[vf.name] = profile
            logging.info("[MOTION]   %d samples", len(profile))

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.motion_db, f)
        logging.info("[MOTION] DB saved: %s", self.db_path)

    def _extract_profile(self, video_path: str) -> list:
        cap      = cv2.VideoCapture(video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / self.SAMPLE_FPS))
        prev     = None
        profile  = []
        idx      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (160, 90))
                if prev is not None:
                    energy = float(np.mean(cv2.absdiff(small, prev))) / 255.0
                    profile.append(round(energy, 4))
                prev = small
            idx += 1
        cap.release()
        return profile

    def load(self):
        with open(self.db_path) as f:
            self.motion_db = json.load(f)
        logging.info("[MOTION] Loaded %d video profiles", len(self.motion_db))

    def add_live_frame(self, gray_frame: np.ndarray):
        small = cv2.resize(gray_frame, (160, 90))
        if self.prev_gray is not None:
            energy = float(np.mean(cv2.absdiff(small, self.prev_gray))) / 255.0
            self.live_motion.append(round(energy, 4))
        self.prev_gray = small

    def verify(self, matched_video: str, matched_sec: float) -> dict:
        if len(self.live_motion) < 5:
            return {"status": "ACCUMULATING",
                    "have": len(self.live_motion)}

        db_profile = self.motion_db.get(matched_video)
        if not db_profile:
            return {"status": "VIDEO_NOT_IN_MOTION_DB"}

        start_idx = int(matched_sec * self.SAMPLE_FPS)
        window    = list(self.live_motion)
        db_window = db_profile[start_idx: start_idx + len(window)]

        if len(db_window) < len(window):
            return {"status": "BEYOND_VIDEO_LENGTH"}

        matches = sum(1 for lv, dv in zip(window, db_window)
                      if abs(lv - dv) <= self.TOLERANCE)
        ratio   = matches / len(window)

        if ratio >= 0.70:
            return {"status":      "MOTION_VERIFIED",
                    "video":       matched_video,
                    "match_ratio": f"{ratio*100:.0f}%"}
        return {"status":      "MOTION_MISMATCH",
                "match_ratio": f"{ratio*100:.0f}%"}


# ===========================================================================
# VERIFICATION — TIER 2b: SPATIAL ZONE BRIGHTNESS
# ===========================================================================

class SpatialZoneVerifier:
    """
    TIER 2 — Step 5
    Divides each frame into a 4×4 grid (16 zones).
    Tracks average brightness per zone over time.
    """

    GRID_X     = 4
    GRID_Y     = 4
    SAMPLE_FPS = 2
    TOLERANCE  = 0.08

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.zone_db = {}

        self.live_zones = deque(maxlen=60)

    def build(self, video_dir: str):
        exts  = {".mp4", ".mkv", ".avi"}
        files = [f for f in Path(video_dir).rglob("*")
                 if f.suffix.lower() in exts]

        for vf in files:
            logging.info("[ZONES] Building zone profile: %s", vf.name)
            profile = self._extract_profile(str(vf))
            self.zone_db[vf.name] = profile
            logging.info("[ZONES]   %d zone samples", len(profile))

        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w") as f:
            json.dump(self.zone_db, f)
        logging.info("[ZONES] DB saved: %s", self.db_path)

    def _extract_profile(self, video_path: str) -> list:
        cap      = cv2.VideoCapture(video_path)
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / self.SAMPLE_FPS))
        profile  = []
        idx      = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                profile.append(self._compute_zones(frame))
            idx += 1
        cap.release()
        return profile

    def _compute_zones(self, frame: np.ndarray) -> list:
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
                if len(frame.shape) == 3 else frame
        h, w  = gray.shape
        zh    = h // self.GRID_Y
        zw    = w // self.GRID_X
        zones = []
        for gy in range(self.GRID_Y):
            for gx in range(self.GRID_X):
                zone = gray[gy*zh:(gy+1)*zh, gx*zw:(gx+1)*zw]
                zones.append(round(float(np.mean(zone)) / 255.0, 4))
        return zones

    def load(self):
        with open(self.db_path) as f:
            self.zone_db = json.load(f)
        logging.info("[ZONES] Loaded %d video zone profiles", len(self.zone_db))

    def add_live_frame(self, frame: np.ndarray):
        self.live_zones.append(self._compute_zones(frame))

    def verify(self, matched_video: str, matched_sec: float) -> dict:
        if len(self.live_zones) < 5:
            return {"status": "ACCUMULATING",
                    "have": len(self.live_zones)}

        db_profile = self.zone_db.get(matched_video)
        if not db_profile:
            return {"status": "VIDEO_NOT_IN_ZONE_DB"}

        start_idx = int(matched_sec * self.SAMPLE_FPS)
        window    = list(self.live_zones)
        db_window = db_profile[start_idx: start_idx + len(window)]

        if len(db_window) < len(window):
            return {"status": "BEYOND_VIDEO_LENGTH"}

        total_zones   = len(window) * self.GRID_X * self.GRID_Y
        matched_zones = sum(
            1 for live_z, db_z in zip(window, db_window)
            for lv, dv in zip(live_z, db_z)
            if abs(lv - dv) <= self.TOLERANCE
        )
        ratio = matched_zones / total_zones

        if ratio >= 0.75:
            return {"status":      "ZONE_VERIFIED",
                    "video":       matched_video,
                    "match_ratio": f"{ratio*100:.0f}%"}
        return {"status":      "ZONE_MISMATCH",
                "match_ratio": f"{ratio*100:.0f}%"}


# ===========================================================================
# COMBINED CONFIDENCE SCORER
# ===========================================================================

def compute_confidence(results: dict) -> dict:
    """
    Aggregates 5 verification results into a single score and verdict.
    Tier 1 must ALL pass for any positive verdict.

    TIER 1 — Foundation (80 pts):
        pHash(20) + Sequence(30) + Scene cuts(30)
    TIER 2 — Strengthening (20 pts):
        Motion(10) + Zones(10)
    """
    score   = 0
    passed  = []
    failed  = []

    # ── TIER 1 — Foundation (80 pts) ───────────────────────
    if results.get("phash") == "VERIFIED":
        score += 20
        passed.append("✓ pHash frame matched")
    else:
        failed.append("✗ pHash frame not matched")

    if results.get("sequence") == "SEQUENCE_VERIFIED":
        score += 30
        passed.append("✓ Consecutive frame sequence verified")
    else:
        failed.append("✗ Frame sequence not verified")

    if results.get("scene_cuts") == "CUT_PATTERN_VERIFIED":
        score += 30
        passed.append("✓ Scene cut timing pattern matched")
    else:
        failed.append("✗ Scene cut pattern not matched")

    # ── TIER 2 — Strengthening (20 pts) ────────────────────
    if results.get("motion") == "MOTION_VERIFIED":
        score += 10
        passed.append("✓ Motion energy pattern matched")
    else:
        failed.append("✗ Motion pattern not matched")

    if results.get("zones") == "ZONE_VERIFIED":
        score += 10
        passed.append("✓ Spatial zone brightness matched")
    else:
        failed.append("✗ Spatial zones not matched")

    # ── Verdict ─────────────────────────────────────────────
    tier1_passed = (
        results.get("phash")      == "VERIFIED" and
        results.get("sequence")   == "SEQUENCE_VERIFIED" and
        results.get("scene_cuts") == "CUT_PATTERN_VERIFIED"
    )

    if not tier1_passed:
        verdict = "NOT VERIFIED ✗✗"
    elif score >= 90:
        verdict = "PROVEN ✓✓✓"
    elif score >= 75:
        verdict = "HIGH CONFIDENCE ✓✓"
    elif score >= 60:
        verdict = "MODERATE ✓"
    else:
        verdict = "LOW CONFIDENCE ✗"

    return {
        "score":        score,
        "max_score":    100,
        "verdict":      verdict,
        "tier1_passed": tier1_passed,
        "passed":       passed,
        "failed":       failed,
    }


# ===========================================================================
# MOVING LOGO DETECTOR
# ===========================================================================

class MovingLogoDetector:
    def __init__(self, cfg: Config):
        self.enabled        = cfg.moving_logo_enabled
        self.min_area       = cfg.logo_min_area
        self.max_area_ratio = cfg.logo_max_area_ratio
        self.sat_thresh     = cfg.logo_sat_threshold
        self.motion_thresh  = cfg.logo_motion_threshold
        self.inpaint_radius = cfg.logo_inpaint_radius
        self.history_size   = cfg.logo_history_size
        self.pad            = cfg.logo_pad
        self.prev_gray      = None
        self.history        = []
        self.miss_streak    = 0
        self.skip_after     = 30

    def _detect_bbox(self, frame):
        if self.miss_streak > self.skip_after and not self.history:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_gray = gray
            if self.miss_streak % 60 != 0:
                return None

        h, w     = frame.shape[:2]
        max_area = w * h * self.max_area_ratio

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, sat_mask = cv2.threshold(
            hsv[:, :, 1], self.sat_thresh, 255, cv2.THRESH_BINARY)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return None

        diff = cv2.absdiff(gray, self.prev_gray)
        self.prev_gray = gray.copy()
        _, motion_mask = cv2.threshold(
            diff, self.motion_thresh, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        combined = cv2.bitwise_and(sat_mask, motion_mask)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,
                                    np.ones((5, 5), np.uint8))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE,
                                    np.ones((20, 20), np.uint8))

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best, best_score = None, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area or area > max_area:
                continue
            bx, by, bw, bh = cv2.boundingRect(cnt)
            score = area
            if self.history:
                avg_x = np.mean([r[0] for r in self.history])
                avg_y = np.mean([r[1] for r in self.history])
                dist  = np.hypot(bx - avg_x, by - avg_y)
                score += max(0, 5000 - dist * 10)
            if score > best_score:
                best_score = score
                best = (bx, by, bw, bh)

        if best:
            bx, by, bw, bh = best
            bx = max(0, bx - self.pad)
            by = max(0, by - self.pad)
            bw = min(w - bx, bw + self.pad * 2)
            bh = min(h - by, bh + self.pad * 2)
            best = (bx, by, bw, bh)
            self.history.append(best)
            if len(self.history) > self.history_size:
                self.history.pop(0)
            self.miss_streak = 0
        else:
            self.miss_streak += 1

        return best

    def process(self, frame: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return frame
        bbox = self._detect_bbox(frame)
        if bbox is None:
            return frame
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255
        return cv2.inpaint(frame, mask, self.inpaint_radius, cv2.INPAINT_TELEA)


# ===========================================================================
# PIPELINE WORKERS
# ===========================================================================

def pick_window() -> str:
    try:
        import win32gui
    except ImportError:
        return input("\n  pywin32 not installed. Type window title: ").strip() or "SABONG"

    windows = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip() and len(title) > 1:
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    if w > 100 and h > 100:
                        windows.append((hwnd, title, w, h))
                except Exception:
                    pass
        return True

    win32gui.EnumWindows(callback, None)

    if not windows:
        return input("  No windows found. Type title: ").strip() or "SABONG"

    print("\n" + "=" * 62)
    print("  SELECT WINDOW TO CAPTURE")
    print("=" * 62)
    for i, (hwnd, title, w, h) in enumerate(windows, 1):
        display = title[:50] + "..." if len(title) > 50 else title
        print(f"  [{i:2d}]  {display:<54} ({w}x{h})")
    print(f"\n  [ 0]  Full screen capture")
    print("=" * 62)

    while True:
        try:
            choice = input("\n  Enter number: ").strip()
            if choice == "0":
                return "__FULLSCREEN__"
            num = int(choice)
            if 1 <= num <= len(windows):
                selected = windows[num - 1]
                print(f"\n  Selected: {selected[1]}")
                return selected[1]
        except ValueError:
            pass
        print(f"  Please enter 0-{len(windows)}")


def capture_worker(frame_q: Queue, stop: Event, cfg: Config):
    interval = 1.0 / cfg.capture_fps
    if cfg.window_title == "__FULLSCREEN__":
        _capture_mss_fullscreen(frame_q, stop, cfg, interval)
        return
    use_win32 = False
    try:
        import win32gui, win32ui, win32con
        use_win32 = True
    except ImportError:
        pass
    if use_win32:
        _capture_win32(frame_q, stop, cfg, interval)
    else:
        _capture_mss(frame_q, stop, cfg, interval)


def _find_hwnd(title: str):
    import win32gui
    result = []
    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            if title.lower() in win32gui.GetWindowText(hwnd).lower():
                result.append(hwnd)
        return True
    win32gui.EnumWindows(cb, None)
    return result[0] if result else None


def _capture_win32(frame_q, stop, cfg, interval):
    import win32gui, win32ui, win32con
    logging.info("[CAPTURE] win32 mode — searching '%s'", cfg.window_title)
    hwnd = None
    while not stop.is_set() and hwnd is None:
        hwnd = _find_hwnd(cfg.window_title)
        if hwnd is None:
            time.sleep(2)
    if stop.is_set():
        return

    recheck = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        recheck += 1
        if recheck % 100 == 0:
            new = _find_hwnd(cfg.window_title)
            if new:
                hwnd = new
        try:
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            w = right - left
            h = bottom - top
            if w <= 0 or h <= 0:
                time.sleep(0.1)
                continue
            hwndDC    = win32gui.GetDC(hwnd)
            mfcDC     = win32ui.CreateDCFromHandle(hwndDC)
            saveDC    = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBitMap)
            import ctypes
            ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)
            bmpinfo = saveBitMap.GetInfo()
            bmpstr  = saveBitMap.GetBitmapBits(True)
            frame   = np.frombuffer(bmpstr, dtype=np.uint8)
            frame   = frame.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))
            frame   = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            fh, fw  = frame.shape[:2]
            if (fw, fh) != (cfg.source_w, cfg.source_h):
                frame = cv2.resize(frame, (cfg.source_w, cfg.source_h),
                                   interpolation=cv2.INTER_AREA)
            if not frame_q.full():
                frame_q.put(frame)
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC(); mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)
        except Exception as e:
            logging.debug("[CAPTURE] %s", e)
            time.sleep(0.1)
            continue
        sleep_t = interval - (time.perf_counter() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)
    logging.info("[CAPTURE] stopped")


def _capture_mss(frame_q, stop, cfg, interval):
    import mss
    try:
        import pygetwindow as gw
    except ImportError:
        gw = None
    sct  = mss.mss()
    def _rect():
        if gw:
            wins = gw.getWindowsWithTitle(cfg.window_title)
            if wins:
                w = wins[0]
                return {"left": w.left, "top": w.top,
                        "width": w.width, "height": w.height}
        mon = sct.monitors[1]
        return {"left": mon["left"], "top": mon["top"],
                "width": cfg.source_w, "height": cfg.source_h}
    rect    = _rect()
    recheck = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        recheck += 1
        if recheck % 50 == 0:
            rect = _rect()
        img   = np.array(sct.grab(rect))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w  = frame.shape[:2]
        if (w, h) != (cfg.source_w, cfg.source_h):
            frame = cv2.resize(frame, (cfg.source_w, cfg.source_h),
                               interpolation=cv2.INTER_AREA)
        if not frame_q.full():
            frame_q.put(frame)
        sleep_t = interval - (time.perf_counter() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)


def _capture_mss_fullscreen(frame_q, stop, cfg, interval):
    import mss
    sct = mss.mss()
    mon = sct.monitors[1]
    while not stop.is_set():
        t0    = time.perf_counter()
        img   = np.array(sct.grab(mon))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        h, w  = frame.shape[:2]
        if (w, h) != (cfg.source_w, cfg.source_h):
            frame = cv2.resize(frame, (cfg.source_w, cfg.source_h),
                               interpolation=cv2.INTER_AREA)
        if not frame_q.full():
            frame_q.put(frame)
        sleep_t = interval - (time.perf_counter() - t0)
        if sleep_t > 0:
            time.sleep(sleep_t)


def preprocess_worker(frame_q: Queue, gray_q: Queue, stop: Event, cfg: Config):
    logging.info("[PREPROCESS] starting (%dx%d → %dx%d)",
                 cfg.source_w, cfg.source_h, cfg.proc_w, cfg.proc_h)
    logo = MovingLogoDetector(cfg)

    while not stop.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except Empty:
            continue

        crop = frame[cfg.focus_y: cfg.focus_y + cfg.focus_h,
                     cfg.focus_x: cfg.focus_x + cfg.focus_w]
        if crop.size == 0:
            continue

        mx1 = max(0, min(cfg.mask_x, crop.shape[1] - 1))
        my1 = max(0, min(cfg.mask_y, crop.shape[0] - 1))
        mw1 = max(1, min(cfg.mask_w, crop.shape[1] - mx1))
        mh1 = max(1, min(cfg.mask_h, crop.shape[0] - my1))
        cv2.rectangle(crop, (mx1, my1), (mx1+mw1, my1+mh1),
                      cfg.mask_color, thickness=-1)

        if cfg.mask2_enabled:
            mx2 = max(0, min(cfg.mask2_x, crop.shape[1] - 1))
            my2 = max(0, min(cfg.mask2_y, crop.shape[0] - 1))
            mw2 = max(1, min(cfg.mask2_w, crop.shape[1] - mx2))
            mh2 = max(1, min(cfg.mask2_h, crop.shape[0] - my2))
            cv2.rectangle(crop, (mx2, my2), (mx2+mw2, my2+mh2),
                          cfg.mask_color, thickness=-1)

        crop = logo.process(crop)

        if (crop.shape[1], crop.shape[0]) != (cfg.proc_w, cfg.proc_h):
            crop = cv2.resize(crop, (cfg.proc_w, cfg.proc_h),
                              interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if not gray_q.full():
            gray_q.put((time.time(), gray, crop))

    logging.info("[PREPROCESS] stopped")


def _phash_one(args):
    ts, gray_bytes, shape, hash_size, hf = args
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(shape)
    h    = compute_phash_bytes(gray, hash_size, hf)
    return (ts, h)


def phash_pool_worker(gray_q: Queue, hash_q: Queue, stop: Event, cfg: Config):
    from concurrent.futures import ThreadPoolExecutor
    logging.info("[PHASH] starting with %d threads", cfg.phash_cores)
    executor = ThreadPoolExecutor(max_workers=cfg.phash_cores)
    batch    = []

    while not stop.is_set():
        try:
            ts, gray, crop = gray_q.get(timeout=0.2)
            batch.append((ts, gray.tobytes(), gray.shape,
                          cfg.phash_size, cfg.highfreq_factor, crop))
        except Empty:
            pass

        if batch:
            futures = []
            for item in batch:
                args = item[:5]
                futures.append((executor.submit(_phash_one, args), item[5]))
            for fut, crop in futures:
                try:
                    ts, h = fut.result()
                    if not hash_q.full():
                        hash_q.put((ts, h, crop))
                except Exception as e:
                    logging.warning("[PHASH] %s", e)
            batch.clear()

    executor.shutdown(wait=False)
    logging.info("[PHASH] stopped")


def match_worker(hash_q: Queue, result_q: Queue, stop: Event, cfg: Config):

    # ── Parallel startup: PKL + verify DBs load simultaneously ───────────────
    db, store, verifiers = parallel_load_all(cfg)

    scene_verifier  = verifiers.get("scene")
    motion_verifier = verifiers.get("motion")
    zone_verifier   = verifiers.get("zones")

    seq_verifier   = SequenceVerifier(sequence_len=5, max_gap_sec=2.0)

    def do_query(phash_bytes: bytes):
        # VP-tree first (fastest), numpy fallback
        if db.vptree:
            return db.vptree.query(phash_bytes, cfg.hamming_threshold)
        return store.query_numpy(phash_bytes)

    logging.info("[MATCH] ready — %d hashes", len(store))

    while not stop.is_set():
        try:
            ts, phash_bytes, crop = hash_q.get(timeout=0.5)
        except Empty:
            continue

        t0   = time.perf_counter()
        best = do_query(phash_bytes)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if motion_verifier:
            motion_verifier.add_live_frame(gray)

        if zone_verifier:
            zone_verifier.add_live_frame(crop)

        if not best:
            logging.debug("[MATCH] no match (%.1f ms)", elapsed_ms)
            continue

        dist, label    = best
        video_name, video_sec = parse_label(label)

        verify_results = {}

        # Step 1 — pHash
        verify_results["phash"] = "VERIFIED" if dist <= cfg.hamming_threshold else "WEAK"

        # Step 2 — Sequence order
        seq_result = seq_verifier.add_match(video_name, video_sec or 0, dist)
        verify_results["sequence"] = seq_result.get("status", "UNKNOWN")

        # Step 3 — Scene cut pattern
        if scene_verifier:
            cut_result = scene_verifier.add_live_frame(gray, ts)
            verify_results["scene_cuts"] = (
                cut_result.get("status", "UNKNOWN") if cut_result else "ACCUMULATING_CUTS"
            )
        else:
            verify_results["scene_cuts"] = "NO_DB"

        # Step 4 — Motion pattern
        if motion_verifier and video_sec is not None:
            m_result = motion_verifier.verify(video_name, video_sec)
            verify_results["motion"] = m_result.get("status", "UNKNOWN")
        else:
            verify_results["motion"] = "NO_DB"

        # Step 5 — Spatial zones
        if zone_verifier and video_sec is not None:
            z_result = zone_verifier.verify(video_name, video_sec)
            verify_results["zones"] = z_result.get("status", "UNKNOWN")
        else:
            verify_results["zones"] = "NO_DB"

        confidence = compute_confidence(verify_results)

        parts      = label.split("|")
        file_name  = parts[0] if len(parts) >= 1 else label
        video_time = parts[1] if len(parts) >= 2 else "unknown"
        similarity = round((1 - dist / 256) * 100, 1)

        if dist <= 5:   match_conf = "EXACT"
        elif dist <= 8: match_conf = "STRONG"
        elif dist <= 12:match_conf = "GOOD"
        else:           match_conf = "WEAK"

        result = {
            "detected_at":     time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime(ts)),
            "file_name":       file_name,
            "video_timestamp": video_time,
            "similarity":      f"{similarity}%",
            "match_confidence":match_conf,
            "hamming_distance":dist,
            "lookup_ms":       round(elapsed_ms, 1),
            "verify_score":    confidence["score"],
            "verify_verdict":  confidence["verdict"],
            "verify_passed":   confidence["passed"],
            "verify_failed":   confidence["failed"],
            "tier1_passed":    confidence["tier1_passed"],
            "verify_details":  verify_results,
        }

        result_q.put(result)
        logging.info("[MATCH] ✓ %s @ %s  dist=%d  verify=%s (%d/100)",
                     file_name, video_time, dist,
                     confidence["verdict"], confidence["score"])

    logging.info("[MATCH] stopped")


def result_consumer(result_q: Queue, stop: Event, cfg: Config):
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.result_dir) / "matches.jsonl"

    print()
    print("=" * 66)
    print("   LIVE FRAME MATCHER v4 — Full Verification Pipeline")
    print(f"   Capture: {cfg.source_w}×{cfg.source_h} → proc: {cfg.proc_w}×{cfg.proc_h}")
    print(f"   Threshold: {cfg.hamming_threshold} | Results: {log_path}")
    print("=" * 66)
    print()

    with open(log_path, "a", encoding="utf-8") as fout:
        while not stop.is_set():
            try:
                r = result_q.get(timeout=1.0)
            except Empty:
                continue

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            fout.flush()

            verdict = r["verify_verdict"]
            score   = r["verify_score"]
            tier1   = r["tier1_passed"]

            print("┌─────────────────────────────────────────────────────────┐")
            print(f"│  ★  MATCH FOUND                                        │")
            print("├─────────────────────────────────────────────────────────┤")
            print(f"│  File:        {r['file_name']:<43}│")
            print(f"│  Time:        {r['video_timestamp']:<43}│")
            print(f"│  Similarity:  {r['similarity']:<43}│")
            print(f"│  Match conf:  {r['match_confidence']:<43}│")
            print(f"│  Detected at: {r['detected_at']:<43}│")
            print(f"│  Lookup:      {r['lookup_ms']} ms{'':<38}│")
            print("├─────────────────────────────────────────────────────────┤")
            print(f"│  VERIFICATION SCORE:  {score}/100  →  {verdict:<25}│")
            print(f"│  Tier 1 passed:       {'YES ✓' if tier1 else 'NO ✗':<43}│")
            print("├─────────────────────────────────────────────────────────┤")
            for p in r["verify_passed"]:
                print(f"│  {p:<57}│")
            for f in r["verify_failed"]:
                print(f"│  {f:<57}│")
            print("└─────────────────────────────────────────────────────────┘")
            print()


# ===========================================================================
# ORCHESTRATOR
# ===========================================================================

def run_pipeline(cfg: Config):
    logging.basicConfig(level=getattr(logging, cfg.log_level),
                        format="%(asctime)s  %(message)s")
    stop    = Event()
    frame_q = Queue(maxsize=30)
    gray_q  = Queue(maxsize=30)
    hash_q  = Queue(maxsize=60)
    result_q= Queue(maxsize=200)

    workers = [
        Process(target=capture_worker,    args=(frame_q, stop, cfg),
                name="capture",    daemon=True),
        Process(target=preprocess_worker, args=(frame_q, gray_q, stop, cfg),
                name="preprocess", daemon=True),
        Process(target=phash_pool_worker, args=(gray_q, hash_q, stop, cfg),
                name="phash",      daemon=True),
        Process(target=match_worker,      args=(hash_q, result_q, stop, cfg),
                name="match",      daemon=True),
    ]

    for w in workers:
        w.start()
        logging.info("Started %s (pid=%d)", w.name, w.pid)

    try:
        result_consumer(result_q, stop, cfg)
    except KeyboardInterrupt:
        logging.info("Ctrl-C — shutting down …")
        stop.set()
        for w in workers:
            w.join(timeout=3)
        logging.info("All workers stopped.")


# ===========================================================================
# BUILD VERIFICATION DATABASES
# ===========================================================================

def build_verify_dbs(video_dir: str, verify_db_dir: str, sample_fps: int = 5):
    """
    Run ONCE on your video files to build all verification databases.
    Creates in verify_db_dir:
        scene_cuts.json   — scene cut timing pattern per video
        motion.json       — motion energy profile per video
        zones.json        — spatial zone brightness profile per video
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    Path(verify_db_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  BUILDING VERIFICATION DATABASES")
    print(f"  Source:  {video_dir}")
    print(f"  Output:  {verify_db_dir}")
    print("=" * 60)

    print("\n  [1/3] Scene cut timing patterns ...")
    sc = SceneCutTimeline(str(Path(verify_db_dir) / "scene_cuts.json"))
    sc.build(video_dir, sample_fps=sample_fps)

    print("\n  [2/3] Motion energy profiles ...")
    mp_ = MotionPatternVerifier(str(Path(verify_db_dir) / "motion.json"))
    mp_.build(video_dir)

    print("\n  [3/3] Spatial zone brightness profiles ...")
    sz = SpatialZoneVerifier(str(Path(verify_db_dir) / "zones.json"))
    sz.build(video_dir)

    print("\n" + "=" * 60)
    print("  ALL VERIFICATION DATABASES BUILT")
    print(f"  Files in {verify_db_dir}:")
    for f in Path(verify_db_dir).glob("*.json"):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name:<25} {size_mb:.2f} MB")
    print("=" * 60 + "\n")


def import_json_to_db(json_dir: str, db_path: str):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    db         = HashDB(db_path)
    db.load()
    json_files = sorted(Path(json_dir).glob("*_hashes.json"))
    if not json_files:
        logging.warning("No *_hashes.json files found in %s", json_dir)
        return
    total = sum(db.import_json(str(jf)) for jf in json_files)
    db.save()
    logging.info("Imported %d total hashes. DB now has %d entries.",
                 total, len(db.hashes))


def build_db_from_videos(video_dir: str, db_path: str,
                         sample_fps: int = 1,
                         hash_size: int = 16, hf: int = 4):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    db   = HashDB(db_path)
    db.load()
    exts = {".mp4", ".avi", ".mkv", ".ts", ".flv", ".webm"}
    files= [f for f in Path(video_dir).rglob("*")
            if f.suffix.lower() in exts]
    logging.info("Found %d video files", len(files))

    for vf in files:
        cap      = cv2.VideoCapture(str(vf))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / sample_fps))
        idx      = 0
        sampled  = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h    = compute_phash_bytes(gray, hash_size, hf)
                sec  = idx / fps
                ts   = "{:02d}:{:02d}:{:02d}".format(
                    int(sec//3600), int((sec%3600)//60), int(sec%60))
                db.add(h, f"{vf.name}|{ts}|t{sec}")
                sampled += 1
            idx += 1
        cap.release()
        logging.info("  indexed %s (%d frames)", vf.name, sampled)

    db.save()
    logging.info("Database now has %d entries", len(db.hashes))


# ===========================================================================
# BENCHMARK + TEST MATCH
# ===========================================================================

def _run_benchmark(db_path: str, n_queries: int = 100):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    print("\n" + "=" * 60)
    print("  BENCHMARK — Live Frame Matcher v4")
    print("=" * 60)

    db = HashDB(db_path, threshold=12)
    t0 = time.perf_counter()
    db.load()
    print(f"\n  Database: {len(db.hashes)} hashes  "
          f"(loaded in {(time.perf_counter()-t0)*1000:.0f} ms)")

    if not db.hashes:
        print("  ERROR: Database empty!")
        return

    print(f"\n  [1/2] Numpy fallback ({n_queries} queries)...")
    np_times = []
    for _ in range(n_queries):
        idx    = np.random.randint(len(db.hashes))
        target = db.hashes[idx]
        t0 = time.perf_counter()
        db.query(target)
        np_times.append((time.perf_counter() - t0) * 1000)
    np_times = np.array(np_times)
    print(f"    Mean: {np_times.mean():.2f} ms  P95: {np.percentile(np_times,95):.2f} ms")

    t0 = time.perf_counter()
    db.build_index()
    print(f"\n  VP-tree: {(time.perf_counter()-t0)*1000:.0f} ms build")

    print(f"\n  [2/2] VP-tree ({n_queries} queries)...")
    vp_times = []
    for _ in range(n_queries):
        idx    = np.random.randint(len(db.hashes))
        target = db.hashes[idx]
        t0 = time.perf_counter()
        db.query(target)
        vp_times.append((time.perf_counter() - t0) * 1000)
    vp_times = np.array(vp_times)
    print(f"    Mean: {vp_times.mean():.2f} ms  P95: {np.percentile(vp_times,95):.2f} ms")

    ratio = np_times.mean() / vp_times.mean() if vp_times.mean() > 0 else 0
    print(f"\n  VP-tree is {ratio:.0f}x faster than numpy fallback")
    print("=" * 60 + "\n")


def _run_test_match(db_path: str, count: int = 10, threshold: int = 12):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
    print("\n" + "=" * 60)
    print("  TEST MATCH — Sample DB lookups")
    print("=" * 60)

    db = HashDB(db_path, threshold=threshold)
    db.load()
    db.build_index()

    if not db.hashes:
        print("  ERROR: Database empty!")
        return

    video_names = Counter(label.split("|")[0] for label in db.labels)
    print(f"\n  Database: {len(db.hashes)} hashes from {len(video_names)} videos")
    for name, count_ in sorted(video_names.items()):
        print(f"    {name:<35} {count_:>6} hashes")

    print(f"\n  Running {count} sample matches...\n")

    for i in range(count):
        idx    = np.random.randint(len(db.hashes))
        target = db.hashes[idx]
        t0     = time.perf_counter()
        result = db.query(target)
        ms     = (time.perf_counter() - t0) * 1000

        if result:
            dist, label = result
            parts = label.split("|")
            sim   = round((1 - dist / 256) * 100, 1)
            conf  = ("EXACT" if dist <= 5 else
                     "STRONG" if dist <= 8 else
                     "GOOD"   if dist <= 12 else "WEAK")
            print(f"  [{i+1:2d}] {parts[0]:<30} @ {parts[1] if len(parts)>1 else '?'}"
                  f"  dist={dist}  {conf}  {sim}%  ({ms:.2f} ms)")
        else:
            print(f"  [{i+1:2d}] No match  ({ms:.2f} ms)")

    print("\n" + "=" * 60 + "\n")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Live Frame Matcher v4 — 6-step verification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples:
  # Step 1 — Build pHash database from video files
  python live_frame_matcher_optimized_4.py build-db C:\videos --db C:\hash_db\phash_database.pkl

  # Step 2 — OR import JSON hashes from video_preprocess.py
  python live_frame_matcher_optimized_4.py import-json C:\hash_db --db C:\hash_db\phash_database.pkl

  # Step 3 — Build all verification databases (scene cuts, motion, zones)
  python live_frame_matcher_optimized_4.py build-verify-db --video-dir C:\videos

  # Step 4 — Run live matching with full verification
  python live_frame_matcher_optimized_4.py run --db C:\hash_db\phash_database.pkl

  # Benchmark
  python live_frame_matcher_optimized_4.py benchmark --db C:\hash_db\phash_database.pkl

  # Test match
  python live_frame_matcher_optimized_4.py test-match --db C:\hash_db\phash_database.pkl
        """
    )

    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Start live matching + verification")
    p_run.add_argument("--window",     default="__pick__")
    p_run.add_argument("--db",         default=r"C:\hash_db\phash_database.pkl")
    p_run.add_argument("--verify-dir", default=r"C:\hash_db\verify")
    p_run.add_argument("--fps",        type=int, default=5)
    p_run.add_argument("--threshold",  type=int, default=12)
    p_run.add_argument("--result-dir", default=r"C:\hash_db\results")

    p_build = sub.add_parser("build-db", help="Build pHash DB from videos")
    p_build.add_argument("video_dir")
    p_build.add_argument("--db",         default=r"C:\hash_db\phash_database.pkl")
    p_build.add_argument("--sample-fps", type=int, default=1)

    p_import = sub.add_parser("import-json", help="Import JSON hashes from video_preprocess.py")
    p_import.add_argument("json_dir")
    p_import.add_argument("--db", default=r"C:\hash_db\phash_database.pkl")

    p_verify = sub.add_parser("build-verify-db",
                               help="Build scene-cut, motion, zone databases")
    p_verify.add_argument("--video-dir",  required=True)
    p_verify.add_argument("--verify-dir", default=r"C:\hash_db\verify")
    p_verify.add_argument("--sample-fps", type=int, default=5)

    p_bench = sub.add_parser("benchmark")
    p_bench.add_argument("--db",      default=r"C:\hash_db\phash_database.pkl")
    p_bench.add_argument("--queries", type=int, default=100)

    p_test = sub.add_parser("test-match")
    p_test.add_argument("--db",        default=r"C:\hash_db\phash_database.pkl")
    p_test.add_argument("--count",     type=int, default=10)
    p_test.add_argument("--threshold", type=int, default=12)

    args = parser.parse_args()

    if args.cmd == "build-db":
        build_db_from_videos(args.video_dir, args.db, args.sample_fps)

    elif args.cmd == "import-json":
        import_json_to_db(args.json_dir, args.db)

    elif args.cmd == "build-verify-db":
        build_verify_dbs(args.video_dir, args.verify_dir, args.sample_fps)

    elif args.cmd == "benchmark":
        _run_benchmark(args.db, args.queries)

    elif args.cmd == "test-match":
        _run_test_match(args.db, args.count, args.threshold)

    elif args.cmd == "run":
        window_title = args.window
        if window_title == "__pick__":
            window_title = pick_window()
            print()

        cfg = Config(
            window_title      = window_title,
            hash_db_path      = args.db,
            verify_db_dir     = args.verify_dir,
            capture_fps       = args.fps,
            hamming_threshold = args.threshold,
            result_dir        = args.result_dir,
        )

        print("\n" + "=" * 66)
        print("  LIVE FRAME MATCHER v4  (1920×784)")
        print("=" * 66)
        print(f"  Window:       {cfg.window_title}")
        print(f"  Hash DB:      {cfg.hash_db_path}")
        print(f"  Verify DB:    {cfg.verify_db_dir}")
        print(f"  Capture FPS:  {cfg.capture_fps}")
        print(f"  Threshold:    {cfg.hamming_threshold}")
        print(f"  Results:      {cfg.result_dir}")
        print("=" * 66)
        print()
        print("  VERIFICATION TIERS:")
        print("  TIER 1 (must pass): pHash + Sequence + Scene cuts")
        print("  TIER 2 (raises confidence): Motion + Zones")
        print("  STARTUP: PKL + verify DBs load in parallel (threads)")
        print("  QUERY:   VP-tree → numpy fallback (auto)")
        print("=" * 66 + "\n")

        run_pipeline(cfg)

    else:
        parser.print_help()


if __name__ == "__main__":
    mp.freeze_support()
    main()
