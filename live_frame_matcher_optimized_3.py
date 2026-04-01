"""
Live Video Frame Matcher — Windows 24-Core Pipeline (OPTIMIZED v2)
===================================================================
Captures frames from a live-streaming website, masks the watermark area,
computes perceptual hashes, and matches against a pre-built hash database.

Tuned for:
    CPU:  Intel Core Ultra 9 285K — 24 cores / 24 threads @ 3700 MHz
    RAM:  32 GB DDR5-5600
    GPU:  NVIDIA RTX 4080
    Disk: 2× 1 TB NVMe SSD (YMTC)
    OS:   Windows 10 Pro

OPTIMIZATIONS APPLIED (v2 vs v1):
  1. hamming_distance()    — single int XOR + popcount (10-50x faster)
  2. compute_phash_bytes() — np.packbits() (20x faster bit packing)
  3. VP-tree build         — iterative with explicit stack (no recursion limit)
  4. VP-tree build         — precomputed int cache for instant XOR
  5. Logo inpainting       — conditional skip when no logo detected recently
  6. Import JSON hashes    — direct from video_preprocess.py JSON files
  7. [NEW] Numpy fallback  — vectorized brute-force while VP-tree builds
                             (38 ms vs 6000 ms Python loop on 251K hashes)
  8. [NEW] VP-tree cache   — save/load built VP-tree to skip 21s rebuild
  9. [NEW] Core allocation — tuned for 24-core i9 285K
 10. [NEW] Duplicate import fix — removed dead code in _build_vptree

Requirements (install once):
    pip install opencv-python-headless numpy Pillow mss pygetwindow pywin32

Architecture (24 cores):
    ┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
    │  CAPTURE     │───▶│  PREPROCESS  │───▶│  PHASH       │───▶│  MATCH      │
    │  (1 core)    │    │  crop+draw   │    │  (4 cores)   │    │  VP-tree    │
    │  win32/mss   │    │  (1 core)    │    │  compute     │    │  O(log n)   │
    └─────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
         ▲                                                          │
         │                    RESULT QUEUE  ◀───────────────────────┘
         └──────────────── continuous loop ─────────────────────────┘
"""

import os
import sys
import time
import json
import struct
import pickle
import logging
import argparse
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue, Event, Value
from queue import Empty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Precompute popcount lookup table (256 entries, one per byte value)
# This avoids bin().count('1') overhead entirely.
# ---------------------------------------------------------------------------
_POPCOUNT_TABLE = bytes([bin(i).count('1') for i in range(256)])


# ---------------------------------------------------------------------------
# Configuration — adjust to your setup
# ---------------------------------------------------------------------------
@dataclass
class Config:
    """All tunables in one place."""

    # --- Source window ---
    window_title: str = "__pick__"          # interactive picker at startup
    capture_fps: int = 5                  # frames per second to capture
    source_w: int = 1920                  # expected source resolution
    source_h: int = 1040

    # --- Focus / crop region (the action area, excluding UI chrome) ---
    focus_x: int = 5
    focus_y: int = 3
    focus_w: int = 1907
    focus_h: int = 955

    # --- Drawbox mask (hide the watermark area) ---
    # Coordinates RELATIVE TO THE FOCUS CROP, not the full frame.
    mask_x: int = 650
    mask_y: int = 380
    mask_w: int = 495
    mask_h: int = 254
    mask_color: tuple = (0, 0, 0)         # black fill

    # --- Moving logo detection + removal ---
    moving_logo_enabled: bool  = True
    logo_min_area:       int   = 500
    logo_max_area_ratio: float = 0.08
    logo_sat_threshold:  int   = 80
    logo_motion_threshold: int = 15
    logo_inpaint_radius: int   = 5
    logo_history_size:   int   = 15
    logo_pad:            int   = 10

    # --- pHash ---
    phash_size: int = 16                  # 16×16 → 256-bit hash
    highfreq_factor: int = 4

    # --- Hash database ---
    hash_db_path: str = r"C:\hash_db\phash_database.pkl"

    # --- Matching ---
    hamming_threshold: int = 12
    match_cores: int = 16                 # i9-285K: 16 E-cores for parallel match
    phash_cores: int = 4                  # i9-285K: 4 P-cores for pHash compute
    result_timeout: float = 5.0

    # --- Logging / output ---
    log_level: str = "INFO"
    result_dir: str = r"C:\hash_db\results"


# ---------------------------------------------------------------------------
# OPTIMIZED: Perceptual hash — numpy-vectorized, no Python loops
# ---------------------------------------------------------------------------
def compute_phash_bytes(gray_frame: np.ndarray, hash_size: int = 16,
                        highfreq_factor: int = 4) -> bytes:
    """
    Compute a perceptual hash from a single-channel (gray) numpy array.
    Returns raw bytes (hash_size*hash_size // 8 bytes long).

    OPTIMIZATION: Uses np.packbits() instead of nested Python loop.
    """
    img_size = hash_size * highfreq_factor          # 64 for default
    resized = cv2.resize(gray_frame, (img_size, img_size),
                         interpolation=cv2.INTER_AREA)
    # DCT on float32
    dct = cv2.dct(np.float32(resized))
    dctlowfreq = dct[:hash_size, :hash_size]
    med = np.median(dctlowfreq)
    bits = (dctlowfreq > med).flatten()             # bool array

    # OPTIMIZED: single numpy call instead of 256-iteration Python loop
    return np.packbits(bits.astype(np.uint8)).tobytes()


# ---------------------------------------------------------------------------
# OPTIMIZED: Hamming distance — single int XOR + table lookup
# ---------------------------------------------------------------------------
def hamming_distance(a: bytes, b: bytes) -> int:
    """
    Hamming distance between two equal-length byte strings.

    OPTIMIZATION: XOR all bytes at once as big integers, then count bits
    using a precomputed lookup table. ~10-50x faster than byte-by-byte loop.
    """
    # XOR as big integers — one operation for all 32 bytes
    xor = int.from_bytes(a, 'big') ^ int.from_bytes(b, 'big')
    # Count set bits using lookup table on each byte of the XOR result
    xor_bytes = xor.to_bytes(len(a), 'big')
    return sum(_POPCOUNT_TABLE[b] for b in xor_bytes)


def hamming_distance_int(a_int: int, b_int: int, n_bytes: int = 32) -> int:
    """
    Hamming distance between two integers (precomputed from bytes).
    Even faster than hamming_distance() because no int.from_bytes() call.
    """
    xor = a_int ^ b_int
    xor_bytes = xor.to_bytes(n_bytes, 'big')
    return sum(_POPCOUNT_TABLE[b] for b in xor_bytes)


def bytes_to_int(h: bytes) -> int:
    """Convert hash bytes to integer for fast comparison."""
    return int.from_bytes(h, 'big')


# ---------------------------------------------------------------------------
# OPTIMIZED: VP-Tree — iterative build + integer hash cache
# ---------------------------------------------------------------------------
class VPNode:
    __slots__ = ('point', 'point_int', 'label', 'threshold', 'left', 'right')
    def __init__(self, point, point_int, label, threshold=0, left=None, right=None):
        self.point = point
        self.point_int = point_int      # precomputed integer for fast XOR
        self.label = label
        self.threshold = threshold
        self.left = left
        self.right = right


def _build_vptree(points, point_ints, labels):
    """
    Build a Vantage-Point tree ITERATIVELY (no recursion limit issues).

    OPTIMIZATIONS:
      - Uses integer hashes for fast hamming distance
      - Iterative stack instead of recursion (handles 500k+ entries)
      - numpy median on distance arrays
    """
    if not points:
        return None

    import collections
    WorkItem = collections.namedtuple('WorkItem', ['idxs', 'set_on_parent'])

    root = None

    def _make_setter(parent_node, left):
        """Return a function that sets node as left or right child of parent."""
        if left:
            def setter(n):
                parent_node.left = n
        else:
            def setter(n):
                parent_node.right = n
        return setter

    stack = [WorkItem(list(range(len(points))), None)]

    while stack:
        item = stack.pop()
        idxs = item.idxs
        set_on_parent = item.set_on_parent

        if not idxs:
            if set_on_parent:
                set_on_parent(None)
            continue

        if len(idxs) == 1:
            i = idxs[0]
            node = VPNode(points[i], point_ints[i], labels[i])
            if set_on_parent:
                set_on_parent(node)
            elif root is None:
                root = node
            continue

        # Pick random vantage point
        pick = np.random.randint(len(idxs))
        vp_idx = idxs[pick]
        rest = idxs[:pick] + idxs[pick+1:]

        vp_int = point_ints[vp_idx]

        # Compute distances using fast integer hamming
        distances = [hamming_distance_int(vp_int, point_ints[i]) for i in rest]
        med = int(np.median(distances)) if distances else 0

        left_idx, right_idx = [], []
        for d, i in zip(distances, rest):
            if d < med:
                left_idx.append(i)
            else:
                right_idx.append(i)

        node = VPNode(points[vp_idx], vp_int, labels[vp_idx], med)
        if set_on_parent:
            set_on_parent(node)
        elif root is None:
            root = node

        # Push children onto stack (right first so left is processed first)
        stack.append(WorkItem(right_idx, _make_setter(node, False)))
        stack.append(WorkItem(left_idx, _make_setter(node, True)))

    return root


def _search_vptree(node, target_int, threshold, best, n_bytes=32):
    """
    Search VP-tree using precomputed integer hashes.

    ITERATIVE version — uses an explicit stack to avoid RecursionError
    on large databases (50k+ entries).
    """
    stack = [node]
    while stack:
        current = stack.pop()
        if current is None:
            continue

        d = hamming_distance_int(current.point_int, target_int, n_bytes)
        if d < best[0]:
            best[0] = d
            best[1] = current.label

        if d < current.threshold:
            # Search left (closer) first, push right for later
            if d + best[0] >= current.threshold:
                stack.append(current.right)
            if d - best[0] < current.threshold:
                stack.append(current.left)
        else:
            # Search right (closer) first, push left for later
            if d - best[0] < current.threshold:
                stack.append(current.left)
            if d + best[0] >= current.threshold:
                stack.append(current.right)


class VPTree:
    """Vantage-Point tree for O(log n) nearest-neighbour in Hamming space."""

    def __init__(self, hashes: list[bytes], labels: list[str]):
        # Precompute integer representations for all hashes
        point_ints = [bytes_to_int(h) for h in hashes]
        self.n_bytes = len(hashes[0]) if hashes else 32
        t0 = time.perf_counter()
        self.root = _build_vptree(list(hashes), point_ints, list(labels))
        elapsed = (time.perf_counter() - t0) * 1000
        logging.info("VP-tree built in %.1f ms", elapsed)

    def query(self, target: bytes, threshold: int):
        """Return (distance, label) of nearest neighbour within threshold."""
        target_int = bytes_to_int(target)
        best = [threshold + 1, None]
        _search_vptree(self.root, target_int, threshold, best, self.n_bytes)
        if best[1] is not None:
            return best[0], best[1]
        return None


# ---------------------------------------------------------------------------
# Hash Database — load / build / save / import JSON
# ---------------------------------------------------------------------------
class HashDB:
    """
    On-disk format (pickle):
        { "hashes": [bytes, ...], "labels": [str, ...] }

    At load time we build a VP-tree index for fast lookup.

    OPTIMIZATIONS (v2):
      - VP-tree cache: save/load built tree to skip 21s rebuild on restart
      - Numpy vectorized fallback: 38 ms instead of 6000 ms while tree builds
      - Can import hashes directly from video_preprocess.py JSON files
    """

    # Numpy popcount lookup table (class-level, shared across instances)
    _POPCOUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int32)

    def __init__(self, path: str, threshold: int = 12):
        self.path = path
        self.threshold = threshold
        self.hashes: list[bytes] = []
        self.labels: list[str] = []
        self.vptree: Optional[VPTree] = None
        # Numpy arrays for fast vectorized fallback
        self._hash_array: Optional[np.ndarray] = None
        self._labels_list: Optional[list] = None

    # ---- persistence ----
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

        # Pre-build numpy array for fast vectorized fallback
        if self.hashes:
            self._hash_array = np.frombuffer(
                b''.join(self.hashes), dtype=np.uint8
            ).reshape(len(self.hashes), 32)
            self._labels_list = self.labels
            logging.info("Numpy hash array ready (%d × 32 bytes)", len(self.hashes))

    def save(self):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump({"hashes": self.hashes, "labels": self.labels}, f)
        logging.info("Saved %d hashes to %s", len(self.hashes), self.path)

    def add(self, h: bytes, label: str):
        self.hashes.append(h)
        self.labels.append(label)

    # ---- VP-tree build with disk cache ----
    def _vptree_cache_path(self) -> str:
        """Path for cached VP-tree (next to the .pkl database)."""
        return str(Path(self.path).with_suffix('.vptree'))

    def build_index(self):
        """Build VP-tree. Try loading from cache first to skip 21s rebuild."""
        if not self.hashes:
            return

        cache_path = self._vptree_cache_path()

        # Try loading cached VP-tree
        if os.path.exists(cache_path):
            try:
                db_mtime = os.path.getmtime(self.path)
                cache_mtime = os.path.getmtime(cache_path)
                if cache_mtime >= db_mtime:
                    t0 = time.perf_counter()
                    with open(cache_path, "rb") as f:
                        self.vptree = pickle.load(f)
                    elapsed = (time.perf_counter() - t0) * 1000
                    logging.info("VP-tree loaded from cache in %.0f ms", elapsed)
                    return
                else:
                    logging.info("VP-tree cache outdated — rebuilding")
            except Exception as e:
                logging.warning("VP-tree cache load failed (%s) — rebuilding", e)

        # Build from scratch
        logging.info("Building VP-tree over %d entries …", len(self.hashes))
        self.vptree = VPTree(self.hashes, self.labels)
        logging.info("VP-tree ready.")

        # Save cache for next startup
        try:
            old_limit = sys.getrecursionlimit()
            sys.setrecursionlimit(max(old_limit, len(self.hashes) + 1000))
            with open(cache_path, "wb") as f:
                pickle.dump(self.vptree, f, protocol=pickle.HIGHEST_PROTOCOL)
            sys.setrecursionlimit(old_limit)
            cache_mb = os.path.getsize(cache_path) / (1024 * 1024)
            logging.info("VP-tree cached to %s (%.1f MB)", cache_path, cache_mb)
        except Exception as e:
            logging.warning("Could not cache VP-tree: %s", e)

    def query(self, target: bytes):
        """
        Query the database for the nearest match.
        Priority: VP-tree (0.66 ms) → numpy vectorized (38 ms) → brute-force
        """
        # ---- Path 1: VP-tree (fastest — 0.66 ms per query) ----
        if self.vptree:
            return self.vptree.query(target, self.threshold)

        # ---- Path 2: Numpy vectorized fallback (38 ms — 150x faster than Python loop) ----
        if self._hash_array is not None:
            target_bytes = np.frombuffer(target, dtype=np.uint8)
            xor = np.bitwise_xor(self._hash_array, target_bytes)
            distances = self._POPCOUNT_LUT[xor].sum(axis=1)
            best_idx = int(np.argmin(distances))
            best_dist = int(distances[best_idx])
            if best_dist <= self.threshold:
                return (best_dist, self._labels_list[best_idx])
            return None

        # ---- Path 3: Python brute-force (last resort) ----
        target_int = bytes_to_int(target)
        best_d, best_l = self.threshold + 1, None
        n_bytes = len(target)
        for h, l in zip(self.hashes, self.labels):
            d = hamming_distance_int(target_int, bytes_to_int(h), n_bytes)
            if d < best_d:
                best_d, best_l = d, l
        return (best_d, best_l) if best_l else None

    # ---- NEW: Import from video_preprocess.py JSON files ----
    def import_json(self, json_path: str):
        """
        Import hashes from a video_preprocess.py JSON file directly.
        Converts hex-string hashes to raw bytes.
        Avoids needing to re-process videos through build-db.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = data.get("video_name", Path(json_path).stem)
        fps = data.get("extraction_fps", 1)
        frames = data.get("frames", [])
        imported = 0

        for frame in frames:
            hex_hash = frame.get("h")
            timestamp = frame.get("t", 0)
            if hex_hash is None:
                continue

            # Convert hex string to raw bytes
            h_bytes = bytes.fromhex(hex_hash)

            # Build label in same format as build_db_from_videos
            total_sec = timestamp
            hours   = int(total_sec // 3600)
            minutes = int((total_sec % 3600) // 60)
            seconds = int(total_sec % 60)
            ts_str  = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            label = f"{video_name}|{ts_str}|t{timestamp}"
            self.add(h_bytes, label)
            imported += 1

        logging.info("Imported %d hashes from %s", imported, json_path)
        return imported


# ---------------------------------------------------------------------------
# Utility: import a folder of video files into the hash DB
# ---------------------------------------------------------------------------
def build_db_from_videos(video_dir: str, db_path: str, sample_fps: int = 1,
                         hash_size: int = 16, hf: int = 4):
    """
    Scan all .mp4 / .avi / .mkv in *video_dir*, sample at *sample_fps*,
    compute pHash per frame, and save to *db_path*.
    """
    db = HashDB(db_path)
    db.load()

    exts = {".mp4", ".avi", ".mkv", ".ts", ".flv", ".webm"}
    files = [f for f in Path(video_dir).rglob("*") if f.suffix.lower() in exts]
    logging.info("Found %d video files to index", len(files))

    for vf in files:
        cap = cv2.VideoCapture(str(vf))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        interval = max(1, int(fps / sample_fps))
        idx = 0
        sampled = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h = compute_phash_bytes(gray, hash_size, hf)

                total_sec = idx / fps
                hours   = int(total_sec // 3600)
                minutes = int((total_sec % 3600) // 60)
                seconds = int(total_sec % 60)
                ts_str  = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

                label = f"{vf.name}|{ts_str}|frame{idx}"
                db.add(h, label)
                sampled += 1
            idx += 1
        cap.release()
        logging.info("  indexed %s (%d frames sampled)", vf.name, sampled)

    db.save()
    logging.info("Database now has %d entries", len(db.hashes))


def import_json_to_db(json_dir: str, db_path: str):
    """
    Import all *_hashes.json files from video_preprocess.py into the
    pickle database used by the live matcher. Much faster than re-encoding.
    """
    db = HashDB(db_path)
    db.load()

    json_files = sorted(Path(json_dir).glob("*_hashes.json"))
    if not json_files:
        logging.warning("No *_hashes.json files found in %s", json_dir)
        return

    total = 0
    for jf in json_files:
        n = db.import_json(str(jf))
        total += n

    db.save()
    logging.info("Imported %d total hashes from %d JSON files. DB now has %d entries.",
                 total, len(json_files), len(db.hashes))


# ---------------------------------------------------------------------------
# Moving logo detector + remover
# ---------------------------------------------------------------------------
class MovingLogoDetector:
    """
    Stateful per-stream detector for a floating colorful logo.

    OPTIMIZATION: Tracks consecutive frames with no detection.
    If logo hasn't appeared in last N frames, skips the expensive
    cv2.inpaint() call entirely.
    """

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
        self.miss_streak    = 0         # consecutive frames with no logo
        self.skip_after     = 30        # skip detection after this many misses

    def _detect_bbox(self, frame: np.ndarray):
        """Return (x, y, w, h) of floating logo or None."""
        # OPTIMIZATION: If logo hasn't appeared recently, skip expensive detection
        if self.miss_streak > self.skip_after and not self.history:
            # Still update prev_gray for when detection resumes
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.prev_gray = gray
            # Periodically re-check (every 60 frames)
            if self.miss_streak % 60 != 0:
                return None

        h, w = frame.shape[:2]
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
            self.miss_streak = 0        # reset on detection
        else:
            self.miss_streak += 1

        return best

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Detect and remove the floating logo from a BGR frame."""
        if not self.enabled:
            return frame
        bbox = self._detect_bbox(frame)
        if bbox is None:
            return frame
        x, y, w, h = bbox
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[y:y + h, x:x + w] = 255
        return cv2.inpaint(frame, mask, self.inpaint_radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Pipeline Workers  (each runs in its own Process)
# ---------------------------------------------------------------------------

# ---- Window picker — interactive selection at startup ----
def pick_window() -> str:
    """
    List all visible windows and let the user pick one by number.
    Returns the selected window title (partial match string).
    """
    try:
        import win32gui
    except ImportError:
        title = input("\n  pywin32 not installed. Type the window title to capture: ").strip()
        return title if title else "SABONG"

    windows = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip() and len(title) > 1:
                # Get window size to filter tiny/hidden windows
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    if w > 100 and h > 100:  # skip tiny windows
                        windows.append((hwnd, title, w, h))
                except Exception:
                    pass
        return True

    win32gui.EnumWindows(callback, None)

    if not windows:
        print("  No visible windows found!")
        title = input("  Type the window title to capture: ").strip()
        return title if title else "SABONG"

    print("\n" + "=" * 62)
    print("  SELECT WINDOW TO CAPTURE")
    print("=" * 62)
    for i, (hwnd, title, w, h) in enumerate(windows, 1):
        # Truncate long titles
        display = title[:50] + "..." if len(title) > 50 else title
        print(f"  [{i:2d}]  {display:<54} ({w}x{h})")
    print(f"\n  [ 0]  Full screen capture (primary monitor)")
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
            else:
                print(f"  Please enter 0-{len(windows)}")
        except ValueError:
            print("  Please enter a number")


# ---- Stage 1: Window capture (works through AnyDesk / remote desktop) ----
def capture_worker(frame_q: Queue, stop: Event, cfg: Config):
    """
    Continuously grab the browser window at *capture_fps* using Windows API.
    Uses win32gui + PrintWindow to capture the window content directly,
    regardless of whether it's visible, behind other windows, or accessed
    through AnyDesk / remote desktop.

    Supports:
      - Specific window by title (partial match)
      - Full screen capture (__FULLSCREEN__)
      - Falls back to mss screen capture if win32 is not available
    """
    interval = 1.0 / cfg.capture_fps

    # Full screen mode — always use mss
    if cfg.window_title == "__FULLSCREEN__":
        logging.info("[CAPTURE] using full screen capture mode")
        _capture_mss_fullscreen(frame_q, stop, cfg, interval)
        return

    # Try win32 window capture first (works through AnyDesk)
    use_win32 = False
    try:
        import win32gui
        import win32ui
        import win32con
        use_win32 = True
        logging.info("[CAPTURE] using win32 window capture (AnyDesk-compatible)")
    except ImportError:
        logging.warning("[CAPTURE] pywin32 not installed — falling back to mss screen capture")
        logging.warning("[CAPTURE] Install with: pip install pywin32")

    if use_win32:
        _capture_win32(frame_q, stop, cfg, interval)
    else:
        _capture_mss(frame_q, stop, cfg, interval)


def _find_hwnd(title: str):
    """Find window handle by partial title match."""
    import win32gui
    result = []

    def callback(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            wtext = win32gui.GetWindowText(hwnd)
            if title.lower() in wtext.lower():
                result.append(hwnd)
        return True

    win32gui.EnumWindows(callback, None)
    return result[0] if result else None


def _capture_win32(frame_q: Queue, stop: Event, cfg: Config, interval: float):
    """Capture window content using Windows API — works through AnyDesk."""
    import win32gui
    import win32ui
    import win32con

    logging.info("[CAPTURE] starting — searching for window '%s'", cfg.window_title)

    hwnd = None
    while not stop.is_set() and hwnd is None:
        hwnd = _find_hwnd(cfg.window_title)
        if hwnd is None:
            logging.warning("[CAPTURE] window '%s' not found — retrying in 2s",
                            cfg.window_title)
            time.sleep(2)

    if stop.is_set():
        return

    logging.info("[CAPTURE] found window hwnd=%d", hwnd)
    recheck = 0

    while not stop.is_set():
        t0 = time.perf_counter()

        # Re-find window periodically (in case browser was restarted)
        recheck += 1
        if recheck % 100 == 0:
            new_hwnd = _find_hwnd(cfg.window_title)
            if new_hwnd:
                hwnd = new_hwnd

        try:
            # Get window dimensions
            left, top, right, bottom = win32gui.GetClientRect(hwnd)
            w = right - left
            h = bottom - top

            if w <= 0 or h <= 0:
                time.sleep(0.1)
                continue

            # Create device contexts
            hwndDC = win32gui.GetDC(hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            # Create bitmap
            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
            saveDC.SelectObject(saveBitMap)

            # PrintWindow with PW_CLIENTONLY flag to get client area only
            # Flag 0x1 = PW_CLIENTONLY, 0x2 = PW_RENDERFULLCONTENT
            result = win32gui.SendMessage(hwnd, win32con.WM_PRINT, saveDC.GetSafeHdc(),
                                          win32con.PRF_CLIENT | win32con.PRF_CHILDREN |
                                          win32con.PRF_OWNED)

            # If WM_PRINT didn't work, try PrintWindow
            if result == 0:
                import ctypes
                ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

            # Convert to numpy array
            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            frame = np.frombuffer(bmpstr, dtype=np.uint8)
            frame = frame.reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

            # BGRA to BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize to expected resolution if needed
            fh, fw = frame.shape[:2]
            if (fw, fh) != (cfg.source_w, cfg.source_h):
                frame = cv2.resize(frame, (cfg.source_w, cfg.source_h),
                                   interpolation=cv2.INTER_AREA)

            if not frame_q.full():
                frame_q.put(frame)

            # Cleanup
            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(hwnd, hwndDC)

        except Exception as e:
            logging.debug("[CAPTURE] frame grab failed: %s", e)
            time.sleep(0.1)
            continue

        elapsed = time.perf_counter() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    logging.info("[CAPTURE] stopped")


def _capture_mss(frame_q: Queue, stop: Event, cfg: Config, interval: float):
    """Fallback: screen capture using mss (requires visible window)."""
    import mss
    try:
        import pygetwindow as gw
    except ImportError:
        gw = None

    logging.info("[CAPTURE] starting (mss fallback) — target window '%s'", cfg.window_title)

    sct = mss.mss()

    def _find_window_rect():
        if gw:
            wins = gw.getWindowsWithTitle(cfg.window_title)
            if wins:
                w = wins[0]
                if w.isMinimized:
                    w.restore()
                return {"left": w.left, "top": w.top,
                        "width": w.width, "height": w.height}
        mon = sct.monitors[1]
        return {"left": mon["left"], "top": mon["top"],
                "width": cfg.source_w, "height": cfg.source_h}

    rect = _find_window_rect()
    recheck = 0

    while not stop.is_set():
        t0 = time.perf_counter()

        recheck += 1
        if recheck % 50 == 0:
            rect = _find_window_rect()

        img = np.array(sct.grab(rect))            # BGRA
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        if (w, h) != (cfg.source_w, cfg.source_h):
            frame = cv2.resize(frame, (cfg.source_w, cfg.source_h),
                               interpolation=cv2.INTER_AREA)

        if not frame_q.full():
            frame_q.put(frame)

        elapsed = time.perf_counter() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    logging.info("[CAPTURE] stopped")


def _capture_mss_fullscreen(frame_q: Queue, stop: Event, cfg: Config, interval: float):
    """Full screen capture — grabs the entire primary monitor."""
    import mss
    logging.info("[CAPTURE] starting (full screen mode)")

    sct = mss.mss()
    mon = sct.monitors[1]   # primary monitor

    while not stop.is_set():
        t0 = time.perf_counter()

        img = np.array(sct.grab(mon))               # BGRA
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        h, w = frame.shape[:2]
        if (w, h) != (cfg.source_w, cfg.source_h):
            frame = cv2.resize(frame, (cfg.source_w, cfg.source_h),
                               interpolation=cv2.INTER_AREA)

        if not frame_q.full():
            frame_q.put(frame)

        elapsed = time.perf_counter() - t0
        sleep_t = interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    logging.info("[CAPTURE] stopped")


# ---- Stage 2: Preprocess (crop + drawbox) ----
def preprocess_worker(frame_q: Queue, gray_q: Queue, stop: Event, cfg: Config):
    """
    Crop to focus area, apply drawbox to mask static watermark,
    detect and remove floating logo, then convert to gray.
    """
    logging.info("[PREPROCESS] starting")

    logo_detector = MovingLogoDetector(cfg)

    while not stop.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except Empty:
            continue

        # 1) Crop to focus region
        crop = frame[cfg.focus_y : cfg.focus_y + cfg.focus_h,
                      cfg.focus_x : cfg.focus_x + cfg.focus_w]

        # 2) Drawbox — black out static watermark rectangle
        cv2.rectangle(
            crop,
            (cfg.mask_x, cfg.mask_y),
            (cfg.mask_x + cfg.mask_w, cfg.mask_y + cfg.mask_h),
            cfg.mask_color,
            thickness=-1
        )

        # 3) Detect and remove floating logo (inpainting)
        crop = logo_detector.process(crop)

        # 4) Convert to grayscale for pHash
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if not gray_q.full():
            gray_q.put((time.time(), gray))

    logging.info("[PREPROCESS] stopped")


# ---- Stage 3: pHash computation (multi-core pool) ----
def _phash_one(args):
    """Standalone function for Pool.map — compute pHash for one frame."""
    ts, gray_bytes, shape, hash_size, hf = args
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(shape)
    h = compute_phash_bytes(gray, hash_size, hf)
    return (ts, h)


def phash_pool_worker(gray_q: Queue, hash_q: Queue, stop: Event, cfg: Config):
    """Pull gray frames, compute pHash using a thread pool.

    FIX: Python 3.13 forbids daemon processes from spawning child processes.
    Use concurrent.futures.ThreadPoolExecutor instead of multiprocessing.Pool.
    pHash is mostly numpy/OpenCV (C-level, releases the GIL), so threads
    give near-identical throughput without the daemon restriction.
    """
    from concurrent.futures import ThreadPoolExecutor
    logging.info("[PHASH] starting with %d threads", cfg.phash_cores)

    executor = ThreadPoolExecutor(max_workers=cfg.phash_cores)
    batch: list = []
    BATCH_SIZE = cfg.phash_cores

    while not stop.is_set():
        try:
            ts, gray = gray_q.get(timeout=0.2)
            batch.append((ts, gray.tobytes(), gray.shape,
                          cfg.phash_size, cfg.highfreq_factor))
        except Empty:
            pass

        if batch:
            futures = [executor.submit(_phash_one, item) for item in batch]
            for fut in futures:
                ts, h = fut.result()
                if not hash_q.full():
                    hash_q.put((ts, h))
            batch.clear()

    executor.shutdown(wait=False)
    logging.info("[PHASH] stopped")


# ---- Stage 4: DB matching ----
def _match_shard(args):
    """Search a shard of the flat hash list (fallback for huge DBs)."""
    target, shard_hashes, shard_labels, threshold = args
    target_int = bytes_to_int(target)
    n_bytes = len(target)
    best_d, best_l = threshold + 1, None
    for h, l in zip(shard_hashes, shard_labels):
        d = hamming_distance_int(target_int, bytes_to_int(h), n_bytes)
        if d < best_d:
            best_d, best_l = d, l
    return (best_d, best_l) if best_l else None


def match_worker(hash_q: Queue, result_q: Queue, stop: Event, cfg: Config):
    """
    Pull pHashes from hash_q, look up in the VP-tree DB.
    For very large DBs (>500 k), falls back to sharded brute-force.

    FIX: Uses ThreadPoolExecutor instead of multiprocessing.Pool to avoid
    'daemonic processes are not allowed to have children' on Python 3.13.
    """
    from concurrent.futures import ThreadPoolExecutor

    logging.info("[MATCH] loading hash database …")
    db = HashDB(cfg.hash_db_path, cfg.hamming_threshold)
    db.load()
    db.build_index()
    logging.info("[MATCH] ready — %d entries, VP-tree built", len(db.hashes))

    USE_BRUTE = len(db.hashes) > 500_000 and db.vptree is None
    executor = ThreadPoolExecutor(max_workers=cfg.match_cores) if USE_BRUTE else None

    # Pre-compute shards for brute-force path
    shards_h, shards_l = [], []
    if USE_BRUTE:
        chunk = len(db.hashes) // cfg.match_cores + 1
        for i in range(cfg.match_cores):
            shards_h.append(db.hashes[i*chunk:(i+1)*chunk])
            shards_l.append(db.labels[i*chunk:(i+1)*chunk])

    while not stop.is_set():
        try:
            ts, phash_bytes = hash_q.get(timeout=0.5)
        except Empty:
            continue

        t0 = time.perf_counter()

        if USE_BRUTE and executor:
            tasks = [(phash_bytes, sh, sl, cfg.hamming_threshold)
                     for sh, sl in zip(shards_h, shards_l)]
            futures = [executor.submit(_match_shard, t) for t in tasks]
            shard_results = [f.result() for f in futures]
            best = None
            for r in shard_results:
                if r and (best is None or r[0] < best[0]):
                    best = r
        else:
            # VP-tree path — O(log n)
            best = db.query(phash_bytes)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if best:
            dist, label = best

            parts = label.split("|")
            if len(parts) == 3:
                file_name   = parts[0]
                video_time  = parts[1]
                frame_ref   = parts[2]
            else:
                file_name  = label
                video_time = "unknown"
                frame_ref  = ""

            similarity = round((1 - dist / 256) * 100, 1)

            if dist <= 5:
                confidence = "EXACT"
            elif dist <= 8:
                confidence = "STRONG"
            elif dist <= 12:
                confidence = "GOOD"
            else:
                confidence = "WEAK"

            result = {
                "detected_at": time.strftime("%Y-%m-%d %H:%M:%S",
                                             time.localtime(ts)),
                "file_name": file_name,
                "video_timestamp": video_time,
                "similarity": f"{similarity}%",
                "confidence": confidence,
                "hamming_distance": dist,
                "lookup_ms": round(elapsed_ms, 1),
            }
            result_q.put(result)
            logging.info("[MATCH] ✓ %s @ %s  (%s %.1f%%)",
                         file_name, video_time, confidence, similarity)
        else:
            logging.debug("[MATCH] no match (%.1f ms)", elapsed_ms)

    if executor:
        executor.shutdown(wait=False)
    logging.info("[MATCH] stopped")


# ---- Result consumer (runs in main process) ----
def result_consumer(result_q: Queue, stop: Event, cfg: Config):
    """Log and persist match results with user-friendly display."""
    Path(cfg.result_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(cfg.result_dir) / "matches.jsonl"

    match_count = 0

    print()
    print("=" * 62)
    print("   LIVE FRAME MATCHER — watching for matches …")
    print(f"   Threshold: {cfg.hamming_threshold} "
          f"(lower = stricter, max 256)")
    print(f"   Results saved to: {log_path}")
    print("=" * 62)
    print()

    with open(log_path, "a", encoding="utf-8") as fout:
        while not stop.is_set():
            try:
                r = result_q.get(timeout=1.0)
            except Empty:
                continue

            match_count += 1

            fout.write(json.dumps(r) + "\n")
            fout.flush()

            print("┌─────────────────────────────────────────────────┐")
            print(f"│  ★ ★ ★ ★ ★   រ ក ឃើ ញ ហើ យ   ★ ★ ★ ★ ★  │")
            print("├─────────────────────────────────────────────────┤")
            print(f"│  ឯកសារ:        {r['file_name']:<35}│")
            print(f"│  ពេលវេលា:    {r['video_timestamp']:<35}│")
            print(f"│  ភាពស្រដៀង:  {r['similarity']:<35}│")
            print(f"│  ទំនុកចិត្ត:  {r['confidence']:<35}│")
            print(f"│  រកឃើញនៅ:    {r['detected_at']:<35}│")
            print(f"│  ល្បឿន:      {r['lookup_ms']} ms{'':<30}│")
            print("└─────────────────────────────────────────────────┘")
            print()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(cfg: Config):
    """Spawn all pipeline stages and supervise."""
    logging.basicConfig(level=getattr(logging, cfg.log_level),
                        format="%(asctime)s  %(message)s")

    stop = Event()

    frame_q = Queue(maxsize=30)
    gray_q  = Queue(maxsize=30)
    hash_q  = Queue(maxsize=60)
    result_q = Queue(maxsize=200)

    workers = [
        Process(target=capture_worker,     args=(frame_q, stop, cfg),
                name="capture",    daemon=True),
        Process(target=preprocess_worker,  args=(frame_q, gray_q, stop, cfg),
                name="preprocess", daemon=True),
        Process(target=phash_pool_worker,  args=(gray_q, hash_q, stop, cfg),
                name="phash",      daemon=True),
        Process(target=match_worker,       args=(hash_q, result_q, stop, cfg),
                name="match",      daemon=True),
    ]

    for w in workers:
        w.start()
        logging.info("Started %s (pid=%d)", w.name, w.pid)

    try:
        result_consumer(result_q, stop, cfg)
    except KeyboardInterrupt:
        logging.info("Ctrl-C received — shutting down …")
        stop.set()
        for w in workers:
            w.join(timeout=3)
        logging.info("All workers stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Live frame matcher — capture, pHash, match on 24 cores (OPTIMIZED v2)")
    sub = parser.add_subparsers(dest="cmd")

    # --- run ---
    p_run = sub.add_parser("run", help="Start the live matching pipeline")
    p_run.add_argument("--window", default="__pick__",
                       help="Window title to capture (default: interactive picker)")
    p_run.add_argument("--db", default=r"C:\hash_db\phash_database.pkl",
                       help="Path to the pHash database file")
    p_run.add_argument("--fps", type=int, default=5,
                       help="Capture frames per second (default 5)")
    p_run.add_argument("--threshold", type=int, default=12,
                       help="Hamming distance threshold for a match")
    p_run.add_argument("--result-dir", default=r"C:\hash_db\results",
                       help="Directory to write match results")

    # --- build-db ---
    p_build = sub.add_parser("build-db",
                             help="Build / extend the hash DB from video files")
    p_build.add_argument("video_dir", help="Folder with video files")
    p_build.add_argument("--db", default=r"C:\hash_db\phash_database.pkl")
    p_build.add_argument("--sample-fps", type=int, default=1,
                         help="Frames per second to sample from each video")

    # --- import-json (NEW) ---
    p_import = sub.add_parser("import-json",
                              help="Import hashes from video_preprocess.py JSON files")
    p_import.add_argument("json_dir",
                          help="Directory containing *_hashes.json files")
    p_import.add_argument("--db", default=r"C:\hash_db\phash_database.pkl")

    # --- benchmark (NEW) ---
    p_bench = sub.add_parser("benchmark",
                             help="Benchmark VP-tree build + query speed")
    p_bench.add_argument("--db", default=r"C:\hash_db\phash_database.pkl")
    p_bench.add_argument("--queries", type=int, default=100,
                         help="Number of random queries to run")

    # --- test-match (NEW v3) ---
    p_test = sub.add_parser("test-match",
                            help="Show sample matches to verify DB is working")
    p_test.add_argument("--db", default=r"C:\hash_db\phash_database.pkl")
    p_test.add_argument("--count", type=int, default=10,
                        help="Number of sample matches to show (default 10)")
    p_test.add_argument("--threshold", type=int, default=12,
                        help="Hamming distance threshold")

    args = parser.parse_args()

    if args.cmd == "build-db":
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(message)s")
        build_db_from_videos(args.video_dir, args.db, args.sample_fps)

    elif args.cmd == "import-json":
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(message)s")
        import_json_to_db(args.json_dir, args.db)

    elif args.cmd == "benchmark":
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(message)s")
        _run_benchmark(args.db, args.queries)

    elif args.cmd == "test-match":
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s  %(message)s")
        _run_test_match(args.db, args.count, args.threshold)

    elif args.cmd == "run":
        # Interactive window picker if --window not specified or set to default
        window_title = args.window
        if window_title == "__pick__":
            window_title = pick_window()
            print()

        cfg = Config(
            window_title=window_title,
            hash_db_path=args.db,
            capture_fps=args.fps,
            hamming_threshold=args.threshold,
            result_dir=args.result_dir,
        )
        run_pipeline(cfg)
    else:
        parser.print_help()


def _run_test_match(db_path: str, count: int = 10, threshold: int = 12):
    """Show sample matches so you can see which recordings are in the database."""
    print("\n" + "=" * 62)
    print("  TEST MATCH — Sample lookups from your database")
    print("  Intel Core Ultra 9 285K | 32 GB DDR5-5600")
    print("=" * 62)

    db = HashDB(db_path, threshold=threshold)
    db.load()
    db.build_index()

    if not db.hashes:
        print("  ERROR: Database is empty!")
        return

    # Show database summary
    video_names = {}
    for label in db.labels:
        parts = label.split("|")
        name = parts[0] if parts else label
        video_names[name] = video_names.get(name, 0) + 1

    print(f"\n  Database: {len(db.hashes)} hashes from {len(video_names)} videos")
    print(f"  ─────────────────────────────────────────")
    for name in sorted(video_names.keys()):
        secs = video_names[name]
        mins = secs // 60
        print(f"    {name:<30} {secs:>6} hashes ({mins} min)")
    print(f"  ─────────────────────────────────────────")

    # Run sample matches
    print(f"\n  Running {count} sample matches...\n")

    for i in range(count):
        idx = np.random.randint(len(db.hashes))
        target = db.hashes[idx]
        source_label = db.labels[idx]

        t0 = time.perf_counter()
        result = db.query(target)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result:
            dist, label = result
            parts = label.split("|")
            file_name = parts[0] if len(parts) >= 1 else label
            video_time = parts[1] if len(parts) >= 2 else "unknown"
            similarity = round((1 - dist / 256) * 100, 1)

            if dist <= 5:
                confidence = "EXACT"
            elif dist <= 8:
                confidence = "STRONG"
            elif dist <= 12:
                confidence = "GOOD"
            else:
                confidence = "WEAK"

            print("┌─────────────────────────────────────────────────┐")
            print(f"│  ★ ★ ★ ★ ★   រ ក ឃើ ញ ហើ យ   ★ ★ ★ ★ ★  │")
            print("├─────────────────────────────────────────────────┤")
            print(f"│  ឯកសារ:        {file_name:<35}│")
            print(f"│  ពេលវេលា:    {video_time:<35}│")
            print(f"│  ភាពស្រដៀង:  {similarity}%{'':<32}│")
            print(f"│  ទំនុកចិត្ត:  {confidence:<35}│")
            print(f"│  ល្បឿន:      {elapsed_ms:.2f} ms{'':<29}│")
            print("└─────────────────────────────────────────────────┘")
            print()
        else:
            print(f"  [{i+1}] No match found (lookup: {elapsed_ms:.2f} ms)")

    print("=" * 62)
    print("  These are the recordings your live matcher will identify")
    print("  when it detects matching frames from the live stream.")
    print("=" * 62 + "\n")


def _run_benchmark(db_path: str, n_queries: int = 100):
    """Benchmark all query strategies on your actual database."""
    print("\n" + "=" * 60)
    print("  BENCHMARK — Optimized Live Frame Matcher v2")
    print("  Intel Core Ultra 9 285K | 32 GB DDR5-5600")
    print("=" * 60)

    # Load database
    db = HashDB(db_path, threshold=12)
    t0 = time.perf_counter()
    db.load()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  Database: {len(db.hashes)} hashes")
    print(f"  Load time: {load_ms:.1f} ms")

    if not db.hashes:
        print("  ERROR: Database is empty!")
        return

    # ---- Test 1: Numpy vectorized fallback (no VP-tree) ----
    print(f"\n  [1/3] Numpy vectorized fallback ({n_queries} queries)...")
    np_times = []
    for i in range(n_queries):
        idx = np.random.randint(len(db.hashes))
        target = db.hashes[idx]
        t0 = time.perf_counter()
        db.query(target)   # VP-tree not built yet, uses numpy fallback
        np_times.append((time.perf_counter() - t0) * 1000)
    np_times = np.array(np_times)
    print(f"    Mean:   {np_times.mean():.2f} ms")
    print(f"    P95:    {np.percentile(np_times, 95):.2f} ms")

    # ---- Build VP-tree ----
    t0 = time.perf_counter()
    db.build_index()
    build_ms = (time.perf_counter() - t0) * 1000
    print(f"\n  VP-tree build: {build_ms:.1f} ms")
    if build_ms < 1000:
        print(f"    (loaded from cache — first build takes ~21 seconds)")

    # ---- Test 2: VP-tree queries ----
    print(f"\n  [2/3] VP-tree queries ({n_queries} queries)...")
    query_times = []
    matches = 0
    for i in range(n_queries):
        idx = np.random.randint(len(db.hashes))
        target = db.hashes[idx]
        t0 = time.perf_counter()
        result = db.query(target)
        elapsed = (time.perf_counter() - t0) * 1000
        query_times.append(elapsed)
        if result:
            matches += 1

    query_times = np.array(query_times)
    print(f"    Matches:  {matches}/{n_queries}")
    print(f"    Mean:     {query_times.mean():.2f} ms")
    print(f"    Median:   {np.median(query_times):.2f} ms")
    print(f"    P95:      {np.percentile(query_times, 95):.2f} ms")
    print(f"    P99:      {np.percentile(query_times, 99):.2f} ms")
    print(f"    Max:      {query_times.max():.2f} ms")
    print(f"    Min:      {query_times.min():.2f} ms")

    # ---- Test 3: Compare with old brute-force ----
    print(f"\n  [3/3] Old Python brute-force (10 queries for comparison)...")
    db_brute = HashDB(db_path, threshold=12)
    db_brute.load()
    db_brute._hash_array = None      # disable numpy fallback
    db_brute.vptree = None            # disable VP-tree
    # Force pure Python brute-force
    brute_times = []
    for i in range(min(n_queries, 10)):
        idx = np.random.randint(len(db_brute.hashes))
        target = db_brute.hashes[idx]
        t0 = time.perf_counter()
        db_brute.query(target)
        brute_times.append((time.perf_counter() - t0) * 1000)

    brute_times = np.array(brute_times)
    print(f"    Brute mean: {brute_times.mean():.2f} ms")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Python brute-force:  {brute_times.mean():>10.2f} ms  (old fallback)")
    print(f"  Numpy vectorized:    {np_times.mean():>10.2f} ms  (new fallback)")
    print(f"  VP-tree:             {query_times.mean():>10.2f} ms  (primary)")
    print(f"  ─────────────────────────────────────────")
    brute_vs_np = brute_times.mean() / np_times.mean() if np_times.mean() > 0 else 0
    brute_vs_vp = brute_times.mean() / query_times.mean() if query_times.mean() > 0 else 0
    print(f"  Numpy vs brute:      {brute_vs_np:>10.0f}x faster")
    print(f"  VP-tree vs brute:    {brute_vs_vp:>10.0f}x faster")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()
