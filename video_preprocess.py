"""
VIDEO QA PRE-PROCESSOR
======================
Extracts frames from screen recording videos and computes perceptual hashes.
Optimized for Intel Xeon E5-2676 v3 (12 cores / 24 threads) with 64GB RAM.

Arena crop is pixel-accurately calibrated for 3440x1440 video resolution.

Original calibration was for 1920x1040. Scaled using:
    scale_x = 3440 / 1920 = 1.7917
    scale_y = 1440 / 1040 = 1.3846

Overlay masking (two layers):
  1. STATIC masks  -- known fixed regions blacked out with FFmpeg drawbox filters.
  2. MOVING logo   -- floating colorful logo detected per-frame with OpenCV and
                      removed via inpainting AFTER FFmpeg extraction.

Requirements:
    pip install imagehash Pillow opencv-python numpy
    FFmpeg must be installed and added to system PATH

Usage (Windows):
    # Process all videos in a folder (arena-only, 1fps, overlays masked)
    python video_preprocess.py --input "D:\\video" --batch

    # Disable overlay masking (raw arena frames)
    python video_preprocess.py --input "D:\\video" --batch --no-mask

    # Disable moving logo removal only
    python video_preprocess.py --input "D:\\video" --batch --no-logo

    # Custom mask regions (replaces defaults, x:y:w:h in post-crop coords)
    python video_preprocess.py --input "D:\\video" --batch --mask 717:479:887:173

    # Single video
    python video_preprocess.py --input "D:\\video\\recording1.mp4"

    # 2 frames per second (higher accuracy)
    python video_preprocess.py --input "D:\\video" --batch --fps 2

    # Scene-change mode (faster)
    python video_preprocess.py --input "D:\\video" --batch --mode scene

    # Custom output folder
    python video_preprocess.py --input "D:\\video" --batch --output "D:\\HashDB"

    # Override crop region manually (x:y:width:height)
    python video_preprocess.py --input "D:\\video" --batch --crop 20:98:2258:1069
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count
from functools import partial

_WIN = sys.platform == "win32"
_SUBPROCESS_KWARGS = {}
if _WIN:
    _SUBPROCESS_KWARGS["creationflags"] = subprocess.CREATE_NO_WINDOW


def _run(cmd, timeout=None):
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        **_SUBPROCESS_KWARGS,
    )


def check_dependencies():
    errors = []

    try:
        result = _run(["ffmpeg", "-version"], timeout=10)
        if result.returncode != 0:
            errors.append("FFmpeg is installed but returned an error.")
    except FileNotFoundError:
        errors.append(
            "FFmpeg not found.\n"
            "  1. Download from https://ffmpeg.org/download.html\n"
            "  2. Extract and add the bin\\ folder to Windows PATH\n"
            "  3. Restart your terminal"
        )
    except Exception as e:
        errors.append(f"FFmpeg check failed: {e}")

    try:
        import imagehash
    except ImportError:
        errors.append("imagehash not installed. Run: pip install imagehash")

    try:
        from PIL import Image
    except ImportError:
        errors.append("Pillow not installed. Run: pip install Pillow")

    try:
        import cv2
    except ImportError:
        errors.append("opencv-python not installed. Run: pip install opencv-python")

    try:
        import numpy
    except ImportError:
        errors.append("numpy not installed. Run: pip install numpy")

    if errors:
        print("\n MISSING DEPENDENCIES:\n")
        for err in errors:
            print(f"  - {err}\n")
        sys.exit(1)
    else:
        print("All dependencies OK\n")


check_dependencies()

import imagehash
from PIL import Image
import cv2
import numpy as np


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
HASH_ALGORITHM   = "phash"
HASH_SIZE        = 16
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}

CONFIG = {
    "num_cores": cpu_count()
}

# ─────────────────────────────────────────────
# Arena crop — scaled for 3440×1440
# ─────────────────────────────────────────────
# Original values (1920×1040):
#   x=11, y=71, width=1260, height=772
#
# Scale factors:
#   scale_x = 3440 / 1920 = 1.7917
#   scale_y = 1440 / 1040 = 1.3846
#
# Scaled values:
#   x      = round(11   * 1.7917) = 20
#   y      = round(71   * 1.3846) = 98
#   width  = round(1260 * 1.7917) = 2258
#   height = round(772  * 1.3846) = 1069
#
# Boundary check: (20+2258)=2278 <= 3440 OK | (98+1069)=1167 <= 1440 OK
# ─────────────────────────────────────────────
ARENA_CROP = {
    "x":      20,
    "y":      98,
    "width":  2258,
    "height": 1069,
}

# ─────────────────────────────────────────────
# Overlay masks — scaled for 3440×1440
# ─────────────────────────────────────────────
# Coordinates are relative to the CROPPED arena (2258×1069).
#
# Original masks (1920×1040 arena of 1260×772):
#   watermark: x=400, y=346, w=495, h=125
#   bottom-right logo: x=1178, y=691, w=82, h=81
#
# Scaled ×1.7917 / ×1.3846:
#   watermark: x=717, y=479, w=887, h=173
#   bottom-right logo: x=2110, y=957, w=147, h=112
# ─────────────────────────────────────────────
OVERLAY_MASKS = [
    # Center watermark + timestamp
    {"x": 717,  "y": 479, "w": 887, "h": 173, "label": "watermark+timestamp"},

    # Bottom-right circular logo
    {"x": 2110, "y": 957, "w": 147, "h": 112, "label": "bottom_right_logo"},
]


# ─────────────────────────────────────────────
# Moving logo detector + remover
# ─────────────────────────────────────────────
MOVING_LOGO_CONFIG = {
    "min_area":        500,
    "max_area_ratio":  0.08,
    "sat_threshold":   80,
    "motion_threshold": 15,
    "inpaint_radius":  5,
    "history_size":    15,
    "pad":             10,
}


class MovingLogoDetector:
    def __init__(self, cfg=None):
        c = cfg or MOVING_LOGO_CONFIG
        self.min_area       = c["min_area"]
        self.max_area_ratio = c["max_area_ratio"]
        self.sat_thresh     = c["sat_threshold"]
        self.motion_thresh  = c["motion_threshold"]
        self.history_size   = c["history_size"]
        self.pad            = c["pad"]
        self.prev_gray      = None
        self.history        = []

    def detect(self, frame):
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

        return best


def remove_moving_logo(frame, bbox):
    if bbox is None:
        return frame
    x, y, w, h = bbox
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return cv2.inpaint(
        frame, mask,
        inpaintRadius=MOVING_LOGO_CONFIG["inpaint_radius"],
        flags=cv2.INPAINT_TELEA,
    )


def apply_moving_logo_removal(frame_data, remove_logo=True):
    if not remove_logo or not frame_data:
        return 0, len(frame_data), 0

    detector = MovingLogoDetector()
    cleaned = skipped = errors = 0

    for fd in frame_data:
        try:
            frame = cv2.imread(fd["path"])
            if frame is None:
                errors += 1
                continue
            bbox         = detector.detect(frame)
            clean_frame  = remove_moving_logo(frame, bbox)
            cv2.imwrite(fd["path"], clean_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if bbox:
                cleaned += 1
            else:
                skipped += 1
        except Exception as e:
            print("  WARNING: moving logo removal failed for {}: {}".format(
                fd["path"], e))
            errors += 1

    return cleaned, skipped, errors


def build_vf(extra, crop, masks=None):
    c = crop
    filters = []
    filters.append("crop={}:{}:{}:{}".format(c["width"], c["height"], c["x"], c["y"]))

    if masks:
        for m in masks:
            filters.append("drawbox=x={}:y={}:w={}:h={}:color=black:t=fill".format(
                m["x"], m["y"], m["w"], m["h"]
            ))

    if extra:
        filters.append(extra)

    return ",".join(filters)


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────
def format_duration(seconds):
    return str(timedelta(seconds=int(seconds)))


def get_video_duration(video_path):
    try:
        r = _run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", str(video_path)],
            timeout=30,
        )
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception as e:
        print("  WARNING: Could not get duration: {}".format(e))
        return None


def _safe_fps(rate_str):
    try:
        if "/" in rate_str:
            num, den = rate_str.split("/")
            return float(num) / float(den)
        return float(rate_str)
    except Exception:
        return 30.0


def get_video_info(video_path):
    try:
        r = _run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", str(video_path)],
            timeout=30,
        )
        s = json.loads(r.stdout)["streams"][0]
        return {
            "width":  s.get("width"),
            "height": s.get("height"),
            "codec":  s.get("codec_name"),
            "fps":    _safe_fps(s.get("r_frame_rate", "30/1")),
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────
def extract_frames_fps(video_path, output_dir, fps=1, crop=None, masks=None):
    c = crop or ARENA_CROP
    print("  Extracting at {} fps | arena {}x{} @ x={}, y={}...".format(
        fps, c["width"], c["height"], c["x"], c["y"]))

    out_pattern = str(Path(output_dir) / "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", build_vf("fps={}".format(fps), c, masks),
        "-q:v", "2",
        "-frame_pts", "1",
        out_pattern,
    ]
    result = _run(cmd)
    if result.returncode != 0:
        print("  ERROR - FFmpeg:\n{}".format(result.stderr[-600:]))
        return []

    frames = sorted(Path(output_dir).glob("frame_*.jpg"))
    frame_data = [
        {"path": str(f), "timestamp": round(i / fps, 2)}
        for i, f in enumerate(frames)
    ]
    print("  Extracted {} frames".format(len(frame_data)))
    return frame_data


def extract_frames_scene(video_path, output_dir, threshold=0.3, crop=None, masks=None):
    c = crop or ARENA_CROP
    print("  Scene-change extraction (threshold={}) | arena {}x{} @ x={}, y={}...".format(
        threshold, c["width"], c["height"], c["x"], c["y"]))

    out_pattern = str(Path(output_dir) / "scene_%06d.jpg")
    vf = build_vf("select='gt(scene\\,{})',showinfo".format(threshold), c, masks)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", "2",
        out_pattern,
    ]
    result = _run(cmd)
    if result.returncode != 0:
        print("  ERROR - FFmpeg:\n{}".format(result.stderr[-600:]))
        return []

    frames = sorted(Path(output_dir).glob("scene_*.jpg"))
    timestamps = re.findall(r"pts_time:(\d+\.?\d*)", result.stderr)

    frame_data = [
        {"path": str(f),
         "timestamp": round(float(timestamps[i]) if i < len(timestamps) else i * 4.0, 2)}
        for i, f in enumerate(frames)
    ]
    print("  Extracted {} scene-change frames".format(len(frame_data)))
    return frame_data


def extract_frames_keyframe(video_path, output_dir, crop=None, masks=None):
    c = crop or ARENA_CROP
    print("  Keyframe extraction | arena {}x{} @ x={}, y={}...".format(
        c["width"], c["height"], c["x"], c["y"]))

    out_pattern = str(Path(output_dir) / "key_%06d.jpg")
    vf = build_vf("select='eq(pict_type\\,I)',showinfo", c, masks)
    cmd = [
        "ffmpeg", "-i", str(video_path),
        "-vf", vf,
        "-vsync", "vfr",
        "-q:v", "2",
        out_pattern,
    ]
    result = _run(cmd)

    frames = sorted(Path(output_dir).glob("key_*.jpg"))
    timestamps = re.findall(r"pts_time:(\d+\.?\d*)", result.stderr)

    frame_data = [
        {"path": str(f),
         "timestamp": round(float(timestamps[i]) if i < len(timestamps) else i * 2.5, 2)}
        for i, f in enumerate(frames)
    ]
    print("  Extracted {} keyframes".format(len(frame_data)))
    return frame_data


def extract_frames_twopass(video_path, output_dir, gap_threshold=10, crop=None, masks=None):
    c = crop or ARENA_CROP
    print("  Two-pass extraction (gap threshold: {}s)...".format(gap_threshold))

    print("      Pass 1: Keyframes...")
    keyframe_data = extract_frames_keyframe(video_path, output_dir, crop=c, masks=masks)

    if not keyframe_data:
        print("      WARNING: No keyframes found, falling back to 1fps")
        return extract_frames_fps(video_path, output_dir, fps=1, crop=c, masks=masks)

    print("      Pass 2: Filling gaps > {}s...".format(gap_threshold))
    gaps = []
    for i in range(1, len(keyframe_data)):
        gap = keyframe_data[i]["timestamp"] - keyframe_data[i - 1]["timestamp"]
        if gap > gap_threshold:
            gaps.append((keyframe_data[i - 1]["timestamp"] + keyframe_data[i]["timestamp"]) / 2)

    gap_frames = []
    if gaps:
        gap_dir = Path(output_dir) / "gaps"
        gap_dir.mkdir(exist_ok=True)

        for idx, gap_time in enumerate(gaps):
            out_file = str(gap_dir / "gap_{:06d}.jpg".format(idx))
            vf = build_vf("", c, masks)
            cmd = [
                "ffmpeg",
                "-ss", str(gap_time),
                "-i", str(video_path),
                "-vf", vf,
                "-frames:v", "1",
                "-q:v", "2",
                out_file,
            ]
            _run(cmd)
            if os.path.exists(out_file):
                gap_frames.append({"path": out_file, "timestamp": round(gap_time, 2)})

    all_frames = sorted(keyframe_data + gap_frames, key=lambda x: x["timestamp"])
    print("      {} keyframes + {} gap-fills = {} total frames".format(
        len(keyframe_data), len(gap_frames), len(all_frames)))
    return all_frames


# ─────────────────────────────────────────────
# Perceptual hashing (parallelized)
# ─────────────────────────────────────────────
def _hash_one_frame(frame_info, hash_size=HASH_SIZE):
    try:
        img = Image.open(frame_info["path"])
        h = imagehash.phash(img, hash_size=hash_size)
        return {"t": frame_info["timestamp"], "h": str(h)}
    except Exception as e:
        return {"t": frame_info["timestamp"], "h": None, "error": str(e)}


def compute_hashes_parallel(frame_data, num_workers=None):
    num_workers = num_workers or CONFIG["num_cores"]
    print("  Computing p-hashes using {} cores...".format(num_workers))
    start = time.time()

    hash_func = partial(_hash_one_frame, hash_size=HASH_SIZE)
    with Pool(processes=num_workers) as pool:
        results = pool.map(hash_func, frame_data)

    valid  = [r for r in results if r.get("h") is not None]
    errors = [r for r in results if r.get("h") is None]

    elapsed = time.time() - start
    rate = len(valid) / elapsed if elapsed > 0 else 0
    print("  Hashed {} frames in {:.1f}s ({:.0f} frames/sec)".format(len(valid), elapsed, rate))
    if errors:
        print("  WARNING: {} frames failed to hash".format(len(errors)))

    return valid


# ─────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────
def process_video(video_path, output_dir, mode="fps", fps=1, scene_threshold=0.3,
                  crop=None, masks=None, no_logo=False):
    c = crop or ARENA_CROP
    m = masks
    video_path = Path(video_path)
    video_name = video_path.name

    os.makedirs(output_dir, exist_ok=True)
    json_path = str(Path(output_dir) / (video_path.stem + "_hashes.json"))
    if os.path.exists(json_path):
        size_mb = os.path.getsize(json_path) / (1024 * 1024)
        print("\n" + "=" * 60)
        print("  SKIPPING: {} (already processed)".format(video_name))
        print("  Found:    {} ({:.2f} MB)".format(json_path, size_mb))
        print("=" * 60)
        return json_path

    print("\n" + "=" * 60)
    print("  Processing: {}".format(video_name))
    print("=" * 60)
    print("  Arena crop:  x={}, y={}, width={}, height={} (3440x1440 pixels)".format(
        c["x"], c["y"], c["width"], c["height"]))
    if m:
        print("  Overlay masks: {} regions masked out".format(len(m)))
        for mask in m:
            print("    - {}: x={}, y={}, {}x{} px".format(
                mask.get("label", "unnamed"), mask["x"], mask["y"], mask["w"], mask["h"]))
    else:
        print("  Overlay masks: DISABLED")

    duration = get_video_duration(video_path)
    info     = get_video_info(video_path)

    if duration:
        print("  Duration:    {}".format(format_duration(duration)))
    if info:
        print("  Resolution:  {}x{}".format(info.get("width"), info.get("height")))
        print("  Codec:       {}".format(info.get("codec")))

    temp_dir = tempfile.mkdtemp(prefix="vqa_")
    print("  Temp dir:    {}".format(temp_dir))
    overall_start = time.time()

    try:
        extract_start = time.time()

        if mode == "fps":
            frame_data = extract_frames_fps(video_path, temp_dir, fps=fps, crop=c, masks=m)
        elif mode == "scene":
            frame_data = extract_frames_scene(video_path, temp_dir, threshold=scene_threshold, crop=c, masks=m)
        elif mode == "keyframe":
            frame_data = extract_frames_keyframe(video_path, temp_dir, crop=c, masks=m)
        elif mode == "twopass":
            frame_data = extract_frames_twopass(video_path, temp_dir, crop=c, masks=m)
        else:
            print("  ERROR: Unknown mode: {}".format(mode))
            return None

        extract_time = time.time() - extract_start
        print("  Extraction time: {:.1f}s".format(extract_time))

        if not frame_data:
            print("  ERROR: No frames extracted!")
            return None

        if not no_logo:
            print("  Detecting and removing floating logo...")
            cleaned, skipped, logo_errors = apply_moving_logo_removal(
                frame_data, remove_logo=True)
            print("  Moving logo: {} frames cleaned | {} no logo | {} errors".format(
                cleaned, skipped, logo_errors))
        else:
            print("  Moving logo removal: DISABLED (--no-logo)")

        hash_start   = time.time()
        hash_results = compute_hashes_parallel(frame_data)
        hash_time    = time.time() - hash_start

        output_data = {
            "video_name":       video_name,
            "video_path":       str(video_path),
            "duration_seconds": duration,
            "duration_human":   format_duration(duration) if duration else None,
            "resolution":       "{}x{}".format(info.get("width"), info.get("height")) if info else None,
            "arena_crop": {
                "x":      c["x"],
                "y":      c["y"],
                "width":  c["width"],
                "height": c["height"],
                "note":   "Scaled for 3440x1440 ultrawide. Excludes top bar, right betting panel, bottom UI.",
            },
            "overlay_masks": [
                {"x": mask["x"], "y": mask["y"], "w": mask["w"], "h": mask["h"],
                 "label": mask.get("label", "")}
                for mask in m
            ] if m else None,
            "moving_logo_removal": not no_logo,
            "hash_algorithm":   HASH_ALGORITHM,
            "hash_size":        HASH_SIZE,
            "extraction_mode":  mode,
            "extraction_fps":   fps if mode == "fps" else None,
            "total_frames":     len(hash_results),
            "processed_at":     datetime.now().isoformat(),
            "processing_time": {
                "extraction_seconds": round(extract_time, 1),
                "hashing_seconds":    round(hash_time, 1),
                "total_seconds":      round(time.time() - overall_start, 1),
            },
            "cores_used": CONFIG["num_cores"],
            "frames":     hash_results,
        }

        os.makedirs(output_dir, exist_ok=True)
        json_path = str(Path(output_dir) / (video_path.stem + "_hashes.json"))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        file_size  = os.path.getsize(json_path) / (1024 * 1024)
        total_time = time.time() - overall_start

        print("\n  " + "-" * 36)
        print("  RESULTS for {}".format(video_name))
        print("  " + "-" * 36)
        print("  Frames hashed:     {}".format(len(hash_results)))
        print("  JSON file:         {}".format(json_path))
        print("  JSON size:         {:.2f} MB".format(file_size))
        print("  Extraction time:   {:.1f}s".format(extract_time))
        print("  Hashing time:      {:.1f}s".format(hash_time))
        print("  Total time:        {}".format(format_duration(total_time)))
        print("  " + "-" * 36)

        return json_path

    finally:
        print("  Cleaning up temp files...")
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_batch(input_dir, output_dir, mode="fps", fps=1, crop=None, masks=None, no_logo=False):
    c = crop or ARENA_CROP
    input_dir   = Path(input_dir)
    video_files = []

    for ext in VIDEO_EXTENSIONS:
        video_files.extend(input_dir.glob("*{}".format(ext)))
        video_files.extend(input_dir.glob("*{}".format(ext.upper())))

    video_files = sorted(set(video_files))

    if not video_files:
        print("ERROR: No video files found in {}".format(input_dir))
        return

    print("\n" + "#" * 60)
    print("  BATCH PROCESSING: {} videos".format(len(video_files)))
    print("  Mode:   {} | Cores: {}".format(mode, CONFIG["num_cores"]))
    print("  Arena:  x={}, y={}, {}x{} px (3440×1440)".format(
        c["x"], c["y"], c["width"], c["height"]))
    if masks:
        print("  Masks:  {} overlay regions will be blacked out".format(len(masks)))
    else:
        print("  Masks:  DISABLED")
    print("#" * 60)

    batch_start = time.time()
    results  = []
    skipped  = []
    failed   = []

    for i, video_file in enumerate(video_files, 1):
        print("\n  [{}/{}]".format(i, len(video_files)))

        json_check = str(Path(output_dir) / (video_file.stem + "_hashes.json"))
        was_already_done = os.path.exists(json_check)

        json_path = process_video(video_file, output_dir, mode=mode, fps=fps,
                                   crop=c, masks=masks, no_logo=no_logo)

        if json_path and was_already_done:
            skipped.append(json_path)
        elif json_path:
            results.append(json_path)
        else:
            failed.append(str(video_file))

    batch_time = time.time() - batch_start

    print("\n\n" + "#" * 60)
    print("  BATCH COMPLETE")
    print("#" * 60)
    print("  Total videos:      {}".format(len(video_files)))
    print("  Newly processed:   {}".format(len(results)))
    print("  Skipped (done):    {}".format(len(skipped)))
    print("  Failed:            {}".format(len(failed)))
    print("  Total time:        {}".format(format_duration(batch_time)))
    print("  Output directory:  {}".format(output_dir))
    if failed:
        print("\n  FAILED videos:")
        for f in failed:
            print("    - {}".format(os.path.basename(f)))
    print("#" * 60 + "\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Video QA Pre-Processor: Extract arena frames and compute perceptual hashes (3440×1440)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Examples (Windows):
  # All videos in folder, 1fps (default, with overlay masking)
  python video_preprocess.py --input "D:\video" --batch

  # Disable overlay masking
  python video_preprocess.py --input "D:\video" --batch --no-mask

  # 2fps (higher accuracy)
  python video_preprocess.py --input "D:\video" --batch --fps 2

  # Scene-change mode
  python video_preprocess.py --input "D:\video" --batch --mode scene

  # Single video
  python video_preprocess.py --input "D:\video\recording1.mp4"

  # Custom output folder
  python video_preprocess.py --input "D:\video" --batch --output "D:\HashDB"

  # Override crop (if your resolution differs from 3440x1440)
  python video_preprocess.py --input "D:\video" --batch --crop 20:98:2258:1069
        """
    )

    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--batch",  "-b", action="store_true")
    parser.add_argument("--mode",   "-m",
                        choices=["fps", "scene", "keyframe", "twopass"], default="fps")
    parser.add_argument("--fps",    "-f", type=float, default=1.0)
    parser.add_argument("--cores",  "-c", type=int, default=None)
    parser.add_argument("--crop", type=str, default=None,
                        help="Override arena crop as x:y:width:height "
                             "(default: 20:98:2258:1069 for 3440x1440 video)")
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-logo", action="store_true")
    parser.add_argument("--mask", type=str, action="append", default=None,
                        help="Add/override mask region as x:y:w:h (replaces built-in masks)")

    args = parser.parse_args()

    if args.cores:
        CONFIG["num_cores"] = min(args.cores, cpu_count())

    crop = None
    if args.crop:
        try:
            x, y, w, h = map(int, args.crop.split(":"))
            crop = {"x": x, "y": y, "width": w, "height": h}
        except Exception:
            print("ERROR: Invalid --crop. Use format x:y:width:height  e.g.  20:98:2258:1069")
            sys.exit(1)

    if args.no_mask:
        masks = None
    elif args.mask:
        masks = []
        for i, m_str in enumerate(args.mask):
            try:
                mx, my, mw, mh = map(int, m_str.split(":"))
                masks.append({"x": mx, "y": my, "w": mw, "h": mh,
                               "label": "custom_{}".format(i + 1)})
            except Exception:
                print("ERROR: Invalid --mask '{}'. Use format x:y:w:h".format(m_str))
                sys.exit(1)
    else:
        masks = OVERLAY_MASKS

    if args.output is None:
        base = args.input if args.batch else str(Path(args.input).parent)
        args.output = str(Path(base) / "hash_db")

    c = crop or ARENA_CROP
    print("\nVIDEO QA PRE-PROCESSOR  (3440×1440 ultrawide)")
    print("   Input:       {}".format(args.input))
    print("   Output:      {}".format(args.output))
    print("   Mode:        {}".format(args.mode))
    print("   Cores:       {}".format(CONFIG["num_cores"]))
    print("   Arena crop:  x={}, y={}, {}x{} px (calibrated for 3440x1440)".format(
        c["x"], c["y"], c["width"], c["height"]))
    if masks:
        print("   Masks:       {} overlay regions".format(len(masks)))
        for mask in masks:
            print("                - {}: x={}, y={}, {}x{} px".format(
                mask.get("label", "unnamed"), mask["x"], mask["y"], mask["w"], mask["h"]))
    else:
        print("   Masks:       DISABLED")
    print("   Moving logo: {}".format("DISABLED (--no-logo)" if args.no_logo else "ON"))
    if args.mode == "fps":
        print("   FPS:         {}".format(args.fps))
    print()

    if args.batch:
        if not os.path.isdir(args.input):
            print("ERROR: Not a directory: {}".format(args.input))
            sys.exit(1)
        process_batch(args.input, args.output, mode=args.mode, fps=args.fps,
                      crop=crop, masks=masks, no_logo=args.no_logo)
    else:
        if not os.path.isfile(args.input):
            print("ERROR: File not found: {}".format(args.input))
            sys.exit(1)
        process_video(args.input, args.output, mode=args.mode, fps=args.fps,
                      crop=crop, masks=masks, no_logo=args.no_logo)


if __name__ == "__main__":
    main()
