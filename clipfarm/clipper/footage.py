"""Footage record v2: what each file of a project IS, from cheap deterministic signals first (the vision model comes second).

One entry point, `facts_for(path, probe, cache_dir, level=...)`, gives a JSON-able record per file: its role (a_roll, broll, voiceover_broll, audio,
music, photo, logo, graphic, screen, proxy ...), speech and faces (is the speaker on camera?), shot cuts, blur / exposure / black / frozen
checks, camera motion and shake, slow-motion eligibility, camera metadata, usable stretches. `finalize` links duplicate shots and alternate
takes across the whole folder, and settles what needs a transcript (words per second, earlier edits).

Two levels. "fast" reads the file once or twice at 480 px wide (ffprobe, one ffmpeg pass for black / frozen / cuts, at most MAX_SAMPLES frames for
faces, blur, exposure and picture hashes, three 2-second looks for camera motion): safe in the daytime. "full" adds the real camera-motion pass
(`cameramotion.extract`, as costly as indexing the file), and lets the caller add a vision description and catalogue embeddings through hooks;
it belongs in the night window (`full_allowed`) or on the M2, never in a hurry on the dev laptop.

The record is cached by path, size and mtime (never by display name, so two cameras that both wrote C0001.MP4 cannot share one).
Every numeric threshold below is a starting point from the improvement plan (docs/plans/editor-improvement-plan.md, WP2); change them from what
real folders show, not from guesses.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.footage")

VERSION = 2                                     # bumped when the record's meaning changes: older cache files are read again
LEVELS = {"fast": 1, "full": 2}
PROXY_W = 480                                   # every analysis decode is this wide (a 4K folder decoded at full size would take hours)
FACE_RATE = 2.0                                 # samples per second for faces, blur, exposure and hashes
MAX_SAMPLES = 400                               # per file: a two-hour file gets one sample every 18 s instead of 14,000 frames in memory
MOTION_WINDOWS, MOTION_WINDOW_S, MOTION_RATE = 3, 2.0, 8.0     # the cheap motion look: three short windows of 8 fps frames
MOTION_W = 160
SCENE_CUT = 0.4                                 # ffmpeg select(scene) threshold when PySceneDetect is not installed (proposal)
MIN_SHOT_S = 0.6                                # shots shorter than this are merged into their neighbour
BLUR_FLOOR = 40.0                               # Laplacian variance, 15th percentile, measured at 480 px wide (the plan's 80 is for 720p; proposal, calibrate)
CLIP_FLAG = 0.02                                # share of pixels at or beyond 254 / 1 that is worth a note
BLACK_MEAN = 16.0                               # a sampled frame this dark (0-255) counts as black when ffmpeg's blackdetect is not available
SPEECH_A_ROLL = 0.35                            # share of the file with speech: at or above, someone is talking in it
ON_CAMERA_MIN = 0.5                             # ... and the speaker is on camera when at least this share of the speech has a moving mouth
ACT_MIN = 1.5                                   # mean mouth-area change (0-255) that counts as a moving mouth
HAMMING_SAME = 10                               # perceptual hashes this close are the same picture
TAKE_SIMILARITY = 0.8                           # sentences this alike are the same line said again
EDIT_REPEAT = 0.6                               # an "earlier edit" repeats this share of another file's sentences
SLOWMO_FPS = 48.0
SHAKE_USABLE = 0.6

VIDEO_EXT = {".mp4", ".mov", ".mxf", ".m4v", ".mts", ".avi", ".mkv", ".webm"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".svg"}
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".aif", ".aiff", ".flac", ".ogg"}
SIDECAR_EXT = {".lrv", ".thm", ".lrf", ".xml", ".xmp", ".klv", ".smi"}
MUSIC_HINT = re.compile(r"music|song|track|score|soundtrack|bgm|theme", re.I)
SCREEN_TAG = re.compile(r"quicktime player|obs|screen ?(?:capture|record)|screencast|camtasia", re.I)
SCREEN_SIZES = ((16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (21, 9), (1, 1), (5, 4), (16, 10))


# --------------------------------------------------------------------------------------------------------------------------------- the cache
def cache_key(path: Path) -> str:
    """Path, size and modification time: a renamed display name is the same file, a touched file is a new one."""
    p = Path(path)
    try:
        st = p.stat()
        raw = f"{p}:{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        raw = f"{p}:0:0"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _cache_file(cache_dir: Path, path: Path) -> Path:
    return Path(cache_dir) / f"{cache_key(path)}.json"


def load_cached(cache_dir: Path, path: Path, level: str = "fast") -> dict | None:
    f = _cache_file(cache_dir, path)
    try:
        rec = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if rec.get("version") != VERSION or rec.get("key") != cache_key(path) or LEVELS.get(rec.get("level"), 0) < LEVELS.get(level, 1):
        return None
    return rec


def _store(cache_dir: Path, path: Path, rec: dict) -> None:
    f = _cache_file(cache_dir, path)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".part")
        tmp.write_text(json.dumps(rec, indent=1))
        tmp.replace(f)
    except OSError as e:
        log.debug(f"could not cache the facts of {path}: {e}")


def full_allowed(now: datetime | None = None) -> bool:
    """The heavy level runs in the night window (19:00 to 07:00), the same rule as the footage catalogue."""
    h = (now or datetime.now()).hour
    return h >= 19 or h < 7


def describe_cap(now: datetime | None = None, level: str = "fast") -> int:
    """How many of the longest mostly-visual files get a vision description: 150 by day, 400 at night or when the full level is asked for."""
    return 400 if (level == "full" or full_allowed(now)) else 150


# --------------------------------------------------------------------------------------------------------------------- sidecars and folders
def scan_folder(folder: Path, skip=None) -> dict:
    """The media files of a folder and their sidecars: {"media": [Path], "sidecars": [{"path", "name", "role": "proxy", "for": media name | None}]}.
    A sidecar is never footage: the camera's low-res proxy (GL010001.LRV for GX010001.MP4), a thumbnail, a Sony clip XML (C0001M01.XML for C0001.MP4)."""
    folder = Path(folder)
    media, side = [], []
    for p in sorted(folder.rglob("*")):
        try:
            if not p.is_file() or p.name.startswith("."):
                continue
            rel = p.relative_to(folder)
            if skip and skip(rel.parts[:-1]):
                continue
        except (OSError, ValueError):
            continue
        ext = p.suffix.lower()
        if ext in SIDECAR_EXT:
            side.append(p)
        elif ext in VIDEO_EXT:
            media.append(p)
    sidecars = []
    for s in side:
        owner = next((m for m in media if m.parent == s.parent and sidecar_matches(m, s)), None)
        sidecars.append({"path": str(s), "name": s.name, "role": "proxy", "for": owner.name if owner else None})
    return {"media": media, "sidecars": sidecars}


def sidecar_matches(media: Path, side: Path) -> bool:
    m, s = media.stem.lower(), side.stem.lower()
    if m == s or (len(m) >= 4 and s.startswith(m)):                      # C0001.MP4 and C0001M01.XML
        return True
    return bool(re.match(r"^g[xhl]\d{2}\d{4}$", m) and re.match(r"^g[xhl]\d{2}\d{4}$", s) and m[2:] == s[2:])     # GoPro: GX010001.MP4 and GL010001.LRV


# ------------------------------------------------------------------------------------------------------------------------------ probe facts
def _fps(vs: dict) -> float:
    for k in ("avg_frame_rate", "r_frame_rate"):
        try:
            a, b = str(vs.get(k) or "0/0").split("/")
            r = float(a) / float(b)
        except (ValueError, ZeroDivisionError):
            continue
        if 5 <= r <= 240:
            return round(r, 3)
    return 0.0


def _tags(probe: dict) -> dict[str, str]:
    t = {k.lower(): str(v) for k, v in ((probe.get("format") or {}).get("tags") or {}).items()}
    for s in probe.get("streams") or []:
        for k, v in (s.get("tags") or {}).items():
            t.setdefault(k.lower(), str(v))
    return t


def _first(tags: dict, *keys: str) -> str:
    return next((tags[k] for k in keys if tags.get(k)), "")


def camera_family(make: str, model: str, tags: dict, name: str) -> str:
    blob = " ".join([make, model, tags.get("encoder", ""), tags.get("handler_name", ""), tags.get("firmware", "")]).lower()
    if "gopro" in blob or re.match(r"^g[xhl]\d{6}", name.lower()):
        return "gopro"
    if "dji" in blob:
        return "dji"
    if SCREEN_TAG.search(blob):
        return "screen"
    if "sony" in blob or re.match(r"^ilce|^zv-|^fx\d|^pxw|^ilme", model.lower()) or re.match(r"^c\d{4}\.", name.lower()):
        return "sony_c"
    if any(w in blob for w in ("apple", "iphone", "samsung", "pixel", "xiaomi", "huawei", "oneplus")) or "com.apple.quicktime.make" in tags:
        return "phone"
    return "unknown"


def _created(probe: dict) -> str | None:
    t = _first(_tags(probe), "creation_time")
    try:
        dt = datetime.fromisoformat(t.split(";")[0].strip().replace("Z", "+00:00")) if t else None
    except ValueError:
        return None
    return dt.isoformat() if dt else None


def probe_facts(path: Path, probe: dict, name: str | None = None) -> dict:
    """Everything the container and its streams say about the file (no decoding)."""
    from .glasses import _display_dims
    vs = ffmpeg.video_stream(probe) or {}
    aus = ffmpeg.audio_stream(probe) or {}
    tags = _tags(probe)
    make = _first(tags, "com.apple.quicktime.make", "make", "manufacturer", "com.android.manufacturer")
    model = _first(tags, "com.apple.quicktime.model", "model", "com.android.model")
    try:
        w, h = _display_dims(vs) if vs else (0, 0)
    except Exception:  # noqa: BLE001
        w, h = int(vs.get("width") or 0), int(vs.get("height") or 0)
    fps = _fps(vs)
    tc = 0.0
    try:
        from .localproj import timecode_df, timecode_s
        tc, tc_df = timecode_s(probe, fps or 30), bool(timecode_df(probe))
    except Exception:  # noqa: BLE001 - timecode is a bonus
        tc_df = False
    log_name = None
    try:
        from . import logcolor
        d = logcolor.detect(path, use_look=False) or {}
        log_name = d.get("profile") or None
    except Exception:  # noqa: BLE001
        pass
    dur = float((probe.get("format") or {}).get("duration") or vs.get("duration") or aus.get("duration") or 0)
    return {
        "name": name or Path(path).name,
        "dur": round(dur, 3), "w": int(w), "h": int(h), "fps": fps,
        "codec": vs.get("codec_name") or "", "pix_fmt": vs.get("pix_fmt") or "",
        "has_video": bool(vs),
        "colour": {"primaries": vs.get("color_primaries") or "", "transfer": vs.get("color_transfer") or "", "log": log_name},
        "audio": {"present": bool(aus), "channels": int(aus.get("channels") or 0), "sample_rate": int(aus.get("sample_rate") or 0),
                  "codec": aus.get("codec_name") or ""},
        "camera": {"make": make, "model": model, "family": camera_family(make, model, tags, Path(path).name),
                   "created": _created(probe), "tc": tc, "tc_df": tc_df, "shutter": None},
        "tags": {k: tags[k] for k in ("encoder", "handler_name") if tags.get(k)},
    }


# ------------------------------------------------------------------------------------------------------------------------------------ speech
def vad(path: Path, dur: float) -> list[list[float]]:
    """Speech segments [[start, end], ...] in file seconds. A band-limited loudness gate (ffmpeg silencedetect on 120-3800 Hz): it hears music and
    loud rooms as speech too, so the role rules also need the face evidence. Replace this function with a real model (Silero VAD) when one is
    installed: everything else only reads its result."""
    cmd = [ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(path), "-vn", "-af", "highpass=f=120,lowpass=f=3800,silencedetect=n=-34dB:d=0.35",
           "-f", "null", "-"]
    try:
        err = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stderr
    except (OSError, subprocess.SubprocessError):
        return []
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", err)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", err)]
    quiet: list[tuple[float, float]] = []
    for i, a in enumerate(starts):
        quiet.append((max(0.0, a), ends[i] if i < len(ends) else dur))
    segs, t = [], 0.0
    for a, b in quiet:
        if a - t >= 0.25:
            segs.append([round(t, 2), round(a, 2)])
        t = max(t, b)
    if dur - t >= 0.25:
        segs.append([round(t, 2), round(dur, 2)])
    return segs


def speech_summary(segments: list[list[float]], dur: float) -> dict:
    spoken = sum(max(0.0, b - a) for a, b in segments)
    return {"ratio": round(spoken / max(0.5, dur), 3), "wps": None, "segments": [[round(a, 2), round(b, 2)] for a, b in segments[:200]]}


# ------------------------------------------------------------------------------------------------------------ frames, hashes and detectors
def _nice() -> list[str]:
    return ["nice", "-n", "15"] if shutil.which("nice") else []


def _even(x: float) -> int:
    return max(2, int(round(x)) // 2 * 2)


def sample_frames(path: Path, dur: float, w: int, h: int, rate: float = FACE_RATE, width: int = PROXY_W, max_frames: int = MAX_SAMPLES):
    """Yields (t, bgr frame [ph, pw, 3]) at `rate` per second (thinned so there are at most `max_frames`), one at a time: the file is never held in memory."""
    rate = min(rate, max_frames / max(1.0, dur))
    pw = min(width, w or width) // 2 * 2
    ph = _even((h or 9) * pw / (w or 16))
    cmd = [*_nice(), ffmpeg.bin_path("ffmpeg"), "-v", "error", "-t", f"{dur:.3f}", "-i", str(path), "-an", "-vf", f"fps={rate:.5f},scale={pw}:{ph}", "-f",
           "rawvideo", "-pix_fmt", "bgr24", "-"]
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return
    n, size = 0, pw * ph * 3
    try:
        while n < max_frames:
            raw = p.stdout.read(size)
            if len(raw) < size:
                break
            yield n / rate, np.frombuffer(raw, np.uint8).reshape(ph, pw, 3)
            n += 1
    finally:
        try:
            p.kill()
        except OSError:
            pass
        p.wait()


def lap_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian (the classic sharpness number): high = sharp, low = blurred or flat."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def phash(gray: np.ndarray) -> int:
    """A 64-bit perceptual hash (32x32 DCT, the top-left 8x8 against its median); no imagehash needed."""
    try:
        import cv2
        small = cv2.resize(gray.astype(np.float32), (32, 32), interpolation=cv2.INTER_AREA)
        d = cv2.dct(small)[:8, :8]
    except ImportError:
        small = gray.astype(np.float32)
        ys, xs = np.linspace(0, small.shape[0] - 1, 8).astype(int), np.linspace(0, small.shape[1] - 1, 8).astype(int)
        d = small[np.ix_(ys, xs)]
    bits = (d > np.median(d)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(int(a) ^ int(b)).count("1")


def _face_detector(models_dir, w: int, h: int):
    """A callable frame -> [(x, y, fw, fh, score)] in pixels, or None when no face model is on disk (it is never downloaded here)."""
    if not models_dir:
        return None
    model = Path(models_dir) / "face_detection_yunet_2023mar.onnx"
    if not model.exists():
        return None
    try:
        import cv2
        det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.7, 0.3, 50)
    except Exception as e:  # noqa: BLE001
        log.debug(f"no face detector: {e}")
        return None

    def run(frame):
        _, faces = det.detect(frame)
        return [(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])) for f in (faces if faces is not None else [])]
    return run


def analyse_samples(path: Path, dur: float, w: int, h: int, models_dir=None, detector=None) -> tuple[list[dict], bool]:
    """(samples, faces_measured). One sample per frame at ~2 per second: {t, blur, hi, lo, mean, hash, faces: [{cx, cy, w, h, score, act, track}]};
    faces_measured is False when no face model was available (then "no faces" means "not looked for")."""
    from . import reframe as R
    out: list[dict] = []
    prev = None
    tracks: dict[int, tuple[float, int]] = {}
    nxt = 0
    det = detector
    for k, (t, frame) in enumerate(sample_frames(path, dur, w, h)):
        gray = frame.astype(np.float32).mean(axis=2)
        ph, pw = gray.shape
        if det is None and k == 0:
            det = _face_detector(models_dir, pw, ph) or False
        boxes = det(frame) if det else []
        faces = []
        for (x, y, fw, fh, sc) in boxes:
            if fw / pw < 0.03:
                continue
            act = R.mouth_motion(prev, gray, (x, y, fw, fh)) if prev is not None else 0.0
            faces.append({"cx": round((x + fw / 2) / pw, 3), "cy": round((y + fh / 2) / ph, 3), "w": round(fw / pw, 3), "h": round(fh / ph, 3),
                          "score": round(sc, 2), "act": round(float(act), 2), "track": -1})
        used: set[int] = set()
        for f in sorted(faces, key=lambda f: -f["w"]):
            cand = [(abs(cx - f["cx"]), tid) for tid, (cx, seen) in tracks.items() if tid not in used and abs(cx - f["cx"]) < 0.15 and k - seen <= 2]
            f["track"] = min(cand)[1] if cand else nxt
            if not cand:
                nxt += 1
            used.add(f["track"])
            tracks[f["track"]] = (f["cx"], k)
        out.append({"t": round(t, 2), "blur": round(lap_var(gray), 1), "hi": round(float((gray >= 254).mean()), 4), "lo": round(float((gray <= 1).mean()), 4),
                    "mean": round(float(gray.mean()), 1), "hash": phash(gray), "faces": faces})
        prev = gray
    return out, bool(det)


# -------------------------------------------------------------------------------------------------------------- black / frozen / cuts, one pass
def scan_pass(path: Path, dur: float) -> dict:
    """ONE decode at 480 px for black stretches, frozen stretches and hard cuts: {"black": [[a, b]], "frozen": [[a, b]], "cuts": [t]}."""
    vf = (f"scale={PROXY_W}:-2,blackdetect=d=0.5:pic_th=0.98:pix_th=0.10,freezedetect=n=0.001:d=1.5,"
          f"select='gt(scene,{SCENE_CUT})',showinfo")
    try:
        err = subprocess.run([*_nice(), ffmpeg.bin_path("ffmpeg"), "-nostats", "-t", f"{dur:.3f}", "-i", str(path), "-an", "-vf", vf, "-f", "null", "-"],
                             capture_output=True, text=True, timeout=1800).stderr
    except (OSError, subprocess.SubprocessError):
        return {"black": [], "frozen": [], "cuts": []}
    black = [[float(a), float(b)] for a, b in re.findall(r"black_start:(-?[\d.]+) black_end:(-?[\d.]+)", err)]
    fs = [float(x) for x in re.findall(r"freeze_start: (-?[\d.]+)", err)]
    fe = [float(x) for x in re.findall(r"freeze_end: (-?[\d.]+)", err)]
    fe += [dur] * (len(fs) - len(fe))
    frozen = [[a, b] for a, b in zip(fs, fe) if b - a >= 1.5]
    cuts = sorted(float(t) for t in re.findall(r"pts_time:(-?[\d.]+)", err) if float(t) > 0.2)
    return {"black": [[round(a, 2), round(b, 2)] for a, b in black], "frozen": [[round(a, 2), round(b, 2)] for a, b in frozen],
            "cuts": [round(c, 2) for c in cuts]}


def shots_from_cuts(cuts: list[float], dur: float, min_len: float = MIN_SHOT_S) -> list[dict]:
    """[{"start", "end", "cut": "none"|"hard"}]: a shot shorter than `min_len` is merged into the one before it. PySceneDetect's AdaptiveDetector
    (3.0, window 2) replaces the ffmpeg cut list when it is installed."""
    marks = [0.0]
    for c in sorted(cuts):
        if 0 < c < dur and c - marks[-1] >= min_len and dur - c >= min_len:
            marks.append(c)
    marks.append(dur)
    return [{"start": round(a, 2), "end": round(b, 2), "cut": "none" if i == 0 else "hard"} for i, (a, b) in enumerate(zip(marks, marks[1:]))]


def scenedetect_cuts(path: Path) -> list[float] | None:
    try:
        from scenedetect import AdaptiveDetector, detect
    except ImportError:
        return None
    try:
        found = detect(str(path), AdaptiveDetector(adaptive_threshold=3.0, window_width=2, min_scene_len=15))
        return [s[0].get_seconds() for s in found[1:]]
    except Exception as e:  # noqa: BLE001
        log.debug(f"PySceneDetect failed on {path}: {e}")
        return None


# ----------------------------------------------------------------------------------------------------------------------------------- motion
def _frames_at(path: Path, start: float, secs: float, rate: float, w: int, h: int) -> list[np.ndarray]:
    pw = MOTION_W
    ph = _even((h or 9) * pw / (w or 16))
    cmd = [*_nice(), ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{secs:.3f}", "-i", str(path), "-an",
           "-vf", f"fps={rate},scale={pw}:{ph},format=gray", "-f", "rawvideo", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    n = len(raw) // (pw * ph)
    return [np.frombuffer(raw[k * pw * ph:(k + 1) * pw * ph], np.uint8).reshape(ph, pw) for k in range(n)]


def motion_fast(path: Path, dur: float, w: int, h: int) -> dict | None:
    """Camera motion from three short looks (8 fps, 160 px, Farneback flow): {"type", "pan_speed" (px/s at 480 px wide), "shake" 0-1, "energy", "zoom"}.
    None when OpenCV is missing or the file is too short."""
    try:
        import cv2
    except ImportError:
        return None
    if dur < 1.0:
        return None
    n = MOTION_WINDOWS if dur >= MOTION_WINDOWS * MOTION_WINDOW_S * 2 else 1
    starts = [max(0.0, (dur - MOTION_WINDOW_S) * (i + 0.5) / n) for i in range(n)] if n > 1 else [max(0.0, (dur - min(dur, MOTION_WINDOW_S)) / 2)]
    tx, ty, zoom, res = [], [], [], []
    for s0 in starts:
        fr = _frames_at(path, s0, min(MOTION_WINDOW_S, dur), MOTION_RATE, w, h)
        win_tx = []
        for a, b in zip(fr, fr[1:]):
            (mx, my), _resp = cv2.phaseCorrelate(a.astype(np.float32), b.astype(np.float32))        # (the whole picture's shift: sturdy on flat colour areas)
            hh, ww = a.shape
            if abs(mx) > ww / 3 or abs(my) > hh / 3:                           # (a cut between the two frames, not a camera move)
                continue
            flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 2, 5, 1.1, 0)
            fx, fy = flow[..., 0], flow[..., 1]
            yy, xx = np.mgrid[0:hh, 0:ww]
            xc, yc = xx - ww / 2, yy - hh / 2
            div = float(((xc * (fx - float(np.median(fx))) + yc * (fy - float(np.median(fy)))).mean()) / max(1.0, (xc ** 2 + yc ** 2).mean()))
            win_tx.append((mx, my))
            tx.append(mx)
            ty.append(my)
            zoom.append(div)
        if len(win_tx) >= 3:                                              # shake: what is left of the movement after a 3-frame average is taken out
            arr = np.array(win_tx)
            sm = np.stack([np.convolve(arr[:, i], np.ones(3) / 3, mode="same") for i in range(2)], axis=1)
            res.extend(np.linalg.norm((arr - sm)[1:-1], axis=1).tolist())
    if not tx:
        return None
    scale = PROXY_W / MOTION_W
    mean_tx, mean_ty = float(np.mean(tx)), float(np.mean(ty))
    speed = float(np.mean(np.hypot(np.array(tx), np.array(ty)))) * MOTION_RATE * scale
    net = float(np.hypot(mean_tx, mean_ty)) * MOTION_RATE * scale
    z = float(np.mean(zoom)) * MOTION_RATE                                  # fraction per second
    step = float(np.mean(np.hypot(np.array(tx), np.array(ty))))                 # (px per sample at 160 wide)
    shake = float(min(1.0, (np.sqrt(np.mean(np.square(res))) if res else 0.0) / (1.2 * step + 3.0)))      # jitter left after a smooth move is taken out, against the size of the move
    if shake >= 0.35 and net < 0.5 * speed + 1e-6 or shake >= 0.5:
        kind = "handheld"
    elif abs(z) >= 0.03 and abs(z) * 120 > net:
        kind = "push" if z > 0 else "pull"
    elif net >= 8.0:
        kind = "pan" if abs(mean_tx) >= abs(mean_ty) else "tilt"
    elif speed < 4.0:
        kind = "static"
    else:
        kind = "moving"
    return {"type": kind, "pan_speed": round(net, 1), "shake": round(shake, 2), "energy": round(speed + abs(z) * 120, 1), "zoom": round(z, 3)}


def motion_full(path: Path) -> dict | None:
    """The real thing, for catalogued-quality analysis: `cameramotion.extract` over the keyframes. As costly as indexing the file; night window or M2 only."""
    from . import cameramotion as CM
    try:
        m = CM.extract(ffmpeg.bin_path("ffmpeg"), path, prefix=_nice())
        prof = CM.profile(m)
    except Exception as e:  # noqa: BLE001
        log.debug(f"full motion of {path} failed: {e}")
        return None
    if not prof:
        return None
    runs = prof.get("runs") or []
    shake = float(np.mean(prof["shake"]))
    moving = sum(b - a for a, b, _s in runs)
    span = float(m[-1, CM.T] - m[0, CM.T]) if len(m) > 1 else 0.0
    dx, dy, ls = float(np.mean(m[:, CM.DX])), float(np.mean(m[:, CM.DY])), float(np.mean(m[:, CM.LOGS]))
    if shake >= 0.4:
        kind = "handheld"
    elif span and moving / span >= 0.3:
        kind = "push" if abs(ls) > 0.5 * (abs(dx) + abs(dy)) and ls > 0 else "pull" if abs(ls) > 0.5 * (abs(dx) + abs(dy)) else ("pan" if abs(dx) >= abs(dy) else "tilt")
    else:
        kind = "static"
    return {"type": kind, "pan_speed": round(float(np.mean(np.hypot(m[:, CM.DX], m[:, CM.DY]) / np.maximum(m[:, CM.DT], 0.05))) * PROXY_W, 1),
            "shake": round(min(1.0, shake), 2), "energy": round(float(np.mean(prof["speed"])) * 100, 1), "zoom": round(ls, 3)}


# ----------------------------------------------------------------------------------------------------------------------- faces and speakers
def shot_size_from_ratio(r: float | None) -> str | None:
    """Textbook shot scales from the largest face's height over the frame height; None without a face (the vision model decides then)."""
    if r is None or r <= 0:
        return None
    return "close" if r >= 0.45 else "medium_close" if r >= 0.25 else "medium" if r >= 0.12 else "wide" if r >= 0.05 else None


def faces_summary(samples: list[dict], speakers: float | None) -> dict:
    counts = [len(s["faces"]) for s in samples]
    big = [max(f["h"] for f in s["faces"]) for s in samples if s["faces"]]
    return {"count_median": int(np.median(counts)) if counts else 0,
            "largest_ratio_median": round(float(np.median(big)), 3) if big else 0.0,
            "on_camera_speaker_ratio": None if speakers is None else round(float(speakers), 3)}


def active_speaker(video: Path, segments: list[list[float]], samples: list[dict] | None = None) -> float | None:
    """Share of the speech during which a face on screen has a moving mouth (None when faces were not measured). The hook for a better model
    (Light-ASD, TalkNet): same arguments, same 0-1 result. This one compares the mouth-area change of the most active face inside each speech
    segment with the same number in the silences; with no silence to compare it only asks for a clearly moving mouth."""
    if not samples or not segments:
        return None if not samples else 0.0
    if not any(s["faces"] for s in samples):
        return 0.0 if segments else None

    def act_at(s):
        return max((f["act"] for f in s["faces"]), default=0.0)

    def inside(t):
        return any(a <= t <= b for a, b in segments)
    quiet = [act_at(s) for s in samples if not inside(s["t"])]
    base = float(np.mean(quiet)) if len(quiet) >= 3 else 0.0
    got = total = 0.0
    for a, b in segments:
        ln = b - a
        if ln < 0.6:
            continue
        acts = [act_at(s) for s in samples if a <= s["t"] <= b]
        total += ln
        if acts and float(np.mean(acts)) >= max(ACT_MIN, 1.3 * base):
            got += ln
    return got / total if total else None


# --------------------------------------------------------------------------------------------------------------------------- the role rules
def _odd_size(w: int, h: int) -> bool:
    if not w or not h:
        return False
    r = w / h
    return not any(abs(r - a / b) / (a / b) < 0.02 for a, b in SCREEN_SIZES)


def classify(rec: dict, rel_parts=(), folder_music: bool | None = None) -> tuple[str, str]:
    """(role, plain reason). Deterministic first: sidecars, audio, pictures, screen recordings; then speech and faces."""
    name = rec.get("name") or ""
    ext = Path(name).suffix.lower()
    if ext in SIDECAR_EXT:
        return "proxy", "a camera sidecar, not footage"
    if ext in IMAGE_EXT:
        return rec.get("role") or "photo", "a picture"
    sp = (rec.get("speech") or {}).get("ratio") or 0.0
    if not rec.get("has_video", True):
        in_music = folder_music if folder_music is not None else bool(any(MUSIC_HINT.search(p) for p in list(rel_parts)[:-1]) or MUSIC_HINT.search(Path(name).stem))
        if sp < 0.05 and in_music:
            return "music", f"no picture, hardly any speech ({sp:.2f}), music-like name or folder"
        return "audio", f"no picture (speech {sp:.2f})"
    faces = rec.get("faces") or {}
    motion = (rec.get("motion") or {}).get("type")
    tags = " ".join((rec.get("tags") or {}).values())
    no_faces = not faces.get("count_median")
    seen_none = bool(faces.get("measured")) and no_faces
    if (SCREEN_TAG.search(tags) and no_faces) or (_odd_size(rec.get("w") or 0, rec.get("h") or 0) and motion == "static" and seen_none):
        return "screen", "looks like a screen recording"
    on_cam = faces.get("on_camera_speaker_ratio")
    if sp >= SPEECH_A_ROLL:
        if on_cam is None:
            return "a_roll", f"speech {sp:.2f} of the file (faces not measured)"
        if on_cam >= ON_CAMERA_MIN:
            return "a_roll", f"speech {sp:.2f} of the file, speaker on camera {on_cam:.2f}"
        return "voiceover_broll", f"speech {sp:.2f} of the file, but the speaker is not on camera ({on_cam:.2f})"
    return "broll", f"hardly any speech ({sp:.2f})"


def slowmo_eligible(rec: dict, median_energy: float = 0.0) -> bool:
    """Slow motion is for 48 fps or more, picture-only shots with some movement and no shake: never for anything with talking in it (60 fps
    conformed to 24 gives 2.5x without interpolation). A known shutter slower than 1/(2 fps) turns it off (it would look smeared)."""
    m = rec.get("motion") or {}
    if (rec.get("fps") or 0) < SLOWMO_FPS or rec.get("role") not in ("broll", "multicam_angle"):
        return False
    if ((rec.get("speech") or {}).get("ratio") or 0.0) >= 0.05:
        return False
    if (m.get("energy") or 0.0) <= median_energy or (m.get("shake") or 0.0) >= 0.4:
        return False
    sh = (rec.get("camera") or {}).get("shutter")
    return not (sh and float(sh) > 1.0 / (2 * rec["fps"]))


def _overlap(a: float, b: float, ranges: list[list[float]]) -> float:
    return sum(max(0.0, min(b, y) - max(a, x)) for x, y in ranges)


def usable_ranges(rec: dict, samples: list[dict]) -> list[dict]:
    """The stretches worth cutting from: inside a shot, not black or frozen, sharp enough, steady enough; for B-roll, not someone talking to the camera."""
    out = []
    shake = (rec.get("motion") or {}).get("shake") or 0.0
    bad = (rec.get("quality") or {}).get("black", []) + (rec.get("quality") or {}).get("frozen", [])
    on_cam_talk = rec.get("role") in ("broll", "voiceover_broll") and ((rec.get("faces") or {}).get("on_camera_speaker_ratio") or 0) >= ON_CAMERA_MIN
    for sh in rec.get("shots") or []:
        a, b = sh["start"], sh["end"]
        if b - a < 0.5 or _overlap(a, b, bad) > 0.5 * (b - a) or shake >= SHAKE_USABLE or on_cam_talk:
            continue
        blur = [s["blur"] for s in samples if a <= s["t"] < b]
        if blur and float(np.percentile(blur, 15)) < BLUR_FLOOR:
            continue
        why = "sharp, steady" + (", no speech" if ((rec.get("speech") or {}).get("ratio") or 0) < 0.35 else "")
        out.append({"start": a, "end": b, "why": why})
    return out


# ---------------------------------------------------------------------------------------------------------------------------- the entry point
def _image_record(path: Path, name: str, rel_parts) -> dict:
    from . import foldergfx as FG
    role = "photo"
    try:
        if FG.role_of(Path(path).name, rel_parts) == "logo":
            role = "logo"
        elif FG.is_graphic(path, rel_parts):
            role = "graphic"
    except Exception:  # noqa: BLE001
        pass
    return {"name": name, "role": role, "role_why": "a picture", "has_video": False}


def facts_for(path: Path, probe: dict | None, cache_dir: Path, *, level: str = "fast", name: str | None = None, rel_parts=(), models_dir=None,
              folder_energy: float = 0.0, sidecars: list[str] | None = None, motion_fn=None, describe=None, now: float | None = None) -> dict:
    """The facts record for one file (see the module docstring). `probe` is `ffmpeg.probe(path)` (None: probed here). Cached under `cache_dir` by
    path, size and mtime. `level="full"` adds the keyframe camera-motion pass and calls `describe(record)` (a vision description hook, returns a
    dict with at least "model") when given."""
    path = Path(path)
    level = level if level in LEVELS else "fast"
    models_dir = models_dir or default_models_dir()
    hit = load_cached(cache_dir, path, level)
    if hit is not None:
        if name:
            hit["name"] = name
        if sidecars is not None:
            hit["sidecars"] = sidecars
        return hit
    if level == "full":                                                  # the fast record exists: only the heavy parts are added to it
        prev = load_cached(cache_dir, path, "fast")
        if prev is not None and prev.get("has_video"):
            return _upgrade_full(prev, path, cache_dir, name, describe, folder_energy, motion_fn)
    ext = path.suffix.lower()
    if ext in IMAGE_EXT:
        rec = _image_record(path, name or path.name, rel_parts)
    else:
        if probe is None:
            probe = ffmpeg.probe(path)
        rec = probe_facts(path, probe, name)
        rec["has_video"] = ffmpeg.video_stream(probe) is not None
        dur, w, h = rec["dur"], rec["w"], rec["h"]
        segs = vad(path, dur) if rec["audio"]["present"] else []
        rec["speech"] = speech_summary(segs, dur)
        samples: list[dict] = []
        if rec["has_video"] and dur > 0.2:
            adur = min(dur, ANALYSE_MAX_S)                                  # (a very long file is looked at over its first 20 minutes)
            scan = scan_pass(path, adur)
            cuts = scenedetect_cuts(path) if dur <= ANALYSE_MAX_S else None
            rec["shots"] = shots_from_cuts(cuts if cuts is not None else scan["cuts"], adur)
            rec["analysed_s"] = round(adur, 1)
            samples, measured = analyse_samples(path, adur, w, h, models_dir)
            blur = [s["blur"] for s in samples]
            p15 = float(np.percentile(blur, 15)) if blur else 0.0
            rec["quality"] = {"blur_p15": round(p15, 1), "blur_floor_ok": bool(blur) and p15 >= BLUR_FLOOR,
                              "clip_hi": round(float(np.mean([s["hi"] for s in samples])), 4) if samples else 0.0,
                              "clip_lo": round(float(np.mean([s["lo"] for s in samples])), 4) if samples else 0.0,
                              "black": scan["black"], "frozen": scan["frozen"]}
            rec["faces"] = faces_summary(samples, active_speaker(path, segs, samples) if (measured and segs) else None)
            rec["faces"]["measured"] = measured
            for sh in rec["shots"]:                                          # up to three picture hashes per shot (the first, middle and last samples inside it)
                inside = [s for s in samples if sh["start"] <= s["t"] < sh["end"]] or []
                pick = [inside[0], inside[len(inside) // 2], inside[-1]] if len(inside) >= 3 else inside
                sh["phash"] = [format(s["hash"], "016x") for s in pick]
            rec["series"] = {"t": [s["t"] for s in samples], "face_h": [max((f["h"] for f in s["faces"]), default=0.0) for s in samples],
                             "face_cx": [max(s["faces"], key=lambda f: f["h"])["cx"] if s["faces"] else 0.0 for s in samples],
                             "face_cy": [max(s["faces"], key=lambda f: f["h"])["cy"] if s["faces"] else 0.0 for s in samples],
                             "blur": [s["blur"] for s in samples]}
            m = (motion_fn or motion_fast)(path, adur, w, h)
            if level == "full":
                m = (motion_full(path) if motion_fn is None else m) or m
            rec["motion"] = m or {"type": "unknown", "pan_speed": 0.0, "shake": 0.0, "energy": 0.0, "zoom": 0.0}
        else:
            rec["shots"], rec["quality"], rec["faces"], rec["motion"] = [], {}, {}, {}
        rec["role"], rec["role_why"] = classify(rec, rel_parts)
        rec["slowmo_eligible"] = slowmo_eligible(rec, folder_energy)
        rec["duplicates_of"] = None
        rec["usable"] = usable_ranges(rec, samples)
        rec["described"] = None
        if level == "full" and describe:
            try:
                rec["described"] = describe(rec)
            except Exception as e:  # noqa: BLE001 - a description is a bonus
                log.debug(f"describing {path.name} failed: {e}")
    st = None
    try:
        st = path.stat()
    except OSError:
        pass
    rec.update({"version": VERSION, "level": level, "key": cache_key(path), "path": str(path), "size": st.st_size if st else 0,
                "mtime": int(st.st_mtime) if st else 0, "sidecars": sidecars or [], "at": int(now or time.time())})
    _store(cache_dir, path, rec)
    return rec


def _upgrade_full(rec: dict, path: Path, cache_dir: Path, name: str | None, describe, folder_energy: float, motion_fn=None) -> dict:
    """Fast record -> full record: the keyframe camera-motion pass and the description hook, nothing re-read that the fast level already knows."""
    m = motion_full(path) if motion_fn is None else motion_fn(path, rec.get("analysed_s") or rec.get("dur") or 0, rec.get("w") or 0, rec.get("h") or 0)
    if m:
        rec["motion"] = m
    rec["slowmo_eligible"] = slowmo_eligible(rec, folder_energy)
    if describe:
        try:
            rec["described"] = describe(rec)
        except Exception as e:  # noqa: BLE001
            log.debug(f"describing {path.name} failed: {e}")
    if name:
        rec["name"] = name
    rec["level"], rec["at"] = "full", int(time.time())
    _store(cache_dir, path, rec)
    return rec


def default_models_dir() -> Path | None:
    """Where the face model lives when the caller has no engine to ask (the repo's data/models); None if it is not there."""
    p = Path(__file__).resolve().parents[2] / "data" / "models"
    return p if p.is_dir() else None


# ----------------------------------------------------------------------------------------------------------------- across the whole folder
def _sent_tokens(s: str) -> frozenset:
    return frozenset(re.findall(r"[a-z']{3,}", str(s).lower()))


def _sent_alike(a: str, b: str) -> float:
    ta, tb = _sent_tokens(a), _sent_tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def alternate_takes(sentences: dict[str, list[str]], threshold: float = TAKE_SIMILARITY) -> dict[str, str]:
    """{file name: the earlier file whose lines it says again}: at least `threshold` of its sentences have an almost identical sentence there."""
    names = list(sentences)
    out: dict[str, str] = {}
    for i, a in enumerate(names):
        mine = [s for s in sentences[a] if len(_sent_tokens(s)) >= 3]
        if len(mine) < 2:
            continue
        for b in names[:i]:
            theirs = [s for s in sentences[b] if len(_sent_tokens(s)) >= 3]
            if not theirs:
                continue
            same = sum(1 for s in mine if any(_sent_alike(s, t) >= threshold for t in theirs))
            if same / len(mine) >= threshold:
                out[a] = b
                break
    return out


def repeats_another(mine: list[str], others: dict[str, list[str]], share: float = EDIT_REPEAT) -> bool:
    """True when at least `share` of this file's sentences appear (nearly word for word) in one other file: it is made from that footage."""
    mine = [s for s in mine if len(_sent_tokens(s)) >= 3]
    if len(mine) < 2:
        return False
    for theirs in others.values():
        t = [x for x in theirs if len(_sent_tokens(x)) >= 3]
        if t and sum(1 for s in mine if any(_sent_alike(s, x) >= 0.8 for x in t)) / len(mine) >= share:
            return True
    return False


def link_duplicates(records: list[dict]) -> None:
    """Sets `duplicates_of` (the earlier file in the list, same folder) when at least half of a file's shots have a picture hash within Hamming
    distance 10 of a shot there and the two lengths are within 30%: a copy, or the same setup shot again."""
    for i, r in enumerate(records):
        r["duplicates_of"] = None
        mine = [h for sh in r.get("shots") or [] for h in sh.get("phash", [])[1:2] or sh.get("phash", [])[:1]]
        if not mine:
            continue
        for q in records[:i]:
            if str(Path(q.get("path") or "").parent) != str(Path(r.get("path") or "").parent) or not q.get("dur") or not r.get("dur"):
                continue
            if abs(q["dur"] - r["dur"]) / max(q["dur"], r["dur"]) > 0.3:
                continue
            theirs = [int(h, 16) for sh in q.get("shots") or [] for h in sh.get("phash", [])]
            hit = sum(1 for h in mine if any(hamming(int(h, 16), t) <= HAMMING_SAME for t in theirs))
            if theirs and hit / len(mine) >= 0.5:
                r["duplicates_of"] = q.get("name")
                break


def finalize(records: list[dict], *, words: list[dict] | None = None, starts: dict[str, float] | None = None,
             sentences: dict[str, list[str]] | None = None, edit_named: set[str] | None = None) -> list[dict]:
    """The folder-wide pass over the records of one scan (changes them in place and returns them): words per second and a transcript-based speech
    ratio (`words` are on the shared timeline, `starts` maps file name to its start there), roles again, duplicates, alternate takes, earlier edits
    (a file whose name looks like an export, in `edit_named`, that repeats another file's lines) and slow-motion eligibility against the folder's median."""
    for r in records:
        if words and starts and r.get("name") in starts and r.get("dur"):
            a = starts[r["name"]]
            ws = [w for w in words if a <= w["start"] < a + r["dur"]]
            sp = r.setdefault("speech", {})
            sp["wps"] = round(len(ws) / max(1.0, r["dur"]), 2)
            sp["ratio"] = round(min(1.0, sum(max(0.0, w["end"] - w["start"]) for w in ws) / max(1.0, r["dur"])), 3)
        if r.get("role") != "proxy":
            r["role"], r["role_why"] = classify(r)
    link_duplicates(records)
    if sentences:
        for a, b in alternate_takes(sentences).items():
            for r in records:
                if r.get("name") == a:
                    r["take_of"] = b
        for r in records:
            if r.get("name") in (edit_named or set()) and repeats_another(sentences.get(r["name"], []), {k: v for k, v in sentences.items() if k != r["name"]}):
                r["role"], r["role_why"] = "earlier_edit", "named like an export and says the same sentences as another file"
    energies = [(r.get("motion") or {}).get("energy") or 0.0 for r in records if r.get("role") in ("broll", "multicam_angle")]
    med = float(np.median(energies)) if energies else 0.0
    for r in records:
        r["slowmo_eligible"] = slowmo_eligible(r, med)
    return records


# ----------------------------------------------------------------------------------------------------------------------------- scenes
def scene_extras(rec: dict | None, start: float, end: float) -> dict:
    """What the file's facts say about one scene (file seconds): the face ratio and shot size measured there, camera motion, and whether it is a
    usable stretch. Merged into a vision scene when the model did not say it itself."""
    if not rec:
        return {}
    ser = rec.get("series") or {}
    ts, fh = ser.get("t") or [], ser.get("face_h") or []
    sel = [h for t, h in zip(ts, fh) if start <= t < end and h > 0]
    ratio = round(float(np.median(sel)), 3) if sel else None
    out = {"motion": (rec.get("motion") or {}).get("type"), "face_ratio": ratio}
    if ratio:
        out["shot_size_measured"] = shot_size_from_ratio(ratio)
    if rec.get("usable") is not None:
        out["usable"] = _overlap(start, end, [[u["start"], u["end"]] for u in rec["usable"]]) >= 0.6 * max(0.1, end - start)
    return out


def summary_line(rec: dict) -> str:
    """One plain line about a file for prompts and the MCP."""
    sp = (rec.get("speech") or {}).get("ratio")
    bits = [rec.get("role") or "?", f"speech {sp:.2f}" if sp is not None else "", f"faces {((rec.get('faces') or {}).get('largest_ratio_median') or 0):.2f}",
            (rec.get("motion") or {}).get("type") or ""]
    return ", ".join(b for b in bits if b)


# --------------------------------------------------------------------------------------------------------------- a whole scan at once
ANALYSE_MAX_S = 1200.0          # a file longer than this is analysed over its first 20 minutes (the rest of its facts come from the audio and the container)
COMPACT_KEYS = ("name", "dur", "w", "h", "fps", "codec", "pix_fmt", "colour", "audio", "camera", "tags", "has_video", "role", "role_why", "speech",
                "faces", "motion", "slowmo_eligible", "duplicates_of", "take_of", "sidecars", "level", "key")


def compact(rec: dict) -> dict:
    """The part of a record that goes into files.json (the cache file keeps the rest: per-sample series, picture hashes, shot lists)."""
    out = {k: rec[k] for k in COMPACT_KEYS if k in rec}
    sp = dict(out.get("speech") or {})
    sp["segments"] = sp.get("segments", [])[:40]
    out["speech"] = sp
    q = rec.get("quality") or {}
    out["quality"] = {k: q[k] for k in ("blur_p15", "blur_floor_ok", "clip_hi", "clip_lo") if k in q} | {"black": len(q.get("black") or []), "frozen": len(q.get("frozen") or [])}
    out["shots"] = len(rec.get("shots") or [])
    out["usable"] = (rec.get("usable") or [])[:12]
    return out


def attach(files: list[dict], cache_dir: Path, models_dir=None, level: str = "fast", workers: int = 2, progress=None, describe=None) -> None:
    """Gives every file entry of a scan its `facts` (compact; the full record is in the cache). Files are read a few at a time; one that cannot be read
    is left without facts (nothing downstream needs them). Duplicates are linked across the folder at the end."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    models_dir = models_dir or default_models_dir()
    todo = [f for f in files if f.get("path") and not f.get("library") and not f.get("photo") and not f.get("stock")]
    full: dict[str, dict] = {}

    def one(f):
        try:
            return f, facts_for(Path(f["path"]), None, cache_dir, level=level, name=f["name"], rel_parts=Path(f.get("rel") or f["name"]).parts,
                                models_dir=models_dir, describe=describe)
        except Exception as e:  # noqa: BLE001 - facts are a bonus, never a reason to stop
            log.warning(f"facts for {f.get('name')} failed: {e}")
            return f, None
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="facts") as ex:
        for fu in as_completed([ex.submit(one, f) for f in todo]):
            f, rec = fu.result()
            done += 1
            if progress:
                progress(done / max(1, len(todo)), f"Looking at {f['name']} ({done} of {len(todo)})")
            if rec:
                full[f["name"]] = rec
    recs = [full[f["name"]] for f in todo if f["name"] in full]
    finalize(recs)
    for f in todo:
        r = full.get(f["name"])
        if r:
            f["facts"] = compact(r)
            if not f.get("created") and (r.get("camera") or {}).get("created"):
                f["created"] = r["camera"]["created"]
            _store(cache_dir, Path(f["path"]), r)                      # (the duplicate links the pass found are kept)


def refine_with_words(files: list[dict], words: list[dict]) -> None:
    """Once the transcript exists: words per second and the spoken share from the words themselves (more exact than the loudness gate), and the
    roles again. `files` are scan entries with `facts`; `words` are on the shared timeline."""
    for f in files:
        fa = f.get("facts")
        if not fa or not f.get("dur"):
            continue
        ws = [w for w in words if f["start"] <= w["start"] < f["end"]]
        sp = fa.setdefault("speech", {})
        sp["wps"] = round(len(ws) / max(1.0, f["dur"]), 2)
        sp["ratio"] = round(min(1.0, sum(max(0.0, w["end"] - w["start"]) for w in ws) / max(1.0, f["dur"])), 3)
        fa["role"], fa["role_why"] = classify(fa, Path(f.get("rel") or f["name"]).parts)
        fa["slowmo_eligible"] = slowmo_eligible({**fa, "role": fa["role"]}, 0.0)


def facts_of(f: dict, cache_dir: Path) -> dict | None:
    """The full record of a scan entry (from the cache), or None."""
    try:
        return load_cached(cache_dir, Path(f["path"]), "fast")
    except Exception:  # noqa: BLE001
        return None


def merge_scene(scene: dict, rec: dict | None) -> dict:
    """A vision scene (file seconds) with the new fields defaulted, what the facts measured there added (camera motion, face ratio, usable), and the
    shot size taken from the measured face ratio when the vision model did not give one."""
    from . import broll as BR
    out = {**BR.with_defaults(scene), **scene_extras(rec, float(scene["start"]), float(scene["end"]))}
    if not out.get("shot_size") and out.get("shot_size_measured"):
        out["shot_size"] = out["shot_size_measured"]
    return out
