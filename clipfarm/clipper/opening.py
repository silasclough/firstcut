"""The first three seconds, and what is burned into the picture: the two things that decide whether a Short is swiped away.

Learned from the channel's own numbers (October 2026): a post that is one still picture with captions on it, or a talking head with another creator's
"SUBSCRIBE" bar and @handle across the bottom, got 5-20 views while the same channel's full-screen real footage with captions from the first frame got 2,900.
Measured on the finished file, so it catches any cause upstream.

    still     how much of the clip the picture does not move (half a second apart, middle of the frame; captions and the title card are left out)
    blank     the first frames are black or one flat colour
    branding  another channel's subscribe / follow / @handle / logo text in the top or bottom strip, in any of six frames (such bars come and go) and not said aloud
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger
from . import ocr

log = get_logger("clipper.opening")
FPS, W, H = 8, 64, 112
LAG = 4                                  # frames: half a second
STILL = 1.0                              # mean brightness change (0-255) between frames half a second apart, below which the picture "does not move"
STILL_CLIP = 0.45                        # share of the clip that is still, above which the clip is a slideshow
STILL_OPEN_S, STILL_OPEN = 3.0, 0.9      # a picture that does not move for the first 3 s
BRAND = re.compile(r"(?i)\b(subscribe|subscribed|follow (?:me|us)|follow for more|link in (?:bio|description)|patreon|onlyfans)\b|@\w{4,}|\bwww\.|\.(?:com|tv|net)\b|"
                   r"\b(?:instagram|tiktok|youtube|twitch|facebook|twitter|snapchat)\b")
TOP, BOTTOM = 0.09, 0.86                 # other channels put their bars above / below these lines (our own hook title starts at 13%, our captions sit at 60-80%)


def _gray(video: Path, t0: float = 0.0, t1: float | None = None) -> np.ndarray:
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{t0:.2f}"] + (["-t", f"{t1 - t0:.2f}"] if t1 else []) + \
          ["-i", str(video), "-an", "-vf", f"fps={FPS},scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    n = len(raw) // (W * H)
    return np.frombuffer(raw[: n * W * H], np.uint8).reshape(n, H, W).astype(np.float32)


def motion(frames: np.ndarray) -> dict:
    """-> still_share (whole clip), still_open (first 3 s), blank_open (first second is black or one flat colour). Looks at the middle of the frame only."""
    if len(frames) <= LAG + 2:
        return {"still_share": 0.0, "still_open": 0.0, "blank_open": False}
    mid = frames[:, int(H * 0.12):int(H * 0.62), :]
    lag = np.abs(mid[LAG:] - mid[:-LAG]).mean(axis=(1, 2))
    first = lag[: int(STILL_OPEN_S * FPS) - LAG] if len(lag) > 4 else lag
    head = frames[: FPS]
    blank = bool(head.std(axis=(1, 2)).max() < 6.0 or head.mean() < 8.0)
    return {"still_share": float((lag < STILL).mean()), "still_open": float((first < STILL).mean()), "blank_open": blank}


def branding(video: Path, duration: float, spoken: str = "") -> list[str] | None:
    """-> the burned-in promo texts found in the top or bottom strip of any of six frames (None if no text reader is available)."""
    if duration < 4:
        return []
    with tempfile.TemporaryDirectory() as td:
        shots = []
        for k, f in enumerate((0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):
            p = Path(td) / f"f{k}.jpg"
            subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{duration * f:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", str(p)],
                           capture_output=True, timeout=60)
            if p.exists():
                shots.append(p)
        pages = ocr.read(shots)
    if pages is None:
        return None
    said = spoken.lower()
    seen: dict[str, int] = {}
    for lines in pages:
        keys = set()
        for ln in lines:
            if ln.get("c", 1) < 0.5 or not (ln["y"] + ln["h"] < TOP or ln["y"] > BOTTOM):
                continue
            for m in BRAND.finditer(ln["t"]):
                w = m.group(0).lower()
                if w.lstrip("@") not in said:
                    keys.add(w)
        for k in keys:
            seen[k] = seen.get(k, 0) + 1
    return sorted(seen)


def check(video: Path, spoken: str = "", frames: np.ndarray | None = None, brand: bool = True, strict: bool = True) -> list[dict]:
    """-> issues ({"level": "bad"|"warn", "what"}) about the opening and the burned-in branding of a finished clip."""
    issues: list[dict] = []
    try:
        dur = ffmpeg.duration(video)
        m = motion(frames if frames is not None else _gray(video))
        if m["blank_open"]:
            issues.append({"level": "bad", "what": "it opens on an empty (black or flat) picture"})
        if not strict:                                   # news reels and originals hold a frame when the stock footage runs short: only a true slideshow counts
            m["still_open"] = 0.0
        if m["still_share"] >= (STILL_CLIP if strict else 0.8):
            issues.append({"level": "bad", "what": f"the picture does not move for {m['still_share'] * 100:.0f}% of the clip (one still with captions)"})
        elif m["still_open"] >= STILL_OPEN and dur > 8:
            issues.append({"level": "bad", "what": f"the picture does not move for the first {STILL_OPEN_S:.0f} seconds"})
    except Exception as e:  # noqa: BLE001 - a check is never a reason to lose a clip
        log.debug(f"motion check of {video.name} failed: {e}")
    try:
        found = branding(video, ffmpeg.duration(video), spoken) if brand else None
        if found:
            issues.append({"level": "bad", "what": "another channel's promo is burned into the picture (" + ", ".join(found[:3]) + ")"})
    except Exception as e:  # noqa: BLE001
        log.debug(f"branding check of {video.name} failed: {e}")
    return issues


def qa_field(video: Path, spoken: str = "", brand: bool = True) -> dict:
    """The `qa` entry of a clip record ({"issues": [...]}) for clips that do not go through the Clipper's own render (originals, news reels)."""
    return {"issues": check(video, spoken, brand=brand, strict=False)}
