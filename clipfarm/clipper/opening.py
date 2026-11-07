"""The first three seconds, and what is burned into the picture: the two things that decide whether a Short is swiped away.

Learned from the channel's own numbers (October 2026): a post that is one still picture with captions on it, or a talking head with another creator's
"SUBSCRIBE" bar and @handle across the bottom, got 5-20 views while the same channel's full-screen real footage with captions from the first frame got 2,900.
Measured on the finished file, so it catches any cause upstream.

    still     how much of the clip the picture does not move (half a second apart, middle of the frame; captions and the title card are left out)
    blank     the first frames are black or one flat colour
    branding  another channel's subscribe / follow / @handle / logo text in the top or bottom strip, in any of six frames (such bars come and go) and not said aloud;
              also a phone number, a name-tag lower third ("Creator of ...", "Let's connect") and a web address in those strips
    doubled   two sets of captions at once (ours, plus the source's own burned in): two lines in the caption zone, far apart, that say the same words
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
PROMO = re.compile(r"(?i)\b(?:creator|founder|co-?founder|ceo|author|host|owner|coach|speaker) of\b|\blet'?s (?:connect|talk|chat)\b|\bconnect with (?:me|us)\b|\b(?:dm|call|text|email) (?:me|us)\b|"
                   r"\bbook a (?:call|demo)\b|\bfree (?:training|guide|ebook|course|webinar)\b|\bsign up\b|\bcell phone\b|\bcontact\b|https?://|\.(?:org|io|co|me|ly|gg|app|shop|store)\b|"
                   r"(?<!\d)(?:\+?1[ .-])?\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}(?!\d)|(?<!\d)\d{3}[ .-]\d{4}(?!\d)")     # a phone number (10 or 7 digits), a name tag, a web address
TOP, BOTTOM = 0.09, 0.86                 # other channels put their bars above / below these lines (our own hook title starts at 13%, our captions sit at 60-80%)
TOP_P, BOTTOM_P = 0.12, 0.82             # name tags, phone numbers and addresses are looked for in a slightly wider strip (our own captions end at 80%)
CAP_TOP, CAP_BOTTOM = 0.50, 0.95         # where captions live (ours at 60-80%, a source's own often lower)
DOUBLE_GAP, DOUBLE_FRAMES = 0.10, 2      # two caption lines this far apart (a share of the height: our own two-line captions are about 0.07 apart) that say the same word, in this many of the six frames, are two sets of captions


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


def _pages(video: Path, duration: float) -> list[list[dict]] | None:
    """-> the text lines of six frames spread over the clip (None if no text reader is available)."""
    with tempfile.TemporaryDirectory() as td:
        shots = []
        for k, f in enumerate((0.1, 0.25, 0.4, 0.55, 0.7, 0.85)):
            p = Path(td) / f"f{k}.jpg"
            subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{duration * f:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", str(p)],
                           capture_output=True, timeout=60)
            if p.exists():
                shots.append(p)
        return ocr.read(shots)


def branding(video: Path, duration: float, spoken: str = "", pages: list[list[dict]] | None = None) -> list[str] | None:
    """-> the burned-in promo texts found in the top or bottom strip of any of six frames (None if no text reader is available).
    `pages` = the text already read from those frames (check() reads them once for both checks)."""
    if duration < 4:
        return []
    if pages is None:
        pages = _pages(video, duration)
    if pages is None:
        return None
    said = spoken.lower()
    seen: dict[str, int] = {}
    for lines in pages:
        keys = set()
        for ln in lines:
            if ln.get("c", 1) < 0.5:
                continue
            y, h = ln["y"], ln.get("h", 0.0)
            edge = y + h < TOP or y > BOTTOM
            wide = y + h < TOP_P or y > BOTTOM_P
            for m in BRAND.finditer(ln["t"]) if edge else ():
                w = m.group(0).lower()
                if w.lstrip("@") not in said:
                    keys.add(w)
            for m in PROMO.finditer(ln["t"]) if wide else ():
                w = m.group(0).lower().strip()
                if w and w not in said:
                    keys.add(w)
        for k in keys:
            seen[k] = seen.get(k, 0) + 1
    return sorted(seen)


STOPWORDS = {"the", "and", "that", "this", "with", "have", "from", "they", "what", "were", "when", "your", "just", "like", "there", "their", "about", "would", "been",
             "then", "than", "them", "will", "into", "over", "only", "some", "very", "because", "which", "could", "should", "where", "while"}


def _words(t: str) -> set[str]:
    """The words of a caption line that say something: 4 letters or more and not a stopword (so "THE ... THE" on two lines of one caption is not a match)."""
    return {w for w in re.findall(r"[a-z0-9']+", t.lower()) if len(w) >= 4 and w not in STOPWORDS}


def double_captions(pages: list[list[dict]] | None) -> bool:
    """Two sets of captions at once (ours at 60-80% of the height, plus the source's own burned in lower down, or the other way round): in the caption zone, two
    lines at least DOUBLE_GAP apart that share a word - it is the same speech written twice. The two lines of one wrapped caption sit closer than that."""
    if not pages:
        return False
    hits = 0
    for lines in pages:
        zone = [ln for ln in lines if ln.get("c", 1) >= 0.5 and CAP_TOP <= ln["y"] <= CAP_BOTTOM and len(ln["t"].strip()) >= 3]
        found = any(abs(a["y"] - b["y"]) >= DOUBLE_GAP and _words(a["t"]) & _words(b["t"]) for i, a in enumerate(zone) for b in zone[i + 1:])
        hits += found
    return hits >= DOUBLE_FRAMES


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
        dur = ffmpeg.duration(video)
        pages = _pages(video, dur) if brand and dur >= 4 else None
        found = branding(video, dur, spoken, pages=pages) if brand else None
        if found:
            issues.append({"level": "bad", "what": "another channel's promo is burned into the picture (" + ", ".join(found[:3]) + ")"})
        if strict and double_captions(pages):
            issues.append({"level": "bad", "what": "two sets of captions on top of each other (ours and the source's own)"})
    except Exception as e:  # noqa: BLE001
        log.debug(f"branding check of {video.name} failed: {e}")
    return issues


def qa_field(video: Path, spoken: str = "", brand: bool = True) -> dict:
    """The `qa` entry of a clip record ({"issues": [...]}) for clips that do not go through the Clipper's own render (originals, news reels)."""
    return {"issues": check(video, spoken, brand=brand, strict=False)}
