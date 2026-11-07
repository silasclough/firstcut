"""Look at the SOURCE before a clip is cut from it, so a bad clip is never made (clipgate.py / opening.py stay as the last look at the finished file).

Learned from the channel's own numbers (October 2026) and the clips behind them:

    still     a ranger story whose "video" is one picture of an office for half an hour: the clip was that picture with captions (8 views).
              `probe_source` samples the whole file (a few short windows) and `screen_moments` the stretch of every candidate, with the same motion
              measure opening.py uses on the result, so a still picture is dropped before it is rendered, and a source that is only stills is skipped.
    strips    a talking head with the creator's own "SUBSCRIBE" bar, @handle, phone number and name tag across the top / bottom (6 views). `burned_in`
              reads a few frames of the stretch and `crop_plan` says how much of the top and bottom to cut away; the render crops it out (the 16:9 to
              9:16 reframe throws most of the width away anyway), so the finished clip has no promo for the gate to refuse.
    captions  the source's own small captions under ours (double captions). Cropped away like a strip when they sit at the bottom, else ours are left off.
    gameplay  real-life content only: a source (YouTube category, tags, title; a Twitch clip's game) is gameplay unless the streamer is just talking to the camera
              (Just Chatting, IRL, music, art).

Everything is ffmpeg + numpy + the text reader (ocr.py); where the reader is missing the text checks are skipped, never failed.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger
from . import clipgate, ocr, opening

log = get_logger("clipper.sourcecheck")

PROBE_WINDOWS, PROBE_SECONDS = 8, 5.0       # a whole-source look: this many short windows spread over the file
SOURCE_STILL_SHARE = 0.8                    # share of those windows that are a still picture, above which the source IS a still picture
WINDOW_STILL = 0.9                          # a window whose picture hardly moves for this share of it is "still"
MAX_CROP = 0.30                             # never cut away more than this share of the picture (top + bottom together); past it the clip is left for the gate
CAP_BOTTOM = 0.70                           # captions below this line are cropped away; captions higher up make us leave ours off instead
FRAMES = 6

PHONE = re.compile(r"(?:\+?\d{1,2}[ .-])?\(?\d{3}\)?[ .-]\d{3,4}[ .-]?\d{0,4}")
PROMO = re.compile(opening.BRAND.pattern + r"|\bwww\b|\bhttps?\b|let'?s connect|\bcall (?:me|us)\b|\bdm (?:me|us)\b|\bjoin (?:my|our|the)\b|\bcourse\b|\bbook a\b|\bfree (?:guide|training|ebook)\b|\bsign ?up\b", re.I)

GAME_CATEGORIES = {"gaming", "video games", "games"}
GAME_WORDS, GAME_TAGS = clipgate.GAME_WORDS, clipgate.GAME_TAGS      # one list for the last look and the source check
TWITCH_IRL = {"509658", "509672", "417752", "509663", "26936", "509660", "509667", "518203", "509659", "516575"}     # Just Chatting, IRL, Talk Shows, Special Events, Music, Art, ASMR, Sports, Travel & Outdoors, Food & Drink


# ------------------------------------------------------------------ a picture that does not move
def still(m: dict, dur: float) -> bool:
    """Is this motion reading (opening.motion) a picture that does not move? The same rule opening.check applies to the finished clip."""
    return bool(m.get("still_share", 0.0) >= opening.STILL_CLIP or (m.get("still_open", 0.0) >= opening.STILL_OPEN and dur > 8))


def motion_of(src: Path, start: float, end: float) -> dict:
    """The picture of the stretch [start, end] of the source: opening.motion's readings plus "still" (True/False) and "dur". Unreadable -> still False (never block on a doubt)."""
    dur = max(0.0, end - start)
    try:
        frames = opening._gray(Path(src), max(0.0, start), start + dur)
        m = opening.motion(frames)
    except Exception as e:  # noqa: BLE001 - a check is never a reason to lose a source
        log.debug(f"motion check of {Path(src).name} {start:.0f}-{end:.0f} failed: {e}")
        return {"still_share": 0.0, "still_open": 0.0, "blank_open": False, "still": False, "dur": dur}
    return {**m, "still": still(m, dur), "dur": dur}


def probe_source(src: Path, duration: float | None = None, n: int = PROBE_WINDOWS, win: float = PROBE_SECONDS) -> dict:
    """-> {"still": the whole source is (nearly) one picture, "windows": how many were looked at, "still_windows": how many of them did not move}."""
    try:
        dur = duration or ffmpeg.duration(Path(src))
    except Exception:  # noqa: BLE001
        return {"still": False, "windows": 0, "still_windows": 0}
    if dur < win * 2:
        return {"still": False, "windows": 0, "still_windows": 0}
    starts = [max(0.0, (dur - win) * (k + 0.5) / n) for k in range(n)]
    seen = stills = 0
    for t in starts:
        m = motion_of(src, t, t + win)
        if m["dur"] <= 0:
            continue
        seen += 1
        stills += 1 if m.get("still_share", 0.0) >= WINDOW_STILL else 0
    return {"still": seen >= 3 and stills / seen >= SOURCE_STILL_SHARE, "windows": seen, "still_windows": stills}


def screen_moments(src: Path, found: list, limit: int = 30) -> tuple[list, list[tuple[object, str]]]:
    """Look at the stretch of each candidate moment (the `limit` best by score; the rest are not looked at and stay).
    -> (the moments to keep, [(moment, why) dropped]). A candidate whose picture does not move is dropped before anything is rendered."""
    ranked = sorted(found, key=lambda m: -float(getattr(m, "score", 0) or 0))[:limit]
    checked = {id(m) for m in ranked}
    kept, dropped = [], []
    for m in found:
        if id(m) not in checked:
            kept.append(m)
            continue
        r = motion_of(src, float(m.start), float(m.end))
        if r["still"]:
            dropped.append((m, f"the picture does not move ({r['still_share'] * 100:.0f}% still)"))
        else:
            kept.append(m)
    return kept, dropped


# ------------------------------------------------------------------ text burned into the source
def _words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", s.lower()) if len(w) > 2}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def frames_of(src: Path, start: float, end: float, td: Path, n: int = FRAMES) -> list[Path]:
    out = []
    dur = max(0.5, end - start)
    for k in range(n):
        p = td / f"s{k}.jpg"
        subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{max(0.0, start + dur * (k + 0.5) / n):.2f}", "-i", str(src), "-frames:v", "1",
                        "-vf", "scale=640:-2", "-q:v", "3", str(p)], capture_output=True, timeout=60)
        if p.exists():
            out.append(p)
    return out


def analyse(pages: list[list[dict]], spoken: str = "") -> dict:
    """The decision on OCR pages (one list of {"t","x","y","w","h","c"} per sampled frame, positions as shares of the picture). Pure: tested with fake pages.
    -> {"promo": [texts], "top": share of the picture to cut from the top, "bottom": same from the bottom, "captions": bool, "cap_band": (y0, y1) or None}"""
    said = _words(spoken)
    n = max(1, len(pages))
    seen: dict[str, dict] = {}                                       # a line that sits in an outer strip, by its letters
    cap_rows: list[list[dict]] = []
    for lines in pages:
        row = []
        for ln in lines:
            if ln.get("c", 1) < 0.5 or not str(ln.get("t", "")).strip():
                continue
            t, y, h = str(ln["t"]).strip(), float(ln["y"]), float(ln["h"])
            w = _words(t)
            spoken_line = bool(w) and len(w & said) / len(w) >= 0.6
            row.append({**ln, "t": t, "spoken": spoken_line})
            low = y + h < 0.25 or y > 0.62                           # in the top quarter or below the middle: where creators put bars and tags
            if not low:
                continue
            e = seen.setdefault(_key(t), {"t": t, "y0": y, "y1": y + h, "n": 0, "spoken": spoken_line, "promo": False})
            e["n"] += 1
            e["y0"], e["y1"] = min(e["y0"], y), max(e["y1"], y + h)
            if (PROMO.search(t) or PHONE.search(t)) and not spoken_line:
                e["promo"] = True
        cap_rows.append(row)
    promo = [e for e in seen.values() if e["promo"] or (e["n"] >= max(2, n // 2) and not e["spoken"] and (e["y1"] < 0.25 or e["y0"] > 0.7))]     # a promo word, or a tag that stays put
    top = max([e["y1"] for e in promo if e["y1"] < 0.30] or [0.0])
    bottom = max([1.0 - e["y0"] for e in promo if e["y0"] > 0.62] or [0.0])
    # the source's own captions: centred lines in the lower half whose words change from frame to frame and are what is being said
    frames_with = 0
    texts: set[str] = set()
    ys: list[tuple[float, float]] = []
    for row in cap_rows:
        hit = [ln for ln in row if 0.45 <= ln["y"] <= 0.93 and 0.25 <= ln["x"] + ln["w"] / 2 <= 0.75 and ln["w"] >= 0.10 and (ln["spoken"] or not said)]
        if hit:
            frames_with += 1
            texts |= {_key(ln["t"]) for ln in hit}
            ys += [(float(ln["y"]), float(ln["y"]) + float(ln["h"])) for ln in hit]
    captions = frames_with >= max(2, n // 2) and len(texts) >= 3
    band = (min(a for a, _ in ys), max(b for _, b in ys)) if captions and ys else None
    out = {"promo": sorted({e["t"] for e in promo if e["promo"]})[:5], "top": round(top + 0.012, 3) if top else 0.0,
           "bottom": round(bottom + 0.012, 3) if bottom else 0.0, "captions": bool(captions), "cap_band": band}
    return out


def burned_in(src: Path, start: float, end: float, spoken: str = "") -> dict | None:
    """Read FRAMES pictures of the stretch -> analyse(). None when nothing can read text here (the text checks are skipped, not failed)."""
    with tempfile.TemporaryDirectory() as td:
        shots = frames_of(Path(src), start, end, Path(td))
        pages = ocr.read(shots) if shots else []
    if pages is None:
        return None
    return analyse(pages, spoken)


def crop_plan(found: dict | None) -> dict:
    """What the render should do about what `burned_in` found -> {"top": share to crop from the top, "bottom": from the bottom, "own_captions": False|True (leave ours off)}.
    Bars and tags are cropped out; the source's own bottom captions too; captions up in the picture cannot be cropped away, so ours are left off. Past MAX_CROP nothing is cropped."""
    plan = {"top": 0.0, "bottom": 0.0, "own_captions": True}
    if not found:
        return plan
    top, bottom = float(found.get("top") or 0.0), float(found.get("bottom") or 0.0)
    if found.get("captions") and found.get("cap_band"):
        y0, y1 = found["cap_band"]
        if y0 >= CAP_BOTTOM:
            bottom = max(bottom, 1.0 - y0 + 0.012)
        else:
            plan["own_captions"] = False
    if top + bottom > MAX_CROP:
        log.info(f"the burned-in text would need {top + bottom:.0%} of the picture cut away: left as it is")
        return {"top": 0.0, "bottom": 0.0, "own_captions": plan["own_captions"] and not found.get("captions")}      # (the source's captions stay in the picture and are readable: ours would be a second set)
    plan["top"], plan["bottom"] = round(top, 3), round(bottom, 3)
    return plan


def crop_filter(sh: int, plan: dict, even: int = 2) -> tuple[str, int, int]:
    """-> (the ffmpeg crop to put right after the trim: "" when nothing is cropped, the picture's new height, how many pixels are cut from the top). Width is untouched."""
    top, bottom = float(plan.get("top") or 0.0), float(plan.get("bottom") or 0.0)
    if top + bottom <= 0.0:
        return "", sh, 0
    y0 = int(round(sh * top)) // even * even
    h = (int(round(sh * (1.0 - top - bottom))) // even) * even
    if h < even * 16 or h >= sh:
        return "", sh, 0
    return f",crop=iw:{h}:0:{y0}", h, y0


# ------------------------------------------------------------------ gameplay
def gameplay_meta(info: dict | None, extra_words: list[str] | None = None) -> str | None:
    """-> why a YouTube source is gameplay (None = real life / unknown): its category, tags or title. `info` is what yt-dlp returned."""
    info = info or {}
    cats = {str(c).lower() for c in (info.get("categories") or [])}
    if cats & GAME_CATEGORIES:
        return "category: " + ", ".join(sorted(cats & GAME_CATEGORIES))
    tags = [str(t).strip().lower() for t in (info.get("tags") or [])[:25]]
    if set(tags) & GAME_TAGS:
        return "gameplay tag: " + ", ".join(sorted(set(tags) & GAME_TAGS))
    text = " ".join([str(info.get("title") or ""), " ".join(tags)])
    hit = GAME_WORDS.search(text)
    if hit:
        return f"gameplay word: {hit.group(0)}"
    for w in extra_words or []:
        if w and re.search(rf"\b{re.escape(str(w).lower())}\b", text.lower()):
            return f"gameplay word: {w}"
    return None


def gameplay_twitch(clip: dict) -> str | None:
    """-> why a Twitch clip is gameplay: it was made in a game, not in Just Chatting / IRL / music / art. No game id -> None (not known, not blocked)."""
    gid = str(clip.get("game_id") or "").strip()
    if not gid or gid in TWITCH_IRL:
        return None
    return f"game {clip.get('game_name') or gid}"
