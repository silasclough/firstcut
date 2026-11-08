"""Learn how a video is EDITED (not what it says) from a link or a file, as numbers, and write it as editing settings plus a few plain lines.

    fetch    stylefetch.py brings the videos down (YouTube, an Instagram reel when it will come without a login, or a dropped file).
    measure  plain code on a small copy (540 px high at most, the first three minutes): how long shots last and how many cuts there are, how the
             first three seconds are cut, how long the pauses between sentences are, whether music sits under the talking, how often the picture
             zooms or shakes, whether and where captions are, how many words at a time and in what colour, and how much of the time a face is on
             screen. Anything that could not be measured stays empty and is said so ("not measured"): nothing is guessed.
    combine  several videos (a channel, or more added to a preset later): the median and the spread of every number; what repeats is the style.
    write    the numbers become a partial style object (styles_profile.py, checked and clamped there) and the plain lines the person reads: one cheap
             AI call that may only state what was measured, with a template that writes them from the numbers when the call fails.

What is kept: the numbers of each video, its title and link, and up to 8 small frames. Never the video, its transcript or its words. The downloads are
deleted when the measuring is done (the person's own files are never touched). Nothing heavy: ffmpeg, numpy, the text reader and, if it is on disk, the
face model, on a small copy.
"""

from __future__ import annotations

import colorsys
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger
from . import cameramotion, footage, ocr, opening, stylefetch
from . import styles_profile as SPF

log = get_logger("clipper.stylelearn")

ANALYSE_S = 180                   # the first three minutes of each video are measured
PROXY_H = 540                     # the small copy is at most this high
MIN_SHOT_S = 0.25
CAPTION_FRAMES = 10               # frames read for captions
KEEP_FRAMES = 8                   # small frames kept with the preset (they help the AI call, and the person)
CONSISTENT = 0.5                  # (high - low) / median at or below this: the number repeats across the videos, so it is the style
TASK = "style_learn"

NUMBERS = ("duration_s", "cuts_per_30s", "shot_median_s", "shot_p25_s", "shot_p75_s", "shot_p90_s", "shot_max_s", "hook_cuts", "speech_ratio",
           "pause_median_s", "pause_p90_s", "speech_db", "gap_db", "zoom_per_30s", "shake", "moves_share", "caption_share", "caption_words_median",
           "caption_words_p25", "caption_words_p75", "caption_y", "caption_height", "face_share", "face_ratio", "no_face_share")
WORDS = ("aspect", "music", "caption_case", "caption_colour")


# ------------------------------------------------------------------------------------------------------------ small helpers
def _q(values: list[float], q: float) -> float | None:
    return round(float(np.percentile(values, q)), 3) if values else None


def _run(cmd: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, timeout=timeout)


def make_proxy(src: Path, dst: Path, secs: float = ANALYSE_S) -> Path:
    """A small copy of the first `secs` seconds (picture and sound) in `dst`; the source is only read."""
    base = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-t", f"{secs:.1f}", "-i", str(src), "-vf", f"scale=-2:'min({PROXY_H},ih)':flags=bilinear",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", "-pix_fmt", "yuv420p"]
    r = _run([*base, "-c:a", "aac", "-b:a", "64k", "-ac", "1", "-ar", "16000", str(dst)], 600)
    if r.returncode != 0 or not dst.exists():
        _run([*base, "-an", str(dst)], 600)                                      # (a file with no sound track: the picture only)
    if not dst.exists():
        raise stylefetch.FetchError("that video could not be opened")
    return dst


def _size(path: Path) -> tuple[int, int, float]:
    info = ffmpeg.probe(path)
    vs = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), {})
    return int(vs.get("width") or 0), int(vs.get("height") or 0), float(ffmpeg.duration(path))


def aspect_name(w: int, h: int) -> str:
    if not w or not h:
        return "unknown"
    r = w / h
    return "9:16" if r < 0.7 else "4:5" if r < 0.9 else "1:1" if r < 1.15 else "16:9" if r < 2.0 else "wide"


# ------------------------------------------------------------------------------------------------------------ the measures
def shot_stats(cuts: list[float], dur: float) -> dict:
    shots = footage.shots_from_cuts(cuts, dur, MIN_SHOT_S)
    lens = [s["end"] - s["start"] for s in shots]
    n_cuts = max(0, len(shots) - 1)
    return {"cuts_per_30s": round(n_cuts / max(1.0, dur) * 30.0, 2), "shot_median_s": _q(lens, 50), "shot_p25_s": _q(lens, 25), "shot_p75_s": _q(lens, 75),
            "shot_p90_s": _q(lens, 90), "shot_max_s": round(max(lens), 2) if lens else None,
            "hook_cuts": sum(1 for s in shots[1:] if s["start"] < 3.0)}


def speech_stats(segs: list[list[float]], dur: float) -> dict:
    """The talking and the pauses between sentences (from the speech stretches). No speech at all gives empty pauses."""
    spoken = sum(max(0.0, b - a) for a, b in segs)
    pauses = [b2[0] - b1[1] for b1, b2 in zip(segs, segs[1:]) if b2[0] - b1[1] > 0.0]
    return {"speech_ratio": round(spoken / max(0.5, dur), 3), "pause_median_s": _q(pauses, 50) if len(pauses) >= 3 else None,
            "pause_p90_s": _q(pauses, 90) if len(pauses) >= 3 else None}


def audio_levels(path: Path, dur: float, segs: list[list[float]]) -> dict:
    """Is there a steady sound bed (music) in the gaps between talking, and how far under the talking is it? From the loudness of every tenth of a second."""
    empty = {"music": None, "speech_db": None, "gap_db": None}
    try:
        r = _run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-t", f"{dur:.2f}", "-i", str(path), "-vn", "-ac", "1", "-ar", "8000", "-f", "f32le", "-"], 300)
    except (OSError, subprocess.SubprocessError):
        return empty
    a = np.frombuffer(r.stdout[: len(r.stdout) // 4 * 4], np.float32)
    if len(a) < 8000:
        return empty
    step = 800                                                                    # 0.1 s
    n = len(a) // step
    rms = np.sqrt((a[: n * step].reshape(n, step) ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-9)
    mask = np.zeros(n, bool)
    for s, e in segs:
        mask[max(0, int(s * 10) - 1): min(n, int(e * 10) + 1)] = True
    gap = db[~mask]
    spk = db[mask]
    out = dict(empty)
    if len(spk) >= 5:
        out["speech_db"] = round(float(np.median(spk)), 1)
    if len(gap) >= 4:
        g = float(np.median(gap))
        out["gap_db"] = round(g, 1)
        steady = float(np.std(gap)) < 6.0
        out["music"] = "under" if (g > -50.0 and steady and len(spk) >= 5) else "none"
    if len(spk) < 5 and len(db) >= 10:                                          # nobody talks: a music video, or silence
        out["music"] = "only" if float(np.median(db)) > -45.0 else "none"
    return out


def motion_stats(path: Path, dur: float, w: int, h: int) -> dict:
    """How often the picture zooms (a punch-in or a push) and how much it shakes: the camera-motion measure the Find footage heat map uses, on 2 frames
    a second. Empty when it cannot be measured."""
    empty = {"zoom_per_30s": None, "shake": None, "moves_share": None}
    if not w or not h:
        return empty
    fw = 192
    fh = max(2, int(round(fw * h / w / 2)) * 2)
    try:
        r = _run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-t", f"{dur:.2f}", "-i", str(path), "-an", "-vf", f"fps=2,scale={fw}:{fh},format=gray", "-f",
                  "rawvideo", "-"], 300)
        raw = np.frombuffer(r.stdout[: len(r.stdout) // (fw * fh) * fw * fh], np.uint8)
        frames = raw.reshape(-1, fh, fw)
        if len(frames) < 6:
            return empty
        times = [i / 2.0 for i in range(len(frames))]
        m = cameramotion.analyze(times, frames)
        prof = cameramotion.profile(m)
    except Exception as e:  # noqa: BLE001 - no OpenCV or an odd file: not measured
        log.debug(f"motion not measured: {e}")
        return empty
    if prof is None:
        return empty
    z = m[:, cameramotion.LOGS].astype(np.float64)
    events, k = 0, 0
    while k < len(z) - 1:                                                       # a punch-in or push: the picture grows (or shrinks) by 5%+ within 1.5 s
        win = z[k + 1: k + 4]
        if abs(float(win.sum())) >= 0.05:
            events += 1
            k += 4
        else:
            k += 1
    return {"zoom_per_30s": round(events / max(1.0, dur) * 30.0, 2), "shake": round(float(np.mean(prof["shake"])), 3),
            "moves_share": round(float(np.mean(prof["move"] >= 0.5)), 3)}


COLOURS = (("red", 0.0), ("orange", 0.06), ("yellow", 0.15), ("green", 0.33), ("cyan", 0.5), ("blue", 0.62), ("purple", 0.78), ("pink", 0.9))


def colour_name(rgb: tuple[int, int, int]) -> str:
    h, s, v = colorsys.rgb_to_hsv(*(c / 255.0 for c in rgb))
    if v < 0.25:
        return "black"
    if s < 0.18:
        return "white" if v > 0.7 else "grey"
    best = min(COLOURS, key=lambda c: min(abs(h - c[1]), 1 - abs(h - c[1])))
    return best[0]


def line_colour(img: np.ndarray, box: tuple[float, float, float, float]) -> str | None:
    """The colour of the words inside a text box (x, y, w, h as shares of the picture): the brightest saturated colour if a clear part of the words has
    one (a highlight), else white or black. None when the box is too small to say."""
    ih, iw = img.shape[:2]
    x0, y0 = max(0, int(box[0] * iw)), max(0, int(box[1] * ih))
    x1, y1 = min(iw, int((box[0] + box[2]) * iw)), min(ih, int((box[1] + box[3]) * ih))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return None
    crop = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32) / 255.0
    mx, mn = crop.max(axis=1), crop.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    vivid = (sat > 0.55) & (mx > 0.6)
    if vivid.mean() > 0.06:
        c = crop[vivid].mean(axis=0) * 255
        return colour_name((int(c[0]), int(c[1]), int(c[2])))
    bright = mx > 0.85
    dark = mx < 0.2
    if bright.mean() > dark.mean():
        return "white"
    return "black"


def caption_stats(video: Path, dur: float, td: Path, ocr_read: Callable = ocr.read) -> dict:
    """Are there burned-in captions, where, how many words at a time, how tall, what case and what highlight colour. Looks at 10 frames with the text reader;
    None for everything when no reader is available."""
    empty = {k: None for k in ("caption_share", "caption_words_median", "caption_words_p25", "caption_words_p75", "caption_y", "caption_height", "caption_case",
                               "caption_colour")}
    shots: list[Path] = []
    for k in range(CAPTION_FRAMES):
        t = dur * (0.05 + 0.9 * k / max(1, CAPTION_FRAMES - 1))
        p = td / f"cap{k}.jpg"
        _run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=540:-2", "-q:v", "3", str(p)], 60)
        if p.exists():
            shots.append(p)
    if len(shots) < 3:
        return empty
    try:
        pages = ocr_read(shots)
    except Exception as e:  # noqa: BLE001
        log.debug(f"text reader failed: {e}")
        pages = None
    if pages is None or len(pages) != len(shots):
        return empty
    from PIL import Image
    groups: list[dict] = []
    colours: list[str] = []
    for shot, lines in zip(shots, pages):
        cap = []
        for ln in lines:
            t = str(ln.get("t") or "").strip()
            if ln.get("c", 1) < 0.5 or not t or opening.BRAND.search(t):
                continue
            yc = ln["y"] + ln["h"] / 2
            if yc < 0.3 or yc > 0.93:                                          # (a title near the top, a watermark at the very bottom)
                continue
            cap.append(ln)
        if not cap:
            continue
        low = max(c["y"] + c["h"] / 2 for c in cap)
        grp = [c for c in cap if low - (c["y"] + c["h"] / 2) <= 0.12]           # the lines that belong together (the lowest block)
        words = sum(len(str(c["t"]).split()) for c in grp)
        letters = [ch for c in grp for ch in str(c["t"]) if ch.isalpha()]
        upper = (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0
        groups.append({"words": words, "y": float(np.mean([c["y"] + c["h"] / 2 for c in grp])), "h": float(np.mean([c["h"] for c in grp])), "upper": upper})
        try:
            img = np.asarray(Image.open(shot).convert("RGB"))
            for c in grp:
                col = line_colour(img, (c["x"], c["y"], c["w"], c["h"]))
                if col:
                    colours.append(col)
        except Exception:  # noqa: BLE001
            pass
    share = len(groups) / len(shots)
    out = dict(empty)
    out["caption_share"] = round(share, 3)
    if groups and share >= 0.2:
        ws = [g["words"] for g in groups]
        out.update({"caption_words_median": _q(ws, 50), "caption_words_p25": _q(ws, 25), "caption_words_p75": _q(ws, 75),
                    "caption_y": round(float(np.median([g["y"] for g in groups])), 3), "caption_height": round(float(np.median([g["h"] for g in groups])), 3),
                    "caption_case": "upper" if np.mean([g["upper"] for g in groups]) >= 0.8 else "mixed"})
        if colours:
            out["caption_colour"] = max(set(colours), key=colours.count)
    return out


def face_stats(path: Path, dur: float, w: int, h: int, models_dir=None, detector=None) -> dict:
    empty = {"face_share": None, "face_ratio": None, "no_face_share": None, "faces_measured": False}
    try:
        samples, measured = footage.analyse_samples(path, min(dur, 60.0), w, h, models_dir if models_dir is not None else footage.default_models_dir(), detector)
    except Exception as e:  # noqa: BLE001
        log.debug(f"faces not measured: {e}")
        return empty
    if not measured or not samples:
        return empty
    share = sum(1 for s in samples if s["faces"]) / len(samples)
    fs = footage.faces_summary(samples, None)
    return {"face_share": round(share, 3), "face_ratio": fs["largest_ratio_median"] or None, "no_face_share": round(1.0 - share, 3), "faces_measured": True}


def measure(video: Path, *, ocr_read: Callable = ocr.read, models_dir=None, detector=None, td: Path | None = None) -> dict:
    """Every number for ONE video (a small copy of it, see make_proxy). Keys are NUMBERS and WORDS; a number that could not be measured is None."""
    w, h, dur = _size(video)
    dur = min(dur, float(ANALYSE_S))
    out: dict[str, Any] = {k: None for k in NUMBERS}
    out.update({k: None for k in WORDS})
    out.update({"duration_s": round(dur, 2), "aspect": aspect_name(w, h), "width": w, "height": h, "faces_measured": False})
    if dur < 2.0:
        raise stylefetch.FetchError("that video is too short to learn from")
    scan = footage.scan_pass(video, dur)
    out.update(shot_stats(scan.get("cuts") or [], dur))
    segs = footage.vad(video, dur)
    out.update(speech_stats(segs, dur))
    out.update(audio_levels(video, dur, segs))
    out.update(motion_stats(video, dur, w, h))
    own = td is None
    tdir = Path(tempfile.mkdtemp(prefix="firstcut-cap-")) if own else td
    try:
        out.update(caption_stats(video, dur, tdir, ocr_read))
    finally:
        if own:
            import shutil
            shutil.rmtree(tdir, ignore_errors=True)
    out.update(face_stats(video, dur, w, h, models_dir, detector))
    return out


# ------------------------------------------------------------------------------------------------------------ combine several videos
def aggregate(measurements: list[dict]) -> dict:
    """Median and spread of every number over the videos, and the commonest word for the word-like ones. `consistent` marks what repeats (the style)
    against what varies from one video to the next."""
    ms = [m for m in measurements if isinstance(m, dict)]
    nums: dict[str, dict] = {}
    for k in NUMBERS:
        vals = [float(m[k]) for m in ms if isinstance(m.get(k), (int, float)) and not isinstance(m.get(k), bool)]
        if not vals:
            continue
        med, lo, hi = float(np.median(vals)), min(vals), max(vals)
        nums[k] = {"median": round(med, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": len(vals),
                   "consistent": len(vals) < 2 or (hi - lo) <= CONSISTENT * max(abs(med), 0.05)}
    words: dict[str, dict] = {}
    for k in WORDS:
        vals = [str(m[k]) for m in ms if m.get(k)]
        if vals:
            top = max(set(vals), key=vals.count)
            words[k] = {"value": top, "share": round(vals.count(top) / len(vals), 2), "n": len(vals)}
    return {"videos": len(ms), "numbers": nums, "words": words, "faces_measured": any(m.get("faces_measured") for m in ms)}


def _med(agg: dict, key: str) -> float | None:
    v = (agg.get("numbers") or {}).get(key)
    return None if not v else float(v["median"])


def _word(agg: dict, key: str) -> str | None:
    v = (agg.get("words") or {}).get(key)
    return None if not v else str(v["value"])


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def to_style(agg: dict) -> tuple[dict, list[str]]:
    """The editing settings the measures state, as a partial style object that styles_profile accepts. -> (partial, notes). Values are clamped to the
    ranges the style allows; the whole partial is then checked and anything the check refuses is dropped (with a note), never passed on."""
    p: dict = {}
    notes: list[str] = []
    cuts = _med(agg, "cuts_per_30s")
    pace: dict = {}
    if cuts is not None:
        pace["name"] = "fast" if cuts >= 14 else "tight" if cuts >= 8 else "natural" if cuts >= 3 else "slow"
    p90 = _med(agg, "shot_p90_s")
    if p90 is not None:
        pace["max_hold_s"] = round(_clamp(p90 * 1.2, 2.5, 30.0), 1)
    pm = _med(agg, "pause_median_s")
    if pm is not None and pm > 0:
        pace["gap_shrink_to_s"] = round(_clamp(pm, 0.12, 0.6), 2)
    if pace:
        p["pace"] = pace
    lo, hi = _med(agg, "shot_p25_s"), _med(agg, "shot_p75_s")
    if lo is not None and hi is not None:
        a = _clamp(lo, 0.5, 20.0)
        b = _clamp(max(hi, a), a, 30.0)
        p["broll"] = {"shot_s": [round(a, 1), round(b, 1)]}
    nf = _med(agg, "no_face_share")
    if nf is not None:
        p.setdefault("broll", {})["share"] = round(_clamp(nf, 0.0, 1.0), 2)
    cs = _med(agg, "caption_share")
    if cs is not None:
        if cs < 0.2:
            p["captions"] = {"style": "none"}
        else:
            w50, w25, w75 = _med(agg, "caption_words_median"), _med(agg, "caption_words_p25"), _med(agg, "caption_words_p75")
            cap: dict = {"style": "karaoke" if (w50 is not None and w50 <= 3) else "phrase"}
            if w25 is not None and w75 is not None:
                a = int(_clamp(round(w25), 1, 8))
                b = int(_clamp(round(w75), a, 12))
                cap["words_per_group"] = [a, b]
            y = _med(agg, "caption_y")
            if y is not None:
                cap["band_y"] = [round(_clamp(y - 0.05, 0.05, 0.9), 2), round(_clamp(y + 0.05, 0.1, 0.95), 2)]
            if _word(agg, "caption_case"):
                cap["case"] = _word(agg, "caption_case")
            col = _word(agg, "caption_colour")
            if col:
                cap["highlight"] = "one_colour" if col not in ("white", "black", "grey") else "none"
            p["captions"] = cap
    mus = _word(agg, "music")
    if mus:
        p["music"] = {"policy": "throughout" if mus in ("under", "only") else "none"}
    if agg.get("videos") and (_word(agg, "aspect") in ("9:16", "4:5", "1:1")):
        d = _med(agg, "duration_s")
        if d is not None and 7 <= d <= 90:
            p["length"] = {"target_s": int(round(d))}
    # check the whole partial; drop a section the check refuses, one at a time
    base = SPF.resolve("short_talk" if _word(agg, "aspect") in ("9:16", "4:5") else "longform_talk")
    for _ in range(len(p) + 1):
        _new, problems = SPF.apply_partial(base, p)
        if not problems:
            break
        bad = next((k for k in list(p) if any(k in pr for pr in problems)), None) or next(iter(p), None)
        if bad is None:
            break
        notes.append(f"left out {bad}: {problems[0]}")
        p.pop(bad, None)
    return p, notes


# ------------------------------------------------------------------------------------------------------------ the words
def _d(v: float | None, nd: int = 1) -> str:
    return "?" if v is None else f"{v:.{nd}f}".rstrip("0").rstrip(".") if nd else f"{v:.0f}"


def evidence(agg: dict) -> list[str]:
    """What was measured, in plain lines (shown next to the points so the person can see why they say what they say)."""
    n = agg.get("videos", 0)
    out: list[str] = []
    cuts, med = _med(agg, "cuts_per_30s"), _med(agg, "shot_median_s")
    if cuts is not None:
        out.append(f"{_d(cuts)} cuts in 30 s; the middle shot lasts {_d(med)} s, the longest hold is about {_d(_med(agg, 'shot_p90_s'))} s.")
    hc = _med(agg, "hook_cuts")
    if hc is not None:
        out.append(f"{_d(hc, 0)} cut{'s' if round(hc) != 1 else ''} in the first 3 seconds.")
    pm = _med(agg, "pause_median_s")
    sr = _med(agg, "speech_ratio")
    if sr is not None:
        out.append(f"Someone talks {sr * 100:.0f}% of the time" + (f"; pauses between sentences are about {_d(pm, 2)} s." if pm is not None else "."))
    mus = _word(agg, "music")
    if mus:
        sd, gd = _med(agg, "speech_db"), _med(agg, "gap_db")
        out.append({"under": "Music or a steady sound runs under the talking" + (f" (the talking is about {sd - gd:.0f} dB above it)." if sd is not None and gd is not None else "."),
                    "only": "Music, no talking.", "none": "No music under the talking."}.get(mus, ""))
    zp = _med(agg, "zoom_per_30s")
    if zp is not None:
        out.append(f"{_d(zp)} zoom or punch-in moves in 30 s; shake {_d(_med(agg, 'shake'), 2)} (0 is steady).")
    cs = _med(agg, "caption_share")
    if cs is not None:
        if cs < 0.2:
            out.append("No burned-in captions.")
        else:
            col = _word(agg, "caption_colour")
            y = _med(agg, "caption_y")
            out.append(f"Captions on screen {cs * 100:.0f}% of the time, about {_d(_med(agg, 'caption_words_median'), 0)} words at a time"
                       + (", in the lower part of the picture" if y is not None and y >= 0.6 else ", around the middle of the picture" if y is not None else "")
                       + (f", {_word(agg, 'caption_case')} case" if _word(agg, "caption_case") else "") + (f", {col} highlight" if col else "") + ".")
    fs = _med(agg, "face_share")
    if fs is not None:
        out.append(f"A face is on screen {fs * 100:.0f}% of the time.")
    asp, dur = _word(agg, "aspect"), _med(agg, "duration_s")
    if asp:
        out.append(f"{asp} picture, {_d(dur, 0)} s measured per video, {n} video{'s' if n != 1 else ''}.")
    if n > 1:
        same = [k for k in ("cuts_per_30s", "shot_median_s", "caption_share", "zoom_per_30s") if (agg["numbers"].get(k) or {}).get("consistent")]
        if same:
            out.append("Repeats from video to video: " + ", ".join(k.replace("_", " ") for k in same) + ".")
    miss = []
    if "face_share" not in (agg.get("numbers") or {}):
        miss.append("faces")
    if "caption_share" not in (agg.get("numbers") or {}):
        miss.append("captions (no text reader here)")
    if "zoom_per_30s" not in (agg.get("numbers") or {}):
        miss.append("zooms and shake")
    miss += ["graphics", "colour grade", "fonts", "sound effects", "words per minute"]
    out.append("Not measured: " + ", ".join(miss) + ".")
    return [x for x in out if x]


def points_from_numbers(agg: dict) -> list[str]:
    """The no-AI way to write the points: a few plain lines from the numbers (used when the AI call fails or is off)."""
    pts: list[str] = []
    cuts, med = _med(agg, "cuts_per_30s"), _med(agg, "shot_median_s")
    if cuts is not None:
        pace = "fast" if cuts >= 14 else "tight" if cuts >= 8 else "natural" if cuts >= 3 else "slow"
        pts.append(f"Cut at a {pace} pace: about every {_d(med)} s, never holding one picture past {_d(_med(agg, 'shot_p90_s'))} s.")
    hc = _med(agg, "hook_cuts")
    if hc is not None and hc >= 1:
        pts.append(f"Open with {_d(hc, 0)} or more cuts in the first 3 seconds.")
    pm = _med(agg, "pause_median_s")
    if pm is not None:
        pts.append(f"Trim pauses between sentences to about {_d(pm, 2)} s.")
    cs = _med(agg, "caption_share")
    if cs is not None:
        if cs < 0.2:
            pts.append("No burned-in captions.")
        else:
            w = _med(agg, "caption_words_median")
            y = _med(agg, "caption_y")
            col = _word(agg, "caption_colour")
            pts.append(f"Captions of about {_d(w, 0)} words at a time" + (", low in the picture" if y is not None and y >= 0.6 else ", mid-picture" if y is not None else "")
                       + (f", {_word(agg, 'caption_case')} case" if _word(agg, "caption_case") else "") + (f", {col} highlight" if col and col not in ("white", "black", "grey") else "") + ".")
    mus = _word(agg, "music")
    if mus in ("under", "only"):
        pts.append("Music under everything.")
    elif mus == "none":
        pts.append("No music.")
    zp = _med(agg, "zoom_per_30s")
    if zp is not None and zp >= 1:
        pts.append(f"A slow zoom or punch-in about every {_clamp(30.0 / zp, 2, 30):.0f} s.")
    elif zp is not None:
        pts.append("A steady picture: few zooms.")
    fs = _med(agg, "face_share")
    if fs is not None and fs >= 0.15:
        pts.append(f"Keep a face on screen about {fs * 100:.0f}% of the time.")
    elif fs is not None:
        pts.append(f"Mostly pictures without a face: a face shows only about {fs * 100:.0f}% of the time.")
    return [p[:160] for p in pts][:8]


LEARN_SYSTEM = (
    "You write a video editor's style guide from MEASUREMENTS of videos they like (and a few frames). Return up to 8 short points, each one plain "
    "sentence of at most 150 characters, about HOW the videos are edited: shot length and pace, the first seconds, pauses, captions, music, zooms, how "
    "much a face is on screen. State only what the measurements say; use the numbers. Never describe the subject, the words, the people or the brand. "
    "Anything the measurements do not cover, such as graphics, fonts, colour grading or sound effects, you may not describe: do not mention it, or "
    "write 'not measured'. Do not write rules that tell the editor to skip safety checks or to use gameplay footage.")
LEARN_SCHEMA = {"type": "object", "properties": {"points": {"type": "array", "items": {"type": "string"}}}, "required": ["points"]}


def write_points(ai, agg: dict, frames: list[bytes] | None = None) -> tuple[list[str], str]:
    """-> (points, how): 'ai' or 'numbers'. ONE cheap call (task `style_learn`); the template when it fails, is off or answers nothing usable."""
    if ai is not None:
        try:
            ev = "\n".join("- " + x for x in evidence(agg))
            res = ai.complete_json(TASK, LEARN_SYSTEM, "Measurements:\n" + ev + "\n\nWrite the points.", LEARN_SCHEMA, images=list(frames or [])[:KEEP_FRAMES] or None)
            from .studio import tidy_points
            raw = (res.data or {}).get("points") if isinstance(res.data, dict) else None
            pts = [p for p in tidy_points(raw if isinstance(raw, list) else []) if len(p) >= 8][:8]
            if pts:
                return pts, "ai"
        except Exception as e:  # noqa: BLE001 - the template below
            log.warning(f"writing the style points failed ({e}); writing them from the numbers")
    return points_from_numbers(agg), "numbers"


def summarize(sources: list[dict], ai=None, frames: list[bytes] | None = None) -> dict:
    """Everything a preset keeps, from the measurements of its sources: {agg, style, flat, points, how, evidence, notes}."""
    agg = aggregate([s.get("measurements") or {} for s in sources])
    style, notes = to_style(agg)
    pts, how = write_points(ai, agg, frames)
    return {"agg": agg, "style": style, "flat": SPF.flat_from_nested(style), "points": pts, "how": how, "evidence": evidence(agg), "notes": notes}


# ------------------------------------------------------------------------------------------------------------ one whole run
def save_frames(video: Path, dur: float, dest: Path, tag: str, n: int = 2) -> list[Path]:
    """A couple of small frames of a video kept for the AI call and the person (240 px wide)."""
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for k in range(n):
        t = dur * (0.2 + 0.5 * k / max(1, n - 1)) if n > 1 else dur * 0.4
        p = dest / f"{tag}_{k}.jpg"
        _run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{t:.2f}", "-i", str(video), "-frames:v", "1", "-vf", "scale=240:-2", "-q:v", "5", str(p)], 60)
        if p.exists():
            out.append(p)
    return out


def learn(source: str, *, ai=None, endpoint: str = "", progress: Callable[[str], None] | None = None, frames_dir: Path | None = None,
          fetch: Callable = stylefetch.fetch, measure_fn: Callable = measure, ocr_read: Callable = ocr.read, work: Path | None = None) -> dict:
    """Fetch the video(s) of `source` (a link or a file path), measure each, delete the downloads. -> {"sources": [{url, title, kind, added, measurements}],
    "frames": [paths]} (summarize() turns the sources into settings and points). Raises stylefetch.FetchError with a plain reason."""
    say = progress or (lambda m: None)
    tmp = work or Path(tempfile.mkdtemp(prefix="firstcut-style-"))
    tmp.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    sources: list[dict] = []
    frames: list[Path] = []
    try:
        items = fetch(source, tmp / "dl", endpoint=endpoint, progress=say)
        for i, it in enumerate(items):
            say(f"Measuring video {i + 1} of {len(items)}" if len(items) > 1 else "Measuring the video")
            try:
                proxy = make_proxy(Path(it["path"]), tmp / f"proxy{i}.mp4")
                m = measure_fn(proxy, ocr_read=ocr_read, td=tmp)
            except stylefetch.FetchError as e:
                log.info(f"skipped one video: {e}")
                continue
            except Exception as e:  # noqa: BLE001 - one odd video never ends the others
                log.warning(f"measuring {it.get('title')} failed: {e}")
                continue
            sources.append({"url": it.get("url") or "", "title": it.get("title") or "", "kind": it.get("kind") or "", "added": int(time.time()), "measurements": m})
            if frames_dir is not None and len(frames) < KEEP_FRAMES:
                frames += save_frames(proxy, float(m.get("duration_s") or 10), frames_dir, f"s{int(time.time())}{i}", 2 if len(items) < 4 else 1)[: KEEP_FRAMES - len(frames)]
        if not sources:
            raise stylefetch.FetchError("none of those videos could be measured")
        return {"sources": sources, "frames": frames}
    finally:
        try:
            stylefetch.cleanup(items, tmp)
        except Exception:  # noqa: BLE001
            pass
