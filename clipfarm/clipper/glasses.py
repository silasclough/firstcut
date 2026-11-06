"""Meta glasses (Ray-Ban Meta / Oakley Meta) footage: detection + moment finding from what's SEEN.

POV glasses footage has no face to follow and often little talking, so the transcript alone can't
find the good parts. Here:

detect()           reads every metadata field ffprobe exposes (make/model, QuickTime/Android keys,
                   encoder, comment...) for Meta / Ray-Ban / Oakley Meta. Separately reports whether
                   the shape fits (portrait 3:4, <= ~3 min) so the GUI can offer "redo as glasses"
                   when an upload path stripped the metadata. All tags are saved on the job so the
                   rules can be tightened once real files are seen.
visual_signals()   per-sample motion (shake), sharpness (blur), brightness (dark) + per-second loudness.
contact_sheets()   frames every ~2 s tiled 4x3 with burned-in timestamps - what the vision model sees.
vision_moments()   vision LLM (OpenAI by day; an Ollama vision model if one is configured) looks at the
                   sheets + transcript + signal notes and picks POV-worthy moments with hook titles.
heuristic_moments() no-AI fallback: loudness + steady + sharp + speech, sliding windows.
"""

from __future__ import annotations

import io
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger
from . import moments as M

log = get_logger("clipper.glasses")

META_RE = re.compile(r"ray[\s_-]?ban|oakley\s*meta|meta\s*(?:ai|view|smart\s*glasses|glasses|wearables)|"
                     r"\bray-ban\s*stories\b|^\s*meta\s*$|meta\s*platforms", re.I)
TAG_HINT = re.compile(r"make|model|manufacturer|software|encoder|comment|description|title|handler|author|"
                      r"artist|copyright|creation_tool|com\.", re.I)


# ----------------------------------------------------------------------------- detection
def probe_tags(info: dict) -> dict[str, str]:
    tags = {f"format.{k.lower()}": str(v) for k, v in (info.get("format", {}).get("tags") or {}).items()}
    for st in info.get("streams", []):
        for k, v in (st.get("tags") or {}).items():
            tags[f"{st.get('codec_type', 'stream')}.{k.lower()}"] = str(v)
    return tags


def _display_dims(vs: dict) -> tuple[int, int]:
    w, h = int(vs.get("width", 0)), int(vs.get("height", 0))
    rot = 0
    try:
        rot = int(float((vs.get("tags") or {}).get("rotate", 0)))
    except ValueError:
        pass
    for sd in vs.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    return (h, w) if abs(rot) % 180 == 90 else (w, h)


def detect(src: Path) -> dict:
    info = ffmpeg.probe(src)
    tags = probe_tags(info)
    hit = next(((k, v) for k, v in tags.items() if TAG_HINT.search(k) and META_RE.search(v)), None)
    vs = ffmpeg.video_stream(info) or {}
    w, h = _display_dims(vs)
    dur = float(info.get("format", {}).get("duration") or 0)
    shape = bool(w and h and h > w and abs(w / h - 0.75) < 0.02 and dur <= 190)
    keep = {k: v[:120] for k, v in tags.items() if TAG_HINT.search(k) and "creation_time" not in k}
    return {"glasses": bool(hit), "evidence": f"{hit[0].split('.', 1)[-1]}={hit[1][:60]}" if hit else None,
            "shape_match": shape, "dims": f"{w}x{h}", "duration": round(dur, 1), "tags": keep}


# ----------------------------------------------------------------------------- signals
def _lap_var(g: np.ndarray) -> float:
    lap = 4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())


def visual_signals(src: Path, rate: float = 2.0, w: int = 96) -> dict:
    vs = ffmpeg.video_stream(ffmpeg.probe(src)) or {}
    sw, sh = _display_dims(vs)
    h = max(2, int(round(w * (sh or 4) / (sw or 3))) // 2 * 2)
    raw = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", str(src), "-vf",
                          f"fps={rate},scale={w}:{h},format=gray", "-f", "rawvideo", "-"],
                         capture_output=True, timeout=1800).stdout
    n = len(raw) // (w * h)
    frames = [np.frombuffer(raw[k * w * h:(k + 1) * w * h], dtype=np.uint8).reshape(h, w).astype(float) for k in range(n)]
    motion = [0.0] + [float(np.abs(a - b).mean() / 255) for a, b in zip(frames, frames[1:])]
    sharp = [_lap_var(f) for f in frames]
    med = float(np.median(sharp)) if sharp else 1.0
    return {"rate": rate, "motion": motion, "sharp": [s / (med or 1.0) for s in sharp],
            "bright": [float(f.mean() / 255) for f in frames]}


def _ranges(flags: list[bool], rate: float, min_s: float = 1.0) -> list[tuple[float, float]]:
    out, run = [], None
    for k, f in enumerate(flags + [False]):
        if f and run is None:
            run = k
        elif not f and run is not None:
            if (k - run) / rate >= min_s:
                out.append((run / rate, k / rate))
            run = None
    return out


def signal_notes(sig: dict, loud: np.ndarray | None, start: float, end: float) -> str:
    r = sig["rate"]
    a, b = int(start * r), int(end * r)
    mo, sh, br = sig["motion"][a:b], sig["sharp"][a:b], sig["bright"][a:b]
    if not mo:
        return ""
    shaky_th = max(0.12, float(np.percentile(sig["motion"], 90)))
    fmt = lambda rs: ", ".join(f"{start + x:.0f}-{start + y:.0f}s" for x, y in rs[:8]) or "none"   # noqa: E731
    notes = [f"very shaky / fast head turns: {fmt(_ranges([m > shaky_th for m in mo], r))}",
             f"blurry: {fmt(_ranges([s < 0.35 for s in sh], r))}",
             f"dark: {fmt(_ranges([x < 0.12 for x in br], r))}"]
    if loud is not None and len(loud):
        seg = loud[int(start):int(end)]
        peaks = [int(start) + i for i, v in enumerate(seg) if v > 2.0]
        merged = []
        for p in peaks:
            if merged and p - merged[-1][1] <= 2:
                merged[-1][1] = p
            else:
                merged.append([p, p])
        notes.append("loud moments (laughs, shouts, impacts, music): " +
                     (", ".join(f"{x}-{y + 1}s" for x, y in merged[:10]) or "none"))
    return "\n".join(notes)


# ----------------------------------------------------------------------------- contact sheets
def contact_sheets(src: Path, start: float, end: float, step: float, cols: int = 4, rows: int = 3,
                   tile_w: int = 256) -> list[tuple[bytes, list[float]]]:
    from PIL import Image, ImageDraw, ImageFont

    from . import logcolor                                          # (log footage seen in normal colour by the AI)

    vs = ffmpeg.video_stream(ffmpeg.probe(src)) or {}
    sw, sh = _display_dims(vs)
    tile_h = int(round(tile_w * (sh or 4) / (sw or 3))) // 2 * 2
    raw = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
                          "-i", str(src), "-vf", f"{logcolor.vf_prefix(src)}fps=1/{step:.4f},scale={tile_w}:{tile_h}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=1800).stdout
    size = tile_w * tile_h * 3
    imgs = [Image.frombytes("RGB", (tile_w, tile_h), raw[k * size:(k + 1) * size]) for k in range(len(raw) // size)]
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    per = cols * rows
    out = []
    for s0 in range(0, len(imgs), per):
        batch = imgs[s0:s0 + per]
        sheet = Image.new("RGB", (cols * tile_w + (cols - 1) * 4, rows * tile_h + (rows - 1) * 4), (20, 20, 20))
        times = []
        for i, im in enumerate(batch):
            t = start + (s0 + i) * step
            times.append(round(t, 1))
            x, y = (i % cols) * (tile_w + 4), (i // cols) * (tile_h + 4)
            sheet.paste(im, (x, y))
            label = f"{int(t // 60)}:{t % 60:04.1f}"
            d = ImageDraw.Draw(sheet)
            d.rectangle((x, y, x + 8 + d.textlength(label, font=font), y + 30), fill=(0, 0, 0))
            d.text((x + 4, y + 3), label, font=font, fill=(255, 235, 60))
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=82)
        out.append((buf.getvalue(), times))
    return out


# ----------------------------------------------------------------------------- vision moments
SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "clips": {"type": "array", "items": {"type": "object", "properties": {
            "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "title": {"type": "string"}, "hook": {"type": "string"}, "caption": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
            "score": {"type": "number"}, "reason": {"type": "string"},
            "mood": {"type": "string", "enum": ["chill", "upbeat", "funny", "suspense", "emotional", "epic", "adventure"]}},
            "required": ["start_s", "end_s", "title", "hook", "caption", "hashtags", "score", "reason", "mood"]}}},
    "required": ["summary", "clips"],
}

SYSTEM = """You edit first-person POV footage filmed on Meta smart glasses (Ray-Ban Meta / Oakley Meta) into \
short vertical clips for TikTok, Reels and Shorts. You see contact sheets: frames sampled in time order, each \
stamped with its timestamp (m:ss.s) in the top-left corner, plus any speech transcript and per-second signals.
POV clips that work: something surprising, funny, beautiful, skillful or risky happens in view; a reveal \
(turning a corner, opening something, arriving somewhere); an interaction with people or animals; a reaction \
that's audible; a "you had to be there" moment that the viewer experiences through the wearer's eyes; a clear \
payoff. Start right before the action so the first second already hooks.
Avoid: frantic head turns and whip pans, staring at the ground or sky, dark/blurry stretches, dead time where \
nothing changes, anything that only makes sense with missing context.
For each clip return start_s/end_s (seconds, from the timestamps), a hook title in POV style (max 55 chars, \
e.g. "POV: the waiter brings THIS out", no emojis, never a lie about what's shown), hook (why viewers keep \
watching), a short caption (max 20 words, no hashtags), 3-5 hashtags without '#', score 1-10 for standalone \
viral potential (use the whole range and rank - no two clips get the same score; 9-10 only for exceptional) \
and a reason (max 10 words), plus the background-music mood that fits (chill, upbeat, funny, suspense, \
emotional, epic or adventure). Titles and captions must be clean (no profanity or slurs). Describe only \
what is visible or audible."""


def fit_time(m: M.Moment, min_len: float, max_len: float, duration: float, words: list[dict]) -> M.Moment | None:
    s, e = max(0.0, m.start), min(duration, m.end)
    if e - s < min_len:
        pad = (min_len - (e - s)) / 2
        s, e = max(0.0, s - pad), min(duration, e + pad)
        if e - s < min_len:                     # against an edge: push the other way
            s, e = (max(0.0, e - min_len), e) if e >= duration - 0.01 else (s, min(duration, s + min_len))
    if e - s > max_len:
        e = s + max_len
    for w in words:                              # never cut a word in half at the edges
        if w["start"] < s < w["end"]:
            s = w["start"]
        if w["start"] < e < w["end"]:
            e = min(duration, w["end"] + 0.05)
    if e - s < min(min_len, duration) * 0.6:
        return None
    m.start, m.end = round(s, 2), round(e, 2)
    return m


def vision_moments(router, src: Path, duration: float, words: list[dict], sig: dict, loud: np.ndarray | None,
                   n: int, min_len: float, max_len: float, title: str = "", window_s: float = 180.0,
                   progress=None, files: list[dict] | None = None) -> list[M.Moment]:
    from .combine import boundary_note
    sents = M.sentences_from_words(words) if words else []
    wins = []
    a = 0.0
    while a < duration - 1:
        wins.append((a, min(duration, a + window_s)))
        a += window_s - 15
    out: list[M.Moment] = []
    per = max(1, min(6, math.ceil(n * 1.5 / len(wins)) + (1 if duration > 40 else 0)))
    for wi, (ws, we) in enumerate(wins):
        step = max(1.5, (we - ws) / 72)
        sheets = contact_sheets(src, ws, we, step)
        speech = "\n".join(f"({int(s.start // 60)}:{s.start % 60:04.1f}) {s.text}" for s in sents
                           if ws <= s.start < we) or "(no speech)"
        prompt = (f"Video: {title}\nThis part: {ws:.0f}-{we:.0f}s of {duration:.0f}s. "
                  f"{len(sheets)} contact sheet(s), one frame every {step:.1f}s.\n\n"
                  f"{boundary_note(files or [], ws, we)}\n"
                  f"Speech transcript:\n{speech}\n\nSignals:\n{signal_notes(sig, loud, ws, we)}\n\n"
                  f"Propose up to {per} clips, each {int(min_len)}-{int(max_len)} seconds, inside {ws:.0f}-{we:.0f}s. "
                  f"Fewer is fine if nothing else is genuinely good.")

        def in_range(c, ws=ws, we=we):
            try:
                return ws - 2 <= float(c["start_s"]) < float(c["end_s"]) <= we + 2
            except (KeyError, TypeError, ValueError):
                return False

        def check(d):
            # drop the odd out-of-range proposal later; only reject an answer with nothing usable in it
            cs = d.get("clips", [])
            if cs and not any(in_range(c) for c in cs):
                raise ValueError(f"every proposed clip is outside {ws:.0f}-{we:.0f}s")

        try:
            res = router.complete_json("glasses_moments", SYSTEM, prompt, SCHEMA, validate=check,
                                       images=[b for b, _ in sheets])
        except Exception as e:  # noqa: BLE001 - heuristic covers this window
            log.warning(f"vision moments failed for {ws:.0f}-{we:.0f}s: {e}")
            if progress:
                progress(wi + 1, len(wins))
            continue
        for c in res.data.get("clips", []):
            if not in_range(c):
                log.info(f"skipping out-of-range proposal {c.get('start_s')}-{c.get('end_s')}s")
                continue
            m = M.Moment(float(c["start_s"]), float(c["end_s"]), -1, -1, title=str(c["title"])[:70].strip(),
                         hook=str(c.get("hook", "")), caption=str(c.get("caption", "")).strip(),
                         hashtags=[h.lstrip("#").strip() for h in c.get("hashtags", []) if str(h).strip()][:6],
                         score=float(c.get("score", 5)), reason=str(c.get("reason", "")), source=f"{res.provider} vision",
                         mood=str(c.get("mood", "")))
            m = fit_time(m, min_len, max_len, duration, words)
            if m:
                out.append(m)
        if progress:
            progress(wi + 1, len(wins))
    return out


def heuristic_moments(sig: dict, loud: np.ndarray | None, words: list[dict], duration: float,
                      min_len: float, max_len: float) -> list[M.Moment]:
    r = sig["rate"]
    secs = int(duration)
    if secs <= 0:
        return []

    def per_sec(arr):
        a = np.array(arr, dtype=float)
        return np.array([a[int(i * r):int((i + 1) * r)].mean() if len(a[int(i * r):int((i + 1) * r)]) else 0
                         for i in range(secs)])

    motion, sharp, bright = per_sec(sig["motion"]), per_sec(sig["sharp"]), per_sec(sig["bright"])
    shaky = motion > max(0.12, np.percentile(motion, 90))
    talk = np.zeros(secs)
    for w in words:
        talk[min(secs - 1, int(w["start"]))] += 1
    lz = np.clip(loud[:secs], -1, 4) if loud is not None and len(loud) >= secs else np.zeros(secs)
    score = 0.35 * lz + 0.25 * np.clip(sharp, 0, 1.5) + 0.2 * np.clip(talk / 3, 0, 1) + \
        0.2 * np.clip(motion / 0.06, 0, 1) - 0.8 * shaky - 0.6 * (bright < 0.12)
    target = int(min(max_len, max(min_len, 15)))
    out = []
    for s in range(0, max(1, secs - target + 1)):
        win = score[s:s + target]
        m = M.Moment(float(s), float(min(duration, s + target)), -1, -1, score=round(4 + 2 * float(win.mean()), 2),
                     source="heuristic", reason="loud / steady / sharp / speech")      # no fake hook title
        out.append(m)
    return M.dedupe(out)


def as_json(d: dict) -> str:
    return json.dumps(d, default=str)
