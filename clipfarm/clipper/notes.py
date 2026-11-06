"""What a finished long video leaves in its Notes folder for the person who finishes it (2026-11-06):

    Chapters.txt          YouTube chapters ("00:00 Title"): the first at 00:00, at least 3, each at least 10 s (YouTube's own rules)
    Description.txt       a 2-3 sentence summary, the chapters, and the credits
    Quality check.md      every problem the render found or measured, each with how bad it is (Fix / Look / Info), WHERE in the finished
                          video (a timecode range), what it is in plain words, and what to do about it

The render's own findings (a shot under a second it could not cover, a stretch no camera filmed, a badly recorded opening line taken out)
are collected here as they happen (note()), next to what is measured on the finished file (qa.check) and on the sources (sources()).

    begin() / note(level, what, start, end, do) / take()      the render's findings
    chapters(ai, words, dur, glossary) -> (chapters, summary)  one cheap AI call; a fallback from the pauses when it fails
    youtube_ok(chapters, dur)                                   YouTube's rules
    sources(files) -> [issue]                                   variable frame rate, mixed frame rates or sizes
    quality_md(issues, stats) / counts(issues)                  the report and its "2 to look at, 1 to fix"
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.notes")

LEVELS = {"bad": "Fix", "fix": "Fix", "warn": "Look", "look": "Look", "info": "Info"}
_found: dict[int, list[dict]] = {}
_lock = threading.Lock()


# ------------------------------------------------------------------------------------------------ the render's own findings
def begin() -> None:
    with _lock:
        _found[threading.get_ident()] = []


def note(level: str, what: str, start: float | None = None, end: float | None = None, do: str = "") -> None:
    """A finding of the render (never raises: a note must never break a render)."""
    try:
        with _lock:
            _found.setdefault(threading.get_ident(), []).append(
                {"level": level, "what": what, "start": start, "end": end, "do": do})
    except Exception:  # noqa: BLE001
        pass


def take() -> list[dict]:
    with _lock:
        return _found.pop(threading.get_ident(), [])


# ------------------------------------------------------------------------------------------------ chapters
def clock(t: float) -> str:
    t = int(max(0.0, t))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60:02d}:{t % 60:02d}"


def youtube_ok(chs: list[tuple[float, str]], dur: float) -> bool:
    """YouTube shows chapters only when the first is at 0:00, there are at least 3, and each lasts at least 10 seconds."""
    if len(chs) < 3 or chs[0][0] != 0:
        return False
    marks = [t for t, _ in chs] + [dur]
    return all(b - a >= 10 for a, b in zip(marks, marks[1:])) and all(str(x).strip() for _, x in chs)


def tidy(chs: list[tuple[float, str]], dur: float) -> list[tuple[float, str]]:
    """Sorted, inside the video, the first at 0:00, none closer than 10 s to the next (a too-short one joins the one before it)."""
    out: list[tuple[float, str]] = []
    for t, title in sorted((max(0.0, float(t)), " ".join(str(title).split())[:60]) for t, title in chs):
        if not title or t >= dur - 10:
            continue
        if out and t - out[-1][0] < 10:
            continue
        out.append((t, title))
    if out:
        out[0] = (0.0, out[0][1])
    while len(out) > 1 and dur - out[-1][0] < 10:
        out.pop()
    return out


def _sentences(words: list[dict]) -> list[tuple[float, str]]:
    out, cur, t0 = [], [], 0.0
    for w in words:
        if not cur:
            t0 = float(w["start"])
        cur.append(str(w.get("text") or "").strip())
        if re.search(r"[.!?]$", cur[-1]):
            out.append((t0, " ".join(x for x in cur if x)))
            cur = []
    if cur:
        out.append((t0, " ".join(x for x in cur if x)))
    return out


def fallback_chapters(words: list[dict], dur: float) -> list[tuple[float, str]]:
    """Without the AI: about one chapter a minute (3 to 8), each starting at a sentence, named after its first words."""
    sents = _sentences(words)
    if dur < 30 or len(sents) < 3:
        return []
    n = max(3, min(8, int(dur // 60) or 3))
    out = []
    for k in range(n):
        want = dur * k / n
        t, text = min(sents, key=lambda s: abs(s[0] - want))
        title = " ".join(re.sub(r"[^\w' -]", "", text).split()[:6])
        out.append((0.0 if k == 0 else t, title[:1].upper() + title[1:] if title else f"Part {k + 1}"))
    return tidy(out, dur)


SYSTEM = ("You write YouTube chapters and a short description for a finished video, from its transcript (each line starts with its time "
          "in the video). Chapters mark where the topic really changes: short, concrete titles (2-6 words, no numbering, no emojis, no "
          "clickbait), the first one at 0 seconds, at least 3, at least 10 seconds apart. The summary is 2-3 plain sentences saying what "
          "the video is about and who is in it, in the speaker's spirit. Spell names exactly as in the glossary when one is given.")
SCHEMA = {"type": "object", "properties": {
    "summary": {"type": "string"},
    "chapters": {"type": "array", "items": {"type": "object", "properties": {"at": {"type": "number"}, "title": {"type": "string"}},
                                            "required": ["at", "title"]}}},
    "required": ["summary", "chapters"]}


def chapters(ai, words: list[dict], dur: float, glossary: list[str] | None = None, brief: str = "") -> tuple[list[tuple[float, str]], str]:
    """(chapters, summary) for the finished video's words (times in the finished video). One cheap AI call; without the AI (or when its
    chapters break YouTube's rules) the chapters come from the transcript itself and the summary is left empty."""
    sents = _sentences(words)
    summary = ""
    if ai is not None and sents:
        listing = "\n".join(f"[{t:.0f}] {s}" for t, s in sents)[:24000]
        prompt = ((f"What the video is: {brief}\n" if brief else "") + (f"Glossary (spell these exactly): {', '.join(glossary)}\n" if glossary else "")
                  + f"The video is {dur:.0f} seconds long.\nTranscript:\n{listing}")
        try:
            res = ai.complete_json("video_chapters", SYSTEM, prompt, SCHEMA)
            summary = " ".join(str(res.data.get("summary") or "").split())
            chs = tidy([(c.get("at", 0), c.get("title", "")) for c in res.data.get("chapters") or [] if isinstance(c, dict)], dur)
            if youtube_ok(chs, dur):
                return chs, summary
        except Exception as e:  # noqa: BLE001 - chapters are a bonus: the transcript gives some anyway
            log.warning(f"chapters: the AI didn't answer ({e}); made from the transcript instead")
    chs = fallback_chapters(words, dur)
    return (chs if youtube_ok(chs, dur) else []), summary


def chapters_txt(chs: list[tuple[float, str]]) -> str:
    return "\n".join(f"{clock(t)} {title}" for t, title in chs) + ("\n" if chs else "")


# ------------------------------------------------------------------------------------------------ the sources
def _rate(s: str) -> float:
    try:
        a, _, b = str(s).partition("/")
        return float(a) / float(b or 1)
    except (ValueError, ZeroDivisionError):
        return 0.0


def sources(files: list[dict]) -> list[dict]:
    """Phone footage with a variable frame rate (it drifts out of sync in an editor), and a mix of frame rates or picture sizes."""
    issues, rates, sizes = [], {}, {}
    for f in files:
        if f.get("photo") or f.get("library") or not f.get("path"):
            continue
        try:
            v = ffmpeg.video_stream(ffmpeg.probe(f["path"])) or {}
        except Exception:  # noqa: BLE001 - an unreadable file is the render's business, not this report's
            continue
        r, avg = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
        name = Path(f["path"]).name
        if r and avg and abs(r - avg) / r > 0.01:
            issues.append({"level": "look", "what": f"{name} has a variable frame rate ({avg:.2f} fps on average, up to {r:.2f}): phone footage often does",
                           "do": "If its sound drifts out of sync in Final Cut, convert it to a constant frame rate first (Compressor or HandBrake)."})
        if avg and not f.get("broll"):
            rates.setdefault(round(avg, 2), []).append(name)
        if v.get("width") and v.get("height") and not f.get("broll"):
            sizes.setdefault(f"{v['width']}x{v['height']}", []).append(name)
    if len(rates) > 1:
        issues.append({"level": "info", "what": "The talking footage mixes frame rates: " + "; ".join(f"{k:g} fps ({', '.join(v[:3])}{'…' if len(v) > 3 else ''})" for k, v in rates.items()),
                       "do": "Nothing needed in the MP4. In Final Cut, set the project to the rate most of the footage uses."})
    if len(sizes) > 1:
        issues.append({"level": "info", "what": "The talking footage mixes picture sizes: " + "; ".join(f"{k} ({', '.join(v[:3])}{'…' if len(v) > 3 else ''})" for k, v in sizes.items()),
                       "do": "Nothing needed: smaller footage is scaled to fill the frame. Check it still looks sharp."})
    return issues


# ------------------------------------------------------------------------------------------------ the report
def counts(issues: list[dict]) -> dict:
    lv = [LEVELS.get(i.get("level"), "Info") for i in issues]
    return {"fix": lv.count("Fix"), "look": lv.count("Look"), "info": lv.count("Info")}


def headline(c: dict) -> str:
    bits = ([f"{c['fix']} to fix"] if c["fix"] else []) + ([f"{c['look']} to look at"] if c["look"] else [])
    return ", ".join(bits) if bits else "Nothing to fix"


def _where(i: dict) -> str:
    a, b = i.get("start"), i.get("end")
    if a is None:
        return "whole video"
    return clock(a) if b is None or abs(float(b) - float(a)) < 1 else f"{clock(a)}–{clock(b)}"


def quality_md(issues: list[dict], stats: dict, title: str = "") -> str:
    order = {"Fix": 0, "Look": 1, "Info": 2}
    rows = sorted(issues, key=lambda i: (order[LEVELS.get(i.get("level"), "Info")], float(i.get("start") or 0)))
    c = counts(issues)
    lines = [f"# Quality check{': ' + title if title else ''}", "", f"**{headline(c)}.**" + (f" ({c['info']} note{'s' if c['info'] != 1 else ''})" if c["info"] else ""), ""]
    for i in rows:
        lines.append(f"- **{LEVELS.get(i.get('level'), 'Info')}** · {_where(i)} · {i['what'][:1].upper() + i['what'][1:]}"
                     + (f"\n  What to do: {i['do']}" if i.get("do") else ""))
    if stats:
        lines += ["", "Length {duration} s · loudness {lufs} LUFS · true peak {peak} dBTP · {cuts} picture changes · longest unchanged picture {longest_hold} s."
                  .format(**{k: stats.get(k, "?") for k in ("duration", "lufs", "peak", "cuts", "longest_hold")})]
    return "\n".join(lines) + "\n"
