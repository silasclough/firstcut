"""Exact edit lists: the person (or the AI working for them through the MCP) says which clips play, in which order, with which trims.

When the list is known, FirstCut does not plan the cut: it plays exactly that. This module is the plain part (no ffmpeg, no AI), so it
can be tested exactly:

    parse()       the entries as given (a list of dicts or of strings) -> clean entries
    find_file()   which file in the folder an entry names (by its number or name, never a guess among several)
    from_steps()  a numbered outline in the person's notes whose numbers ARE file names ("1.01 Welcome" and a file "1.01 Welcome.mov"):
                  the edit list is chosen automatically, nobody asks for it
    items_for()   the plan's items for the entries (each a stretch of one file, in the order given), plus what was wrong with the list
    clean_pool()  the B-roll that may be used: nothing shorter than the minimum, nothing the person ruled out

The runner (localproj.make_edit_list) owns scanning, transcribing, describing B-roll and the render.
"""

from __future__ import annotations

import re
from pathlib import Path

LABEL = re.compile(r"^\s*(?:step\s*)?(\d{1,2}\.\d{1,3})\b")
AUDIO = ("own", "silent")


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if x >= 0 else None


def parse(raw) -> list[dict]:
    """[{file, start, end, audio, label, note}] from what was given: dicts, or plain strings (the file's name or number). Bad entries raise
    ValueError with the entry's place in the list, so the caller can fix it."""
    if not isinstance(raw, list):
        raise ValueError("edit_list must be a list of clips")
    out = []
    for k, e in enumerate(raw, 1):
        if isinstance(e, str):
            e = {"file": e}
        if not isinstance(e, dict):
            raise ValueError(f"edit_list entry {k} must be a clip name or an object with a file")
        ref = str(e.get("file") or e.get("clip") or e.get("name") or "").strip()
        if not ref:
            raise ValueError(f"edit_list entry {k} has no file")
        a, b = _num(e.get("start", e.get("in"))), _num(e.get("end", e.get("out")))
        if a is not None and b is not None and b <= a:
            raise ValueError(f"edit_list entry {k} ({ref}): end must be after start")
        audio = str(e.get("audio") or "own").lower()
        if audio not in AUDIO:
            raise ValueError(f"edit_list entry {k} ({ref}): audio must be 'own' (the clip's own sound) or 'silent'")
        out.append({"file": ref, "start": a, "end": b, "audio": audio, "label": str(e.get("label") or "")[:80], "note": str(e.get("note") or "")[:200]})
    return out


MEDIA = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".mxf", ".webm", ".insv", ".lrv", ".mp3", ".wav", ".m4a", ".jpg", ".jpeg", ".png", ".heic"}


def _key(s: str) -> str:
    """A name for comparing: lower case, spaces made one, and the extension removed only when it IS a media extension ("1.02" is a number, not a
    file called "1" with the extension ".02")."""
    name = Path(str(s)).name
    if Path(name).suffix.lower() in MEDIA:
        name = name[: -len(Path(name).suffix)]
    return re.sub(r"\s+", " ", name.lower()).strip()


def find_file(ref: str, files: list[dict]) -> tuple[dict | None, list[str]]:
    """(the file, []) when `ref` names exactly one file; (None, [candidates]) when it names none or several. Tried in this order: the whole name,
    the name without its extension, a name that STARTS with it as a whole token ("1.01" is "1.01 Welcome.mov" but never "1.011.mov"), a name
    that contains it. The first step that finds anything decides: two files there is a question, not a guess."""
    r = str(ref).strip()
    rl, rk = r.lower(), _key(r)
    names = [(f, f.get("rel") or f["name"]) for f in files]
    steps = (
        lambda f, n: n.lower() == rl or Path(n).name.lower() == rl,
        lambda f, n: _key(n) == rk,
        lambda f, n: re.match(r"^" + re.escape(rk) + r"(?![0-9])", _key(n)) is not None,
        lambda f, n: rk in _key(n),
    )
    for test in steps:
        hit = [f for f, n in names if test(f, n)]
        if len(hit) == 1:
            return hit[0], []
        if hit:
            return None, [h.get("rel") or h["name"] for h in hit][:8]
    return None, []


def from_steps(steps: list[str], files: list[dict]) -> list[dict] | None:
    """The edit list an outline already is: every beat starts with a number ("1.01 ...") and most of those numbers name a file in the folder.
    None when the outline is about what is said (no numbers, or not file names): the ordinary outline editing then looks beats up in the talk.
    Beats whose number is no file are left out; the person is told by the runner."""
    if len(steps) < 3:
        return None
    entries, hit = [], 0
    for s in steps:
        m = LABEL.match(s)
        if not m:
            continue
        f, _ = find_file(m.group(1), files)
        if f is not None:
            hit += 1
            entries.append({"file": f.get("rel") or f["name"], "start": None, "end": None, "audio": "own", "label": s[:80], "note": ""})
    return entries if hit >= 3 and hit >= 0.6 * len(steps) else None


def items_for(entries: list[dict], files: list[dict], min_shot_s: float = 0.0) -> tuple[list[dict], list[str], dict[str, str]]:
    """(items, problems, how each used file sounds). Each entry becomes one item: a stretch of its file, in the order given. problems are things
    to tell the person (a clip not found, an entry shorter than the minimum, a trim past the end of the file); a clip not found stops the list."""
    items, problems = [], []
    sound: dict[str, set[str]] = {}
    missing = []
    for k, e in enumerate(entries, 1):
        f, cands = find_file(e["file"], files)
        if f is None:
            missing.append(f"entry {k} '{e['file']}': " + (f"could be any of {', '.join(cands)}" if cands else "no such file in the folder"))
            continue
        dur = float(f["dur"])
        a = min(max(0.0, e["start"] if e["start"] is not None else 0.0), dur)
        b = min(dur, e["end"] if e["end"] is not None else dur)
        if e["end"] is not None and e["end"] > dur + 0.01:
            problems.append(f"entry {k} ({f['name']}): ends at {e['end']:g} s but the clip is only {dur:.1f} s: cut at its end")
        if b - a < 0.2:
            problems.append(f"entry {k} ({f['name']}): nothing left to play between {a:.1f} and {b:.1f} s: left out")
            continue
        if min_shot_s and b - a < min_shot_s:
            problems.append(f"entry {k} ({f['name']}): only {b - a:.1f} s, under the minimum of {min_shot_s:g} s (kept: you listed it)")
        items.append({"kind": "visual", "start": round(float(f["start"]) + a, 3), "end": round(float(f["start"]) + b, 3), "file": f["name"],
                      **({"label": e["label"]} if e.get("label") else {})})
        sound.setdefault(f["name"], set()).add(e["audio"])
    if missing:
        raise ValueError("the edit list names clips that can't be found: " + "; ".join(missing))
    how = {}
    for name, kinds in sound.items():
        how[name] = "silent" if kinds == {"silent"} else "own"
        if len(kinds) > 1:
            problems.append(f"{name} is listed both with its own sound and silent: it plays with its own sound")
    return items, problems, how


def clean_pool(pool: list[dict], avoid: list[str] | None, min_shot_s: float = 0.0, exclude_files: list[str] | None = None) -> list[dict]:
    """The B-roll scenes that may be used: not shorter than `min_shot_s`, nothing whose description or file name has one of the words the
    person ruled out (case ignored), and nothing from a file or scene id they ruled out (`exclude_files`, e.g. "C0003" or "s0041"). Photos have
    no length and stay."""
    from . import poolrules as PR
    words = [str(a).lower().strip() for a in (avoid or []) if str(a).strip()]
    out = []
    for s in pool:
        if exclude_files and PR.file_excluded(s, [str(x) for x in exclude_files]):
            continue
        what = f"{s.get('what', '')} {s.get('file', '')}".lower()
        if words and any(w in what for w in words):
            continue
        if min_shot_s and not s.get("photo") and float(s.get("end", 0)) - float(s.get("start", 0)) < min_shot_s:
            continue
        out.append(s)
    return out
