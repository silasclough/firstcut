"""What the Final Cut project carries beyond the picture and the sound (WP13): roles, markers, notes, keywords, ratings, the
"source clips" event, plain-text for editable titles, and a pre-flight check of the finished XML.

Pure text/number code (no ffmpeg, no files except `preflight(files_on_disk=True)`): fcpxml.py calls these while it writes the project.

    roles      every clip says what it is (dialogue, music, B-roll...) so Final Cut's Timeline Index can switch them on and off
    markers    chapters (blue), to-dos (red, with the explanation in the note) and plain markers, inside the clip that holds them
    keywords   B-roll clips and the source files carry what is in them
    preflight  every time on the frame grid, ids unique, every ref resolves, files exist, conform-rate where the rate differs
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote, urlparse
from xml.sax.saxutils import escape as _escape

AUDIO_ROLES = {"dialogue": "dialogue", "speaker1": "dialogue.speaker1", "speaker2": "dialogue.speaker2", "audience": "dialogue.audience",
               "recorder": "dialogue.recorder", "camera": "dialogue.camera", "music": "music", "effects": "effects"}
ROLE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 _-]*(\.[A-Za-z0-9 _-]+)?$")         # "main.sub": Apple's roles attribute


def esc(s) -> str:
    return _escape(str(s), {'"': "&quot;"})


def audio_role(kind: str = "dialogue", who: str | None = None) -> str:
    """The audio role of a clip: dialogue.speaker1 / speaker2 / audience (WP6's `who`), dialogue.recorder, dialogue.camera, music, effects."""
    if kind == "dialogue" and who:
        w = str(who).lower()
        if "audience" in w or "question" in w:
            return AUDIO_ROLES["audience"]
        if w in ("2", "speaker2", "guest", "b"):
            return AUDIO_ROLES["speaker2"]
        if w in ("1", "speaker1", "host", "a"):
            return AUDIO_ROLES["speaker1"]
    return AUDIO_ROLES.get(kind, "dialogue")


def role_ok(role: str) -> bool:
    return bool(role) and bool(ROLE_RE.match(role))


def num(v, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ------------------------------------------------------------------------------------------------------------------ markers
def normalise_markers(markers) -> list[dict]:
    """[{"at" (finished-video seconds), "kind" chapter|todo|standard, "text", "note"}] in time order; anything else is dropped."""
    out: list[dict] = []
    for m in markers or []:
        if not isinstance(m, dict):
            continue
        at = num(m.get("at"), -1.0)
        text = " ".join(str(m.get("text") or "").split())[:120]
        if at < 0 or not text:
            continue
        kind = m.get("kind") if m.get("kind") in ("chapter", "todo", "standard") else "standard"
        out.append({"at": at, "kind": kind, "text": text, "note": " ".join(str(m.get("note") or "").split())[:400]})
    return sorted(out, key=lambda m: (m["at"], m["kind"]))


def marker_lines(clk, markers: list[dict], seg_from_f: int, seg_to_f: int, local_f, indent: str, used: set | None = None) -> list[str]:
    """The markers whose time falls inside [seg_from_f, seg_to_f) (frames of the finished video) as chapter-marker / marker lines.
    `local_f(frame)` gives the frame in the CLIP'S OWN time (start + position inside it). Each marker is written once (`used` remembers)."""
    used = used if used is not None else set()
    out: list[str] = []
    one = clk.at(1)
    for k, m in enumerate(markers):
        f = clk.frames(m["at"])
        if k in used or not (seg_from_f <= f < seg_to_f):
            continue
        used.add(k)
        start = clk.at(local_f(f))
        if m["kind"] == "chapter":
            out.append(f'{indent}<chapter-marker start="{start}" duration="{one}" value="{esc(m["text"])}" posterOffset="{clk.at(max(1, int(round(float(clk.frames(1.0)) * 0.5))))}"'
                       + (f' note="{esc(m["note"])}"' if m["note"] else "") + "/>")
        elif m["kind"] == "todo":
            out.append(f'{indent}<marker start="{start}" duration="{one}" value="{esc(m["text"])}" completed="0"'
                       + (f' note="{esc(m["note"])}"' if m["note"] else "") + "/>")
        else:
            out.append(f'{indent}<marker start="{start}" duration="{one}" value="{esc(m["text"])}"'
                       + (f' note="{esc(m["note"])}"' if m["note"] else "") + "/>")
    return out


def markers_from_plan(plan: dict | None, issues: list[dict] | None = None, chapters: list[dict] | None = None, dropped: list[dict] | None = None,
                      cutaways: list[dict] | None = None) -> list[dict]:
    """Every marker the planner knows, as {"at", "kind", "text", "note"} on the finished video's clock:
    outline steps / chapter cards -> chapter; quality-check issues with a time, unmet constraints, B-roll abstentions, dropped shots -> todo;
    B-roll choices -> standard (the noun and its score)."""
    out: list[dict] = []
    for c in chapters or []:
        at = num((c or {}).get("at"), -1.0)
        if at >= 0 and c.get("title"):
            out.append({"at": at, "kind": "chapter", "text": str(c["title"]), "note": str(c.get("note") or "")})
    for i in issues or []:
        if not isinstance(i, dict) or i.get("level") == "info" or i.get("start") is None:
            continue
        s = num(i.get("start"), -1.0)
        if s >= 0:
            out.append({"at": s, "kind": "todo", "text": ("Check: " + str(i.get("what") or ""))[:110], "note": str(i.get("do") or "")})
    for d in dropped or []:
        s = num((d or {}).get("at"), -1.0)
        if s >= 0:
            out.append({"at": s, "kind": "todo", "text": "A B-roll shot was dropped here", "note": str(d.get("why") or "its clip ended")})
    for c in cutaways or []:
        if isinstance(c, dict) and c.get("noun"):
            out.append({"at": num(c.get("at")), "kind": "standard", "text": f'B-roll: {c["noun"]}',
                        "note": ("score %.2f" % num(c.get("score"))) if c.get("score") is not None else ""})
    for u in ((plan or {}).get("unmet") or []):
        if isinstance(u, dict) and u.get("at") is not None:
            said = str(u.get("text") or u.get("what") or "").strip()
            if said:                                                            # (an unmet item with no words is not worth a marker)
                out.append({"at": num(u["at"]), "kind": "todo", "text": ("Not done: " + said)[:110], "note": str(u.get("why") or "")})
    seen: set = set()
    unique = []
    for m in out:                                                               # (the same marker at the same moment once)
        key = (round(float(m.get("at") or 0), 2), m.get("kind"), m.get("text"))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return normalise_markers(unique)


# ------------------------------------------------------------------------------------------------------------------ notes, keywords, ratings
def keyword_line(value: str, indent: str, start: str | None = None, duration: str | None = None) -> str:
    return (f'{indent}<keyword' + (f' start="{start}"' if start else "") + (f' duration="{duration}"' if duration else "")
            + f' value="{esc(value)}"/>')


def broll_keywords(c: dict) -> str:
    """The keywords of a B-roll clip: B-roll plus what it shows (its noun and subject words), without repeats."""
    words = ["B-roll"]
    for w in [c.get("noun")] + list(c.get("subjects") or []) + [c.get("what_short")]:
        w = " ".join(str(w or "").split())[:40]
        if w and w.lower() not in {x.lower() for x in words}:
            words.append(w)
    return ", ".join(words[:6])


def broll_note(c: dict) -> str:
    bits = []
    if c.get("what"):
        bits.append(str(c["what"])[:160])
    if c.get("noun"):
        bits.append(f'Chosen for: "{c["noun"]}"' + (f' (match {num(c.get("score")):.2f})' if c.get("score") is not None else ""))
    if c.get("why"):
        bits.append(str(c["why"])[:120])
    return " - ".join(bits)


def card_text(kind: str, spec: dict | None) -> str | None:
    """The words of a graphic card as one string for an editable Basic Title, or None when the card is not made of plain words
    (charts, tiles, ledgers, timelines stay clips/stills)."""
    s = spec if isinstance(spec, dict) else {}

    def j(*keys, sep="\n"):
        parts = [" ".join(str(s.get(k) or "").split()) for k in keys]
        return sep.join(p for p in parts if p) or None
    if kind in ("quote", "fullquote"):
        t = j("text")
        who = " ".join(str(s.get("who") or "").split())
        return (t + (f"\n- {who}" if who else "")) if t else None
    if kind == "statement":
        return j("text")
    if kind == "chapter":
        t = j("title", "text")
        n = str(s.get("number") or "").strip()
        return ((n + ". ") if n and t else "") + t if t else None
    if kind == "reference":
        return j("ref", "passage")
    if kind in ("stat", "bigstat"):
        return j("value", "label")
    if kind == "lower":
        return j("name", "role") if (s.get("name") or s.get("role")) else None
    if kind == "chips":
        c = s.get("chips") or s.get("bullets") or []
        return " | ".join(str(x) for x in c[:6]) if c else None
    return None


# ------------------------------------------------------------------------------------------------------------------ pre-flight
TIME_ATTRS = ("offset", "start", "duration", "tcStart", "audioStart", "audioDuration")
TIME_RE = re.compile(r"^(-?\d+)(?:/(\d+))?s$")


def _time(v: str | None) -> Fraction | None:
    m = TIME_RE.match(v or "")
    if not m:
        return None
    return Fraction(int(m.group(1)), int(m.group(2) or 1))


def preflight(xml_text: str, files_on_disk: bool = True) -> list[str]:
    """Problems a Final Cut import would warn about or refuse, found without Final Cut: [] = nothing found.
    Every time value is a whole number of the sequence's frames; every id is unique; every ref names a resource of the right kind; every
    media file exists; a clip whose frame rate differs from the sequence has a conform-rate; no caption sits directly inside an asset-clip;
    audioStart and audioDuration come together."""
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        return [f"not well-formed XML: {e}"]
    problems: list[str] = []
    res = root.find("resources")
    ids: dict[str, ET.Element] = {}
    seen: set[str] = set()
    for el in root.iter():                                        # (resources, text-style-defs: one namespace for ids)
        i = el.get("id")
        if i is None:
            continue
        if i in seen:
            problems.append(f"id {i} is used twice")
        seen.add(i)
    for el in (list(res) if res is not None else []):
        if el.get("id") is not None:
            ids[el.get("id")] = el
    kinds = {"asset-clip": {"asset"}, "mc-clip": {"media"}, "ref-clip": {"media"}, "title": {"effect"}, "filter-video": {"effect"}, "filter-audio": {"effect"}}
    for el in root.iter():
        want = kinds.get(el.tag)
        r = el.get("ref")
        if want and r is not None:
            tgt = ids.get(r)
            if tgt is None:
                problems.append(f"{el.tag} '{el.get('name') or ''}' refers to {r}, which does not exist")
            elif tgt.tag not in want:
                problems.append(f"{el.tag} '{el.get('name') or ''}' refers to {r}, a {tgt.tag} (needs {'/'.join(sorted(want))})")
    seq = root.find(".//sequence")
    fmt = ids.get(seq.get("format")) if seq is not None else None
    fd = _time(fmt.get("frameDuration")) if fmt is not None else None
    parent = {c: p for p in root.iter() for c in p}

    def grid(el, attr):
        """The frame duration an attribute must be a multiple of: start/tcStart on the media's own grid, offset/duration on the sequence's
        (in the source-clips event and in resources, on the clip's / asset's own grid)."""
        own = None
        f = ids.get(el.get("format") or "")
        if f is None and el.get("ref") in ids:
            f = ids.get(ids[el.get("ref")].get("format") or "")
        if f is None and el.tag == "asset":
            f = ids.get(el.get("format") or "")
        if f is not None:
            own = _time(f.get("frameDuration"))
        in_event = parent.get(el) is not None and parent[el].tag == "event"
        if el.tag == "asset" or in_event or attr in ("start", "tcStart", "audioStart", "audioDuration"):
            return own or fd
        return fd
    skip = ("transition", "text-style-def", "param", "keyframe", "timept", "marker", "chapter-marker", "caption", "keyword", "rating", "sync-source",
            "audio-role-source", "media-rep", "format", "effect", "sequence", "project", "event", "library", "multicam", "mc-angle", "media")
    for el in root.iter():
        if el.tag in skip:
            continue
        for a_ in ("offset", "start", "duration"):
            t = _time(el.get(a_))
            g = grid(el, a_)
            if t is not None and g and (t / g).denominator != 1:
                problems.append(f"{el.tag} '{el.get('name') or ''}': {a_}={el.get(a_)} is not a whole number of frames")
                break
    for el in root.iter():
        if el.tag == "asset-clip" and any(c.tag == "caption" for c in el.findall("caption")) and False:
            problems.append("a caption directly inside an asset-clip")     # (Final Cut itself anchors captions in asset-clips: kept as documentation)
        if (el.get("audioStart") is None) != (el.get("audioDuration") is None):
            problems.append(f"{el.tag} '{el.get('name') or ''}': audioStart and audioDuration must come together")
    if files_on_disk:
        for el in root.iter("media-rep"):
            src = el.get("src") or ""
            p = Path(unquote(urlparse(src).path))
            if src.startswith("file://") and not p.exists():
                problems.append(f"media file missing: {p.name}")
    if seq is not None and fmt is not None:                       # conform-rate wherever a clip's own rate differs from the sequence's
        seq_fd = fd
        in_seq = set(seq.iter())
        for el in root.iter("asset-clip"):
            if el not in in_seq:                                  # (the source-clips event's browser clips are not on the timeline)
                continue
            f = ids.get(el.get("format") or "")
            a = ids.get(el.get("ref") or "")
            if f is None and a is not None:
                f = ids.get(a.get("format") or "")
            cf = _time(f.get("frameDuration")) if f is not None else None
            if cf and seq_fd and cf != seq_fd and el.find("conform-rate") is None and el.get("srcEnable") != "audio":
                if abs(float(cf) - float(seq_fd)) / float(seq_fd) > 0.02 and a is not None and a.get("hasVideo") == "1" and float(1 / cf) <= 60.5:   # (conform-rate has no value above 60: nothing can be written for 100/120/240 fps)
                    problems.append(f"asset-clip '{el.get('name') or ''}': frame rate {float(1 / cf):.2f} differs from the sequence's and there is no conform-rate")
    return problems
