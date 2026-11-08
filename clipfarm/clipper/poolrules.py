"""Which footage may be cut to as B-roll, and which scenes of it (WP3 of docs/plans/editor-improvement-plan.md).

Before: a file was B-roll when its folder or name said so, or when it was "sparse" (few words per second). A short clip of the speaker saying
"okay" was sparse, so the speaker was offered as B-roll; a named B-roll file the vision model failed on silently disappeared.

Now the footage record (footage.py) decides, and the words test is only the fallback for files without a record:
  - a file joins the pool when its role is broll / voiceover_broll / photo, or it was named B-roll (folder, name, the person's list);
  - a scene joins it only when it is not a talking head and lies inside a usable stretch of its file (a file keeps all its scenes when none of
    them is usable, so one shaky B-roll file is not silently emptied);
  - a named B-roll file the vision model could not describe still counts: a placeholder scene over its usable stretches, quality 4 (never a
    story shot);
  - everything kept out is listed (`excluded`) so the report can say "12 shots of the speaker were kept out of the B-roll";
  - orientation: a portrait shot on a landscape timeline (or the reverse) is marked and loses two quality points where shots are chosen by quality.

Pure functions: no AI, no files, nothing to fake in tests.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..core.words import STOP, tokens

POOL_ROLES = ("broll", "voiceover_broll", "photo")
A_ROLL_ROLES = ("a_roll", "multicam_angle")
PLACEHOLDER_QUALITY = 4.0
MISMATCH_PENALTY = 2.0                      # (proposal) quality points a wrong-orientation shot loses where shots are picked by quality
MIN_PLACEHOLDER_S = 0.6


def role_of(f: dict) -> str:
    """The role the footage record gave a scan entry ('' when it has no record)."""
    return str(((f.get("facts") or {}).get("role")) or "")


def describe_wanted(f: dict, words: list[dict], sparse) -> bool:
    """Is this file worth showing to the vision model as possible B-roll? (library footage is described by the catalogue)"""
    if f.get("library"):
        return False
    if f.get("broll"):
        return True
    role = role_of(f)
    if role in POOL_ROLES:
        return True
    if role in A_ROLL_ROLES:
        return False                                        # (the speaker's own footage is never B-roll, however few words a stretch has)
    if role:
        return False                                        # audio, music, logo, screen recording ...: not footage to cut to
    return bool(sparse(f, words))                           # (no record: the words-per-second test of before)


def _usable_windows(f: dict) -> list[tuple[float, float]]:
    out = []
    for u in ((f.get("facts") or {}).get("usable") or []):
        try:
            a, b = float(u["start"]), float(u["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b - a >= MIN_PLACEHOLDER_S:
            out.append((a, b))
    return out


def placeholder_scenes(f: dict) -> list[dict]:
    """Scenes for a named B-roll file nobody described: its usable stretches (the whole file when the record has none), quality 4."""
    start = float(f.get("start") or 0.0)
    dur = float(f.get("dur") or (float(f.get("end") or 0.0) - start) or 0.0)
    wins = _usable_windows(f) or ([(0.0, dur)] if dur >= MIN_PLACEHOLDER_S else [])
    what = f"(not described) {f.get('name')}"
    return [{"start": round(start + a, 3), "end": round(start + min(b, dur), 3), "what": what, "quality": PLACEHOLDER_QUALITY, "file": f.get("name"),
             "fstart": round(a, 3), "fend": round(min(b, dur), 3), "described": False, "w": f.get("w"), "h": f.get("h")} for a, b in wins if min(b, dur) - a >= MIN_PLACEHOLDER_S]


TALKING_FACE = 0.30                         # a face this tall (share of the picture) in a file where words were said is someone talking to the camera
SOME_SPEECH = 0.03


def talking_head(f: dict, s: dict) -> bool:
    """Is this scene a person talking to the camera? The vision model's own word (`is_talking_head`), or the measurements: a big face in a file
    that has speech in it (a short clip of the speaker saying "okay" has too few words to be called A-roll, but it is still the speaker)."""
    if s.get("is_talking_head") is True:
        return True
    try:
        face = float(s.get("face_ratio") or 0.0)
        said = float((((f.get("facts") or {}).get("speech")) or {}).get("ratio") or 0.0)
    except (TypeError, ValueError):
        return False
    return face >= TALKING_FACE and said >= SOME_SPEECH


def _scene_filter(f: dict, scenes: list[dict], allow_talking: bool) -> tuple[list[dict], list[dict]]:
    """(kept, [{sid/file, why}]) for one file's scenes."""
    kept, out = [], []
    usable_any = any(s.get("usable") is True for s in scenes)
    for s in scenes:
        why = ""
        if not allow_talking and talking_head(f, s):
            why = "the speaker talking to the camera"
        elif usable_any and s.get("usable") is False:
            why = "blurry, shaky, black or frozen there"
        if why:
            out.append({"file": f.get("name"), "sid": s.get("sid"), "start": s.get("start"), "why": why})
        else:
            kept.append(s)
    return kept, out


def select(files: list[dict], scenes: dict[str, list[dict]], words: list[dict], sparse, keep_cut: bool = False) -> dict:
    """-> {"named": {index}, "silent": {index}, "scenes": {file name: scenes}, "excluded": [{file, why, ...}]}.
    `named` are files the person (or their folder) called B-roll; `silent` are other files that really are pictures, not talking.
    `keep_cut` ("add B-roll to a video"): every other video is B-roll by the person's own words, talking or not, so a talking scene stays."""
    named: set[int] = set()
    silent: set[int] = set()
    out_scenes = {k: list(v) for k, v in (scenes or {}).items()}
    excluded: list[dict] = []
    for k, f in enumerate(files):
        name = f.get("name")
        have = out_scenes.get(name) or []
        if f.get("broll"):
            if not have and not f.get("library") and not f.get("photo") and not f.get("stock"):
                have = placeholder_scenes(f)                   # (named B-roll the model could not describe: still B-roll, never a story shot)
            kept, ex = _scene_filter(f, have, allow_talking=keep_cut)
            excluded += ex
            if kept:
                out_scenes[name] = kept
                named.add(k)
            else:
                out_scenes.pop(name, None)
            continue
        role = role_of(f)
        if role in A_ROLL_ROLES:
            if have or sparse(f, words):
                excluded.append({"file": name, "why": "the speaker's own footage is never used as B-roll"})
            out_scenes.pop(name, None)
            continue
        if role in POOL_ROLES or (not role and have and sparse(f, words)):
            kept, ex = _scene_filter(f, have, allow_talking=False)
            excluded += ex
            if kept:
                out_scenes[name] = kept
                silent.add(k)
            else:
                out_scenes.pop(name, None)
        elif role:
            out_scenes.pop(name, None)                         # audio, music, a logo, a screen recording ...
    return {"named": named, "silent": silent, "scenes": out_scenes, "excluded": excluded}


def excluded_line(excluded: list[dict]) -> str:
    """One plain sentence for the report ('' when nothing was kept out)."""
    if not excluded:
        return ""
    talk = sum(1 for e in excluded if "speaker" in str(e.get("why")))
    other = len(excluded) - talk
    bits = []
    if talk:
        bits.append(f"{talk} shot(s) of the speaker were kept out of the B-roll")
    if other:
        bits.append(f"{other} other shot(s) were left out (not usable)")
    return "; ".join(bits)


# --------------------------------------------------------------------------------------------------------------- rules from the notes / the MCP
def rules_from(opts: dict | None) -> dict:
    """The exclusion rules of a video, from its options: {"avoid": [words], "exclude_files": [names], "exclude_windows": [[s, e]], "min_shot_s": float}."""
    o = opts or {}

    def strs(v):
        return [str(x).strip() for x in v if str(x).strip()][:80] if isinstance(v, (list, tuple)) else []
    wins = []
    for w in (o.get("exclude_windows") or []) if isinstance(o.get("exclude_windows"), (list, tuple)) else []:
        se = window_se(w)
        if not se:
            continue
        a, b = se
        if b > a >= 0:
            wins.append([round(a, 2), round(b, 2)])
    try:
        mn = max(0.0, float(o.get("min_shot_s") or 0))
    except (TypeError, ValueError):
        mn = 0.0
    return {"avoid": strs(o.get("avoid")), "exclude_files": strs(o.get("exclude_files")), "exclude_windows": wins[:40], "min_shot_s": mn}


def file_excluded(scene: dict, names: list[str]) -> bool:
    """Is this scene one of the files / scene ids the person ruled out ('C0003', 'C0003.MP4', 's0041', case ignored)?"""
    if not names:
        return False
    keys = {str(scene.get("sid") or "").lower(), str(scene.get("file") or "").lower(), Path(str(scene.get("file") or "")).stem.lower()}
    keys.discard("")
    return any(n.lower() in keys or Path(n).stem.lower() in keys for n in names)


def window_se(w) -> tuple[float, float] | None:
    """(start, end) seconds of one window given as a dict ({"start_s","end_s"} from the grounded notes, or {"a","b"} / {"start","end"}) or as a
    pair [s, e]; None when it is neither or empty. One reader for every place that takes `exclude_windows`."""
    try:
        if isinstance(w, dict):
            a = next(w[k] for k in ("start_s", "a", "start", "s") if w.get(k) is not None)
            b = next(w[k] for k in ("end_s", "b", "end", "e") if w.get(k) is not None)
        else:
            a, b = w[0], w[1]
        a, b = float(a), float(b)
    except (TypeError, ValueError, IndexError, KeyError, StopIteration):
        return None
    if a != a or b != b or b <= a:
        return None
    return (a, b)


def window_pairs(wins, to_out=None) -> list[tuple[float, float]]:
    """Every usable window as (start, end). A dict window from the grounded notes is in the recording's seconds: `to_out(start_s)` maps it to
    the finished video's seconds when given (a window that maps to nothing is dropped). A plain pair is taken as finished-video seconds."""
    out: list[tuple[float, float]] = []
    for w in (wins or []) if isinstance(wins, (list, tuple)) else []:
        se = window_se(w)
        if not se:
            continue
        if isinstance(w, dict) and to_out is not None:
            try:
                a, b = to_out(se[0]), to_out(se[1])
            except Exception:  # noqa: BLE001 - a window that maps to nothing blocks nothing
                continue
            if a is None or b is None or not b > a:
                continue
            se = (float(a), float(b))
        out.append(se)
    return out


def outside_windows(cutaways: list[dict], windows: list[list[float]] | None) -> tuple[list[dict], int]:
    """(cutaways with the ones overlapping a no-B-roll window of the finished video taken out, how many were taken out)."""
    wins = window_pairs(windows)
    if not wins:
        return list(cutaways), 0
    kept = [c for c in cutaways if not any(float(c["at"]) < b and float(c["at"]) + float(c["length"]) > a for a, b in wins)]
    return kept, len(cutaways) - len(kept)


# ------------------------------------------------------------------------------------------------------------------------- orientation
def shape(s: dict) -> str:
    """'portrait' / 'landscape' / '' of a pool scene, from its own size."""
    try:
        w, h = float(s.get("w") or 0), float(s.get("h") or 0)
    except (TypeError, ValueError):
        return ""
    if w <= 0 or h <= 0 or abs(w - h) / max(w, h) < 0.08:
        return ""
    return "portrait" if h > w else "landscape"


def mismatch(s: dict, landscape: bool | None) -> bool:
    """A scene whose shape is the opposite of the timeline's (a square one fits both)."""
    if landscape is None:
        return False
    sh = shape(s)
    if not sh:
        ok = str(s.get("orientation_ok") or "both")
        return (ok == "portrait" and landscape) or (ok == "landscape" and not landscape)
    return (sh == "portrait") == bool(landscape)


def effective_quality(s: dict) -> float:
    """The quality shots are ranked by where they are chosen without the AI: a wrong-orientation shot loses MISMATCH_PENALTY points."""
    try:
        q = float(s.get("quality") if s.get("quality") is not None else 6.0)
    except (TypeError, ValueError):
        q = 6.0
    return q - (MISMATCH_PENALTY if s.get("mismatch") else 0.0) - (2.0 if s.get("described") is False else 0.0)


def mark_orientation(pool: list[dict], landscape: bool | None) -> list[dict]:
    """Pool scenes with `mismatch` set (copies)."""
    return [({**s, "mismatch": True} if mismatch(s, landscape) else dict(s)) for s in pool]


# --------------------------------------------------------------------------------------------------------------------------- loose music
MUSIC_NAME = re.compile(r"music|song|soundtrack|bgm|\bscore\b|instrumental|background", re.I)


def music_like(parts) -> bool:
    """Is an audio file (given as folder names then the file name) a song rather than a recorder's track? A folder or file name that says music.
    (The word "track" alone is not enough: a recorder's channels are called Track 1, Track 2.)"""
    parts = [str(x) for x in parts]
    if not parts:
        return False
    return any(MUSIC_NAME.search(x) for x in parts[:-1]) or bool(MUSIC_NAME.search(Path(parts[-1]).stem))


# --------------------------------------------------------------------------------------------------------------------------- caches
def cache_key(f: dict) -> str:
    """The identity of a scan entry's file for the per-file caches (words.json, broll.json): the footage record's key (path, size, modified time)."""
    from .footage import cache_key as _k
    try:
        return _k(Path(str(f.get("path") or f.get("name") or "")))
    except Exception:  # noqa: BLE001
        return ""


def cache_lookup(cache: dict, f: dict) -> dict | None:
    """The entry of a per-file cache that belongs to this file, or None. Entries stay under the file's display name (other readers use them so),
    and each now carries the file's identity key: an entry whose name matches but whose key says another file (two cameras that both wrote
    C0001.MP4 once shared one) is not used; a file that was renamed finds its entry by its key and the entry moves to the new name; an older entry
    with no key is accepted by name and size and gets its key stamped (the one-time migration)."""
    name, key = f.get("name"), cache_key(f)
    e = cache.get(name)
    if isinstance(e, dict) and e.get("size") == f.get("size"):
        if not e.get("key"):
            e["key"] = key                                        # (migration: stamped, the entry itself is unchanged)
        if e["key"] == key:
            return e
    if key:
        for other, x in list(cache.items()):
            if other != name and isinstance(x, dict) and x.get("key") == key and x.get("size") == f.get("size"):
                cache[name] = dict(x)                             # (renamed: its entry follows it)
                return cache[name]
    return None


# -------------------------------------------------------------------------------------------------------------------------- photos
def photo_fits(scene: dict, spoken_tokens: frozenset) -> bool:
    """Is this photo forced into the cut? Only when it was a good photo (its own quality 6 or more) or it is about something that is said."""
    try:
        raw = float(scene.get("raw_quality") if scene.get("raw_quality") is not None else scene.get("quality", 6.0))
    except (TypeError, ValueError):
        raw = 6.0
    if raw >= 6.0:
        return True
    words = {w for w in (scene.get("subject_words") or [])} | set(tokens(str(scene.get("what") or "")))
    return bool(words & spoken_tokens)


def spoken_tokens(text: str) -> frozenset:
    return frozenset(t for t in tokens(text) if t not in STOP)


__all__ = ["music_like", "cache_key", "cache_lookup", "select", "describe_wanted", "placeholder_scenes", "rules_from", "file_excluded", "outside_windows", "mark_orientation", "effective_quality",
           "photo_fits", "spoken_tokens", "excluded_line", "mismatch", "shape"]
