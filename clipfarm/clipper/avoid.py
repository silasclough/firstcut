"""Shots a person said "don't use this" about, for one video (the live view and the Replace sheet).

An entry names the shot the way the project's own files do: {"name": "Library - Athletics/C5211.MP4" (the file as the edit calls it),
"start", "end": seconds in that file, "photo": True for a still, "what", "id": the footage catalog's id, "at": when}. The list lives in the
project's work folder (avoid.json) so a render, a re-render and the Replace sheet all read the same one:
  - the library search leaves these shots out (add_library),
  - a render never keeps one as a cutaway: it is swapped for the next best unused shot, or taken out (swap),
  - the Replace sheet doesn't offer them again.
Pure functions over dicts; the render passes its own pool and file list.
"""
from __future__ import annotations

import time

FILE = "avoid.json"
OVERLAP_S = 0.3                          # a video shot that overlaps an avoided one by more than this is the same shot


def load(st) -> list[dict]:
    got = st.get(FILE) or {}
    return [x for x in (got.get("shots") if isinstance(got, dict) else got) or [] if isinstance(x, dict) and x.get("name")]


def _same(a: dict, name: str, start: float, end: float, photo: bool) -> bool:
    if a.get("name") != name:
        return False
    if photo or a.get("photo"):
        return True
    return min(end, float(a.get("end") or 0)) - max(start, float(a.get("start") or 0)) > OVERLAP_S


def record(st, shot: dict) -> list[dict]:
    """Remember a shot (once); returns the whole list."""
    name = str(shot.get("name") or "").strip()
    if not name:
        raise ValueError("which shot?")
    cur = load(st)
    photo = bool(shot.get("photo"))
    start, end = float(shot.get("start") or 0), float(shot.get("end") or 0)
    if not any(_same(a, name, start, end, photo) for a in cur):
        cur.append({"name": name[:300], "start": round(start, 2), "end": round(end, 2), "photo": photo, "what": str(shot.get("what") or "")[:200],
                    "id": str(shot.get("id") or "")[:64], "at": int(time.time())})
        st.put(FILE, {"shots": cur})
    return cur


def forget(st, shot: dict) -> list[dict]:
    """Take a shot off the list again (the person changed their mind)."""
    name, photo = str(shot.get("name") or ""), bool(shot.get("photo"))
    start, end = float(shot.get("start") or 0), float(shot.get("end") or 0)
    cur = [a for a in load(st) if not _same(a, name, start, end, photo)]
    st.put(FILE, {"shots": cur})
    return cur


def is_avoided(avoid: list[dict], name: str, start: float, end: float, photo: bool = False) -> bool:
    return any(_same(a, name, start, end, photo) for a in avoid)


def drop_found(avoid: list[dict], shots: list[dict]) -> list[dict]:
    """The footage library's found shots without the avoided ones (add_library)."""
    if not avoid:
        return shots
    return [s for s in shots if not is_avoided(avoid, f"Library - {s.get('rel')}", float(s.get("start") or 0), float(s.get("end") or 0),
                                               s.get("kind") == "photo")]


def bad_pool(avoid: list[dict], pool: list[dict], files: list[dict]) -> set[int]:
    """The numbers of the pool shots that are avoided. A pool shot is on the timeline of its file (file["start"] + seconds in the file)."""
    if not avoid:
        return set()
    by = {f.get("name"): f for f in files}
    out = set()
    for i, sc in enumerate(pool):
        name = sc.get("file") or ""
        f = by.get(name)
        off = float(f["start"]) if f and not sc.get("photo") else 0.0
        a, b = float(sc.get("start") or 0) - off, float(sc.get("end") or 0) - off
        if is_avoided(avoid, name, a, b, bool(sc.get("photo"))):
            out.add(i)
    return out


def swap(cutaways: list[dict], pool: list[dict], bad: set[int], exclude: set[int] | None = None) -> tuple[list[dict], dict]:
    """The placed shots with every avoided one swapped for the best unused pool shot that fits (video for video, photo for photo, long
    enough), or taken out when nothing fits (the speaker is shown instead). -> (cutaways, {"swapped": n, "removed": n})."""
    if not bad or not any(c.get("scene") in bad for c in cutaways):
        return cutaways, {"swapped": 0, "removed": 0}
    used = {c.get("scene") for c in cutaways} | set(exclude or ())
    out, swapped, removed = [], 0, 0
    for c in cutaways:
        if c.get("scene") not in bad:
            out.append(c)
            continue
        want_photo = bool(pool[c["scene"]].get("photo")) if isinstance(c.get("scene"), int) and 0 <= c["scene"] < len(pool) else False
        need = float(c.get("length") or 0)
        best = None
        for i, sc in enumerate(pool):
            if i in used or i in bad or bool(sc.get("photo")) != want_photo:
                continue
            room = float(sc["end"]) - float(sc["start"])
            if not want_photo and room < min(need, 2.0):
                continue
            q = float(sc.get("quality") or 0) + (2.0 if want_photo or room >= need else 0.0)
            if best is None or q > best[0]:
                best = (q, i, sc, room)
        if best is None:
            removed += 1
            continue
        _, i, sc, room = best
        used.add(i)
        swapped += 1
        out.append({**c, "src": round(float(sc["start"]), 2), "scene": i, "what": sc.get("what", ""), "why": "the other shot was ruled out",
                    "file": sc.get("file"), "photo": sc.get("photo"), "length": round(need if want_photo else min(need, room), 2)})
    return out, {"swapped": swapped, "removed": removed}
