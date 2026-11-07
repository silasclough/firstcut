"""Montage and slideshow videos: footage with no one speaking (a camp, an event, a folder of photos) cut to music.

Not a fourth video type: the person picks a folder and says what they want, and when there is nobody talking (or the notes say
"montage" / "slideshow", or most of the folder is photos) the Long form flow plans one of these instead of an interview edit.

What lives here is the planning, with no ffmpeg and no AI in the arithmetic, so it can be tested exactly:

    detect()      is this a montage, a slideshow, or a normal edit?
    candidates()  every usable shot: a described scene of a clip, or a photo
    select()      the keepers: poor ones out, never the same file or the same look more than twice
    arc_order()   chronological, opening on an arrival and closing on the group
    layout()      how long each shot is held: the pace setting, cut on the music's beats or bars, the length setting
    card()        the plan in plain words and thumbnails, for the person to check before anything renders

The runner (localproj.py) owns the AI calls, the music choice, the photo animation and the render; plan["montage"] is its state.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.montage")

WORDS_PER_10_MIN = 25            # fewer spoken words than this per 10 minutes of footage: nobody is talking
MAX_AUTO_S = 480.0               # "AI decides" never makes a montage longer than this (the best shots are kept, and the card says so)
MIN_QUALITY = 4.0
MIN_SHOT_S = 0.8
MAX_PER_FILE = 2                 # the same file is used at most this many times (the same seated man, eight times, was the bug report)
ALIKE_CLIP, ALIKE_PHOTO = 0.6, 0.8     # how alike two descriptions are (word overlap) before they count as the same look
MAX_ALIKE = 2

PHOTO_BASE = 2_000_000.0         # photos live far past the footage on the shared timeline (photos.VIRTUAL_BASE is for B-roll photos)
PHOTO_STEP = 100.0
EFFECTS = ["zoom_in", "pan_right", "zoom_out", "pan_left"]       # a gentle move: no polaroid frames in a slideshow

# What "fast", "regular" and "slow" mean: seconds a shot is held, and what the cut lands on (a beat of the music or a whole bar).
PACE = {"fast": {"clip": (1.0, 2.5), "photo": (1.0, 2.2), "unit": "beat", "label": "Fast-paced: quick cuts on the beat"},
        "auto": {"clip": (2.5, 4.0), "photo": (2.5, 4.0), "unit": "bar", "label": "Regular pace: cuts on the bars of the music"},
        "slow": {"clip": (5.0, 8.0), "photo": (5.0, 7.0), "unit": "bar", "label": "Taking it slow: long holds, cuts on the bars"}}

# Words in the notes. STRONG ones decide on their own; WEAK ones ("recap") only count when almost nobody is talking.
STRONG = re.compile(r"\b(montage|slide ?show|highlight reel|sizzle( reel)?|music video|photo (video|show)|no talking|just (the )?(b-?roll|clips|photos))\b", re.I)
WEAK = re.compile(r"\b(recap|highlights?|best of|memories|aftermovie|after-movie)\b", re.I)
SLIDE = re.compile(r"\b(slide ?show|photo (video|show)|photos? only|just (the )?photos)\b", re.I)
KEEP_SOUND = re.compile(r"\b(keep (the )?(natural |original )?(sound|audio)|with (the )?(natural |original )?(sound|audio)|natural sound|ambient sound)\b", re.I)
SOFT = re.compile(r"\b(dissolves?|cross ?fades?|soft transitions?)\b", re.I)

OPENING = re.compile(r"\b(entrance|sign|banner|welcome|arriv\w*|building|exterior|sunrise|gate|lake|field|wide|landscape|check-?in|bus|van)\b", re.I)
CLOSING = re.compile(r"\b(group|everyone|together|all of|sunset|goodbye|hug\w*|cheer\w*|circle|finale|wave|waving|campfire|team photo)\b", re.I)
STOP = set("with that this from they them then than have been were their there into over under about while which where what when"
           "some more most very just only also each other both same like near next onto upon within without".split())


# ---------------------------------------------------------------------------------------------------------------- detection
def pace_key(pace) -> str:
    """The engine's pace word (natural / fast / slow) or the page's (auto / fast / slow) as one of PACE's keys."""
    p = str(pace or "").lower()
    return "fast" if p == "fast" else "slow" if p == "slow" else "auto"


def detect(notes: str, *, n_words: int | None = None, footage_s: float = 0.0, n_photos: int = 0, n_videos: int = 0, audio_clips: int | None = None,
           setting: str = "auto") -> dict | None:
    """None for a normal edit, else {"style": "montage" | "slideshow", "why": ..., "line": ...} (line: one plain sentence for the page).
    n_words is None while nothing has been transcribed yet (then only what can be known without it decides)."""
    setting = str(setting or "auto").lower()
    notes = str(notes or "")
    if setting == "off":
        return None
    media = n_photos + n_videos
    slideshow = bool(SLIDE.search(notes)) or setting == "slideshow" or (media >= 4 and n_photos >= 6 and n_photos / max(1, media) > 0.6)
    if slideshow and (n_photos or setting == "slideshow"):
        why = "notes" if SLIDE.search(notes) else "photos"
        line = ("This looks like a slideshow: photos" + (" with a few clips" if n_videos else "") + ". Music comes first and the cuts follow it."
                if why == "photos" else "You asked for a slideshow: the photos" + (" and a few clips" if n_videos else "") + " are cut to music.")
        return {"style": "slideshow", "why": why, "line": line}
    low_speech = n_words is not None and footage_s >= 20 and n_words / max(0.1, footage_s / 600.0) < WORDS_PER_10_MIN
    silent = audio_clips == 0 and n_videos > 0                              # no clip has a sound track at all
    if STRONG.search(notes) or setting == "montage":
        return {"style": "montage", "why": "notes", "line": "You asked for a montage: the clips are cut to music, with no talking."}
    if silent or low_speech or (n_words is not None and WEAK.search(notes) and n_words / max(0.1, footage_s / 600.0) < 4 * WORDS_PER_10_MIN):
        return {"style": "montage", "why": "no_speech" if (silent or low_speech) else "notes",
                "line": "This looks like a montage: no one is speaking. Music comes first and the cuts follow it."}
    return None


def wants_sound(notes: str) -> bool:
    return bool(KEEP_SOUND.search(notes or ""))


def wants_dissolves(notes: str) -> bool:
    return bool(SOFT.search(notes or ""))


# ------------------------------------------------------------------------------------------------------------------ the shots
def tokens(text: str) -> frozenset:
    return frozenset(w for w in re.findall(r"[a-z]{4,}", str(text or "").lower()) if w not in STOP)


def alike(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


def candidates(files: list[dict], scenes: dict, photos: list[dict], when=None) -> list[dict]:
    """Every shot that could be used, as plain dicts (kept in plan.json): {id, kind clip|photo, name, path, start, end (virtual seconds of a clip
    scene), what, quality, t (when it was shot)}. `when(path, kind, f)` gives the shot time (camera clock / photo EXIF, else the file date)."""
    when = when or (lambda path, kind, f: float((f or {}).get("mtime") or 0))
    out: list[dict] = []
    for f in files:
        if f.get("photo") or f.get("library") or f.get("stock"):
            continue
        shot = when(f.get("path"), "clip", f)
        sc = [s for s in (scenes.get(f["name"]) or []) if float(s.get("end", 0)) - float(s.get("start", 0)) >= MIN_SHOT_S]
        if not sc:
            sc = [{"start": f["start"], "end": f["end"], "what": Path(f["name"]).stem.replace("_", " ").replace("-", " "), "quality": 5.5}]
        for s in sc:
            out.append({"id": f"c{len(out)}", "kind": "clip", "name": f["name"], "path": f.get("path"), "start": float(s["start"]), "end": float(s["end"]),
                        "what": str(s.get("what") or "")[:200], "quality": float(s.get("quality") or 6.0), "t": [shot, float(s["start"])]})
    for p in photos:
        out.append({"id": f"c{len(out)}", "kind": "photo", "name": p["name"], "path": p["path"], "start": None, "end": None,
                    "what": str(p.get("what") or "")[:200], "quality": float(p.get("quality") or 6.0), "t": [when(p["path"], "photo", p), 0.0]})
    return out


def select(cands: list[dict], *, max_per_file: int = MAX_PER_FILE, max_alike: int = MAX_ALIKE) -> tuple[list[dict], list[dict]]:
    """(keepers, left out). Left out: [{id, what, why}]. The best shots first, so when two look alike the better one is the one kept."""
    keep: list[dict] = []
    gone: list[dict] = []
    per_file: dict[str, int] = {}
    for c in sorted(cands, key=lambda c: (-c["quality"], c["t"])):
        if c["quality"] < MIN_QUALITY:
            gone.append({"id": c["id"], "what": c["what"], "why": "poor quality"})
            continue
        if c["kind"] == "clip" and per_file.get(c["name"], 0) >= max_per_file:
            gone.append({"id": c["id"], "what": c["what"], "why": "this clip is already used twice"})
            continue
        thr = ALIKE_PHOTO if c["kind"] == "photo" else ALIKE_CLIP
        if sum(1 for k in keep if k["kind"] == c["kind"] and alike(k["what"], c["what"]) >= thr) >= max_alike:
            gone.append({"id": c["id"], "what": c["what"], "why": "looks like shots already chosen"})
            continue
        keep.append(c)
        if c["kind"] == "clip":
            per_file[c["name"]] = per_file.get(c["name"], 0) + 1
    return keep, gone


def arc_order(keep: list[dict]) -> list[str]:
    """Ids in play order: as it happened, opening on an arrival or an establishing shot and closing on the group, and never two of the same look
    side by side when the next shot is different."""
    order = sorted(keep, key=lambda c: tuple(c["t"]))
    n = len(order)
    if n >= 6:
        head, tail = max(1, n // 5), max(1, n // 5)
        op = next((c for c in order[:head] if OPENING.search(c["what"])), None)
        if op is not None:
            order.remove(op)
            order.insert(0, op)
        cl = next((c for c in reversed(order[-tail:]) if CLOSING.search(c["what"])), None)
        if cl is not None and cl is not order[0]:
            order.remove(cl)
            order.append(cl)
    for i in range(1, len(order) - 1):                                        # break up runs of the same look: swap with a different one just ahead
        if alike(order[i - 1]["what"], order[i]["what"]) >= ALIKE_CLIP:
            for j in range(i + 1, min(len(order) - 1, i + 4)):
                if alike(order[i - 1]["what"], order[j]["what"]) < ALIKE_CLIP:
                    order[i], order[j] = order[j], order[i]
                    break
    return [c["id"] for c in order]



def vary(cutaways: list[dict], pool: list[dict], exclude: set[int] | None = None, max_per_file: int = MAX_PER_FILE, max_alike: int = MAX_ALIKE) -> tuple[list[dict], int]:
    """The B-roll placed over an interview, with no file and no look used more than twice: a shot that would be the third is swapped for the best
    unused shot that fits (video for video, photo for photo, long enough). Nothing is ever taken out (something must change on screen every 12 s),
    and a shot with no alternative stays. -> (cutaways, how many were swapped). The same seated man used eight times was the bug report."""
    used = {c.get("scene") for c in cutaways} | set(exclude or ())
    per_file: dict = {}
    seen: list[tuple[str, str]] = []                               # (kind, what) of the shots kept so far
    out, swapped = [], 0

    def bad(sc: dict) -> bool:
        kind = "photo" if sc.get("photo") else "clip"
        if kind == "clip" and per_file.get(sc.get("file"), 0) >= max_per_file:
            return True
        thr = ALIKE_PHOTO if kind == "photo" else ALIKE_CLIP
        return sum(1 for k, w in seen if k == kind and alike(w, sc.get("what", "")) >= thr) >= max_alike

    for c in sorted(cutaways, key=lambda c: c["at"]):
        i = c.get("scene")
        sc = pool[i] if isinstance(i, int) and 0 <= i < len(pool) else None
        if sc is not None and bad(sc):
            want_photo, need = bool(sc.get("photo")), float(c.get("length") or 0)
            best = None
            for j, alt in enumerate(pool):
                if j in used or bool(alt.get("photo")) != want_photo or bad(alt):
                    continue
                room = float(alt["end"]) - float(alt["start"])
                if not want_photo and room < min(need, 2.0):
                    continue
                q = float(alt.get("quality") or 0) + (2.0 if want_photo or room >= need else 0.0)
                if best is None or q > best[0]:
                    best = (q, j, alt, room)
            if best is not None:
                _, j, alt, room = best
                used.add(j)
                swapped += 1
                c = {**c, "src": round(float(alt["start"]), 2), "scene": j, "what": alt.get("what", ""), "file": alt.get("file"), "photo": alt.get("photo"),
                     "why": "so no look is used more than twice", "length": round(need if want_photo else min(need, room), 2)}
                sc = alt
        if sc is not None:
            if not sc.get("photo"):
                per_file[sc.get("file")] = per_file.get(sc.get("file"), 0) + 1
            seen.append(("photo" if sc.get("photo") else "clip", sc.get("what", "")))
        out.append(c)
    return out, swapped


# ------------------------------------------------------------------------------------------------------------ timing and length
def hold(span: tuple[float, float], period: float | None, unit: str) -> float:
    """How long to hold one shot: a whole number of bars (or beats) of the music inside the range, the one nearest its middle; plain seconds
    when the song has no steady beat."""
    lo, hi = span
    mid = (lo + hi) / 2.0
    if not period or period <= 0:
        return round(mid, 3)
    steps = [period * 4] if unit == "bar" and period * 4 <= hi else []
    steps.append(period)
    for step in steps:
        ks = [k for k in range(1, 400) if lo - 1e-9 <= k * step <= hi + 1e-9]
        if ks:
            return round(min(ks, key=lambda k: abs(k * step - mid)) * step, 3)
    return round(max(period, round(mid / period) * period), 3)


def _snap_down(d: float, period: float | None) -> float:
    """A shot shorter than its hold (a short scene): the nearest whole beat below it."""
    if not period or period <= 0:
        return round(d, 3)
    k = math.floor(d / period + 1e-9)
    return round(max(period, k * period) if d >= period else d, 3)


def layout(m: dict) -> dict:
    """Decide every shot's hold from the plan's settings and put the result in m: m['items'] (what the render plays), m['total_s'], m['dropped'],
    m['note']. Reads m['cands'] (id -> candidate), m['order'], m['removed'], m['pace'], m['period'], m['target_s']. Never speeds footage up."""
    cands = {c["id"]: c for c in m["cands"]}
    pace = PACE[pace_key(m.get("pace"))]
    period = m.get("period") or None
    removed = set(m.get("removed") or [])
    seq = [cands[i] for i in m["order"] if i in cands and i not in removed]
    rows: list[dict] = []
    for c in seq:
        want = hold(pace["photo" if c["kind"] == "photo" else "clip"], period, pace["unit"])
        if c["kind"] == "clip":
            room = float(c["end"]) - float(c["start"])
            d = want if room >= want + 0.2 else _snap_down(room - 0.2, period)
            if d < MIN_SHOT_S:
                continue
            lo = float(c["start"]) + max(0.0, (room - d)) * 0.4                    # inside the scene, away from the handheld start and end
            rows.append({"id": c["id"], "kind": "clip", "d": d, "start": round(lo, 3), "end": round(lo + d, 3), "max": min(room - 0.2, pace["clip"][1])})
        else:
            rows.append({"id": c["id"], "kind": "photo", "d": want, "max": pace["photo"][1]})
    total = sum(r["d"] for r in rows)
    notes: list[str] = []
    dropped: list[dict] = []
    target = float(m["target_s"]) if m.get("target_s") else None
    limit = target if target else MAX_AUTO_S
    while rows and total > limit + 1e-6 and len(rows) > 2:                       # too long: the weakest shot (never the first or the last) goes
        worst = min(range(1, len(rows) - 1) or [0], key=lambda i: cands[rows[i]["id"]]["quality"])
        gone = rows.pop(worst)
        total -= gone["d"]
        dropped.append({"id": gone["id"], "what": cands[gone["id"]]["what"], "why": "to fit the length" if target else "to keep it under 8 minutes"})
    if dropped:
        notes.append(f"{len(dropped)} shot{'s' if len(dropped) != 1 else ''} left out " + ("to fit the length you asked for." if target else "to keep the video under 8 minutes."))
    if target and rows and total < target * 0.9:                                   # too short: hold the shots a little longer, up to the pace's longest
        for _round in range(6):
            grew = False
            for r in rows:
                step = period if period else 0.5
                if r["d"] + step <= r["max"] + 1e-6 and total < target - 1e-6:
                    r["d"] = round(r["d"] + step, 3)
                    if r["kind"] == "clip":
                        r["end"] = round(r["start"] + r["d"], 3)
                    total += step
                    grew = True
            if not grew or total >= target * 0.98:
                break
        if total < target * 0.9:
            notes.append(f"There are not enough shots for {fmt_len(target)} at this pace: it comes to {fmt_len(total)}.")
    clock = 0.0
    items = []
    for r in rows:
        c = cands[r["id"]]
        if c["kind"] == "photo":
            idx = int(c["id"][1:])
            src = PHOTO_BASE + idx * PHOTO_STEP
            items.append({"kind": "visual", "start": src, "end": round(src + r["d"], 3), "id": r["id"], "photo": c["path"], "at": round(clock, 3)})
        else:
            items.append({"kind": "visual", "start": r["start"], "end": r["end"], "id": r["id"], "file": c["name"], "at": round(clock, 3)})
        clock += r["d"]
    m["items"], m["total_s"], m["dropped"], m["note"] = items, round(clock, 3), dropped, " ".join(notes)
    return m


def fmt_len(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


# ------------------------------------------------------------------------------------------------------------- what the page shows
def default_summary(m: dict) -> str:
    cands = {c["id"]: c for c in m["cands"]}
    ids = [i["id"] for i in m.get("items") or []]
    if not ids:
        return ""
    first, last = cands[ids[0]]["what"], cands[ids[-1]]["what"]
    mid = [cands[i]["what"] for i in ids[len(ids) // 4: len(ids) * 3 // 4: max(1, len(ids) // 6)]][:3]
    low = lambda s: (s[:1].lower() + s[1:]).rstrip(".")                   # noqa: E731
    return f"It opens on {low(first)}" + (f", moves through {', '.join(low(x) for x in mid)}" if mid else "") + f", and ends on {low(last)}."


def card(m: dict, *, title: str = "", notes: str = "", job: str = "") -> dict:
    """The plan for the person to check: plain words and the shots in order. Pure data; the page draws it."""
    cands = {c["id"]: c for c in m["cands"]}
    items = m.get("items") or []
    n_photo = sum(1 for i in items if "photo" in i)
    n_clip = len(items) - n_photo
    style = m.get("style") or "montage"
    song = m.get("song") or {}
    pace = pace_key(m.get("pace"))
    parts = []
    if n_photo:
        parts.append(f"{n_photo} photo{'s' if n_photo != 1 else ''}")
    if n_clip:
        parts.append(f"{n_clip} clip{'s' if n_clip != 1 else ''}")
    head = f"A {fmt_len(m.get('total_s') or 0)} {style} from {' and '.join(parts) or 'no shots yet'}"
    head += f", {PACE[pace]['label'].split(':')[0].lower()}"
    if song.get("title"):
        head += f", to “{song['title']}”" + (f" ({round(song['bpm'])} BPM)" if song.get("bpm") else "")
    removed = set(m.get("removed") or [])
    out = {"style": style, "title": title, "headline": head + ".", "summary": m.get("summary") or default_summary(m), "you_said": str(notes or "").strip()[:400],
           "why": m.get("why") or "", "line": m.get("line") or "", "total_s": m.get("total_s"), "pace": pace, "pace_label": PACE[pace]["label"],
           "target_s": m.get("target_s"), "dissolves": bool(m.get("dissolves")), "keep_sound": bool(m.get("keep_sound")),
           "song": {k: song.get(k) for k in ("title", "bpm", "why", "mood", "file")}, "note": m.get("note") or "",
           "strip": [{"id": i["id"], "kind": "photo" if "photo" in i else "clip", "what": cands[i["id"]]["what"], "d": round(i["end"] - i["start"], 1),
                      "thumb": f"mt_{i['id']}.jpg"} for i in items],
           "left_out": [{"id": g["id"], "what": g.get("what", ""), "why": g.get("why", ""), "thumb": f"mt_{g['id']}.jpg"} for g in (m.get("left_out") or [])
                        if g["id"] in cands][:60],
           "removed": [{"id": i, "what": cands[i]["what"], "thumb": f"mt_{i}.jpg"} for i in (m.get("removed") or []) if i in cands and i in removed],
           "job": job}
    reasons = {}
    for g in (m.get("left_out") or []) + (m.get("dropped") or []):
        reasons[g.get("why", "")] = reasons.get(g.get("why", ""), 0) + 1
    out["left_out_line"] = ("; ".join(f"{n} left out ({w})" for w, n in reasons.items() if w)) if reasons else ""
    return out


# ------------------------------------------------------------------------------------------------------------------- folders
def scan_photos(folder: Path, skipped, limit: int = 400) -> list[dict]:
    """Every photo in the folder (any subfolder that is not ours or Final Cut's): [{name, path, size, mtime}]. A montage folder is often
    all photos in its top level, unlike a normal edit, where photos only count inside a B-roll or photos folder."""
    from .foldergfx import is_graphic
    from .photos import PHOTO_EXT
    out = []
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in PHOTO_EXT or p.name.startswith(".") or not p.is_file():
            continue
        rel = p.relative_to(folder)
        if skipped(rel.parts[:-1]):
            continue
        try:
            if is_graphic(p, rel.parts):
                continue
            st = p.stat()
        except OSError:
            continue
        out.append({"name": p.name, "path": str(p), "size": st.st_size, "mtime": int(st.st_mtime)})
        if len(out) >= limit:
            break
    seen: dict[str, int] = {}
    for o in out:                                                           # two photos with the same name in different folders stay apart
        seen[o["name"]] = seen.get(o["name"], 0) + 1
        if seen[o["name"]] > 1:
            o["name"] = f"{Path(o['name']).stem} ({seen[o['name']]}){Path(o['name']).suffix}"
    return out


SUPPORT_DIRS = {"b-roll", "broll", "b roll", "graphics", "logos", "logo", "overlays", "stock"}


def quick_counts(folder: Path, skipped) -> dict:
    """What is in a folder, cheaply (no probing): {videos, photos, audio_files} for the page's first guess before anything is read."""
    from .localproj import VIDEO_EXT
    from .photos import PHOTO_EXT
    v = p = 0
    for f in Path(folder).rglob("*"):
        try:
            if not f.is_file() or f.name.startswith("."):
                continue
            rel = f.relative_to(folder)
            if skipped(rel.parts[:-1]):
                continue
        except (OSError, ValueError):
            continue
        ext = f.suffix.lower()
        v += ext in VIDEO_EXT
        if ext in PHOTO_EXT and not ({x.lower() for x in rel.parts[:-1]} & SUPPORT_DIRS) and "logo" not in f.stem.lower():
            p += 1                                                        # (photos in B-roll / Graphics folders and logos are not what the video is made of)
        if v + p > 5000:
            break
    return {"videos": v, "photos": p}


def shot_time(path: str | None, kind: str, f: dict | None = None) -> float:
    """When a shot was taken: a photo's EXIF date, a clip's camera clock, else the file's date. Used to play the day in order."""
    import time as _t
    fallback = float((f or {}).get("mtime") or 0)
    if not path:
        return fallback
    try:
        if kind == "photo":
            from PIL import Image
            with Image.open(path) as im:
                ex = im.getexif()
                raw = ex.get(36867) or ex.get(306) or (ex.get_ifd(0x8769).get(36867) if hasattr(ex, "get_ifd") else None)
            if raw:
                return _t.mktime(_t.strptime(str(raw)[:19], "%Y:%m:%d %H:%M:%S"))
        else:
            from ..core import ffmpeg
            from .combine import _created
            c = _created(ffmpeg.probe(path))
            if c:
                return c.timestamp()
    except Exception:  # noqa: BLE001 - the file's date is a fine answer
        pass
    return fallback
