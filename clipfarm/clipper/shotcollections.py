"""Collections: shots you want to find again, kept by name ("Student interaction", "Campus in autumn").

A collection is a list of REFERENCES into the footage catalogue: a catalogue id (the sha1 of the file's full path, so re-rooting a drive
keeps them), and for a video the moment that was saved (start, end in seconds). Nothing is copied or moved and the footage is never
touched; deleting a collection only forgets the list. An item whose file has left the catalogue (a drive that is unplugged, a file that
moved or was deleted) stays in the list, shown as missing with the name it had, until the person removes it: the catalogue's prune
never touches this file.

One small JSON file, `<app home>/data/collections.json`, written atomically (a temp file, then a rename), so a crash never leaves half a
list. Names are unique without regard to case. The class is thread-safe (the page and the engine share one).

    {"collections": [{"id": "c1a2b3c4d5", "name": "Student interaction", "created": 1.7e9, "updated": 1.7e9, "used": 1.7e9,
                      "items": [{"file_id": "<sha1>", "kind": "video", "start": 12.0, "end": 17.5, "k": 6, "rel": "B-roll/Quad/clip.mp4",
                                 "what": "two students laughing on a bench", "dur": 93.2, "note": "", "added": 1.7e9}]}]}
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import uuid
from pathlib import Path

NAME_MAX = 60
FILE_ID = re.compile(r"[0-9a-f]{8,64}")
COLL_ID = re.compile(r"c[0-9a-f]{6,32}")
MAX_ITEMS_PER_CALL = 500
MAX_ITEMS = 20000                    # a collection past this is a mistake (it is a list of shots to find again, not an archive)
MAX_COLLECTIONS = 500
MIN_CHANCE = 0.002                   # SigLIP's own odds that the picture shows the words: ~0.1 is a clear match, under 0.001 is not


def clean_name(name) -> str:
    """The name as stored: trimmed, inner whitespace collapsed. ValueError (in plain words) when there is nothing to keep."""
    n = " ".join(str(name or "").split())
    if not n:
        raise ValueError("give the collection a name")
    if len(n) > NAME_MAX:
        raise ValueError(f"keep the name under {NAME_MAX} letters")
    return n


def item_key(it: dict) -> str:
    """What makes two items the same shot: the file and the moment (a photo has no moment)."""
    return f"{it['file_id']}@{float(it.get('start') or 0):.2f}@{float(it.get('end') or 0):.2f}"


def clean_item(raw: dict) -> dict:
    """One item as the page sends it ({id | file_id, kind, start, end, k, rel, what, dur, note}) -> the stored shape. ValueError if unusable."""
    if not isinstance(raw, dict):
        raise ValueError("an item must be an object")
    fid = str(raw.get("file_id") or raw.get("id") or "")
    if not FILE_ID.fullmatch(fid):
        raise ValueError("that isn't a shot from your footage library")
    kind = str(raw.get("kind") or "video")
    if kind not in ("video", "photo"):
        raise ValueError("a shot is a video or a photo")
    try:
        start, end, dur = (float(raw.get(n) or 0) for n in ("start", "end", "dur"))
        k = max(0, int(raw.get("k") or 0))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("a shot's times are numbers") from None
    if not all(math.isfinite(x) and abs(x) < 1e7 for x in (start, end, dur)):       # (NaN, infinity, absurd)
        raise ValueError("a shot's times are numbers")
    start, end, dur = max(0.0, start), max(0.0, end), max(0.0, dur)
    if kind == "photo":
        start = end = 0.0
    elif end <= start:
        raise ValueError("a video shot needs an end after its start")
    return {"file_id": fid, "kind": kind, "start": round(start, 2), "end": round(end, 2), "k": k,
            "rel": str(raw.get("rel") or "")[:300], "what": " ".join(str(raw.get("what") or "").split())[:200],
            "dur": round(dur, 2), "note": " ".join(str(raw.get("note") or "").split())[:300]}


class Collections:
    def __init__(self, path: Path, now=time.time):
        self.path = Path(path)
        self.now = now
        self.lock = threading.RLock()
        self._data: dict | None = None

    # ------------------------------------------------------------------------------------------------ the file
    def _load(self) -> dict:
        if self._data is not None:
            return self._data
        data = {"collections": []}
        try:
            raw = json.loads(self.path.read_text())
            if isinstance(raw, dict) and isinstance(raw.get("collections"), list):
                data = {"collections": [self._sane(c) for c in raw["collections"] if isinstance(c, dict) and COLL_ID.fullmatch(str(c.get("id") or ""))]}
        except (OSError, ValueError):
            pass                                                     # (no file yet, or a damaged one: start empty rather than refuse to open)
        self._data = data
        return data

    @staticmethod
    def _sane(c: dict) -> dict:
        items = []
        for it in c.get("items") or []:
            try:
                ci = clean_item(it)
            except ValueError:
                continue
            ci["added"] = float(it.get("added") or 0) if isinstance(it.get("added"), (int, float)) else 0.0
            items.append(ci)
        t = lambda k: float(c.get(k) or 0) if isinstance(c.get(k), (int, float)) else 0.0
        return {"id": str(c["id"]), "name": " ".join(str(c.get("name") or "Collection").split())[:NAME_MAX] or "Collection",
                "created": t("created"), "updated": t("updated"), "used": t("used"), "items": items}

    def _save(self) -> None:
        data = self._load()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".part")
        tmp.write_text(json.dumps(data, indent=1))
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        tmp.replace(self.path)

    def reload(self) -> None:
        with self.lock:
            self._data = None

    # ------------------------------------------------------------------------------------------------ reading
    def _find(self, cid: str) -> dict:
        for c in self._load()["collections"]:
            if c["id"] == cid:
                return c
        raise KeyError("no such collection")

    def get(self, cid: str) -> dict:
        with self.lock:
            return json.loads(json.dumps(self._find(str(cid))))

    def list(self) -> list[dict]:
        """Every collection (a copy), the one used last first."""
        with self.lock:
            out = json.loads(json.dumps(self._load()["collections"]))
        return sorted(out, key=lambda c: (-(c["used"] or c["updated"] or c["created"]), c["name"].lower()))

    def holding(self, items: list[dict]) -> dict[str, int]:
        """{collection id: how many of these items it already holds} (the 'already in' ticks of the Add menu)."""
        keys = []
        for raw in items or []:
            try:
                keys.append(item_key(clean_item(raw)))
            except ValueError:
                continue
        out = {}
        with self.lock:
            for c in self._load()["collections"]:
                have = {item_key(i) for i in c["items"]}
                out[c["id"]] = sum(1 for k in keys if k in have)
        return out

    # ------------------------------------------------------------------------------------------------ changing
    def _unique(self, name: str, except_id: str = "") -> str:
        n = clean_name(name)
        low = n.lower()
        for c in self._load()["collections"]:
            if c["id"] != except_id and c["name"].lower() == low:
                raise ValueError(f"there is already a collection called “{c['name']}”")
        return n

    def create(self, name: str) -> dict:
        with self.lock:
            data = self._load()
            if len(data["collections"]) >= MAX_COLLECTIONS:
                raise ValueError("that's a lot of collections: remove one first")
            n = self._unique(name)
            t = self.now()
            c = {"id": "c" + uuid.uuid4().hex[:10], "name": n, "created": t, "updated": t, "used": t, "items": []}
            data["collections"].append(c)
            self._save()
            return json.loads(json.dumps(c))

    def rename(self, cid: str, name: str) -> dict:
        with self.lock:
            c = self._find(str(cid))
            c["name"] = self._unique(name, except_id=c["id"])
            c["updated"] = self.now()
            self._save()
            return json.loads(json.dumps(c))

    def delete(self, cid: str) -> None:
        """Forgets the list. The shots' files are not touched (a collection only ever pointed at them)."""
        with self.lock:
            data = self._load()
            c = self._find(str(cid))
            data["collections"].remove(c)
            self._save()

    def add(self, cid: str, items: list[dict]) -> dict:
        """-> {"added": [item_key...], "duplicates": n, "collection": ...}. The same shot twice is not an error: it is left as it was."""
        if not isinstance(items, list) or not items:
            raise ValueError("pick a shot to add")
        if len(items) > MAX_ITEMS_PER_CALL:
            raise ValueError("add fewer shots at once")
        with self.lock:
            c = self._find(str(cid))
            have = {item_key(i) for i in c["items"]}
            added, dup = [], 0
            t = self.now()
            clean = [clean_item(raw) for raw in items]                 # (all or nothing: one bad item changes nothing)
            for it in clean:
                k = item_key(it)
                if k in have:
                    dup += 1
                    continue
                if len(c["items"]) >= MAX_ITEMS:
                    raise ValueError("this collection is full")
                it["added"] = t
                c["items"].append(it)
                have.add(k)
                added.append(k)
            if added:
                c["updated"] = t
            c["used"] = t
            self._save()
            return {"added": added, "duplicates": dup, "collection": json.loads(json.dumps(c))}

    def remove(self, cid: str, items: list[dict]) -> int:
        """Takes shots out of the list (by file and moment). Returns how many went."""
        if not isinstance(items, list):
            raise ValueError("say which shots")
        with self.lock:
            c = self._find(str(cid))
            drop = set()
            for raw in items:
                try:
                    drop.add(item_key(clean_item(raw)))
                except ValueError:
                    continue
            before = len(c["items"])
            c["items"] = [i for i in c["items"] if item_key(i) not in drop]
            gone = before - len(c["items"])
            if gone:
                c["updated"] = self.now()
                self._save()
            return gone

    def set_note(self, cid: str, item: dict, note: str) -> None:
        with self.lock:
            c = self._find(str(cid))
            key = item_key(clean_item(item))
            for i in c["items"]:
                if item_key(i) == key:
                    i["note"] = " ".join(str(note or "").split())[:300]
                    c["updated"] = self.now()
                    self._save()
                    return
            raise KeyError("that shot isn't in this collection")

    def move(self, cid: str, item: dict, to: int) -> None:
        """Puts one shot at position `to` (0 = first): the order the collection shows and sends to Final Cut."""
        with self.lock:
            c = self._find(str(cid))
            key = item_key(clean_item(item))
            cur = next((n for n, i in enumerate(c["items"]) if item_key(i) == key), None)
            if cur is None:
                raise KeyError("that shot isn't in this collection")
            it = c["items"].pop(cur)
            c["items"].insert(max(0, min(int(to), len(c["items"]))), it)
            c["updated"] = self.now()
            self._save()


# ---------------------------------------------------------------------------------------------------- with the catalogue
def shot_of(cat, it: dict) -> dict:
    """One stored item as the shot the Find footage grid and zoom player take: {id, path, rel, kind, k, t, start, end, dur, step, frames,
    what, quality, w, h, ...} read from the catalogue's record now (so a re-described file shows its new words), or, when the file isn't
    in the catalogue any more, the item as it was saved with `missing: True`."""
    rec = cat.load(it["file_id"]) if cat is not None else None
    base = {"id": it["file_id"], "kind": it["kind"], "start": it["start"], "end": it["end"], "note": it.get("note") or "", "added": it.get("added") or 0.0}
    if not rec or rec.get("error") or not rec.get("path"):
        rel = it.get("rel") or ""
        return {**base, "missing": True, "path": "", "rel": rel or "(a file that left the library)", "what": it.get("what") or "",
                "k": it.get("k") or 0, "t": it["start"], "dur": it.get("dur") or 0.0, "step": 0, "frames": 0, "score": 0, "chance": 0, "quality": None}
    step = float(rec.get("step") or 0) or 2.0
    frames = int(rec.get("frames") or 0)
    k = min(max(0, int(it.get("k") or round(it["start"] / step))), max(0, frames - 1)) if frames else 0
    t = k * step
    scene = next((s for s in (rec.get("scenes") or []) if s.get("start", 0) - 0.5 <= t <= s.get("end", 0) + 0.5), None) if it["kind"] == "video" else \
        ((rec.get("scenes") or [None])[0])
    return {**base, "missing": False, "path": rec["path"], "rel": rec.get("rel") or it.get("rel") or "", "k": k, "t": round(t, 2),
            "dur": float(rec.get("dur") or it.get("dur") or 0), "step": step, "frames": frames,
            "what": (scene or {}).get("what") or it.get("what") or None, "quality": (scene or {}).get("quality"), "score": 0, "chance": 0,
            "w": rec.get("w"), "h": rec.get("h"), "fps": rec.get("fps")}


def shots_of(cat, coll: dict) -> list[dict]:
    return [shot_of(cat, it) for it in coll["items"]]


def cover_of(cat, coll: dict, n: int = 4) -> list[dict]:
    """The first few shots that can be shown (not missing), as {id, k}: the card's thumbnails."""
    out = []
    for it in coll["items"]:
        if len(out) >= n:
            break
        rec = cat.load(it["file_id"]) if cat is not None else None
        if rec and not rec.get("error") and rec.get("frames"):
            out.append({"id": it["file_id"], "k": shot_of(cat, it)["k"], "kind": it["kind"]})
    return out


def words_of(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", str(text or "").lower()) if len(w) > 1]


def filter_shots(cat, shots: list[dict], q: str) -> list[dict]:
    """The collection's shots that go with the words, best first: the catalogue's picture search ranks the files (the same search as Find
    footage), and a word that is in a shot's description, folder names or note counts too (so a shot whose file ranks low still shows when
    you type its words). No words: the list as it is."""
    q = " ".join(str(q or "").split())[:200]
    if not q:
        return shots
    rank: dict[str, float] = {}
    try:
        for s in cat.search(q, n=2000, per_file=1):
            if float(s.get("chance") or 0) < MIN_CHANCE:                # (the catalogue always returns its best n: a file that clearly isn't it is left out)
                continue
            rank[s["id"]] = max(rank.get(s["id"], -9.0), float(s.get("score") or 0))
    except Exception:  # noqa: BLE001 - no picture model right now: the words alone still filter
        rank = {}
    qw = words_of(q)
    out = []
    for s in shots:
        hay = " ".join([str(s.get("what") or ""), str(s.get("rel") or ""), str(s.get("note") or "")]).lower()
        hits = sum(1 for w in qw if w in hay)
        if s["id"] in rank or hits:
            out.append((rank.get(s["id"], -1.0) + 0.05 * hits, s))
    out.sort(key=lambda x: -x[0])
    return [s for _, s in out]
