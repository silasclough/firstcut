"""Editing-style presets: named, global, kept in the app's own folder (`<home>/styles/presets.json`) and shared by short and long videos.

A preset grows: the person learns a style from a link, renames it, adds another video to it (its numbers are combined with the first), edits the plain
lines it wrote, adds rules of their own. Applying a preset to "How it edits" (short or long) fills that mode's points box and the settings the points
state, the same way typed points do; nothing is applied until the person presses "Use this".

    {id, name, points [text], rules [text], style {partial style object}, flat {settings}, sources [{url, title, kind, added, measurements}],
     evidence [text], how "ai"|"numbers", edited bool, learned_points [text], frames [file names], created, updated}

`points` are what the person sees and edits; `learned_points` are what the last learning wrote. Adding a video re-learns (`learned_points`, `style`,
`evidence` change); `points` are replaced by the new lines only while the person has not edited them.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from ..core.log import get_logger
from . import stylelearn as SL
from . import styles_profile as SPF

log = get_logger("clipper.stylepresets")
MAX_PRESETS = 40
MAX_SOURCES = 40
MAX_RULES = 20
LINE_MAX = 160


def _tidy(text: str) -> str:
    return " ".join(str(text or "").split()).strip(" -•*")[:LINE_MAX]


def _lines(raw) -> list[str]:
    """Plain lines from a text (one per line) or a list; empty and repeated lines dropped."""
    if isinstance(raw, str):
        raw = raw.splitlines()
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[str] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, (str, int, float)):
            continue
        t = _tidy(str(x))
        if t and t not in out:
            out.append(t)
    return out[:30]


class Presets:
    def __init__(self, root: Path):
        self.dir = Path(root) / "styles"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.file = self.dir / "presets.json"
        self.frames = self.dir / "preset_frames"
        self.lock = threading.RLock()

    # -------------------------------------------------------------------------------------------- storage
    def _load(self) -> dict:
        try:
            d = json.loads(self.file.read_text())
        except (OSError, ValueError):
            d = {}
        pres = d.get("presets") if isinstance(d, dict) else None
        return {"presets": [p for p in (pres or []) if isinstance(p, dict) and p.get("id")]}

    def _save(self, d: dict) -> None:
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(d, indent=1))
        tmp.replace(self.file)

    def _find(self, d: dict, pid: str) -> dict:
        p = next((x for x in d["presets"] if x["id"] == pid), None)
        if p is None:
            raise KeyError("no such style")
        return p

    # -------------------------------------------------------------------------------------------- reading
    def all(self) -> list[dict]:
        return self._load()["presets"]

    def get(self, pid: str) -> dict:
        return self._find(self._load(), pid)

    def view(self, p: dict) -> dict:
        """What the page shows of one preset (never the raw measurements)."""
        return {"id": p["id"], "name": p.get("name") or "Untitled style", "points": p.get("points") or [], "rules": p.get("rules") or [],
                "evidence": p.get("evidence") or [], "how": p.get("how") or "", "edited": bool(p.get("edited")), "updated": p.get("updated"),
                "learned_points": p.get("learned_points") or [], "style": p.get("style") or {},
                "sources": [{"title": s.get("title") or "", "url": s.get("url") or "", "kind": s.get("kind") or "", "added": s.get("added")} for s in p.get("sources") or []],
                "frames": list(p.get("frames") or [])}

    def summary(self) -> list[dict]:
        return [self.view(p) for p in sorted(self.all(), key=lambda x: -(x.get("updated") or 0))]

    def frame_path(self, pid: str, name: str) -> Path | None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}\.jpg", str(name or "")):
            return None
        p = self.frames / pid / name
        return p if p.is_file() else None

    # -------------------------------------------------------------------------------------------- changes
    def _name(self, d: dict, name: str, keep: str | None = None) -> str:
        base = _tidy(name)[:60] or "My style"
        names = {(p.get("name") or "").lower() for p in d["presets"] if p["id"] != keep}
        out, k = base, 2
        while out.lower() in names:
            out, k = f"{base} {k}", k + 1
        return out

    def create(self, name: str = "", points=None, rules=None) -> dict:
        with self.lock:
            d = self._load()
            if len(d["presets"]) >= MAX_PRESETS:
                raise ValueError("that is as many styles as can be kept: delete one first")
            now = int(time.time())
            pts = _lines(points)
            p = {"id": "st" + uuid.uuid4().hex[:10], "name": self._name(d, name or "My style"), "points": pts, "rules": _lines(rules)[:MAX_RULES], "style": {}, "flat": {},
                 "sources": [], "evidence": [], "how": "", "edited": bool(pts), "learned_points": [], "frames": [], "created": now, "updated": now}
            d["presets"].append(p)
            self._save(d)
            return self.view(p)

    def rename(self, pid: str, name: str) -> dict:
        if not _tidy(name):
            raise ValueError("give the style a name")
        with self.lock:
            d = self._load()
            p = self._find(d, pid)
            p["name"] = self._name(d, name, keep=pid)
            p["updated"] = int(time.time())
            self._save(d)
            return self.view(p)

    def delete(self, pid: str) -> None:
        with self.lock:
            d = self._load()
            self._find(d, pid)
            d["presets"] = [p for p in d["presets"] if p["id"] != pid]
            self._save(d)
        shutil.rmtree(self.frames / pid, ignore_errors=True)

    def set_points(self, pid: str, points) -> dict:
        """The person edits the plain lines (a text or a list). Their words stay: later learning does not replace edited points."""
        with self.lock:
            d = self._load()
            p = self._find(d, pid)
            p["points"] = _lines(points)
            p["edited"] = p["points"] != (p.get("learned_points") or [])
            p["updated"] = int(time.time())
            self._save(d)
            return self.view(p)

    def reset_points(self, pid: str) -> dict:
        """Back to what the last learning wrote."""
        with self.lock:
            d = self._load()
            p = self._find(d, pid)
            p["points"] = list(p.get("learned_points") or [])
            p["edited"] = False
            p["updated"] = int(time.time())
            self._save(d)
            return self.view(p)

    def add_rule(self, pid: str, text: str) -> dict:
        t = _tidy(text)
        if not t:
            raise ValueError("write the rule first")
        with self.lock:
            d = self._load()
            p = self._find(d, pid)
            rules = p.get("rules") or []
            if len(rules) >= MAX_RULES:
                raise ValueError("that is as many rules as one style keeps: remove one first")
            if t not in rules:
                rules.append(t)
            p["rules"] = rules
            p["updated"] = int(time.time())
            self._save(d)
            return self.view(p)

    def remove_rule(self, pid: str, index: int) -> dict:
        with self.lock:
            d = self._load()
            p = self._find(d, pid)
            rules = p.get("rules") or []
            if not 0 <= int(index) < len(rules):
                raise ValueError("no such rule")
            rules.pop(int(index))
            p["rules"] = rules
            p["updated"] = int(time.time())
            self._save(d)
            return self.view(p)

    def add_learned(self, pid: str | None, name: str, sources: list[dict], frames: list[Path], ai=None) -> dict:
        """Put freshly measured videos into a preset (a new one when `pid` is None), combine ALL its sources again and rewrite its style, evidence and
        learned points. The person's own points and rules are kept. -> the preset's view plus `kept_edits` (True when the new points were not used
        because the person had edited theirs)."""
        with self.lock:
            d = self._load()
            now = int(time.time())
            if pid is None:
                if len(d["presets"]) >= MAX_PRESETS:
                    raise ValueError("that is as many styles as can be kept: delete one first")
                p = {"id": "st" + uuid.uuid4().hex[:10], "name": self._name(d, name or (sources[0].get("title") if sources else "") or "Learned style"), "points": [], "rules": [], "style": {}, "flat": {},
                     "sources": [], "evidence": [], "how": "", "edited": False, "learned_points": [], "frames": [], "created": now, "updated": now}
                d["presets"].append(p)
            else:
                p = self._find(d, pid)
            have = {(s.get("url"), s.get("title")) for s in p["sources"]}
            for s in sources:
                if (s.get("url"), s.get("title")) in have and s.get("url"):
                    continue
                p["sources"].append(s)
            p["sources"] = p["sources"][-MAX_SOURCES:]
            kept = self._keep_frames(p["id"], frames, p.get("frames") or [])
            p["frames"] = kept
            images = []
            for n in kept:
                try:
                    images.append((self.frames / p["id"] / n).read_bytes())
                except OSError:
                    pass
            out = SL.summarize(p["sources"], ai, images)
            p["style"], p["flat"], p["evidence"], p["how"] = out["style"], out["flat"], out["evidence"], out["how"]
            kept_edits = bool(p.get("edited")) and bool(p.get("points"))
            p["learned_points"] = out["points"]
            if not kept_edits:
                p["points"], p["edited"] = list(out["points"]), False
            p["updated"] = now
            self._save(d)
            v = self.view(p)
            v["kept_edits"] = kept_edits
            v["notes"] = out["notes"]
            return v

    def _keep_frames(self, pid: str, new: list[Path], old: list[str]) -> list[str]:
        dest = self.frames / pid
        dest.mkdir(parents=True, exist_ok=True)
        names = list(old)
        for src in new:
            try:
                tgt = dest / Path(src).name
                if Path(src).resolve() != tgt.resolve():
                    shutil.copyfile(src, tgt)
                if tgt.name not in names:
                    names.append(tgt.name)
            except OSError:
                continue
        names = names[-SL.KEEP_FRAMES:]
        for f in dest.glob("*.jpg"):                                         # (only the kept frames stay on disk)
            if f.name not in names:
                f.unlink(missing_ok=True)
        return names

    # -------------------------------------------------------------------------------------------- applying
    def lines_for(self, p: dict) -> list[str]:
        """What goes into a "How it edits" box: the person's rules first, then the style's lines."""
        out: list[str] = []
        for t in list(p.get("rules") or []) + list(p.get("points") or []):
            if t and t not in out:
                out.append(t)
        return out[:30]

    def settings_for(self, p: dict) -> dict:
        """The flat editing settings the preset states (styles_profile.SETTING_KEYS), checked."""
        flat = SPF.clean_settings(p.get("flat") or SPF.flat_from_nested(p.get("style") or {}))
        flat.pop("profile", None)
        flat.pop("target_seconds", None)                                     # (how long the learned videos were is no default for yours: a 30 s reel must not shorten every long video)
        return flat
