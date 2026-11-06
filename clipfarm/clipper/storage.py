"""Storage: take the rendered videos of old projects off the disk, keep everything needed to render them again.

A rendered MP4 is big; the plan behind it (the cuts, the B-roll that was placed, the words, the settings) is a few kilobytes. So an old
project can lose its videos and still be made again, the same cut, with one button (Re-render): nothing is planned again.

Per project (long video) or job (short clips) the files fall into four groups:
    videos    the rendered MP4s: <project>/Output/*.mp4, the job folder's clip, short clips' mp4 and Clips/ folders
    finalcut  <project>/Final Cut/** and the .fcpxml next to it. Left alone unless "Also remove Final Cut folders" is on: a project
              already imported into Final Cut points at these files.
    working   caches that are rebuilt when they are missing: photo animations, cleaned voice, rough cuts, end cards, fcp zips,
              and a downloaded copy of a link's video (downloaded again)
    keep      plan.json (and its versions), the words, files.json, notes, credits, chapters, the quality report, small thumbnails: small, never removed

Never touched: the footage itself (it is read where it lives; project folders only hold links to it, and a link is never followed),
anything inside a project that is being made, has changes waiting for Re-render, or is pinned ("Keep"). Archiving only runs after the
plan that renders the video again has been read back and found whole.

`Storage.archive_*` removes; `after_run` clears the "removed" mark when the video exists again; `check_restore` says plainly what is
missing (the footage folder, a drive, the link) before a re-render is started.
"""

from __future__ import annotations

import json
import re
import os
import time
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.storage")

DAYS = (0, 7, 14, 30, 60)                                            # "Remove old videos after": 0 = never (the default)
VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
CACHE_EXT = VIDEO_EXT | {".wav", ".m4a", ".mp3", ".aac", ".png", ".jpg", ".jpeg", ".webp", ".zip"}     # (inside .aivideo: media caches only)
JOB_WORK_EXT = {".zip", ".wav", ".m4a", ".mp3", ".aac"}              # (in a job folder: rebuilt; its .jpg thumbnails are the posters and stay)
ACTIVE = ("queued", "joining", "downloading", "transcribing", "finding", "rendering", "running", "uploading")
GROUPS = ("videos", "finalcut", "working")


def _files(root: Path):
    """The regular files under `root`. A symlink is never followed (a project folder holds links to the original footage) and never counted."""
    try:
        if not root.is_dir() or root.is_symlink():
            return
    except OSError:
        return
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = [d for d in dns if not os.path.islink(os.path.join(dp, d))]
        for f in fns:
            p = Path(dp) / f
            try:
                if not p.is_symlink() and p.is_file():
                    yield p
            except OSError:
                continue


def _size(paths) -> int:
    n = 0
    for p in paths:
        try:
            n += p.stat().st_size
        except OSError:
            pass
    return n


def _json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------------------------------------- what is where
VERSIONED_NAME = re.compile(r"\sv\d{1,3}$")                                  # ("Fall Recap v3")


def long_files(proj: dict, jobdir: Path) -> dict[str, list[Path]]:
    """The removable files of a long-video project, by group."""
    home = Path(proj.get("home") or proj.get("folder") or "")
    out: dict[str, list[Path]] = {g: [] for g in GROUPS}
    # Only what FirstCut itself wrote: "<title> v3.mp4", "Final Cut/<title> v3/", "<title> v3.fcpxml". A project without a folder of its own works
    # inside the footage folder, so an "Output" or "Final Cut" folder there could be someone else's: anything that isn't named like our
    # versions, or that is one of the project's own source files, stays.
    mine = lambda name: bool(VERSIONED_NAME.search(name))
    try:
        from .localproj import State
        sources = {str(Path(f.get("path") or "").resolve()) for f in (State(home).get("files.json") or []) if f.get("path")}
    except Exception:  # noqa: BLE001 - no list of sources: only our own names are touched anyway
        sources = set()
    ours = lambda p: mine(p.stem) and str(p.resolve()) not in sources
    out["videos"] += [p for p in _files(home / "Output") if p.suffix.lower() in VIDEO_EXT and ours(p)]
    fc = home / "Final Cut"
    out["finalcut"] += [p for p in _files(fc) if p.relative_to(fc).parts and mine(p.relative_to(fc).parts[0]) and str(p.resolve()) not in sources]
    try:
        out["finalcut"] += [p for p in home.glob("*.fcpxml") if p.is_file() and not p.is_symlink() and mine(p.stem)]
    except OSError:
        pass
    work = home / ".aivideo"
    for p in _files(work):
        if p.relative_to(work).parts[0] != "backups" and p.suffix.lower() in CACHE_EXT:
            out["working"].append(p)
    for p in _files(jobdir):
        ext = p.suffix.lower()
        if ext in VIDEO_EXT and p.parent == jobdir:                  # (the finished clip; a video deeper down is a render's scratch: working)
            out["videos"].append(p)
        elif ext in VIDEO_EXT or ext in JOB_WORK_EXT:
            out["working"].append(p)
    return out


def short_files(job: dict, jobdir: Path) -> dict[str, list[Path]]:
    """The removable files of a short-clips job, by group."""
    out: dict[str, list[Path]] = {g: [] for g in GROUPS}
    url = bool((job.get("source") or {}).get("url"))
    for p in _files(jobdir):
        ext = p.suffix.lower()
        if p.parent == jobdir and p.stem == "source":
            if url and ext in VIDEO_EXT:                             # (a download: fetched again. A local file is only a link here, and links are skipped)
                out["working"].append(p)
        elif ext in VIDEO_EXT and p.parent == jobdir:
            out["videos"].append(p)
        elif ext in VIDEO_EXT or ext in JOB_WORK_EXT:
            out["working"].append(p)
    home = Path(job.get("app_home") or "")
    if job.get("app_home") and home.is_dir():
        try:
            for d in home.iterdir():
                if d.is_dir() and not d.is_symlink() and (d.name == "Clips" or d.name.startswith("Clips (before")):
                    out["videos"] += [p for p in _files(d) if p.suffix.lower() in VIDEO_EXT]
        except OSError:
            pass
    return out


def _sizes(cats: dict[str, list[Path]]) -> dict[str, int]:
    return {g: _size(v) for g, v in cats.items()}


# ----------------------------------------------------------------------------------------------------- the manager
class Storage:
    def __init__(self, cfg, store, projects, runner=None, settings=None, now=time.time):
        self.cfg, self.store, self.projects, self.runner = cfg, store, projects, runner
        self.settings = settings or (lambda: {})
        self.now = now

    # -- the setting
    def options(self) -> tuple[int, bool]:
        s = (self.settings() or {}).get("storage") or {}
        days = s.get("auto_days")
        return (int(days) if days in DAYS else 0), bool(s.get("also_finalcut"))

    # -- what exists
    def long_projects(self) -> list[dict]:
        out = []
        for pid in self.projects.ids():
            try:
                p = self.projects.get(pid)
            except (KeyError, OSError, ValueError):
                continue
            if p.get("mode") == "local":
                out.append(p)
        return out

    def short_jobs(self) -> list[dict]:
        try:
            return [j for j in self.store.list() if (j.get("source") or {}).get("app") == "studio" and not (j.get("options") or {}).get("project")]
        except Exception:  # noqa: BLE001
            return []

    def files_of(self, kind: str, obj: dict) -> dict[str, list[Path]]:
        if kind == "long":
            return long_files(obj, self.store.dir(obj["job"]))
        return short_files(obj, self.store.dir(obj["id"]))

    def last_activity(self, kind: str, obj: dict) -> float:
        """When it was last made or changed: the newest rendered clip (a file's own date is only used when no render time was saved), else when it was created."""
        stamps: list[float] = []
        job = obj
        if kind == "long":
            try:
                job = self.store.get(obj["job"])
            except KeyError:
                job = {}
        for c in job.get("clips") or []:
            try:
                if c.get("rendered_at"):
                    stamps.append(float(c["rendered_at"]))
            except (TypeError, ValueError):
                pass
        if not stamps:                                                    # (no render time saved: the newest video on disk says when)
            for p in self.files_of(kind, obj)["videos"][:60]:
                try:
                    stamps.append(p.stat().st_mtime)
                except OSError:
                    pass
        if not stamps:
            stamps.append(float(obj.get("created") or 0))
        return max(stamps)

    # -- what may be removed
    def _plan_ok(self, kind: str, obj: dict) -> str:
        """"" when the saved edit that renders it again is whole, else why not."""
        if kind == "long":
            home = Path(obj.get("home") or obj.get("folder") or "")
            work = home / ".aivideo"
            plan = _json(work / "plan.json")
            if not isinstance(plan, dict) or not isinstance(plan.get("items"), list) or not plan["items"]:
                return "its saved edit (the plan) is missing, so it couldn't be rendered again"
            pw = _json(work / "plan_words.json") or _json(work / "words.json")
            words = pw.get("words") if isinstance(pw, dict) else pw
            if not isinstance(words, list) or not words:
                return "its words aren't saved, so it couldn't be rendered again"
            if not isinstance(_json(work / "files.json"), list):
                return "its list of clips isn't saved, so it couldn't be rendered again"
            return ""
        d = self.store.dir(obj["id"])
        tr = _json(d / "transcript.json")
        if not isinstance(tr, dict) or not tr.get("words"):
            return "its words aren't saved, so the clips couldn't be made again"
        clips = obj.get("clips") or []
        if not clips or any(not isinstance(c.get("moment"), dict) for c in clips):
            return "the cuts of its clips aren't saved, so they couldn't be made again"
        return ""

    def can_archive(self, kind: str, iid: str, also_finalcut: bool | None = None) -> dict:
        """{"ok": bool, "why": str, "bytes": int, "sizes": {...}} for one project or job."""
        if also_finalcut is None:
            also_finalcut = self.options()[1]
        try:
            obj = self.projects.get(iid) if kind == "long" else self.store.get(iid)
        except (KeyError, OSError, ValueError):
            return {"ok": False, "why": "it isn't there", "bytes": 0, "sizes": {}}
        cats = self.files_of(kind, obj)
        sizes = _sizes(cats)
        gone = sizes["videos"] + sizes["working"] + (sizes["finalcut"] if also_finalcut else 0)
        res = {"ok": False, "why": "", "bytes": gone, "sizes": sizes}
        if obj.get("keep"):
            res["why"] = "it is kept (pinned)"
            return res
        try:
            job = self.store.get(obj["job"]) if kind == "long" else obj
        except KeyError:
            res["why"] = "it has no render job"
            return res
        status = job.get("status")
        if status in ACTIVE or (kind == "long" and self.runner is not None and getattr(self.runner, "running", None) == iid):
            res["why"] = "it is being made right now"
            return res
        if status in (None, "draft"):
            res["why"] = "it hasn't been made yet"
            return res
        if kind == "long":
            plan = _json(Path(obj.get("home") or obj.get("folder") or "") / ".aivideo" / "plan.json") or {}
            if plan.get("pending_render") or plan.get("pending_changes"):
                res["why"] = "it has changes waiting for Re-render"
                return res
        why = self._plan_ok(kind, obj)
        if why:
            res["why"] = why
            return res
        if gone <= 0:
            res["why"] = "there is nothing to remove" if not obj.get("archived") else "its video is already removed"
            return res
        res["ok"] = True
        return res

    # -- usage
    def usage(self, days: int | None = None, also_finalcut: bool | None = None) -> dict:
        """Totals and one row per project / job (biggest first): sizes by group, age, whether it can be removed (and why not), pinned."""
        auto, fc = self.options()
        days = auto if days is None else days
        if also_finalcut is None:
            also_finalcut = fc
        now = self.now()
        rows = []
        for kind, items in (("long", self.long_projects()), ("short", self.short_jobs())):
            for obj in items:
                iid = obj["id"]
                try:
                    cats = self.files_of(kind, obj)
                    sizes = _sizes(cats)
                    chk = self.can_archive(kind, iid, also_finalcut)
                    act = self.last_activity(kind, obj)
                except Exception:  # noqa: BLE001 - one odd project never hides the others
                    log.exception(f"storage: could not measure {kind} {iid}")
                    continue
                total = sum(sizes.values())
                title = obj.get("title") or (obj.get("source") or {}).get("filename") or iid
                if kind == "long":
                    try:
                        title = self.store.get(obj["job"]).get("title") or title
                    except KeyError:
                        pass
                age_days = max(0.0, (now - act) / 86400.0) if act else 0.0
                rows.append({"kind": kind, "id": iid, "title": str(title)[:90], "bytes": total, "sizes": sizes, "age_days": round(age_days, 1),
                             "last": act, "keep": bool(obj.get("keep")), "archived": obj.get("archived") or None, "can": chk["ok"], "why": chk["why"],
                             "gain": chk["bytes"], "due": bool(days and chk["ok"] and age_days >= days)})
        rows.sort(key=lambda r: -r["bytes"])
        tot = {g: sum(r["sizes"].get(g, 0) for r in rows) for g in GROUPS}
        return {"rows": rows[:200], "projects": len(rows), "bytes": sum(tot.values()), "sizes": tot, "days": days, "also_finalcut": also_finalcut,
                "due": sum(r["gain"] for r in rows if r["due"]), "due_n": sum(1 for r in rows if r["due"]), "at": now}

    # -- removing
    def _remove(self, paths: list[Path], roots: list[Path]) -> tuple[int, int]:
        freed = n = 0
        for p in paths:
            try:
                if p.is_symlink() or not p.is_file():
                    continue
                size = p.stat().st_size
                p.unlink()
                freed += size
                n += 1
            except OSError as e:
                log.warning(f"storage: couldn't remove {p}: {e}")
        for root in roots:                                              # folders that are empty now (never the root itself)
            if not root.is_dir() or root.is_symlink():
                continue
            for dp, dns, fns in os.walk(root, topdown=False, followlinks=False):
                if Path(dp) == root:
                    continue
                try:
                    if not os.path.islink(dp) and not os.listdir(dp):
                        os.rmdir(dp)
                except OSError:
                    pass
        return freed, n

    def archive(self, kind: str, iid: str, also_finalcut: bool | None = None) -> dict:
        """Remove the regenerable files of one project/job. Raises ValueError (plainly) when it may not be done."""
        if also_finalcut is None:
            also_finalcut = self.options()[1]
        chk = self.can_archive(kind, iid, also_finalcut)
        if not chk["ok"]:
            raise ValueError(chk["why"][:1].upper() + chk["why"][1:] if chk["why"] else "it can't be removed")
        obj = self.projects.get(iid) if kind == "long" else self.store.get(iid)
        cats = self.files_of(kind, obj)
        groups = ["videos", "working"] + (["finalcut"] if also_finalcut else [])
        paths = [p for g in groups for p in cats[g]]
        if kind == "long":
            home = Path(obj.get("home") or obj.get("folder") or "")
            roots = [home / "Output", home / "Final Cut", home / ".aivideo", self.store.dir(obj["job"])]
        else:
            home = Path(obj.get("app_home") or "")
            roots = [self.store.dir(iid)] + ([home / "Clips"] if obj.get("app_home") else [])
            if obj.get("app_home") and home.is_dir():
                roots += [d for d in home.iterdir() if d.is_dir() and not d.is_symlink() and d.name.startswith("Clips (before")]
        freed, n = self._remove(paths, roots)
        mark = {"at": self.now(), "freed": int(freed + (obj.get("archived") or {}).get("freed", 0)), "what": groups, "files": n}
        self._set(kind, iid, archived=mark)
        log.info(f"storage: {kind} {iid}: removed {n} files, {freed / 1e6:.0f} MB freed ({', '.join(groups)})")
        return {"ok": True, "freed": int(freed), "files": n, "archived": mark}

    def _set(self, kind: str, iid: str, **fields) -> None:
        """Write fields into a project's / job's record WITHOUT touching its "updated" time (the lists are sorted by it: removing a video or
        pinning must not make an old project look new)."""
        if kind == "long":
            with self.projects.lock:
                proj = self.projects.get(iid)
                proj.update(fields)
                self.projects._save(proj)
        else:
            with self.store._lock:
                job = self.store.get(iid)
                job.update(fields)
                self.store._write(job)

    def pin(self, kind: str, iid: str, keep: bool) -> dict:
        self._set(kind, iid, keep=bool(keep))
        return {"ok": True, "keep": bool(keep)}

    def due(self, days: int | None = None, also_finalcut: bool | None = None) -> list[dict]:
        """The rows that are old enough (and allowed) for the chosen number of days."""
        days = self.options()[0] if days is None else days
        if not days:
            return []
        return [r for r in self.usage(days, also_finalcut)["rows"] if r["due"]]

    def auto_pass(self, busy=lambda: "", days: int | None = None, also_finalcut: bool | None = None) -> dict:
        """The quiet background pass: with a number of days chosen (the default is never), remove the videos of what is older than that,
        one at a time, and stop at once if the app starts making something."""
        days = self.options()[0] if days is None else days
        if not days:
            return {"removed": 0, "freed": 0, "skipped": "off"}
        if busy():
            return {"removed": 0, "freed": 0, "skipped": "busy"}
        removed = freed = 0
        for r in self.due(days, also_finalcut):
            if busy():
                break
            try:
                res = self.archive(r["kind"], r["id"], also_finalcut)
                removed += 1
                freed += res["freed"]
            except ValueError as e:
                log.info(f"storage: left {r['kind']} {r['id']}: {e}")
        return {"removed": removed, "freed": freed}

    # -- making it again
    def missing_footage(self, kind: str, iid: str) -> str:
        """"" when the originals a re-render reads are there, else one plain sentence saying what is missing."""
        try:
            obj = self.projects.get(iid) if kind == "long" else self.store.get(iid)
        except (KeyError, OSError, ValueError):
            return "it isn't there any more"
        if kind == "long":
            folder = Path(obj.get("folder") or "")
            if not folder.is_dir():
                return f"The footage folder “{folder.name or folder}” isn't there. Plug in its drive (or put the folder back), then press Re-render."
            files = _json(Path(obj.get("home") or obj.get("folder") or "") / ".aivideo" / "files.json") or []
            gone = [f for f in files if isinstance(f, dict) and f.get("path") and not f.get("library") and not f.get("stock") and not f.get("photo")
                    and not Path(f["path"]).exists()]
            if gone:
                where = Path(gone[0]["path"]).parent.name or str(Path(gone[0]["path"]).parent)
                return (f"{len(gone)} of the original clips aren't in “{where}” any more (for example {Path(gone[0]['path']).name}). "
                        "Plug in the drive or put them back, then press Re-render.")
            return ""
        src = obj.get("source") or {}
        if src.get("url"):
            return ""                                                   # (downloaded again: the page says so)
        path = Path(src.get("path") or "")
        if not path.is_file():
            return f"The original video “{path.name or path}” isn't there. Plug in its drive (or put the file back), then press Re-render."
        return ""

    def after_run(self, pid: str, ok: bool) -> None:
        """A long video rendered again: it has its videos back, so it is no longer marked as removed."""
        if not ok:
            return
        try:
            if self.projects.get(pid).get("archived"):
                self.projects.update(pid, archived=None)
        except (KeyError, OSError, ValueError):
            pass


def after_run(runner, pid: str, kind: str, ok: bool) -> None:
    """LocalRunner hook (next to transcript_edit.after_run)."""
    if not ok:
        return
    try:
        proj = runner.projects.get(pid)
        if proj.get("archived"):
            runner.projects.update(pid, archived=None)
    except (KeyError, OSError, ValueError):
        pass


def human(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1000
    return ""
