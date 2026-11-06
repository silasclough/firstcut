"""Finished clips -> a titled folder per job on the archive drive (clipper.library_dir).

    <library_dir>/2026-10-09 Russell Peters - Blocks Podcast (Neal Brennan)/
        01 - Why Women Like Me.mp4      the clip
        01 - Why Women Like Me.txt      post caption (+ hashtags + credits)
        01 - Why Women Like Me.srt      captions
        info.txt                        source, license / credit, when, settings
        Rejected/                       auto-clipper clips you rejected

Clipper keeps working in its own folder (data/clipper/<job id>) - the archive copy is what you keep.
If the drive isn't mounted the export waits (job["library_pending"]) and is retried.
"""

from __future__ import annotations

import re
import shutil
import time
from datetime import datetime
from pathlib import Path

from ..core.log import get_logger

log = get_logger("clipper.library")
BAD = re.compile(r'[\\/:*?"<>|\x00-\x1f]+')


def library_dir(cfg) -> Path | None:
    d = cfg.get("clipper.library_dir")
    return Path(d) if d else None


def drive_ready(path: Path | str) -> bool:
    """Is it safe to write under `path`? For anything on an external volume (/Volumes/<name>/...) the volume itself must be
    MOUNTED. A path that merely 'exists' isn't enough: if the drive is unplugged, writing there creates a real folder on
    the internal disk and macOS then mounts the returning drive as '<name> 1'. Never create /Volumes/<name> ourselves."""
    import os
    p = Path(path)
    parts = p.parts
    if len(parts) >= 3 and parts[1] == "Volumes":
        mount = Path(parts[0], parts[1], parts[2])
        return os.path.ismount(str(mount)) and p.exists()
    return p.exists()


def safe(name: str, limit: int = 90) -> str:
    name = BAD.sub(" ", name or "").replace("..", ".")
    name = re.sub(r"\s+", " ", name).strip(" .-")
    return (name[:limit].rstrip(" .-") or "Untitled")


def folder_title(job: dict) -> str:
    src = job.get("source") or {}
    when = datetime.fromtimestamp(job.get("created") or time.time()).strftime("%Y-%m-%d")
    title = job.get("title") or src.get("title") or src.get("filename") or job["id"]
    if src.get("auto"):
        title = src.get("title") or title
        if src.get("channel"):
            title = f"{title} ({src['channel']})"
    return f"{when} {safe(title)}"


def _clip_stem(c: dict) -> str:
    return f"{int(c['index']):02d} - {safe(c.get('title') or 'clip', 70)}"


def _target(cfg, job: dict) -> Path | None:
    root = library_dir(cfg)
    if not root:
        return None
    if job.get("library"):
        return Path(job["library"])
    base = root / folder_title(job)
    d, k = base, 2
    while d.exists():
        d = base.with_name(f"{base.name} ({k})")
        k += 1
    return d


def feed_job(cfg, job: dict) -> bool:
    """Auto-clipper jobs in swipe-feed mode are archived clip by clip, only once you like them."""
    return bool(cfg.get("clipper.feed.enabled", True)) and bool((job.get("source") or {}).get("auto"))


def export(cfg, store, job_id: str, only: set[int] | None = None) -> Path | None:
    """Copy the clips of a finished job into its archive folder (all of them, or just `only`).
    Returns the folder (None = not now)."""
    root = library_dir(cfg)
    if not root:
        return None
    job = store.get(job_id)
    if not job.get("clips") or (job.get("options") or {}).get("project"):
        return None                                       # AI-video projects have their own folder (clipper/aivideo.py)
    if only is None and feed_job(cfg, job):
        only = {c["index"] for c in job["clips"] if (c.get("review") or {}).get("state") == "approved"}
        if not only:
            return None                                   # nothing liked yet: nothing to keep
    if not drive_ready(root):
        log.warning(f"archive {root} not mounted; {job_id} will be copied later")
        store.update(job_id, library_pending=True)
        return None
    dest = _target(cfg, job)
    dest.mkdir(parents=True, exist_ok=True)
    src_dir = store.dir(job_id)
    for c in job["clips"]:
        if c.get("deleted") or (only is not None and c["index"] not in only):
            continue
        _copy_clip(src_dir, dest, c, rejected=((c.get("review") or {}).get("state") == "rejected"))
    src = job.get("source") or {}
    lines = [f"Title: {job.get('title')}", f"Made: {datetime.fromtimestamp(job.get('created') or time.time()):%Y-%m-%d %H:%M}",
             f"Clipper job: {job_id}"]
    if src.get("url"):
        lines.append(f"Source: {src['url']}")
    if src.get("files"):
        lines.append("Uploaded files: " + ", ".join(src["files"]))
    if src.get("credit") or src.get("license"):
        lines.append(f"License / credit: {src.get('credit') or src.get('license')}")
    opts = job.get("options") or {}
    lines.append("Settings: " + ", ".join(f"{k}={v}" for k, v in opts.items()))
    (dest / "info.txt").write_text("\n".join(lines) + "\n")
    store.update(job_id, library=str(dest), library_pending=False)
    log.info(f"{job_id}: {len(job['clips'])} clips archived to {dest}")
    return dest


def _copy_clip(src_dir: Path, dest: Path, c: dict, rejected: bool = False) -> None:
    stem = _clip_stem(c)
    out = dest / "Rejected" if rejected else dest
    out.mkdir(parents=True, exist_ok=True)
    for old in list(dest.glob(f"{int(c['index']):02d} - *")) + list((dest / "Rejected").glob(f"{int(c['index']):02d} - *")):
        old.unlink(missing_ok=True)                     # a re-render / rename / reject replaces the old copy
    if c.get("file") and (src_dir / c["file"]).exists():
        shutil.copy2(src_dir / c["file"], out / f"{stem}.mp4")
    if c.get("srt") and (src_dir / c["srt"]).exists():
        shutil.copy2(src_dir / c["srt"], out / f"{stem}.srt")
    (out / f"{stem}.txt").write_text((c.get("caption") or "").strip() + "\n")


def update_clip(cfg, store, job_id: str, index: int) -> None:
    """After a restyle / reject: refresh that one clip in the archive folder."""
    job = store.get(job_id)
    if not job.get("library"):
        export(cfg, store, job_id, only={index} if feed_job(cfg, job) else None)
        return
    dest = Path(job["library"])
    if not drive_ready(dest.parent):
        store.update(job_id, library_pending=True)
        return
    c = next((c for c in job["clips"] if c["index"] == index), None)
    if c and not c.get("deleted"):
        _copy_clip(store.dir(job_id), dest, c, rejected=((c.get("review") or {}).get("state") == "rejected"))


def export_pending(cfg, store, limit: int | None = None) -> list[str]:
    """Copy archives that had to wait (drive was missing). Trickles: at most `clipper.library_flush_per_cycle` jobs per
    call (default 1, called every few minutes), so a drive that just came back isn't hit with a huge burst of writes."""
    limit = int(cfg.get("clipper.library_flush_per_cycle", 1)) if limit is None else limit
    root = library_dir(cfg)
    if not root or not drive_ready(root):
        return []                                            # nothing to do (and no churn) until the drive is back
    done = []
    for j in store.list():
        if len(done) >= limit:
            break
        if j.get("status") == "done" and j.get("clips") and not (j.get("options") or {}).get("project") and (
                j.get("library_pending") or (not j.get("library") and not feed_job(cfg, j))):
            if export(cfg, store, j["id"]):
                done.append(j["id"])
    return done
