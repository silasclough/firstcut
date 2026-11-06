"""Nightly auto-clipper (runs on the M2 after its one cartoon): find a licensed long video, cut ~10 clips,
send them to the posting server, repeat until the night's stop time.

Each finished job is copied to the server's inbox/clipper/<job>/ (clips + thumbnails + manifest.json,
manifest last so a half-copied folder is never ingested). The server shows them in Clipper with
Reject buttons and posts whatever isn't rejected (see relay.py).
"""

from __future__ import annotations

import json
import random
import subprocess
import time
from datetime import datetime

from ..core.log import get_logger
from ..core.state import JsonState
from .discover import Finder, credit_line
from .engine import Engine
from .store import Store

log = get_logger("clipper.auto")


def _target(cfg) -> tuple[str, str] | None:
    """(user@host, remote inbox dir) of the posting server."""
    host = cfg.get("clipper.auto.push_host")
    if not host:
        t = cfg.get("handoff.target") or ""
        host = t.split(":", 1)[0] if ":" in t else None
    return (host, cfg.get("clipper.auto.push_dir", "clipfarm/inbox/clipper")) if host else None


def push(cfg, store: Store, job: dict, src: dict, approve: list[int] | None = None,
         remote_id: str | None = None, extra: dict | None = None) -> bool:
    tgt = _target(cfg)
    if not tgt:
        log.warning("no push target (clipper.auto.push_host / handoff.target); clips stay on this machine")
        return False
    host, inbox = tgt
    d = store.dir(job["id"])
    files = [d / c[k] for c in job["clips"] for k in ("file", "thumb", "srt", "preview") if c.get(k) and (d / c[k]).exists()]
    if not files:
        return False
    rid = remote_id or job["id"]
    manifest = d / "manifest.json"
    manifest.write_text(json.dumps({"job": {**job, "id": rid}, "source": src, "credit": credit_line(src),
                                    "made_on": "M2", "made_at": datetime.now().isoformat(timespec="seconds"),
                                    **({"approve": approve} if approve else {}), **(extra or {})}))
    remote = f"{inbox}/{rid}"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host]
    try:
        subprocess.run(ssh + [f"mkdir -p {remote}"], check=True, capture_output=True, timeout=60)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", *map(str, files), f"{host}:{remote}/"],
                       check=True, capture_output=True, timeout=3600)
        subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(manifest), f"{host}:{remote}/manifest.json"],
                       check=True, capture_output=True, timeout=300)
    except Exception as e:  # noqa: BLE001 - retried by the next run (job stays unpushed)
        log.error(f"pushing {rid} to {host} failed: {e}")
        return False
    log.info(f"pushed {len(job['clips'])} clips of {rid} to {host}")
    return True


def push_replace(cfg, store: Store, job_id: str, index: int, rid: str) -> bool:
    """Send a re-made version of a clip the server already holds (it swaps the file if the clip hasn't been scheduled yet)."""
    job = store.get(job_id)
    c = next(c for c in job["clips"] if c["index"] == index)
    sub = {**job, "clips": [c]}
    return push(cfg, store, sub, job["source"], remote_id=f"{rid}~r{int(time.time())}", extra={"replace": [index], "replace_job": rid})


def push_own(cfg, store: Store, job_id: str, index: int) -> bool:
    """One of YOUR uploads' clips -> the posting server, approved and at the FRONT of the line (no review, no credit line)."""
    job = store.get(job_id)
    c = next(c for c in job["clips"] if c["index"] == index)
    rid = f"{job_id}+own{index}"
    sub = {**job, "clips": [c]}
    src = {**(job.get("source") or {}), "own": True, "title": job.get("title")}
    ok = push(cfg, store, sub, src, approve=[index], remote_id=rid, extra={"priority": True, "own": True, "credit": ""})
    return ok


def _mark_sent(store: Store, job_id: str, indices: list[int], rid: str) -> None:
    with store._lock:
        job = store.get(job_id)
        for c in job["clips"]:
            if c["index"] in indices:
                c["feed"] = {**(c.get("feed") or {}), "sent_as": rid}
        store._write(job)


def push_verdicts(cfg, store: Store, job_id: str, likes: list[int], dislikes: list[int]) -> bool:
    """Tell the posting server what you liked / deleted in the feed on this machine.
    Liked clips it has never seen travel with their files (a small job it approves on arrival); for clips it already
    holds, a tiny verdicts file is enough. Deleted clips the server never had need nothing."""
    tgt = _target(cfg)
    if not tgt:
        return False
    host, inbox = tgt
    job = store.get(job_id)
    d = store.dir(job_id)
    by_idx = {c["index"]: c for c in job["clips"]}

    def sent_as(i):
        return ((by_idx.get(i) or {}).get("feed") or {}).get("sent_as") or (job_id if job.get("pushed") else None)

    new = [i for i in likes if not sent_as(i) and i in by_idx and (d / by_idx[i]["file"]).exists()]
    if new:
        first = not any(sent_as(i) for i in by_idx)
        rid = job_id if first else f"{job_id}+{int(time.time())}"
        sub = {**job, "clips": [by_idx[i] for i in new]}
        if not push(cfg, store, sub, job["source"], approve=new, remote_id=rid):
            return False
        _mark_sent(store, job_id, new, rid)
        job = store.get(job_id)
        by_idx = {c["index"]: c for c in job["clips"]}
    per: dict[str, dict[str, list[int]]] = {}
    for i in likes:
        if i not in new and sent_as(i):
            per.setdefault(sent_as(i), {"approve": [], "discard": []})["approve"].append(i)
    for i in dislikes:
        if sent_as(i):
            per.setdefault(sent_as(i), {"approve": [], "discard": []})["discard"].append(i)
    if not per:
        return True
    vdir = f"{inbox.rsplit('/', 1)[0]}/verdicts"
    try:
        subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, f"mkdir -p {vdir}"],
                       check=True, capture_output=True, timeout=60)
        for rid, v in per.items():
            local = d / "verdicts.json"
            local.write_text(json.dumps({"job": rid, **v}))
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", str(local), f"{host}:{vdir}/{rid}__{int(time.time())}.json"],
                           check=True, capture_output=True, timeout=120)
    except Exception as e:  # noqa: BLE001 - retried
        log.error(f"sending verdicts for {job_id} to {host} failed: {e}")
        return False
    return True


class Pace:
    """Downloads are spread over the whole day - one every 80-140 minutes, a daily limit, and a long pause after YouTube
    refuses a download - so the machine never looks like a scraper. Shared by the night run and the daytime drip."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.path = cfg.path("state") / "auto_pace.json"
        a = cfg.get("clipper.auto", {}) or {}
        gap = a.get("download_gap_min", [80, 140])
        self.lo, self.hi = float(gap[0]), float(gap[-1])
        self.per_day = int(a.get("max_videos_per_day", a.get("max_videos_per_night", 6)))

    def _d(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

    def ok(self) -> tuple[bool, str]:
        d, now = self._d(), time.time()
        if now < d.get("blocked_until", 0):
            return False, f"paused until {datetime.fromtimestamp(d['blocked_until']):%H:%M} after YouTube refused a download"
        recent = [t for t in d.get("downloads", []) if now - t < 86400]
        if len(recent) >= self.per_day:
            return False, f"{len(recent)} videos in the last 24 h (limit {self.per_day})"
        wait = d.get("next_gap_s", self.lo * 60) - (now - max(recent)) if recent else 0
        if wait > 0:
            return False, f"next download in {wait / 60:.0f} min"
        return True, ""

    def record(self) -> None:
        d, now = self._d(), time.time()
        d["downloads"] = [t for t in d.get("downloads", []) if now - t < 86400] + [now]
        d["next_gap_s"] = random.uniform(self.lo, self.hi) * 60
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d))

    def block(self, hours: float) -> None:
        d = self._d()
        d["blocked_until"] = time.time() + hours * 3600
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d))


def run_auto(ctx, deadline: datetime | None = None, should_stop=lambda: False, max_jobs: int | None = None,
             paced: bool | None = None) -> list[dict]:
    cfg = ctx.cfg
    acfg = cfg.get("clipper.auto", {}) or {}
    store = Store(cfg.root / cfg.get("clipper.data_dir", "data/clipper"))
    engine = Engine(cfg, store)
    finder = Finder(cfg, cfg.path("state"))
    hist = JsonState(cfg.path("state") / "clipper_auto.json")
    results: list[dict] = []
    from . import taste
    if not cfg.get("clipper.feed.local", False):
        taste.pull(cfg)                              # what you've liked / deleted so far steers tonight's choices

    local_feed = bool(cfg.get("clipper.feed.local", False))     # you swipe on this machine; only liked clips travel
    for j in store.list():                       # finished earlier but never delivered (server was down)
        if local_feed:
            break
        if j["status"] == "done" and j["source"].get("auto") and not j.get("pushed") and j["clips"]:
            if push(cfg, store, j, j["source"]):
                store.update(j["id"], pushed=True)

    per_night = int(acfg.get("max_videos_per_night", 6))
    paced = (max_jobs is None) if paced is None else paced             # a hand-started `--max N` run isn't paced
    pace = Pace(cfg)
    max_jobs = per_night if max_jobs is None else max_jobs
    max_clips = int(acfg.get("max_clips_per_night", 30))
    used_lists: set[str] = set()
    while not should_stop() and sum(1 for r in results if not r.get("twitch")) < max_jobs and sum(r["clips"] for r in results) < max_clips:
        walls = sorted(hist.get("walls", []) or [2400])
        est = walls[len(walls) // 2]
        if deadline and time.time() + est > deadline.timestamp():
            log.info(f"auto-clipper: next video needs ~{est / 60:.0f} min, past the stop time")
            break
        src = finder.twitch_source()                      # streamer clips first: small files from Twitch, no YouTube download pacing needed
        is_tw = bool(src)
        if not src:
            if paced:
                ok, why = pace.ok()
                if not ok:
                    log.info(f"auto-clipper: waiting - {why}")
                    break
            src = finder.next_source(avoid=used_lists, twitch=False)
        if src and src.get("list_url"):
            used_lists.add(src["list_url"])
            if len(used_lists) >= len([c for c in (acfg.get("channels") or []) if c.get("url")]):
                used_lists.clear()                    # every creator has had a turn tonight: start another round
        if not src:
            log.info("auto-clipper: no new licensed source found")
            break
        finder.mark(src["id"], "used")
        if paced and not is_tw:
            pace.record()
        opts = {"clips": int(acfg.get("clips_per_video", 10)), "min_len": float(acfg.get("min_len", 20)),
                "max_len": float(acfg.get("max_len", 55)), "layout": "auto", "caption_style": "mix",
                "pace": "tight", "hook_title": True, "language": "en", "source_type": "auto"}
        if is_tw:                                                  # the clip IS the moment: keep it whole, light touch
            opts.update(clips=1, min_len=6.0, max_len=65.0, pace="natural")
        job = store.create({**src, "auto": True}, opts)
        log.info(f"auto-clipper: {src['kind']} source '{src['title']}' ({src['channel']}) -> {job['id']}")
        t0 = time.time()
        job = engine.process(job["id"])
        wall = time.time() - t0
        if job["status"] == "failed" and "DownloadError" in str(job.get("error")) and not is_tw:
            finder.state.data["seen"].pop(src["id"], None)      # YouTube said no this time (403 / rate limit): try it another night
            finder.state.save()
            pace.block(float(acfg.get("block_hours_after_refusal", 6)))
            log.warning("auto-clipper: YouTube refused a download - no more downloads for a few hours")
            results.append({"job": job["id"], "status": job["status"], "clips": 0, "pushed": False,
                            "source": src["url"], "minutes": round(wall / 60, 1)})
            break
        hist.set("walls", (hist.get("walls", []) + [round(wall)])[-20:])
        pushed = job["status"] == "done" and not local_feed and push(cfg, store, job, src)
        if pushed:
            store.update(job["id"], pushed=True)
        results.append({"job": job["id"], "status": job["status"], "clips": len(job["clips"]), "pushed": pushed,
                        "source": src["url"], "minutes": round(wall / 60, 1), **({"twitch": True} if is_tw else {})})
    return results
