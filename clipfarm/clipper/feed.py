"""The swipe feed: scroll through the night's clips like TikTok; what you like gets posted, the rest is deleted.

    like      -> approved (posts at the next random slot) + archived to the drive
    dislike   -> the clip's video files are deleted (thumbnail + metadata stay so the page keeps its history)
    scrolled past without rating -> stays in the feed (behind the unseen ones) until you rate it

A rating takes effect after a short undo window (clipper.feed.grace_s). Every rating trains taste.py, which
orders the feed and steers the M2's nightly choices.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from ..core.log import get_logger
from ..core.queue import JobQueue
from ..core.state import atomic_write_json, read_json
from . import clipgate, preview, taste as T

log = get_logger("clipper.feed")
FORWARD_RETRY_S = 300
from .engine import FRAMING_V
STYLE_V = 2             # 2 = looked at for graphics with the fixed cards (1 had stat cards with clipped text over the captions: those clips are redone)
COMMENTARY_V = 3        # 3 = only clips with little talking get a voice (2 = lines never overlap each other and the clip's own sound is ducked almost to silence under them
PREVIEW_V = 2          # 1 = 540 px / ~1 Mbit/s, 2 = 432 px / ~0.5 Mbit/s (loads about twice as fast on a phone)


def _fresh(c: dict) -> bool:
    """Cut with the current recipe (full word ends + the wide framing)."""
    return bool(c.get("word_end_pad")) and int(c.get("framing_v", 1)) >= FRAMING_V


def enabled(cfg) -> bool:
    return bool(cfg.get("clipper.feed.enabled", True))


def _podcast(job: dict) -> bool:
    from .commentary import is_podcast
    return is_podcast(job.get("source"))


class Feed:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        self.taste = T.Taste(cfg.path("state"))
        self.queue = JobQueue(cfg.path("queue"))
        self.lock = threading.RLock()
        self.pending_path = cfg.path("state") / "feed_pending.json"
        self.pending: dict[str, dict] = read_json(self.pending_path) or {}
        self._qcache: dict[str, tuple[int, str]] = {}
        self.grace = float(cfg.get("clipper.feed.grace_s", 20))
        self._prev_failed: set[str] = set()
        self._last_prev = 0.0
        # local mode (the M2): the clips live here, so you swipe here. Verdicts are forwarded to the posting
        # server when it is reachable - the feed itself never needs the server.
        self.local = bool(cfg.get("clipper.feed.local", False))
        self.fwd_path = cfg.path("state") / "feed_forward.json"
        self.forward_q: dict[str, str] = read_json(self.fwd_path) or {}
        self._fwd_next = 0.0
        self.min_score = float(cfg.get("clipper.auto.min_score", 6))

    @property
    def on(self) -> bool:
        return enabled(self.cfg)

    # ---------------------------------------------------------------- helpers
    def queue_info(self, qid: str | None) -> dict:
        """{state, approve_at, slot} of a posting-queue job, cached by file mtime (no Context needed)."""
        if not qid:
            return {"state": ""}
        p = self.queue.dir / f"{qid}.json"
        try:
            mt = p.stat().st_mtime_ns
        except OSError:
            return {"state": "missing"}
        hit = self._qcache.get(qid)
        if hit and hit[0] == mt:
            return hit[1]
        d = read_json(p) or {}
        info = {"state": d.get("state", ""), "approve_at": (d.get("data") or {}).get("auto_approve_at"),
                "slot": (d.get("publish") or {}).get("slot")}
        self._qcache[qid] = (mt, info)
        return info

    def queue_state(self, qid: str | None) -> str:
        return self.queue_info(qid)["state"]

    def _save_pending(self) -> None:
        atomic_write_json(self.pending_path, self.pending)

    @staticmethod
    def _key(job_id: str, index: int) -> str:
        return f"{job_id}:{int(index)}"

    def _clip(self, job_id: str, index: int) -> tuple[dict, dict]:
        job = self.store.get(job_id)
        c = next((c for c in job["clips"] if c["index"] == int(index)), None)
        if c is None:
            raise KeyError(f"{job_id}#{index}")
        return job, c

    def _patch_clip(self, job_id: str, index: int, **feed_fields) -> None:
        with self.store._lock:
            job, c = self._clip(job_id, index)
            c = dict(c)
            c["feed"] = {**(c.get("feed") or {}), **feed_fields}
            self.store.set_clip(job_id, c)

    # ---------------------------------------------------------------- listing
    def _rank(self, c: dict, src: dict, job_id: str, today: str) -> tuple[float, float, list[str]]:
        m = self.taste.model()
        feats = T.features(c, src)
        pred = self.taste.predict(feats) if (m.n >= 5 or self.taste.audience_weight() > 0) else 0.5
        ai = min(1.0, float(c.get("score", 5)) / 10)
        trust = self.taste.trust()
        jitter = random.Random(hashlib.md5(f"{job_id}{c['index']}{today}".encode()).hexdigest()).random()
        rank = (1 - trust) * ai + trust * pred + 0.10 * jitter       # stable for the day, a little exploring
        return rank, pred, feats

    def items(self, limit: int = 30, exclude: set[str] | None = None) -> list[dict]:
        exclude = exclude or set()
        today = datetime.now().strftime("%Y-%m-%d")
        out = []
        avoid = [w.lower() for w in (self.cfg.get("clipper.auto.avoid_words") or [])]
        avoid_ch = {c.lower() for c in (self.cfg.get("clipper.auto.avoid_channels") or [])}
        for j in self.store.list():
            src = j.get("source") or {}
            if not src.get("auto") or j.get("status") != "done":
                continue
            t = f" {(src.get('title') or j.get('title') or '').lower()} "
            if str(src.get("channel") or "").lower() in avoid_ch or any((f" {w} " in t or (len(w) > 4 and w in t)) for w in avoid):
                continue                                                   # gaming: kept on disk but never shown or posted
            d = self.store.dir(j["id"])
            for c in j.get("clips", []):
                r = c.get("review") or {}
                key = self._key(j["id"], c["index"])
                if c.get("deleted") or key in self.pending or key in exclude or (c.get("feed") or {}).get("verdict"):
                    continue
                if not src.get("own") and clipgate.problems(c, src):
                    continue                                               # what the posting gate would refuse is not shown to be swiped (a like could never post it)
                if self.local:
                    fd = c.get("feed") or {}
                    auto_q = bool(fd.get("auto")) and r.get("state") == "liked"        # queued by the model: you can veto it
                    if (r.get("state") not in (None, "", "review") and not auto_q) or float(c.get("score", 0)) < self.min_score:
                        continue
                else:
                    if r.get("state") != "review":
                        continue
                    if r.get("queue_job") and self.queue_state(r["queue_job"]) != "review":
                        continue
                vid = c.get("preview") if c.get("preview") and (d / c["preview"]).exists() else c["file"]
                if not (d / vid).exists():
                    continue
                rank, pred, feats = self._rank(c, src, j["id"], today)
                if src.get("kind") == "original":
                    rank += 0.2                                   # our own original videos are the majority of what gets posted
                streamer = src.get("kind") == "twitch_clip"
                if streamer:
                    rank += 1.5                                   # streamer moments first (clipper.twitch): fans already rated them
                hot = bool(src.get("trend") and str(src.get("hot_until") or "") > datetime.now(timezone.utc).isoformat(timespec="seconds"))
                if hot:
                    rank += 2.0 + 0.2 * float(src.get("trend_heat") or 0)    # about a search that is hot right now: first in your feed, before it goes cold
                info = self._info_value(pred, feats)
                seen = int((c.get("feed") or {}).get("seen", 0))
                m = self.taste.model()
                v = c.get("rendered_at") or 0
                out.append({"key": key, "job": j["id"], "index": c["index"], "title": c.get("title", ""),
                            "caption": c.get("caption", ""), "score": c.get("score"), "duration": c.get("duration"),
                            "channel": src.get("channel"), "source_title": src.get("title"), "source_url": src.get("url"),
                            "topic": src.get("topic"), "license": src.get("license"), "trend": src.get("trend") if hot else None,
                            "video": f"/media/{j['id']}/{vid}?v={v}", "thumb": f"/media/{j['id']}/{c['thumb']}?v={v}",
                            "is_preview": vid != c["file"], "seen": seen, "match": round(pred * 100) if m.n >= 10 else None,
                            "why": m.explain(feats) if m.n >= 10 else [], "_rank": rank, "_info": info, "_pred": pred,
                            "auto": (c.get("feed") or {}).get("auto_pred") if (c.get("feed") or {}).get("auto") else None,
                            "fresh": _fresh(c), "prev_v": int(c.get("preview_v", 1)), "streamer": streamer})
        for x in out:
            if x["fresh"]:
                x["_rank"] += 0.15                     # nudge: clips with the wide framing / full word ends a little earlier
        out = self._schedule(out)
        out = [x for x in out if x.get("streamer")] + [x for x in out if not x.get("streamer")]       # streamer clips at the front of the feed; the rest follow when there are not enough
        out = self._diversify(out)
        for x in out:
            for k in ("_rank", "_info", "_pred"):
                x.pop(k, None)
        return out[:limit] if limit else out

    def _info_value(self, pred: float, feats: list[str]) -> float:
        """How much would rating this teach the model? Clips it is unsure about (near 50/50) and clips with creators /
        topics / words it has barely seen teach the most. A clip it is 97% sure you'll delete teaches almost nothing."""
        m = self.taste.model()
        uncertainty = 4 * pred * (1 - pred)
        key = [f for f in feats if f.split(":", 1)[0] in ("channel", "topic", "tx", "words")]
        novelty = (sum(1 for f in key if m.counts.get(f, (0, 0))[1] < 3) / len(key)) if key else 0.5
        return 0.6 * uncertainty + 0.4 * novelty

    def _schedule(self, out: list[dict]) -> list[dict]:
        """Order for learning as well as liking. Every 10 clips: 4 best guesses (what you'll probably like - keeps it worth
        scrolling and posts good stuff), 4 that teach the model the most, 1 random clip (an honest check of its blind
        spots - even a 'sure no' is sometimes wrong), 1 more best guess. While the model is young, more of the teaching ones."""
        auto = [x for x in out if x["auto"] is not None]
        rest = [x for x in out if x["auto"] is None]
        n = self.taste.model().n
        pattern = "ETETRETETE" if n >= 60 else "TETETETETR" if n >= 20 else "TTETTETTER"
        exploit = sorted(rest, key=lambda x: -x["_rank"])
        teach = sorted(rest, key=lambda x: -x["_info"])
        rnd = sorted(rest, key=lambda x: hashlib.md5((x["key"] + datetime.now().strftime("%Y-%m-%d-%H")).encode()).hexdigest())
        used: set[str] = set()
        queues = {"E": iter(exploit), "T": iter(teach), "R": iter(rnd)}
        tag = {"E": "best guess", "T": "helps me learn", "R": "random check"}
        res: list[dict] = []
        while len(used) < len(rest):
            for ch in pattern:
                for x in queues[ch]:
                    if x["key"] not in used:
                        used.add(x["key"])
                        x["shown_for"] = tag[ch] if ch != "E" or x["_pred"] >= 0.3 else "best guess left"
                        res.append(x)
                        break
                if len(used) >= len(rest):
                    break
        return res + sorted(auto, key=lambda x: -x["_rank"])

    @staticmethod
    def _diversify(ordered: list[dict], window: int = 40) -> list[dict]:
        """Keep the order (best first) but never serve the same creator / same source video / same topic over and over:
        each pick is penalised for repeating what was shown in the last few, so the feed stays varied."""
        groups: list[list[dict]] = []
        for x in ordered:                                          # unseen / seen / auto stay in their own blocks
            key = (x["auto"] is not None,)
            if not groups or groups[-1][0]["_g"] != key:
                groups.append([])
            x["_g"] = key
            groups[-1].append(x)
        out: list[dict] = []
        for g in groups:
            rest = list(g)
            picked: list[dict] = []
            while rest:
                best, best_score = None, -1e9
                for k, cand in enumerate(rest[:window]):
                    pen = (0.30 * sum(1 for p in picked[-4:] if p["channel"] == cand["channel"]) +
                           0.20 * sum(1 for p in picked[-6:] if p["job"] == cand["job"]) +
                           0.10 * sum(1 for p in picked[-3:] if p["topic"] == cand["topic"]))
                    score = -0.02 * k - pen                              # keep the scheduled order; only skip past repeats
                    if score > best_score:
                        best, best_score = k, score
                picked.append(rest.pop(best))
            out.extend(picked)
        for x in out:
            x.pop("_g", None)
        return out

    def maintain(self) -> int:
        """(M2) give older clips the transcript the model now learns from, then recompute old ratings with it."""
        if not self.local or time.time() - getattr(self, "_maint", 0) < 30:
            return 0
        self._maint = time.time()
        n = 0
        for j in self.store.list():
            if j.get("status") != "done" or not (j.get("source") or {}).get("auto"):
                continue
            todo = [c for c in j.get("clips", []) if not c.get("transcript") and not c.get("deleted_transcript")]
            if not todo:
                continue
            tp = self.store.dir(j["id"]) / "transcript.json"
            try:
                words = json.loads(tp.read_text())["words"]
            except (OSError, ValueError, KeyError):
                with self.store._lock:
                    job = self.store.get(j["id"])
                    for c in job["clips"]:
                        c["deleted_transcript"] = True            # no transcript on disk: don't look again
                    self.store._write(job)
                continue
            with self.store._lock:
                job = self.store.get(j["id"])
                for c in job["clips"]:
                    if not c.get("transcript"):
                        a, b = c.get("source_start", 0), c.get("source_end", 0)
                        c["transcript"] = " ".join(w["text"] for w in words if w["start"] >= a - 0.05 and w["end"] <= b + 0.05)[:1800]
                        n += 1
                self.store._write(job)
            if n >= 60:
                break

        def recompute(key):
            jid, idx = key.rsplit(":", 1)
            try:
                job, c = self._clip(jid, int(idx))
            except (KeyError, ValueError):
                return None
            return T.features(c, job.get("source")) if c.get("transcript") else None
        return n + self.taste.relearn(recompute)

    def refresh_perf(self) -> int:
        """(M2) copy the server's posted-clip analytics and learn from them (every few hours)."""
        every = float(self.cfg.get("clipper.performance.pull_hours", 6)) * 3600
        if not self.local or time.time() - getattr(self, "_perf_at", 0) < every:
            return 0
        self._perf_at = time.time()
        from . import taste
        if not taste.pull_perf(self.cfg):
            return 0
        path = self.cfg.path("state") / "clip_perf.json"
        perf = read_json(path) or {}
        from . import performance
        if performance.fill_public(perf):                          # Metricool is days behind: use the public counts meanwhile
            atomic_write_json(path, perf)
        return self.taste.ingest_perf(perf, float(self.cfg.get("clipper.performance.min_age_h", 24)))

    def stats(self) -> dict:
        m = self.taste.model()
        s = m.summary()
        unrated = len(self.items(limit=0))
        ev = self.taste._events()
        day = time.time() - 86400
        liked = sum(1 for e in ev.values() if e.get("kind") == "rating" and e["label"] == 1 and e["t"] > day)
        disliked = sum(1 for e in ev.values() if e.get("kind") == "rating" and e["label"] == 0 and e["t"] > day)
        midnight = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        liked_today = sum(1 for e in ev.values() if e.get("kind") == "rating" and e["label"] == 1 and e["t"] >= midnight)
        goal = int(self.cfg.get("clipper.feed.daily_goal", 3))
        return {"unrated": unrated, "liked_24h": liked, "disliked_24h": disliked, "model": s,
                "liked_today": liked_today, "auto_today": self._auto_today(), "daily_goal": goal,
                "pending": len(self.pending), "trust": round(self.taste.trust(), 2), "autopost": self.autopost_status(),
                "performance": self._perf_summary(), "audience": self._audience_summary()}

    def _audience_summary(self) -> dict:
        a = self.taste.audience()
        return {"posts": a.n, "weight": round(self.taste.audience_weight(), 2),
                "works": [r for r in a.summary()["likes"][:6]], "flops": [r for r in a.summary()["dislikes"][:6]]}

    def _perf_summary(self) -> dict | None:
        perf = read_json(self.cfg.path("state") / "clip_perf.json")
        if not perf:
            return None
        from .performance import summary
        return summary(perf)

    # ---------------------------------------------------------------- ratings
    @staticmethod
    def watch_weight(verdict: str, frac: float | None) -> float:
        """How much a rating counts, from how much of the clip you watched (like other feeds do).
        A like after watching it through / rewatching is a strong yes; a quick like is a weaker one.
        Scrolling away within a moment is a firm no; watching it all and then scrolling is only a mild no."""
        if frac is None:
            return 1.0
        if verdict == "like":
            return 1.4 if frac >= 1.0 else 1.2 if frac >= 0.7 else 1.0 if frac >= 0.3 else 0.7
        return 1.5 if frac < 0.15 else 1.2 if frac < 0.4 else 0.9 if frac < 0.8 else 0.55

    def rate(self, job_id: str, index: int, verdict: str, immediate: bool = False,
             watched: float | None = None, duration: float | None = None) -> dict:
        if verdict not in ("like", "dislike"):
            raise ValueError("verdict must be like or dislike")
        job, c = self._clip(job_id, index)
        key = self._key(job_id, index)
        frac = (watched / duration) if (watched is not None and duration) else None
        self.taste.record(key, 1 if verdict == "like" else 0, T.features(c, job.get("source")),
                          weight=self.watch_weight(verdict, frac))
        if frac is not None:
            self._patch_clip(job_id, index, watch_frac=round(frac, 2), watch_s=round(float(watched), 1))
        with self.lock:
            self.pending[key] = {"verdict": verdict, "apply_at": time.time() + (0 if immediate else self.grace)}
            self._save_pending()
        return {"key": key, "verdict": verdict, "undo_s": 0 if immediate else self.grace}

    def undo(self, job_id: str, index: int) -> bool:
        key = self._key(job_id, index)
        with self.lock:
            if key not in self.pending:
                return False
            self.pending.pop(key)
            self._save_pending()
        self.taste.forget(key)
        return True

    def seen(self, job_id: str, index: int, watched_s: float, duration: float) -> None:
        job, c = self._clip(job_id, index)
        f = c.get("feed") or {}
        frac = watched_s / duration if duration else 0.0
        self._patch_clip(job_id, index, seen=int(f.get("seen", 0)) + 1, watch_s=round(watched_s, 1),
                         watch_frac=round(frac, 2), last_seen=time.time())
        if frac < 0.25 and watched_s >= 0.7 and not f.get("verdict"):
            self.taste.record(self._key(job_id, index), 0, T.features(c, job.get("source")), kind="skip",
                              weight=T.PASSIVE_W)

    # ---------------------------------------------------------------- applying ratings
    def sweep(self) -> int:
        """Apply ratings whose undo window has passed (called from the Clipper worker loop)."""
        due = [k for k, p in list(self.pending.items()) if p["apply_at"] <= time.time()]
        if not due:
            return 0
        from . import relay
        ctx = None
        if not self.local:
            from ..context import Context
            ctx = Context(self.cfg)
        n = 0
        for key in due:
            p = self.pending.get(key)
            job_id, idx = key.rsplit(":", 1)
            try:
                if self.local:
                    self._apply_local(job_id, int(idx), p["verdict"])
                elif p["verdict"] == "like":
                    relay.approve_clip(ctx, job_id, int(idx), rated=True)
                else:
                    relay.discard_clip(ctx, job_id, int(idx))
                self._patch_clip(job_id, int(idx), verdict=p["verdict"], rated_at=time.time())
                n += 1
            except KeyError:
                log.warning(f"feed: {key} no longer exists")
            except Exception as e:  # noqa: BLE001 - try again on the next sweep
                log.error(f"feed: applying {p['verdict']} to {key} failed: {e}")
                continue
            with self.lock:
                self.pending.pop(key, None)
        with self.lock:
            self._save_pending()
        return n

    # ---------------------------------------------------------------- local mode (the M2)
    def _apply_local(self, job_id: str, index: int, verdict: str) -> None:
        job, c = self._clip(job_id, index)
        d = self.store.dir(job_id)
        if verdict == "dislike":
            for name in (c.get("file"), c.get("preview"), c.get("srt")):
                if name:
                    (d / name).unlink(missing_ok=True)
            with self.store._lock:
                job, c = self._clip(job_id, index)
                self.store.set_clip(job_id, {**c, "deleted": True,
                                             "review": {**(c.get("review") or {}), "state": "disliked"}})
        else:
            with self.store._lock:
                job, c = self._clip(job_id, index)
                self.store.set_clip(job_id, {**c, "review": {**(c.get("review") or {}), "state": "liked"}})
        with self.lock:
            self.forward_q[self._key(job_id, index)] = verdict
            atomic_write_json(self.fwd_path, self.forward_q)

    def queue_own(self, job_id: str, index: int) -> str:
        """The 'Post next' button on your own upload: first in line for the next posting slot."""
        job, c = self._clip(job_id, index)
        if (job.get("source") or {}).get("auto"):
            return "this button is for your own uploads"
        if not c.get("file") or not (self.store.dir(job_id) / c["file"]).exists():
            return "the video file is missing"
        if self.local:
            self._set_post_next(job_id, index, "pending")
            self._fwd_next = 0.0
            return "sent" if self.send_own() else "queued - will send as soon as the posting Mac is reachable"
        from ..context import Context
        from . import relay
        relay.queue_own(Context(self.cfg), job_id, index)
        self._set_post_next(job_id, index, "queued")
        return "queued"

    def _set_post_next(self, job_id: str, index: int, state: str) -> None:
        with self.store._lock:
            job, c = self._clip(job_id, index)
            self.store.set_clip(job_id, {**c, "post_next": {"state": state, "at": time.time()}})

    def send_own(self) -> bool:
        """(M2) deliver every 'Post next' clip that hasn't reached the posting server yet. True if none are left waiting."""
        from . import auto
        left = False
        for j in self.store.list():
            for c in j.get("clips", []):
                if (c.get("post_next") or {}).get("state") == "pending":
                    try:
                        if auto.push_own(self.cfg, self.store, j["id"], c["index"]):
                            self._set_post_next(j["id"], c["index"], "sent")
                            continue
                    except Exception as e:  # noqa: BLE001
                        log.warning(f"sending {j['id']}#{c['index']} failed: {e}")
                    left = True
        return not left

    def next_backlog_clip(self) -> tuple[str, int, str] | None:
        """(local job, index, server job id) of the next approved-but-unposted clip that hasn't had its commentary added yet."""
        from . import taste
        path = self.cfg.path("state") / "clip_queue.json"
        if time.time() - getattr(self, "_queue_at", 0) > 3600:
            self._queue_at = time.time()
            taste.pull_queue_list(self.cfg)
        data = read_json(path) or {}
        for it in data.get("items", []):
            rid, idx = str(it.get("job") or ""), it.get("index")
            local = rid.split("+")[0].split("~")[0]
            if idx is None or it.get("kind") == "original" or f"{local}:{idx}" in self._prev_failed:
                continue
            try:
                job, c = self._clip(local, int(idx))
            except KeyError:
                continue
            done = c.get("commentary_done") == COMMENTARY_V and c.get("style_done") == STYLE_V
            if done or c.get("deleted") or not (self.store.dir(local) / c.get("file", "")).exists():
                continue
            return local, int(idx), rid
        return None

    def commentate_backlog_one(self, engine) -> bool:
        """(M2, daytime) bring ONE approved clip up to date - voiced commentary (never on podcasts) and, if what it says is informational, graphics - and swap it
        into the posting line. A clip that needs neither is just marked done (no re-render). True if it looked at one."""
        from . import auto, commentary as CM, editstyle
        nxt = self.next_backlog_clip()
        if not nxt:
            return False
        local, idx, rid = nxt
        key = f"{local}:{idx}"
        try:
            job, c = self._clip(local, idx)
            density = len(str(c.get("transcript") or "").split()) / max(1.0, float(c.get("duration") or 30))
            podcast = not CM.allowed(job.get("source"), density)             # a podcast, or a clip with a lot of talking: no voice-over
            need_comm = c.get("commentary_done") != COMMENTARY_V and not podcast
            strip_comm = bool(c.get("commentary")) and podcast               # it was given one before this rule: make it again without
            style = c.get("edit_style") or "standard"
            if c.get("style_done") != STYLE_V:
                style, why = editstyle.decide(engine.router_for_now()[0], job.get("source"), c.get("title", ""), c.get("transcript", ""))
                style = "standard" if style == "none" else style
                log.info(f"graphics for {key}: {style} ({why})")
            need_style = style != "standard" and (c.get("style_done") != STYLE_V)
            if style == "standard" and (c.get("edit_style") or "standard") != "standard" and c.get("style_done") != STYLE_V:
                need_style = True                                  # it had graphics from an older version and should have none now: re-render it plain
            if need_comm or need_style or strip_comm:
                engine.restyle(local, idx, c["layout"], c["caption_style"], c["pace"], bool(c.get("hook_title", True)),
                               commentary=True if need_comm else bool(c.get("commentary")) and not podcast, edit_style=style)
            with self.store._lock:
                job, c = self._clip(local, idx)
                self.store.set_clip(local, {**c, "commentary_done": COMMENTARY_V, "style_done": STYLE_V})
            if (need_comm or need_style or strip_comm) and not auto.push_replace(self.cfg, self.store, local, idx, rid):
                with self.store._lock:
                    job, c = self._clip(local, idx)
                    self.store.set_clip(local, {**c, "commentary_done": False, "style_done": False})
                self._prev_failed.add(key)                           # server unreachable: try again after a restart
        except Exception as e:  # noqa: BLE001
            log.warning(f"re-edit of {key} failed: {e}")
            self._prev_failed.add(key)
        return True

    def _needs_recut(self, job_id: str, index: int) -> bool:
        if not self.cfg.get("clipper.feed.recut_old", True) or f"{job_id}:{index}" in self._prev_failed:
            return False
        try:
            c = self._clip(job_id, index)[1]
        except KeyError:
            return False
        return not _fresh(c) and not (c.get("feed") or {}).get("sent_as") and not c.get("deleted")

    def forward(self) -> int:
        """Send likes / dislikes to the posting server (retried every few minutes while it is unreachable)."""
        if self.local and time.time() >= self._fwd_next:
            try:
                self.send_own()
            except Exception as e:  # noqa: BLE001
                log.warning(f"own uploads: {e}")
        if not self.local or not self.forward_q or time.time() < self._fwd_next:
            return 0
        from . import auto
        by_job: dict[str, dict[str, list[int]]] = {}
        for key, v in self.forward_q.items():
            jid, idx = key.rsplit(":", 1)
            if v == "like" and self._needs_recut(jid, int(idx)):
                continue                                             # wait until it has been re-cut with the word-end fix
            by_job.setdefault(jid, {"like": [], "dislike": []})[v].append(int(idx))
        sent = 0
        for jid, vs in by_job.items():
            try:
                ok = auto.push_verdicts(self.cfg, self.store, jid, vs["like"], vs["dislike"])
            except Exception as e:  # noqa: BLE001
                log.warning(f"forwarding verdicts for {jid} failed: {e}")
                ok = False
            if not ok:
                self._fwd_next = time.time() + FORWARD_RETRY_S
                break
            with self.lock:
                for i in vs["like"] + vs["dislike"]:
                    self.forward_q.pop(self._key(jid, i), None)
                atomic_write_json(self.fwd_path, self.forward_q)
            sent += len(vs["like"]) + len(vs["dislike"])
        return sent

    # ---------------------------------------------------------------- posting on its own, once it has earned it
    def _auto_cfg(self) -> dict:
        a = self.cfg.get("clipper.feed.autopost", {}) or {}
        return {"enabled": bool(a.get("enabled", True)) and self.local, "min_ratings": int(a.get("min_ratings", 50)),
                "target": float(a.get("target_hit_rate", 0.8)), "max_per_day": int(a.get("max_per_day", 10)), "floor": float(a.get("floor", 0.6))}

    def autopost_status(self) -> dict:
        a = self._auto_cfg()
        if not a["enabled"]:
            return {"enabled": False, "why": "runs on the machine that holds the clips" if not self.local else "off"}
        g = self.taste.autopost_gate(a["min_ratings"], a["target"])
        return {"enabled": True, **g, "today": self._auto_today(), "max_per_day": a["max_per_day"], "floor": a["floor"],
                "need": self._need() if g["n"] >= a["min_ratings"] else None}

    def _auto_today(self) -> int:
        d = read_json(self.cfg.path("state") / "feed_auto.json") or {}
        return int(d.get(datetime.now().strftime("%Y-%m-%d"), 0))

    def _need(self) -> dict:
        """How many more clips the posting server wants for the next day (it posts YOUR liked clips first; only the
        shortfall is filled with the model's picks). Refreshed from the server; a stale / missing file falls back."""
        from . import taste
        path = self.cfg.path("state") / "clip_need.json"
        if time.time() - getattr(self, "_need_at", 0) > 1200:
            self._need_at = time.time()
            taste.pull_need(self.cfg)
        d = read_json(path) or {}
        a = self.cfg.get("clipper.feed.autopost", {}) or {}
        if not d or time.time() - float(d.get("at", 0)) > 14 * 3600:
            return {"need": int(a.get("fallback_need", 2)), "known": False}
        since = float(d.get("at", 0))
        sent = 0                                                   # liked / auto-queued since that snapshot: already covers some
        for j in self.store.list():
            for c in j.get("clips", []):
                f = c.get("feed") or {}
                if (f.get("verdict") == "like" and f.get("rated_at", 0) > since) or (f.get("auto") and f.get("auto_at", 0) > since):
                    sent += 1
        return {"need": max(0, int(d.get("need", 0)) - sent), "known": True, "approved": d.get("approved"),
                "per_day": d.get("per_day")}

    def autopost(self) -> int:
        """Post on its own, but only to fill the gap: your liked clips go first; if there aren't enough for the
        next day's slots, queue the clips the model expects you to like best (and only re-cut ones)."""
        a = self._auto_cfg()
        if not a["enabled"]:
            return 0
        g = self.taste.autopost_gate(a["min_ratings"], a["target"])
        if g["n"] < a["min_ratings"]:
            return 0
        need = self._need()
        room = min(need["need"], a["max_per_day"] - self._auto_today())
        if room <= 0:
            return 0
        floor = g["threshold"] if g.get("on") else a["floor"]      # proven confidence if it has it, else a sensible floor
        floor = max(floor, a["floor"])
        n = 0
        ranked = []
        for it in self.items(limit=0):
            if it["auto"] is not None or not it["fresh"] or it["key"] in self.forward_q:
                continue
            job, c = self._clip(it["job"], it["index"])
            ranked.append((self.taste.predict(T.features(c, job.get("source"))), it["job"], it["index"], c, job))
        for pred, job_id, idx, c, job in sorted(ranked, key=lambda x: -x[0]):
            if n >= room:
                break
            if pred < floor:
                break
            self._apply_local(job_id, idx, "like")
            self._patch_clip(job_id, idx, auto=True, auto_pred=round(pred * 100), auto_at=time.time())
            n += 1
            log.info(f"feed: auto-queued {job_id}#{idx} ({pred:.0%} sure)")
        if n:
            day = datetime.now().strftime("%Y-%m-%d")
            atomic_write_json(self.cfg.path("state") / "feed_auto.json", {day: self._auto_today() + n})
        return n

    # ---------------------------------------------------------------- re-cut clips made before the word-end fix
    def next_stale(self) -> tuple[str, int] | None:
        for key, v in self.forward_q.items():                       # clips you liked go first: they're about to be sent
            jid, idx = key.rsplit(":", 1)
            try:
                job, c = self._clip(jid, int(idx))
            except KeyError:
                continue
            if v == "like" and not _fresh(c) and not c.get("deleted") and key not in self._prev_failed:
                return jid, int(idx)
        for it in self.items(limit=0):
            if not it["fresh"] and it["key"] not in self._prev_failed and it["auto"] is None:
                return it["job"], it["index"]
        return None

    # ---------------------------------------------------------------- previews for clips that arrived without one
    def ensure_previews(self, limit: int = 1) -> int:
        if time.time() - self._last_prev < 4:
            return 0
        self._last_prev = time.time()
        made = 0
        for it in self.items(limit=120):
            if (it["is_preview"] and it["prev_v"] >= PREVIEW_V) or it["key"] in self._prev_failed:
                continue
            job_id, idx = it["job"], it["index"]
            try:
                job, c = self._clip(job_id, idx)
                d = self.store.dir(job_id)
                dst = d / c["file"].replace(".mp4", ".preview.mp4")
                src_file = d / c["file"] if (d / c["file"]).exists() else dst         # (a liked clip may live only as its preview)
                if preview.make(src_file, dst, nice=True):
                    with self.store._lock:
                        job, c = self._clip(job_id, idx)
                        self.store.set_clip(job_id, {**c, "preview": dst.name, "preview_v": PREVIEW_V})
                    made += 1
                else:
                    self._prev_failed.add(it["key"])
            except Exception as e:  # noqa: BLE001
                self._prev_failed.add(it["key"])
                log.warning(f"preview for {job_id}#{idx} failed: {e}")
            if made >= limit:
                break
        return made
