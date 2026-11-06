"""Learn from how posted clips actually perform (views / watch time on My Channel), not only from your swipes.

Server (it posts, so it has the Metricool access):   refresh(ctx)  -> state/clip_perf.json
    pulls the per-video analytics, matches every video to the clip it was made from, and stores the clip's features
    next to its numbers (at most every `clipper.performance.every_hours`).
M2 (it picks and cuts):   Feed.refresh_perf() copies that file and feeds it to taste.ingest_perf().
    A clip that did clearly better than the channel's median counts as a "like" for its creator / topic / length /
    title words, one that did clearly worse as a "no" - so tomorrow's downloads, moments and feed order follow what
    actually works, not just what you happened to like. Never trained on before a post is `min_age_h` old.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timedelta

from ..core.log import get_logger
from ..core.state import atomic_write_json, read_json
from . import taste as T
from .store import Store

log = get_logger("clipper.performance")
MIN_MATURE = 20                     # no conclusions before this many posts are old enough: a channel this young is mostly noise


def _match(ctx, rows: list[dict], jobs) -> list[tuple[dict, object]]:
    out = []
    for r in rows:
        hit = next((j for j in jobs if r.get("url") and (j.publish or {}).get("public_url") == r["url"]), None)
        if not hit and r.get("text"):
            key = str(r["text"]).strip()[:60]
            hit = next((j for j in jobs if (j.data.get("caption") or "").strip()[:60] == key), None)
        if not hit and r.get("text"):                              # the video's title / description starts with our title
            first = str(r["text"]).strip().lower()[:40]
            hit = next((j for j in jobs if first and (j.data.get("title") or "").strip().lower()[:40] == first), None)
        if not hit and r.get("published"):
            try:
                t = datetime.fromisoformat(str(r["published"])).replace(tzinfo=None)
                near = sorted(((abs((datetime.fromisoformat(j.publish["slot"]).replace(tzinfo=None) - t).total_seconds()), j)
                               for j in jobs), key=lambda x: x[0])
                hit = near[0][1] if near and near[0][0] < 45 * 60 else None
            except (ValueError, KeyError):
                pass
        if hit:
            out.append((r, hit))
    return out


def refresh(ctx, force: bool = False) -> dict | None:
    """Server: fetch analytics for posted clipper videos and store them with their clip features."""
    pcfg = ctx.cfg.get("clipper.performance", {}) or {}
    if not pcfg.get("enabled", True):
        return None
    path = ctx.cfg.path("state") / "clip_perf.json"
    old = read_json(path) or {}
    if not force and time.time() - float(old.get("fetched_at", 0)) < float(pcfg.get("every_hours", 12)) * 3600:
        return None
    brand = ctx.cfg.get("formats.clipper.brand", "clipper_channel")
    blog = ctx.cfg.get(f"brands.{brand}.metricool_blog_id")
    if not blog:
        return None
    from ..publish.metricool_mcp import fetch_stats
    end = ctx.cfg.now()
    rows = fetch_stats(ctx, str(blog), ["youtube"], end - timedelta(days=int(pcfg.get("days", 28))), end)
    if rows:
        log.info(f"performance: first analytics row: { {k: (str(v)[:50] if v is not None else None) for k, v in rows[0].items()} }")
    jobs = [j for j in ctx.queue.all() if j.format == "clipper" and j.brand == brand and (j.publish or {}).get("slot")]
    store = Store(ctx.cfg.root / ctx.cfg.get("clipper.data_dir", "data/clipper"))
    posts = {}

    def entry(j, r):
        cj = str(j.data.get("clipper_job") or "").split("+")[0]
        try:
            job = store.get(cj)
            c = next(c for c in job["clips"] if c["index"] == int(j.data.get("clip_index", -1)))
        except (KeyError, StopIteration, ValueError):
            return None
        slot = datetime.fromisoformat(j.publish["slot"])
        age_h = (end - slot.astimezone(end.tzinfo)).total_seconds() / 3600

        def num(k):
            v = r.get(k)
            if isinstance(v, bool) or v is None:
                return None
            try:
                return float(str(v).replace(",", "").strip().rstrip("%"))
            except ValueError:
                return None
        return {"url": r.get("url"), "views": num("views"), "likes": num("likes"), "comments": num("comments"), "shares": num("shares"),
                "avg_watch_s": num("avg_watch_s"), "duration": c.get("duration"), "age_h": round(age_h, 1),
                "slot_hour": slot.astimezone(end.tzinfo).hour, "picked_as": j.data.get("picked_as"),
                "channel": (job.get("source") or {}).get("channel"), "title": c.get("title"),
                "feats": T.features(c, job.get("source"))}
    for r, j in _match(ctx, rows, jobs):
        e = entry(j, r)
        if e:
            posts[j.id] = e
    # Metricool's numbers are days behind: a clip that is live but has no analytics row yet is listed too (views unknown), and its public
    # view / like counts are read from the watch page within hours (fill_public), so the channel learns from the FIRST day, not the third.
    first_day = int(pcfg.get("first_day_min_age_h", 3))
    for j in jobs:
        url = (j.publish or {}).get("public_url")
        if j.id in posts or not url or str(getattr(j.state, "value", j.state)) != "posted":
            continue
        e = entry(j, {"url": url})
        if e and e["age_h"] >= first_day:
            posts[j.id] = e
    try:
        fill_public({"posts": posts})
    except Exception as ex:  # noqa: BLE001 - no watch-page access on this machine: the M2 fills them in
        log.info(f"performance: public counts not read here ({ex})")
    if len(posts) < len(rows):                                    # say WHY some rows didn't match (so the matching can be fixed)
        matched = {id(r) for r, _ in _match(ctx, rows, jobs)}
        for r in [r for r in rows if id(r) not in matched][:3]:
            log.info(f"performance: unmatched row published={r.get('published')} url={r.get('url')} text={str(r.get('text'))[:60]!r}")
        for j in jobs[:3]:
            log.info(f"performance: job {j.id} slot={(j.publish or {}).get('slot')} title={(j.data.get('title') or '')[:40]!r}")
    data = {"fetched_at": time.time(), "posts": posts}
    atomic_write_json(path, data)
    log.info(f"performance: {len(posts)} posted clips matched to analytics ({len(rows)} rows)")
    return data


def fill_public(perf: dict, details=None, every_h: float = 3.0) -> int:
    """Metricool's YouTube numbers lag by days on a new channel, but a video's public view / like / comment counts are
    there within hours. Fill in the missing ones from the watch page (no login needed). Returns how many were filled."""
    if details is None:
        from .discover import details as _d
        details = _d
    n = 0
    for p in (perf.get("posts") or {}).values():
        if p.get("views") is not None and not p.get("public"):
            continue                                              # Metricool has real numbers for this one
        if not p.get("url") or time.time() - float(p.get("public_at", 0)) < every_h * 3600:
            continue
        try:
            info = details(p["url"]) or {}
        except Exception as e:  # noqa: BLE001
            log.warning(f"public stats for {p['url']} failed: {e}")
            continue
        if info.get("view_count") is None:
            continue
        p.update({"views": float(info["view_count"]), "likes": float(info.get("like_count") or 0),
                  "comments": float(info.get("comment_count") or 0), "public": True, "public_at": time.time()})
        n += 1
    return n


def labels(perf: dict, min_age_h: float = 48) -> dict[str, tuple[int, list[str]]]:
    """job id -> (1 = clearly better than the channel's median, 0 = clearly worse, features). Middle ground is skipped."""
    mature = {k: p for k, p in (perf.get("posts") or {}).items()
              if p.get("views") is not None and p.get("age_h", 0) >= min_age_h}
    if len(mature) < MIN_MATURE:
        return {}
    rate = {k: p["views"] / max(6.0, min(p.get("age_h", 24), 72.0)) for k, p in mature.items()}   # views per hour (first 3 days)
    med = statistics.median(rate.values()) or 1.0
    ret = {k: (p["avg_watch_s"] / p["duration"]) for k, p in mature.items() if p.get("avg_watch_s") and p.get("duration")}
    rmed = statistics.median(ret.values()) if len(ret) >= MIN_MATURE else None
    out = {}
    for k, p in mature.items():
        v = rate[k] / med
        r = (ret[k] / rmed) if (rmed and k in ret) else 1.0
        score = 0.9 * v + 0.1 * r                       # reach decides; retention hardly counts yet (AI-edited clips mostly lose people early)
        if score >= 1.6:
            out[k] = (1, p["feats"])
        elif score <= 0.4:
            out[k] = (0, p["feats"])
    return out


def summary(perf: dict) -> dict:
    posts = list((perf.get("posts") or {}).values())
    views = [p["views"] for p in posts if p.get("views") is not None]
    by: dict[str, list[float]] = {}
    for p in posts:
        if p.get("channel") and p.get("views") is not None:
            by.setdefault(p["channel"], []).append(p["views"])
    best = sorted(((statistics.median(v), c, len(v)) for c, v in by.items() if len(v) >= 2), reverse=True)
    return {"posts": len(posts), "median_views": statistics.median(views) if views else None,
            "by_channel": [{"channel": c, "median_views": m, "posts": n} for m, c, n in best[:6]],
            "updated": perf.get("fetched_at")}
