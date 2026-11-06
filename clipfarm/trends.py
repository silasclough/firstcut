"""What people are searching for right now, and how fresh it is.

Source: Google's public "Daily Search Trends" feed (https://trends.google.com/trending/rss?geo=US - no key, no login), which lists each trending
search with a rough traffic figure and the news stories behind it. We keep the list in state/trends.json and remember when each topic was first
seen, because a topic is only worth a video while it is fresh: `hot_hours` (default 48) after it first appears, it falls back to normal.

Used by:
  * the news / cartoon topic picker (pipelines/news.py): a headline that matches a trend is picked first, and the trend's own news stories are
    headlines too;
  * the clip finder (clipper/discover.py): recent uploads about a trend are tried first and the trend is searched for;
  * the scheduler (publish/scheduler.py): an approved video tagged with a hot trend jumps to the front of the line, and may take an extra slot,
    until the trend goes cold.
Only politics / elections and medical-advice / self-harm / sexual-abuse topics are skipped (TREND_EXCLUDE); tragedies and crimes are fine. Matching is by words, no AI involved.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone

import requests

from .core.log import get_logger
from .core.state import atomic_write_json, read_json

log = get_logger("trends")
FEED = "https://trends.google.com/trending/rss?geo={geo}"
HT = "{https://trends.google.com/trending/rss}"
# What trends never use: elections / politics and medical-advice topics. Tragedies, disasters and crimes are fine (they are what gets searched
# and clicked); the news pipeline's research step still keeps the telling factual and respectful. Override with trends.exclude_keywords.
TREND_EXCLUDE = ["election", "ballot", "vote", "voter", "polls", "campaign", "candidate", "primary", "president", "prime minister", "minister",
                 "parliament", "congress", "senate", "tory", "tories", "labour", "democrat", "republican", "politic", "chancellor", "vaccine",
                 "cancer cure", "treatment for", "overdose", "suicide", "child abuse", "rape"]
STOP = set("""the a an and or of to in on at for from with by vs versus is are was were be this that it its as new live today score game match
news update updates latest watch how what who when why where""".split())


def _cfg(ctx) -> dict:
    return ctx.cfg.get("trends", {}) or {}


def _traffic(s: str) -> int:
    m = re.match(r"\s*([\d,.]+)\s*([KMB]?)", s or "", re.I)
    if not m:
        return 0
    n = float(m.group(1).replace(",", ""))
    return int(n * {"": 1, "K": 1e3, "M": 1e6, "B": 1e9}[m.group(2).upper()])


def parse(xml_text: str) -> list[dict]:
    out = []
    for it in ET.fromstring(xml_text).iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        pub = None
        try:
            from email.utils import parsedate_to_datetime
            pub = parsedate_to_datetime(it.findtext("pubDate") or "").astimezone(timezone.utc).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            pass
        news = [{"title": (n.findtext(f"{HT}news_item_title") or "").strip(), "url": (n.findtext(f"{HT}news_item_url") or "").strip(),
                 "source": (n.findtext(f"{HT}news_item_source") or "").strip()} for n in it.findall(f"{HT}news_item")]
        out.append({"title": title, "traffic": _traffic(it.findtext(f"{HT}approx_traffic") or ""), "pub": pub,
                    "news": [n for n in news if n["title"]]})
    return out


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", str(text).lower().replace("'", "")) if w not in STOP and (len(w) > 2 or w.isdigit())]


def refresh(ctx, force: bool = False) -> list[dict]:
    """Fetch (at most every `refresh_minutes`) and merge into state/trends.json. -> the current trends, hottest first."""
    cfg = _cfg(ctx)
    path = ctx.cfg.path("state") / "trends.json"
    saved = read_json(path) or {"fetched_at": 0, "trends": {}}
    now = datetime.now(timezone.utc)
    age = now.timestamp() - float(saved.get("fetched_at", 0))
    if cfg.get("enabled", True) is False:
        return []
    if force or age >= float(cfg.get("refresh_minutes", 45)) * 60:
        try:
            r = requests.get(FEED.format(geo=cfg.get("geo", "US")), timeout=30, headers={"User-Agent": "Mozilla/5.0 (Clipfarm trends)"})
            r.raise_for_status()
            seen = {}
            for t in parse(r.text):
                key = " ".join(tokens(t["title"])) or t["title"].lower()
                old = saved["trends"].get(key, {})
                seen[key] = {**t, "key": key, "first_seen": old.get("first_seen") or now.isoformat(timespec="seconds"),
                             "last_seen": now.isoformat(timespec="seconds")}
            for key, old in saved["trends"].items():              # keep recently-gone ones a little while (they are still "recent")
                if key not in seen and now - datetime.fromisoformat(old["last_seen"]) < timedelta(hours=36):
                    seen[key] = old
            saved = {"fetched_at": now.timestamp(), "trends": seen}
            if not ctx.dry_run:
                atomic_write_json(path, saved)
            log.info(f"trends: {len(seen)} topics ({sum(1 for t in seen.values() if t['last_seen'] == now.isoformat(timespec='seconds'))} live)")
        except Exception as e:  # noqa: BLE001 - old list is fine
            log.warning(f"trends fetch failed: {e}")
    return current(ctx, saved)


def heat(t: dict, now: datetime | None = None) -> float:
    """0..~6: size of the search (log scale) plus a bonus for being new. Two thousand searches is ~1, a million is ~4."""
    now = now or datetime.now(timezone.utc)
    size = max(0.0, math.log10(max(int(t.get("traffic") or 0), 100)) - 1.5)
    age_h = (now - datetime.fromisoformat(t["first_seen"])).total_seconds() / 3600
    fresh = 1.5 if age_h < 12 else 1.0 if age_h < 24 else 0.5 if age_h < 48 else 0.0
    return round(size + fresh, 2)


def exclusions(ctx=None) -> list[str]:
    return (_cfg(ctx).get("exclude_keywords") if ctx else None) or TREND_EXCLUDE


def safe(t: dict, ctx=None) -> bool:
    from .pipelines import news
    ex = exclusions(ctx)
    item = {"title": t["title"], "summary": " ".join(n["title"] for n in t.get("news", [])[:3])}
    return news.excluded(item, ex) is None


def current(ctx, saved: dict | None = None, only_safe: bool = True) -> list[dict]:
    saved = saved or read_json(ctx.cfg.path("state") / "trends.json") or {"trends": {}}
    now = datetime.now(timezone.utc)
    hot_h = float(_cfg(ctx).get("hot_hours", 48))
    out = []
    for t in saved.get("trends", {}).values():
        age_h = (now - datetime.fromisoformat(t["first_seen"])).total_seconds() / 3600
        if only_safe and not safe(t, ctx):
            continue
        out.append({**t, "heat": heat(t, now), "age_h": round(age_h, 1), "hot": age_h <= hot_h,
                    "hot_until": (datetime.fromisoformat(t["first_seen"]) + timedelta(hours=hot_h)).isoformat(timespec="seconds")})
    return sorted(out, key=lambda t: -t["heat"])


def match(ctx, text: str, trends: list[dict] | None = None) -> dict | None:
    """The hottest hot trend this text is about (about = at least half the trend's distinctive words, and at least one real word), else None."""
    trends = [t for t in (trends if trends is not None else current(ctx)) if t["hot"]]
    have = set(tokens(text))
    best = None
    for t in trends:
        want = tokens(t["title"])
        if not want:
            continue
        got = [w for w in want if w in have]
        if len(got) * 2 >= len(want) and sum(len(w) for w in got) >= 5:
            if best is None or t["heat"] > best["heat"]:
                best = t
    return best


def tag(ctx, text: str, trends: list[dict] | None = None) -> dict:
    """Fields to put on a job / source: {} if the text is not about a hot trend, else trend name, heat and when it goes cold."""
    t = match(ctx, text, trends)
    if not t or t["heat"] < float(_cfg(ctx).get("min_heat", 1.5)):
        return {}
    return {"trend": t["title"], "trend_heat": t["heat"], "hot_until": t["hot_until"]}


def is_hot(job, now: datetime) -> bool:
    hu = (job.data or {}).get("hot_until")
    if not hu or not job.data.get("trend"):
        return False
    try:
        return datetime.fromisoformat(hu) > now.astimezone(timezone.utc)
    except ValueError:
        return False
