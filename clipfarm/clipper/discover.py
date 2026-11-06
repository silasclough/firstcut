"""Find long videos the nightly auto-clipper may use - only ones licensed for reuse.

Sources, in order:
  1. clipper.auto.channels       your allowlist: clipping campaigns (Whop etc.) and creators who said
                                 "clip me". Each entry carries a `why`, kept on every clip.
  2. clipper.auto.public_domain  public-domain publishers (e.g. NASA - US government works).
  3. Creative Commons search     YouTube search with the Creative Commons filter, rotating through
                                 clipper.auto.topics for variety. Every hit is re-checked: the video's
                                 own license field must say Creative Commons, or it is skipped.
Never used twice (data/state/clipper_seen.json). Lives, upcoming streams and videos outside
min/max length are skipped.
"""

from __future__ import annotations

import re
import time
from urllib.parse import quote_plus

from ..core.log import get_logger
from ..core.state import JsonState

log = get_logger("clipper.discover")
CC_FILTER = "EgIwAQ%3D%3D"           # YouTube search "Features: Creative Commons"


def _ydl(extra: dict | None = None):
    import yt_dlp
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True, "socket_timeout": 30,
                             **(extra or {})})


def _flat(url: str, n: int) -> list[dict]:
    try:
        with _ydl({"extract_flat": "in_playlist", "playlistend": n}) as y:
            info = y.extract_info(url, download=False) or {}
        return [e for e in (info.get("entries") or []) if e and e.get("id")]
    except Exception as e:  # noqa: BLE001 - one bad source shouldn't end the night
        log.warning(f"listing {url} failed: {e}")
        return []


def uploads_playlist(channel_url: str) -> str | None:
    """https://www.youtube.com/channel/UCxxxx/videos -> the channel's uploads playlist (newest first)."""
    m = re.search(r"/channel/UC([\w-]+)", channel_url or "")
    return f"https://www.youtube.com/playlist?list=UU{m.group(1)}" if m else None


def details(video_url: str) -> dict | None:
    try:
        with _ydl() as y:
            return y.extract_info(video_url, download=False)
    except Exception as e:  # noqa: BLE001
        log.warning(f"metadata for {video_url} failed: {e}")
        return None


def is_creative_commons(info: dict) -> bool:
    return "creative commons" in str(info.get("license") or "").lower()


def credit_line(src: dict) -> str:
    """The credit block that goes at the end of every post description: who made the original, where it is, its license."""
    kind = src.get("kind")
    who = src.get("channel") or "the original creator"
    lines = [f"Credit: {who}" + (f' - "{src["title"]}"' if src.get("title") else "")]
    lines.append(f"Original video: {src.get('url')}")
    if src.get("channel_url"):
        lines.append(f"{who}'s channel: {src['channel_url']}")
    if kind == "cc":
        lines.append("Licensed under Creative Commons Attribution (CC BY). Clipped and edited for Shorts.")
    elif kind == "public_domain":
        lines.append("Public domain. Clipped and edited for Shorts.")
    elif kind == "twitch_clip":
        lines.append("Twitch clip, edited for Shorts with the creator's permission.")
    else:
        lines.append("Clipped and edited for Shorts with the creator's permission.")
    return "\n".join(lines)


# Story clips beat everything else on the channel (a bodycam, a "fell off the roof" story: 1,000-2,800 views and the longest watch times; recipe, vlog and "time flew by" clips:
# about 5 seconds watched). So titles and sentences that sound like something HAPPENED rank higher, and recipes / vlogs / hauls lower. Not a filter: a push.
STORY_RE = re.compile(r"\b(story ?time|true story|what happened|survived|survival|caught|bodycam|body cam|dashcam|arrested|police|officer|cops?|confession|confess(?:ed)?|"
                      r"interrogat\w+|investigat\w+|accident|rescued?|escap(?:e|ed)|missing|disappear\w*|murder\w*|trial|lawsuit|scam(?:med)?|lied|betray\w*|"
                      r"blacked out|fell off|almost died|nearly died|went wrong|worst|terrifying|scariest|shocking|shocked|never saw|turned out|turns out|"
                      r"the day (?:i|we|he|she)|when i was|i was (?:in|at|on|driving|walking)|he told me|she told me|then (?:he|she|they)|emergency|crash(?:ed)?|fire|attack(?:ed)?)\b", re.I)
NOT_STORY_RE = re.compile(r"\b(recipes?|ingredients?|grwm|get ready with me|haul|unboxing|tutorial|workout|outfit|skincare|makeup|meal prep|vlog|day in (?:my|the) life|tips?|hacks?)\b", re.I)


def story_score(*texts: str) -> float:
    """-0.5 .. 1: how much this text sounds like something HAPPENED (1 = clearly a story, 0 = nothing special, negative = recipe / vlog / haul talk)."""
    t = " ".join(x or "" for x in texts)
    hits = len({m.group(0).lower() for m in STORY_RE.finditer(t)})
    bad = len({m.group(0).lower() for m in NOT_STORY_RE.finditer(t)})
    return max(-0.5, min(1.0, hits / 3.0 - 0.25 * bad))


class Finder:
    def __init__(self, cfg, state_dir):
        self.cfg = cfg
        self.tcfg = cfg.get("clipper.twitch", {}) or {}
        self.acfg = cfg.get("clipper.auto", {}) or {}
        self.state = JsonState(state_dir / "clipper_seen.json")
        self.state.data.setdefault("seen", {})
        self.state.data.setdefault("topic_i", 0)
        from .taste import Taste
        self.taste = Taste(state_dir)                 # what you liked / deleted in the swipe feed (pulled from the server)
        from types import SimpleNamespace
        self._tctx = SimpleNamespace(cfg=cfg, dry_run=False)            # what trends.py needs: the config (for the state folder) and nothing else

    def seen(self, vid: str) -> bool:
        return vid in self.state.data["seen"]

    def mark(self, vid: str, why: str) -> None:
        self.state.data["seen"][vid] = {"at": time.time(), "why": why}
        self.state.save()

    def is_avoided(self, title: str, channel: str = "") -> bool:
        """Real-life content only: gaming titles / channels are never downloaded."""
        t = f" {title.lower()} "
        words = [w.lower() for w in (self.acfg.get("avoid_words") or [])]
        bad_ch = {c.lower() for c in (self.acfg.get("avoid_channels") or [])}
        return channel.lower() in bad_ch or any((f" {w} " in t or (len(w) > 4 and w in t)) for w in words)

    def _ok_length(self, info: dict, max_minutes: float | None = None) -> bool:
        d = float(info.get("duration") or 0)
        top = float(max_minutes or self.acfg.get("max_minutes", 150))
        return float(self.acfg.get("min_minutes", 8)) * 60 <= d <= top * 60

    def _accept(self, entry: dict, kind: str, why: str, max_minutes: float | None = None) -> dict | None:
        vid = entry["id"]
        if self.seen(vid):
            return None
        url = entry.get("url") if str(entry.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={vid}"
        info = details(url)
        if not info:
            self.mark(vid, "metadata failed")
            return None
        if info.get("live_status") in ("is_live", "is_upcoming") or info.get("is_live"):
            return None                                   # come back once the stream is a VOD
        if not self._ok_length(info, max_minutes):
            self.mark(vid, f"length {info.get('duration')}s outside limits")
            return None
        if self.is_avoided(info.get("title") or entry.get("title") or "", info.get("channel") or info.get("uploader") or ""):
            self.mark(vid, "avoided topic (gaming)")
            return None
        chan = info.get("channel") or info.get("uploader")
        if chan and self.taste.channel_blocked(chan):
            self.mark(vid, f"taste: you usually delete clips from {chan}")
            return None
        if kind == "cc" and not is_creative_commons(info):
            self.mark(vid, f"not Creative Commons ({info.get('license')})")
            return None
        return {**self._trend_tag(f"{info.get('title') or ''} {info.get('description') or ''}"[:600]),
                "url": info.get("webpage_url") or url, "id": vid, "title": info.get("title"),
                "channel": info.get("channel") or info.get("uploader"), "duration": info.get("duration"),
                "channel_url": info.get("channel_url") or info.get("uploader_url"),
                "upload_date": info.get("upload_date"),
                "license": info.get("license") or ("public domain" if kind == "public_domain" else "allowlisted"),
                "kind": kind, "why": why}

    def _rank_entries(self, entries: list[dict], channel: str, kind: str) -> list[dict]:
        """Of a channel's recent uploads, try first the ones whose TITLE looks like what you like (words and creators you
        rate up) - not just the newest. A quarter of the time the order is left alone, so nothing is ever ruled out."""
        import random
        from . import taste as T
        m = self.taste.model()
        bias = float(self.acfg.get("story_bias", 0.6))
        if len(entries) < 2:
            return entries
        rng = random.Random()
        if bias > 0 and ((m.n < 30 and self.taste.audience_weight() <= 0) or rng.random() >= 0.15):      # story-sounding titles first (a few times in ten the order is left alone)
            if m.n < 30 and self.taste.audience_weight() <= 0:
                return sorted(entries, key=lambda e: -story_score(e.get("title")))          # stable: videos that tie keep their order
        elif (m.n < 30 and self.taste.audience_weight() <= 0):
            return entries
        if rng.random() < 0.25:
            return entries                                           # explore: newest first, as before
        # how well the creator's own uploads did with viewers tells us which videos are worth clipping: relative to the others
        views = sorted(v for v in (e.get("view_count") for e in entries) if v)
        med = views[len(views) // 2] if views else None

        def popularity(e):
            v = e.get("view_count")
            return max(-0.2, min(0.25, 0.12 * (v / med - 1))) if (v and med) else 0.0
        scored = [(self.taste.predict(T.features({"title": e.get("title") or ""}, {"channel": channel, "kind": kind}))
                   + popularity(e) + bias * story_score(e.get("title")) + 0.12 * rng.random(), e) for e in entries]
        return [e for _, e in sorted(scored, key=lambda x: -x[0])]

    def old_source(self, ch: dict, years_ago: float = 3.0, window: tuple[float, float] = (1.5, 5.0)) -> dict | None:
        """A usable video from this creator's channel from about `years_ago` years back (for "then vs now").
        Uploads are in date order, so a binary search over the channel's playlist finds that period in ~12 lookups."""
        from datetime import date, timedelta
        pl = uploads_playlist(ch.get("url", ""))
        if not pl:
            return None
        entries = _flat(pl, 6000)                                   # newest first
        if len(entries) < 20:
            return None
        target = date.today() - timedelta(days=int(years_ago * 365.25))

        def when(i: int):
            info = details(entries[i].get("url") if str(entries[i].get("url", "")).startswith("http")
                           else f"https://www.youtube.com/watch?v={entries[i]['id']}")
            d = str((info or {}).get("upload_date") or "")
            return date(int(d[:4]), int(d[4:6]), int(d[6:8])) if len(d) == 8 else None
        lo, hi = 0, len(entries) - 1
        for _ in range(14):
            if hi - lo <= 3:
                break
            mid = (lo + hi) // 2
            d = when(mid)
            if d is None:
                mid += 1
                d = when(min(mid, len(entries) - 1))
            if d is None:
                break
            if d > target:                                            # too recent: go further back (higher index)
                lo = mid
            else:
                hi = mid
        centre = (lo + hi) // 2
        order = [centre]
        for k in range(1, 40):
            order += [centre + k, centre - k]
        today = date.today()
        for i in order:
            if not (0 <= i < len(entries)):
                continue
            src = self._accept(entries[i], "allowlist", ch.get("why", "allowlisted by you"), ch.get("max_minutes"))
            if not src:
                continue
            d = str(src.get("upload_date") or "")
            if len(d) != 8:
                continue
            age = (today - date(int(d[:4]), int(d[4:6]), int(d[6:8]))).days / 365.25
            if window[0] <= age <= window[1]:
                src["list_url"] = ch["url"]
                src["era"] = "then"
                return src
            self.state.data["seen"].pop(src["id"], None)             # outside the wanted years: leave it for other uses
        return None

    def _allowlist_order(self, avoid: set[str]) -> list[dict]:
        """Allowlisted channels in the order to try them: ones not used tonight first, then the one used longest ago
        (a little shuffled), so consecutive videos - and the posts cut from them - come from different creators."""
        import random
        last = self.state.data.setdefault("chan_last", {})
        chans = [c for c in (self.acfg.get("channels", []) or []) if c.get("url")]
        rng = random.Random()
        rng.shuffle(chans)
        m = self.taste.model()

        def due(c):                                               # a creator that does well gets its turn ~6 h sooner
            f = f"channel:{str(c.get('name', '')).strip().lower()}"
            w = max(-1.5, min(1.5, m.weight(f) + 2.0 * self.taste.audience().weight(f))) if (m.n >= 8 or self.taste.audience_weight() > 0) else 0.0
            return last.get(c["url"], 0.0) - 14400 * w
        return sorted(chans, key=lambda c: (c["url"] in avoid, due(c)))

    def _trend_tag(self, text: str) -> dict:
        """{"trend", "trend_heat", "hot_until"} when this text is about a hot search trend (trends.py), else {}."""
        try:
            from .. import trends as TR
            return TR.tag(self._tctx, text, getattr(self, "_hot", None))
        except Exception as e:  # noqa: BLE001 - trends are a bonus
            log.debug(f"trend tag failed: {e}")
            return {}

    def trending_source(self, avoid: set[str]) -> dict | None:
        """A fresh video about something people are searching for right now: from your allowlisted channels' newest uploads (a title that matches a
        hot trend), or a Creative Commons search for the trend itself. Looked at most every `trend_pass_minutes`, so a quiet hour costs nothing."""
        from .. import trends as TR
        tcfg = self.acfg.get("trending", {}) or {}
        if tcfg.get("enabled", True) is False:
            return None
        try:
            hot = [t for t in TR.refresh(self._tctx) if t["hot"] and t["heat"] >= float(self._tctx.cfg.get("trends.min_heat", 1.5))]
        except Exception as e:  # noqa: BLE001
            log.warning(f"trends unavailable: {e}")
            return None
        self._hot = hot
        if not hot:
            return None
        last = float(self.state.data.get("trend_pass_at", 0))
        if time.time() - last < float(tcfg.get("pass_minutes", 45)) * 60:
            return None
        self.state.data["trend_pass_at"] = time.time()
        self.state.save()
        log.info("trending now: " + ", ".join(f"{t['title']} ({t['heat']})" for t in hot[:6]))
        for ch in self._allowlist_order(avoid):                                  # your permitted channels first
            if ch.get("name") and self.taste.channel_blocked(ch["name"]):
                continue
            entries = [e for e in _flat(ch["url"], int(tcfg.get("scan", 8))) if TR.match(self._tctx, e.get("title") or "", hot)]
            for e in entries:
                src = self._accept(e, "allowlist", ch.get("why", "allowlisted by you"), ch.get("max_minutes"))
                if src and src.get("trend"):
                    src["list_url"] = ch["url"]
                    return src
        for t in hot[: int(tcfg.get("searches", 3))]:                            # then a Creative Commons search for the trend itself
            url = f"https://www.youtube.com/results?search_query={quote_plus(t['title'])}&sp={CC_FILTER}"
            for e in _flat(url, int(tcfg.get("per_search", 8))):
                if not TR.match(self._tctx, e.get("title") or "", [t]):
                    continue
                src = self._accept(e, "cc", "YouTube Creative Commons (CC BY) license")
                if src and src.get("trend"):
                    src["topic"] = t["title"]
                    return src
        return None

    def twitch_source(self) -> dict | None:
        """The best unused Twitch clip of a whitelisted streamer (clipper.twitch): fans have already cut and rated these moments. None when Twitch is not set up, nothing
        clears the view bar, or today's quota (clipper.twitch.per_day) is used up - then the other sources take over."""
        from .twitch import Twitch
        tw = Twitch(self.cfg, self.state.path.parent)
        if not tw.configured():
            return None
        day = time.strftime("%Y-%m-%d")
        used = self.state.data.setdefault("twitch_used", {})
        for d in [d for d in used if d != day]:
            used.pop(d)
        today = used.setdefault(day, {})
        if sum(today.values()) >= int(self.tcfg.get("per_day", 12)):
            return None
        try:
            cands = tw.candidates(self.seen, lambda login: today.get(login, 0), self.is_avoided)
        except Exception as e:  # noqa: BLE001 - Twitch down: the old sources still work
            log.warning(f"twitch source failed: {e}")
            return None
        if not cands:
            return None
        src = cands[0]
        today[src["twitch_login"]] = today.get(src["twitch_login"], 0) + 1
        self.state.save()
        return src

    def next_source(self, avoid: set[str] | None = None, twitch: bool = True) -> dict | None:
        """`avoid`: allowlist channel urls already used in this run (tried again only when nothing else is left)."""
        tw_src = self.twitch_source() if twitch else None                      # 0a. a streamer moment the fans already cut and rated
        if tw_src:
            return tw_src
        hot_src = self.trending_source(avoid or set())                         # 0. something people are searching for right now
        if hot_src:
            return hot_src
        for ch in self._allowlist_order(avoid or set()):                       # 1. allowlist (permission granted)
            if ch.get("name") and self.taste.channel_blocked(ch["name"]):
                continue
            entries = self._rank_entries(_flat(ch["url"], int(ch.get("scan", self.acfg.get("per_channel_scan", 16)))),
                                         str(ch.get("name", "")), "allowlist")
            for e in entries:
                src = self._accept(e, "allowlist", ch.get("why", "allowlisted by you"), ch.get("max_minutes"))
                if src:
                    src["list_url"] = ch["url"]
                    self.state.data.setdefault("chan_last", {})[ch["url"]] = time.time()
                    self.state.save()
                    return src
        for ch in self.acfg.get("public_domain", []) or []:                    # 2. public domain
            for e in _flat(ch["url"], int(self.acfg.get("per_channel_scan", 8))):
                src = self._accept(e, "public_domain", ch.get("why", "public domain"))
                if src:
                    return src
        topics = list(self.acfg.get("topics", []) or [])                       # 3. Creative Commons
        i0 = int(self.state.data.get("topic_i", 0))
        order = [topics[(i0 + k) % len(topics)] for k in range(len(topics))] if topics else []
        if topics and self.taste.model().n >= 8:                               # liked topics first, still exploring
            import random
            order = self.taste.topic_order(topics, random.Random())
        if order and float(self.acfg.get("story_bias", 0.6)) > 0:              # searches that sound like stories go first, the rest keep their order after them
            order = [q for q in order if story_score(q) > 0] + [q for q in order if story_score(q) <= 0]
        for q in order:
            i = topics.index(q)
            url = f"https://www.youtube.com/results?search_query={quote_plus(q)}&sp={CC_FILTER}"
            for e in self._rank_entries(_flat(url, int(self.acfg.get("per_search", 15))), "", "cc"):
                src = self._accept(e, "cc", "YouTube Creative Commons (CC BY) license")
                if src:
                    src["topic"] = q
                    self.state.data["topic_i"] = i + 1                 # next night starts on another topic
                    self.state.save()
                    return src
        return None
