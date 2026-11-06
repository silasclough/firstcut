"""Twitch clips of the whitelisted streamers: the moments their fans already cut and rated.

    Twitch(cfg).candidates(seen=lambda id: bool, today=lambda login: int) -> [source dicts, best first]

A Twitch clip is 10-60 seconds, already a moment, with a view count that says how many people cared: far better raw material than guessing at moments in a 10-hour stream, and a small
file from Twitch's own servers (no YouTube download limits). Needs TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET (a free app at dev.twitch.tv/console, "client credentials" login).
Which streamers: the allowlisted channels in clipper.auto.channels (permission granted by their management); a channel entry may set `twitch: <login>`, otherwise the name is used
(spaces removed, a trailing "Live" dropped); a name Twitch does not know is skipped and remembered for a week. Config: clipper.twitch.*
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from ..core.log import get_logger

log = get_logger("clipper.twitch")
API = "https://api.twitch.tv/helix"
OVERRIDES = {"live speedy": "livespeedy", "fresh & fit": "freshandfit", "adin live": "adinross", "asmongold tv": "asmongold", "stable ronaldo live": "stableronaldo", "plaqueboymax live": "plaqueboymax",
             "dukedennis live": "dukedennis"}


def login_for(ch: dict) -> str:
    if ch.get("twitch"):
        return str(ch["twitch"]).lower()
    name = str(ch.get("name") or "").strip().lower()
    if name in OVERRIDES:
        return OVERRIDES[name]
    name = re.sub(r"\s+live$", "", name)
    return re.sub(r"[^a-z0-9_]", "", name)


class Twitch:
    def __init__(self, cfg, state_dir: Path | None = None, http=requests):
        self.cfg = cfg
        self.tcfg = cfg.get("clipper.twitch", {}) or {}
        self.dir = Path(state_dir or cfg.path("state"))
        self.http = http
        self._tok: tuple[str, float] | None = None

    def configured(self) -> bool:
        return bool(os.environ.get("TWITCH_CLIENT_ID") and os.environ.get("TWITCH_CLIENT_SECRET")) and self.tcfg.get("enabled", True) is not False

    # -- api -----------------------------------------------------------------------
    def _token(self) -> str:
        if self._tok and self._tok[1] > time.time() + 600:
            return self._tok[0]
        f = self.dir / "twitch_token.json"
        try:
            d = json.loads(f.read_text())
            if d.get("expires_at", 0) > time.time() + 600:
                self._tok = (d["access_token"], d["expires_at"])
                return d["access_token"]
        except (OSError, ValueError):
            pass
        r = self.http.post("https://id.twitch.tv/oauth2/token", timeout=30, data={"client_id": os.environ["TWITCH_CLIENT_ID"], "client_secret": os.environ["TWITCH_CLIENT_SECRET"],
                                                                                  "grant_type": "client_credentials"})
        r.raise_for_status()
        j = r.json()
        self._tok = (j["access_token"], time.time() + int(j.get("expires_in", 3600)))
        self.dir.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({"access_token": j["access_token"], "expires_at": self._tok[1]}))
        return j["access_token"]

    def _get(self, path: str, params: dict) -> list[dict]:
        r = self.http.get(f"{API}/{path}", params=params, timeout=30, headers={"Client-Id": os.environ["TWITCH_CLIENT_ID"], "Authorization": "Bearer " + self._token()})
        if r.status_code == 401:
            self._tok = None
            (self.dir / "twitch_token.json").unlink(missing_ok=True)
            r = self.http.get(f"{API}/{path}", params=params, timeout=30, headers={"Client-Id": os.environ["TWITCH_CLIENT_ID"], "Authorization": "Bearer " + self._token()})
        r.raise_for_status()
        return r.json().get("data", [])

    def _users(self) -> dict:
        try:
            return json.loads((self.dir / "twitch_users.json").read_text())
        except (OSError, ValueError):
            return {}

    def user_id(self, login: str) -> str | None:
        users = self._users()
        hit = users.get(login)
        if hit and (hit.get("id") or time.time() - hit.get("checked", 0) < 7 * 86400):
            return hit.get("id")
        data = self._get("users", {"login": login})
        users[login] = {"id": data[0]["id"] if data else None, "name": data[0]["display_name"] if data else None, "checked": time.time()}
        (self.dir / "twitch_users.json").write_text(json.dumps(users))
        return users[login]["id"]

    def clips(self, login: str, hours: float = 48, first: int = 25) -> list[dict]:
        uid = self.user_id(login)
        if not uid:
            return []
        since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self._get("clips", {"broadcaster_id": uid, "first": first, "started_at": since})

    # -- which ones ------------------------------------------------------------------
    def candidates(self, seen, today, avoid=lambda title, channel: False) -> list[dict]:
        """Sources (best first): recent clips of the whitelisted streamers with enough views, a sensible length, not used before, not past each streamer's daily share."""
        if not self.configured():
            return []
        t = self.tcfg
        min_views, lo, hi = int(t.get("min_views", 300)), float(t.get("min_s", 10)), float(t.get("max_s", 60))
        per_day, hours = int(t.get("per_streamer_per_day", 2)), float(t.get("hours", 48))
        out: list[dict] = []
        done_logins: set[str] = set()
        for ch in (self.cfg.get("clipper.auto.channels", []) or []):
            login = login_for(ch)
            if not login or login in done_logins or today(login) >= per_day:
                continue
            done_logins.add(login)
            try:
                clips = self.clips(login, hours)
            except Exception as e:  # noqa: BLE001 - one streamer failing must not stop the rest
                log.warning(f"twitch clips for {login} failed: {e}")
                continue
            for c in clips:
                cid, views, dur = "tw_" + c["id"], int(c.get("view_count") or 0), float(c.get("duration") or 0)
                title = (c.get("title") or "").strip()
                if views < min_views or not (lo <= dur <= hi) or seen(cid) or avoid(title, c.get("broadcaster_name") or ""):
                    continue
                age_h = max(2.0, (datetime.now(timezone.utc) - datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))).total_seconds() / 3600)
                out.append({"id": cid, "url": c["url"], "title": title, "channel": c.get("broadcaster_name") or ch.get("name"), "duration": dur, "kind": "twitch_clip",
                            "license": "allowlisted (permission granted by their management)", "views": views, "twitch_login": login, "twitch_clip": c["id"],
                            "channel_url": f"https://www.twitch.tv/{login}", "upload_date": c["created_at"][:10].replace("-", ""),
                            "why": f"Twitch clip, {views:,} views", "rank": math.log10(views + 1) + 0.5 * math.log10(views / age_h + 1)})
        return sorted(out, key=lambda s: -s["rank"])
