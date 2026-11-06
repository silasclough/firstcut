"""Twitch clips of the whitelisted streamers: candidates, the daily quota, the whole-clip moment, and the front of the line."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import twitch as TW


class Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code, self.ok = data, status, status < 400

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeHTTP:
    def __init__(self, clips):
        self.clips, self.calls = clips, []

    def post(self, url, **k):
        return Resp({"access_token": "t", "expires_in": 3600})

    def get(self, url, params=None, **k):
        self.calls.append((url, params))
        if url.endswith("/users"):
            return Resp({"data": [{"id": "1" if params["login"] == "kaicenat" else "2", "display_name": params["login"].title()}] if params["login"] != "nosuchperson" else []})
        return Resp({"data": self.clips.get(params["broadcaster_id"], [])})


def clip(cid, views, dur=30, age_h=5, title="Wild moment"):
    return {"id": cid, "url": f"https://clips.twitch.tv/{cid}", "title": title, "view_count": views, "duration": dur, "broadcaster_name": "Kai",
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=age_h)).strftime("%Y-%m-%dT%H:%M:%SZ")}


class Cfg:
    def __init__(self, tmp, chans, **t):
        self.tmp, self.chans, self.t = tmp, chans, t

    def get(self, k, d=None):
        return {"clipper.twitch": self.t, "clipper.auto.channels": self.chans}.get(k, d)

    def path(self, n):
        return self.tmp


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("TWITCH_CLIENT_ID", "id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "secret")


def test_logins_come_from_the_name_unless_set():
    assert TW.login_for({"name": "Kai Cenat Live"}) == "kaicenat" and TW.login_for({"name": "xQc"}) == "xqc" and TW.login_for({"name": "X", "twitch": "Other"}) == "other"


def test_candidates_are_the_best_unused_clips_within_the_limits(tmp_path):
    http = FakeHTTP({"1": [clip("a", 5000), clip("b", 100), clip("c", 4000, dur=90), clip("d", 3000, dur=20), clip("e", 9000)]})
    tw = TW.Twitch(Cfg(tmp_path, [{"name": "Kai Cenat"}, {"name": "Kai Cenat Live"}, {"name": "NoSuchPerson"}]), tmp_path, http)
    out = tw.candidates(seen=lambda cid: cid == "tw_e", today=lambda login: 0)
    assert [c["id"] for c in out] == ["tw_a", "tw_d"]                      # b: too few views, c: too long, e: already used
    assert out[0]["kind"] == "twitch_clip" and out[0]["views"] == 5000 and out[0]["twitch_login"] == "kaicenat"
    assert sum(1 for u, p in http.calls if u.endswith("/clips")) == 1       # "Kai Cenat Live" is the same streamer: asked once; the unknown name asks nothing


def test_a_streamer_past_the_daily_share_is_skipped_and_titles_with_avoided_words_too(tmp_path):
    http = FakeHTTP({"1": [clip("a", 5000, title="Ranked gameplay of the day"), clip("b", 4000)]})
    tw = TW.Twitch(Cfg(tmp_path, [{"name": "Kai Cenat"}]), tmp_path, http)
    assert [c["id"] for c in tw.candidates(lambda c: False, lambda l: 0, avoid=lambda t, ch: "ranked" in t.lower())] == ["tw_b"]
    assert tw.candidates(lambda c: False, lambda l: 2) == []


def test_nothing_without_keys(tmp_path, monkeypatch):
    monkeypatch.delenv("TWITCH_CLIENT_ID")
    assert TW.Twitch(Cfg(tmp_path, [{"name": "Kai Cenat"}]), tmp_path, FakeHTTP({})).candidates(lambda c: False, lambda l: 0) == []


def test_the_finder_hands_out_streamer_clips_first_and_stops_at_the_daily_quota(tmp_path, monkeypatch):
    from clipfarm.clipper import discover as D
    monkeypatch.setattr(TW.Twitch, "candidates", lambda self, seen, today, avoid=None: [{"id": "tw_a", "twitch_login": "kaicenat", "url": "u", "title": "t", "kind": "twitch_clip"},
                                                                                         {"id": "tw_b", "twitch_login": "xqc", "url": "u2", "title": "t2", "kind": "twitch_clip"}])
    f = D.Finder(Cfg(tmp_path, [], per_day=1), tmp_path)
    assert f.next_source()["id"] == "tw_a"
    assert f.twitch_source() is None                                        # per_day 1: used up, the old sources take over


def test_a_twitch_clip_becomes_one_whole_moment_and_skips_the_text_gate():
    from clipfarm.clipper import clipgate
    assert clipgate.problems({"duration": 20, "transcript": "no way he just did that and then"}, {"kind": "twitch_clip"}) == []
    assert clipgate.problems({"duration": 20, "transcript": "no way he just did that and then"}, {"kind": "cc"}) != []
