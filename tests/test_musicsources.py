"""Online music for FirstCut (clipper/musicsources.py): Openverse search keeps only licences fine for a monetised video, results are cached,
a fetched song lands in the library with its credit (sidecar + tags, so a copy keeps it), a bad download leaves nothing behind, Epidemic
Sound's free developer key never downloads, and a long video that uses a fetched CC BY song writes its credit line. No real network."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipfarm.clipper import musicsources as MS
from clipfarm.core import ffmpeg


class FakeResp:
    def __init__(self, status=200, data=None, content=b"", headers=None):
        self.status_code, self._data, self.content, self.headers = status, data, content, headers or {}
        self.ok = 200 <= status < 300

    def json(self):
        if self._data is None:
            raise ValueError("no json")
        return self._data

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ov(i, lic="by", ver="4.0", dur=150, title=None, tags=(), url=True, source="jamendo"):
    return {"id": f"ov{i}", "title": title or f"Song {i}", "creator": f"Artist {i}", "license": lic, "license_version": ver,
            "license_url": f"https://creativecommons.org/licenses/{lic}/{ver}/", "duration": dur * 1000, "foreign_landing_url": f"https://jamendo.com/track/{i}",
            "url": f"https://files.example/ov{i}.mp3" if url else None, "source": source, "filetype": "mp3", "tags": [{"name": t} for t in tags]}


@pytest.fixture(autouse=True)
def _fresh_cache():
    MS._cache.clear()
    yield
    MS._cache.clear()


@pytest.fixture
def mp3_bytes(tmp_path):
    p = tmp_path / "src.mp3"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "sine=f=440:d=40", "-c:a", "libmp3lame", "-b:a", "128k", str(p)],
                   check=True)
    return p.read_bytes()


def test_openverse_offers_only_licences_fine_for_a_monetised_video_and_caches(monkeypatch):
    calls = []
    results = [_ov(1), _ov(2, "cc0", "1.0"), _ov(3, "by-nc"), _ov(4, "by-sa"), _ov(5, "pdm", "1.0"), _ov(6, dur=12), _ov(7, dur=2000),
               _ov(8, tags=["sound effect"]), _ov(9, url=False), _ov(10, "by-nd")]
    monkeypatch.setattr(MS.requests, "get", lambda url, params=None, headers=None, timeout=None: calls.append(params) or FakeResp(200, {"results": results}))
    out = MS.openverse_search("piano", n=30)
    assert [t["id"] for t in out] == ["ov1", "ov2", "ov5"]                             # no NC / ND / SA, nothing under 30 s or over 15 min
    t = out[0]
    assert set(t) >= {"source", "id", "title", "artist", "dur", "preview", "license", "license_url", "credit", "page", "tags"}
    assert t["source"] == "openverse" and t["license"] == "CC BY 4.0" and t["preview"].startswith("https://") and t["dur"] == 150
    assert '"Song 1" by Artist 1' in t["credit"] and "CC BY 4.0" in t["credit"] and "jamendo.com/track/1" in t["credit"]
    assert out[1]["credit"] == "" and out[2]["license"].startswith("Public Domain")  # CC0 / public domain: no credit needed
    assert calls[0]["category"] == "music" and calls[0]["license"] == "by,cc0,pdm" and "UA" not in str(calls[0])
    MS.openverse_search("piano", n=30)
    assert len(calls) == 1                                                            # the same search again: from the cache


def test_several_words_also_search_each_word_and_rank_the_best_match_first(monkeypatch):
    def get(url, params=None, headers=None, timeout=None):
        q = params["q"]
        if q == "calm piano":
            return FakeResp(200, {"results": [_ov(1, title="Morning")]})
        if q == "calm":
            return FakeResp(200, {"results": [_ov(2, title="Calm sea"), _ov(3, title="Calm piano hour")]})
        return FakeResp(200, {"results": [_ov(4, title="Piano rain"), _ov(3, title="Calm piano hour")]})
    monkeypatch.setattr(MS.requests, "get", get)
    out = MS.openverse_search("calm piano", n=10)
    assert out[0]["id"] == "ov1" and [t["id"] for t in out].count("ov3") == 1        # the phrase itself first; no song twice
    assert [t["id"] for t in out].index("ov3") < [t["id"] for t in out].index("ov2")  # matching both words beats one


def test_a_busy_openverse_says_so(monkeypatch):
    monkeypatch.setattr(MS.time, "sleep", lambda s: None)
    monkeypatch.setattr(MS.requests, "get", lambda *a, **k: FakeResp(429, {}, headers={"Retry-After": "1"}))
    with pytest.raises(MS.MusicSourceError, match="busy"):
        MS.openverse_search("upbeat")


def test_a_fetched_song_lands_once_with_its_credit_and_a_copy_keeps_it(monkeypatch, tmp_path, mp3_bytes):
    gets = []
    monkeypatch.setattr(MS.requests, "get", lambda url, **k: gets.append(url) or FakeResp(200, content=mp3_bytes))
    t = {**MS._openverse_track(_ov(7, title='Bright "Day"/One')), "credit": 'Music: "Bright" by Artist 7 – CC BY 4.0'}
    p = MS.fetch(t, tmp_path / "Online")
    assert p.parent == tmp_path / "Online" / "openverse" and p.suffix == ".mp3" and "/" not in p.stem.replace("Online", "")
    side = json.loads(p.with_suffix(".json").read_text())
    assert side["source"] == "openverse" and side["id"] == "ov7" and side["license"] == "CC BY 4.0" and side["fetched_at"] > 0
    assert MS.credit_for(p) == t["credit"] and MS.license_for(p) == "CC BY 4.0"
    assert MS.fetch(t, tmp_path / "Online") == p and len(gets) == 1                   # already there: no second download
    copy = tmp_path / "Project" / "Music" / p.name
    copy.parent.mkdir(parents=True)
    shutil.copy2(p, copy)                                                             # (the app copies the song into the project)
    assert MS.credit_for(copy) == t["credit"]                                         # the credit is in the file's own tags too
    own = tmp_path / "mine.mp3"
    own.write_bytes(mp3_bytes)
    assert MS.credit_for(own) is None and MS.license_for(own) is None                 # your own music: no credit line
    from clipfarm.clipper import music as MU
    tr = MU.song(str(copy))
    assert tr.credit == t["credit"] and tr.license == "CC BY 4.0"
    assert MU.song(str(own)).credit == ""


def test_a_bad_download_leaves_nothing_behind(monkeypatch, tmp_path):
    monkeypatch.setattr(MS.requests, "get", lambda url, **k: FakeResp(200, content=b"<html>gone</html>" * 4000))
    t = MS._openverse_track(_ov(8))
    with pytest.raises(MS.MusicSourceError, match="isn't a song"):
        MS.fetch(t, tmp_path / "Online")
    assert not [p for p in (tmp_path / "Online" / "openverse").iterdir()]             # no partial file, no sidecar
    monkeypatch.setattr(MS.requests, "get", lambda url, **k: FakeResp(404))
    with pytest.raises(MS.MusicSourceError, match="HTTP 404"):
        MS.fetch(t, tmp_path / "Online")
    assert not [p for p in (tmp_path / "Online" / "openverse").iterdir()]


def test_epidemic_searches_and_previews_but_a_free_key_never_downloads(monkeypatch, tmp_path):
    seen = []

    def get(url, params=None, headers=None, timeout=None, **k):
        seen.append((url, params, headers))
        if headers and headers.get("Authorization") != "Bearer good-key":
            return FakeResp(401, {})
        if url.endswith("/v0/tracks/search"):
            return FakeResp(200, {"tracks": [{"id": "es1", "title": "Solaris", "mainArtists": [{"name": "Marc Torch"}], "bpm": 110, "length": 178,
                                              "moods": [{"id": "epic", "name": "Epic"}], "genres": [{"id": "rock", "name": "Rock"}]}]})
        if url.endswith("/stream"):
            return FakeResp(200, {"url": "https://audio.example/es1.m3u8", "expires": "x"})
        return FakeResp(404)
    monkeypatch.setattr(MS.requests, "get", get)
    assert MS.epidemic_status("bad-key")["ok"] is False and "didn't accept" in MS.epidemic_status("bad-key")["error"]
    assert MS.epidemic_status("good-key") == {"ok": True, "error": None, "name": "Epidemic Sound API key"}
    out = MS.epidemic_search("good-key", "epic rock")
    assert out[0]["source"] == "epidemic" and out[0]["artist"] == "Marc Torch" and out[0]["dur"] == 178 and "Epic" in out[0]["tags"]
    assert MS.epidemic_preview("good-key", "es1").startswith("https://")
    with pytest.raises(MS.MusicSourceError, match="aren't licensed"):
        MS.fetch(out[0], tmp_path / "Online", key="good-key")                       # (Epidemic's API Developer License)
    assert not any("/download" in u for u, _, _ in seen)
    assert all("good-key" not in json.dumps(v) for v in MS._cache.values())           # the key itself isn't kept


def test_a_long_video_with_a_fetched_cc_by_song_writes_its_credit(tmp_path, monkeypatch, mp3_bytes):
    import test_aivideo as TA

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    monkeypatch.setattr(MS.requests, "get", lambda url, **k: FakeResp(200, content=mp3_bytes))
    t = {**MS._openverse_track(_ov(11, title="Open Road")), "credit": 'Music: "Open Road" by Artist 11 (Jamendo via Openverse) – CC BY 4.0'}
    fetched = MS.fetch(t, tmp_path / "Online")
    song = tmp_path / "copied" / fetched.name                                        # a copy without its sidecar, as in a project
    song.parent.mkdir()
    shutil.copy2(fetched, song)
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state"), "music": str(tmp_path / "no-bundled-music")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = TA._keep_cut_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))

    class AI(TA._FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task == "broll_plan":
                return TA._Res({"cutaways": [{"at": 5.0, "scene": 0, "length": 2.0, "why": "x"}]})
            if task == "broll_pick":
                return TA._Res({"scene": 0, "why": "x"})
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    monkeypatch.setattr(LP, "LocalAI", lambda c: AI())
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: [{"text": f"w{i}.", "start": 0.5 + i, "end": 0.9 + i} for i in range(18)])
    runner = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "Open house", "", {"transcriber": "openai"}, {"clips": 0})
    projects.prepare_local(proj["id"], {"keep_cut": True, "aroll": "Interview.mov", "graphics": "", "music": "", "library": False, "song": str(song)})
    runner.make(proj["id"], {})
    assert store.get(proj["job"])["status"] == "done"
    credits = (folder / "Notes" / "credits.txt").read_text()
    assert t["credit"] in credits                                                     # CC BY: the credit line goes in the description


def test_fetched_songs_join_the_music_library_with_their_credit(monkeypatch, tmp_path, mp3_bytes):
    from types import SimpleNamespace

    from clipfarm.clipper import musiclib as ML
    monkeypatch.setattr(MS.requests, "get", lambda url, **k: FakeResp(200, content=mp3_bytes))
    home = tmp_path / "Music library"
    t = {**MS._openverse_track(_ov(12, title="Night Drive")), "credit": 'Music: "Night Drive" by Artist 12 – CC BY 4.0'}
    fetched = MS.fetch(t, home / "Online")
    mine = home / "Mine" / "My Song.mp3"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(mp3_bytes)
    data = {"clipper.music_lib": {"roots": [str(home / "Online"), str(home / "Mine")], "dir": str(tmp_path / "lib")}}
    lib = ML.MusicLibrary(SimpleNamespace(root=tmp_path, get=lambda k, d=None: data.get(k, d), path=lambda k: tmp_path / k))
    r = lib.index()
    assert r["tracks"] == 2 and r["failed"] == 0                                      # the song, not its .json sidecar
    online = lib.pick("", exclude={mine.name})
    assert online.path == fetched and online.credit == t["credit"] and online.license == "CC BY 4.0"
    own = lib.pick("", exclude={fetched.name})
    assert own.path == mine and own.credit == "" and own.license == "your music library"


def test_the_music_scrub_bar_gets_a_real_waveform(tmp_path, monkeypatch):
    """Music page: each song's waveform (160 peaks, 0-1) for the scrub bar; your songs by id, Openverse songs only by a recent result's id."""
    import subprocess
    from types import SimpleNamespace
    from clipfarm.clipper import studio_web as W
    from clipfarm.clipper import musicsources as MS
    from clipfarm.core import ffmpeg
    wav = tmp_path / "song.wav"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-f", "lavfi", "-i", "sine=f=220:d=3", "-af", "volume='if(lt(t,1.5),0.1,1)':eval=frame",
                    str(wav)], check=True)
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.music_file = lambda tid: wav
    w = api.music_wave("mine", "abc")
    assert len(w["peaks"]) == 160 and abs(w["dur"] - 3.0) < 0.1 and max(w["peaks"]) <= 1.0
    assert sum(w["peaks"][:70]) < sum(w["peaks"][90:])                                 # quiet first half, loud second half
    assert api.music_wave("mine", "abc") is w                                           # measured once
    import pytest
    with pytest.raises(KeyError):
        api.music_wave("openverse", "not-a-recent-result")                             # never a link from the page
    with pytest.raises(KeyError):
        api.music_wave("epidemic", "x")
