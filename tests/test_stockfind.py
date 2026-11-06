"""Stock in FirstCut: Find footage's stock search (clipper/stockfind.py: Pixabay, Pexels, Openverse, one normalized list, licences kept
safe, cached), downloading a result into the app's B-roll folder or ~/Downloads with its credit, the Music page's favourites (settings
music.favorites) and its Download button. All offline: every provider answer is made up here."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import stockfind as SF
from clipfarm.clipper import studio as SD


# ---------------------------------------------------------------------------------------------- made-up provider answers
PIXABAY_VIDEOS = {"hits": [{"id": 101, "pageURL": "https://pixabay.com/videos/id-101/", "tags": "graduation, caps, students", "duration": 14,
                            "user": "Lena", "picture_id": "", "videos": {
                                "large": {"url": "https://cdn.pixabay.com/video/101_large.mp4", "width": 3840, "height": 2160, "thumbnail": "https://cdn.pixabay.com/video/101_large.jpg"},
                                "medium": {"url": "https://cdn.pixabay.com/video/101_medium.mp4", "width": 1920, "height": 1080, "thumbnail": "https://cdn.pixabay.com/video/101_medium.jpg"},
                                "small": {"url": "https://cdn.pixabay.com/video/101_small.mp4", "width": 1280, "height": 720},
                                "tiny": {"url": "https://cdn.pixabay.com/video/101_tiny.mp4", "width": 640, "height": 360}}}]}
PIXABAY_PHOTOS = {"hits": [{"id": 202, "pageURL": "https://pixabay.com/photos/id-202/", "tags": "campus, library", "user": "Omar",
                            "webformatURL": "https://pixabay.com/get/202_640.jpg", "largeImageURL": "https://pixabay.com/get/202_1280.jpg",
                            "imageWidth": 6000, "imageHeight": 4000}]}
PEXELS_VIDEOS = {"videos": [{"id": 303, "url": "https://www.pexels.com/video/students-throwing-caps-303/", "image": "https://images.pexels.com/videos/303/pic.jpg",
                             "duration": 9, "user": {"name": "Ana"}, "video_files": [
                                 {"link": "https://videos.pexels.com/303_uhd.mp4", "file_type": "video/mp4", "width": 3840, "height": 2160},
                                 {"link": "https://videos.pexels.com/303_hd.mp4", "file_type": "video/mp4", "width": 1920, "height": 1080},
                                 {"link": "https://videos.pexels.com/303_sd.mp4", "file_type": "video/mp4", "width": 640, "height": 360}]}]}
PEXELS_PHOTOS = {"photos": [{"id": 404, "url": "https://www.pexels.com/photo/404/", "photographer": "Kai", "alt": "Students on a lawn",
                             "width": 5000, "height": 3333, "src": {"medium": "https://images.pexels.com/404/m.jpg", "large2x": "https://images.pexels.com/404/l2.jpg"}}]}
OPENVERSE = {"results": [
    {"id": "ov-ok", "title": "Old campus", "creator": "Sam", "license": "by", "license_version": "4.0", "url": "https://live.staticflickr.com/ok.jpg",
     "thumbnail": "https://api.openverse.org/v1/images/ov-ok/thumb/", "width": 2000, "height": 1500, "foreign_landing_url": "https://flickr.com/photos/ok",
     "attribution": '"Old campus" by Sam is licensed under CC BY 4.0.'},
    {"id": "ov-pd", "title": "Bell tower", "creator": "Unknown", "license": "cc0", "url": "https://upload.wikimedia.org/pd.jpg", "width": 900, "height": 600},
    {"id": "ov-nc", "title": "No commercial use", "creator": "X", "license": "by-nc", "url": "https://example.org/nc.jpg"},
    {"id": "ov-sa", "title": "Share alike", "creator": "Y", "license": "by-sa", "url": "https://example.org/sa.jpg"}]}


@pytest.fixture
def providers(monkeypatch):
    """The five search endpoints, answered from the tables above; every call is recorded (url, params, headers)."""
    calls = []
    answers = {"https://pixabay.com/api/videos/": PIXABAY_VIDEOS, "https://pixabay.com/api/": PIXABAY_PHOTOS,
               "https://api.pexels.com/videos/search": PEXELS_VIDEOS, "https://api.pexels.com/v1/search": PEXELS_PHOTOS,
               "https://api.openverse.org/v1/images/": OPENVERSE}

    def get(url, **kw):
        calls.append((url, kw.get("params") or {}, kw.get("headers") or {}))
        if url in answers.get("_down", ()):
            raise ConnectionError("down")
        return json.loads(json.dumps(answers[url]))

    monkeypatch.setattr(SF, "_get", get)
    monkeypatch.setattr(SF, "_CACHE", {})
    monkeypatch.setattr(SF, "_ITEMS", {})
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    return SimpleNamespace(calls=calls, answers=answers)


def test_without_keys_stock_is_openverse_photos_with_only_safe_licences(providers):
    assert SF.sources() == {"pixabay": False, "pexels": False, "openverse": True}
    assert SF.search("graduation", ("video",)) == []                     # no videos without a Pixabay or Pexels key
    got = SF.search("graduation", ("video", "photo"))
    assert [g["id"] for g in got] == ["openverse:photo:ov-ok", "openverse:photo:ov-pd"]     # NC and SA are never offered
    ok, pd = got
    assert ok["license"] == "CC BY 4.0" and ok["credit"] == '"Old campus" by Sam is licensed under CC BY 4.0.' and ok["page"] == "https://flickr.com/photos/ok"
    assert pd["license"] == "Public domain" and pd["credit"] == ""        # nothing to credit
    assert pd["download"] == "https://upload.wikimedia.org/pd.jpg"                           # (no thumbnail: Openverse's small one,
    assert pd["thumb"] == "https://api.openverse.org/v1/images/ov-pd/thumb/"                #  never the original, which can be 20 MB)
    assert {c[0] for c in providers.calls} == {"https://api.openverse.org/v1/images/"}
    assert providers.calls[-1][1]["license"] == "cc0,pdm,by" and providers.calls[-1][2] == {}


def test_every_provider_is_normalized_the_same_way(providers, monkeypatch):
    monkeypatch.setenv("PIXABAY_API_KEY", "pk-secret")
    monkeypatch.setenv("PEXELS_API_KEY", "px-secret")
    got = SF.search("students", ("video", "photo"), n=24)
    by = {g["id"]: g for g in got}
    keys = {"source", "kind", "id", "what", "thumb", "preview", "w", "h", "dur", "page", "license", "credit", "download", "creator",
            "match", "best"}                                                            # (how well the picture shows the words; top 3)
    assert all(set(g) == keys for g in got)
    assert [g["kind"] for g in got[:2]] == ["video", "video"]            # one from each provider in turn, videos first
    v = by["pixabay:video:101"]
    assert v["download"].endswith("101_medium.mp4") and (v["w"], v["h"]) == (1920, 1080)       # the best file at or under 1080p
    assert v["preview"].endswith("101_tiny.mp4") and v["thumb"].endswith("101_medium.jpg") and v["dur"] == 14.0
    assert v["what"] == "graduation caps students" and v["credit"] == "Video by Lena from Pixabay" and v["license"] == "Pixabay Content License"
    pv = by["pexels:video:303"]
    assert pv["download"].endswith("303_hd.mp4") and pv["preview"].endswith("303_sd.mp4") and pv["what"] == "students throwing caps"
    assert pv["credit"] == "Video by Ana from Pexels" and pv["page"] == "https://www.pexels.com/video/students-throwing-caps-303/"
    pp = by["pexels:photo:404"]
    assert pp["kind"] == "photo" and pp["download"].endswith("l2.jpg") and pp["thumb"].endswith("m.jpg") and pp["preview"] == ""
    assert by["pixabay:photo:202"]["download"].endswith("202_1280.jpg") and by["pixabay:photo:202"]["credit"] == "Photo by Omar from Pixabay"
    # the keys go to their own provider only, and never into a result
    pex = [c for c in providers.calls if "pexels" in c[0]]
    assert all(c[2].get("Authorization") == "px-secret" for c in pex) and all("key" not in c[1] for c in pex)
    assert all(c[1].get("key") == "pk-secret" for c in providers.calls if "pixabay" in c[0])
    assert "secret" not in json.dumps(got)


def test_results_are_cached_for_ten_minutes_and_a_new_key_searches_again(providers, monkeypatch):
    SF.search("bell tower", ("photo",))
    n = len(providers.calls)
    SF.search("  Bell   Tower ", ("photo",))                              # the same words: no new request
    assert len(providers.calls) == n
    SF.search("bell tower", ("photo",), page=2)                          # another page is another search
    assert len(providers.calls) == n + 1
    monkeypatch.setenv("PIXABAY_API_KEY", "k")                          # connecting a source: the next search asks it too
    SF.search("bell tower", ("photo",))
    assert any("pixabay" in c[0] for c in providers.calls)
    for k, (at, val) in list(SF._CACHE.items()):
        SF._CACHE[k] = (at - SF.CACHE_S - 1, val)
    m = len(providers.calls)
    SF.search("bell tower", ("photo",))                                  # (after ten minutes: asked again)
    assert len(providers.calls) > m
    assert SF.search("   ", ("photo",)) == []


def test_one_provider_down_still_answers_with_the_others(providers, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "k")
    providers.answers["_down"] = ("https://api.pexels.com/v1/search",)
    got = SF.search("campus", ("photo",))
    assert {g["source"] for g in got} == {"openverse"}


def test_only_results_this_app_showed_can_be_downloaded(providers):
    assert SF.known("openverse:photo:ov-ok") is None
    SF.search("campus", ("photo",))
    assert SF.known("openverse:photo:ov-ok")["download"] == "https://live.staticflickr.com/ok.jpg"
    assert SF.known("openverse:photo:ov-nc") is None and SF.known("") is None


class FakeDownload:
    def __init__(self, body, status=200):
        self.body, self.status = body, status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            raise SF.requests.HTTPError(f"HTTP {self.status}")

    def iter_content(self, n):
        yield from (self.body[i:i + n] for i in range(0, len(self.body), n))


def test_fetch_saves_the_file_once_with_its_licence_and_credit(tmp_path, monkeypatch):
    got = []
    monkeypatch.setattr(SF.requests, "get", lambda url, **kw: got.append((url, kw)) or FakeDownload(b"\0" * 20_000))
    item = {"source": "pexels", "kind": "video", "id": "pexels:video:303", "what": "students throwing caps", "download": "https://videos.pexels.com/303_hd.mp4",
            "page": "https://www.pexels.com/video/303/", "license": "Pexels License", "credit": "Video by Ana from Pexels", "creator": "Ana", "w": 1920, "h": 1080, "dur": 9.0}
    p = SF.fetch(item, tmp_path / "Stock")
    assert p == tmp_path / "Stock" / "pexels" / "students throwing caps - pexels 303.mp4" and p.stat().st_size == 20_000
    side = json.loads(p.with_suffix(".json").read_text())
    assert side["credit"] == "Video by Ana from Pexels" and side["license"] == "Pexels License" and side["page"].endswith("/303/")
    assert SF.credit_for(p) == "Video by Ana from Pexels"
    assert got[0][1]["headers"]["User-Agent"].startswith("FirstCut/")
    assert SF.fetch(item, tmp_path / "Stock") == p and len(got) == 1       # already there: no second download
    # a failed or empty download leaves nothing behind
    monkeypatch.setattr(SF.requests, "get", lambda url, **kw: FakeDownload(b"tiny"))
    bad = {**item, "id": "pexels:video:9", "what": "empty"}
    with pytest.raises(ValueError):
        SF.fetch(bad, tmp_path / "Stock")
    monkeypatch.setattr(SF.requests, "get", lambda url, **kw: FakeDownload(b"x" * 20_000, 404))
    with pytest.raises(SF.requests.HTTPError):
        SF.fetch({**item, "id": "pexels:video:10", "what": "gone"}, tmp_path / "Stock")
    assert sorted(x.name for x in (tmp_path / "Stock" / "pexels").iterdir()) == sorted([p.name, p.with_suffix(".json").name])
    for wrong in ({**item, "download": "http://videos.pexels.com/x.mp4"}, {**item, "source": "somewhere"}, "not a result"):
        with pytest.raises(ValueError):
            SF.fetch(wrong, tmp_path / "Stock")


# ---------------------------------------------------------------------------------------------- favourites (settings)
def test_favourites_keep_small_checked_records_each_song_once():
    raw = [{"source": "openverse", "id": "ab-12", "title": "  Calm   piano ", "artist": "Mia", "dur": "95.25", "preview": "https://cdn.example/a.mp3",
            "license": "CC BY 4.0", "credit": "Music: Calm piano by Mia", "page": "javascript:alert(1)", "extra": "x" * 5000},
           {"source": "openverse", "id": "ab-12", "title": "the same song again"},
           {"source": "library", "id": "0123abcd", "title": "Mine", "path": "/Users/x/Music/mine.mp3"},
           {"source": "openverse", "id": "../../etc", "title": "bad id"}, {"source": "spotify", "id": "1"}, "nonsense", {"source": "epidemic", "id": "e1", "dur": "nan"}]
    got = SD.music_favorites(raw)
    assert [f["id"] for f in got] == ["ab-12", "0123abcd", "e1"]
    a = got[0]
    assert a["title"] == "Calm piano" and abs(a["dur"] - 95.25) < 0.1
    assert a["page"] == "" and a["preview"] == "https://cdn.example/a.mp3" and "extra" not in a and a["path"] == ""   # links: web only
    assert got[1]["path"] == "/Users/x/Music/mine.mp3" and got[2]["dur"] == 0.0
    assert len(SD.music_favorites([{"source": "openverse", "id": f"t{i}"} for i in range(800)])) == SD.FAV_MAX
    with pytest.raises(ValueError):
        SD.music_favorites("not a list")
    assert SD.clean({"music": {"favorites": raw}})["music"] == {"favorites": got}          # (folders untouched when not given)


# ---------------------------------------------------------------------------------------------- the app's endpoints
@pytest.fixture
def app(tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as W
    h = tmp_path / "FirstCut"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    monkeypatch.setattr(SD, "_answers", lambda url: False)
    monkeypatch.setattr(SD, "write_overlay", lambda root=None: None)
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.cfg = SimpleNamespace(get=lambda k, d=None: d, data={})
    a.runner, a.worker, a.store, a.chatgpt = None, SimpleNamespace(busy_with=None), None, None
    a.root, a.settings, a._cache, a.lock = h, SD.Settings(h), {}, threading.Lock()
    return a


def test_hearting_a_song_puts_it_first_in_favourites_and_unhearting_removes_it(app):
    t1 = {"source": "openverse", "id": "s1", "title": "Sunrise", "artist": "Ola", "dur": 120, "preview": "https://x/1.mp3"}
    t2 = {"source": "mine", "id": "abcdef12", "title": "My song", "path": "/Users/me/Music/My song.mp3"}
    app.music_favorite({"track": t1})
    favs = app.music_favorite({"track": t2})["favorites"]
    assert [(f["source"], f["id"]) for f in favs] == [("library", "abcdef12"), ("openverse", "s1")]
    assert app.music_favorite({"track": t1})["favorites"][0]["id"] == "s1"          # hearted again: first, still once
    assert [f["id"] for f in app.music_favorite({"track": t1, "on": False})["favorites"]] == ["abcdef12"]
    assert app.settings.get()["music"]["favorites"][0]["path"] == "/Users/me/Music/My song.mp3"
    with pytest.raises(ValueError):
        app.music_favorite({"track": {"source": "elsewhere", "id": "1"}})


def test_download_puts_an_online_song_in_downloads_with_its_credit(app, tmp_path, monkeypatch):
    fetched = tmp_path / "FirstCut" / "Music library" / "Online" / "openverse" / "Ola - Sunrise.mp3"
    fetched.parent.mkdir(parents=True)
    fetched.write_bytes(b"ID3" + b"\0" * 5000)
    monkeypatch.setattr(app, "music_fetch", lambda t: {"path": str(fetched), "name": t["title"], "credit": 'Music: "Sunrise" by Ola – CC BY 4.0'})
    t = {"source": "openverse", "id": "s1", "title": "Sunrise", "artist": "Ola", "license": "CC BY 4.0", "page": "https://jamendo.com/s1"}
    r = app.music_download({"track": t})
    dl = tmp_path / "me" / "Downloads"
    assert r["path"] == str(dl / "Ola - Sunrise.mp3") and Path(r["path"]).read_bytes() == fetched.read_bytes()
    credit = (dl / "Ola - Sunrise - credit.txt").read_text()
    assert 'Music: "Sunrise" by Ola' in credit and "Licence: CC BY 4.0" in credit and "https://jamendo.com/s1" in credit
    assert app.music_download({"track": t})["path"] == r["path"]                    # the same song twice: the same file
    fetched.write_bytes(b"ID3" + b"\1" * 6000)                                       # a different file with that name: kept apart
    assert app.music_download({"track": t})["path"] == str(dl / "Ola - Sunrise (2).mp3")


def test_download_copies_your_own_song_only_from_your_music_folders(app, tmp_path):
    folder = tmp_path / "Music"
    folder.mkdir()
    song = folder / "theme.mp3"
    song.write_bytes(b"ID3" + b"\0" * 3000)
    app.settings.update({"music": {"folders": [str(folder)]}})
    app.musiclib = lambda: None
    r = app.music_download({"track": {"source": "library", "id": "x", "title": "Theme", "path": str(song)}})
    assert Path(r["path"]).name == "Theme.mp3" and r["credit"] is None and not list((tmp_path / "me" / "Downloads").glob("*credit*"))
    secret = tmp_path / "elsewhere.mp3"
    secret.write_bytes(b"ID3")
    with pytest.raises(ValueError):                                                   # a path the page made up: refused
        app.music_download({"track": {"source": "library", "id": "x", "path": str(secret)}})


def test_stock_add_joins_the_footage_library_and_download_goes_to_downloads(app, tmp_path, providers, monkeypatch):
    firsts = []
    cat = SimpleNamespace(roots=[], window=None, first=lambda folders, hours=None: firsts.append(list(folders)), enabled=lambda: True)
    app.runner = SimpleNamespace(catalog=cat, running=False, library=lambda: cat)
    timers = []
    monkeypatch.setattr(threading, "Timer", lambda s, fn: timers.append(fn) or SimpleNamespace(cancel=lambda: None, start=lambda: None, daemon=True))
    monkeypatch.setattr(SF.requests, "get", lambda url, **kw: FakeDownload(b"\xff\xd8" + b"\0" * 4000))
    r = app.stock_search("old campus", "photo", 1)
    assert [x["id"] for x in r["results"]] == ["openverse:photo:ov-ok", "openverse:photo:ov-pd"] and r["sources"]["openverse"] and not r["sources"]["pexels"]
    with pytest.raises(ValueError):
        app.stock_action("add", {"id": "openverse:photo:never-shown"})
    got = app.stock_action("add", {"item": {"id": "openverse:photo:ov-ok", "download": "https://evil.example/x"}})   # (the page's URL is ignored)
    stock = tmp_path / "FirstCut" / "Footage library" / "Stock"
    assert Path(got["path"]).parent == stock / "openverse" and Path(got["path"]).is_file()
    assert got["credit"].startswith('"Old campus" by Sam') and json.loads(Path(got["path"]).with_suffix(".json").read_text())["license"] == "CC BY 4.0"
    assert app.settings.get()["library"]["folders"] == [str(stock)] and cat.roots == [stock]
    assert app.settings.get()["library"]["look"] == ""                          # (where Find looks isn't changed by adding stock)
    timers[-1]()
    assert firsts == [[str(stock)]]
    app.stock_action("use", {"id": "openverse:photo:ov-ok"})
    assert app.settings.get()["library"]["folders"] == [str(stock)]             # once
    d = app.stock_action("download", {"id": "openverse:photo:ov-pd"})
    dl = tmp_path / "me" / "Downloads"
    assert Path(d["path"]).parent == dl and Path(d["path"]).name == "Bell tower - openverse ov-pd.jpg"
    assert not (dl / "Bell tower - openverse ov-pd - credit.txt").read_text().startswith("\n")
    assert not list((tmp_path / "FirstCut").glob("stock-*"))                   # the temporary folder is gone
    assert not (stock / "openverse" / "Bell tower - openverse ov-pd.jpg").exists()   # downloading isn't adding to your B-roll
    with pytest.raises(KeyError):
        app.stock_action("delete", {})


def test_stock_search_and_actions_through_the_app_routes(app, providers):
    assert app.post(["music", "favorite"], {"track": {"source": "openverse", "id": "z1", "title": "Z"}})[1]["favorites"][0]["id"] == "z1"
    with pytest.raises(KeyError):
        app.post(["stock", "nope"], {})
    assert app.stock_search("", "", 1)["results"] == []


def test_the_server_answers_stock_search_and_music_pages(tmp_path, monkeypatch, providers):
    import urllib.request
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import musicsources as MS
    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    from clipfarm.clipper.localproj import LocalRunner
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "FirstCut"))
    monkeypatch.setattr(SD, "_answers", lambda url: False)
    pages = []
    monkeypatch.setattr(MS, "openverse_search", lambda q, n=30, page=1: pages.append((q, page)) or [{"source": "openverse", "id": f"p{page}-{i}"} for i in range(12)])
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["app"] = "studio"
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI"), "local": True, "data_dir": "ai", "local_projects": True}
    store = Store(tmp_path / "jobs")
    eng = Engine(cfg, store)
    projects = AV.Projects(cfg, store, eng)
    worker = S.Worker(eng, store, Feed(cfg, store), projects)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, store, worker, worker.feed, 100, projects, LocalRunner(cfg, store, projects)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/app/stock/search?q=old%20campus&kinds=photo&page=1", timeout=10) as r:
            got = json.loads(r.read())
        assert [x["id"] for x in got["results"]] == ["openverse:photo:ov-ok", "openverse:photo:ov-pd"] and got["sources"] == SF.sources()
        with urllib.request.urlopen(base + "/api/app/music/online?source=openverse&q=calm&page=3", timeout=10) as r:
            got = json.loads(r.read())
        assert pages == [("calm", 3)] and got["page"] == 3 and got["more"] is True and len(got["tracks"]) == 12
    finally:
        httpd.shutdown()
