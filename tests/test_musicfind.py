"""Find my music downloads: songs from licensed libraries are found by what their tags and names say, rips are never offered, an import
COPIES, the same song is not offered twice, and "Import new" takes only the sure, not-yet-imported, not-declined songs."""
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from clipfarm.clipper import musicfind as MF
from clipfarm.core import ffmpeg

FF = ffmpeg.bin_path("ffmpeg")


def tone(path: Path, hz=440, tags=None, fmt=None, secs=0.6):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f={hz}:d={secs}"]
    for k, v in (tags or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    if fmt:
        cmd += ["-f", fmt]
    subprocess.run(cmd + [str(path)], check=True)
    return path


@pytest.fixture
def downloads(tmp_path):
    d = tmp_path / "Downloads"
    tone(d / "Ryan Rivers - Affection.wav", 300, {"artist": "Ryan Rivers", "title": "Affection", "copyright": "Licensed for video by Artlist.io"})
    tone(d / "Soundstripe - Open Road.mp3", 350, {"artist": "Mia Hart", "title": "Open Road"})
    tone(d / "Quiet Harbor.mp3", 400, {"title": "Quiet Harbor", "album_artist": "Art-list"})
    tone(d / "Pro Tools Song - Maybe.wav", 450, {"title": "Maybe", "encoded_by": "Pro Tools"})
    tone(d / "Plain song.mp3", 500, {"title": "Plain"})
    tone(d / "New Recording 12.m4a", 550)
    tone(d / "YTDown.com_YouTube_Some-Song.m4a", 600)
    tone(d / "ripped" / "Great Song.mp3", 650, {"title": "Great Song"})               # (a rip by its container brand, below)
    # a rip: major_brand dash in an MP4-family container (what yt-dlp leaves behind)
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=700:d=0.6", "-f", "mp4", "-brand", "dash", "-metadata", "title=Another Song",
                    str(d / "ripped" / "Another Song.m4a")], check=True)
    with zipfile.ZipFile(d / "Artlist Pack.zip", "w") as z:
        z.write(d / "Plain song.mp3", "Artlist/Cool Track.mp3")
        z.write(d / "Plain song.mp3", "Artlist/__MACOSX/._junk.mp3")
    return d


def names(items):
    return sorted(i["name"] for i in items)


def run_scan(downloads, tmp_path, **kw):
    return MF.scan([downloads], state_dir=tmp_path / "state", **kw)


def test_licensed_songs_are_found_by_tag_name_and_zip(downloads, tmp_path):
    r = run_scan(downloads, tmp_path)
    found = {i["name"]: i for i in r["found"]}
    assert set(found) == {"Ryan Rivers - Affection.wav", "Soundstripe - Open Road.mp3", "Quiet Harbor.mp3", "Cool Track.mp3"}
    assert found["Ryan Rivers - Affection.wav"]["source"] == "Artlist" and "Licensed for video by Artlist.io" in found["Ryan Rivers - Affection.wav"]["license_line"]
    assert found["Ryan Rivers - Affection.wav"]["why"] and found["Ryan Rivers - Affection.wav"]["artist"] == "Ryan Rivers"
    assert found["Soundstripe - Open Road.mp3"]["source"] == "Soundstripe"
    assert found["Quiet Harbor.mp3"]["source"] == "Artlist"                               # (album artist "Art-list")
    assert found["Cool Track.mp3"]["from_zip"] and found["Cool Track.mp3"]["member"] == "Artlist/Cool Track.mp3"
    assert all(i["confidence"] == "sure" for i in r["found"])


def test_a_pro_tools_song_with_no_site_is_only_a_maybe(downloads, tmp_path):
    r = run_scan(downloads, tmp_path)
    assert names(r["maybe"]) == ["Pro Tools Song - Maybe.wav"]
    assert r["maybe"][0]["confidence"] == "maybe" and r["maybe"][0]["reason"]


def test_rips_recordings_and_unknown_songs_are_never_offered(downloads, tmp_path):
    r = run_scan(downloads, tmp_path)
    offered = names(r["found"] + r["maybe"])
    for bad in ("YTDown.com_YouTube_Some-Song.m4a", "Another Song.m4a", "New Recording 12.m4a", "Plain song.mp3", "Great Song.mp3"):
        assert bad not in offered, bad
    assert r["skipped"]["youtube"] == 2 and r["skipped"]["recordings"] == 1 and r["skipped"]["other"] >= 2


def test_classify_unit():
    assert MF.classify("a.mp3", {"major_brand": "dash"})["bucket"] == "youtube"
    assert MF.classify("elevenlabs_voice.mp3", {})["bucket"] == "recordings"
    assert MF.classify("x.mp3", {"copyright": "Licensed for video by Artlist.io"})["source"] == "Artlist"
    assert MF.classify("x.mp3", {}, folders="Downloads/Musicbed picks")["source"] == "Musicbed"
    assert MF.classify("Epidemic Sound - Song.mp3", {})["source"] == "Epidemic Sound"
    assert MF.classify("x.mp3", {"comment": "downloaded with youtube-dl", "copyright": "Artlist"})["bucket"] == "youtube"      # (a rip wins)


def test_a_second_scan_uses_the_cache(downloads, tmp_path, monkeypatch):
    run_scan(downloads, tmp_path)
    calls = []
    real = MF.probe_tags
    monkeypatch.setattr(MF, "probe_tags", lambda p, run=subprocess.run: calls.append(p) or real(p, run))
    r = run_scan(downloads, tmp_path)
    assert calls == [] and len(r["found"]) == 4


def test_import_copies_never_moves_and_keeps_a_sidecar(downloads, tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    r = run_scan(downloads, tmp_path)
    res = MF.import_items(r["found"], lib, state)
    assert len(res["imported"]) == 4 and not res["failed"]
    assert (downloads / "Ryan Rivers - Affection.wav").exists() and (downloads / "Artlist Pack.zip").exists()       # originals stay
    dst = lib / "Artlist" / "Ryan Rivers - Affection.wav"
    assert dst.is_file() and (lib / "Artlist" / "Cool Track.mp3").is_file()                                        # (zip member unpacked)
    assert (lib / "Soundstripe" / "Soundstripe - Open Road.mp3").is_file()
    side = json.loads(dst.with_name(dst.name + ".json").read_text())
    assert side["provider"] == "Artlist" and side["credit_line"] == "Affection, Ryan Rivers (licensed via Artlist)" and "Artlist.io" in side["license_line"]
    assert MF.credit_line_for(dst) == "Affection, Ryan Rivers (licensed via Artlist)"
    assert MF.last_imported(state)


def test_the_same_song_is_not_offered_or_imported_twice(downloads, tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    MF.import_items(run_scan(downloads, tmp_path)["found"], lib, state)
    again = run_scan(downloads, tmp_path)
    assert names(again["found"]) == ["Cool Track.mp3"] or again["found"] == []        # (zip members have no fingerprint: only they can come back)
    assert again["skipped"]["already"] == 3
    one = run_scan(downloads, tmp_path)["found"]
    res = MF.import_items([i for i in one if not i["from_zip"]], lib, state)
    assert res["imported"] == []


def test_a_name_clash_gets_a_counter(tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    a = tone(tmp_path / "one" / "Song.mp3", 300, {"copyright": "Artlist"})
    b = tone(tmp_path / "two" / "Song.mp3", 800, {"copyright": "Artlist"})
    for src in (a, b):
        r = MF.scan([src.parent], state_dir=state)
        MF.import_items(r["found"], lib, state)
    assert sorted(p.name for p in (lib / "Artlist").glob("*.mp3")) == ["Song (2).mp3", "Song.mp3"]


def test_duplicate_downloads_are_one_offer(tmp_path):
    d = tmp_path / "Downloads"
    a = tone(d / "Song.mp3", 300, {"copyright": "Artlist"})
    (d / "Song (1).mp3").write_bytes(a.read_bytes())
    r = MF.scan([d], state_dir=tmp_path / "state")
    assert len(r["found"]) == 1 and r["skipped"]["duplicates"] == 1


def test_files_already_in_the_library_folder_are_skipped(downloads, tmp_path):
    lib = downloads / "Mine"
    tone(lib / "Inside.mp3", 300, {"copyright": "Artlist"})
    r = MF.scan([downloads], state_dir=tmp_path / "state", library_roots=[lib])
    assert "Inside.mp3" not in names(r["found"])


def test_depth_is_limited_and_hidden_folders_skipped(tmp_path):
    d = tmp_path / "Downloads"
    tone(d / "a" / "b" / "c" / "d" / "e" / "Too Deep.mp3", 300, {"copyright": "Artlist"})
    tone(d / ".hidden" / "Hidden.mp3", 300, {"copyright": "Artlist"})
    tone(d / "a" / "b" / "Fine.mp3", 300, {"copyright": "Artlist"})
    assert names(MF.scan([d], state_dir=tmp_path / "s")["found"]) == ["Fine.mp3"]


def test_a_time_budget_gives_a_partial_result(downloads, tmp_path):
    r = MF.scan([downloads], state_dir=tmp_path / "state", budget_s=-1)
    assert r["partial"] is True


# ------------------------------------------------------------------------------------------------------------------- Import new
def test_import_new_takes_only_sure_songs_and_never_a_maybe_or_a_rip(downloads, tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    r = run_scan(downloads, tmp_path)
    res = MF.import_new(r, lib, state)
    got = sorted(p["path"].split("/")[-1] for p in res["imported"])
    assert got == ["Cool Track.mp3", "Quiet Harbor.mp3", "Ryan Rivers - Affection.wav", "Soundstripe - Open Road.mp3"]
    assert res["maybes"] == 1 and res["sources"] == {"Artlist": 3, "Soundstripe": 1}
    assert not list(lib.rglob("Pro Tools Song*")) and not list(lib.rglob("*YTDown*")) and not list(lib.rglob("Another Song*"))


def test_declined_songs_never_come_back_even_in_import_new(downloads, tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    r = run_scan(downloads, tmp_path)
    no = [i for i in r["found"] if i["name"] in ("Soundstripe - Open Road.mp3", "Cool Track.mp3")]
    MF.decline(no, state)
    again = run_scan(downloads, tmp_path)
    assert "Soundstripe - Open Road.mp3" not in names(again["found"]) and "Cool Track.mp3" not in names(again["found"])
    assert again["skipped"]["declined"] == 2
    res = MF.import_new(again, lib, state)
    assert sorted(p["title"] for p in res["imported"]) == ["Affection", "Quiet Harbor"]
    MF.undecline(state)
    assert "Soundstripe - Open Road.mp3" in names(run_scan(downloads, tmp_path)["found"])


def test_import_new_with_nothing_new_imports_nothing(downloads, tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    MF.import_new(run_scan(downloads, tmp_path), lib, state)
    again = run_scan(downloads, tmp_path)
    res = MF.import_new({**again, "found": [i for i in again["found"] if not i["from_zip"]]}, lib, state)
    assert res["imported"] == []


def test_a_bad_file_is_reported_and_the_rest_still_import(downloads, tmp_path):
    state, lib = tmp_path / "state", tmp_path / "Library"
    r = run_scan(downloads, tmp_path)
    bad = {**r["found"][0], "path": str(downloads / "gone.mp3"), "fp": None, "name": "gone.mp3"}
    res = MF.import_items([bad] + r["found"][1:], lib, state)
    assert len(res["failed"]) == 1 and len(res["imported"]) == 3


# ------------------------------------------------------------------------------------------------------------------- the app's side
class FakeSettings:
    def __init__(self):
        self.s = {"music": {"folders": []}, "places": {"catalog": "/tmp/x"}}

    def get(self):
        return self.s

    def update(self, ch):
        for k, v in ch.items():
            self.s.setdefault(k, {}).update(v)
        return self.s


def api(tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as SW
    a = SW.StudioAPI.__new__(SW.StudioAPI)
    a.root, a.settings, a._cache = tmp_path / "home", FakeSettings(), {}
    a.musiclib = lambda: None
    a.apply_music = lambda s: None
    monkeypatch.setattr(SW.SD, "write_overlay", lambda root: None)
    monkeypatch.setattr(MF, "DEFAULT_LIBRARY", tmp_path / "Music" / "FirstCut Library")
    return a


def wait(f):
    import time
    for _ in range(300):
        if not f.running():
            return
        time.sleep(0.05)
    raise AssertionError("the scan did not finish")


def test_the_app_scans_allows_playback_and_imports_by_id(downloads, tmp_path, monkeypatch):
    a = api(tmp_path, monkeypatch)
    a.musicfind_start({"folders": [str(downloads)]})
    wait(a._mf())
    st = a.musicfind_state()
    assert st["done"] and len(st["found"]) == 4 and st["n_maybe"] == 1
    song = next(i for i in st["found"] if i["name"] == "Quiet Harbor.mp3")
    assert a.drop_file(song["path"]).name == "Quiet Harbor.mp3"                          # (playable from where it is)
    with pytest.raises(KeyError):
        a.drop_file(str(downloads / "Plain song.mp3"))                                    # (never anything that was not found)
    res = a.musicfind_import({"ids": [song["id"], "not-an-id"]})
    assert [i["title"] for i in res["imported"]] == ["Quiet Harbor"]
    assert (tmp_path / "Music" / "FirstCut Library" / "Artlist" / "Quiet Harbor.mp3").exists()
    assert str(tmp_path / "Music" / "FirstCut Library") in a.settings.s["music"]["folders"]            # (added as a music folder)


def test_import_new_through_the_app_and_decline_by_id(downloads, tmp_path, monkeypatch):
    a = api(tmp_path, monkeypatch)
    a.musicfind_start({"folders": [str(downloads)]})
    wait(a._mf())
    st = a.musicfind_state()
    skip = next(i for i in st["found"] if i["name"] == "Soundstripe - Open Road.mp3")
    a.musicfind_decline({"ids": [skip["id"]]})
    a.musicfind_import_new({"folders": [str(downloads)]})
    wait(a._mf())
    after = a.musicfind_state()["after"]
    assert sorted(i["title"] for i in after["imported"]) == ["Affection", "Cool Track", "Quiet Harbor"] and after["maybes"] == 1
    lib = tmp_path / "Music" / "FirstCut Library"
    assert not list(lib.rglob("Soundstripe - Open Road*")) and not list(lib.rglob("Pro Tools Song*"))
    summary = a.musicfind_summary()
    assert summary["last_imported"] and summary["scanned_before"]


def test_the_summary_counts_new_songs_and_never_imports_by_itself(downloads, tmp_path, monkeypatch):
    a = api(tmp_path, monkeypatch)
    a.settings.s["music"]["watch_downloads"] = True
    import time
    a._mf_auto = time.time()                                                              # (the daily quiet scan already ran: this test starts its own)
    a._mf().start([str(downloads)])
    wait(a._mf())
    s = a.musicfind_summary()
    assert s["n"] == 4 and s["sources"] == {"Artlist": 3, "Soundstripe": 1} and s["watch"] is True
    assert not (tmp_path / "Music").exists()                                              # (a scan alone copies nothing)


def test_a_folder_macos_refuses_is_reported_not_hung(tmp_path, monkeypatch):
    d = tmp_path / "Downloads"
    d.mkdir()
    import os
    real = os.scandir

    def deny(p):
        if str(p) == str(d):
            raise PermissionError("Operation not permitted")
        return real(p)
    monkeypatch.setattr(MF.os, "scandir", deny)
    r = MF.scan([d], state_dir=tmp_path / "s")
    assert r["denied"] == [str(d)] and r["found"] == [] and r["scanned"] == 0


def test_the_finder_says_when_it_is_waiting_for_macos(tmp_path, monkeypatch):
    import threading
    import time
    gate = threading.Event()
    monkeypatch.setattr(MF, "scan", lambda *a, **k: (gate.wait(5), {"found": [], "maybe": [], "skipped": {}, "partial": False, "denied": [], "scanned": 0, "folders": []})[1])
    f = MF.Finder(tmp_path / "s", lambda: [])
    assert f.start()
    assert f.start() is False                                     # (one scan at a time)
    f.started = time.time() - 10                                  # (it has been stuck in the folder for 10 s)
    assert f.state()["waiting"] is True and f.state()["running"] is True
    gate.set()
    f.thread.join(3)
    st = f.state()
    assert st["done"] and not st["waiting"]


def test_import_new_is_labelled_by_the_summary_counts(downloads, tmp_path):
    f = MF.Finder(tmp_path / "state", lambda: [])
    f.start([str(downloads)])
    f.thread.join(30)
    s = f.new_summary()
    assert s["n"] == 4 and s["n_maybe"] == 1 and s["scanned_before"] and s["sources"] == {"Artlist": 3, "Soundstripe": 1}


def test_the_music_routes_reach_the_right_methods(downloads, tmp_path, monkeypatch):
    a = api(tmp_path, monkeypatch)
    assert a.music_action("find", {"folders": [str(downloads)]})["started"] is True
    wait(a._mf())
    st = a.musicfind_state()
    one = st["found"][0]
    assert a.music_action("decline", {"ids": [one["id"]]})["declined_total"] == 1
    assert a.music_action("import", {"ids": [i["id"] for i in st["found"][1:2]]})["imported"]
    assert a.music_action("undecline", {}) == {"ok": True}
    assert a.music_action("import_new", {"folders": [str(downloads)]})["started"] is True
    wait(a._mf())


def test_the_import_new_count_drops_once_songs_are_added(downloads, tmp_path, monkeypatch):
    a = api(tmp_path, monkeypatch)
    a.musicfind_start({"folders": [str(downloads)]})
    wait(a._mf())
    assert a.musicfind_summary()["n"] == 4
    a.musicfind_import_new({"folders": [str(downloads)]})
    wait(a._mf())
    assert a.musicfind_summary()["n"] == 0
