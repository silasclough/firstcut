"""Edges of the footage catalog and the work Video Builder, found by trying to break them: odd file names, broken files, camera frame
rates and timecode in the Final Cut event, vertical and landscape frames, previews past the end, an empty catalog, an unreachable
Ollama, a drive unplugged mid-run, a picture model that can't load, hostile ids and malformed requests to the server, the personal
/ai page staying as it was, the night window, the plain worker loop, the work overlay config, a Claude login that was never made,
and the `vb` command. The picture model and the vision model are faked as in test_catalog (colours stand in for meaning)."""

import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
from test_catalog import FakeEmbedder, FakeVision, _cfg, _photo, _video  # noqa: F401  (the neighbour's fakes)

from clipfarm.clipper import catalog as C
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

REPO = Path(__file__).resolve().parents[1]
FF = ffmpeg.bin_path("ffmpeg")


def _clip(path: Path, color="red", secs=2.0, size="320x180", rate="25", extra=(), audio=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d={secs}:r={rate}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={secs}", "-c:a", "aac", "-shortest"]
    subprocess.run([*cmd, "-c:v", "libx264", "-g", "12", "-pix_fmt", "yuv420p", *extra, str(path)], check=True)


def _xml_root(xml: str):
    return ET.fromstring(xml.split("\n", 2)[2])                       # (past the declaration and the bare DOCTYPE)


def _t(s: str) -> Fraction:
    return Fraction(s.rstrip("s"))


def _clips_inside_their_assets(root) -> None:
    assets = {a.get("id"): a for a in root.findall("./resources/asset")}
    for c in root.findall("./library/event/asset-clip"):
        a = assets[c.get("ref")]
        assert _t(a.get("start")) <= _t(c.get("start")), c.get("name")
        assert _t(c.get("start")) + _t(c.get("duration")) <= _t(a.get("start")) + _t(a.get("duration")), c.get("name")


# ---------------------------------------------------------------------------------------------- files
ODD = ["clip with spaces.mp4", "10:30 interview.mp4", "café ñ 日本語.mp4", "quote's \"double\".mp4", "hash#and?q%20x.mp4", "amp&<lt>.mov",
       "é decomposed.mp4"]


def test_odd_file_names_survive_scan_index_search_and_the_final_cut_event(tmp_path):
    from clipfarm.clipper import fcpxml as FX
    root = tmp_path / "Drive" / "Video"
    for n in ODD:
        _clip(root / "Odd" / n)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    r = cat.index()
    assert r["new"] == len(ODD) and r["failed"] == 0
    shots = cat.search("red", n=50, kind="video")
    assert sorted(Path(s["rel"]).name for s in shots) == sorted(ODD)
    xml = cat.fcpxml(shots, 'Found "footage" & <odd> names')
    root_el = _xml_root(xml)
    assert root_el.find("./library/event").get("name") == 'Found "footage" & <odd> names'
    for a in root_el.findall("./resources/asset"):                      # every media URL leads back to the real file
        assert Path(unquote(urlparse(a.find("media-rep").get("src")).path)).is_file(), a.get("name")
    assert sorted(a.get("name") for a in root_el.findall("./resources/asset")) == sorted(ODD)
    p = tmp_path / "odd.fcpxml"
    p.write_text(xml)
    assert FX.validate(p) == []


def test_broken_files_are_noted_and_skipped_not_the_end_of_the_run(tmp_path):
    root = tmp_path / "Drive" / "Video"
    _clip(root / "good.mp4", secs=3)
    (root / "empty.mp4").write_bytes(b"")                                # (under 1 KB: never even listed)
    (root / "corrupt.mp4").write_bytes(os.urandom(4096))
    _clip(root / "whole.mp4", secs=4)
    data = (root / "whole.mp4").read_bytes()
    (root / "truncated.mp4").write_bytes(data[: len(data) // 2])        # a copy that stopped half way: no moov atom
    (root / "whole.mp4").unlink()
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3", "-c:a", "aac", str(root / "audio_only.mp4")], check=True)
    _clip(root / "blink.mp4", secs=0.3)
    (root / "bad.jpg").write_bytes(os.urandom(5000))                    # a photo neither Pillow nor sips can open
    (root / "notreally.png").write_bytes(b"\x89PNG\r\n\x1a\n" + os.urandom(3000))
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder(), vision=FakeVision())
    r = cat.index()
    assert r["found"] == 7 and r["new"] == 1 and r["failed"] == 6 and r["left"] == 0
    errs = {Path(x["path"]).name: x.get("error") for x in cat.records() if x.get("error")}
    assert set(errs) == {"corrupt.mp4", "truncated.mp4", "audio_only.mp4", "blink.mp4", "bad.jpg", "notreally.png"}
    st = cat.stats()
    assert st["videos"] == 1 and st["photos"] == 0
    assert [Path(s["rel"]).name for s in cat.search("red")] == ["good.mp4"]
    assert cat.describe()["described"] == 1                             # the broken ones are never sent to the vision model
    bad = next(x for x in cat.records() if x["path"].endswith("corrupt.mp4"))
    assert cat.preview(bad["id"], 0, 3) is None


@pytest.mark.parametrize("rate,tc,nominal,frames", [
    ("24000/1001", "01:00:00:00", 24, 86400),                           # 23.976 NDF
    ("30000/1001", "01:00:00;02", 30, 3600 * 30 + 2 - 2 * (60 - 6)),     # 29.97 drop-frame: 2 labels skipped a minute, not every tenth
    ("60000/1001", "10:59:59;58", 60, 39599 * 60 + 58 - 4 * (659 - 65)),  # 59.94 drop-frame
    ("30000/1001", "23:59:59:29", 30, 86399 * 30 + 29),                  # 29.97 NDF at the end of the day
])
def test_camera_rates_and_timecode_land_frame_exact_in_the_final_cut_event(tmp_path, rate, tc, nominal, frames):
    from clipfarm.clipper import fcpxml as FX
    root = tmp_path / "Drive" / "Video"
    _clip(root / "cam.mov", secs=6, size="640x360", rate=rate, extra=["-timecode", tc])
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    rec = next(cat.records())
    assert rec["tc_df"] is (";" in tc)
    shot = {**cat.search("red")[0], "start": 1.3, "end": 4.7}
    root_el = _xml_root(cat.fcpxml([shot]))
    frame = Fraction(1001, nominal * 1000)
    assert root_el.find("./resources/format").get("frameDuration") == f"{frame.numerator}/{frame.denominator}s"
    asset = root_el.find("./resources/asset")
    assert _t(asset.get("start")) == frames * frame                     # where Final Cut expects the clip by its timecode
    clip = root_el.find("./library/event/asset-clip")
    offset = _t(clip.get("start")) - _t(asset.get("start"))
    assert (offset / frame).denominator == 1 and abs(float(offset) - 1.3) < float(frame)
    _clips_inside_their_assets(root_el)
    p = tmp_path / "cam.fcpxml"
    p.write_text(cat.fcpxml([shot]))
    assert FX.validate(p) == []


def test_vertical_and_landscape_frames_keep_their_shape(tmp_path):
    root = tmp_path / "Drive" / "Video"
    _clip(root / "phone.mp4", "blue", secs=1.5, size="1080x1920")
    _clip(root / "camera.mp4", "green", secs=1.5, size="1920x1080", audio=True)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    recs = {Path(r["path"]).name: r for r in cat.records()}
    assert (recs["phone.mp4"]["w"], recs["phone.mp4"]["h"]) == (1080, 1920) and not recs["phone.mp4"]["audio"]
    assert (recs["camera.mp4"]["w"], recs["camera.mp4"]["h"]) == (1920, 1080) and recs["camera.mp4"]["audio"]
    shots = [cat.search("blue")[0], cat.search("green")[0]]
    root_el = _xml_root(cat.fcpxml(shots))
    assert {(f.get("width"), f.get("height")) for f in root_el.findall("./resources/format")} == {("1080", "1920"), ("1920", "1080")}
    with_audio = [a for a in root_el.findall("./resources/asset") if a.get("hasAudio") == "1"]
    assert [a.get("name") for a in with_audio] == ["camera.mp4"]
    for name, portrait in (("phone.mp4", True), ("camera.mp4", False)):
        pv = cat.preview(recs[name]["id"], 0, 1.5)
        info = ffmpeg.probe(pv)
        v = ffmpeg.video_stream(info)
        assert v["height"] == 540 and (v["width"] < v["height"]) is portrait
        assert (ffmpeg.audio_stream(info) is not None) is (name == "camera.mp4")


def test_previews_past_the_end_or_backwards_are_short_real_clips(tmp_path):
    root = tmp_path / "Drive" / "Video"
    _clip(root / "five.mp4", secs=5)
    _photo(root / "still.jpg", (0, 255, 0))
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    vid = next(r for r in cat.records() if r["kind"] == "video")
    photo = next(r for r in cat.records() if r["kind"] == "photo")

    def dur(p):
        return float(ffmpeg.probe(p)["format"]["duration"])
    assert 0.3 < dur(cat.preview(vid["id"], 100, 200)) <= 0.6          # past the end: the last half second
    assert 0.3 < dur(cat.preview(vid["id"], 3, 1)) <= 0.6              # backwards: half a second from the start
    assert dur(cat.preview(vid["id"], -5, 2)) == pytest.approx(2, abs=0.1)
    assert cat.preview(photo["id"], 0, 3) is None and cat.preview("0123456789abcdef", 0, 3) is None
    (root / "five.mp4").unlink()                                        # the source is gone: no preview, no crash, nothing half-written
    assert cat.preview(vid["id"], 1, 2.5) is None
    assert not list((cat.dir / "previews").glob("*.part.mp4"))


def test_validate_finds_final_cuts_dtd_under_another_app_name(tmp_path, monkeypatch):
    """The work Mac has "Final Cut Pro Creator Studio.app", not "Final Cut Pro.app": validate() must still check (or Motion's copy)."""
    import glob

    from clipfarm.clipper import fcpxml as FX
    name = "Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_11.dtd"
    if not (glob.glob(f"/Applications/Final Cut Pro*.app/{name}") + glob.glob(f"/Applications/Motion*.app/{name}")) or not shutil.which("xmllint"):
        pytest.skip("neither Final Cut nor Motion is installed here")
    monkeypatch.setattr(FX, "DTD_DIR", tmp_path / "Final Cut Pro.app (not installed)")
    bad = tmp_path / "bad.fcpxml"
    bad.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n<fcpxml version="1.11"><nonsense/></fcpxml>\n')
    assert FX.validate(bad)                                             # found the DTD elsewhere, so the nonsense is reported


def test_an_empty_catalog_answers_calmly(tmp_path):
    from clipfarm.clipper import fcpxml as FX
    from clipfarm.clipper.moments import Sentence
    cat = C.Catalog(_cfg(tmp_path, [tmp_path / "Empty"]), embedder=FakeEmbedder())
    assert cat.search("red") == [] and cat.stats()["videos"] == 0       # never indexed: no index files at all
    (tmp_path / "Empty").mkdir()
    assert cat.index() == {"found": 0, "new": 0, "failed": 0, "left": 0, "forgotten": 0}
    assert cat.search("red") == [] and cat.stats()["frames"] == 0

    class AI:
        def complete_json(self, task, system, prompt, schema):
            return SimpleNamespace(data={"passages": [{"n": 1, "looks": ["red wall"]}]})
    assert cat.shots_for(AI(), [Sentence(0, 0.0, 5.0, "Hello there.")]) == []
    assert cat.shots_for(AI(), []) == []
    p = tmp_path / "none.fcpxml"
    p.write_text(cat.fcpxml([]))
    assert FX.validate(p) == [] and _xml_root(p.read_text()).find("./library/event") is not None


def test_an_unreachable_ollama_leaves_everything_to_try_again(tmp_path, drive):
    from clipfarm.clipper.moments import Sentence

    class Down:
        def complete_json(self, *a, **k):
            raise ConnectionRefusedError("[Errno 61] Connection refused (localhost:11434)")
    cat = C.Catalog(_cfg(tmp_path, [drive], min_score=0.9), embedder=FakeEmbedder(), vision=Down())
    cat.index()
    assert cat.describe() == {"described": 0, "left": 3}
    assert not any(r.get("described") for r in cat.records())

    class AI:
        def complete_json(self, task, system, prompt, schema):
            return SimpleNamespace(data={"passages": [{"n": 1, "looks": ["red gym wall"]}]})
    shots = cat.shots_for(AI(), [Sentence(0, 0.0, 6.0, "We trained in the gym.")])
    assert shots and shots[0]["what"].startswith("(found searching for 'red gym wall')")   # the search words stand in

    class Broken:
        def complete_json(self, *a, **k):
            raise TimeoutError("the AI took too long")
    assert cat.shots_for(Broken(), [Sentence(0, 0.0, 6.0, "We trained in the gym.")]) == []
    cat._vision = FakeVision()
    assert cat.describe()["described"] == 3                              # back up: everything is described now


@pytest.fixture
def drive(tmp_path):
    root = tmp_path / "Drive" / "Video"
    _video(root / "B-roll" / "Athletics" / "red_gym.mp4", "red")
    _video(root / "B-roll" / "Student Life" / "blue_sky.mp4", "blue", secs=12)
    _photo(root / "Photos" / "green_lawn.jpg", (0, 255, 0))
    return root


# ---------------------------------------------------------------------------------------------- failures that must not stick
def test_a_drive_unplugged_mid_run_is_catalogued_when_it_comes_back(tmp_path):
    """The work drive is a network share: if it drops out half way through a night, the files not reached must be looked at again
    once it is back - not remembered for ever as unreadable ("No such file or directory") and left out of every search."""
    root = tmp_path / "Drive" / "Video"
    for k in range(5):
        _clip(root / f"clip{k}.mp4", secs=2)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.workers = 1                                                      # (one file at a time: the unplug is timed by the progress calls)
    away = tmp_path / "Drive" / "Video-unplugged"

    def unplug(k, n, what):
        if k == 2 and root.exists():
            root.rename(away)
    cat.index(progress=unplug)
    away.rename(root)
    r = cat.index()
    assert r["new"] == 3, r
    assert cat.stats()["videos"] == 5


def test_a_picture_model_that_cannot_load_does_not_mark_the_drive_as_unreadable(tmp_path):
    """SigLIP not downloaded yet (no network the first night), transformers missing, or the GPU out of memory: that is not the files'
    fault. Once the model works, every file must still be catalogued."""
    root = tmp_path / "Drive" / "Video"
    for k in range(3):
        _clip(root / f"c{k}.mp4", secs=2)

    class NoModel:
        def images(self, ims):
            raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=NoModel())
    cat.index()
    cat._embedder = FakeEmbedder()
    r = cat.index()
    assert r["new"] == 3, r
    assert cat.stats()["videos"] == 3


def test_a_long_gop_export_is_catalogued_to_its_end(tmp_path):
    """An export or screen recording (x264's default: a keyframe every 250 frames, plus one at each scene change) must be looked at
    all the way through. Today only keyframes are decoded and the last keyframe never becomes a frame: the blue second half of this
    clip is not in the catalog at all, and searching 'blue' offers the red first half."""
    root = tmp_path / "Drive" / "Video"
    root.mkdir(parents=True)
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x180:d=6:r=25", "-f", "lavfi", "-i",
                    "color=c=blue:s=320x180:d=6:r=25", "-filter_complex", "[0][1]concat=n=2:v=1[v]", "-map", "[v]", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(root / "clip.mp4")], check=True)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    rec = next(cat.records())
    from PIL import Image
    colours = [Image.open(cat.thumb(rec["id"], k)).convert("RGB").resize((1, 1)).getpixel((0, 0)) for k in range(rec["frames"])]
    assert any(b > 200 and r < 60 for r, g, b in colours), colours
    assert cat.search("blue")[0]["start"] >= 5.0


def test_library_b_roll_that_fails_never_fails_the_video(tmp_path, drive):
    """Library B-roll is a bonus (like the photos): if the picture model fails while the edit is searching the library (GPU out of
    memory next to the vision and text models), the video is made from its own footage instead of failing."""
    from clipfarm.clipper import localproj as LP
    cfg = _cfg(tmp_path, [drive], min_score=0.9)
    cat = C.Catalog(cfg, embedder=FakeEmbedder(), vision=FakeVision())
    cat.index()

    class OOM:
        def text(self, texts):
            raise RuntimeError("MPS backend out of memory")
    cat._embedder = OOM()
    runner = LP.LocalRunner(cfg, None, None)
    runner.catalog = cat

    class AI:
        def complete_json(self, task, system, prompt, schema):
            return SimpleNamespace(data={"passages": [{"n": 1, "looks": ["red gym"]}]})
    files = [{"name": "talk.mp4", "path": "/x/talk.mp4", "start": 0.0, "end": 60.0, "dur": 60.0, "w": 1920, "h": 1080, "fps": 25.0,
              "audio": True, "broll": False}]
    words = [{"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate("we trained in the gym every morning.".split())]
    out, scenes = runner.add_library(files, words, {}, AI(), LP.State(tmp_path / "proj"), "brief", lambda f, m: None)
    assert out == files and scenes == {}


# ---------------------------------------------------------------------------------------------- the server
def _server(tmp_path, cfg, catalog=None):
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    from clipfarm.clipper.localproj import LocalRunner
    from clipfarm.clipper.store import Store
    store = Store(tmp_path / "jobs")
    eng = Engine(cfg, store)
    projects = AV.Projects(cfg, store, eng)
    runner = LocalRunner(cfg, store, projects)
    runner.catalog = catalog
    worker = S.Worker(eng, store, Feed(cfg, store), projects)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, store, worker, worker.feed, 100, projects, runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _call(base, path, body=None):
    """(status, body); 'dropped' when the server closed the connection without an answer."""
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (http.client.RemoteDisconnected, ConnectionResetError, urllib.error.URLError):
        return "dropped", b""


def _app_cfg(tmp_path, roots):
    cfg = _cfg(tmp_path, roots)
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI"), "local": True, "data_dir": "ai", "local_projects": True,
                                        "out_dir": str(tmp_path / "Saves")}
    cfg.data["clipper"]["app"] = "work"
    return cfg


def test_the_server_refuses_hostile_ids_and_names(tmp_path, drive):
    cfg = _app_cfg(tmp_path, [drive])
    cat = C.Catalog(cfg, embedder=FakeEmbedder())
    cat.index()
    (drive / "B-roll" / "broken.mp4").write_bytes(os.urandom(4096))
    cat.index()
    vid = next(r for r in cat.records() if r["path"].endswith("red_gym.mp4"))
    broken = next(r for r in cat.records() if r.get("error"))
    httpd, base = _server(tmp_path, cfg, cat)
    try:
        for bad in (f"/api/library/media/{vid['id'].upper()}", f"/api/library/media/{broken['id']}", "/api/library/media/..%2F..%2Fetc%2Fpasswd",
                    "/api/library/../../etc/passwd", f"/api/library/media/{vid['id']}/../../x", f"/api/library/thumb/{vid['id']}/..%2F..%2Findex.json",
                    "/api/library/thumb/..%2Ffiles/0", f"/api/library/preview/{vid['id'][:15]}?s=0&e=3"):
            assert _call(base, bad)[0] == 404, bad
        st, body = _call(base, "/api/library/fcpxml", {"shots": [{"id": vid["id"], "start": 1, "end": 4}], "name": "../../../../tmp/evil"})
        path = Path(json.loads(body)["path"])
        assert st == 200 and path.parent == tmp_path / "Saves" / "Found footage" and path.is_file()
        for shots in ([{"id": broken["id"], "start": 0, "end": 3}], [{"id": vid["id"], "start": 6.8, "end": 99}], [{"id": "../../x"}], []):
            assert _call(base, "/api/library/fcpxml", {"shots": shots})[0] == 400, shots
        assert _call(base, "/api/library/reveal", {"id": "../../etc"})[0] == 404
        st, body = _call(base, "/api/library/search?q=red&n=99999")
        assert st == 200 and len(json.loads(body)["shots"]) <= 120 and all("path" not in s for s in json.loads(body)["shots"])
    finally:
        httpd.shutdown()


def test_malformed_requests_get_an_answer_not_a_dropped_connection(tmp_path, drive):
    """The page shows the server's error text; a closed connection only says 'Load failed'. Today a bad number, an odd frame
    number, a non-object shot or a picture model that can't load (the comment in serve() promises 'search says why') all close
    the connection without any HTTP answer."""
    cfg = _app_cfg(tmp_path, [drive])
    cat = C.Catalog(cfg, embedder=FakeEmbedder())
    cat.index()
    vid = next(r for r in cat.records() if r["kind"] == "video")
    httpd, base = _server(tmp_path, cfg, cat)
    try:
        got = {
            "n=abc": _call(base, "/api/library/search?q=red&n=abc")[0],
            "thumb ²": _call(base, f"/api/library/thumb/{vid['id']}/%C2%B2")[0],
            "shots as text": _call(base, "/api/library/fcpxml", {"shots": "abc"})[0],
            "a shot that is a number": _call(base, "/api/library/fcpxml", {"shots": [123]})[0],
        }

        class NoModel:
            def text(self, t):
                raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")
        cat._embedder = NoModel()
        st, body = _call(base, "/api/library/search?q=red")
        got["search without the picture model"] = st
        assert all(isinstance(v, int) and 400 <= v < 600 for v in got.values()), got
        assert b"huggingface" in body or b"picture" in body or b"model" in body
    finally:
        httpd.shutdown()


def test_the_personal_ai_page_is_unchanged_without_clipper_app(tmp_path):
    from clipfarm.clipper.server import WEB
    cfg = _cfg(tmp_path, [])
    cfg.data["clipper"].pop("app", None)
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI"), "local": True, "data_dir": "ai", "local_projects": True}
    httpd, base = _server(tmp_path, cfg, None)
    try:
        assert _call(base, "/")[1] == (WEB / "index.html").read_bytes()               # My Clips stays the home page
        st, html = _call(base, "/ai")
        html = html.decode()
        assert st == 200 and html == (WEB / "ai.html").read_text()
        for el in ('id="tab-up"', 'id="usage"', 'id="chatgpt-plan"', 'id="l-tr"', "Upload videos", 'href="/feed"'):
            assert el in html, el
        assert "body.app .personal { display: none !important; }" in html and "body:not(.app) .app-only { display: none !important; }" in html
        for el_id in ("app-tabs", "ai-who", "l-lib-wrap"):                             # the app's controls are all marked app-only
            tag = re.search(rf'<[^>]*id="{el_id}"[^>]*>', html).group(0)
            assert "app-only" in tag, tag
        d = json.loads(_call(base, "/api/ai/projects")[1])
        assert not d["app"] and d["library"] is None
        assert _call(base, "/api/library/status")[0] == 404 and _call(base, "/api/library/fcpxml", {"shots": []})[0] == 404
        if shutil.which("node"):                                                        # the page's script parses
            js = tmp_path / "ai.mjs"
            js.write_text("\n".join(re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)))
            r = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------------------------- running it
class _At(datetime):
    when = datetime(2026, 11, 5, 23, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.when


def test_tick_only_catalogues_inside_the_night_window(tmp_path, monkeypatch):
    (tmp_path / "Drive").mkdir()
    cat = C.Catalog(_cfg(tmp_path, [tmp_path / "Drive"], window=["19:00", "07:00"]), embedder=FakeEmbedder())
    started = []
    cat.start = lambda hours=None, **k: started.append(hours) or True
    monkeypatch.setattr(C, "datetime", _At)
    _At.when = _At(2026, 11, 5, 12, 0)
    cat.tick()
    assert started == []                                                # midday: no
    _At.when = _At(2026, 11, 5, 23, 0)
    cat.tick(busy=True)
    assert started == []                                                # a video is being made: no
    cat.tick()
    assert started == [pytest.approx(8.0)]                              # 23:00: until 07:00 tomorrow
    cat.status["finished"] = __import__("time").time()
    cat.tick()
    assert len(started) == 1                                            # just finished: not again within half an hour
    gone = C.Catalog(_cfg(tmp_path, [tmp_path / "Unplugged"], window=["19:00", "07:00"]), embedder=FakeEmbedder())
    gone.start = lambda **k: pytest.fail("nothing to catalogue on a drive that is not plugged in")
    gone.tick()


def test_the_plain_worker_loop_never_touches_the_feed_or_the_relay(tmp_path):
    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    from clipfarm.clipper.store import Store
    cfg = _cfg(tmp_path, [])
    cfg.data["clipper"]["background"] = False
    store = Store(tmp_path / "jobs")
    worker = S.Worker(Engine(cfg, store), store, Feed(cfg, store), None)
    worker._relay = lambda: pytest.fail("no relay in the work app")
    worker.feed.sweep = lambda: pytest.fail("no feed in the work app")
    worker._drip = lambda: pytest.fail("no drip in the work app")
    ran = []

    def boom():
        raise RuntimeError("a side job failed")

    def stop():
        raise SystemExit
    worker.side_jobs += [lambda: ran.append(1), boom, lambda: ran.append(2), stop]
    with pytest.raises(SystemExit):
        worker.run()
    assert ran == [1, 2]                                                # a failing side job is logged, the next one still runs


def test_the_work_overlay_config_applies_from_a_relative_path(monkeypatch):
    monkeypatch.setenv("CLIPFARM_OVERLAY", "config/work.yaml")
    cfg = load_config()
    assert cfg.get("clipper.app") == "work" and cfg.get("clipper.background") is False
    assert cfg.get("clipper.ai_videos.openai.enabled") is False and cfg.get("catalog.window") == ["19:00", "07:00"]
    assert cfg.get("catalog.dir") == "data/work/catalog" and cfg.get("clipper.data_dir") == "data/work/clipper"
    assert cfg.get("catalog.frame_every_s") == 3.0                     # the defaults from config.yaml are still there
    monkeypatch.delenv("CLIPFARM_OVERLAY")
    assert load_config().get("catalog.roots") in ([], None)            # without it: no catalog on a personal machine


def test_two_waiting_videos_with_the_same_title_never_share_a_save_folder(tmp_path):
    """Make the video while another one is being made: the project is created (an empty <title>/ folder), the start is refused
    ('another local video is being made'), and pressing the button again later makes a second project with the same title. Both
    then write .aivideo/, Output/ and the .fcpxml into the same folder."""
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper.store import Store
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI videos"), "local": True, "data_dir": "ai"}
    projects = AV.Projects(cfg, Store(tmp_path / "jobs"), None)
    footage = tmp_path / "Interview"
    footage.mkdir()
    a = projects.create_local(str(footage), "", "", {}, {}, out_dir=str(tmp_path / "Saves"))
    b = projects.create_local(str(footage), "", "", {}, {}, out_dir=str(tmp_path / "Saves"))
    assert a["home"] != b["home"]


def test_a_claude_login_that_was_never_made_is_not_retried_on_every_call(monkeypatch, tmp_path):
    """Until someone signs the work account in, `claude -p` answers {"is_error": true, "result": "Not logged in · Please run /login"}
    (seen on this Mac with an empty CLAUDE_CONFIG_DIR). Each AI call of a video then tries Claude four times with 3+6+9 s of
    waiting before the local model answers - dozens of calls per video - while the page's pill says 'Claude (your plan)'."""
    from clipfarm.clipper import localai as LA
    from clipfarm.core import llm as L
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "vision": "ollama", "fallback": "ollama", "data_dir": "ai"})
    cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "skip_tasks": ["broll_describe"], "fallback": True})
    monkeypatch.setattr(L.OllamaLLM, "complete_json", lambda self, *a, **k: L.LLMResult({"ok": True}, "ollama", self.cfg["model"], {}))
    from clipfarm.clipper.claude_cli import ClaudeCLI
    monkeypatch.setattr(ClaudeCLI, "available", lambda self: self.enabled and __import__("time").time() >= self.tripped_until)
    runs, sleeps = [], []
    monkeypatch.setattr(LA.time, "sleep", lambda s: sleeps.append(s))
    ai = LA.LocalAI(cfg)

    def not_logged_in(cmd, **k):
        runs.append(cmd)
        return subprocess.CompletedProcess(cmd, 1, json.dumps({"type": "result", "is_error": True, "result": "Not logged in · Please run /login"}).encode(), b"")
    ai.claude.run = not_logged_in
    assert ai.complete_json("story_plan", "s", "p", {}).data == {"ok": True}
    assert ai.complete_json("broll_plan", "s", "p", {}).data == {"ok": True}
    assert len(runs) <= 2 and sum(sleeps) < 3, (len(runs), sleeps)


def test_vb_talks_to_the_server_and_says_what_went_wrong(tmp_path, drive, monkeypatch):
    # (vb now talks to the Video Builder app's engine, clipper.app = "studio", rewritten 2026-11-05)
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path / "VBHome"))
    cfg = _app_cfg(tmp_path, [drive])
    cfg.data["clipper"]["app"] = "studio"
    cat = C.Catalog(cfg, embedder=FakeEmbedder())
    cat.index()
    httpd, base = _server(tmp_path, cfg, cat)
    vb = REPO / "handoff" / "videobuilder" / "bin" / "vb"
    env = {**os.environ, "VIDEOBUILDER_URL": base}

    def run(*args):
        return subprocess.run([sys.executable, "-I", str(vb), *args], capture_output=True, text=True, env=env, timeout=60)
    try:
        r = run("search", "red", "wall", "-n", "3")
        assert r.returncode == 0 and "red_gym.mp4" in r.stdout.splitlines()[0]
        r = run("search", "--photos", "green")
        assert r.returncode == 0 and "photo" in r.stdout and "green_lawn.jpg" in r.stdout and ".mp4" not in r.stdout
        r = run("status")
        assert r.returncode == 0 and "2 videos, 1 photo" in r.stdout and "ok " in r.stdout and "AI for edit decisions" in r.stdout
        r = run("make", str(tmp_path / "no such folder"), "--notes", "x")
        assert r.returncode != 0 and "choose the folder with your footage" in r.stderr
        r = run("clips", str(tmp_path / "notes.txt"))
        assert r.returncode != 0 and "choose a video" in r.stderr
        r = subprocess.run([sys.executable, "-I", str(vb), "status"], capture_output=True, text=True, timeout=60,
                           env={**os.environ, "VIDEOBUILDER_URL": "http://127.0.0.1:9"})
        assert r.returncode != 0 and "not answering" in r.stderr
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------------------------- footage sources (Find footage's "In" menu)
# Added by the tester 2026-11-05. Tests marked xfail(strict=True) record a real defect: they XPASS (and fail the run) the day it is fixed,
# so remove the mark then.
class _SlowEmbedder(FakeEmbedder):
    """Takes a moment per file, so a run is still going when the next request comes in."""
    def __init__(self, secs=0.4):
        self.secs = secs

    def images(self, ims):
        import time
        time.sleep(self.secs)
        return super().images(ims)


def _settle(cat, secs=60):
    """Wait until no catalogue run (and no first() hand-over thread) is left."""
    import time
    end = time.time() + secs
    while time.time() < end:
        busy = [t for t in threading.enumerate() if t.name in ("catalog", "catalog-first") and t.is_alive()]
        if not busy:
            return
        time.sleep(0.1)
    raise AssertionError("the catalogue run did not finish")


def _per_root(cat):
    return {Path(r["path"]).name: r["videos"] for r in cat.stats()["roots"]}


def test_sources_whose_names_share_a_prefix_are_kept_apart(tmp_path):
    a, b = tmp_path / "X" / "B-roll", tmp_path / "X" / "B-roll 2"
    _clip(a / "red.mp4", "red")
    _clip(b / "blue.mp4", "blue")
    cat = C.Catalog(_cfg(tmp_path, [a, b]), embedder=FakeEmbedder())
    cat.index()
    assert _per_root(cat) == {"B-roll": 1, "B-roll 2": 1}                       # (a plain startswith would give B-roll 2 files)
    for under, want in ((str(a), "red.mp4"), (str(a) + "/", "red.mp4"), (str(b), "blue.mp4")):
        assert [Path(h["path"]).name for h in cat.search("red blue", under=under)] == [want], under
    assert cat.search("red", under=str(tmp_path / "X" / "B-roll 3")) == []
    assert not C.Catalog.under(str(b / "blue.mp4"), a) and C.Catalog.under(str(b / "blue.mp4"), b)
    assert {Path(h["path"]).name for h in cat.search("red", under="B-roll 2")} == {"blue.mp4"}   # (not absolute: a part of the name)


def test_a_source_inside_another_counts_in_both_and_each_file_is_catalogued_once(tmp_path):
    top = tmp_path / "Footage"
    inner = top / "Weddings"
    _clip(top / "red.mp4", "red")
    _clip(inner / "green.mp4", "green")
    cat = C.Catalog(_cfg(tmp_path, [top, inner]), embedder=FakeEmbedder())
    assert len(cat.scan()) == 2
    assert cat.start(only=[inner], rest=False, describe=False)
    cat.thread.join(60)
    assert _per_root(cat) == {"Footage": 1, "Weddings": 1}                      # the inner one first, alone
    assert cat.start(describe=False)
    cat.thread.join(60)
    assert _per_root(cat) == {"Footage": 2, "Weddings": 1} and len(list(cat.records())) == 2
    assert [Path(h["path"]).name for h in cat.search("green", under=str(inner))] == ["green.mp4"]


def test_a_source_inside_a_skipped_folder_is_not_forgotten_by_the_next_run(tmp_path):
    top = tmp_path / "Movies"
    exports = top / "Exports"
    _clip(top / "raw" / "red.mp4", "red")
    _clip(exports / "green.mp4", "green")
    cat = C.Catalog(_cfg(tmp_path, [top, exports]), embedder=FakeEmbedder())
    assert cat.start(only=[exports], rest=False, describe=False)
    cat.thread.join(60)
    assert _per_root(cat)["Exports"] == 1
    assert cat.start(describe=False)
    cat.thread.join(60)
    assert cat.status["last_index"]["forgotten"] == 0 and _per_root(cat)["Exports"] == 1


def test_catalogue_it_now_during_a_run_stops_it_runs_that_folder_and_resets_the_scope(tmp_path):
    a, b = tmp_path / "A", tmp_path / "B"
    for k in range(4):
        _clip(a / f"red{k}.mp4", "red")
    _clip(b / "blue.mp4", "blue")
    cat = C.Catalog(_cfg(tmp_path, [a, b]), embedder=_SlowEmbedder(0.5), vision=FakeVision())   # (no window: daytime rules)
    assert cat.start(only=[a], rest=False, describe=False)
    import time
    time.sleep(0.8)
    assert cat.status["running"] and cat.status["scope"] == [str(a)]
    cat.first([b])
    _settle(cat)
    st = cat.stats()
    assert _per_root(cat)["B"] == 1 and _per_root(cat)["A"] == 4                # A was cut short, so it comes along: both done
    assert cat.only is None and st["status"]["scope"] is None and not st["status"]["running"]
    assert not any(r["cataloguing"] for r in st["roots"])
    assert cat.start(describe=False)                                             # nothing is stuck: a later run still starts
    _settle(cat)
    assert _per_root(cat) == {"A": 4, "B": 1}


def test_two_catalogue_now_requests_in_a_row_both_get_catalogued_one_run_at_a_time(tmp_path):
    s1, s2, s3 = tmp_path / "S1", tmp_path / "S2", tmp_path / "S3"
    for s, c in ((s1, "red"), (s2, "green"), (s3, "blue")):
        for k in range(2):
            _clip(s / f"{c}{k}.mp4", c)
    cat = C.Catalog(_cfg(tmp_path, [s1, s2, s3]), embedder=_SlowEmbedder(0.4), vision=FakeVision())
    active, most, lock = [0], [0], threading.Lock()
    real_index = cat.index

    def index(*a, **k):                                                          # how many runs are inside index() at once
        with lock:
            active[0] += 1
            most[0] = max(most[0], active[0])
        try:
            return real_index(*a, **k)
        finally:
            with lock:
                active[0] -= 1
    cat.index = index
    assert cat.start(only=[s1], rest=False, describe=False)
    import time
    time.sleep(0.5)
    cat.first([s2])
    cat.first([s3])
    _settle(cat)
    per = _per_root(cat)
    assert most[0] == 1 and per["S2"] == 2 and per["S3"] == 2 and not cat.status.get("error"), (most[0], per, cat.status.get("error"))


class _Night(datetime):
    when = datetime(2026, 11, 5, 23, 0)

    @classmethod
    def now(cls, tz=None):
        return cls.when


def _first_args(cat, monkeypatch, when):
    import time
    seen = []
    monkeypatch.setattr(cat, "start", lambda **k: seen.append(k) or True)
    monkeypatch.setattr(C, "datetime", _Night)
    _Night.when = when
    cat.first([str(cat.roots[0])])
    end = time.time() + 10
    while not seen and time.time() < end:
        time.sleep(0.02)
    return seen[0]


def test_catalogue_it_now_in_the_daytime_ends_with_that_folder_and_at_night_goes_on(tmp_path, monkeypatch):
    (tmp_path / "D").mkdir()
    night = C.Catalog(_cfg(tmp_path, [tmp_path / "D"], window=["19:00", "07:00"]), embedder=FakeEmbedder())
    assert _first_args(night, monkeypatch, _Night(2026, 11, 5, 12, 0))["rest"] is False
    assert _first_args(night, monkeypatch, _Night(2026, 11, 5, 23, 0))["rest"] is True
    off = C.Catalog(_cfg(tmp_path, [tmp_path / "D"], window=None), embedder=FakeEmbedder())     # night cataloguing switched off
    assert _first_args(off, monkeypatch, _Night(2026, 11, 5, 23, 0))["rest"] is False


def test_catalogue_it_now_at_night_stops_at_the_end_of_the_window(tmp_path, monkeypatch):
    (tmp_path / "D").mkdir()
    cat = C.Catalog(_cfg(tmp_path, [tmp_path / "D"], window=["19:00", "07:00"]), embedder=FakeEmbedder())
    k = _first_args(cat, monkeypatch, _Night(2026, 11, 5, 23, 0))
    assert k["rest"] is True and k["hours"] is not None and k["hours"] <= 8.01


def test_a_removed_source_is_no_longer_searched_or_counted(tmp_path):
    a, b = tmp_path / "Keep", tmp_path / "Removed"
    _clip(a / "red.mp4", "red")
    _clip(b / "blue.mp4", "blue")
    cat = C.Catalog(_cfg(tmp_path, [a, b]), embedder=FakeEmbedder())
    cat.index()
    cat.roots = [a]                                                              # what StudioAPI.apply_library does when it is removed
    assert all(Path(h["path"]).parent == a for h in cat.search("blue")), [h["rel"] for h in cat.search("blue")]
    assert cat.stats()["videos"] == 1
