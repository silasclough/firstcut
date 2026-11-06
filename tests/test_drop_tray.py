"""Dropping files into the window: the tab you are on decides where they go (never the file's shape), what a dropped path is (probe), the
little thumbnails and the preview file (only for paths that were dropped or chosen), and several videos making several sets of clips."""
import json
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from clipfarm.core import ffmpeg
from tests.test_studio_edges import _api, _call, _studio_server, home  # noqa: F401  (the fixtures and helpers the other app tests use)

REPO = Path(__file__).resolve().parents[1]
FF = ffmpeg.bin_path("ffmpeg")


def _video(path: Path, w=320, h=180, secs=2, audio=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FF, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s={w}x{h}:d={secs}:r=25"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=f=440:d={secs}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"] + (["-c:a", "aac", "-shortest"] if audio else []) + [str(path)]
    subprocess.run(cmd, check=True)
    return path


# ------------------------------------------------------------------------------------------------ where a drop goes (the page's own function, run by node)
NODE = shutil.which("node")


def _plans(cases):
    src = (REPO / "clipfarm/clipper/web/studio.html").read_text()
    a = src.index("function dropPlan(c)")
    b = src.index("async function dropInfo(p)")
    js = src[a:b] + "\nconst cases = " + json.dumps(cases) + ";\nconsole.log(JSON.stringify(cases.map(dropPlan)));"
    r = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


@pytest.mark.skipif(not NODE, reason="node is not installed")
def test_the_tab_you_are_on_decides_never_the_files_shape():
    cases = [
        {"page": "new", "arg": "short", "mode": "long", "kind": "video"},             # a vertical Photos clip on Short form: stays short
        {"page": "new", "arg": "long", "mode": "short", "kind": "video"},             # the same clip on Long form: long footage
        {"page": "home", "arg": "", "mode": "short", "kind": "video"},                # no composer: the last mode you used
        {"page": "home", "arg": "", "mode": "long", "kind": "video"},
        {"page": "short", "arg": "abc", "mode": "long", "kind": "video"},             # on a short project's page: short
        {"page": "long", "arg": "abc", "mode": "short", "kind": "video"},
        {"page": "settings", "arg": "ai", "mode": "long", "kind": "video"},
        {"page": "find", "arg": "", "mode": "long", "kind": "video"},                 # Find footage ignores drops, as it did
    ]
    out = [p.get("act") + ":" + str(p.get("mode")) for p in _plans(cases)]
    assert out == ["add:short", "add:long", "add:short", "add:long", "add:short", "add:long", "add:long", "ignore:None"]


@pytest.mark.skipif(not NODE, reason="node is not installed")
def test_photos_other_kinds_and_special_modes():
    cases = [
        {"page": "new", "arg": "short", "mode": "short", "kind": "image"},            # the preview JPEG: a toast, never a tab change
        {"page": "new", "arg": "long", "mode": "long", "kind": "image"},
        {"page": "new", "arg": "short", "mode": "short", "kind": "folder"},
        {"page": "new", "arg": "long", "mode": "long", "kind": "folder"},
        {"page": "new", "arg": "short", "mode": "short", "kind": "link"},
        {"page": "new", "arg": "long", "mode": "long", "kind": "link"},
        {"page": "new", "arg": "short", "mode": "short", "kind": "audio"},
        {"page": "new", "arg": "short", "mode": "short", "kind": "other"},
        {"page": "new", "arg": "long", "mode": "long", "longMode": "transcript", "kind": "video"},
        {"page": "new", "arg": "long", "mode": "long", "longMode": "transcript", "kind": "image"},
        {"page": "new", "arg": "long", "mode": "long", "longMode": "broll", "kind": "video"},
        {"page": "new", "arg": "long", "mode": "long", "longMode": "broll", "kind": "folder"},
    ]
    got = _plans(cases)
    acts = [p["act"] for p in got]
    assert acts == ["toast", "toast", "toast", "add-folder", "add", "toast", "song", "toast", "transcript", "toast", "broll-video", "broll-folder"]
    assert "photo" in got[0]["toast"] and "photo" in got[1]["toast"]
    assert got[3]["mode"] == "long" and got[4]["mode"] == "short"


# ------------------------------------------------------------------------------------------------ probe / thumb / localfile
def test_probe_says_what_a_path_is(home, tmp_path):
    api = _api(home, tmp_path)
    mp4 = _video(tmp_path / "Photos export" / "IMG 0001.MOV", 180, 320, 2)
    info = api.drop_probe(str(mp4))
    assert info["kind"] == "video" and info["w"] == 180 and info["h"] == 320 and 1.5 < info["duration"] < 2.6 and info["audio"] is True
    nameless = tmp_path / "Photos export" / "2F3A9C"                                 # no extension at all
    shutil.copy(mp4, nameless)
    assert api.drop_probe(str(nameless))["kind"] == "video"
    jpg = tmp_path / "p.jpg"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=64x48:d=1", "-frames:v", "1", str(jpg)], check=True)
    assert api.drop_probe(str(jpg))["kind"] == "image"
    nameless_jpg = tmp_path / "7D"
    shutil.copy(jpg, nameless_jpg)
    assert api.drop_probe(str(nameless_jpg))["kind"] == "image"
    wav = tmp_path / "s.wav"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=1", str(wav)], check=True)
    assert api.drop_probe(str(wav))["kind"] == "audio"
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    assert api.drop_probe(str(txt))["kind"] == "other"
    d = api.drop_probe(str(tmp_path))
    assert d["kind"] == "folder" and set(d) == {"kind", "name", "path"}                # (a folder says only that it is one)
    for bad in ("", "relative/clip.mp4", str(tmp_path / "missing.mp4")):
        with pytest.raises(ValueError):
            api.drop_probe(bad)


def test_only_paths_that_were_dropped_are_served_back(home, tmp_path):
    api = _api(home, tmp_path)
    mp4 = _video(tmp_path / "a.mp4")
    secret = _video(tmp_path / "other" / "secret.mp4")
    api.drop_probe(str(mp4))
    assert api.drop_file(str(mp4)) == mp4.resolve()
    with pytest.raises(KeyError):
        api.drop_file(str(secret))                                                    # never probed: never served
    with pytest.raises(KeyError):
        api.drop_thumb(str(secret))
    for tricky in (str(tmp_path / "other" / ".." / "a.mp4x"), "/etc/hosts", str(tmp_path)):
        with pytest.raises((KeyError, ValueError)):
            api.drop_file(tricky)
    api.drop_probe(str(tmp_path / "other" / ".." / "a.mp4"))                         # (the same file by another spelling is the same file)
    assert api.drop_file(str(tmp_path / "a.mp4")) == mp4.resolve()
    txt = tmp_path / "notes.txt"
    txt.write_text("x")
    api.drop_probe(str(txt))
    with pytest.raises(KeyError):
        api.drop_file(str(txt))                                                       # not media, even though it was probed


def test_thumbnails_are_the_right_shape_and_kept(home, tmp_path):
    from PIL import Image
    api = _api(home, tmp_path)
    vert = _video(tmp_path / "vertical.mp4", 180, 320, 3)
    api.drop_probe(str(vert))
    t = api.drop_thumb(str(vert), 1.0)
    with Image.open(t) as im:
        assert im.width == 320 and im.height > im.width                               # scaled to 320 wide, still portrait
    assert api.drop_thumb(str(vert), 1.0) == t                                         # kept: the same file again
    assert api.drop_thumb(str(vert), 99.0).exists()                                     # past the end: still a picture
    jpg = tmp_path / "p.jpg"
    subprocess.run([FF, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=s=640x480:d=1", "-frames:v", "1", str(jpg)], check=True)
    api.drop_probe(str(jpg))
    with Image.open(api.drop_thumb(str(jpg))) as im:
        assert max(im.size) <= 640 and im.width <= 320


def test_the_routes_over_http(tmp_path, home, monkeypatch):
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        mp4 = _video(tmp_path / "clips" / "IMG_0001.MOV", 180, 320, 2)
        other = _video(tmp_path / "clips" / "IMG_0002.MOV", 180, 320, 2)
        q = lambda p: urllib.parse.quote(str(p), safe="")  # noqa: E731
        st, body = _call(base, f"/api/app/probe?path={q(mp4)}")
        assert st == 200 and json.loads(body)["kind"] == "video"
        assert _call(base, f"/api/app/probe?path={q(tmp_path / 'nope.mp4')}")[0] == 400
        assert _call(base, "/api/app/probe?path=..%2F..%2Fetc%2Fhosts")[0] == 400
        st, body = _call(base, f"/api/app/thumb?path={q(mp4)}&t=0.5")
        assert st == 200 and body[:3] == b"\xff\xd8\xff"
        assert _call(base, f"/api/app/thumb?path={q(other)}")[0] == 404              # not dropped
        assert _call(base, f"/api/app/localfile?path={q(other)}")[0] == 404
        assert _call(base, "/api/app/localfile?path=%2Fetc%2Fhosts")[0] == 404
        full = urllib.request.urlopen(base + f"/api/app/localfile?path={q(mp4)}", timeout=20)
        assert full.status == 200 and full.headers["Accept-Ranges"] == "bytes" and full.headers["Content-Type"].startswith("video/")
        req = urllib.request.Request(base + f"/api/app/localfile?path={q(mp4)}", headers={"Range": "bytes=0-99"})
        part = urllib.request.urlopen(req, timeout=20)
        assert part.status == 206 and len(part.read()) == 100
    finally:
        httpd.shutdown()


# ------------------------------------------------------------------------------------------------ several videos -> several sets of clips
def test_several_videos_make_several_sets_of_clips(tmp_path, home, monkeypatch):
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        a, b = _video(tmp_path / "x" / "one.mp4"), _video(tmp_path / "y" / "two.mov")
        st, body = _call(base, "/api/app/short", {"sources": [str(a), str(b), str(a), str(tmp_path / "gone.mp4"), "https://example.com/watch?v=1"]})
        d = json.loads(body)
        assert st == 201 and len(d["ids"]) == 3 and d["id"] == d["ids"][0]               # (a repeated path counts once)
        assert len(d["errors"]) == 1 and "gone.mp4" in d["errors"][0]
        shorts = json.loads(_call(base, "/api/app")[1])["shorts"]
        assert sorted(s["id"] for s in shorts) == sorted(d["ids"])
        assert _call(base, "/api/app/short", {"sources": [str(tmp_path / "gone.mp4")]})[0] == 400      # nothing usable: refused, plainly
        assert _call(base, "/api/app/short", {"sources": ["x"] * 3 + [str(i) for i in range(50)]})[0] == 400
    finally:
        httpd.shutdown()


def test_a_dropped_video_without_a_usual_extension_can_be_made(tmp_path, home, monkeypatch):
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        nameless = tmp_path / "export" / "9C1F"
        nameless.parent.mkdir()
        shutil.copy(_video(tmp_path / "v.mp4"), nameless)
        assert _call(base, "/api/app/short", {"source": str(nameless)})[0] == 400          # not known to be a video yet
        assert _call(base, "/api/app/probe?path=" + urllib.parse.quote(str(nameless), safe=""))[0] == 200
        st, body = _call(base, "/api/app/short", {"source": str(nameless)})
        assert st == 201
        shorts = json.loads(_call(base, "/api/app")[1])["shorts"]
        assert [x["id"] for x in shorts] == [json.loads(body)["id"]] and shorts[0]["status"] == "queued"
    finally:
        httpd.shutdown()
