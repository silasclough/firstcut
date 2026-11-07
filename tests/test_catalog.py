"""The footage catalog (Video Builder): scanning a drive, cataloguing frames, searching in words, B-roll for an interview, the Final Cut
event of found shots, the work app's AI routing (local only, its own Claude login) and the "save to" folder. The picture model and the
vision model are faked: colours stand in for meaning, so 'red' finds the red clip."""

import json
import os
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipfarm.clipper import catalog as C
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

COLORS = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}


class FakeEmbedder:
    """Pictures -> their mean colour; words -> the colour they name. Cosine of a pure-red frame with 'red' is 1."""
    name = "fake"

    def images(self, ims):
        out = []
        for im in ims:
            a = np.asarray(im.convert("RGB"), dtype="float32").reshape(-1, 3).mean(0)
            v = np.array([*a, 1.0], dtype="float32")
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype="float32")

    def text(self, texts):
        out = []
        for t in texts:
            c = next((v for k, v in COLORS.items() if k in t.lower()), (0.3, 0.3, 0.3))
            v = np.array([*(x * 255 for x in c), 1.0], dtype="float32")
            out.append(v / np.linalg.norm(v))
        return np.array(out, dtype="float32")

    def chance(self, score):
        return score


class FakeVision:
    def __init__(self):
        self.calls = []

    def complete_json(self, system, prompt, schema, images=None):
        self.calls.append(prompt)
        if "tiles" in schema["properties"]:
            n = int(prompt.split(" ")[0])
            return SimpleNamespace(data={"tiles": [{"n": k + 1, "what": f"tile {k + 1} described", "quality": 7} for k in range(n)]})
        return SimpleNamespace(data={"scenes": [{"start_s": 0, "end_s": 6, "what": "a red wall in a gym", "quality": 8}]})


def _video(path: Path, color: str, secs: int = 7, size: str = "320x180"):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={size}:d={secs}:r=25",
                    "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", str(path)], check=True)


def _photo(path: Path, rgb):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (400, 300), rgb).save(path, quality=90)


def _cfg(tmp_path, roots, **extra):
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["catalog"] = {"roots": [str(r) for r in roots], "dir": str(tmp_path / "cat"), "frame_every_s": 2.0, "max_frames": 50,
                           "min_score": 0.5, **extra}
    return cfg


@pytest.fixture
def drive(tmp_path):
    root = tmp_path / "Drive" / "Video"
    _video(root / "B-roll" / "Athletics" / "red_gym.mp4", "red")
    _video(root / "B-roll" / "Student Life" / "blue_sky.mp4", "blue", secs=12)
    _video(root / "B-roll" / "Generic Library.fcpbundle" / "inside.mp4", "green")           # Final Cut's own media: never catalogued
    _video(root / "Projects" / "Proxy Media" / "proxy.mp4", "green")
    _video(root / "$RECYCLE.BIN" / "gone.mp4", "green")
    _photo(root / "Photos" / "green_lawn.jpg", (0, 255, 0))
    (root / "Photos" / "green_lawn.CR3").write_bytes(b"x" * 4096)                               # the RAW twin of the JPEG: counted once
    (root / "B-roll" / "._red_gym.mp4").write_bytes(b"x" * 4096)                                 # macOS litter on exFAT drives
    return root


def test_scan_leaves_out_final_cut_media_proxies_the_bin_and_raw_twins(tmp_path, drive):
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    found = cat.scan()
    rels = sorted(f["rel"] for f in found)
    assert rels == ["Video/B-roll/Athletics/red_gym.mp4", "Video/B-roll/Student Life/blue_sky.mp4", "Video/Photos/green_lawn.jpg"]
    assert {f["kind"] for f in found} == {"video", "photo"}


def test_index_is_resumable_notices_changes_and_forgets_deleted_files(tmp_path, drive):
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    r = cat.index()
    assert r["new"] == 3 and r["failed"] == 0 and r["left"] == 0
    rec = next(x for x in cat.records() if x["rel"].endswith("blue_sky.mp4"))
    assert rec["frames"] >= 5 and rec["dur"] == pytest.approx(12, abs=0.2) and rec["w"] == 320
    assert cat.thumb(rec["id"], 0).exists() and (cat.dir / "emb" / f"{rec['id']}.npy").exists()
    assert cat.index()["new"] == 0                                                   # nothing to do the second time
    red = drive / "B-roll" / "Athletics" / "red_gym.mp4"
    os.utime(red, (time.time() + 60, time.time() + 60))
    assert cat.index()["new"] == 1                                                   # a changed file is looked at again
    red.unlink()
    assert cat.index()["forgotten"] == 1
    assert not any(x["rel"].endswith("red_gym.mp4") for x in cat.records())


def test_an_unplugged_drive_forgets_nothing(tmp_path, drive):
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    cat.index()
    gone = C.Catalog(_cfg(tmp_path, [tmp_path / "NotPluggedIn"]), embedder=FakeEmbedder())
    gone.dir = cat.dir
    assert gone.index()["forgotten"] == 0
    assert len(list(cat.records())) == 3


def test_search_finds_the_matching_clip_first_with_a_usable_shot(tmp_path, drive):
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    cat.index()
    shots = cat.search("a red wall")
    assert shots[0]["rel"].endswith("red_gym.mp4") and shots[0]["score"] > 0.9
    s = shots[0]
    assert 0 <= s["start"] < s["end"] <= s["dur"] and s["end"] - s["start"] >= C.SHOT_MIN - 0.01
    assert len({x["id"] for x in shots}) == len(shots)                               # one shot per file
    assert [x["kind"] for x in cat.search("green", kind="photo")] == ["photo"]
    assert cat.search("green", kind="photo")[0]["rel"].endswith("green_lawn.jpg")
    assert all("Student Life" in x["rel"] for x in cat.search("red", under="student life"))
    assert cat.search("") == []


def test_a_word_in_the_folder_names_lifts_a_file(tmp_path, drive):
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    cat.index()
    plain = {s["rel"]: s["score"] for s in cat.search("purple things")}
    athletic = {s["rel"]: s["score"] for s in cat.search("purple athletics")}
    red = "Video/B-roll/Athletics/red_gym.mp4"
    assert athletic[red] > plain[red]


def test_descriptions_come_from_the_local_vision_model_and_show_in_results(tmp_path, drive):
    vision = FakeVision()
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder(), vision=vision)
    cat.index()
    r = cat.describe()
    assert r["described"] == 3 and r["left"] == 0
    assert any("contact sheet" in p for p in vision.calls)                         # videos: their frames as contact sheets
    assert cat.search("red")[0]["what"] == "a red wall in a gym"
    photo = cat.search("green", kind="photo")[0]
    assert photo["what"] == "tile 1 described" and photo["quality"] == 7
    assert cat.describe()["described"] == 0                                         # done once


def test_shots_for_an_interview_searches_what_the_ai_asks_for(tmp_path, drive):
    from clipfarm.clipper.moments import Sentence

    class AI:
        def __init__(self):
            self.prompts = []

        def complete_json(self, task, system, prompt, schema):
            assert task == "broll_queries"
            self.prompts.append(prompt)
            return SimpleNamespace(data={"passages": [{"n": 1, "looks": ["red gym wall"]}, {"n": 2, "looks": ["blue sky", "grey nothing"]}]})
    cat = C.Catalog(_cfg(tmp_path, [drive], min_score=0.9), embedder=FakeEmbedder(), vision=FakeVision())
    cat.index()
    sents = [Sentence(0, 0.0, 19.0, "We trained every morning in the gym."), Sentence(1, 20.0, 30.0, "Then we looked up at the sky.")]
    ai = AI()
    shots = cat.shots_for(ai, sents, brief="A college admissions video")
    assert "college admissions" in ai.prompts[0] and "P1" in ai.prompts[0] and "P2" in ai.prompts[0]
    assert [Path(s["rel"]).name for s in shots] == ["red_gym.mp4", "blue_sky.mp4"]          # 'grey nothing' matched nothing well enough
    assert all(s["what"] for s in shots) and shots[0]["query"] == "red gym wall"


def test_the_found_shots_become_a_final_cut_event(tmp_path, drive):
    import xml.etree.ElementTree as ET

    from clipfarm.clipper import fcpxml as FX
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    cat.index()
    shots = [cat.search("red", kind="video")[0], cat.search("blue", kind="video")[0]]
    xml = cat.fcpxml(shots, "Found footage - gyms & skies")
    root = ET.fromstring(xml.split("\n", 2)[2])
    clips = root.findall("./library/event/asset-clip")
    assert root.find("./library/event").get("name") == "Found footage - gyms & skies"
    assert len(clips) == 2 and len(root.findall("./resources/asset")) == 2
    assert all(a.find("media-rep").get("src").startswith("file://") for a in root.findall("./resources/asset"))
    p = tmp_path / "found.fcpxml"
    p.write_text(xml)
    assert FX.validate(p) == []                                                      # (checked against Final Cut's DTD when it is installed)


def test_the_night_window_runs_past_midnight():
    w = ["19:00", "07:00"]
    assert C.in_window(w, datetime(2026, 11, 5, 23, 30)) and C.in_window(w, datetime(2026, 11, 5, 3, 0))
    assert not C.in_window(w, datetime(2026, 11, 5, 12, 0)) and not C.in_window(w, datetime(2026, 11, 5, 7, 0))
    assert C.in_window(["09:00", "17:00"], datetime(2026, 11, 5, 9, 0)) and C.in_window(None)


def test_the_catalogue_waits_while_a_video_is_being_made(tmp_path, drive, monkeypatch):
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    busy = {"n": 2}

    def is_busy():
        busy["n"] -= 1
        return busy["n"] >= 0
    cat.busy = is_busy
    monkeypatch.setattr(C.time, "sleep", lambda s: None)
    assert cat.index(limit=1)["new"] == 1 and busy["n"] < 0


# ---------------------------------------------------------------------------------------------- the work app's AI
def test_the_work_app_keeps_pictures_local_and_never_reaches_openai(monkeypatch, tmp_path):
    from clipfarm.clipper import localai as LA
    from clipfarm.core import llm as L
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "vision": "ollama", "fallback": "ollama", "data_dir": "ai"})
    cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "skip_tasks": ["broll_describe"], "fallback": True})
    calls = []

    def fake(self, system, prompt, schema, images=None):
        calls.append((self.cfg["model"], bool(images)))
        return L.LLMResult({"ok": True}, "ollama", self.cfg["vision_model"] if images else self.cfg["model"], {})
    monkeypatch.setattr(L.OllamaLLM, "complete_json", fake)
    monkeypatch.setattr(L.RemoteJson, "complete_json", lambda *a, **k: pytest.fail("never the posting server"))
    ai = LA.LocalAI(cfg)
    assert ai.host is None and ai.key is None and not ai.direct()
    ok, why = ai.available()
    assert ok and "local" in why                                                    # Claude not signed in: the local model answers
    assert ai.complete_json("broll_describe", "s", "p", {}, images=[b"jpg"]).model == "qwen3-vl:8b"
    assert ai.complete_json("story_plan", "s", "p", {}).data == {"ok": True}
    monkeypatch.setattr(LA.ClaudeCLI if hasattr(LA, "ClaudeCLI") else ai.claude.__class__, "available", lambda self: True)
    seen = []
    monkeypatch.setattr(ai.claude.__class__, "complete_json", lambda self, *a, **k: (seen.append(a[0]) or ({"from": "claude"}, "sonnet")))
    assert ai.complete_json("broll_plan", "s", "p", {}).data == {"from": "claude"}  # signed in: the work Claude plan decides
    assert ai.complete_json("broll_describe", "s", "p", {}, images=[b"jpg"]).provider == "ollama"   # ...but pictures stay here
    cfg.data["clipper"]["ai_videos"]["fallback"] = None
    monkeypatch.setattr(ai.claude.__class__, "available", lambda self: False)
    assert LA.LocalAI(cfg).available()[0] is False


def test_claude_command_can_be_the_desktop_apps_newest_copy(tmp_path):
    from clipfarm.clipper.claude_cli import ClaudeCLI
    for v in ("2.1.9", "2.1.317", "2.1.40"):
        b = tmp_path / "claude-code" / v / "claude"
        b.parent.mkdir(parents=True)
        b.write_text("#!/bin/sh\n")
        b.chmod(0o755)
    cfg = SimpleNamespace(get=lambda k, d=None: {"llm.claude": {"enabled": True, "bin": str(tmp_path / "claude-code" / "*" / "claude"),
                                                                 "skip_tasks": ["broll_describe"]}}.get(k, d))
    c = ClaudeCLI(cfg)
    assert c.bin.endswith("2.1.317/claude")
    assert not c.handles("broll_describe") and c.handles("broll_plan")


def test_an_overlay_config_is_merged_last(tmp_path, monkeypatch):
    (tmp_path / "over.yaml").write_text("clipper:\n  data_dir: data/work/clipper\ncatalog:\n  roots: [/Volumes/X]\n")
    monkeypatch.setenv("CLIPFARM_OVERLAY", str(tmp_path / "over.yaml"))
    cfg = load_config()
    assert cfg.get("clipper.data_dir") == "data/work/clipper" and cfg.get("catalog.roots") == ["/Volumes/X"]
    assert cfg.get("clipper.min_len") is not None                                   # everything else stays


def test_the_shipped_work_config_is_local_only():
    import yaml
    w = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "work.yaml").read_text())
    av = w["clipper"]["ai_videos"]
    assert av["openai"]["enabled"] is False and av["vision"] == "ollama" and av["fallback"] == "ollama"
    assert w["clipper"]["background"] is False and w["clipper"]["auto"]["enabled"] is False and w["clipper"]["feed"]["local"] is False
    assert w["clipper"]["data_dir"] != "data/clipper" and w["llm"]["claude"]["config_dir"]
    assert "broll_describe" in w["llm"]["claude"]["skip_tasks"]


# ---------------------------------------------------------------------------------------------- the editor
def test_library_shots_join_the_project_past_its_own_footage(tmp_path, drive):
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.moments import Sentence  # noqa: F401
    cfg = _cfg(tmp_path, [drive], min_score=0.9)
    cat = C.Catalog(cfg, embedder=FakeEmbedder(), vision=FakeVision())
    cat.index()
    runner = LP.LocalRunner(cfg, None, None)
    runner.catalog = cat

    class AI:
        def complete_json(self, task, system, prompt, schema):
            return SimpleNamespace(data={"passages": [{"n": 1, "looks": ["red gym", "green lawn"]}]})
    st = LP.State(tmp_path / "proj")
    files = [{"name": "talk.mp4", "path": "/x/talk.mp4", "start": 0.0, "end": 60.0, "dur": 60.0, "w": 1920, "h": 1080, "fps": 25.0,
              "audio": True, "broll": False}]
    words = [{"text": w, "start": i * 0.5, "end": i * 0.5 + 0.4} for i, w in enumerate("we trained in the gym every single morning.".split())]
    files2, scenes = runner.add_library(files, words, {}, AI(), st, "brief", lambda f, m: None)
    lib = [f for f in files2 if f.get("library")]
    assert len(lib) == 1 and lib[0]["start"] >= LP.LIB_BASE and lib[0]["broll"] and lib[0]["path"].endswith("red_gym.mp4")
    sc = scenes[lib[0]["name"]][0]
    assert lib[0]["start"] <= sc["start"] < sc["end"] <= lib[0]["end"]
    saved = st.get("library.json")
    assert saved["photos"] and saved["photos"][0]["photo"].endswith("green_lawn.jpg") and saved["photos"][0]["start"] >= LP.LIB_PHOTO_BASE
    # the cut: a cutaway at 2 s showing the library shot puts the library file's picture there (its own time), the voice stays
    parts = LP.build_parts(files2, None, [], [(0.0, 10.0)], [{"at": 2.0, "length": 2.0, "src": sc["start"], "scene": 0}])
    cut = next(p for p in parts if p["v"].get("cut"))
    assert cut["v"]["path"].endswith("red_gym.mp4") and cut["v"]["ss"] == pytest.approx(sc["start"] - lib[0]["start"])
    assert cut["a"]["path"] == "/x/talk.mp4"
    # a re-render reads the same shots back (describe merges them; library files are never described again)
    again = runner.describe(files2, words, None, st, lambda f, m: None)
    assert again[lib[0]["name"]] == scenes[lib[0]["name"]]


def test_save_to_folder_gets_a_folder_per_video(tmp_path):
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.store import Store
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI videos"), "local": True, "data_dir": "ai"}
    projects = AV.Projects(cfg, Store(tmp_path / "jobs"), None)
    footage = tmp_path / "Interview"
    footage.mkdir()
    a = projects.create_local(str(footage), "Garden story", "", {}, {}, out_dir=str(tmp_path / "Saves"))
    assert a["home"] == str(tmp_path / "Saves" / "Garden story") and Path(a["home"]).is_dir()
    (Path(a["home"]) / "something").write_text("x")
    b = projects.create_local(str(footage), "Garden story", "", {}, {}, out_dir=str(tmp_path / "Saves"))
    assert b["home"] == str(tmp_path / "Saves" / "Garden story (2)")
    assert LP.home_of(a) == Path(a["home"]) and projects.view(a)["saved_to"] == a["home"]
    c = projects.create_local(str(footage), "In place", "", {}, {})
    assert c["home"] is None and LP.home_of(c) == footage
    with pytest.raises(ValueError):
        projects.create_local(str(footage), "x", "", {}, {}, out_dir="relative/path")


def test_the_server_searches_previews_and_never_serves_other_files(tmp_path, drive):
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    from clipfarm.clipper.localproj import LocalRunner
    from clipfarm.clipper.store import Store
    cfg = _cfg(tmp_path, [drive])
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "AI"), "local": True, "data_dir": "ai", "local_projects": True,
                                        "out_dir": str(tmp_path / "Saves")}
    cfg.data["clipper"]["app"] = "work"
    store = Store(tmp_path / "jobs")
    eng = Engine(cfg, store)
    projects = AV.Projects(cfg, store, eng)
    runner = LocalRunner(cfg, store, projects)
    runner.catalog = C.Catalog(cfg, embedder=FakeEmbedder())
    runner.catalog.index()
    worker = S.Worker(eng, store, Feed(cfg, store), projects)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, store, worker, worker.feed, 100, projects, runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def get(path):
        with urllib.request.urlopen(base + path, timeout=20) as r:
            return r.status, r.headers.get("Content-Type"), r.read()

    def post(path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            return json.loads(r.read())
    try:
        assert b"Video Builder" in get("/")[2]                                       # the app opens on the builder
        d = json.loads(get("/api/ai/projects")[2])
        assert d["app"] == "work" and d["library"]["videos"] == 2 and d["out_dir"].endswith("Saves")
        shots = json.loads(get("/api/library/search?q=red&n=5")[2])["shots"]
        assert shots[0]["rel"].endswith("red_gym.mp4") and "path" not in shots[0]
        st, ctype, body = get(f"/api/library/thumb/{shots[0]['id']}/0")
        assert st == 200 and ctype == "image/jpeg" and body[:2] == b"\xff\xd8"
        st, ctype, body = get(f"/api/library/media/{shots[0]['id']}")
        assert st == 200 and ctype == "video/mp4" and len(body) == (drive / "B-roll" / "Athletics" / "red_gym.mp4").stat().st_size
        st, ctype, body = get(f"/api/library/preview/{shots[0]['id']}?s=1&e=4")
        assert st == 200 and ctype == "video/mp4" and len(body) > 1000
        assert ffmpeg.probe(next((tmp_path / "cat" / "previews").glob("*.mp4")))["streams"][0]["height"] == 540
        for bad in ("/api/library/media/..%2F..%2Fetc%2Fpasswd", "/api/library/preview/zzzz?s=0&e=3", "/api/library/thumb/zzzz/0", "/api/library/media/0123456789abcdef"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(bad)
            assert e.value.code == 404
        r = post("/api/library/fcpxml", {"shots": [{"id": shots[0]["id"], "start": 1, "end": 4, "what": "red"}], "name": "Found footage - red"})
        assert r["clips"] == 1 and Path(r["path"]).exists() and Path(r["path"]).parent == tmp_path / "Saves" / "Found footage"
        with pytest.raises(urllib.error.HTTPError) as e:
            post("/api/library/fcpxml", {"shots": [{"id": "../../x", "start": 0, "end": 3}]})
        assert e.value.code == 400
    finally:
        httpd.shutdown()


def test_a_chosen_folder_is_catalogued_first_and_searched_alone(tmp_path, drive):
    other = tmp_path / "Other drive" / "B-roll"
    _video(other / "green_field.mp4", "green")
    cat = C.Catalog(_cfg(tmp_path, [drive, other]), embedder=FakeEmbedder())
    assert cat.start(only=[other], rest=False, describe=False)
    cat.thread.join(60)
    st = cat.stats()
    per = {r["path"]: r for r in st["roots"]}
    assert per[str(other)]["videos"] == 1 and per[str(drive)]["videos"] == 0       # only the chosen folder, nothing else yet
    assert cat.only is None and not st["status"]["running"]
    assert cat.start(describe=False)
    cat.thread.join(60)
    assert {r["path"]: r["videos"] for r in cat.stats()["roots"]}[str(drive)] == 2
    hits = cat.search("green", under=str(other))
    assert hits and all(h["path"].startswith(str(other) + "/") for h in hits)
    assert not any(h["rel"].endswith("red_gym.mp4") for h in cat.search("red", under=str(other)))   # the red clip is in the other source
    assert cat.search("red")[0]["rel"].endswith("red_gym.mp4")


def test_a_source_runs_alone_forgets_nothing_elsewhere(tmp_path, drive):
    other = tmp_path / "Other" / "Clips"
    _video(other / "blue.mp4", "blue")
    cat = C.Catalog(_cfg(tmp_path, [drive, other]), embedder=FakeEmbedder())
    cat.index()
    n = len(list(cat.records()))
    cat.only = [other]
    (other / "blue.mp4").unlink()
    assert cat.index()["forgotten"] == 1                                             # gone from the chosen folder: forgotten
    cat.only = None
    assert len(list(cat.records())) == n - 1                                          # everything in the other folder stays


def test_a_catalogue_run_asked_for_carries_on_after_a_restart(tmp_path, drive):
    other = tmp_path / "New drive" / "B-roll"
    _video(other / "blue.mp4", "blue")
    cfg = _cfg(tmp_path, [drive, other])
    cat = C.Catalog(cfg, embedder=FakeEmbedder())
    cat.dir.mkdir(parents=True, exist_ok=True)
    json.dump({"only": [str(other)], "rest": False, "describe": False, "frames": True, "limit": None, "until": None},
              open(cat.dir / C.PENDING, "w"))                                         # the app quit in the middle of this run
    again = C.Catalog(cfg, embedder=FakeEmbedder())
    again.tick()                                                                      # (outside the night window too)
    again.thread.join(60)
    per = {r["path"]: r["videos"] for r in again.stats()["roots"]}
    assert per[str(other)] == 1 and per[str(drive)] == 0 and not (cat.dir / C.PENDING).exists()
    json.dump({"only": [str(other)], "rest": False, "until": time.time() - 5}, open(cat.dir / C.PENDING, "w"))
    again.tick()                                                                      # its time ran out while the app was closed
    assert not (again.thread and again.thread.is_alive()) and not (cat.dir / C.PENDING).exists()
    assert again.start(only=[str(other)], rest=False, describe=False)                 # a run that ends here leaves nothing behind
    again.thread.join(60)
    assert not (cat.dir / C.PENDING).exists()


def test_the_heat_map_follows_the_words_and_the_picture(tmp_path):
    root = tmp_path / "Drive"
    clip = root / "mix.mp4"
    clip.parent.mkdir(parents=True)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=blue:s=320x180:d=6:r=25", "-f", "lavfi", "-i",
                    "color=c=red:s=320x180:d=6:r=25", "-f", "lavfi", "-i", "testsrc2=s=320x180:d=6:r=25", "-filter_complex",
                    "[0][1][2]concat=n=3:v=1:a=0", "-c:v", "libx264", "-g", "25", "-pix_fmt", "yuv420p", str(clip)], check=True)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    rec = next(cat.records())
    h = cat.heat(rec["id"], "something red")
    n = len(h["heat"])
    assert n == rec["frames"] and all(0 <= x <= 1 for x in h["heat"])
    assert 6 <= h["peak"] * h["step"] < 12                                            # the red part matches "red" best
    plain = cat.heat(rec["id"])                                                       # no words: the busiest, sharpest picture wins
    assert plain["peak"] * plain["step"] >= 12
    assert cat.load(rec["id"])["look"] == cat.look(rec["id"])                         # worked out once, kept with the record
    assert cat.heat("0" * 16) == {"step": cat.every, "heat": [], "peak": 0}



def test_always_catalogues_whenever_the_app_is_open(tmp_path, monkeypatch):
    """2026-11-06: "I want it to run whenever the app is open" (window "always"): any time of day, until everything is done, then a
    look for new footage every ten minutes; never while a video is being made."""
    from clipfarm.clipper import catalog as C
    assert C.in_window("always")
    cat = C.Catalog.__new__(C.Catalog)
    cat.window, cat.status, cat.thread = "always", {"finished": 0}, None
    started = []
    monkeypatch.setattr(cat, "enabled", lambda: True, raising=False)
    monkeypatch.setattr(cat, "mounted", lambda: True, raising=False)
    monkeypatch.setattr(cat, "start", lambda **k: started.append(k) or True, raising=False)
    cat.dir = tmp_path
    cat.tick(busy=True)
    assert started == []                                                              # a video is being made: wait
    cat.tick()
    assert started == [{"hours": None, "night": True}]                                # until it is all done
    cat.status["finished"] = __import__("time").time()
    cat.tick()
    assert len(started) == 1                                                          # just finished: not again for ten minutes


def test_the_account_can_describe_the_catalogue_but_only_when_chosen(tmp_path, monkeypatch):
    """Settings › Footage library › Describe shots with: this Mac (default) or the AI account (task catalog_describe, offline: this Mac)."""
    from clipfarm.clipper import catalog as C
    from clipfarm.clipper import studio as SD
    from clipfarm.clipper import localai as LA
    s = SD.defaults()
    s["ai"]["use"] = "claude"
    assert s["library"]["describe_with"] == "account"                                    # the default: the account (cheap, fast)
    s["library"]["describe_with"] = "local"
    assert SD.overlay(s, tmp_path)["catalog"]["describe_with"] == "local"
    s["library"]["describe_with"] = "account"
    assert SD.overlay(s, tmp_path)["catalog"]["describe_with"] == "account"
    assert "catalog_describe" in SD.overlay(s, tmp_path)["llm"]["claude"]["cheap_tasks"]
    s["ai"]["use"] = "local"
    assert SD.overlay(s, tmp_path)["catalog"]["describe_with"] == "local"                 # only this Mac: always this Mac
    asked = []
    monkeypatch.setattr(LA.LocalAI, "complete_json", lambda self, task, system, prompt, schema, images=None, **k: asked.append((task, len(images or []))) or "ok")
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["catalog"] = {"dir": str(tmp_path / "cat"), "describe_with": "account"}
    cat = C.Catalog(cfg)
    assert cat.vision().complete_json("sys", "p", {}, images=[b"jpg"]) == "ok" and asked == [("catalog_describe", 1)]
    cfg.data["catalog"]["describe_with"] = "local"
    assert type(C.Catalog(cfg).vision()).__name__ == "OllamaLLM"


def test_log_clips_catalogued_in_grey_are_looked_at_again(tmp_path, monkeypatch):
    """Clips catalogued before log footage was converted were seen grey (worse search and descriptions): they are done again once."""
    from clipfarm.clipper import catalog as C
    from clipfarm.clipper import logcolor
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["catalog"] = {"dir": str(tmp_path / "cat")}
    cat = C.Catalog(cfg)
    log_clip = {"id": "a1", "path": "/x/FX30.MP4", "kind": "video", "size": 10, "mtime": 1}
    normal = {"id": "b2", "path": "/x/A6500.MP4", "kind": "video", "size": 10, "mtime": 1}
    for f in (log_clip, normal):
        cat.save({**f, "model": cat.model, "frames": 30, "indexed": 1})                     # (no "colour": catalogued before)
    monkeypatch.setattr(logcolor, "vf_prefix", lambda path, cfg=None: "lut3d=file='x.cube'," if "FX30" in path else "")
    assert [f["id"] for f in cat.todo([log_clip, normal])] == ["a1"]
    cat.save({**log_clip, "model": cat.model, "frames": 30, "indexed": 2, "colour": "converted"})
    assert cat.todo([log_clip, normal]) == []                                               # done once


def test_a_run_that_finds_nothing_says_why(tmp_path):
    """BUG (reported 2026-11-07): "Catalogue it now" on a folder did nothing, with no word of why. A folder that cannot be read, or holds only
    finished edits, now says so in plain words (the page shows it where it used to say 'isn't catalogued yet')."""
    drive = tmp_path / "Drive"
    drive.mkdir()
    (drive / "Event FINAL.mp4").write_bytes(b"x" * 4000)
    cat = C.Catalog(_cfg(tmp_path, [drive]), embedder=FakeEmbedder())
    assert cat.scan() == [] and "finished edits" in cat.nothing_found() and "1 video" in cat.nothing_found()
    empty = tmp_path / "Empty"
    empty.mkdir()
    cat2 = C.Catalog(_cfg(tmp_path, [empty]), embedder=FakeEmbedder())
    cat2.scan()
    assert "no videos or photos" in cat2.nothing_found()
    locked = tmp_path / "Locked"
    (locked / "sub").mkdir(parents=True)
    (locked / "sub" / "a.mp4").write_bytes(b"x" * 4000)
    (locked / "sub").chmod(0)
    try:
        cat3 = C.Catalog(_cfg(tmp_path, [locked]), embedder=FakeEmbedder())
        cat3.scan()
        if os.geteuid() != 0:                                                     # (root can read anything)
            assert "won't let FirstCut read" in cat3.nothing_found()
    finally:
        (locked / "sub").chmod(0o700)


def test_setup_is_not_done_while_pytorch_is_missing(monkeypatch):
    """BUG (reported 2026-11-07): transformers was in the app's Python but torch was not; Setup called the footage search done, so nothing
    installed it and every catalogue run stopped with "No module named 'torch'". Both packages are checked now."""
    import importlib.util as iu

    from clipfarm.clipper import studio as SD
    real = iu.find_spec
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: None if name == "torch" else real("os"))
    assert SD.search_packages_ready() is False
    monkeypatch.setattr(iu, "find_spec", lambda name, *a, **k: real("os"))
    assert SD.search_packages_ready() is True
