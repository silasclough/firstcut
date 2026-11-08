"""Video Builder (the app): its settings and what they turn into for the engine, the "how it edits" points per kind of video, importing a
ChatGPT skill, project folders, short clips saved into them, and the shipped settings having nothing personal in them."""

import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import studio as SD


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "VB"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    monkeypatch.setattr(SD, "_answers", lambda url: False)                 # no Ollama running in tests
    monkeypatch.setattr(SD, "ram_gb", lambda: 16.0)
    return h


class FakeAI:
    def __init__(self, answer):
        self.answer, self.prompts = answer, []

    def complete_json(self, task, system, prompt, schema, **kw):
        self.prompts.append((task, system, prompt))
        return SimpleNamespace(data={"points": self.answer})


def test_settings_keep_only_known_values(home):
    s = SD.Settings()
    out = s.update({"appearance": "dark", "ai": {"use": "claude", "fallback": False, "local_model": "rm -rf /"}, "places": {"videos": "relative"},
                    "defaults": {"short": {"min_len": "25", "caption_style": "pop", "evil": 1}}, "library": {"folders": ["~/Footage", ""]}, "hack": 1})
    assert out["appearance"] == "dark" and out["ai"] == {"use": "claude", "fallback": False, "local_model": None,
                                                          "claude_edit": "sonnet", "claude_quick": "haiku", "chatgpt_model": "", "look": "account",
                                                          "backup": "auto"}
    ai = s.update({"ai": {"claude_edit": "opus", "claude_quick": "opus", "chatgpt_model": "gpt-5; rm -rf"}})["ai"]
    assert ai["claude_edit"] == "opus" and ai["claude_quick"] == "haiku" and ai["chatgpt_model"] == ""   # only known models
    ov = SD.overlay(s.get(), home)
    assert ov["llm"]["claude"]["models"] == {"script": "opus", "default": "opus", "cheap": "haiku"}
    s.update({"ai": {"claude_edit": "sonnet"}})
    assert out["places"]["videos"].endswith("Movies/FirstCut")                          # a relative path is ignored
    assert out["defaults"]["short"]["min_len"] == 25 and out["defaults"]["short"]["caption_style"] == "pop" and "evil" not in out["defaults"]["short"]
    assert out["defaults"]["long"]["caption_style"] == "none"                          # long form keeps its own defaults
    assert out["library"]["folders"] == [str(Path("~/Footage").expanduser())] and "hack" not in out
    assert json.loads((home / "settings.json").read_text())["appearance"] == "dark"


def test_the_ai_choice_decides_what_the_engine_may_use(home):
    s = SD.Settings()
    ov = SD.overlay(s.get(), home)
    av = ov["clipper"]["ai_videos"]
    assert av["openai"]["enabled"] is False and av["chatgpt_plan"]["enabled"] is True and ov["llm"]["claude"]["enabled"] is True   # both accounts
    assert av["primary"] == "chatgpt" and av["backup"] == "auto"
    assert av["vision"] == "" and av["fallback"] == "ollama" and av["ollama_model"] == "qwen3:8b" and av["vision_model"] == "qwen3-vl:4b"   # (the account looks at footage by default)
    assert ov["llm"]["ollama"]["base_url"].endswith(str(SD.OWN_OLLAMA_PORT))          # no Ollama app running: our own
    assert ov["clipper"]["background"] is False and ov["clipper"]["auto"]["enabled"] is False and ov["catalog"]["roots"] == []
    s.update({"ai": {"use": "claude", "fallback": False}})
    ov = SD.overlay(s.get(), home)
    assert ov["llm"]["claude"]["enabled"] is True and ov["llm"]["claude"]["config_dir"] == str(home / "claude")
    assert ov["clipper"]["ai_videos"]["primary"] == "claude" and ov["clipper"]["ai_videos"]["fallback"] == ""
    assert SD.overlay(s.get(), home)["catalog"]["window"] == "always"                 # by default: whenever the app is open
    s.update({"library": {"when": "night"}})
    assert SD.overlay(s.get(), home)["catalog"]["window"] == ["19:00", "07:00"]
    s.update({"ai": {"use": "local"}, "library": {"folders": ["/Volumes/X/B-roll"], "when": "off"}})
    ov = SD.overlay(s.get(), home)
    assert ov["clipper"]["ai_videos"]["fallback"] == "ollama" and ov["catalog"]["roots"] == ["/Volumes/X/B-roll"] and ov["catalog"]["window"] is None
    p = SD.write_overlay(home)
    assert "Made by FirstCut" in p.read_text()


def test_who_looks_at_the_footage(home):
    """Settings › AI › Looking at footage: this Mac by default (free, private); the account when chosen (about 2,000 tokens a shot)."""
    s = SD.Settings()
    s.update({"ai": {"use": "claude", "look": "local"}})
    ov = SD.overlay(s.get(), home)
    assert ov["clipper"]["ai_videos"]["vision"] == "ollama" and "broll_describe" in ov["llm"]["claude"]["skip_tasks"]
    assert s.update({"ai": {"look": "anything"}})["ai"]["look"] == "local"                 # unknown values are ignored
    s.update({"ai": {"look": "account"}})
    ov = SD.overlay(s.get(), home)
    assert ov["clipper"]["ai_videos"]["vision"] == "" and ov["llm"]["claude"]["skip_tasks"] == []
    assert "broll_describe" in ov["llm"]["claude"]["cheap_tasks"] and ov["llm"]["claude"]["models"]["cheap"] == "haiku"   # the quick model
    s.update({"ai": {"use": "local"}})                                                      # only this Mac: everything stays here
    ov = SD.overlay(s.get(), home)
    assert ov["clipper"]["ai_videos"]["vision"] == "ollama" and "broll_describe" in ov["llm"]["claude"]["skip_tasks"]


def test_models_fit_the_memory():
    assert SD.local_models(8)["text"] == "qwen3:4b" and SD.local_models(8)["vision"] == "qwen3-vl:2b"
    assert SD.local_models(16)["text"] == "qwen3:8b"
    assert SD.local_models(32)["vision"] == "qwen3-vl:8b" and SD.local_models(64)["text"] == "qwen3:32b"


def test_how_it_edits_keeps_short_points_per_kind(home):
    st = SD.Styles()
    ai = FakeAI(["Start on the punchline.", "  - Captions bold, one word at a time  ", "start on the punchline.", "x" * 300])
    d = st.add("short", "i like it when it starts on the punchline and captions are bold one word at a time, love you", ai)
    assert d["points"][:2] == ["Start on the punchline.", "Captions bold, one word at a time"] and len(d["points"]) == 3
    assert len(d["points"][2]) <= 160 and d["added"] == d["points"]
    assert "short vertical clips" in ai.prompts[0][1] and "(none yet)" in ai.prompts[0][2]
    assert st.get("long")["points"] == []                                              # the other kind never sees it
    assert "Start on the punchline." in st.brief("short") and st.brief("long") == ""
    d = st.remove("short", 0)
    assert d["points"][0].startswith("Captions")
    d = st.undo("short")
    assert d["points"][0] == "Start on the punchline."
    with pytest.raises(ValueError):
        st.add("short", "   ", ai)
    with pytest.raises(ValueError):
        st.add("reels", "x", ai)
    keep = st.get("short")["points"]
    assert st.add("short", "something", FakeAI([]))["points"] == keep                  # an empty answer never wipes the list


def test_a_chatgpt_skill_is_read_as_text():
    md = b"---\nname: reels\ndescription: x\n---\n# How I edit\nCut every pause.\n```python\nprint(1)\n```\nBold captions.\n"
    t = SD.skill_text("SKILL.md", md)
    assert "Cut every pause." in t and "Bold captions." in t and "print(1)" not in t and "name: reels" not in t
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("reels/notes.txt", "Music low.")
        z.writestr("reels/SKILL.md", md)
        z.writestr("__MACOSX/reels/._SKILL.md", "junk")
        z.writestr("reels/run.py", "import os")
    t = SD.skill_text("reels.zip", buf.getvalue())
    assert t.index("Cut every pause") < t.index("Music low") and "junk" not in t and "import os" not in t


def test_every_project_gets_its_own_folder_and_links_to_the_originals(tmp_path):
    a = SD.project_folder(tmp_path / "Videos", "Garden story")
    b = SD.project_folder(tmp_path / "Videos", "Garden story")
    assert a.name == "Garden story" and b.name == "Garden story (2)" and a.is_dir() and b.is_dir()
    src = tmp_path / "Footage" / "C0001.MP4"
    src.parent.mkdir()
    src.write_bytes(b"x")
    other = tmp_path / "Other" / "C0001.MP4"
    other.parent.mkdir()
    other.write_bytes(b"y")
    assert SD.link_raw(a, [src, other, tmp_path / "gone.mp4"]) == 2
    raw = sorted((a / "Raw footage").iterdir())
    assert [p.name for p in raw] == ["C0001 (2).MP4", "C0001.MP4"] and all(p.is_symlink() for p in raw)
    assert SD.link_raw(a, [src]) == 0                                                  # already linked


def test_setup_only_plans_what_is_needed(home, monkeypatch):
    monkeypatch.setattr(SD.Setup, "_pulled", lambda self: set())
    plan = {s["id"]: s for s in SD.Setup().plan()}
    assert plan["search"]["state"] == "skipped" and plan["speech"]["state"] == "waiting" and plan["vision"]["model"] == "qwen3-vl:4b"
    assert plan["local"]["state"] == "waiting"                                         # the fallback is on by default
    SD.Settings().update({"ai": {"fallback": False}, "library": {"folders": ["/Volumes/X"]}})
    plan = {s["id"]: s for s in SD.Setup().plan()}
    assert plan["local"]["state"] == "skipped" and plan["search"]["state"] != "skipped"
    monkeypatch.setattr(SD.Setup, "_pulled", lambda self: {"qwen3-vl:4b"})
    assert {s["id"]: s for s in SD.Setup().plan()}["vision"]["state"] == "done"


def test_the_shipped_settings_have_nothing_personal(tmp_path):
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "app.yaml"
    r = subprocess.run([sys.executable, str(root / "app" / "VideoBuilder" / "make_config.py"), str(root / "config" / "config.yaml"), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text()
    for bad in ("100.79.", "100.66.", "example", "1000001", "1000002", "owner", "metricool", "timezone:"):
        assert bad not in text.lower()
    import yaml
    d = yaml.safe_load(text)
    assert d["clipper"]["app"] == "studio" and d["catalog"]["roots"] == [] and "publishing" not in d and "brands" not in d


# ---------------------------------------------------------------------------------------------- short clips in the app
@pytest.fixture
def api(home, tmp_path):
    from clipfarm.clipper import studio_web as W
    from clipfarm.clipper.server import clean_options
    from clipfarm.clipper.engine import default_options
    from clipfarm.clipper.store import Store
    store = Store(tmp_path / "jobs")
    cfg = SimpleNamespace(get=lambda k, d=None: {"clipper": {}}.get(k, d))
    worker = SimpleNamespace(busy_with=None, wake=SimpleNamespace(set=lambda: None))
    projects = SimpleNamespace(rules_path=lambda: tmp_path / "rules.md", list=lambda: [])
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.cfg, a.store, a.worker, a.projects, a.runner, a.chatgpt = cfg, store, worker, projects, None, None
    a.default_options, a.clean_options = default_options, clean_options
    a.root, a.settings, a.styles, a.setup = home, SD.Settings(home), SD.Styles(home), SD.Setup(home)
    a.mac, a.moving, a.restart_wanted, a._cache = SD.mac_info(), "", False, {}
    import threading
    a.lock = threading.Lock()
    a.settings.update({"places": {"videos": str(tmp_path / "Videos")}})
    return a


def test_a_video_on_this_mac_becomes_a_clip_job_read_in_place(api, tmp_path):
    src = tmp_path / "Podcast 12.mp4"
    src.write_bytes(b"not really a video")
    api.styles._save("short", ["Start on the punchline."], [], [])
    r = api.make_short({"source": str(src), "notes": "only the funny parts", "options": {"len": "30-60", "caption_style": "pop"}})
    j = api.store.get(r["id"])
    assert j["status"] == "queued" and j["source"]["app"] == "studio" and "local" not in j["source"]  # ("local" = a long-form project)
    assert (api.store.dir(r["id"]) / "source.mp4").resolve() == src.resolve()          # a link: nothing copied
    o = j["options"]
    assert o["min_len"] == 30 and o["max_len"] == 60 and o["caption_style"] == "pop" and o["mode"] == "clips"
    assert "only the funny parts" in o["prompt"] and "Start on the punchline." in o["prompt"]
    home = Path(j["app_home"])
    assert home.parent == tmp_path / "Videos" and (home / "Raw footage" / src.name).is_symlink()
    assert api.store.next_queued()["id"] == r["id"]
    with pytest.raises(ValueError):
        api.make_short({"source": str(tmp_path / "notes.txt")})
    link = api.make_short({"source": "https://www.youtube.com/watch?v=abc"})
    assert api.store.get(link["id"])["source"]["url"].startswith("https://")


def test_finished_clips_are_saved_and_untick_moves_them_aside(api, tmp_path):
    src = tmp_path / "talk.mp4"
    src.write_bytes(b"x")
    r = api.make_short({"source": str(src)})
    jid = r["id"]
    d = api.store.dir(jid)
    for i in range(2):
        (d / f"clip_{i}.mp4").write_bytes(b"video%d" % i)
        (d / f"clip_{i}.srt").write_text("1\n")
        api.store.set_clip(jid, {"index": i, "file": f"clip_{i}.mp4", "srt": f"clip_{i}.srt", "title": f"Funny bit {i}", "rendered_at": 1})
    api.store.update(jid, status="done")
    api.save_clips(api.store.get(jid))
    home = Path(api.store.get(jid)["app_home"])
    assert sorted(p.name for p in (home / "Clips").glob("*.mp4")) == ["01 Funny bit 0.mp4", "02 Funny bit 1.mp4"]
    api.keep_clip(jid, {"index": 1, "keep": False})
    assert not (home / "Clips" / "02 Funny bit 1.mp4").exists() and (home / "Clips" / "Not kept" / "02 Funny bit 1.mp4").exists()
    clip = next(c for c in api.store.get(jid)["clips"] if c["index"] == 0)
    (d / "clip_0.mp4").write_bytes(b"changed")
    api.store.set_clip(jid, {**clip, "rendered_at": 2})                                # a clip changed in plain words is copied again
    api.save_clips(api.store.get(jid))
    assert (home / "Clips" / "01 Funny bit 0.mp4").read_bytes() == b"changed"
    api.remake_short(jid, {"options": {"len": "15-30"}})
    j = api.store.get(jid)
    assert j["status"] == "queued" and j["clips"] == [] and j["options"]["max_len"] == 30
    assert not (home / "Clips").exists() and any(p.name.startswith("Clips (before") for p in home.iterdir())


def test_a_drive_that_is_not_plugged_in_is_named(tmp_path):
    assert SD.missing_drive("/Volumes/NoSuchDrive-xyz/Models") == "NoSuchDrive-xyz"
    assert SD.missing_drive(tmp_path / "Models") is None and SD.missing_drive("/Volumes") is None


def test_the_app_never_answers_posting_or_feed_routes(home, tmp_path):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    from clipfarm.clipper.localproj import LocalRunner
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
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
        for path, body in (("/feed", None), ("/api/feed", None), ("/longform", None), ("/api/jobs/x/clips/2/post_next", b"{}"), ("/api/feed/rate", b"{}")):
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(urllib.request.Request(base + path, data=body, headers={"Content-Type": "application/json"}), timeout=10)
            assert e.value.code == 404, path
        with urllib.request.urlopen(base + "/api/app/ping", timeout=10) as r:
            assert json.loads(r.read())["ok"] is True
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            assert b"<title>FirstCut</title>" in r.read()
    finally:
        httpd.shutdown()


def test_a_new_footage_folder_is_used_at_once_without_a_restart(api, tmp_path):
    first = []
    cat = SimpleNamespace(roots=[], window=None, first=lambda folders, hours=None: first.append(list(folders)),
                          enabled=lambda: bool(cat.roots))
    api.runner = SimpleNamespace(catalog=cat, running=False, library=lambda: cat if cat.roots else None)
    a, b = tmp_path / "Big B-roll", tmp_path / "Drive2" / "B-roll"
    a.mkdir(); b.mkdir(parents=True)
    s = api.library_action("add", {"path": str(a)})["settings"]
    assert s["library"]["folders"] == [str(a)] and s["library"]["look"] == str(a) and first == [[str(a)]]
    assert cat.roots == [a] and cat.window == "always" and not api.restart_wanted
    api.library_action("add", {"path": str(b)})
    api.library_action("use", {"path": str(a)})
    s = api.settings.get()["library"]
    assert s["folders"] == [str(a), str(b)] and s["look"] == str(a) and s["recent"][str(a)] >= s["recent"][str(b)]
    assert api.library_action("use", {"path": ""})["settings"]["library"]["look"] == ""
    api.library_action("catalog", {"path": str(b)})
    assert first[-1] == [str(b)]
    with pytest.raises(KeyError):
        api.library_action("use", {"path": "/nowhere"})
    with pytest.raises(ValueError):
        api.library_action("add", {"path": str(tmp_path / "missing")})
    s = api.settings.update({"library": {"when": "off"}})
    api.apply_library(s)
    assert cat.window is None and not api.restart_wanted


def test_settings_remember_where_find_footage_looked(home):
    s = SD.Settings(home)
    out = s.update({"library": {"folders": ["/A", "/A/B"], "look": "/A/B", "recent": {"/A": 5, "rel": 1, "/B": "x"}}})
    assert out["library"]["folders"] == ["/A", "/A/B"]                                 # a folder inside another may be its own source
    assert out["library"]["look"] == "/A/B" and out["library"]["recent"] == {"/A": 5.0}
    assert s.update({"library": {"look": "relative"}})["library"]["look"] == ""


def test_a_mac_that_had_video_builder_keeps_its_folder(tmp_path, monkeypatch):
    monkeypatch.delenv("VIDEOBUILDER_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    support = tmp_path / "Library" / "Application Support"
    assert SD.home() == support / "FirstCut"                                          # a new Mac: the app's own name
    (support / "Video Builder").mkdir(parents=True)
    assert SD.home() == support / "FirstCut"                                          # an empty leftover folder doesn't count
    (support / "Video Builder" / "settings.json").write_text("{}")
    assert SD.home() == support / "Video Builder"                                     # the first version's Python and models stay put
    (support / "FirstCut").mkdir()
    assert SD.home() == support / "FirstCut"


def test_a_link_is_named_after_its_video_straight_away(api, monkeypatch):
    from clipfarm.clipper import studio_web as W
    monkeypatch.setattr(W, "link_title", lambda url: "  How We   Built It  ")
    r = api.make_short({"source": "https://www.youtube.com/watch?v=abc"})
    import time as _t
    for _ in range(50):
        if api.store.get(r["id"])["title"] != "Clips from a link":
            break
        _t.sleep(0.05)
    assert api.store.get(r["id"])["title"] == "How We Built It"
    monkeypatch.setattr(W, "link_title", lambda url: (_ for _ in ()).throw(OSError("offline")))
    r2 = api.make_short({"source": "https://www.youtube.com/watch?v=def"})
    _t.sleep(0.2)
    assert api.store.get(r2["id"])["title"] == "Clips from a link"                      # the download names it later


def test_show_in_finder_opens_the_nearest_folder_that_exists(api, tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as W
    opened = []
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **kw: opened.append(cmd))
    home = tmp_path / "Videos" / "Talk"
    home.mkdir(parents=True)
    assert api.post(["open"], {"path": str(home / "Clips"), "nearest": True})[0] == 200
    assert opened[-1] == ["open", str(home)]                                          # the clips aren't made yet: the project folder
    assert api.post(["open"], {"path": str(home / "Clips")})[0] == 404                 # (without "nearest" it says it isn't there)


def test_setup_carries_on_after_the_app_quits(home, monkeypatch):
    st = SD.Setup(home)
    ran = []
    monkeypatch.setattr(SD.Setup, "plan", lambda self: [{"id": "speech", "state": "done" if ran else "waiting"}])
    monkeypatch.setattr(SD.Setup, "_do_speech", lambda self, s: ran.append(1), raising=False)
    assert not st.resume()                                                           # never started: nothing to carry on with
    (home / SD.WANTED).parent.mkdir(parents=True, exist_ok=True)
    (home / SD.WANTED).write_text("1")                                               # it was downloading when the app quit
    again = SD.Setup(home)
    assert again.resume()
    again.thread.join(5)
    assert ran == [1] and not (home / SD.WANTED).exists()                            # finished: nothing left to resume
    assert not SD.Setup(home).resume()


def test_a_long_video_carries_on_after_the_app_quits(tmp_path):
    import threading
    from clipfarm.clipper.localproj import LocalRunner
    from clipfarm.clipper.store import Store
    store = Store(tmp_path / "jobs")
    job = store.create({"local": True, "app": "studio"}, {}, status="draft")
    projects = SimpleNamespace(get=lambda pid: {"id": pid, "job": job["id"]},
                               list=lambda: [{"id": "p1", "job": job["id"], "pending_render": state["pending"]}])
    state = {"pending": False}
    runner = LocalRunner(SimpleNamespace(get=lambda k, d=None: d), store, projects)
    done, gate = [], threading.Event()
    runner.make = lambda pid, body=None: (gate.wait(5), done.append(("make", body)))
    runner.rerender = lambda pid: done.append(("rerender", None))
    runner.start("p1", "make", body={"notes": "2 minutes"})
    run = store.get(job["id"])["run"]
    assert {k: run[k] for k in ("kind", "kw")} == {"kind": "make", "kw": {"body": {"notes": "2 minutes"}}} and run["at"] > 0   # (and since when)
    store.update(job["id"], status="rendering")                                       # ...and the app quits here
    assert store.requeue_interrupted(resume_local=True) == []
    j = store.get(job["id"])
    assert j["status"] == "queued" and j["stage"] == "Picking up where it left off"
    gate.set()
    for _ in range(100):
        if not runner.running:
            break
        threading.Event().wait(0.02)
    store.update(job["id"], status="queued", run={"kind": "make", "kw": {"body": {"notes": "2 minutes"}}})
    state["pending"] = True                                                           # the edit was already decided
    assert runner.resume() == "p1"
    for _ in range(100):
        if not runner.running:
            break
        threading.Event().wait(0.02)
    assert done[-1] == ("rerender", None) and store.get(job["id"])["run"] is None    # rendered as decided, not planned again
    other = store.create({"local": True}, {}, status="transcribing")
    store.requeue_interrupted(resume_local=False)                                      # the personal Clipper: says what happened
    assert store.get(other["id"])["status"] == "failed"


def test_every_graphics_style_has_a_look_and_a_preview(api, tmp_path):
    from PIL import Image
    from clipfarm.clipper import gfxcards as G
    from clipfarm.clipper.graphics import STYLE_NAMES, THEMES
    assert set(STYLE_NAMES) <= set(THEMES) and {"broadcast", "glass_amber", "pastel_yellow"} <= set(STYLE_NAMES)
    assert all(k in G.MODES for k in THEMES)                                          # every look is drawn by the card renderer
    api.settings.update({"places": {"work": str(tmp_path / "work")}})
    for st in api.gfx_styles():
        for kind in ("lower", "point", "stat"):
            im = Image.open(api.gfx_preview(st["id"], kind + ".jpg"))
            assert im.size == (640, 360)
    with pytest.raises(KeyError):
        api.gfx_preview("no-such-look", "point.jpg")
    with pytest.raises(KeyError):
        api.gfx_preview("glass", "chart.jpg")


def test_the_style_preview_draws_the_caption_and_the_safe_box_in_both_shapes(api, tmp_path):
    from PIL import Image
    import numpy as np
    api.settings.update({"places": {"work": str(tmp_path / "work")}})
    wide = Image.open(api.gfx_preview("broadcast", "lower.jpg"))
    tall = Image.open(api.gfx_preview("broadcast", "lower.jpg", tall=True))
    assert wide.size == (640, 360) and tall.size == (360, 640)
    assert api.gfx_preview("broadcast", "lower.jpg") != api.gfx_preview("broadcast", "lower.jpg", tall=True)
    for im in (wide, tall):                                                        # a white caption sits in the caption band: bright pixels with a dark outline around them
        a = np.asarray(im.convert("L"))
        H = a.shape[0]
        band = a[int(H * (0.55 if im is tall else 0.8)):int(H * (0.68 if im is tall else 0.92))]
        assert (band > 240).sum() > 30


def test_the_picked_graphics_style_reaches_the_project(home):
    s = SD.Settings(home)
    d = s.update({"defaults": {"long": {"graphics": "all", "gfx_style": "broadcast"}}})["defaults"]["long"]
    assert d["graphics"] == "all" and d["gfx_style"] == "broadcast"
    d = s.update({"defaults": {"long": {"graphics": "sparkles", "gfx_style": "../../x"}}})["defaults"]["long"]
    assert d["graphics"] == "" and d["gfx_style"] == "auto"


def test_the_plans_usage_is_read_from_claudes_answers(tmp_path):
    from clipfarm.clipper.claude_cli import ClaudeCLI
    lines = [{"type": "system", "subtype": "init"},
             {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed_warning", "rateLimitType": "seven_day", "utilization": 0.87,
                                                              "unifiedWindows": {"five_hour": {"utilization": 0.26, "resetsAt": 1000},
                                                                                 "seven_day": {"utilization": 0.87, "resetsAt": 2000}}}},
             {"type": "result", "structured_output": {"word": "ok"}, "modelUsage": {"claude-haiku": {}}}]
    out = "\n".join(json.dumps(x) for x in lines).encode()
    cfg = SimpleNamespace(get=lambda k, d=None: {"llm.claude": {"enabled": True, "config_dir": str(tmp_path)}}.get(k, d))
    c = ClaudeCLI(cfg, run=lambda cmd, **kw: SimpleNamespace(stdout=out, stderr=b"", returncode=0))
    data, model = c.complete_json("s", "p", {"type": "object"}, model="haiku")
    assert data == {"word": "ok"} and model == "claude-haiku"
    u = json.loads((tmp_path / "usage.json").read_text())
    assert u["five_hour"] == {"used": 0.26, "resets": 1000.0} and u["week"] == {"used": 0.87, "resets": 2000.0} and u["status"] == "allowed_warning"


def test_chatgpt_usage_comes_from_its_headers(tmp_path):
    from clipfarm.clipper.chatgpt_plan import ChatGPTPlan
    cfg = SimpleNamespace(root=tmp_path, get=lambda k, d=None: d)
    plan = ChatGPTPlan(cfg)
    plan.save_usage({"x-codex-primary-used-percent": "40", "x-codex-primary-window-minutes": "300", "x-codex-primary-reset-after-seconds": "600",
                     "x-codex-secondary-used-percent": "75", "x-codex-secondary-window-minutes": "10080"})
    u = plan.usage()
    assert u["five_hour"]["used"] == 0.4 and u["week"]["used"] == 0.75
    plan.save_usage({"content-type": "text/event-stream"})                            # no usage headers: the last figures stay
    assert plan.usage()["week"]["used"] == 0.75


def test_a_window_that_has_reset_reads_empty(api):
    (api.root / "claude").mkdir(parents=True, exist_ok=True)
    (api.root / "claude" / "usage.json").write_text(json.dumps({"at": 1, "five_hour": {"used": 0.99, "resets": 10}, "week": {"used": 0.5, "resets": 4e9}}))
    u = api.usage("claude")
    assert u["five_hour"]["used"] == 0.0 and u["five_hour"]["reset"] and u["week"]["used"] == 0.5
    assert api.usage("local") is None


def test_a_long_video_opens_in_each_editor_on_this_mac(api, tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as W
    home = tmp_path / "Videos" / "Open house"
    home.mkdir(parents=True)
    (home / "Open house.fcpxml").write_text("<fcpxml/>")
    api.projects = SimpleNamespace(get=lambda pid: {"id": pid, "job": "j1", "home": str(home), "folder": str(tmp_path / "Footage")})
    api.store.create({}, {}, status="done")
    monkeypatch.setattr(api.store, "get", lambda jid: {"id": jid, "status": "done"})
    ran = []
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **kw: ran.append(cmd))
    monkeypatch.setattr(SD, "editors", lambda: [{"id": "fcp", "name": "Final Cut Pro", "app": "/A/FCP.app", "how": "opens"},
                                                {"id": "resolve", "name": "DaVinci Resolve", "app": "/A/Resolve.app", "how": "imports"}])
    assert api.item_action("long", "p1", "open-in", {"editor": "fcp"}) == {"ok": True}
    assert ran[-1] == ["open", "-a", "/A/FCP.app", str(home / "Open house.fcpxml")]
    r = api.item_action("long", "p1", "open-in", {"editor": "resolve"})
    assert "Import" in r["note"] and ran[-2] == ["open", "-a", "/A/Resolve.app"] and ran[-1][:2] == ["open", "-R"]
    with pytest.raises(ValueError):
        api.item_action("long", "p1", "open-in", {"editor": "premiere"})


def test_a_project_is_renamed_and_removed_but_its_footage_never_trashed(api, tmp_path):
    src = tmp_path / "talk.mp4"
    src.write_bytes(b"x")
    jid = api.make_short({"source": str(src)})["id"]
    api.store.update(jid, status="done")
    assert api.item_action("short", jid, "rename", {"title": "  Best   bits "})["title"] == "Best bits"
    assert api.store.get(jid)["title"] == "Best bits"
    folder = api.item_action("short", jid, "folder", {})["folder"]
    assert folder and Path(folder).parent == tmp_path / "Videos"                     # only the folder FirstCut made
    proj = {"id": "p9", "job": jid, "home": str(tmp_path / "Footage"), "folder": str(tmp_path / "Footage")}
    (tmp_path / "Footage").mkdir()
    api.projects = SimpleNamespace(get=lambda pid: proj)
    assert api.item_action("long", "p9", "folder", {})["folder"] is None              # the footage folder itself: never offered for the Trash
    assert api.item_action("short", jid, "remove", {}) == {"ok": True}
    with pytest.raises(KeyError):
        api.store.get(jid)


def test_earlier_edits_in_a_project_folder_are_not_used_as_footage(tmp_path):
    from clipfarm.clipper import localproj as LP
    proj = tmp_path / "Weekend Workshop"
    for rel in ("Raw Video Clips/C0001.MP4", "Raw Video Clips/Draft take.mp4", "Draft 1.mov", "Weekend Workshop Video V2.mov",
                "Interview.mov", "Photographs/DSC0001.jpg", "Audio Clips/ZOOM0001.WAV"):
        (proj / rel).parent.mkdir(parents=True, exist_ok=True)
        (proj / rel).write_bytes(b"x" * 2048)
    assert sorted(LP.earlier_edits(proj)) == ["Draft 1.mov", "Weekend Workshop Video V2.mov"]
    keep, left = LP._videos(proj)
    assert sorted(p.name for p in keep) == ["C0001.MP4", "Draft take.mp4", "Interview.mov"]       # (a "draft" among camera files stays)
    assert LP.summarize(proj, files=[])["left_out"] == LP.earlier_edits(proj)
    only = tmp_path / "Banquet"
    (only / "Final interview.mov").parent.mkdir(parents=True)
    (only / "Final interview.mov").write_bytes(b"x" * 2048)
    assert LP._videos(only)[1] == [] and len(LP._videos(only)[0]) == 1                # all that's there: used, not left out
    for name, edit in (("MVI_0012.MOV", False), ("Masterclass interview.mov", False), ("Final cut v3.mov", True), ("GX010012.MP4", False),
                       ("Weekend recap.mov", False), ("Weekend Workshop - FINAL.mp4", True), ("A-roll/Final thoughts.mov", False)):
        assert LP.earlier_edit(Path(name), proj) is edit, name


def test_picked_shots_and_add_broll_become_folders_of_links_to_the_originals(api, tmp_path):
    a, b = tmp_path / "Drive" / "Graduation", tmp_path / "Drive" / "Campus"
    a.mkdir(parents=True), b.mkdir(parents=True)
    for f in (a / "C0001.MP4", b / "C0001.MP4", a / "notes.txt"):
        f.write_bytes(b"x" * 2048)
    folder = Path(api.picked_folder([str(a / "C0001.MP4"), str(b / "C0001.MP4"), str(a / "C0001.MP4")]))
    links = sorted(folder.iterdir())
    assert [p.name for p in links] == ["C0001.MP4", "Campus - C0001.MP4"] and all(p.is_symlink() for p in links)   # same name twice: both kept
    assert {Path(os.readlink(p)) for p in links} == {a / "C0001.MP4", b / "C0001.MP4"}
    with pytest.raises(ValueError):
        api.picked_folder([str(a / "notes.txt")])
    with pytest.raises(ValueError):
        api.picked_folder([])
    talk = tmp_path / "Edit" / "Interview final.mov"
    talk.parent.mkdir()
    talk.write_bytes(b"x" * 2048)
    folder, aroll = api.broll_folder(str(talk), str(tmp_path / "Drive"))
    names = sorted(p.name for p in Path(folder).iterdir())
    assert aroll == "Interview final.mov" and aroll in names and len(names) == 3                # the video + both B-roll clips
    with pytest.raises(ValueError):
        api.broll_folder(str(tmp_path / "Drive"))                                            # a folder is not "your video"
    with pytest.raises(ValueError):
        api.broll_folder(str(talk), "relative/folder")


def test_new_footage_in_a_source_is_counted_for_the_find_page(api, tmp_path):
    import time as _t
    added = []
    cat = SimpleNamespace(roots=[tmp_path / "A", tmp_path / "B"], under=lambda path, root: str(path).startswith(str(root) + "/"),
                          scan=lambda: [{"path": str(tmp_path / "A" / "x.mp4")}, {"path": str(tmp_path / "A" / "y.mp4")}, {"path": str(tmp_path / "B" / "z.mp4")}],
                          todo=lambda found: [f for f in found if not f["path"].endswith("z.mp4")])
    for d in cat.roots:
        d.mkdir()
    api.runner = SimpleNamespace(library=lambda: cat, running=False)
    first = api.library_new()
    assert first["checking"] is True                                                  # counted in the background: the page asks again
    api._new_thread.join(5)
    r = api.library_new()
    assert r["checking"] is False and {x["path"]: x["new"] for x in r["roots"]} == {str(tmp_path / "A"): 2, str(tmp_path / "B"): 0}
    shot = {"id": "ab12cd34ef567890", "start": 12.5, "end": 16.5, "what": "graduates", "rel": "B-roll/x.mp4"}
    api.library_action("picked", {"shot": shot})
    api.library_action("picked", {"shot": {**shot, "start": 30}})
    api.library_action("picked", {"shot": shot})                                       # picked again: first, once
    got = api.settings.get()["library"]["picked"]
    assert [x["start"] for x in got] == [12.5, 30.0] and got[0]["what"] == "graduates"


def test_pace_choice_reaches_the_engine(home):
    """Pace: Automatic / Fast-paced cuts / Take it slow (2026-11-05: "Is there an option to edit faster or slower? There should be.")."""
    from clipfarm.clipper.engine import PACES
    s = SD.Settings()
    assert s.get()["defaults"]["long"]["pace"] == "auto" and s.get()["defaults"]["short"]["pace"] == "auto"
    assert s.update({"defaults": {"long": {"pace": "warp"}}})["defaults"]["long"]["pace"] == "auto"
    assert s.update({"defaults": {"short": {"pace": "slow"}}})["defaults"]["short"]["pace"] == "slow"
    for kind, m in SD.ENGINE_PACE.items():
        assert set(m) == set(SD.PACE_CHOICES) and all(v in PACES for v in m.values())
    assert PACES["slow"]["gap"] > PACES["natural"]["gap"] > PACES["tight"]["gap"] >= PACES["fast"]["gap"]
