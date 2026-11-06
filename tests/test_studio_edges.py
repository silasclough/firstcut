"""Edges of Video Builder (the app), found by trying to break it: settings with the wrong types, library folders given as a string or twice,
the engine settings (overlay.yaml) for every AI choice read back through load_config, the Setup plan on an 8 GB and a 64 GB Mac, "how it
edits" with an AI that answers badly, ChatGPT skills that are not what they claim, odd paths and titles for videos and folders, keep and
make-again on clips that are running or don't exist, moving the models folder, the folder picker, the real HTTP routes (/ and /api/app),
and real ffmpeg clips saved into the project folder. The AI, Ollama, Claude and every download are faked; nothing leaves this Mac.

Tests marked "BUG" describe what the app should do and fail on the code as it is (see the tester's report)."""

import base64
import io
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from clipfarm.clipper import studio as SD
from clipfarm.clipper import studio_web as W
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

REPO = Path(__file__).resolve().parents[1]
ODD = ['clip with spaces.mp4', '10:30 interview.MP4', 'café ñ 日本語.mov', 'quote\'s "double".mp4', 'hash#and?q%20x.m4v']


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "VB home"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    monkeypatch.setattr(SD, "_answers", lambda url: False)                 # no Ollama running in tests
    monkeypatch.setattr(SD, "ram_gb", lambda: 16.0)
    return h


class FakeAI:
    def __init__(self, data):
        self.data, self.calls = data, []

    def complete_json(self, task, system, prompt, schema, **kw):
        self.calls.append((task, prompt))
        return SimpleNamespace(data=self.data)


def _api(home, tmp_path, projects=None, runner=None, chatgpt=None):
    """A StudioAPI without its background thread (that thread ends the process with code 75 when a restart is wanted)."""
    from clipfarm.clipper.engine import default_options
    from clipfarm.clipper.server import clean_options
    from clipfarm.clipper.store import Store
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.cfg = SimpleNamespace(get=lambda k, d=None: {"clipper": {}}.get(k, d))
    a.store = Store(tmp_path / "jobs")
    a.worker = SimpleNamespace(busy_with=None, wake=SimpleNamespace(set=lambda: None))
    a.projects = projects or SimpleNamespace(rules_path=lambda: tmp_path / "rules.md", list=lambda: [])
    a.runner, a.chatgpt = runner, chatgpt
    a.default_options, a.clean_options = default_options, clean_options
    a.root, a.settings, a.styles, a.setup = home, SD.Settings(home), SD.Styles(home), SD.Setup(home)
    a.claude = SD.ClaudeLogin(home)
    a.mac, a.moving, a.restart_wanted, a._cache = {"chip": "Apple M3", "ram_gb": 16, "disk_free_gb": 100, "arm64": True}, "", False, {}
    a.lock = threading.Lock()
    a.settings.update({"places": {"videos": str(tmp_path / "Videos ✓")}})
    return a


@pytest.fixture
def api(home, tmp_path):
    return _api(home, tmp_path)


def _done(api, jid, titles, files=True):
    d = api.store.dir(jid)
    for i, t in enumerate(titles):
        if files:
            (d / f"clip_{i:02d}.mp4").write_bytes(b"video %d" % i)
            (d / f"clip_{i:02d}.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        api.store.set_clip(jid, {"index": i, "file": f"clip_{i:02d}.mp4", "srt": f"clip_{i:02d}.srt", "title": t, "rendered_at": time.time()})
    api.store.update(jid, status="done")
    api.save_clips(api.store.get(jid))


# ---------------------------------------------------------------------------------------------- settings
def test_settings_values_are_clamped_and_unknown_words_ignored(home):
    s = SD.Settings()
    out = s.update({"appearance": "neon", "ai": {"use": "gemini", "local_model": "Qwen3:8B; rm"},
                    "defaults": {"short": {"min_len": "nan", "max_len": 1e9, "clips": -4, "caption_style": "x" * 99, "library": "yes"},
                                 "reels": {"min_len": 5}}})
    assert out["appearance"] == "system" and out["ai"]["use"] == "chatgpt" and out["ai"]["local_model"] is None
    d = out["defaults"]["short"]
    assert 0 <= d["min_len"] <= 600 and d["max_len"] == 600 and d["clips"] == 0 and len(d["caption_style"]) == 20
    assert "reels" not in out["defaults"]
    assert json.loads((home / "settings.json").read_text())["appearance"] == "system"
    (home / "settings.json").write_text("{not json")                           # a damaged file: the defaults, not a crash
    assert SD.Settings().get()["ai"]["use"] == "chatgpt"


@pytest.mark.parametrize("bad", [{"ai": "local"}, {"places": ["/tmp"]}, {"defaults": "short"}, {"setup": "yes"}, {"library": {"folders": 3}}])
def test_settings_with_the_wrong_types_are_refused_or_ignored_not_a_crash(home, bad):
    """BUG: Settings.update / clean() calls .get / .items on whatever arrives; POST /api/app/settings {"ai": "local"} answers 500
    "'str' object has no attribute 'get'" instead of 400 (or ignoring it)."""
    s = SD.Settings()
    before = s.get()
    try:
        after = s.update(bad)
    except (ValueError, TypeError):                                             # (the server answers these with 400)
        after = s.get()
    assert after == before


def test_library_folders_given_as_one_string_never_become_single_letters(home):
    """BUG: {"library": {"folders": "/Volumes/Footage"}} is iterated letter by letter and saves ["/", "V", "o", ...]: the footage catalog
    then has "/" (the whole disk) as a root. Seen for real through POST /api/app/settings and overlay.yaml."""
    out = SD.Settings().update({"library": {"folders": "/Volumes/Footage"}})
    assert "/" not in out["library"]["folders"] and all(len(f) > 1 for f in out["library"]["folders"])


def test_library_folders_are_full_paths_and_listed_once(home):
    """BUG: the same folder added twice (the page's "Add a folder…" appends without checking) is kept twice, and a relative folder is
    kept (places refuse relative paths; library folders don't)."""
    out = SD.Settings().update({"library": {"folders": ["/Volumes/X/B-roll", "/Volumes/X/B-roll", "relative/folder"]}})
    assert out["library"]["folders"] == ["/Volumes/X/B-roll"]


def test_settings_that_change_the_engine_ask_for_a_restart_and_appearance_does_not(api, home):
    code, out = api.post(["settings"], {"appearance": "dark"})
    assert code == 200 and out["settings"]["appearance"] == "dark" and api.restart_wanted is False
    code, out = api.post(["settings"], {"defaults": {"short": {"caption_style": "pop"}}})
    assert api.restart_wanted is False                                          # defaults are read per video, no restart
    code, out = api.post(["settings"], {"ai": {"use": "local"}})
    assert api.restart_wanted is True
    ov = yaml.safe_load((home / "overlay.yaml").read_text())
    assert ov["clipper"]["ai_videos"]["primary"] == "local" and ov["clipper"]["ai_videos"]["fallback"] == "ollama"


def test_restart_waits_while_something_is_being_made(api):
    api.worker.busy_with = "job1"
    with pytest.raises(ValueError):
        api.post(["restart"], {})
    assert api.post(["restart"], {"force": True})[0] == 200 and api.restart_wanted is True
    assert api.get(["busy"]) == (200, {"what": "clips are being made"})
    assert api.get(["ping"]) == (200, {"ok": True}) and api.get(["nope"])[0] == 404 and api.post(["nope"], {})[0] == 404


# ---------------------------------------------------------------------------------------------- the engine's settings
def _app_cfg(home, tmp_path, monkeypatch):
    app = tmp_path / "cfg" / "app.yaml"
    app.parent.mkdir(exist_ok=True)
    r = subprocess.run([sys.executable, str(REPO / "app" / "VideoBuilder" / "make_config.py"), str(REPO / "config" / "config.yaml"), str(app)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    monkeypatch.setenv("CLIPFARM_OVERLAY", str(SD.write_overlay(home)))
    return load_config(app, root=home)


@pytest.mark.parametrize("use,fallback", [("chatgpt", True), ("chatgpt", False), ("claude", True), ("claude", False), ("local", False)])
def test_overlay_yaml_is_read_back_by_load_config_for_every_ai_choice(home, tmp_path, monkeypatch, use, fallback):
    odd = tmp_path / 'Vidéos "finished" 10:30'
    SD.Settings().update({"ai": {"use": use, "fallback": fallback}, "places": {"videos": str(odd), "work": str(tmp_path / "work ✓"),
                                                                               "models": str(tmp_path / "M o d"), "catalog": str(tmp_path / "Cat")}})
    cfg = _app_cfg(home, tmp_path, monkeypatch)
    assert cfg.get("clipper.app") == "studio" and cfg.get("clipper.background") is False and cfg.get("clipper.auto.enabled") is False
    assert cfg.get("clipper.ai_videos.out_dir") == str(odd) and cfg.get("clipper.data_dir") == str(tmp_path / "work ✓" / "clips")
    assert cfg.get("clipper.ai_videos.openai.enabled") is False                    # never the API or the posting server
    assert cfg.get("clipper.ai_videos.chatgpt_plan.enabled") is True and cfg.get("llm.claude.enabled") is True   # both, when signed in
    assert cfg.get("clipper.ai_videos.primary") == use and cfg.get("clipper.ai_videos.backup") == "auto"
    assert cfg.get("llm.claude.config_dir") == str(home / "claude")
    assert cfg.get("clipper.ai_videos.fallback") == ("ollama" if (fallback or use == "local") else "")
    assert cfg.get("catalog.roots") == [] and cfg.get("clipper.ai_videos.default_rules") is False
    assert cfg.get("clipper.ai_videos.vision") == ("ollama" if use == "local" else "")     # (the account looks at footage by default)
    assert cfg.get("clipper.ai_videos.vision_model") == "qwen3-vl:4b"
    assert cfg.path("logs") == home / "logs"


def test_a_chatgpt_user_without_the_fallback_is_told_about_chatgpt_not_claude(home, tmp_path, monkeypatch):
    """BUG: with ChatGPT chosen, not connected yet and the fallback off, LocalAI.available() answers "Claude is not available on this Mac
    and the local AI fallback is off (clipper.ai_videos.fallback: ollama)" - the text a clip job fails with on the page."""
    from clipfarm.clipper import localai as LA
    SD.Settings().update({"ai": {"use": "chatgpt", "fallback": False}})
    ok, why = LA.LocalAI(_app_cfg(home, tmp_path, monkeypatch)).available()
    assert ok is False and "ChatGPT" in why and "Claude" not in why and "clipper.ai_videos" not in why


def test_the_fallback_finishes_the_video_when_the_account_fails(home, tmp_path, monkeypatch):
    """Done means 6: with the fallback on, ChatGPT failing (expired sign-in) or Claude out of usage hands the call to the local model;
    with it off the call fails."""
    from clipfarm.clipper import chatgpt_plan as CP
    from clipfarm.clipper import claude_cli as CC
    from clipfarm.clipper import localai as LA
    from clipfarm.core import llm as L
    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    local = []
    monkeypatch.setattr(L.OllamaLLM, "complete_json", lambda self, s, p, sc, images=None: local.append(self.cfg["model"]) or
                        L.LLMResult({"from": "local"}, "ollama", self.cfg["model"], {}))
    monkeypatch.setattr(CP.ChatGPTPlan, "connected", lambda self: True)

    def expired(self, *a, **k):
        raise CP.ChatGPTPlanError("the ChatGPT sign-in expired")
    monkeypatch.setattr(CP.ChatGPTPlan, "complete_json", expired)
    SD.Settings().update({"ai": {"use": "chatgpt", "fallback": True}})
    ai = LA.LocalAI(_app_cfg(home, tmp_path, monkeypatch))
    assert ai.complete_json("clip_title", "s", "p", {}).data == {"from": "local"} and local == ["qwen3:8b"]
    SD.Settings().update({"ai": {"fallback": False}})
    with pytest.raises(L.LLMUnavailable):
        LA.LocalAI(_app_cfg(home, tmp_path, monkeypatch)).complete_json("clip_title", "s", "p", {})
    SD.Settings().update({"ai": {"use": "claude", "fallback": True}})
    monkeypatch.setattr(CC.ClaudeCLI, "available", lambda self: self.enabled and time.time() >= self.tripped_until)

    def out_of_usage(self, *a, **k):
        raise CC.ClaudeError("You've hit your usage limit")
    monkeypatch.setattr(CC.ClaudeCLI, "complete_json", out_of_usage)
    assert LA.LocalAI(_app_cfg(home, tmp_path, monkeypatch)).complete_json("broll_plan", "s", "p", {}).provider == "ollama"


def test_the_overlay_with_duplicate_and_unplugged_library_folders(home, tmp_path):
    lib = tmp_path / "Footage"
    lib.mkdir()
    SD.Settings().update({"library": {"folders": [str(lib), "/Volumes/Unplugged Drive/B-roll", "", "  "]}})
    ov = SD.overlay(SD.Settings().get(), home)
    assert ov["catalog"]["roots"] == [str(lib), "/Volumes/Unplugged Drive/B-roll"]          # an unplugged drive stays listed
    assert ov["catalog"]["window"] == "always"                                          # (whenever the app is open)
    p = SD.write_overlay(home)
    assert yaml.safe_load(p.read_text())["catalog"]["roots"][1] == "/Volumes/Unplugged Drive/B-roll"


def test_the_catalog_lists_a_file_once_when_folders_repeat_or_nest(tmp_path):
    """BUG: catalog roots are not de-duplicated anywhere (settings, overlay, Catalog): the same folder twice, or a folder and its own
    subfolder, find the same file several times and todo() catalogues it once per copy (seen for real: 3 entries, 1 unique id)."""
    from clipfarm.clipper import catalog as C
    lib = tmp_path / "lib folder"
    (lib / "deep" / "er").mkdir(parents=True)
    (lib / "deep" / "er" / "shot.mp4").write_bytes(b"x" * 4096)
    cfg = SimpleNamespace(root=tmp_path, get=lambda k, d=None: {"catalog": {"roots": [str(lib), str(lib), str(lib / "deep")],
                                                                              "dir": str(tmp_path / "cat")}}.get(k, d))
    found = C.Catalog(cfg).scan()
    assert len(found) == len({f["id"] for f in found}) == 1


def test_environment_puts_models_in_the_chosen_folder(home, tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOBUILDER_BIN", "/Applications/Video Builder.app/Contents/Resources/bin")
    SD.Settings().update({"places": {"models": str(tmp_path / "External ✓" / "Models")}})
    env = SD.environment(home)
    assert env["HF_HOME"] == str(tmp_path / "External ✓" / "Models" / "huggingface") and env["OLLAMA_MODELS"].endswith("Models/ollama")
    assert env["PATH"].split(os.pathsep)[:2] == [str(home / "runtime" / "bin"), "/Applications/Video Builder.app/Contents/Resources/bin"]


# ---------------------------------------------------------------------------------------------- setup
@pytest.mark.parametrize("ram,text,vision,total", [(8.0, "qwen3:4b", "qwen3-vl:2b", 5.8), (16.0, "qwen3:8b", "qwen3-vl:4b", 9.9),
                                                  (24.0, "qwen3:8b", "qwen3-vl:4b", 9.9), (36.0, "qwen3:14b", "qwen3-vl:8b", 16.8),
                                                  (64.0, "qwen3:32b", "qwen3-vl:8b", 27.5), (128.0, "qwen3:32b", "qwen3-vl:8b", 27.5)])
def test_the_setup_plan_fits_the_macs_memory(home, monkeypatch, ram, text, vision, total):
    monkeypatch.setattr(SD, "ram_gb", lambda: ram)
    monkeypatch.setattr(SD.Setup, "_pulled", lambda self: set())
    plan = {s["id"]: s for s in SD.Setup().plan()}
    assert plan["local"]["model"] == text and plan["vision"]["model"] == vision
    assert round(sum(s["gb"] for s in plan.values() if s["state"] != "skipped"), 1) == total     # (the Read me says "about 5 to 10 GB")
    SD.Settings().update({"ai": {"use": "chatgpt", "fallback": False}})
    plan = {s["id"]: s for s in SD.Setup().plan()}
    assert plan["local"]["state"] == "skipped" and plan["speech"]["state"] == "waiting"
    SD.Settings().update({"ai": {"use": "local", "local_model": "qwen3:4b"}})
    plan = {s["id"]: s for s in SD.Setup().plan()}
    assert plan["local"]["state"] == "waiting" and plan["local"]["model"] == "qwen3:4b"


def test_a_failed_setup_step_says_why_and_the_others_still_run(home, monkeypatch):
    monkeypatch.setattr(SD.Setup, "_pulled", lambda self: set())
    s = SD.Setup()
    ran = []

    def boom(st):
        raise RuntimeError("no internet")
    monkeypatch.setattr(s, "_do_speech", boom)
    for sid in ("ollama", "vision", "local"):
        monkeypatch.setattr(s, f"_do_{sid}", lambda st, sid=sid: ran.append(sid))
    assert s.start() is True
    s.thread.join(5)
    assert ran == ["ollama", "vision", "local"] and not s.running()
    plan = {p["id"]: p for p in s.plan()}
    assert plan["speech"]["state"] == "failed" and plan["speech"]["error"] == "no internet"


def test_setup_without_the_bundled_installer_says_so(home, monkeypatch):
    monkeypatch.delenv("VIDEOBUILDER_UV", raising=False)
    monkeypatch.setattr(SD.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="reinstall"):
        SD.Setup()._uv()
    with pytest.raises(RuntimeError, match="isn't installed"):
        SD.OllamaRunner(home).ensure()


def test_download_progress_counts_only_the_model_being_downloaded(home, tmp_path, monkeypatch):
    """BUG: _hf_download measures the whole models/huggingface folder, so the footage-search download starts at 99% when Parakeet
    (about 1.2 GB, in the same folder) is already there; st_gb("search") is 0.75 while the plan says 1.9 GB."""
    SD.Settings().update({"places": {"models": str(tmp_path / "Models")}})
    blob = tmp_path / "Models" / "huggingface" / "hub" / "models--mlx-community--parakeet-tdt-0.6b-v3" / "blobs" / "weights"
    blob.parent.mkdir(parents=True)
    with open(blob, "wb") as fh:
        fh.truncate(1_300_000_000)                                              # (sparse: takes no disk space)
    polls = iter([None, 0])

    class FakeProc:
        returncode = 0
        stderr = io.BytesIO(b"")

        def __init__(self, *a, **k):
            pass

        def poll(self):
            return next(polls)
    monkeypatch.setattr(SD.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(SD.time, "sleep", lambda s: None)
    s = SD.Setup()
    s._hf_download("google/siglip2-base-patch16-224", "search", 0.4, 1.0)
    assert s.steps["search"]["pct"] < 0.5                                       # nothing of the search model has arrived yet


# ---------------------------------------------------------------------------------------------- how it edits
def test_style_points_from_an_ai_that_answers_badly(home):
    st = SD.Styles()
    st.add("short", "fast cuts", FakeAI({"points": ["Cut fast.", 3, "", "  •  Bold captions  ", "cut fast."]}))
    assert st.get("short")["points"] == ["Cut fast.", "3", "Bold captions"]
    keep = st.get("short")["points"]
    assert st.add("short", "x", FakeAI({}))["points"] == keep                    # no "points" at all: nothing lost
    assert st.add("short", "x", FakeAI({"points": None}))["points"] == keep
    many = st.add("short", "x", FakeAI({"points": [f"Point {i}" for i in range(40)]}))
    assert len(many["points"]) == SD.MAX_POINTS


def test_style_points_given_as_one_string_are_not_split_into_letters(home):
    """BUG: tidy_points iterates whatever "points" is; a model that answers {"points": "Cut every pause."} leaves the list as single
    letters ("C", "u", "t", ...)."""
    d = SD.Styles().add("short", "cut pauses", FakeAI({"points": "Cut every pause."}))
    assert all(len(p) > 1 for p in d["points"])


def test_a_null_point_is_dropped_not_kept_as_the_word_none(home):
    """BUG: tidy_points does str(p) first, so a null in the AI's list becomes the point "None"."""
    d = SD.Styles().add("short", "fast cuts", FakeAI({"points": ["Cut fast.", None]}))
    assert d["points"] == ["Cut fast."]


def test_undo_walks_back_and_the_history_is_capped(home):
    st = SD.Styles()
    st._save("long", [f"Point {i}" for i in range(15)], [], [])
    for _ in range(14):
        st.remove("long", 0)
    assert st.get("long")["points"] == ["Point 14"]
    for _ in range(3):
        st.undo("long")
    assert st.get("long")["points"][0] == "Point 11"
    for _ in range(30):
        st.undo("long")
    assert len(st.get("long")["history"]) == 0 and st.undo("long")["points"] == st.get("long")["points"]
    with pytest.raises(ValueError):
        st.remove("long", 99)
    with pytest.raises(ValueError):
        st.remove("long", -1)
    assert st.brief("short") == ""


def test_each_kind_only_sees_its_own_points_and_long_points_reach_the_rules_file(api, tmp_path, monkeypatch):
    from clipfarm.clipper import localai as LA
    monkeypatch.setattr(LA, "LocalAI", lambda cfg: FakeAI({"points": ["Open with the strongest line."]}))
    api.post(["styles", "long", "add"], {"text": "open with the best line"})
    assert (tmp_path / "rules.md").read_text() == "- Open with the strongest line.\n"
    assert api.styles.get("short")["points"] == [] and api.styles.brief("short") == ""
    api.post(["styles", "long", "remove"], {"index": 0})
    assert (tmp_path / "rules.md").read_text() == ""
    assert api.post(["styles", "long", "undo"], {})[1]["points"] == ["Open with the strongest line."]
    assert api.post(["styles", "reels", "add"], {"text": "x"})[0] == 404           # an unknown kind (the server says 404, not 400)
    with pytest.raises(KeyError):
        api.post(["styles", "short", "frobnicate"], {})
    with pytest.raises(ValueError):
        api.post(["styles", "short", "remove"], {"index": "first"})


# ---------------------------------------------------------------------------------------------- ChatGPT skills
def _zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, data in entries.items():
            z.writestr(n, data)
    return buf.getvalue()


def test_skill_zips_that_are_not_what_they_claim(home, api, tmp_path, monkeypatch):
    from clipfarm.clipper import localai as LA
    ai = FakeAI({"points": ["Bold captions."]})
    monkeypatch.setattr(LA, "LocalAI", lambda cfg: ai)
    only_code = _zip({"skill/run.py": "print(1)", "skill/logo.png": b"\x89PNG"})
    with pytest.raises(ValueError, match="no text"):
        api.post(["styles", "short", "import"], {"name": "skill.zip", "data": base64.b64encode(only_code).decode()})
    sneaky = _zip({"../../../../outside.md": "Bold captions.", "/abs/SKILL.md": "Short clips.", "a/../../b.txt": "Music low."})
    before = set(tmp_path.rglob("*"))
    api.post(["styles", "short", "import"], {"name": "x.zip", "data": base64.b64encode(sneaky).decode()})
    assert set(tmp_path.rglob("*")) - before <= {p for p in tmp_path.rglob("*") if "VB home" in str(p)}     # read, never extracted
    assert "Short clips." in ai.calls[-1][1] and "Music low." in ai.calls[-1][1]
    t = SD.skill_text("SKILL.md", b"\xff\xfe broken \xc3 utf8 Cut pauses.")
    assert "Cut pauses." in t
    big = SD.skill_text("big.zip", _zip({"SKILL.md": "Cut every pause. " * 3_000_000}))       # ~50 MB of text inside a small zip
    assert len(big) <= 20000
    with pytest.raises(ValueError):
        api.post(["styles", "short", "import"], {"name": "x.zip", "data": "not base64!!"})


def test_a_file_that_is_not_a_zip_is_a_plain_400_not_a_500(api, monkeypatch):
    """BUG: a .zip that isn't one raises zipfile.BadZipFile, which the server answers with 500 "File is not a zip file"."""
    from clipfarm.clipper import localai as LA
    monkeypatch.setattr(LA, "LocalAI", lambda cfg: FakeAI({"points": ["x"]}))
    with pytest.raises(ValueError):
        api.post(["styles", "short", "import"], {"name": "skill.zip", "data": base64.b64encode(b"# just markdown").decode()})


# ---------------------------------------------------------------------------------------------- folders and links
@pytest.mark.parametrize("title,expect", [("a/b:c", "a b c"), ("..", "Untitled"), ("", "Untitled"), ("🎬", "🎬"), ("x" * 300, "x" * 70),
                                          ('Café "quoted" 12:30', 'Café quoted 12 30')])
def test_project_folders_for_odd_titles_stay_inside_the_videos_folder(tmp_path, title, expect):
    videos = tmp_path / "Videos ✓"
    a, b, c = (SD.project_folder(videos, title) for _ in range(3))
    assert a.parent == b.parent == c.parent == videos and len({a, b, c}) == 3
    assert a.name == expect and b.name.endswith("(2)") and c.name.endswith("(3)")


def test_raw_footage_links_for_odd_names_and_a_folder(tmp_path):
    proj = SD.project_folder(tmp_path / "Videos", "Odd")
    srcs = []
    for n in ODD:
        p = tmp_path / "Footage" / n
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(b"x")
        srcs.append(p)
    card = tmp_path / "Footage" / "CARD 1"
    card.mkdir()
    assert SD.link_raw(proj, srcs + [card]) == len(ODD) + 1
    raw = proj / "Raw footage"
    assert sorted(p.name for p in raw.iterdir()) == sorted(ODD + ["CARD 1"]) and all((raw / n).resolve().exists() for n in ODD)
    assert SD.link_raw(proj, srcs + [card]) == 0


# ---------------------------------------------------------------------------------------------- short clips
def test_odd_video_paths_become_jobs_that_read_the_original(api, tmp_path):
    for n in ODD:
        src = tmp_path / "Footage" / n
        src.parent.mkdir(exist_ok=True)
        src.write_bytes(b"x")
        j = api.store.get(api.make_short({"source": str(src)})["id"])
        link = api.store.dir(j["id"]) / ("source" + src.suffix.lower())
        assert link.is_symlink() and link.resolve() == src.resolve() and j["status"] == "queued"
        assert Path(j["app_home"]).parent == tmp_path / "Videos ✓" and j["source"]["path"] == str(src)
    tilde = api.make_short({"source": "~/" + os.path.relpath(tmp_path / "Footage" / ODD[0], Path.home())})
    assert (api.store.dir(tilde["id"]) / "source.mp4").resolve() == (tmp_path / "Footage" / ODD[0]).resolve()


@pytest.mark.parametrize("src", ["", "   ", "ftp://example.com/v.mp4", "file:///etc/hosts", "javascript:alert(1)", "https://x.com/a b",
                                 "/no/such/video.mp4", "{dir}/a folder.mp4", "{dir}/notes.txt"])
def test_things_that_are_not_a_video_or_a_link_are_refused_and_leave_nothing(api, tmp_path, src):
    (tmp_path / "a folder.mp4").mkdir()
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError):
        api.make_short({"source": src.format(dir=tmp_path)})
    assert api.store.list() == [] and not (tmp_path / "Videos ✓").exists()


def test_a_relative_video_path_still_links_to_the_original(api, tmp_path, monkeypatch):
    """BUG: make_short links the path as given; a relative path makes a broken link inside the job folder (and in Raw footage)."""
    (tmp_path / "talk.mp4").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    j = api.make_short({"source": "talk.mp4"})
    assert (api.store.dir(j["id"]) / "source.mp4").resolve().exists()


def test_links_and_same_titles_get_folders_of_their_own(api):
    a = api.store.get(api.make_short({"source": "https://www.youtube.com/watch?v=abc"})["id"])
    b = api.store.get(api.make_short({"source": "http://example.com/podcast.mp3?x=1"})["id"])
    assert a["source"]["url"].startswith("https://") and not list(api.store.dir(a["id"]).glob("source*"))
    assert Path(a["app_home"]).name == "Clips from a link" and Path(b["app_home"]).name == "Clips from a link (2)"


def test_clip_options_come_from_defaults_then_the_video_then_the_style(api):
    api.settings.update({"defaults": {"short": {"min_len": 25, "max_len": 40, "caption_style": "pop", "clips": 5}}})
    api.styles._save("short", ["Start on the punchline."], [], [])
    src = api.root.parent / "v.mp4"
    src.write_bytes(b"x")
    o = api.store.get(api.make_short({"source": str(src), "notes": "the funny bits", "options": {"layout": "track"}})["id"])["options"]
    assert (o["min_len"], o["max_len"], o["caption_style"], o["layout"], o["clips"]) == (25, 40, "pop", "track", 5)
    assert o["prompt"].index("the funny bits") < o["prompt"].index("Start on the punchline.")
    o = api.store.get(api.make_short({"source": str(src), "options": {"len": "45-90", "caption_style": "evil", "layout": "nope"}})["id"])["options"]
    assert (o["min_len"], o["max_len"]) == (45, 90) and o["caption_style"] != "evil" and o["layout"] in ("auto", "track", "wide", "fit", "center")
    with pytest.raises((ValueError, TypeError)):
        api.make_short({"source": str(src), "options": ["not", "a", "dict"]})


def test_keep_on_a_clip_that_does_not_exist(api, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    jid = api.make_short({"source": str(src)})["id"]
    _done(api, jid, ["One"])
    with pytest.raises(KeyError):
        api.post(["short", jid, "keep"], {"index": 7, "keep": False})
    with pytest.raises(ValueError):
        api.post(["short", jid, "keep"], {"index": "first"})
    with pytest.raises(TypeError):
        api.post(["short", jid, "keep"], {"index": None})
    with pytest.raises(KeyError):
        api.post(["short", "20990101-000000-abcdef", "keep"], {"index": 0})
    with pytest.raises(KeyError):
        api.post(["short", "../../etc", "keep"], {"index": 0})


@pytest.mark.parametrize("status", ["queued", "downloading", "transcribing", "finding", "rendering", "joining"])
def test_make_again_is_refused_while_the_clips_are_being_made(api, tmp_path, status):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    jid = api.make_short({"source": str(src)})["id"]
    api.store.update(jid, status=status)
    with pytest.raises(ValueError, match="still being made"):
        api.post(["short", jid, "remake"], {})
    assert api.store.get(jid)["status"] == status


def test_make_again_only_for_the_apps_own_jobs(api):
    job = api.store.create({"url": "https://x"}, {})
    api.store.update(job["id"], status="done")
    with pytest.raises(KeyError):
        api.remake_short(job["id"], {})


def test_make_again_twice_in_the_same_minute_keeps_both_earlier_versions(api, tmp_path, monkeypatch):
    """BUG: the earlier clips move to "Clips (before HH.MM)"; a second Make again in the same minute renames onto that folder and fails
    with "[Errno 66] Directory not empty" (seen for real through POST /api/app/short/<job>/remake: 500)."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    jid = api.make_short({"source": str(src)})["id"]
    _done(api, jid, ["First"])
    monkeypatch.setattr(W.time, "strftime", lambda fmt, *a: "10.00")
    api.remake_short(jid, {})
    _done(api, jid, ["Second"])
    api.remake_short(jid, {})
    home = Path(api.store.get(jid)["app_home"])
    assert len([p for p in home.iterdir() if p.name.startswith("Clips (before")]) == 2


def test_saved_clips_for_odd_titles_untick_and_tick_again(api, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    jid = api.make_short({"source": str(src)})["id"]
    _done(api, jid, ["Funny: bit / one?", "Funny: bit / one?", "🎬🎬", "<b>x</b> & 'y'", ".."])
    clips = Path(api.store.get(jid)["app_home"]) / "Clips"
    names = sorted(p.name for p in clips.glob("*.mp4"))
    assert len(names) == 5 and all(n[:3] in ("01 ", "02 ", "03 ", "04 ", "05 ") for n in names) and not any("/" in n for n in names)
    assert "03 clip.mp4" in names                                                # nothing left of the title: "clip"
    api.keep_clip(jid, {"index": 0, "keep": False})
    assert sorted(p.name for p in (clips / "Not kept").iterdir()) == ["01 Funny bit one.mp4", "01 Funny bit one.srt"]
    api.keep_clip(jid, {"index": 0, "keep": True})
    assert not list((clips / "Not kept").iterdir()) and (clips / "01 Funny bit one.mp4").exists()
    api.save_clips(api.store.get(jid))                                          # nothing changed: nothing copied again
    assert len(list(clips.glob("*.mp4"))) == 5


def _real_clip(path: Path, size: str, audio: bool) -> None:
    ff = ffmpeg.bin_path("ffmpeg")
    cmd = [ff, "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=15:duration=1"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-c:a", "aac", "-shortest"]
    subprocess.run([*cmd, "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)], check=True)


def test_real_vertical_and_landscape_clips_arrive_whole_in_the_project_folder(api, tmp_path):
    src = tmp_path / "Interview 01.mov"
    _real_clip(src, "1920x1080", True)
    jid = api.make_short({"source": str(src)})["id"]
    d = api.store.dir(jid)
    _real_clip(d / "clip_00.mp4", "1080x1920", True)
    _real_clip(d / "clip_01.mp4", "1920x1080", False)
    for i, t in enumerate(["Vertical one", "Landscape, no audio"]):
        api.store.set_clip(jid, {"index": i, "file": f"clip_{i:02d}.mp4", "title": t, "rendered_at": 1})
    api.store.update(jid, status="done")
    api.save_clips(api.store.get(jid))
    out = Path(api.store.get(jid)["app_home"]) / "Clips"
    probe = ffmpeg.bin_path("ffprobe")

    def info(p):
        r = subprocess.run([probe, "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json", str(p)],
                           capture_output=True, text=True, check=True)
        return json.loads(r.stdout)
    v, l = info(out / "01 Vertical one.mp4"), info(out / "02 Landscape, no audio.mp4")
    assert (v["streams"][0]["width"], v["streams"][0]["height"]) == (1080, 1920) and any(s["codec_type"] == "audio" for s in v["streams"])
    assert (l["streams"][0]["width"], l["streams"][0]["height"]) == (1920, 1080) and not any(s["codec_type"] == "audio" for s in l["streams"])
    assert abs(float(v["format"]["duration"]) - 1.0) < 0.2
    assert (Path(api.store.get(jid)["app_home"]) / "Raw footage" / "Interview 01.mov").resolve() == src.resolve()


# ---------------------------------------------------------------------------------------------- long videos
class _Projects:
    """create_local makes the save folder like the real one; nothing else runs."""

    def __init__(self, tmp_path):
        self.tmp, self.made = tmp_path, []

    def rules_path(self):
        return self.tmp / "rules.md"

    def list(self):
        return []

    def create_local(self, folder, title, notes, opts, defaults, out_dir=None, **kw):
        home = SD.project_folder(out_dir, Path(folder).name)
        self.made.append(home)
        return {"id": f"ai-{len(self.made)}", "home": str(home)}

    def prepare_local(self, pid, body):
        self.prepared = (pid, body)


def test_a_long_video_from_an_odd_folder_with_the_defaults_and_this_videos_choices(home, tmp_path):
    started = []
    projects = _Projects(tmp_path)
    a = _api(home, tmp_path, projects=projects, runner=SimpleNamespace(start=lambda pid, kind, body: started.append((pid, kind, body)),
                                                                        running=None, library=lambda: None))
    folder = tmp_path / 'Footage "A" 10:30 ✓'
    folder.mkdir()
    a.settings.update({"defaults": {"long": {"caption_style": "clean", "long_minutes": 3}}})
    assert a.post(["long"], {"source": str(folder), "notes": "Open with her best line.", "options": {"music": "chill"}}) == (201, {"id": "ai-1"})
    body = started[0][2]
    assert started[0][1] == "make" and body["caption_style"] == "clean" and body["long_minutes"] == 3.0 and body["music"] == "chill"
    assert body["fcp"] is True and body["library"] is True and body["notes"] == "Open with her best line."
    for bad in ("", str(tmp_path / "missing"), str(tmp_path / "file.mp4"), "https://youtube.com/x"):
        with pytest.raises(ValueError):
            a.make_long({"source": bad})


def test_a_long_video_with_a_bad_option_leaves_no_project_behind(home, tmp_path):
    """BUG: make_long creates the project (and its folder in the videos folder) before reading the options; long_minutes "abc" answers
    400 but leaves an empty "<title> (2)" folder and a draft project nobody sees (seen for real)."""
    projects = _Projects(tmp_path)
    a = _api(home, tmp_path, projects=projects, runner=SimpleNamespace(start=lambda *x, **k: None, running=None, library=lambda: None))
    folder = tmp_path / "Footage"
    folder.mkdir()
    with pytest.raises(ValueError):
        a.make_long({"source": str(folder), "options": {"long_minutes": "abc"}})
    assert projects.made == [] and not (tmp_path / "Videos ✓").exists()


# ---------------------------------------------------------------------------------------------- places
def test_places_refuse_relative_and_unknown_and_change_the_videos_folder(api, tmp_path):
    for bad in ({"key": "videos", "path": "relative/dir"}, {"key": "evil", "path": str(tmp_path)}, {"key": "videos", "path": ""}, {}):
        with pytest.raises(ValueError):
            api.post(["places"], bad)
    new = tmp_path / 'New "Videos" ✓'
    code, out = api.post(["places"], {"key": "videos", "path": str(new)})
    assert code == 200 and out["settings"]["places"]["videos"] == str(new) and new.is_dir() and api.restart_wanted


def test_moving_the_models_moves_them_and_restarts(api, tmp_path, monkeypatch):
    old = tmp_path / "Models"
    (old / "huggingface" / "hub").mkdir(parents=True)
    (old / "huggingface" / "hub" / "w.bin").write_bytes(b"w" * 1000)
    api.settings.update({"places": {"models": str(old)}})
    new = tmp_path / "External drive ✓" / "Models"
    api.post(["places"], {"key": "models", "path": str(new)})
    for t in threading.enumerate():
        if t.name == "move-place":
            t.join(10)
    assert (new / "huggingface" / "hub" / "w.bin").read_bytes() == b"w" * 1000 and not (old / "huggingface").exists()
    assert api.settings.get()["places"]["models"] == str(new) and api.restart_wanted
    api.worker.busy_with = "job"
    (new / "x").write_text("x")
    with pytest.raises(ValueError, match="wait until"):
        api.post(["places"], {"key": "models", "path": str(tmp_path / "Elsewhere")})


def test_moving_the_models_into_their_own_subfolder_is_refused(api, tmp_path):
    """BUG: moving the models folder into a folder inside it starts a move that stops with "Cannot move a directory ... into itself"
    and leaves an empty subfolder (seen for real); it should be refused up front."""
    old = tmp_path / "Models"
    (old / "ollama").mkdir(parents=True)
    (old / "ollama" / "m").write_text("m")
    api.settings.update({"places": {"models": str(old)}})
    with pytest.raises(ValueError):
        api.post(["places"], {"key": "models", "path": str(old / "inner")})


def test_a_folder_that_cannot_be_made_is_a_plain_400(api, tmp_path):
    """BUG: a folder on a drive that isn't plugged in (/Volumes/<missing>/...) answers 500 "[Errno 13] Permission denied: '/Volumes/...'"
    (seen for real); it should be a ValueError the page can show ("that drive isn't plugged in")."""
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        with pytest.raises(ValueError):
            api.post(["places"], {"key": "videos", "path": str(locked / "Videos")})
    finally:
        locked.chmod(0o700)


# ---------------------------------------------------------------------------------------------- sign-ins and the picker
def test_claude_sign_in_status_uses_the_apps_own_login(home, tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "fakeclaude"
    fake.parent.mkdir()
    fake.write_text('#!/bin/sh\nif [ "$CLAUDE_CONFIG_DIR" = "' + str(home / "claude") + '" ]; then '
                    'echo \'{"loggedIn": true, "email": "a@b.c", "subscriptionType": "max"}\'; else echo \'{"loggedIn": false}\'; fi\n')
    fake.chmod(0o755)
    monkeypatch.setattr(SD, "claude_bin", lambda: str(fake))
    assert SD.ClaudeLogin(home).status() == {"status": "connected", "email": "a@b.c", "plan": "max"}
    fake.write_text("#!/bin/sh\necho 'not json'\n")
    assert SD.ClaudeLogin(home).status()["status"] == "error"
    monkeypatch.setattr(SD, "claude_bin", lambda: None)
    assert SD.ClaudeLogin(home).status()["status"] == "not installed"


def test_claude_that_will_not_install_says_why(home, monkeypatch):
    monkeypatch.setattr(SD, "claude_bin", lambda: None)
    monkeypatch.setattr(SD.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="curl: (6) Could not resolve host"))
    c = SD.ClaudeLogin(home)
    c.start()
    c.thread.join(5)
    assert c.state["status"] == "idle" and "Could not resolve host" in c.state["error"]


def test_connecting_chatgpt_switches_the_ai_to_it(api, monkeypatch):
    api.settings.update({"ai": {"use": "local"}})
    monkeypatch.setattr(SD, "write_overlay", lambda root=None: None)
    connected = threading.Event()
    api.chatgpt = SimpleNamespace(connect=lambda: connected.set(), disconnect=lambda: None)
    assert api.post(["ai", "chatgpt", "connect"], {}) == (200, {"ok": True})
    connected.wait(5)
    for t in threading.enumerate():
        if t.name == "chatgpt-connect":
            t.join(5)
    assert api.settings.get()["ai"]["use"] == "chatgpt" and api.restart_wanted
    assert api.post(["ai", "openai", "connect"], {})[0] == 404 and api.post(["ai", "chatgpt", "hack"], {})[0] == 404


def test_the_folder_picker_survives_a_quote_in_the_start_folder(tmp_path, monkeypatch):
    """BUG: _osascript_pick puts the start folder into the AppleScript as it is; a folder named with a double quote breaks the script
    (the picker silently returns nothing) and lets the folder name add AppleScript of its own. server.py's own picker strips quotes."""
    start = tmp_path / 'Client "A" footage'
    start.mkdir()
    seen = []
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **k: seen.append(cmd) or SimpleNamespace(returncode=0, stdout="/Volumes/X/\n"))
    assert W._osascript_pick("folder", 'Choose "it"', str(start)) == "/Volumes/X"
    script = seen[0][2]
    assert 'Client "A"' not in script


# ---------------------------------------------------------------------------------------------- the real server
def _studio_server(tmp_path, home, monkeypatch):
    from http.server import ThreadingHTTPServer

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import server as S
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.feed import Feed
    from clipfarm.clipper.localproj import LocalRunner
    from clipfarm.clipper.store import Store
    monkeypatch.setattr(W.StudioAPI, "_watch", lambda self: None)              # (it would end the test run with code 75)
    monkeypatch.setattr(SD.ClaudeLogin, "status", lambda self: {"status": "disconnected"})
    monkeypatch.setattr(SD.Setup, "_pulled", lambda self: set())
    SD.Settings(home).update({"setup": {"done": True}, "places": {"videos": str(tmp_path / "Videos"), "work": str(tmp_path / "work")}})
    cfg = _app_cfg(home, tmp_path, monkeypatch)
    store = Store(Path(cfg.get("clipper.data_dir")))
    eng = Engine(cfg, store)
    projects = AV.Projects(cfg, store, eng)
    runner = LocalRunner(cfg, store, projects)
    worker = S.Worker(eng, store, Feed(cfg, store), projects)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(eng, store, worker, worker.feed, 100, projects, runner))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _call(base, path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_the_app_page_and_its_api_over_http(tmp_path, home, monkeypatch):
    from clipfarm.clipper.server import WEB
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        assert _call(base, "/") == (200, (WEB / "studio.html").read_bytes())
        st, body = _call(base, "/api/app")
        d = json.loads(body)
        assert st == 200 and d["ai"]["use"] == "chatgpt" and d["library"] is None and d["shorts"] == [] and d["restarting"] is False
        assert [s["id"] for s in d["setup"]["steps"]] == ["speech", "ollama", "vision", "local", "search", "music", "voices"]
        assert _call(base, "/api/app/ping") == (200, b'{"ok": true}')
        assert _call(base, "/api/app/styles/reels/add", {"text": "x"})[0] == 404
        assert _call(base, "/api/app/short", {"source": str(tmp_path)})[0] == 400
        assert _call(base, "/api/app/places", {"key": "models", "path": "relative"})[0] == 400
        assert _call(base, "/api/library/status")[0] == 404                       # no library: no Find footage
        src = tmp_path / "talk.mp4"
        src.write_bytes(b"x")
        st, body = _call(base, "/api/app/short", {"source": str(src)})
        assert st == 201 and json.loads(_call(base, "/api/app")[1])["shorts"][0]["status"] == "queued"
    finally:
        httpd.shutdown()


def test_settings_with_the_wrong_types_answer_400_over_http(tmp_path, home, monkeypatch):
    """BUG: POST /api/app/settings {"ai": "local"} answers 500 "'str' object has no attribute 'get'" (seen for real)."""
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        assert _call(base, "/api/app/settings", {"ai": "local"})[0] == 400
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------------------------- what ships
def test_make_config_stops_on_personal_details(tmp_path):
    for leak in ("host: user@server", "url: https://box.example.ts.net/x", "dir: /Users/someone/clipfarm", "who: '@handle'"):
        src = tmp_path / "config.yaml"
        src.write_text(f"clipper:\n  {leak}\n")
        r = subprocess.run([sys.executable, str(REPO / "app" / "VideoBuilder" / "make_config.py"), str(src), str(tmp_path / "out.yaml")],
                           capture_output=True, text=True)
        assert r.returncode != 0 and "personal details" in r.stderr, leak


@pytest.mark.skipif(not (REPO / "dist" / "FirstCut.app").is_dir(), reason="no build in dist/ (scripts/package_videobuilder.sh)")
def test_the_built_app_has_nothing_personal_and_no_tests():
    res = REPO / "dist" / "FirstCut.app" / "Contents" / "Resources"
    bad = ("example", "you", "workuser", "1000001", "1000002", "/Users/you", "100.66.", "100.79.")
    for p in (res / "app").rglob("*"):
        if p.is_file() and p.suffix in (".py", ".yaml", ".html", ".sh", ".txt", ".js", ".json", ".md"):
            text = p.read_text(errors="replace")
            assert not [b for b in bad if b in text], p
    assert not list((res / "app").rglob("test_*.py")) and (res / "bin" / "ffmpeg").is_file() and (res / "bin" / "uv").is_file()
    assert yaml.safe_load((res / "app" / "config" / "app.yaml").read_text())["clipper"]["app"] == "studio"


# ---------------------------------------------------------------------------------------------- footage sources (Find footage, 2026-11-05)
# Added by the tester. xfail(strict=True) = a real defect found today: the test XPASSes (and fails the run) once it is fixed; then drop the mark.
def _lib_api(home, tmp_path, cat=None):
    """A StudioAPI whose runner holds `cat` (a real Catalog, or None: apply_library makes one) and records first() calls."""
    runner = SimpleNamespace(catalog=cat, running=None)
    runner.library = lambda: runner.catalog if runner.catalog is not None and runner.catalog.enabled() else None
    a = _api(home, tmp_path, runner=runner)
    return a, runner


def _fake_first(cat):
    calls = []
    cat.first = lambda folders, hours=None: calls.append([str(f) for f in folders])
    return calls


def test_adding_a_source_refuses_what_is_not_a_folder_and_names_an_unplugged_drive(home, tmp_path):
    from clipfarm.clipper import catalog as C
    good = tmp_path / "Footage"
    good.mkdir()
    (good / "clip.mp4").write_bytes(b"x" * 2048)
    cat = C.Catalog(SimpleNamespace(root=tmp_path, get=lambda k, d=None: {"catalog": {"roots": [str(good)], "dir": str(tmp_path / "cat")}}.get(k, d)))
    a, runner = _lib_api(home, tmp_path, cat)
    calls = _fake_first(cat)
    for bad in ("relative/folder", str(good / "clip.mp4"), str(tmp_path / "missing"), "", None, ["/tmp"], 7, {"p": 1}):
        with pytest.raises(ValueError, match="choose a folder"):
            a.library_action("add", {"path": bad})
    with pytest.raises(ValueError, match="NoSuchDrive-vb"):
        a.library_action("add", {"path": "/Volumes/NoSuchDrive-vb/B-roll"})
    assert a.settings.get()["library"]["folders"] == [] and calls == [] and not a.restart_wanted
    s = a.library_action("add", {"path": str(good) + "/./", "now": False})["settings"]["library"]
    assert s["folders"] == [str(good)] and s["look"] == str(good) and calls == []                    # normalised; now=False: not started
    a.library_action("add", {"path": str(good)})
    assert a.settings.get()["library"]["folders"] == [str(good)] and calls == [[str(good)]]          # added twice: listed once


def test_use_and_catalog_only_for_known_sources(home, tmp_path):
    from clipfarm.clipper import catalog as C
    good = tmp_path / "Footage"
    good.mkdir()
    cat = C.Catalog(SimpleNamespace(root=tmp_path, get=lambda k, d=None: {"catalog": {"roots": [], "dir": str(tmp_path / "cat")}}.get(k, d)))
    a, runner = _lib_api(home, tmp_path, cat)
    calls = _fake_first(cat)
    a.settings.update({"library": {"folders": [str(good), "/Volumes/NoSuchDrive-vb/B-roll"]}})
    a.apply_library(a.settings.get())
    assert cat.roots == [good, Path("/Volumes/NoSuchDrive-vb/B-roll")]
    assert a.library_action("use", {"path": str(good) + "/"})["settings"]["library"]["look"] == str(good)
    for what, path in (("use", "/nowhere"), ("use", "relative"), ("catalog", "/nowhere"), ("catalog", ""), ("bogus", str(good))):
        with pytest.raises(KeyError):
            a.library_action(what, {"path": path})
    with pytest.raises(ValueError, match="NoSuchDrive-vb"):                                       # a source whose drive is unplugged
        a.library_action("catalog", {"path": "/Volumes/NoSuchDrive-vb/B-roll"})
    assert a.library_action("catalog", {"path": str(good)}) == {"ok": True} and calls == [[str(good)]]
    runner.catalog = None                                                                       # no catalog at all: 404, not a crash
    with pytest.raises(KeyError):
        a.library_action("catalog", {"path": str(good)})


def test_removing_a_source_in_settings_changes_the_live_catalog_and_the_overlay_without_a_restart(home, tmp_path):
    from clipfarm.clipper import catalog as C
    x, y = tmp_path / "X", tmp_path / "Y"
    x.mkdir()
    y.mkdir()
    cat = C.Catalog(SimpleNamespace(root=tmp_path, get=lambda k, d=None: {"catalog": {"roots": [str(x), str(y)], "dir": str(tmp_path / "cat")}}.get(k, d)))
    a, runner = _lib_api(home, tmp_path, cat)
    a.settings.update({"library": {"folders": [str(x), str(y)]}})
    code, out = a.post(["settings"], {"library": {"folders": [str(x)]}})
    assert code == 200 and cat.roots == [x] and not a.restart_wanted
    assert yaml.safe_load((home / "overlay.yaml").read_text())["catalog"]["roots"] == [str(x)]
    a.post(["settings"], {"library": {"folders": []}})
    assert cat.roots == [] and runner.library() is None and not a.restart_wanted                  # Find footage goes back to "Choose a folder"


def test_the_51st_source_is_refused_not_silently_dropped(home, tmp_path):
    from clipfarm.clipper import catalog as C
    folders = []
    for k in range(51):
        f = tmp_path / "src" / f"F{k:02d}"
        f.mkdir(parents=True)
        folders.append(str(f))
    cat = C.Catalog(SimpleNamespace(root=tmp_path, get=lambda k, d=None: {"catalog": {"roots": folders[:1], "dir": str(tmp_path / "cat")}}.get(k, d)))
    a, runner = _lib_api(home, tmp_path, cat)
    calls = _fake_first(cat)
    a.settings.update({"library": {"folders": folders[:50]}})
    try:
        s = a.library_action("add", {"path": folders[50]})["settings"]["library"]
    except ValueError:
        return                                                                                  # refused with a reason: fine
    assert folders[50] in s["folders"] or (s["look"] != folders[50] and not calls)


def test_recent_with_nan_or_infinity_never_reaches_the_page(tmp_path, home, monkeypatch):
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        req = urllib.request.Request(base + "/api/app/settings", data=b'{"library": {"recent": {"/A": NaN, "/B": Infinity, "/C": 5}}}',
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req, timeout=20).read()
        except urllib.error.HTTPError as e:
            assert e.code == 400                                                                # refusing it is fine too
        st, body = _call(base, "/api/app")

        def strict(c):
            raise ValueError(f"not JSON for a browser: {c}")
        assert st == 200 and json.loads(body, parse_constant=strict)["settings"]["library"]["recent"].get("/C") in (5, 5.0, None)
    finally:
        httpd.shutdown()


def test_recent_stays_capped(home):
    s = SD.Settings(home)
    s.update({"library": {"recent": {f"/A{k}": float(k) for k in range(80)}}})
    out = s.update({"library": {"recent": {f"/B{k}": float(k) for k in range(80)}}})
    assert len(out["library"]["recent"]) <= 100


def test_footage_sources_over_http(tmp_path, home, monkeypatch):
    """add / use / catalog through the real server: a catalog is made on the first add (no restart), catalogued, searched per source."""
    import urllib.parse

    from test_catalog import FakeEmbedder, FakeVision, _video

    from clipfarm.clipper import catalog as C
    monkeypatch.setattr(C, "SigLIP", lambda name=None: FakeEmbedder())                         # offline: no model download, no Ollama
    monkeypatch.setattr(C.Catalog, "vision", lambda self: FakeVision())
    a, b = tmp_path / "X" / "B-roll", tmp_path / "X" / "B-roll 2"
    _video(a / "red.mp4", "red", secs=3)
    _video(b / "blue.mp4", "blue", secs=3)
    httpd, base = _studio_server(tmp_path, home, monkeypatch)

    def wait_idle():
        end = time.time() + 60
        while time.time() < end:
            st = json.loads(_call(base, "/api/library/status")[1])
            if not st["status"].get("running") and not [t for t in threading.enumerate() if t.name == "catalog-first" and t.is_alive()]:
                return st
            time.sleep(0.2)
        raise AssertionError("cataloguing did not finish")
    try:
        assert _call(base, "/api/library/status")[0] == 404                                      # no source yet: no catalog
        assert _call(base, "/api/app/library/add", {"path": "relative"})[0] == 400
        assert _call(base, "/api/app/library/add", {"path": "/Volumes/NoSuchDrive-vb/x"})[0] == 400
        st, body = _call(base, "/api/app/library/add", {"path": str(a)})
        assert st == 200 and json.loads(body)["settings"]["library"]["look"] == str(a)
        wait_idle()
        assert _call(base, "/api/app/library/add", {"path": str(b)})[0] == 200
        st = wait_idle()
        assert {Path(r["path"]).name: r["videos"] for r in st["roots"]} == {"B-roll": 1, "B-roll 2": 1} and st["status"].get("scope") is None
        d = json.loads(_call(base, "/api/app")[1])
        assert d["restarting"] is False and [r["path"] for r in d["library"]["roots"]] == [str(a), str(b)]
        for under, want in ((a, "red.mp4"), (b, "blue.mp4")):
            shots = json.loads(_call(base, "/api/library/search?q=red%20blue&under=" + urllib.parse.quote(str(under)))[1])["shots"]
            assert [Path(s["rel"]).name for s in shots] == [want] and all("path" not in s for s in shots)
        assert _call(base, "/api/app/library/use", {"path": str(b)})[0] == 200
        assert _call(base, "/api/app/library/use", {"path": "/nowhere"})[0] == 404
        assert _call(base, "/api/app/library/catalog", {"path": "/nowhere"})[0] == 404
        assert _call(base, "/api/app/library/bogus", {})[0] == 404
        assert _call(base, "/api/app/library/catalog", {"path": str(b)}) == (200, b'{"ok": true}')
        wait_idle()
        lib = json.loads(_call(base, "/api/app")[1])["settings"]["library"]
        assert lib["look"] == str(b) and lib["recent"][str(b)] >= lib["recent"][str(a)]
    finally:
        httpd.shutdown()


# ---------------------------------------------------------------------------------------------- the Find page's source logic (node)
def _find_logic(cases: list[dict]) -> list[dict]:
    """Runs studio.html's shortName / sources / lookIn / findStatus in node against stand-in page states."""
    import shutil
    if not shutil.which("node"):
        pytest.skip("no node on this Mac")
    from clipfarm.clipper.server import WEB
    html = (WEB / "studio.html").read_text()
    start, end = html.index("const shortName"), html.index("function paintFindStatus")
    js = ("let A, FIND;\n" + html[start:end] + "\nconst cases = " + json.dumps(cases) + ";\n"
          "console.log(JSON.stringify(cases.map(c => { A = c.A; FIND = c.FIND; const s = sources().map(r => r.path); "
          "const u = lookIn(); return {sources: s, under: u, status: findStatus()}; })));\n")
    r = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _page(folders, roots=None, look="", recent=None, status=None, videos=0, under=None):
    lib = None if roots is None else {"videos": videos, "photos": 0, "roots": roots, "status": status or {}}
    return {"A": {"settings": {"library": {"folders": folders, "look": look, "recent": recent or {}}}, "library": lib},
            "FIND": {"under": under}}


def test_the_find_pages_sources_with_none_one_and_many_folders():
    a, b, c, d = "/Volumes/Work/B-roll", "/Volumes/Work/B-roll 2", "/Users/me/Movies/Weddings", "/Volumes/Gone/Clips"
    root = lambda p, v=0, **k: {"path": p, "mounted": k.get("mounted", True), "videos": v, "photos": 0, "cataloguing": k.get("cat", False)}  # noqa: E731
    out = _find_logic([
        _page([]),                                                                                   # 0: nothing yet
        _page([a], [root(a, 12)], look=a, videos=12),                                                # 1: one catalogued source
        _page([a, b, c], [root(a, 3), root(b, 0), root(c, 5)], recent={b: 30, c: 20}, videos=8),    # 2: many, Everything
        _page([a, b], [root(a, 3), root(b, 0, cat=True)], look=b, videos=3,
              status={"running": True, "done": 2, "total": 9, "stage": "Looking at x", "scope": [b]}),  # 3: the chosen one is being catalogued
        _page([a, d], [root(a, 3), root(d, 7, mounted=False)], look=d, videos=10),                  # 4: its drive is unplugged
        _page([a, b], [root(a, 3), root(b, 0)], look="/Volumes/Work/Removed", videos=3),              # 5: the last look was removed
        _page([a, b], None, look=b),                                                                 # 6: settings say two, no catalog yet
    ])
    assert out[0] == {"sources": [], "under": "", "status": {"text": "0 videos catalogued.", "btn": ""}}
    assert out[1]["under"] == a and out[1]["status"] == {"text": "12 videos catalogued here.", "btn": "more"}
    assert out[2]["sources"] == [b, c, a] and out[2]["under"] == "" and out[2]["status"]["text"] == "8 videos catalogued in 3 folders."
    assert out[3]["status"]["btn"] == "stop" and "2 of 9" in out[3]["status"]["text"]
    assert out[4]["status"] == {"text": "That folder's drive isn't plugged in. Plug it in to search it.", "btn": ""}
    assert out[5]["under"] == ""
    assert out[6]["sources"] == [a, b] and out[6]["status"]["btn"] == "run"


def test_one_source_with_no_saved_choice_is_looked_in():
    a = "/Volumes/Work/B-roll"
    out = _find_logic([_page([a], [{"path": a, "mounted": True, "videos": 0, "photos": 0, "cataloguing": False}], look="")])
    assert out[0]["under"] == a and out[0]["status"]["btn"] == "run"


def test_the_app_pages_script_parses():
    import re
    import shutil
    if not shutil.which("node"):
        pytest.skip("no node on this Mac")
    from clipfarm.clipper.server import WEB
    for k, js in enumerate(re.findall(r"<script[^>]*>(.*?)</script>", (WEB / "studio.html").read_text(), re.S)):
        r = subprocess.run(["node", "--check", "-"], input=js, capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, (k, r.stderr)


class _StopLoop(Exception):
    pass


@pytest.mark.parametrize("looping,want", [(False, [False]), (True, [])])
def test_the_watch_loop_ticks_a_catalog_made_after_startup_only(home, tmp_path, monkeypatch, looping, want):
    ticks = []
    cat = SimpleNamespace(tick=lambda busy=False: ticks.append(busy))
    if looping:
        cat.looping = True                                                     # (server.py's own loop ticks the startup catalog)
    a, runner = _lib_api(home, tmp_path, cat)
    runner.running = None
    naps = []

    def nap(s):
        naps.append(s)
        if len(naps) > 1:
            raise _StopLoop
    monkeypatch.setattr(W, "time", SimpleNamespace(sleep=nap, time=time.time, strftime=time.strftime))   # (only this module's clock)
    with pytest.raises(_StopLoop):
        a._watch()
    assert ticks == want
