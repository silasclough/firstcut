"""FirstCut on someone else's Mac (added by the tester, 2026-11-05): nothing the app path uses may come from this Mac's own installs (Homebrew,
the repo's assets/ and data/, the SF Compact fonts from Apple's developer download, the Xcode command line tools), the app bundle stays
untouched after it is signed, and what the app fetches by itself is checked and fails with a reason when there is no internet.

xfail(strict=True) = a real defect found today (with the evidence in its reason): the test XPASSes, and fails the run, once it is fixed;
then drop the mark."""

import io
import os
import subprocess
import sys
import tarfile
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from clipfarm.clipper import studio as SD

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "dist" / "FirstCut.app"


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "FirstCut"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    monkeypatch.setattr(SD, "_answers", lambda url: False)                 # no Ollama running in tests
    monkeypatch.setattr(SD, "ram_gb", lambda: 8.0)
    monkeypatch.setattr(SD.Setup, "_pulled", lambda self: set())
    return h


def _app_config(tmp_path, monkeypatch, home):
    """The settings a new install runs with: config/app.yaml as the build makes it, plus the overlay the app writes from its settings."""
    out = tmp_path / "app.yaml"
    r = subprocess.run([sys.executable, str(REPO / "app" / "VideoBuilder" / "make_config.py"), str(REPO / "config" / "config.yaml"), str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    monkeypatch.setenv("CLIPFARM_OVERLAY", str(SD.write_overlay(home)))
    from clipfarm.core.config import load_config
    return load_config(out, root=home)


# ---------------------------------------------------------------------------------------------- what ships
@pytest.mark.skipif(not APP.is_dir(), reason="no build in dist/ (scripts/package_videobuilder.sh)")
def test_the_built_firstcut_app_has_nothing_personal_no_tests_and_its_own_tools():
    """(tests/test_studio_edges.py::test_the_built_app_has_nothing_personal_and_no_tests looks for dist/"Video Builder.app", which no build
    makes since the rename to FirstCut, so it always skips.)"""
    res = APP / "Contents" / "Resources"
    bad = ("example", "you", "workuser", "1000001", "1000002", "/Users/you", "100.66.", "100.79.")   # (drive names such as
    # "/Volumes/Backup" in aivideo.DEFAULT_DIR and "/Volumes/Work" in ai.html's placeholders still ship; they are not reached in the app)
    for p in (res / "app").rglob("*"):
        if p.is_file() and p.suffix in (".py", ".yaml", ".html", ".sh", ".txt", ".js", ".json", ".md", ".swift"):
            text = p.read_text(errors="replace")
            assert not [b for b in bad if b in text], p
    assert not list((res / "app").rglob("test_*.py"))
    assert not (res / "app" / "data").exists()                                   # a fresh build carries no records and no compiled helpers
    for tool in ("ffmpeg", "ffprobe", "uv"):
        assert os.access(res / "bin" / tool, os.X_OK), tool
    assert yaml.safe_load((res / "app" / "config" / "app.yaml").read_text())["clipper"]["app"] == "studio"


def test_a_new_install_reads_parakeet_from_its_own_home_not_this_macs_clipfarm(tmp_path, monkeypatch, home):
    """config/app.yaml still carries ~/clipfarm/.venv-parakeet (this Mac's layout); the overlay must win over it."""
    cfg = _app_config(tmp_path, monkeypatch, home)
    from clipfarm.clipper import parakeet
    assert cfg.get("clipper.ai_videos.parakeet_bin") == str(home / "runtime" / "parakeet" / "bin" / "parakeet-mlx")
    assert parakeet.binary(cfg) is None                                          # not installed yet: None, never ~/clipfarm's copy
    assert cfg.get("clipper.ai_videos.openai.enabled") is False and cfg.get("llm.provider_chain") == ["ollama"]
    from clipfarm.clipper.auto import _target
    assert _target(cfg) is None                                                  # no posting server to relay through


# ---------------------------------------------------------------------------------------------- the text reader (opening checks of every clip)
@pytest.mark.xfail(strict=True, reason="ocr.BIN = <code>/../data/models/cf_ocr: in the app that is FirstCut.app/Contents/Resources/app/data/models, "
                                       "so the first rendered clip compiles cf_ocr into the signed bundle (seen 2026-11-05: codesign --verify then "
                                       "says 'a sealed resource is missing or invalid'); on a translocated or read-only app it can't be written at all")
def test_the_text_reader_is_compiled_into_the_apps_home_not_into_the_app(tmp_path):
    home = tmp_path / "FirstCut"
    code = r"""
from clipfarm.clipper import ocr
seen = []
ocr.shutil.which = lambda name: "/usr/bin/swiftc" if name == "swiftc" else None
def run(cmd, *a, **k):
    seen.append([str(c) for c in cmd])
    raise RuntimeError("nothing is compiled in a test")
ocr.subprocess.run = run
got = ocr.helper()
outs = [c[c.index("-o") + 1] for c in seen if "-o" in c]
print(outs[0] if outs else str(got))
"""
    env = {**os.environ, "CLIPFARM_ROOT": str(home), "VIDEOBUILDER_HOME": str(home), "PYTHONPATH": str(REPO)}   # as main.swift starts the engine
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr
    where = Path(r.stdout.strip().splitlines()[-1])
    assert not where.is_relative_to(REPO), where                                # (REPO plays the app bundle's Resources/app here)


def test_a_mac_without_the_command_line_tools_is_never_asked_to_install_them(tmp_path, monkeypatch):
    from clipfarm.clipper import ocr
    monkeypatch.setattr(ocr, "BIN", tmp_path / "models" / "cf_ocr")
    monkeypatch.setattr(ocr.shutil, "which", lambda name: f"/usr/bin/{name}" if name in ("swiftc", "swift", "xcrun", "xcode-select") else None)
    ran = []

    def run(cmd, *a, **k):
        cmd = [str(c) for c in (cmd if isinstance(cmd, (list, tuple)) else [cmd])]
        ran.append(cmd)
        out = "" if k.get("text") else b""
        if Path(cmd[0]).name in ("swiftc", "swift"):                             # the stub: it asks to install, and fails
            r = subprocess.CompletedProcess(cmd, 1, out, out)
        else:                                                                     # xcode-select -p / xcrun --find: no developer tools
            r = subprocess.CompletedProcess(cmd, 2, out, out)
        if k.get("check") and r.returncode:
            raise subprocess.CalledProcessError(r.returncode, cmd)
        return r
    monkeypatch.setattr(ocr.subprocess, "run", run)
    no_tools = ("/Library/Developer", "/Applications/Xcode")
    real_exists, real_isdir, real_path_exists = os.path.exists, os.path.isdir, Path.exists
    monkeypatch.setattr(os.path, "exists", lambda p: False if str(p).startswith(no_tools) else real_exists(p))
    monkeypatch.setattr(os.path, "isdir", lambda p: False if str(p).startswith(no_tools) else real_isdir(p))
    monkeypatch.setattr(Path, "exists", lambda self, *a, **k: False if str(self).startswith(no_tools) else real_path_exists(self, *a, **k))
    assert ocr.helper() is None
    assert not [c for c in ran if Path(c[0]).name in ("swiftc", "swift")], ran


# ---------------------------------------------------------------------------------------------- music, animations, faces
@pytest.mark.xfail(strict=True, reason="the page offers Chill / Upbeat / Emotional / Funny music, but the app has no music library: assets/music is not "
                                       "in the bundle and Setup doesn't download it, so every clip says 'no licensed music in assets/music "
                                       "(./cf music openverse) - rendering without music' (seen in the clean-install run)")
def test_music_the_page_offers_is_there_in_a_new_install(tmp_path, monkeypatch, home):
    from clipfarm.clipper import music as MU
    page = (REPO / "clipfarm" / "clipper" / "web" / "studio.html").read_text()
    offered = [m for m in MU.MOODS if f'["{m}", "{m.capitalize()}"]' in page]
    cfg = _app_config(tmp_path, monkeypatch, home)
    for mood in offered:                                                         # (offering none is also a fix)
        assert MU.pick(cfg, mood=mood) is not None, mood


@pytest.mark.xfail(strict=True, reason="animate_hf.make asks the AI to design the diagram (up to 3 rounds) before it finds Hyperframes missing; the app "
                                       "ships no Hyperframes, node or assets/hyperframes, so every 'animation' card costs AI calls and is dropped")
def test_an_animation_card_without_hyperframes_costs_no_ai_call(tmp_path, monkeypatch):
    from clipfarm.clipper import animate_hf, localai, polish
    from clipfarm.core.config import Config
    calls = []

    class FakeAI:
        def __init__(self, cfg, *a, **k):
            pass

        def complete_json(self, task, system, prompt, schema, **kw):
            calls.append(task)
            return SimpleNamespace(data={"script": "tl.to('#a', {opacity: 1}, 0);", "svg": "<svg></svg>", "notes": ""})
    monkeypatch.setattr(localai, "LocalAI", FakeAI)
    monkeypatch.setattr(animate_hf, "cli", lambda cfg: None)                     # a Mac without Hyperframes (every friend's Mac)
    spec = {"concept": "Show water rising four feet inside the bakery and the ovens going under", "narration": "The ovens were under four feet of water.",
            "timed": []}
    out = polish.render_animation(Config({}, root=tmp_path), spec, 8.0, tmp_path / "anim.mov", 1920, 1080, "slate")
    assert out is None
    assert calls == []


@pytest.mark.xfail(strict=True, reason="reframe.yunet_path saves whatever the URL answers (no size or checksum check) and never asks again, so a "
                                       "captive-portal page is kept as the face model: no face framing and graphics may cover faces from then on")
def test_a_face_model_download_that_is_not_the_model_is_not_kept(tmp_path, monkeypatch):
    import requests

    from clipfarm.clipper import reframe
    page = SimpleNamespace(status_code=200, content=b"<html><body>Sign in to the hotel wifi</body></html>", raise_for_status=lambda: None)
    monkeypatch.setattr(requests, "get", lambda *a, **k: page)
    assert reframe.yunet_path(tmp_path) is None
    assert not (tmp_path / "face_detection_yunet_2023mar.onnx").exists()


def test_no_internet_for_the_face_model_means_the_blurred_fit_not_a_crash(tmp_path, monkeypatch):
    import requests

    from clipfarm.clipper import reframe

    def offline(*a, **k):
        raise requests.ConnectionError("nodename nor servname provided")
    monkeypatch.setattr(requests, "get", offline)
    assert reframe.yunet_path(tmp_path / "models") is None
    assert not (tmp_path / "models" / "face_detection_yunet_2023mar.onnx").exists()


# ---------------------------------------------------------------------------------------------- fonts: a friend's Mac has no SF Compact
@pytest.mark.parametrize("W,H", [(1920, 1080), (1080, 1920)])
def test_the_stat_cards_accent_bar_stays_clear_of_the_number_without_sf_compact(monkeypatch, W, H):
    from clipfarm.clipper import gfxcards as G
    monkeypatch.setattr(G, "FONT_FILES", {k: [p for p in v if not p.startswith("/Library/Fonts/SF-")] for k, v in G.FONT_FILES.items()})
    monkeypatch.setattr(G, "_FONTS", {})
    look = G.Look("slate", W, H)
    spec = {"kind": "stat", "value": "92%", "label": "of graduates are hired within six months", "len": 4.0, "style": "wipe", "at": 0.0}
    im = np.asarray(G.frame_at(G.build_scene("stat", look, spec, 4.0), 1.6).convert("RGBA")).astype(int)
    bar = (np.abs(im[..., :3] - np.array(look.deco[:3])).sum(2) < 60) & (im[..., 3] > 200)
    assert bar.sum() >= 0.95 * look.px(120) * look.px(8)                        # the whole bar shows, nothing is drawn over it


def test_every_card_font_has_a_fallback_that_every_mac_ships():
    """/Library/Fonts/SF-Compact-* come from Apple's developer font download (only this Mac has them); each weight must also list a font
    from /System/Library/Fonts, which every macOS has."""
    from clipfarm.clipper import gfxcards as G
    for weight, files in G.FONT_FILES.items():
        assert [p for p in files if p.startswith("/System/Library/Fonts/")], weight


# ---------------------------------------------------------------------------------------------- Setup without internet
class _Resp(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _ollama_tgz() -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in (("ollama", b"#!/bin/sh\necho ollama\n"), ("lib/ollama/libggml-base.dylib", b"\0" * 64), ("GO_LICENSE", b"BSD")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_the_ollama_download_offline_fails_with_a_reason_and_the_next_try_installs_it(home, monkeypatch):
    import urllib.request
    s = SD.Setup()
    for sid in ("speech", "vision", "local"):
        monkeypatch.setattr(s, f"_do_{sid}", lambda st: None)
    monkeypatch.setattr(SD.OllamaRunner, "ensure", lambda self: "http://127.0.0.1:1")

    def offline(*a, **k):
        raise urllib.error.URLError("[Errno 8] nodename nor servname provided, or not known")
    monkeypatch.setattr(urllib.request, "urlopen", offline)
    s._run()
    st = {p["id"]: p for p in s.plan()}["ollama"]
    assert st["state"] == "failed" and "nodename" in st["error"]
    assert not (home / "runtime" / "ollama" / "ollama").exists()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp(_ollama_tgz()))
    s.steps.clear()
    s._run()
    exe = home / "runtime" / "ollama" / "ollama"
    assert {p["id"]: p for p in s.plan()}["ollama"]["state"] == "done"
    assert exe.is_file() and os.access(exe, os.X_OK) and not list((home / "runtime" / "ollama").glob("*.part"))


@pytest.mark.xfail(strict=True, reason="a failed `uv pip install parakeet-mlx` is shown as CalledProcessError's text (\"Command '[.../uv', 'pip', "
                                       "'install', ...\"), cut at 300 characters, without uv's reason; offline that is all Setup shows")
def test_a_failed_speech_install_says_why(home, monkeypatch):
    def run(cmd, *a, **k):
        if "pip" in cmd:
            raise subprocess.CalledProcessError(2, cmd, output=b"", stderr=b"error: Failed to fetch: `https://pypi.org/simple/parakeet-mlx/`\n"
                                                                      b"  Caused by: dns error: failed to lookup address information")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    monkeypatch.setattr(SD.subprocess, "run", run)
    monkeypatch.setenv("VIDEOBUILDER_UV", "/usr/bin/true")
    s = SD.Setup()
    for sid in ("ollama", "vision", "local"):
        monkeypatch.setattr(s, f"_do_{sid}", lambda st: None)
    s._run()
    err = {p["id"]: p for p in s.plan()}["speech"]["error"] or ""
    assert "Failed to fetch" in err or "internet" in err.lower(), err


# ---------------------------------------------------------------------------------------------- clips saved into the project folder
@pytest.fixture
def api(home, tmp_path):
    from clipfarm.clipper import studio_web as W
    from clipfarm.clipper.engine import default_options
    from clipfarm.clipper.server import clean_options
    from clipfarm.clipper.store import Store
    import threading
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.cfg = SimpleNamespace(get=lambda k, d=None: {"clipper": {}}.get(k, d))
    a.store, a.runner, a.chatgpt = Store(tmp_path / "jobs"), None, None
    a.worker = SimpleNamespace(busy_with=None, wake=SimpleNamespace(set=lambda: None))
    a.projects = SimpleNamespace(rules_path=lambda: tmp_path / "rules.md", list=lambda: [])
    a.default_options, a.clean_options = default_options, clean_options
    a.root, a.settings, a.styles, a.setup = home, SD.Settings(home), SD.Styles(home), SD.Setup(home)
    a.mac, a.moving, a.restart_wanted, a._cache, a.lock = SD.mac_info(), "", False, {}, threading.Lock()
    a.settings.update({"places": {"videos": str(tmp_path / "Videos")}})
    return a


def test_the_first_clip_the_engine_makes_is_saved_as_01(api, tmp_path):
    src = tmp_path / "talk.mp4"
    src.write_bytes(b"x")
    jid = api.make_short({"source": str(src)})["id"]
    d = api.store.dir(jid)
    for i in (1, 2):                                                             # as Engine.process numbers them
        (d / f"clip_{i:02d}.mp4").write_bytes(b"video%d" % i)
        api.store.set_clip(jid, {"index": i, "file": f"clip_{i:02d}.mp4", "title": f"Bit {i}", "rendered_at": 1})
    api.store.update(jid, status="done")
    api.save_clips(api.store.get(jid))
    home = Path(api.store.get(jid)["app_home"])
    assert sorted(p.name for p in (home / "Clips").glob("*.mp4")) == ["01 Bit 1.mp4", "02 Bit 2.mp4"]
