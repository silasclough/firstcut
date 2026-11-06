"""FirstCut for AI apps (mcp_server.py): the MCP protocol over stdio, the tools against a stand-in engine (a local test server), the four
things the user said they will ask their AI, the off switch, Claude Desktop's settings, and the launcher."""
import base64
import io
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from clipfarm.clipper import mcp_server as M

REPO = Path(__file__).resolve().parents[1]


def _jpeg() -> bytes:
    from PIL import Image
    out = io.BytesIO()
    Image.new("RGB", (640, 360), (40, 120, 200)).save(out, "JPEG")
    return out.getvalue()


@pytest.fixture
def engine(tmp_path):
    """A stand-in FirstCut engine on a random local port, a home with server.json and the switch on."""
    home = tmp_path / "home"
    home.mkdir()
    saved = tmp_path / "Weekend Workshop"
    (saved / "Output").mkdir(parents=True)
    (saved / "Notes").mkdir()
    (saved / "Output" / "Ready to Share v2.mp4").write_bytes(b"x")
    (saved / "Ready to Share v2.fcpxml").write_text("<fcpxml/>")
    (saved / "Notes" / "Chapters.txt").write_text("00:00 Welcome\n00:41 Going out in pairs\n")
    calls = []
    jpeg = _jpeg()
    state = {"projects": [{"id": "p1", "title": "Weekend Workshop", "updated": 200, "saved_to": str(saved),
                           "job_state": {"status": "done", "progress": 1.0}},
                          {"id": "p0", "title": "Campus tour", "updated": 100, "saved_to": str(tmp_path / "x"), "job_state": {"status": "done"}}],
             "shorts": [], "transcripts": [{"id": "t00000000a1", "title": "Talk", "status": "done", "updated": 50}],
             "library": {"videos": 3034, "photos": 2, "described": 525, "status": {"running": True, "stage": "Describing", "done": 40, "total": 2800},
                         "roots": [{"path": "/Volumes/Footage/Video"}]}}
    shots = [{"k": 0, "at": 4.0, "length": 3.0, "what": "A person waving at the camera in a hallway"},
             {"k": 1, "at": 20.0, "length": 2.5, "what": "Students walking across the lawn"},
             {"k": 2, "at": 40.0, "length": 3.0, "what": "A crowd in the chapel"}]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj, raw=None, ctype="application/json"):
            body = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            calls.append(("GET", u.path, q))
            p = u.path
            if p == "/api/app/ping":
                return self._send(200, {"ok": True})
            if p == "/api/app":
                return self._send(200, state)
            if p == "/api/library/search":
                return self._send(200, {"shots": [{"id": "abc0000000000001", "k": 3, "rel": "Video/B-roll/Lib.MP4", "start": 2.0, "end": 6.0,
                                                   "what": "Students in the library"}]})
            if p.startswith("/api/library/thumb/"):
                return self._send(200, None, raw=jpeg, ctype="image/jpeg")
            if p == "/api/app/long/p1/shots":
                return self._send(200, {"shots": shots})
            if p.startswith("/api/app/long/p1/shots/") and p.endswith("/search"):
                return self._send(200, {"results": [{"choice": "lib:abc0000000000002:1.00:5.00", "what": "Two people talking on a bench",
                                                     "where": "library", "folder": "Video/B-roll"}]})
            if p.startswith("/api/app/long/p1/shots/") and p.endswith("/choices"):
                return self._send(200, {"choices": [{"choice": 3, "what": "Coffee being poured"}]})
            if p == "/api/app/music/search":
                return self._send(200, {"tracks": [{"id": "m1", "title": "Slow Rain", "artist": "Me", "path": "/x/Slow Rain.mp3"}]})
            if p == "/api/app/music/online":
                return self._send(200, {"tracks": [{"source": "openverse", "id": "ov1", "title": "Open Road", "artist": "Lee Rosevere",
                                                    "license": "CC BY 4.0", "credit": "Music: “Open Road” by Lee Rosevere – CC BY 4.0"}]})
            if p == "/api/app/transcripts/t00000000a1":
                return self._send(200, {"id": "t00000000a1", "title": "Talk", "status": "done", "words": 4, "paragraphs": [{"text": "Hello there everyone."}],
                                        "out": {"txt": "/x/Talk.txt"}})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            calls.append(("POST", urlparse(self.path).path, body))
            p = urlparse(self.path).path
            if p == "/api/app/short":
                return self._send(201, {"id": "s9"})
            if p == "/api/app/long":
                return self._send(201, {"id": "p9"})
            if p == "/api/app/music/fetch":
                return self._send(200, {"path": "/home/Music library/Online/openverse/Lee Rosevere - Open Road.mp3", "name": "Open Road",
                                        "credit": "Music: “Open Road” by Lee Rosevere – CC BY 4.0"})
            if p == "/api/ai/projects/p1/open-fcp":
                return self._send(200, {"ok": True, "path": "/x/a.fcpxml"})
            return self._send(200, {"ok": True})

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    (home / "server.json").write_text(json.dumps({"port": srv.server_address[1]}))
    (home / "settings.json").write_text(json.dumps({"mcp": {"enabled": True}}))
    eng = M.Engine(home, opener=lambda: None, wait_s=0.5)
    yield {"server": M.Server(eng), "calls": calls, "home": home, "shots": shots, "state": state, "tmp": tmp_path}
    srv.shutdown()


def rpc(server, *msgs):
    inp = io.StringIO("".join(json.dumps(m) + "\n" for m in msgs))
    out = io.StringIO()
    M.serve(inp, out, server)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def call(server, name, **args):
    r = rpc(server, {"jsonrpc": "2.0", "id": 9, "method": "tools/call", "params": {"name": name, "arguments": args}})[0]["result"]
    return r, "\n".join(c.get("text", "") for c in r["content"] if c["type"] == "text")


def test_the_handshake_over_a_pipe(engine):
    out = rpc(engine["server"],
              {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t"}}},
              {"jsonrpc": "2.0", "method": "notifications/initialized"},
              {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
              {"jsonrpc": "2.0", "id": 3, "method": "ping"},
              {"jsonrpc": "2.0", "id": 4, "method": "no/such"})
    assert [m["id"] for m in out] == [1, 2, 3, 4]                                         # (the notification gets no answer)
    assert out[0]["result"]["serverInfo"]["name"] == "firstcut" and out[0]["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in out[0]["result"]["capabilities"] and len(out[1]["result"]["tools"]) == len(M.TOOLS)
    assert out[2]["result"] == {} and out[3]["error"]["code"] == -32601
    bad = io.StringIO()
    M.serve(io.StringIO("not json\n"), bad, engine["server"])
    assert json.loads(bad.getvalue())["error"]["code"] == -32700


def test_every_tool_has_a_valid_schema():
    names = [t["name"] for t in M.TOOLS]
    assert len(names) == len(set(names)) and set(names) == set(M.HANDLERS)
    for t in M.TOOLS:
        sch = t["inputSchema"]
        assert t["description"] and sch["type"] == "object" and set(sch["required"]) <= set(sch["properties"])
        for k, v in sch["properties"].items():
            assert v.get("type") in ("string", "number", "integer", "boolean") or "enum" in v, (t["name"], k)


def test_the_off_switch(engine):
    (engine["home"] / "settings.json").write_text(json.dumps({"mcp": {"enabled": False}}))
    r, text = call(engine["server"], "list_projects")
    assert r["isError"] and "Settings › Connect an AI" in text and not engine["calls"]


def test_projects_by_title_or_the_most_recent(engine):
    r, text = call(engine["server"], "project_status")                                     # no project: the most recent
    assert "Weekend Workshop" in text and "Ready to Share v2.mp4" in text and "00:41 Going out in pairs" in text
    r, text = call(engine["server"], "project_status", project="the campus tour video")
    assert "Campus tour" in text
    r, text = call(engine["server"], "project_status", project="t00000000a1")
    assert "Hello there everyone." in text
    r, text = call(engine["server"], "open_in_final_cut", project="nonexistent thing")
    assert r["isError"] and "Weekend Workshop" in text                                  # (says what is there)


def test_1_swap_the_shot_of_the_person_waving(engine):
    """"change out the B-roll shot of the person waving to a shot of two people talking" """
    r, text = call(engine["server"], "replace_shot", which="person waving", **{"with": "two people talking"})
    assert not r["isError"] and "0:04" in text and "Two people talking on a bench" in text and "footage library" in text
    post = [c for c in engine["calls"] if c[0] == "POST"][-1]
    assert post[1] == "/api/app/long/p1/shots/0" and post[2] == {"choice": "lib:abc0000000000002:1.00:5.00"}
    engine["shots"].append({"k": 3, "at": 60.0, "length": 2.0, "what": "Another person waving goodbye"})
    n = len(engine["calls"])
    r, text = call(engine["server"], "replace_shot", which="person waving", **{"with": "two people talking"})
    assert "Which one?" in text and "0:04" in text and "1:00" in text
    assert not [c for c in engine["calls"][n:] if c[0] == "POST"]                         # ambiguous: nothing swapped
    r, text = call(engine["server"], "replace_shot", which="60")
    assert "Coffee being poured" in text                                                   # by time, the editor's suggestion


def test_2_make_it_shorter_and_more_fast_paced(engine):
    """"the video is too long, make it shorter and more fast-paced" """
    r, text = call(engine["server"], "change_video", instruction="the video is too long, make it shorter and more fast-paced",
                   length_minutes=2.4, pace="fast")
    post = [c for c in engine["calls"] if c[0] == "POST"][-1]
    assert post[1] == "/api/ai/projects/p1/revise"
    sent = post[2]["instruction"]
    assert "too long" in sent and "144 seconds" in sent and "fast-paced" in sent and "Weekend Workshop" in text


def test_3_a_clip_from_the_raw_folder_on_the_desktop(engine, monkeypatch):
    """"pull a clip from the raw folder in my videos folder on my desktop" """
    home = engine["tmp"] / "user"
    raw = home / "Desktop" / "Videos" / "RAW"
    raw.mkdir(parents=True)
    (raw / "C0001.MP4").write_bytes(b"x" * 2_000_000)
    (raw / "notes.txt").write_text("not a video")
    monkeypatch.setattr(M.Path, "home", classmethod(lambda cls: home))
    r, text = call(engine["server"], "find_files", where="desktop/videos/raw")
    assert not r["isError"] and "C0001.MP4" in text and "notes.txt" not in text and "2 MB" in text
    r, text = call(engine["server"], "make_short_clips", source="Desktop/videos/raw/c0001.mp4")
    post = [c for c in engine["calls"] if c[0] == "POST"][-1]
    assert post[1] == "/api/app/short" and post[2]["source"] == str(raw / "C0001.MP4")
    r, text = call(engine["server"], "find_files", where="/etc")
    assert r["isError"] and "home folder" in text                                          # never outside home or a drive


def test_4_an_upbeat_song_behind_the_video(engine):
    """"add an upbeat song behind the video": your music first (nothing upbeat there), then free stock music, credited."""
    r, text = call(engine["server"], "add_song", query="upbeat")
    posts = [c for c in engine["calls"] if c[0] == "POST"]
    assert posts[-2][1] == "/api/app/music/fetch" and posts[-1][1] == "/api/app/item/long/p1/song"
    assert posts[-1][2]["song"].endswith("Lee Rosevere - Open Road.mp3")
    assert "Open Road" in text and "Lee Rosevere" in text and "CC BY 4.0" in text and "Credit" in text
    r, text = call(engine["server"], "add_song", query="slow rain")                         # a song of yours that fits: yours
    assert [c for c in engine["calls"] if c[0] == "POST"][-1][2] == {"track": "m1"} and "your music" in text


def test_search_returns_small_pictures(engine):
    r, text = call(engine["server"], "search_footage", query="students in the library", n=1)
    imgs = [c for c in r["content"] if c["type"] == "image"]
    assert "Students in the library" in text and len(imgs) == 1 and imgs[0]["mimeType"] == "image/jpeg"
    from PIL import Image
    assert Image.open(io.BytesIO(base64.b64decode(imgs[0]["data"]))).width == 320


def test_resources_are_the_notes_and_transcripts(engine):
    out = rpc(engine["server"], {"jsonrpc": "2.0", "id": 1, "method": "resources/list"})[0]["result"]["resources"]
    uris = [x["uri"] for x in out]
    assert "firstcut://project/p1/notes/Chapters.txt" in uris and "firstcut://transcript/t00000000a1" in uris
    got = rpc(engine["server"], {"jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": {"uri": "firstcut://project/p1/notes/Chapters.txt"}})
    assert "Going out in pairs" in got[0]["result"]["contents"][0]["text"]


def test_a_closed_app_is_said_plainly(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "settings.json").write_text(json.dumps({"mcp": {"enabled": True}}))
    opened = []
    srv = M.Server(M.Engine(home, opener=lambda: opened.append(1), wait_s=0.2))
    r, text = call(srv, "catalogue_status")
    assert r["isError"] and "Open FirstCut" in text and opened == [1]


def test_claude_desktop_keeps_its_other_tools_and_a_backup(tmp_path):
    cfg = tmp_path / "Claude" / "claude_desktop_config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps({"mcpServers": {"github": {"command": "gh-mcp"}}, "globalShortcut": "Cmd+K"}))
    r = M.add_to_claude_desktop("/Applications/FirstCut.app/Contents/Resources/bin/firstcut-mcp", cfg)
    new = json.loads(cfg.read_text())
    assert new["mcpServers"]["github"] == {"command": "gh-mcp"} and new["globalShortcut"] == "Cmd+K"
    assert new["mcpServers"]["firstcut"]["command"].endswith("firstcut-mcp") and Path(r["backup"]).exists()
    assert M.claude_desktop_state(cfg)["added"]
    cfg.write_text("{ not json")
    with pytest.raises(ValueError, match="isn't valid JSON"):
        M.add_to_claude_desktop("/x", cfg)
    fresh = tmp_path / "New" / "claude_desktop_config.json"
    assert M.add_to_claude_desktop("/x", fresh)["backup"] is None and json.loads(fresh.read_text())["mcpServers"]["firstcut"]["command"] == "/x"


def _fake_app(tmp_path) -> Path:
    app = tmp_path / "FirstCut.app" / "Contents" / "Resources"
    (app / "bin").mkdir(parents=True)
    (app / "app" / "clipfarm").mkdir(parents=True)
    launcher = app / "bin" / "firstcut-mcp"
    launcher.write_text((REPO / "app" / "VideoBuilder" / "firstcut-mcp").read_text())
    launcher.chmod(0o755)
    return launcher


def test_the_launcher_finds_the_apps_home(tmp_path):
    launcher = _fake_app(tmp_path)
    home = tmp_path / "user"
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    r = subprocess.run([str(launcher)], capture_output=True, text=True, env=env, timeout=20)
    assert r.returncode == 1 and "open FirstCut once" in r.stderr and not r.stdout             # (nothing but protocol ever goes to stdout)
    for name, other in (("Video Builder", None), ("FirstCut", "Video Builder")):            # the first name's folder, or the new one
        h = home / "Library" / "Application Support" / name
        py = h / "runtime" / "env" / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.write_text('#!/bin/bash\necho "$VIDEOBUILDER_HOME|$PYTHONPATH|$*"\n')
        py.chmod(0o755)
        (h / "settings.json").write_text("{}")
        r = subprocess.run([str(launcher)], capture_output=True, text=True, env=env, timeout=20)
        home_seen, code, args = r.stdout.strip().split("|")
        assert home_seen == str(h) and code.endswith("Contents/Resources/app") and args == "-m clipfarm mcp"


def test_the_settings_page_route_tests_the_launcher_like_an_ai_app(tmp_path):
    fake = tmp_path / "firstcut-mcp"
    fake.write_text('#!/bin/bash\nwhile read -r l; do\n  case "$l" in *\'"id": 2\'*) echo \'{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"a"},{"name":"b"}]}}\';; '
                    '*\'"id": 1\'*) echo \'{"jsonrpc":"2.0","id":1,"result":{}}\';; esac\ndone\n')
    fake.chmod(0o755)
    assert M.test_launcher(str(fake)) == {"ok": True, "tools": 2, "error": None}
    assert M.test_launcher(str(tmp_path / "missing"))["ok"] is False


def test_the_setting_is_off_until_turned_on_and_the_page_gets_its_state(tmp_path, monkeypatch):
    from clipfarm.clipper import studio as SD
    from clipfarm.clipper import studio_web as W
    assert SD.defaults()["mcp"] == {"enabled": False}
    assert SD.clean({"mcp": {"enabled": 1, "evil": 2}}) == {"mcp": {"enabled": True}}
    api = W.StudioAPI.__new__(W.StudioAPI)
    monkeypatch.setattr(M, "CLAUDE_DESKTOP", tmp_path / "Claude" / "claude_desktop_config.json")
    st = api.mcp_state()
    assert st["launcher"].endswith("firstcut-mcp") and st["tools"] == len(M.TOOLS) and st["claude_desktop"]["added"] is False


def test_codex_gets_its_table_and_everything_else_stays(tmp_path):
    import tomllib
    cfg = tmp_path / ".codex" / "config.toml"
    assert M.codex_state(cfg)["added"] is False
    cfg.parent.mkdir()
    cfg.write_text('# my settings\nmodel = "gpt-5-codex"\n\n[mcp_servers.firstcut]\ncommand = "/old/path"\n\n[mcp_servers.firstcut.env]\nX = "1"\n\n'
                   '[mcp_servers.github]  # keep me\ncommand = "gh"\n[profiles.fast]\nmodel = "o4-mini"\n')
    launcher = '/Applications/First "Cut".app/Contents/Resources/bin/firstcut-mcp'
    r1 = M.add_to_codex(launcher, cfg)
    M.add_to_codex(launcher, cfg)                                                       # a second run never duplicates
    text = cfg.read_text()
    data = tomllib.loads(text)
    assert data["mcp_servers"]["firstcut"] == {"command": launcher, "args": [], "startup_timeout_sec": 60, "tool_timeout_sec": 180}
    assert data["mcp_servers"]["github"] == {"command": "gh"} and data["profiles"]["fast"] == {"model": "o4-mini"} and data["model"] == "gpt-5-codex"
    assert "# my settings" in text and "# keep me" in text and "/old/path" not in text and "[mcp_servers.firstcut.env]" not in text
    assert text.count("[mcp_servers.firstcut]") == 1 and text.count("# FirstCut for AI apps") == 1 and Path(r1["backup"]).exists()
    assert M.codex_state(cfg)["added"]
    fresh = tmp_path / "new" / ".codex" / "config.toml"
    assert M.add_to_codex("/x", fresh)["backup"] is None and tomllib.loads(fresh.read_text())["mcp_servers"]["firstcut"]["command"] == "/x"
