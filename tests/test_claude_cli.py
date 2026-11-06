"""How FirstCut's Claude calls are made: lean (no connectors or skills), pictures inside the message at about half a megapixel."""
import base64
import io
import json
import subprocess
from types import SimpleNamespace

from PIL import Image

from clipfarm.clipper import claude_cli as CC


def _cli(calls, answer=None):
    def run(cmd, input=None, **kw):
        calls.append((cmd, input))
        events = [{"type": "system", "subtype": "init"},
                  {"type": "result", "is_error": False, "structured_output": answer or {"ok": True}, "modelUsage": {"claude-haiku": {}}}]
        return subprocess.CompletedProcess(cmd, 0, "\n".join(json.dumps(e) for e in events).encode(), b"")
    cfg = SimpleNamespace(get=lambda k, d=None: {"llm.claude": {"enabled": True, "bin": "claude"}}.get(k, d))
    return CC.ClaudeCLI(cfg, run=run)


def _jpeg(w, h):
    out = io.BytesIO()
    Image.new("RGB", (w, h), (120, 90, 60)).save(out, "JPEG")
    return out.getvalue()


def test_every_call_skips_connectors_and_skills():
    calls = []
    data, used = _cli(calls).complete_json("sys", "hello", {"type": "object"}, model="haiku")
    cmd, stdin = calls[0]
    assert data == {"ok": True} and used == "claude-haiku"
    assert "--strict-mcp-config" in cmd and "--disable-slash-commands" in cmd
    assert cmd[cmd.index("--tools") + 1] == ""                        # no tools at all
    assert "--input-format" not in cmd and stdin == b"hello"          # plain text prompt when there are no pictures


def test_pictures_go_in_the_message_at_half_a_megapixel():
    calls = []
    big, small = _jpeg(6000, 4000), _jpeg(800, 450)                   # a 24 MP photo and a contact sheet already under the cap
    _cli(calls).complete_json("sys", "describe", {"type": "object"}, images=[big, small])
    cmd, stdin = calls[0]
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert "Read" not in cmd and "--allowedTools" not in cmd
    msg = json.loads(stdin.decode())
    blocks = msg["message"]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "text"] and blocks[-1]["text"] == "describe"
    sizes = [Image.open(io.BytesIO(base64.b64decode(b["source"]["data"]))).size for b in blocks[:2]]
    w, h = sizes[0]
    assert 450_000 < w * h <= 500_000 and abs(w / h - 1.5) < 0.01     # shrunk, same shape
    assert sizes[1] == (800, 450)                                     # small ones are sent as they are
    assert base64.b64decode(blocks[1]["source"]["data"]) == small


def test_a_broken_picture_is_sent_unchanged():
    assert CC._small(b"not a jpeg") == b"not a jpeg"


def test_only_describing_footage_stays_local_and_a_failing_local_model_gives_up_fast(monkeypatch, tmp_path):
    """The Weekend Workshop run (2026-11-05): with "pictures stay on this Mac" the camera director (a decision that comes with
    pictures) went to the local picture model, which answered with nothing four times, five minutes each, at 59%."""
    from clipfarm.clipper import localai as LA
    from clipfarm.core import llm as L
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "vision": "ollama", "fallback": "ollama", "data_dir": "ai"})
    cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "skip_tasks": ["broll_describe", "photo_describe"], "fallback": True})
    local, claude = [], []

    def bad_local(self, system, prompt, schema, images=None):
        local.append(bool(images))
        raise ValueError("Expecting value: line 1 column 1 (char 0)")
    monkeypatch.setattr(L.OllamaLLM, "complete_json", bad_local)
    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    ai = LA.LocalAI(cfg)
    monkeypatch.setattr(ai.claude.__class__, "available", lambda self: True)
    monkeypatch.setattr(ai.claude.__class__, "complete_json", lambda self, *a, **k: (claude.append(k.get("images")) or ({"cams": []}, "sonnet")))
    assert ai.complete_json("multicam_director", "s", "p", {}, images=[b"sheet"]).provider == "claude_plan"
    assert claude == [[b"sheet"]] and local == []
    with __import__("pytest").raises(LA.LLMUnavailable):
        ai.complete_json("broll_describe", "s", "p", {}, images=[b"jpg"])
    assert local == [True] * LA.LOCAL_TRIES and LA.LOCAL_TRIES == 2


def test_the_progress_line_says_which_ai_step_it_is_waiting_on(monkeypatch, tmp_path):
    from clipfarm.clipper import localai as LA
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "vision": "ollama", "fallback": "ollama", "data_dir": "ai"})
    cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "fallback": False})
    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    ai = LA.LocalAI(cfg)
    seen = []

    def flaky(self, *a, **k):
        seen.append(ai.now())
        if len(seen) == 1:
            raise RuntimeError("a bad answer")
        return {"ok": True}, "sonnet"
    monkeypatch.setattr(ai.claude.__class__, "available", lambda self: True)
    monkeypatch.setattr(ai.claude.__class__, "complete_json", flaky)
    assert ai.now() == ""
    ai.complete_json("multicam_director", "s", "p", {})
    assert seen == ["Choosing camera angles · asking Claude", "Choosing camera angles · asking Claude · try 2 of 4"]
    assert ai.now() == ""                                                             # done: nothing to wait on
    assert LA.task_words("flow_script_draft") == "Writing the script" and LA.task_words("something_new") == "Thinking"
