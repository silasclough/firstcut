"""Both AI accounts (2026-11-06: "make it so I can use both Claude and ChatGPT accounts, or I could toggle between them, or it switches
over when one gets out of usage"): the primary answers; when it runs out of usage the backup takes over until the primary's reset time,
then the primary is used again by itself."""
import json
import time
from types import SimpleNamespace

import pytest

from clipfarm.clipper import chatgpt_plan as CP
from clipfarm.clipper import claude_cli as CC
from clipfarm.clipper import localai as LA
from clipfarm.clipper import net
from clipfarm.core import llm as L
from clipfarm.core.config import load_config


@pytest.fixture
def two(monkeypatch, tmp_path):
    """A LocalAI as the app makes it, with both accounts signed in (fakes) and this Mac's model downloaded."""
    calls = []
    state = {"claude": "ok", "chatgpt": "ok", "local_ready": True}

    def make(primary="claude", backup="auto", fallback=""):
        cfg = load_config()
        cfg.root = tmp_path
        cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "chatgpt_plan": {"enabled": True}, "vision": "ollama",
                                                 "fallback": fallback, "data_dir": str(tmp_path / "ai"), "primary": primary, "backup": backup,
                                                 "ollama_model": "qwen3:14b", "vision_model": "qwen3-vl:8b"})
        cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "skip_tasks": ["broll_describe", "photo_describe"],
                                          "config_dir": str(tmp_path / "claude"), "fallback": True, "fallback_on_limit": True})
        return LA.LocalAI(cfg)

    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    monkeypatch.setattr(CC.ClaudeCLI, "bin", property(lambda self: "/usr/local/bin/claude"))
    monkeypatch.setattr(CC.ClaudeCLI, "signed_in", lambda self: True)
    monkeypatch.setattr(CC.ClaudeCLI, "available", lambda self: self.enabled and time.time() >= self.tripped_until)

    def claude(self, system, prompt, schema, **k):
        calls.append("claude")
        if state["claude"] == "out":
            raise CC.ClaudeError("claude reported an error: You've hit your usage limit · resets 3:40pm")
        return {"from": "claude"}, "claude-sonnet"
    monkeypatch.setattr(CC.ClaudeCLI, "complete_json", claude)
    monkeypatch.setattr(CP.ChatGPTPlan, "connected", lambda self: state["chatgpt"] != "gone")

    def chatgpt(self, system, prompt, schema, **k):
        calls.append("chatgpt")
        if state["chatgpt"] == "out":
            raise CP.ChatGPTPlanError("ChatGPT plan usage limit reached")
        return {"from": "chatgpt"}, "gpt-5.4"
    monkeypatch.setattr(CP.ChatGPTPlan, "complete_json", chatgpt)
    monkeypatch.setattr(LA.LocalAI, "local_ready", lambda self, images=False: state["local_ready"])
    monkeypatch.setattr(L.OllamaLLM, "complete_json", lambda self, s, p, sc, images=None: calls.append("local") or
                        L.LLMResult({"from": "local"}, "ollama", self.cfg["model"], {}))
    return SimpleNamespace(make=make, calls=calls, state=state, dir=tmp_path)


def _usage(dir_, name, used, resets):
    d = dir_ / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "usage.json").write_text(json.dumps({"at": time.time(), "five_hour": {"used": used, "resets": resets}, "week": {"used": 0.4, "resets": resets + 86400 * 3}}))


def test_the_primary_answers_first(two):
    assert two.make("claude").complete_json("highlight_assemble", "s", "p", {}).data == {"from": "claude"}
    assert two.make("chatgpt").complete_json("highlight_assemble", "s", "p", {}).data == {"from": "chatgpt"}
    assert two.calls == ["claude", "chatgpt"]


def test_claude_runs_out_and_chatgpt_takes_over_until_the_reset_then_claude_again(two):
    reset = time.time() + 3600
    _usage(two.dir, "claude", 1.0, reset)
    ai = two.make("claude")
    two.state["claude"] = "out"
    assert ai.complete_json("highlight_assemble", "s", "p", {}).data == {"from": "chatgpt"}
    assert two.calls == ["claude", "chatgpt"]                                         # one try: out of usage is not retried
    trips = LA.read_trips(two.dir / "ai")
    assert abs(trips["claude"]["until"] - reset) < 1                                  # until the full window resets
    assert "Claude is out of usage until" in ai.offline_note and "ChatGPT is making the decisions" in ai.offline_note
    other = two.make("claude")                                                        # the next video knows too: straight to ChatGPT
    assert other.complete_json("video_title", "s", "p", {}).data == {"from": "chatgpt"} and two.calls[-1] == "chatgpt" and two.calls.count("claude") == 1
    assert other.in_use()[0] == "chatgpt" and other.available() == (True, f"ChatGPT plan (Claude is out of usage until {LA.clock(reset)})")
    LA.write_trip(two.dir / "ai", "claude", time.time() - 1, "out of usage")          # its window reset
    two.state["claude"] = "ok"
    assert two.make("claude").complete_json("video_title", "s", "p", {}).data == {"from": "claude"}


def test_chatgpt_primary_with_claude_as_the_backup(two):
    two.state["chatgpt"] = "out"
    ai = two.make("chatgpt", "claude")
    t0 = time.time()
    assert ai.complete_json("highlight_assemble", "s", "p", {}).data == {"from": "claude"}
    until = LA.read_trips(two.dir / "ai")["chatgpt"]["until"]
    assert 1700 < until - t0 < 1900                                                   # no usage data from ChatGPT: half an hour


def test_this_mac_as_the_backup(two):
    two.state["claude"] = "out"
    assert two.make("claude", "local").complete_json("highlight_assemble", "s", "p", {}).data == {"from": "local"}


def test_no_backup_says_until_when_in_plain_words(two):
    two.state["claude"] = "out"
    reset = time.time() + 2 * 3600
    _usage(two.dir, "claude", 1.0, reset)
    ai = two.make("claude", "")
    with pytest.raises(LA.LLMUnavailable, match=r"Claude is out of usage until .*(am|pm).*Pick a backup in Settings › AI"):
        ai.complete_json("highlight_assemble", "s", "p", {})
    assert ai.available()[0] is False and "out of usage until" in ai.available()[1]


def test_auto_backup_is_the_other_account_else_this_mac(two):
    assert two.make("claude", "auto").backup() == "chatgpt"
    two.state["chatgpt"] = "gone"                                                     # ChatGPT not signed in
    assert two.make("claude", "auto").backup() == "local"
    two.state["local_ready"] = False
    assert two.make("claude", "auto").backup() == ""
    assert two.make("claude", "claude").backup() == ""                               # (never itself)


def test_offline_still_goes_to_this_mac(two, monkeypatch):
    monkeypatch.setattr(net, "online", lambda *a, **k: False)
    assert two.make("claude", "chatgpt").complete_json("highlight_assemble", "s", "p", {}).data == {"from": "local"}
    assert two.calls == ["local"]


def test_describing_footage_stays_local_when_chosen(two):
    assert two.make("claude").complete_json("broll_describe", "s", "p", {}, images=[b"jpg"]).data == {"from": "local"}


def test_chatgpt_says_plainly_when_it_is_out_of_usage(monkeypatch, tmp_path):
    cfg = SimpleNamespace(root=tmp_path, get=lambda k, d=None: d)
    plan = CP.ChatGPTPlan(cfg)
    monkeypatch.setattr(plan, "_connection", lambda: {"credentials": {"access_token": "t"}})
    monkeypatch.setattr(plan, "models", lambda: [{"slug": "gpt-5.4"}])
    monkeypatch.setattr(CP.requests, "post", lambda *a, **k: SimpleNamespace(status_code=429, ok=False, headers={}))
    with pytest.raises(CP.ChatGPTPlanError, match="usage limit reached"):
        plan.complete_json("s", "p", {})


def test_settings_and_overlay(tmp_path):
    from clipfarm.clipper import studio as SD
    s = SD.defaults()
    assert s["ai"]["backup"] == "auto"
    assert SD.clean({"ai": {"backup": "chatgpt"}})["ai"]["backup"] == "chatgpt"
    assert SD.clean({"ai": {"backup": "rm -rf"}}).get("ai", {}).get("backup") is None
    s["ai"].update(use="claude", backup="chatgpt")
    av = SD.overlay(s, tmp_path)["clipper"]["ai_videos"]
    assert av["primary"] == "claude" and av["backup"] == "chatgpt" and av["chatgpt_plan"]["enabled"] is True
    assert SD.overlay(s, tmp_path)["llm"]["claude"]["enabled"] is True
    s["ai"].update(backup="local", fallback=False)
    assert SD.overlay(s, tmp_path)["clipper"]["ai_videos"]["fallback"] == "ollama"      # a local backup needs the local model


def test_the_sidebar_shows_the_account_in_use(tmp_path):
    from clipfarm.clipper import studio as SD
    from clipfarm.clipper import studio_web as W
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.root, api._cache = tmp_path, {}
    api.chatgpt = SimpleNamespace(status=lambda: {"status": "connected", "model": "gpt-5.4"}, usage=lambda: None)
    api.claude = SimpleNamespace(status=lambda: {"status": "connected"})
    api.setup = SimpleNamespace(_pulled=lambda: set())
    s = SD.defaults()
    s["ai"]["use"] = "claude"
    st = api.ai_state(s)
    assert st["use"] == "claude" and st["backup"] == "chatgpt" and st["out"] is None
    until = time.time() + 3600
    LA.write_trip(tmp_path / "data" / "ai_videos", "claude", until, "out of usage")
    st = api.ai_state(s)
    assert st["use"] == "chatgpt" and st["name"] == "ChatGPT" and st["primary"] == "claude" and abs(st["out"]["until"] - until) < 1
    s["ai"]["backup"] = ""                                                            # nobody takes over: still Claude, out
    st = api.ai_state(s)
    assert st["use"] == "claude" and st["backup"] == "" and st["out"]["acct"] == "claude"


def test_with_finish_on_this_mac_on_a_failing_backup_still_ends_on_this_mac(two):
    """The old "finish the video with the model on this Mac" switch is the last resort after the backup."""
    two.state["claude"] = "out"
    two.state["chatgpt"] = "out"
    assert two.make("claude", "chatgpt", fallback="ollama").complete_json("highlight_assemble", "s", "p", {}).data == {"from": "local"}
    assert two.calls == ["claude", "chatgpt", "local"]
