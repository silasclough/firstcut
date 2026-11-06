"""Offline (2026-11-05: "make sure everything works properly when it's offline. It realizes it's offline. And if there is an AI
model installed, you can still edit ... you can still do stuff like the transcription")."""
import json
import subprocess
from pathlib import Path

import pytest

from clipfarm.clipper import localai as LA
from clipfarm.clipper import net
from clipfarm.core import llm as L
from clipfarm.core.config import load_config


def _ai(monkeypatch, tmp_path, local_models=("qwen3:14b", "qwen3-vl:8b")):
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "vision": "ollama", "fallback": "", "data_dir": "ai",
                                             "ollama_model": "qwen3:14b", "vision_model": "qwen3-vl:8b"})
    cfg.data["llm"]["claude"].update({"enabled": True, "tasks": [], "skip_tasks": ["broll_describe"], "fallback": False})
    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    ai = LA.LocalAI(cfg)
    monkeypatch.setattr(ai.claude.__class__, "available", lambda self: True)
    monkeypatch.setattr(ai, "local_ready", lambda images=False: ("qwen3-vl:8b" if images else "qwen3:14b") in local_models)
    calls = []
    monkeypatch.setattr(L.OllamaLLM, "complete_json", lambda self, s, p, sc, images=None: calls.append(self.cfg["model"]) or
                        L.LLMResult({"from": "this mac"}, "ollama", self.cfg["model"], {}))
    return ai, calls


def test_offline_with_the_model_downloaded_it_still_edits(monkeypatch, tmp_path):
    ai, calls = _ai(monkeypatch, tmp_path)
    monkeypatch.setattr(ai.claude.__class__, "complete_json", lambda *a, **k: pytest.fail("the account is never asked offline"))
    monkeypatch.setattr(net, "online", lambda *a, **k: False)
    assert ai.complete_json("highlight_assemble", "s", "p", {}).data == {"from": "this mac"}
    assert calls == ["qwen3:14b"] and "offline" in ai.offline_note


def test_offline_without_a_model_says_so_plainly(monkeypatch, tmp_path):
    ai, _ = _ai(monkeypatch, tmp_path, local_models=())
    monkeypatch.setattr(net, "online", lambda *a, **k: False)
    with pytest.raises(LA.LLMUnavailable, match="you're offline, and this Mac has no editing model"):
        ai.complete_json("highlight_assemble", "s", "p", {})


def test_the_connection_dropping_mid_video_hands_over_to_this_mac(monkeypatch, tmp_path):
    from clipfarm.clipper.claude_cli import ClaudeError
    ai, calls = _ai(monkeypatch, tmp_path)
    state = {"online": True}
    monkeypatch.setattr(net, "online", lambda *a, **k: state["online"])

    def dropped(*a, **k):
        state["online"] = False                                                  # the Wi-Fi went while it was asking
        raise ClaudeError("claude reported an error: API Error: Connection error.")
    monkeypatch.setattr(ai.claude.__class__, "complete_json", dropped)
    assert ai.complete_json("multicam_director", "s", "p", {}).provider == "ollama" and calls == ["qwen3:14b"]


def test_a_real_error_is_not_mistaken_for_offline(monkeypatch, tmp_path):
    from clipfarm.clipper.claude_cli import ClaudeError
    ai, calls = _ai(monkeypatch, tmp_path)
    monkeypatch.setattr(ai.claude.__class__, "complete_json", lambda *a, **k: (_ for _ in ()).throw(ClaudeError("the answer was not the requested JSON")))
    with pytest.raises(LA.LLMUnavailable, match="requested JSON"):
        ai.complete_json("highlight_assemble", "s", "p", {})
    assert calls == []
    assert net.looks_offline("getaddrinfo ENOTFOUND api.anthropic.com") and not net.looks_offline("rate limit reached")


def test_the_online_check_is_cached_and_cheap(monkeypatch):
    import importlib
    real = importlib.reload(net)                                                   # (conftest replaced online(); test the real one)
    tried = []
    monkeypatch.setattr(real, "_reach", lambda h, p, timeout=1.5: tried.append(h) or False)
    real.changed()
    assert real.online() is False and real.peek() is False and len(tried) == len(real.HOSTS)
    assert real.online() is False and len(tried) == len(real.HOSTS)               # cached: no new attempts
    monkeypatch.setattr(real, "_reach", lambda h, p, timeout=1.5: True)
    real.changed()
    assert real.online() is True
    real._state.update(at=0.0, online=True)


def test_speech_to_text_uses_the_model_on_disk_when_offline(monkeypatch, tmp_path):
    from clipfarm.clipper import parakeet as P
    exe = tmp_path / "parakeet-mlx"
    exe.write_text("#!/bin/sh\n")
    monkeypatch.setattr(P, "binary", lambda cfg=None: exe)
    monkeypatch.setattr(P.ffmpeg, "run", lambda *a, **k: None)
    monkeypatch.setattr(P.ffmpeg, "duration", lambda p: 10.0)
    envs = []

    def run(cmd, **kw):
        envs.append(kw.get("env") or {})
        out = Path(cmd[cmd.index("--output-dir") + 1]) / "audio.json"
        out.write_text(json.dumps({"sentences": [{"tokens": [{"text": " hi", "start": 0.1, "end": 0.4}]}]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(P.subprocess, "run", run)
    monkeypatch.setattr(net, "online", lambda *a, **k: False)
    P.transcribe(tmp_path / "a.wav")
    assert envs[-1].get("HF_HUB_OFFLINE") == "1"
    monkeypatch.setattr(net, "online", lambda *a, **k: True)
    P.transcribe(tmp_path / "a.wav")
    assert "HF_HUB_OFFLINE" not in envs[-1] or envs[-1]["HF_HUB_OFFLINE"] != "1" or __import__("os").environ.get("HF_HUB_OFFLINE") == "1"


def test_a_local_answer_has_a_length_cap(monkeypatch):
    """A JSON answer that loops would run for ever and block every request behind it (work Mac, 2026-11-06)."""
    sent = []

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}}
    monkeypatch.setattr(L.requests, "post", lambda url, json=None, timeout=None: sent.append(json) or R())
    L.OllamaLLM({"model": "qwen3:14b", "vision_model": "qwen3-vl:8b"}).complete_json("s", "p", {}, images=[b"x"])
    L.OllamaLLM({"model": "qwen3:14b"}).complete_json("s", "p", {})
    assert sent[0]["options"]["num_predict"] == 2048 and sent[0]["options"]["temperature"] == 0.5
    assert sent[1]["options"]["num_predict"] == 8192


def test_this_macs_speed_is_measured_from_each_local_answer(monkeypatch):
    """The sidebar on "only this Mac": tokens a second, from Ollama's own counts (eval_count over eval_duration)."""
    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "{}"}, "eval_count": 400, "eval_duration": 20_000_000_000}
    monkeypatch.setattr(L.requests, "post", lambda url, json=None, timeout=None: R())
    L.LOCAL_SPEEDS.clear()
    L.OllamaLLM({"model": "qwen3:14b"}).complete_json("s", "p", {})
    assert L.LOCAL_SPEED["tps"] == 20.0 and L.LOCAL_SPEED["model"] == "qwen3:14b" and L.LOCAL_SPEEDS == [20.0]
    L.note_speed("qwen3:14b", 3, 1_000_000)                                              # a tiny answer says nothing about speed
    assert L.LOCAL_SPEEDS == [20.0]
