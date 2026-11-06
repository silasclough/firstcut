"""Edge cases found by the tester on 2026-11-06 for the FirstCut / Clipper changes of 2026-11-05/06 (offline, sound grading, transcripts,
no-flash B-roll, lean Claude calls). Each test names the behaviour it expects; the ones that fail show a bug in the product code."""
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clipfarm.clipper import localai as LA
from clipfarm.clipper import localproj as LP
from clipfarm.clipper import net
from clipfarm.clipper import soundcheck as S
from clipfarm.clipper import transcript as T
from clipfarm.core import ffmpeg
from clipfarm.core import llm as L
from clipfarm.core.config import load_config

FF = ffmpeg.bin_path("ffmpeg")


# ------------------------------------------------------------------------------------------------ offline: the ChatGPT plan
def _chatgpt_ai(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "chatgpt_plan": {"enabled": True}, "vision": "", "fallback": "",
                                             "data_dir": "ai", "ollama_model": "qwen3:14b", "vision_model": "qwen3-vl:8b"})
    cfg.data["llm"]["claude"].update({"enabled": False})
    monkeypatch.setattr(LA.time, "sleep", lambda s: None)
    ai = LA.LocalAI(cfg)
    monkeypatch.setattr(ai.chatgpt_plan, "connected", lambda: True)
    monkeypatch.setattr(ai, "local_ready", lambda images=False: True)
    calls = []
    monkeypatch.setattr(L.OllamaLLM, "complete_json", lambda self, s, p, sc, images=None: calls.append(self.cfg["model"]) or
                        L.LLMResult({"from": "this mac"}, "ollama", self.cfg["model"], {}))
    return ai, calls


def test_chatgpt_plan_losing_the_connection_mid_video_hands_over_to_this_mac(monkeypatch, tmp_path):
    """The Claude case is covered (test_offline); the ChatGPT plan's own network error ("could not reach ChatGPT for this editor request",
    chatgpt_plan.py) must be recognised the same way, or the whole video fails when the Wi-Fi drops."""
    from clipfarm.clipper.chatgpt_plan import ChatGPTPlanError
    ai, calls = _chatgpt_ai(monkeypatch, tmp_path)
    state = {"online": True}
    monkeypatch.setattr(net, "online", lambda *a, **k: state["online"])

    def dropped(*a, **k):
        state["online"] = False
        raise ChatGPTPlanError("could not reach ChatGPT for this editor request")
    monkeypatch.setattr(ai.chatgpt_plan, "complete_json", dropped)
    assert ai.complete_json("highlight_assemble", "s", "p", {}).provider == "ollama" and calls == ["qwen3:14b"]


def test_looks_offline_reads_the_apps_own_network_errors():
    assert net.looks_offline("could not reach ChatGPT for this editor request")
    assert net.looks_offline("API Error: Connection error.")
    assert not net.looks_offline("claude did not answer within 1200s")
    assert not net.looks_offline("ChatGPT-plan request was not accepted; check your plan usage limit or reconnect")


def test_peek_before_any_check_is_online_and_changed_forces_a_new_check(monkeypatch):
    import importlib
    real = importlib.reload(net)
    try:
        real._state.update(at=0.0, online=True)
        assert real.peek() is True                                                  # nothing checked yet: assume online
        n = []
        monkeypatch.setattr(real, "_reach", lambda h, p, timeout=1.5: n.append(h) or h == "8.8.8.8")
        assert real.online() is True and n == [h for h, _ in real.HOSTS]           # any one host is enough
        real.changed()
        real.online()
        assert len(n) == 2 * len(real.HOSTS)
    finally:
        real._state.update(at=0.0, online=True)


# ------------------------------------------------------------------------------------------------ the progress line
def test_now_reports_parallel_calls_and_minutes(monkeypatch, tmp_path):
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["clipper"]["ai_videos"].update({"openai": {"enabled": False}, "data_dir": "ai"})
    ai = LA.LocalAI(cfg)
    ai._doing = {1: {"task": "broll_describe", "via": "this Mac's model", "try": 2, "of": 2, "since": time.time() - 400},
                 2: {"task": "broll_describe", "via": "this Mac's model", "try": 1, "of": 2, "since": time.time() - 10}}
    assert ai.now() == "Looking at your footage (2 at once) · asking this Mac's model · try 2 of 2 · 7 min"


# ------------------------------------------------------------------------------------------------ Claude CLI
def _cli(out: bytes, code=0):
    calls = []
    from clipfarm.clipper import claude_cli as CC

    def run(cmd, input=None, **kw):
        calls.append((cmd, input, kw))
        return subprocess.CompletedProcess(cmd, code, out, b"boom")
    cfg = SimpleNamespace(get=lambda k, d=None: {"llm.claude": {"enabled": True, "bin": "claude"}}.get(k, d))
    return CC.ClaudeCLI(cfg, run=run), calls, CC


def test_claude_with_no_output_or_an_error_result_raises_plainly():
    import json
    cli, _, CC = _cli(b"")
    with pytest.raises(CC.ClaudeError, match="no JSON"):
        cli.complete_json("s", "p", {})
    cli, _, CC = _cli((json.dumps({"type": "result", "is_error": True, "result": "Not logged in · Please run /login"}) + "\n").encode())
    with pytest.raises(CC.ClaudeError, match="Not logged in"):
        cli.complete_json("s", "p", {})
    cli, _, CC = _cli((json.dumps({"type": "result", "is_error": False, "result": "```json\n{}```"}) + "\n").encode())
    with pytest.raises(CC.ClaudeError, match="not the requested JSON"):
        cli.complete_json("s", "p", {})


def test_claude_never_gets_the_api_key_and_a_timeout_is_a_claude_error(monkeypatch):
    import json
    from clipfarm.clipper import claude_cli as CC
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-pass")
    out = (json.dumps({"type": "result", "structured_output": {"a": 1}}) + "\n").encode()
    cli, calls, _ = _cli(out)
    cli.complete_json("s", "p", {})
    assert "ANTHROPIC_API_KEY" not in calls[0][2]["env"]

    def slow(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 1)
    cli.run = slow
    with pytest.raises(CC.ClaudeError, match="did not answer"):
        cli.complete_json("s", "p", {})


# ------------------------------------------------------------------------------------------------ sound grading
def test_a_recording_that_cannot_be_read_is_never_graded_bad():
    """Grades: "unreadable: say nothing about it (never a reason to drop a line)". A missing file (a drive unplugged mid-run) decodes to
    nothing, which judge() calls "no sound" / bad, and that bad grade is cached into sound_grades.json for good."""
    g = S.Grades()
    r = g.grade("/Volumes/NoSuchDrive/C0001.MP4", 12.0)
    assert r["grade"] != "bad", r


def test_short_and_quiet_inputs_do_not_crash_the_grader():
    assert S.judge(np.zeros(0, np.float32))["why"] == "no sound"
    assert S.judge(np.full(S.SR * 3, 0.5, np.float32))["why"] == "nobody talking"                # a steady hum: nobody talking
    assert S.rate_file("/nope.wav", 0)["grade"] == "bad"
    assert S._window_points(5.0, 5.0, 10.0) == [5.0]
    pts = S._window_points(3.0, 37.0, 10.0)
    assert pts[0] == 3.0 and pts[-1] == 30.0 and len(pts) == 4


# ------------------------------------------------------------------------------------------------ B-roll flashes
def test_two_cutaways_starting_together_never_leave_a_flash():
    pool = [{"start": 0, "end": 20, "quality": 7, "what": "a", "file": "a.mp4"}, {"start": 0, "end": 20, "quality": 7, "what": "b", "file": "b.mp4"}]
    out = LP.no_flash_gaps([{"at": 10.0, "length": 3.0, "src": 2.0, "scene": 0}, {"at": 10.1, "length": 3.0, "src": 2.0, "scene": 1}], pool, 60.0)
    assert all(c["length"] >= 1.0 for c in out), out


def test_fix_flashes_leaves_one_piece_and_back_to_back_files_alone():
    one = [{"d": 0.4, "v": {"path": "/a", "ss": 0.0}, "a": {"path": "/a", "ss": 0.0}}]
    assert LP.fix_flashes([dict(p) for p in one], {"/a": 10.0}) == (one, [])
    assert LP.merge_tiny([(0.0, 0.5)]) == [(0.0, 0.5)]                                  # a lone piece stays (nothing to join)
    assert LP.out_time([(10.0, 20.0), (30.0, 40.0)], 25.0) is None and LP.out_time([(10.0, 20.0), (30.0, 40.0)], 35.0) == 15.0


# ------------------------------------------------------------------------------------------------ transcripts
def _jobs(tmp_path, transcribe):
    out = tmp_path / "Movies"
    return T.Transcripts(tmp_path / "data", None, lambda: out, transcribe=transcribe, download=None), out


def _wait(j, tid, n=300):
    for _ in range(n):
        r = j.get(tid)
        if r["status"] in ("done", "failed"):
            return r
        time.sleep(0.02)
    return j.get(tid)


def test_a_transcript_added_just_as_the_last_one_finishes_still_runs(tmp_path, monkeypatch):
    """_loop reads the queue (list()) outside the lock and then exits; start() in that moment sees the thread alive and does not start
    another, so the new job stays "queued" until the app is restarted."""
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 1.0)
    j, _ = _jobs(tmp_path, lambda p: [{"text": "hi.", "start": 0.0, "end": 0.4}])
    real_list = j.list
    gate, go = threading.Event(), threading.Event()

    def list_():
        r = real_list()
        if threading.current_thread().name == "transcripts" and not any(x["status"] == "queued" for x in r) and not gate.is_set():
            gate.set()
            go.wait(2)                                                              # the worker has seen an empty queue...
        return r
    j.list = list_
    (tmp_path / "a.wav").write_bytes(b"x")
    (tmp_path / "b.wav").write_bytes(b"x")
    first = j.start(str(tmp_path / "a.wav"))
    assert gate.wait(5)
    second = j.start(str(tmp_path / "b.wav"))                                       # ...when the next one is added
    go.set()
    assert _wait(j, first)["status"] == "done"
    assert _wait(j, second)["status"] == "done", j.get(second)


def test_a_video_without_sound_fails_in_plain_words(tmp_path, monkeypatch):
    from clipfarm.clipper import parakeet as P
    exe = tmp_path / "parakeet-mlx"
    exe.write_text("#!/bin/sh\nexit 1\n")
    exe.chmod(0o755)
    monkeypatch.setattr(P, "binary", lambda cfg=None: exe)
    v = tmp_path / "silent picture.mp4"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=15:duration=1", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(v)], check=True, timeout=60)
    j = T.Transcripts(tmp_path / "data", None, lambda: tmp_path / "Movies")
    r = _wait(j, j.start(str(v)))
    assert r["status"] == "failed"
    assert "Output file does not contain any stream" not in r["error"] and "exited" not in r["error"], r["error"]


def test_transcript_formatting_edges():
    assert T.to_srt([]) == "" and T.paragraphs([]) == [] and T.to_txt([]) == "" and T.to_timed([]) == ""
    assert T.paragraphs([{"text": "  ", "start": 0, "end": 1}]) == []
    long = [{"text": "Supercalifragilisticexpialidocious" * 3, "start": 0.0, "end": 1.0}]
    srt = T.to_srt(long)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,000\n")
    words = [{"text": f"w{i}", "start": i * 0.1, "end": i * 0.1 + 0.05} for i in range(200)]       # fast talk: captions never overlap
    blocks = [b.split("\n") for b in T.to_srt(words).strip().split("\n\n")]

    def secs(s):
        h, m, rest = s.split(":")
        sec, ms = rest.split(",")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000
    times = [tuple(secs(x) for x in b[1].split(" --> ")) for b in blocks]
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(times, times[1:])), times[:5]
    assert T.clock(-5) == "00:00" and T.clock(3600) == "1:00:00"
    assert T._free_name(Path("/nonexistent"), "a/b:c", (".txt",)) == "a-b-c" and T._free_name(Path("/nonexistent"), " ..", (".txt",)) == "Transcript"


def test_removing_a_running_transcript_is_refused_and_bad_ids_never_touch_files(tmp_path, monkeypatch):
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 1.0)
    hold = threading.Event()
    j, _ = _jobs(tmp_path, lambda p: hold.wait(5) and [{"text": "hi", "start": 0.0, "end": 0.3}])
    (tmp_path / "a.wav").write_bytes(b"x")
    tid = j.start(str(tmp_path / "a.wav"))
    with pytest.raises(ValueError, match="still being written"):
        j.remove(tid)
    hold.set()
    assert _wait(j, tid)["status"] == "done"
    for bad in ("t0123", "../t0123456789", "t0123456789.words", "T0123456789", ""):
        with pytest.raises(ValueError):
            j.get(bad)


# ------------------------------------------------------------------------------------------------ studio: setup offline, busy, online state
def test_setup_waits_for_the_internet_then_carries_on_and_retries_a_dropped_download(tmp_path, monkeypatch):
    from clipfarm.clipper import studio as SD
    su = SD.Setup(tmp_path)
    state = {"plan": [{"id": "speech", "state": "todo"}]}
    monkeypatch.setattr(su, "plan", lambda: [dict(s, **({"state": su.steps.get(s["id"], {}).get("state", s["state"])})) for s in state["plan"]])
    ticks = iter([False, False, True, True, True, True])
    monkeypatch.setattr(net, "online", lambda *a, **k: next(ticks))
    seen = []
    monkeypatch.setattr(SD.time, "sleep", lambda s: seen.append(su.steps.get("speech", {}).get("stage")))
    tries = []

    def do(st):
        tries.append(1)
        if len(tries) == 1:
            raise RuntimeError("Connection reset: ECONNRESET")
    su._do_speech = do
    su._run()
    assert seen == ["waiting for the internet"] * 2 and len(tries) == 2 and su.steps["speech"]["state"] == "done"


def test_busy_counts_a_transcript_being_written(tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as W
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = None
    api.worker = SimpleNamespace(busy_with=None)
    api.setup = SimpleNamespace(running=lambda: False)
    hold = threading.Event()
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 1.0)
    api.transcripts, _ = _jobs(tmp_path, lambda p: hold.wait(5) and [{"text": "hi", "start": 0.0, "end": 0.3}])
    assert api.busy() == ""
    (tmp_path / "a.wav").write_bytes(b"x")
    tid = api.transcripts.start(str(tmp_path / "a.wav"))
    assert api.busy() == "a transcript is being written"
    hold.set()
    _wait(api.transcripts, tid)
    t = api.transcripts.thread                                                 # (gone once the queue is empty)
    if t is not None:
        t.join(2)
    assert api.busy() == ""


def test_online_state_says_what_still_works_offline(monkeypatch):
    from clipfarm.clipper import studio_web as W
    api = W.StudioAPI.__new__(W.StudioAPI)
    api._cache = {}
    api.setup = SimpleNamespace(plan=lambda: [{"id": "local", "state": "done"}, {"id": "vision", "state": "todo"}])
    monkeypatch.setattr(net, "peek", lambda: True)
    assert api.online_state({}) == {"online": True}
    monkeypatch.setattr(net, "peek", lambda: False)
    assert api.online_state({}) == {"online": False, "local_editing": True, "local_looking": False}


def test_transcript_routes(tmp_path, monkeypatch):
    from clipfarm.clipper import studio_web as W
    monkeypatch.setattr(T.ffmpeg, "duration", lambda p: 1.0)
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = None
    api.transcripts, _ = _jobs(tmp_path, lambda p: [{"text": "Hi there.", "start": 0.0, "end": 0.5}])
    (tmp_path / "a.mov").write_bytes(b"x")
    code, body = api.post(["transcripts"], {"src": str(tmp_path / "a.mov")})
    assert code == 200 and _wait(api.transcripts, body["id"])["status"] == "done"
    code, rec = api.get(["transcripts", body["id"]])
    assert code == 200 and rec["paragraphs"][0]["text"] == "Hi there."
    assert api.get(["transcripts", "t0000000000"])[0] == 404 and api.get(["transcripts", "../x"])[0] == 404
    with pytest.raises(ValueError):
        api.post(["transcripts"], {"src": ""})
    assert api.post(["transcripts", body["id"], "remove"], {}) == (200, {"ok": True})
    assert api.get(["transcripts", body["id"]])[0] == 404
