"""Live dictation: the words appear while they are said. The speech process is faked (a tiny script that speaks the same JSON-lines
protocol as clipfarm/clipper/dictate_worker.py); nothing here needs the model, the network or a microphone."""

import json
import sys
import threading
import urllib.error
import urllib.request

import pytest

from clipfarm.clipper import parakeet as PK
from clipfarm.clipper import studio_web as W

# a fake worker: every "feed" it has heard one more word of the sentence (the last two words are always "still changing")
FAKE = r'''
import sys, json, base64
SENTENCE = "cut the part about parking and make it shorter".split()
n = 0
started = False
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    m = json.loads(line)
    if m["op"] == "quit":
        break
    if m["op"] == "start":
        n, started = 0, True
    if m["op"] == "feed":
        if not started:
            print(json.dumps({"error": "no dictation started"}), flush=True)
            continue
        if m.get("pcm") == base64.b64encode(b"BOOM").decode():
            print(json.dumps({"error": "RuntimeError: out of memory"}), flush=True)
            continue
        n += 1
        words = SENTENCE[:n]
        lead = lambda ws: "".join(" " + w for w in ws)          # (Parakeet's tokens carry their own leading space)
        print(json.dumps({"stable": lead(words[:-2]), "tail": lead(words[-2:])}), flush=True)
        continue
    print(json.dumps({"ok": True}), flush=True)
'''


@pytest.fixture
def live(tmp_path):
    script = tmp_path / "fake_worker.py"
    script.write_text(FAKE)
    # the "recording" grows by one second of silence for every 100 bytes written
    d = PK.LiveDictation(None, tmp_path / "work", cmd=[sys.executable, str(script)],
                         decode=lambda p: b"\0\0" * 16000 * (p.stat().st_size // 100))
    yield d
    d.close()


def piece(n=100):
    return b"x" * n


def test_the_words_arrive_while_the_person_speaks_and_settle_once_two_passes_agree(live):
    sid = live.start()
    seen = [live.push(sid, piece()) for _ in range(5)]
    assert [len(r["stable"].split()) for r in seen] == [0, 1, 2, 3, 4]   # each pass settles the words the one before it already had
    assert seen[0]["tail"] == "cut"
    assert seen[-1]["stable"] == "cut the part about" and seen[-1]["tail"] == "parking"
    stable = [r["stable"].split() for r in seen]
    for x, y in zip(stable, stable[1:]):
        assert y[:len(x)] == x                                       # settled words never change or go away
    assert not any(r["capped"] for r in seen)


def test_only_one_dictation_at_a_time_and_a_new_one_after_the_last_ends(live):
    sid = live.start()
    with pytest.raises(RuntimeError, match="another dictation"):
        live.start()
    live.end(sid)
    assert live.start() != sid


def test_a_dictation_that_went_quiet_is_replaced(tmp_path):
    t = [100.0]
    script = tmp_path / "w.py"
    script.write_text(FAKE)
    d = PK.LiveDictation(None, tmp_path, cmd=[sys.executable, str(script)], decode=lambda p: b"", now=lambda: t[0])
    try:
        d.start()
        t[0] += PK.LIVE_IDLE_S + 1
        assert d.start()
    finally:
        d.close()


def test_a_piece_that_arrives_while_a_pass_runs_just_gets_the_last_words(live):
    sid = live.start()
    first = live.push(sid, piece())
    live._pass.acquire()                                              # a pass is running
    try:
        again = live.push(sid, piece())
    finally:
        live._pass.release()
    assert again["tail"] == first["tail"] and again["stable"] == first["stable"]
    assert live.push(sid, piece())["tail"] != first["tail"]           # and the next one catches up


def test_a_wrong_session_a_huge_piece_and_a_finished_dictation_are_refused(live):
    sid = live.start()
    with pytest.raises(KeyError):
        live.push("nope", piece())
    with pytest.raises(ValueError):
        live.push(sid, b"x" * (PK.LIVE_CHUNK_MAX + 1))
    live.end(sid)
    with pytest.raises(KeyError):
        live.push(sid, piece())


def test_the_recording_file_is_removed_when_it_ends(live, tmp_path):
    sid = live.start()
    live.push(sid, piece())
    assert list((tmp_path / "work").glob("live-*.part"))
    live.end(sid)
    assert not list((tmp_path / "work").glob("live-*.part"))


def test_a_very_long_dictation_stops_showing_live_words(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(FAKE)
    d = PK.LiveDictation(None, tmp_path, cmd=[sys.executable, str(script)], decode=lambda p: b"\0\0" * int(16000 * (PK.LIVE_MAX_S + 5)))
    try:
        sid = d.start()
        assert d.push(sid, piece())["capped"] is True
    finally:
        d.close()


def test_a_speech_process_that_fails_raises_so_the_page_falls_back_to_the_final_text(tmp_path):
    script = tmp_path / "w.py"
    script.write_text(FAKE)
    d = PK.LiveDictation(None, tmp_path, cmd=[sys.executable, str(script)], decode=lambda p: b"BOOM")
    try:
        sid = d.start()
        with pytest.raises(RuntimeError, match="out of memory"):
            d.push(sid, piece())
    finally:
        d.close()
    d2 = PK.LiveDictation(None, tmp_path, cmd=[sys.executable, "-c", "import sys; sys.exit(3)"], decode=lambda p: b"\0\0" * 16000)
    try:
        sid = d2.start()
        with pytest.raises(RuntimeError):
            d2.push(sid, piece())
    finally:
        d2.close()


def test_without_parakeet_installed_the_start_says_so(tmp_path):
    d = PK.LiveDictation(None, tmp_path)                                # no cmd, and no parakeet-mlx next to it
    with pytest.raises(RuntimeError, match="isn't installed"):
        d.start()


def test_the_editors_words_are_spelled_the_way_editors_write_them():
    assert W.StudioAPI._dictation_text(None, "swap the be roll for the final cut  version , ok") == "swap the B-roll for the Final Cut version, ok"


def test_the_live_routes_over_http_start_push_and_end(tmp_path, monkeypatch):
    from test_studio_edges import _studio_server
    home = tmp_path / "VB home"
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(home))
    from clipfarm.clipper import studio as SD
    monkeypatch.setattr(SD, "_answers", lambda url: False)
    monkeypatch.setattr(SD, "ram_gb", lambda: 16.0)
    script = tmp_path / "w.py"
    script.write_text(FAKE)
    httpd, base = _studio_server(tmp_path, home, monkeypatch)
    try:
        monkeypatch.setattr(PK, "worker_python", lambda cfg=None: None)
        monkeypatch.setattr(PK.LiveDictation, "_command", lambda self: [sys.executable, str(script)])
        monkeypatch.setattr(PK, "_decode_pcm", lambda p: b"\0\0" * 16000 * (p.stat().st_size // 100))

        def post(path, data=b""):
            req = urllib.request.Request(base + path, data=data, method="POST", headers={"Content-Type": "audio/mp4"})
            try:
                with urllib.request.urlopen(req, timeout=20) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read() or b"{}")
        st, out = post("/api/app/dictate/live/start")
        assert st == 200 and out["id"]
        sid = out["id"]
        for _ in range(3):
            st, out = post(f"/api/app/dictate/live/{sid}", b"x" * 100)
        assert st == 200 and (out["stable"] + " " + out["tail"]).split() == ["cut", "the", "part"] and out["capped"] is False
        assert post("/api/app/dictate/live/start")[0] == 409                         # another is going
        assert post("/api/app/dictate/live/nope", b"x" * 100)[0] == 404
        assert post(f"/api/app/dictate/live/{sid}/end")[0] == 200
        assert post("/api/app/dictate/live/start")[0] == 200
    finally:
        httpd.shutdown()
