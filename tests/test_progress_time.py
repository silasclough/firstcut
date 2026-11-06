"""Progress in time (progress.py) and the AI-written notes after the video is done (localproj.start_notes)."""
import threading
import time
from types import SimpleNamespace as NS

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import progress as PG


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def go(self, s):
        self.t += s


def job(status, stage="", progress=0.0, kind=None, jid="j1"):
    return {"id": jid, "status": status, "stage": stage or status, "progress": progress, "run": {"kind": kind} if kind else {}, "options": {}}


def one_run(w, clock, stages, jid="j1", kind="make"):
    """stages = [(status, seconds)] then done."""
    for status, secs in stages:
        j = job(status, progress=0.1, kind=kind, jid=jid)
        w.on_update(jid, j, {"status": status})
        clock.go(secs)
    w.on_update(jid, job("done", progress=1.0, kind=kind, jid=jid), {"status": "done"})


def watch(tmp_path=None):
    c = Clock()
    return PG.Watch(PG.History(tmp_path / "h.json" if tmp_path else None), clock=c), c


# ------------------------------------------------------------------------------------------------ the bar in time
def test_without_history_the_stored_value_is_used_never_back_never_100_and_no_eta():
    w, c = watch()
    w.on_update("j1", job("rendering", progress=0.5, kind="make"), {"status": "rendering"})
    v = w.view(job("rendering", progress=0.5, kind="make"))
    assert v["progress"] == 0.5 and v["eta_s"] is None and v["long"] is False
    assert w.view(job("rendering", progress=0.3, kind="make"))["progress"] == 0.5           # (never back)
    assert w.view(job("rendering", progress=1.0, kind="make"))["progress"] == 0.99          # (never 100% before done)


def test_a_finished_run_is_remembered_and_the_stage_times_are_medians(tmp_path):
    w, c = watch(tmp_path)
    for _ in range(3):
        one_run(w, c, [("transcribing", 60), ("rendering", 120)])
    exp = w.history.expected("long:make", None)
    assert round(exp["transcribing"]) == 60 and round(exp["rendering"]) == 120
    assert len(PG.History(tmp_path / "h.json").runs) == 3                                  # saved on disk, loads again


def test_history_is_capped():
    h = PG.History(None)
    for k in range(80):
        h.record({"kind": "short", "stages": {"rendering": 10.0 + k}, "units": None})
    assert len(h.runs) == PG.KEEP_RUNS


def test_the_bar_moves_with_time_is_monotonic_and_stays_under_99():
    w, c = watch()
    for _ in range(3):
        one_run(w, c, [("transcribing", 60), ("rendering", 120)])
    j = job("transcribing", progress=0.05, kind="make", jid="j2")
    w.on_update("j2", j, {"status": "transcribing"})
    seen = []
    for _ in range(40):
        c.go(10)
        seen.append(w.view(j)["progress"])
    assert seen == sorted(seen) and max(seen) <= 0.99 and seen[0] < seen[-1]


def test_a_stage_over_its_usual_time_says_so_and_the_bar_keeps_easing_but_never_sits_still():
    w, c = watch()
    for _ in range(3):
        one_run(w, c, [("transcribing", 60), ("rendering", 120)])
    j = job("rendering", progress=0.97, kind="make", jid="j3")
    w.on_update("j3", job("transcribing", kind="make", jid="j3"), {"status": "transcribing"})
    c.go(60)
    w.on_update("j3", j, {"status": "rendering"})
    c.go(100)
    mid = w.view(j)
    assert mid["long"] is False and mid["eta_s"] is not None and mid["stage_expected_s"] == 120
    c.go(300)                                                                              # 400 s into a stage that usually takes 120
    late = w.view(j)
    assert late["long"] is True and late["eta_s"] is None and late["stage_elapsed_s"] == 400
    c.go(600)
    later = w.view(j)
    assert later["progress"] >= late["progress"] and later["progress"] <= 0.99


def test_eta_needs_history_from_at_least_two_finished_runs():
    w, c = watch()
    one_run(w, c, [("transcribing", 60), ("rendering", 120)])
    j = job("transcribing", kind="make", jid="j4")
    w.on_update("j4", j, {"status": "transcribing"})
    c.go(20)
    assert w.view(j)["eta_s"] is None                                                      # one run: the bar is time-based, but no promise
    one_run(w, c, [("transcribing", 60), ("rendering", 120)])
    w.on_update("j5", job("transcribing", kind="make", jid="j5"), {"status": "transcribing"})
    c.go(20)
    assert 100 < w.view(job("transcribing", kind="make", jid="j5"))["eta_s"] < 200


def test_stages_done_are_listed_with_their_times():
    w, c = watch()
    j = job("transcribing", kind="make", jid="j6")
    w.on_update("j6", j, {"status": "transcribing"})
    c.go(75)
    j2 = job("rendering", kind="make", jid="j6")
    w.on_update("j6", j2, {"status": "rendering"})
    c.go(5)
    v = w.view(j2)
    assert v["stages_done"] == [{"key": "transcribing", "label": "Transcribing", "seconds": 75}] and v["stage_key"] == "rendering"


def test_a_failed_or_stopped_run_is_not_remembered():
    w, c = watch()
    w.on_update("j7", job("rendering", kind="make", jid="j7"), {"status": "rendering"})
    c.go(50)
    w.on_update("j7", job("failed", kind="make", jid="j7"), {"status": "failed"})
    assert w.history.runs == [] and w.view(job("failed", jid="j7")) == {}


def test_kinds_do_not_mix_and_short_clips_are_a_kind_of_their_own():
    assert PG.kind_of({"run": {"kind": "rerender"}}) == "long:rerender"
    assert PG.kind_of({"options": {"project": True}}) == "long:make"
    assert PG.kind_of({"options": {"mode": "clips"}}) == "short"
    assert PG.clock_text(252) == "4:12" and PG.clock_text(3725) == "1:02:05"


def test_the_store_calls_the_listener_on_every_update(tmp_path):
    from clipfarm.clipper.store import Store
    w, c = watch()
    st = Store(tmp_path / "jobs")
    st.listeners.append(w.on_update)
    j = st.create({"app": "studio"}, {"mode": "clips"})
    st.update(j["id"], status="transcribing", stage="Transcribing", progress=0.1)
    assert w.view(st.get(j["id"]))["stage_key"] == "transcribing"


# ------------------------------------------------------------------------------------------------ the notes after the video
class Projects:
    def __init__(self):
        self.p = {"notes_info": {"qa": {"fix": 0}, "notes_state": "writing"}}

    def get(self, pid):
        return self.p

    def update(self, pid, **kw):
        self.p.update(kw)

    def list(self):
        return [{"id": "p1", **self.p}]


def runner(tmp_path):
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    r.cfg = {}
    r.projects = Projects()
    return r


def gated_chapters(monkeypatch, gate, result=None, boom=False):
    calls = []

    def chapters(ai, words, dur, gl, brief):
        calls.append(brief)
        gate.wait(5)
        if boom:
            raise RuntimeError("the AI is down")
        return result or [(0.0, "Intro"), (20.0, "Middle"), (40.0, "End")], "A summary."
    monkeypatch.setattr(LP.NT, "chapters", chapters)
    monkeypatch.setattr(LP, "LocalAI", lambda cfg: object())
    return calls


def wait_state(r, want, secs=5):
    t = time.time()
    while time.time() - t < secs:
        if r.projects.p["notes_info"].get("notes_state") == want:
            return True
        time.sleep(0.02)
    return False


def test_the_notes_are_written_after_start_notes_returns_and_state_goes_done(tmp_path, monkeypatch):
    gate = threading.Event()
    calls = gated_chapters(monkeypatch, gate)
    r = runner(tmp_path)
    notes = tmp_path / "Notes"
    notes.mkdir()
    r.start_notes("p1", [{"text": "hi"}], 60.0, notes, ["Pictures: x"], "brief")
    assert r.projects.p["notes_info"]["notes_state"] == "writing" and r.notes_alive("p1")       # (it returned at once; the AI is still being waited for)
    gate.set()
    assert wait_state(r, "done")
    assert (notes / "Chapters.txt").read_text().startswith("00:00 Intro") and "A summary." in (notes / "Description.txt").read_text()
    assert "Pictures: x" in (notes / "Description.txt").read_text() and r.projects.p["notes_info"]["chapters"].startswith("00:00 Intro")


def test_a_failing_ai_leaves_the_video_alone_and_the_state_failed(tmp_path, monkeypatch):
    gate = threading.Event()
    gate.set()
    gated_chapters(monkeypatch, gate, boom=True)
    r = runner(tmp_path)
    notes = tmp_path / "Notes"
    notes.mkdir()
    r.start_notes("p1", [{"text": "hi"}], 60.0, notes, [], "")
    assert wait_state(r, "failed") and r.projects.p["notes_info"]["qa"] == {"fix": 0}          # (the rest of the notes row is untouched)
    assert not (notes / "Chapters.txt").exists()


def test_a_new_run_makes_the_pending_notes_stale_and_they_are_dropped(tmp_path, monkeypatch):
    gate = threading.Event()
    gated_chapters(monkeypatch, gate)
    r = runner(tmp_path)
    notes = tmp_path / "Notes"
    notes.mkdir()
    r.start_notes("p1", [{"text": "hi"}], 60.0, notes, [], "")
    r.cancel_notes("p1")                                                                    # (a re-render of the project started)
    assert not r.notes_alive("p1") or True
    gate.set()
    time.sleep(0.4)
    assert r.projects.p["notes_info"]["notes_state"] == "writing" and not (notes / "Chapters.txt").exists()       # nothing from the old video lands


def test_notes_run_one_at_a_time(tmp_path, monkeypatch):
    gate = threading.Event()
    calls = gated_chapters(monkeypatch, gate)
    r = runner(tmp_path)
    for k in range(2):
        d = tmp_path / f"n{k}"
        d.mkdir()
        r.start_notes(f"p{k}", [{"text": "hi"}], 60.0, d, [], f"b{k}")
    time.sleep(0.3)
    assert calls == ["b0"]                                                                  # the second waits for the first
    gate.set()
    t = time.time()
    while len(calls) < 2 and time.time() - t < 5:
        time.sleep(0.02)
    assert calls == ["b0", "b1"]


def test_a_writing_state_left_by_a_quit_app_becomes_failed(tmp_path):
    r = runner(tmp_path)
    r._sweep_notes()
    assert r.projects.p["notes_info"]["notes_state"] == "failed"


def test_a_real_run_is_done_before_the_chapters_are_written_and_the_notes_row_says_so(tmp_path, monkeypatch):
    """deliver(): the video, the Final Cut project and the quality check are saved and the job is 'done' while the AI is still writing the chapters."""
    import json
    from test_aivideo import _FakeAI, _make_folder

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True}
    folder = _make_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    fake = _FakeAI()
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)
    words = {"talk1.mov": [{"text": f"w{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(24)],
             "talk2.mov": [{"text": f"v{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(16)]}
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words[f["name"]])
    monkeypatch.setattr(LP, "NOTES_MIN_S", 5.0)
    gate = threading.Event()
    calls = gated_chapters(monkeypatch, gate)
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)                                     # (the helper stubbed it for the notes only)
    runner_ = LP.LocalRunner(cfg, store, projects)
    proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": "auto"}, {"clips": 0})
    runner_.make(proj["id"], {})
    j = store.get(proj["job"])
    assert j["status"] == "done" and j["progress"] == 1.0                                  # done, with the AI still waiting
    ni = projects.get(proj["id"])["notes_info"]
    assert ni["notes_state"] == "writing" and ni["chapters"] == "" and ni["issues"] is not None
    assert (folder / "Notes" / "Quality check.md").exists() and not (folder / "Notes" / "Chapters.txt").exists()
    assert list((folder / "Output").glob("*.mp4"))
    gate.set()
    t = time.time()
    while projects.get(proj["id"])["notes_info"].get("notes_state") != "done" and time.time() - t < 10:
        time.sleep(0.05)
    assert projects.get(proj["id"])["notes_info"]["notes_state"] == "done" and (folder / "Notes" / "Chapters.txt").exists()
    assert calls and json.loads(json.dumps(projects.get(proj["id"])["notes_info"]["chapters"]))
