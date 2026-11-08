"""Asking for a second video while one is being made puts it in the queue (the sidebar's 'Making it now' list) instead of refusing."""
import threading
from types import SimpleNamespace as NS

import pytest

from clipfarm.clipper import localproj as LP


class Store:
    def __init__(self):
        self.jobs = {"j1": {}, "j2": {}}

    def update(self, job, **kw):
        self.jobs[job].update(kw)

    def get(self, job):
        return self.jobs[job]


def runner():
    projects = NS(get=lambda pid: {"job": {"p1": "j1", "p2": "j2"}[pid]})
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    r.cfg, r.store, r.projects = {}, Store(), projects
    r.lock = threading.Lock()
    r.running = None
    return r


def test_a_second_video_waits_in_the_queue():
    r = runner()
    r.lock.acquire()                                  # (one is being made)
    r.running = "p1"
    r.start("p2", "rerender")
    j = r.store.jobs["j2"]
    assert j["status"] == "queued" and j["run"]["kind"] == "rerender"
    assert "Waiting" in j["stage"]


def test_the_one_being_made_cannot_be_started_again():
    r = runner()
    r.lock.acquire()
    r.running = "p1"
    with pytest.raises(ValueError):
        r.start("p1", "rerender")


def test_a_kept_video_has_all_its_words_in_it():
    from clipfarm.clipper import transcript_edit as TE
    words = [{"text": f"w{k}", "start": float(k), "end": k + 0.8} for k in range(10)]
    plan = {"items": [{"kind": "visual", "start": 0.0, "end": 10.0}], "keep_cut": {"file": "a.mp4", "start": 0.0, "end": 10.0}}
    assert all(TE.kept_flags(plan, [], words))
    plan["keep_cut"] = {"file": "a.mp4", "start": 3.0, "end": 6.0}          # (only the stretch it keeps)
    assert [k for k, v in enumerate(TE.kept_flags(plan, [], words)) if v] == [3, 4, 5]


def test_stop_ends_a_frozen_run_for_good(monkeypatch):
    """A run stuck in something that never returns (an AI answer, an encode) is given up on after the grace time: the project reads Stopped,
    the runner is free for the next video, and when the old run ever ends it changes nothing."""
    import time
    r = runner()
    r.store.jobs["j1"] = {"status": "rendering"}
    r._units_of = lambda pid: 0
    r.cancel_notes = lambda pid, wait=0.0: None
    r._render_children = lambda: []                                   # (nothing to kill in the test)
    release = threading.Event()
    r.frozen = lambda pid: release.wait(30)
    monkeypatch.setattr(LP.ST, "after_run", lambda *a, **k: None)         # (undone after the test: a leaked patch made later tests miss these hooks)
    monkeypatch.setattr(LP.TE, "after_run", lambda *a, **k: None)
    r.start("p1", "frozen")
    for _ in range(50):
        if r.running == "p1":
            break
        time.sleep(0.02)
    assert r.running == "p1"
    assert r.hard_stop("p1", grace=0.05)["stopping"] is True
    for _ in range(200):
        if r.running is None:
            break
        time.sleep(0.05)
    assert r.running is None and r.store.jobs["j1"]["status"] == "cancelled"
    assert r.lock.acquire(blocking=False)                              # free for the next video
    r.lock.release()
    release.set()                                                      # the frozen run finally ends...
    time.sleep(0.3)
    assert r.store.jobs["j1"]["status"] == "cancelled"                 # ...and changes nothing
