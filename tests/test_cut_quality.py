"""The long-form cut problems found in the Weekend Workshop video (2026-11-05), one test each:
a voice with nobody on screen, a bad-sounding opening, half-second flashes, jump cuts left bare, B-roll that ignores the pace, and a
progress bar that sat at 59% for 20 minutes."""
import threading
import time

import numpy as np
import pytest

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import multicam as MC
from clipfarm.clipper import soundcheck as S


def _group():
    cam = {"path": "/f/C0071.MP4", "t0": 1846.0, "dur": 609.0, "w": 1920, "h": 1080, "fps": 24.0}
    rec = {"path": "/f/DJI_08.WAV", "t0": 0.0, "dur": 1864.0}
    return {"angles": [{"id": "Camera 3", "files": [cam]}], "recorders": [rec], "dur": 2464.0, "files": ["C0071.MP4"]}


def _pool(n=12, photo=False):
    return [{"start": 100.0 + i * 10, "end": 108.0 + i * 10, "quality": 7, "what": f"shot {i}", "file": f"B{i}.MP4",
             **({"photo": f"/f/p{i}.jpg"} if photo else {})} for i in range(n)]


# 1. nobody on screen ---------------------------------------------------------------------------------------------------
def test_sound_that_no_camera_filmed_is_found_and_never_gets_a_wrong_camera():
    g = _group()
    files = [{"name": "Multicam", "path": None, "start": 0.0, "end": 2464.0, "dur": 2464.0, "multicam": True, "broll": False, "audio": True}]
    spans = LP.no_picture_spans(files, g, [(907.4, 915.1), (1850.0, 1860.0), (1840.0, 1848.0)])
    assert spans == [(907.4, 915.1), (1840.0, 1846.0)]
    parts = LP.mc_parts(g, [{"start": 0, "end": 2464, "angle": "Camera 3"}], 907.4, 915.1)
    assert parts and all(p["v"].get("nopic") and p["v"]["path"] is None for p in parts)        # never C0071 at -938 s
    assert LP.bad_pictures(parts, {}) and "no camera filmed" in LP.bad_pictures(parts, {})[0]
    assert LP.bad_pictures([{"d": 2.0, "v": {"path": "/f/C0071.MP4", "ss": -938.8}}], {"/f/C0071.MP4": 609.0})
    assert not LP.bad_pictures([{"d": 2.0, "v": {"path": "/f/C0071.MP4", "ss": 10.0}}], {"/f/C0071.MP4": 609.0})
    assert LP.subtract_spans([(900.0, 920.0)], [(907.4, 915.1)]) == [(900.0, 907.4), (915.1, 920.0)]


def test_a_span_nobody_filmed_is_covered_from_start_to_end():
    pool = _pool(4)
    taste = [{"at": 5.0, "length": 3.0, "src": 100.0, "scene": 0, "what": "x", "why": ""}]
    cut, missed = LP.cover_spans(taste, [(6.0, 14.0)], pool)
    assert not missed and LP._covered(cut, 6.0, 14.0)
    assert all(c["at"] + c["length"] <= 5.85 or c.get("must") for c in cut)          # the taste shot gives way to the needed ones
    again, _ = LP.cover_spans(cut, [(6.0, 14.0)], pool)                                 # a re-render changes nothing
    assert again == cut
    _, missed = LP.cover_spans([], [(0.0, 60.0)], _pool(2))
    assert missed == [(0.0, 60.0)]


# 2. sound -----------------------------------------------------------------------------------------------------------------
def test_the_cleanest_recording_is_heard_not_always_the_recorder():
    g = _group()
    g["recorders"][0].update(t0=1800.0, dur=700.0)
    g["sound_r"] = {"/f/DJI_08.WAV": {"score": 1.4, "grade": "bad"}, "/f/C0071.MP4": {"score": 59.0, "grade": "good"}}
    assert MC.sound_at(g, 1900.0)["path"] == "/f/C0071.MP4"                         # a recorder in a pocket loses
    g["sound_r"]["/f/DJI_08.WAV"]["score"] = 57.0
    assert MC.sound_at(g, 1900.0)["path"] == "/f/DJI_08.WAV"                         # a near tie: the recorder (the mic on the person)
    g["sound_r"] = {"/f/DJI_08.WAV": {"score": 0.0}, "/f/C0071.MP4": {"score": 0.0}}
    assert MC.sound_at(g, 1900.0)["path"] == "/f/DJI_08.WAV"                         # nothing measured: as before


def test_rating_sources_covers_cameras_and_recorders():
    g = _group()
    MC.rate_sources(g, rate=lambda path, dur: {"score": 10.0 if path.endswith("WAV") else 50.0, "grade": "good"})
    assert set(g["sound_r"]) == {"/f/C0071.MP4", "/f/DJI_08.WAV"}


def test_grades_are_cached_per_window_and_spans_take_the_common_grade():
    calls = []

    def fake(path, start, secs):
        calls.append(start)
        return {"grade": "bad" if start < 20 else "good", "snr": 10.0, "bright": 5.0, "level": -44.0 if start < 20 else -24.0, "why": "muffled"}
    g = S.Grades({}, measure_fn=fake)
    assert g.span("/a.mp4", 0.0, 25.0)["grade"] == "bad" and g.span("/a.mp4", 21.0, 45.0)["grade"] == "good"
    n = len(calls)
    g.span("/a.mp4", 0.0, 45.0)
    assert len(calls) == n and g.dirty                                                 # cached
    assert S.score({"snr": 41.8, "bright": 34.3, "level": -25.9, "grade": "good"}) > S.score({"snr": 15.7, "bright": 13.3, "level": -44.0, "grade": "bad"}) + 40


def test_the_opening_never_plays_badly_recorded_sound():
    iv = [(2607.0, 2613.6), (2613.8, 2615.9), (2071.9, 2076.5), (2077.5, 2080.8), (2250.0, 2290.0)]
    bad, good = {"grade": "bad"}, {"grade": "good"}
    out, gone = LP.opening_without_bad_sound(iv, [bad, bad, good, good, good])
    assert gone == [0, 1] and out[0] == (2071.9, 2076.5)                              # C0018's camera mic no longer opens it
    out, gone = LP.opening_without_bad_sound(iv[:2], [bad, bad])                      # all of it is the bad recording: keep it
    assert gone == []


def test_every_stretch_comes_out_at_the_same_speech_level():
    gains = LP.level_gains([-24.0, -25.0, -9.0, -44.0, -99.0], [10, 10, 10, 5, 3])
    assert gains[0] == 0.0 and gains[1] == 0.0 and gains[2] == -10.0 and gains[3] == 12.0 and gains[4] == 0.0


def test_the_planner_reads_which_moments_sound_bad(tmp_path, monkeypatch):
    from clipfarm.core.config import load_config
    cfg = load_config()
    r = LP.LocalRunner(cfg, None, None)
    st = LP.State(tmp_path)
    files = [{"name": "C0018.MP4", "path": "/f/C0018.MP4", "start": 0.0, "end": 108.0, "dur": 108.0, "audio": True, "broll": False},
             {"name": "C0071.MP4", "path": "/f/C0071.MP4", "start": 108.0, "end": 700.0, "dur": 592.0, "audio": True, "broll": False}]
    monkeypatch.setattr(S, "measure", lambda path, start, secs: {"grade": "bad" if "C0018" in path else "good", "snr": 15.7, "bright": 13.3,
                                                                  "level": -44.0, "why": "far from the mic or noisy, very quiet"})
    cands = [{"id": 0, "start": 2.5, "end": 18.0, "score": 8.0, "what": "a strong line"}, {"id": 1, "start": 300.0, "end": 310.0, "score": 6.0, "what": "fine"}]
    out = r.mark_sound(cands, files, st)
    assert out[0]["sound"] == "bad" and out[0]["score"] == 5.0 and out[0]["what"].startswith("[poor sound: far from the mic")
    assert "sound" not in out[1] and (st.dir / "sound_grades.json").exists()


# 3. flashes ----------------------------------------------------------------------------------------------------------------
def test_photos_three_seconds_apart_no_longer_flash_the_speaker():
    pool = _pool(3, photo=True)
    cut = [{"at": 60.0, "length": 2.5, "src": 100.0, "scene": 0, "photo": "/f/p0.jpg"},
           {"at": 63.0, "length": 2.5, "src": 110.0, "scene": 1, "photo": "/f/p1.jpg"},
           {"at": 66.0, "length": 2.5, "src": 120.0, "scene": 2, "photo": "/f/p2.jpg"}]
    out = LP.no_flash_gaps(cut, pool, 200.0)
    assert [(c["at"], round(c["at"] + c["length"], 2)) for c in out] == [(60.0, 63.0), (63.0, 66.0), (66.0, 68.5)]


def test_a_gap_that_cannot_be_closed_loses_the_later_shot():
    pool = [{"start": 0.0, "end": 3.0, "quality": 7, "what": "a", "file": "A"}, {"start": 10.0, "end": 13.0, "quality": 7, "what": "b", "file": "B"}]
    cut = [{"at": 5.0, "length": 3.0, "src": 0.0, "scene": 0}, {"at": 8.6, "length": 3.0, "src": 10.0, "scene": 1}]
    assert [c["scene"] for c in LP.no_flash_gaps(cut, pool, 100.0)] == [0]


def test_slivers_of_talk_join_their_neighbour_but_never_across_someone_else():
    words = [{"text": "um", "start": 10.1, "end": 10.3}]
    assert LP.merge_tiny([(0.0, 10.0), (10.4, 11.1), (20.0, 30.0)], words) == [(0.0, 11.1), (20.0, 30.0)]
    other = [{"text": "Right,", "start": 10.1, "end": 10.3}]                                 # the interviewer between them
    assert LP.merge_tiny([(0.0, 10.0), (10.4, 11.1), (20.0, 30.0)], other) == [(0.0, 10.0), (10.4, 11.1), (20.0, 30.0)]


def test_the_last_look_runs_broll_on_over_a_flash():
    spk = lambda ss, d: {"d": d, "v": {"path": "/f/cam.mp4", "ss": ss, "fill": False}, "a": {"path": "/f/cam.mp4", "ss": ss}}   # noqa: E731
    br = lambda ss, d: {"d": d, "v": {"path": "/f/b.mp4", "ss": ss, "fill": True, "cut": True}, "a": {"path": "/f/cam.mp4", "ss": 0}}   # noqa: E731
    parts = [spk(0.0, 5.0), br(10.0, 2.5), spk(30.0, 0.5), br(40.0, 2.5), spk(33.0, 5.0)]
    parts[1]["a"]["ss"], parts[3]["a"]["ss"] = 5.0, 30.5
    out, left = LP.fix_flashes(parts, {"/f/b.mp4": 60.0, "/f/cam.mp4": 100.0})
    assert out[2]["v"]["path"] == "/f/b.mp4" and out[2]["v"]["ss"] == 12.5 and out[2]["v0"]["path"] == "/f/cam.mp4" and not left
    # a speaker's own picture never runs on over a different moment (the lips would not match)
    parts = [spk(0.0, 5.0), spk(50.0, 0.5), spk(80.0, 5.0)]
    out, left = LP.fix_flashes(parts, {"/f/cam.mp4": 100.0})
    assert out[1]["v"]["ss"] == 50.0 and left == [5.0]


# 4. jump cuts -----------------------------------------------------------------------------------------------------------------
def test_jump_cuts_inside_one_shot_get_broll_from_just_before_the_cut():
    files = [{"name": "A.MP4", "path": "/f/A.MP4", "start": 0.0, "end": 300.0, "dur": 300.0, "broll": False, "audio": True},
             {"name": "B.MP4", "path": "/f/B.MP4", "start": 300.0, "end": 400.0, "dur": 100.0, "broll": False, "audio": True}]
    iv = [(10.0, 20.0), (25.0, 40.0), (40.1, 50.0), (310.0, 330.0)]
    assert LP.jump_cuts(iv, files) == [(25.0, 5.0)]                                       # 40->40.1 is no jump; A->B is another shot
    pool = _pool(4)
    cut, _ = LP.keep_moving([], [], iv, pool, None, None, 55.0, jumps=[25.0])
    c = next(c for c in cut if c["why"] == "cover the jump cut")
    assert c["at"] == pytest.approx(9.2) and c["at"] + c["length"] >= 11.0               # the cut is at 10.0 s of the video


def test_with_no_broll_a_shoot_switches_camera_at_the_jump():
    g = {"angles": [{"id": "A", "files": [{"path": "/a", "t0": 0.0, "dur": 100.0}]}, {"id": "B", "files": [{"path": "/b", "t0": 0.0, "dur": 100.0}]}],
         "recorders": [], "dur": 100.0}
    cuts = [{"start": 0.0, "end": 100.0, "angle": "A"}]
    _, cuts2 = LP.keep_moving([], cuts, [(0.0, 20.0), (30.0, 60.0)], [], None, g, 50.0, jumps=[30.0], limit=1000)
    assert {"start": 30.0, "end": 100.0, "angle": "B"} in cuts2


def test_jump_covers_stop_before_broll_covers_most_of_the_video():
    pool = _pool(12)
    iv = [(k * 10.0, k * 10.0 + 8.0) for k in range(10)]                                    # nine jumps in 80 s
    cut, _ = LP.keep_moving([], [], iv, pool, None, None, 80.0, jumps=[a for a, _ in iv[1:]], share=0.3, limit=1000)
    assert sum(c["length"] for c in cut) <= 0.3 * 80 + 3.2


# 5. pace ------------------------------------------------------------------------------------------------------------------------
def test_broll_follows_the_pace():
    assert LP.PACE_BROLL["fast"] < LP.PACE_BROLL["natural"] < LP.PACE_BROLL["slow"]
    cut = [{"at": t, "length": 3.0, "src": 0.0, "scene": i, "why": ""} for i, t in enumerate([20.0, 25.0, 30.0, 36.0])]
    assert [c["at"] for c in LP.thin_cutaways(cut, "natural")] == [20.0, 30.0, 36.0]     # 2 s of speaker between: dropped unless fast
    cut[3]["at"] = 35.0
    assert [c["at"] for c in LP.thin_cutaways(cut, "natural")] == [20.0, 30.0]
    assert len(LP.thin_cutaways(cut, "fast")) == 4
    cut[3]["why"] = "cover the jump cut"
    assert [c["at"] for c in LP.thin_cutaways(cut, "natural")] == [20.0, 30.0, 35.0]     # a jump cover always stays


# 6. progress ---------------------------------------------------------------------------------------------------------------------
def test_a_long_step_creeps_on_but_never_reaches_the_next():
    vals = [LP.creep(0.59, t) for t in (0, 60, 600, 1200, 99999)]
    assert vals[0] == 0.59 and all(b > a for a, b in zip(vals, vals[1:])) and vals[-1] < 0.63 + 1e-9
    assert LP.creep(0.97, 9999) == 0.97


def test_the_ticker_writes_what_the_ai_waits_on(monkeypatch):
    class Store:
        def __init__(self):
            self.job = {"progress": 0.59}

        def get(self, j):
            return self.job

        def update(self, j, **kw):
            self.job.update(kw)

    class AI:
        offline_note = ""

        def now(self):
            return "Choosing camera angles · asking Claude · try 2 of 4"
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    r.store, r.cur_ai, r.TICK_S = Store(), AI(), 0.02
    r._pbase = ("j", 0.59, time.time() - 600)
    stop = threading.Event()
    th = threading.Thread(target=r._tick, args=("j", stop))
    th.start()
    time.sleep(0.15)
    stop.set()
    th.join()
    assert r.store.job["detail"].startswith("Choosing camera angles") and 0.59 < r.store.job["progress"] < 0.63


def test_every_recorder_that_heard_a_moment_is_compared():
    g = _group()
    g["recorders"] = [{"path": "/f/DJI_08.WAV", "t0": 0.0, "dur": 1864.0}, {"path": "/f/DJI_11.WAV", "t0": 1819.0, "dur": 645.0}]
    g["sound_r"] = {"/f/DJI_08.WAV": {"score": 26.2}, "/f/DJI_11.WAV": {"score": 49.1}, "/f/C0071.MP4": {"score": 51.0}}
    assert MC.sound_at(g, 1850.0)["path"] == "/f/DJI_11.WAV"                          # (the first recorder used to be the only one asked)


def test_one_recording_played_straight_on_gets_one_gain():
    a = lambda ss, d: {"d": d, "v": {"path": "/v", "ss": ss}, "a": {"path": "/r.wav", "ss": ss}}   # noqa: E731
    parts = [a(10.0, 0.17), a(10.17, 9.8), {"d": 3.0, "v": {"path": "/v", "ss": 50.0}, "a": {"path": "/o.wav", "ss": 5.0}}]
    assert LP.sound_runs(parts) == [[0, 1], [2]]
