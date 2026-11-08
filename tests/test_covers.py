"""WP7: hiding jump cuts (covers.py), punch-ins on the face, multicam switching rules (switching.py)."""
import subprocess

import pytest

from clipfarm.clipper import covers as COV
from clipfarm.clipper import localproj as LP
from clipfarm.clipper import switching as SW
from clipfarm.core import ffmpeg

FF = ffmpeg.bin_path("ffmpeg")


def _pool(n):
    return [{"start": k * 10.0, "end": k * 10.0 + 6.0, "what": f"shot {k}", "quality": 7, "file": f"b{k}.mp4"} for k in range(n)]


def _devices(**kw):
    return {"punch": True, "prefer": ("angle", "broll", "punch"), "last": None, **kw}


def _run(jumps, pool, devices, group=None, cuts=None, protected=None, choose=None, iv=None, out_len=120.0):
    report = []
    iv = iv or [(0.0, 20.0)] + [(a, a + 20.0) for a in range(30, 30 + 10 * len(jumps), 10)]
    cut, cuts2 = LP.keep_moving([], cuts or [], iv, pool, None, group, out_len, jumps=jumps, limit=1000, devices=devices, report=report,
                                protected=protected, choose=choose)
    return cut, cuts2, report


# --- the cover planner ---------------------------------------------------------------------------------------------------------
def test_devices_rotate_and_never_repeat_in_a_row():
    out, notes = COV.plan_covers([{"t": float(k), "can": {"punch": True, "angle": True, "broll": True}} for k in range(8)])
    assert len(out) == 8 and all(a["device"] != b["device"] for a, b in zip(out, out[1:]))
    # five jump cuts, one angle (no other camera), a pool with two matching scenes: punch and B-roll take turns
    jumps = [30.0, 40.0, 50.0, 60.0, 70.0]
    dev = _devices(prefer=("punch", "angle", "broll"))
    cut, _, _ = _run(jumps, _pool(2), dev)
    events = sorted([(c["at"] + 0.8, "broll") for c in cut] + [(t, "punch") for t in dev["punch_at"]])
    seq = [e[1] for e in events]
    assert len(seq) == 5 and all(a != b for a, b in zip(seq, seq[1:])), seq
    assert seq.count("broll") == 2 and seq.count("punch") == 3


def test_no_broll_cover_inside_a_protected_span_or_exclude_window():
    assert COV.blocked(29.0, 32.0, [(30.0, 40.0)]) and not COV.blocked(10.0, 12.0, [(30.0, 40.0)])
    dev = _devices(prefer=("broll", "punch", "angle"))
    cut, _, _ = _run([30.0, 40.0], _pool(3), dev, protected=[(25.0, 35.0)])
    assert not any(abs(c["at"] + 0.8 - 30.0) < 1.0 for c in cut)                    # nothing over the protected jump
    assert any(abs(t - 30.0) < 0.5 for t in dev["punch_at"])                         # a punch-in took it instead


def test_when_nothing_matches_the_jump_cut_stays_and_is_reported_not_filled():
    dev = _devices(punch=False)
    cut, cuts2, report = _run([30.0], _pool(3), dev, choose=lambda t, free: None)
    assert cut == [] and not dev.get("punch_at") and cuts2 == []
    assert report and "stays on the speaker" in report[0]
    out, notes = COV.plan_covers([{"t": 41.2, "can": {"punch": False, "angle": False, "broll": False}}])
    assert out == [] and "41.2" in notes[0]


# --- punch-ins on the face -----------------------------------------------------------------------------------------------------
def test_punch_in_centres_on_the_face_and_keeps_the_eye_line():
    face = {"cx": 0.3, "cy": 0.35, "h": 0.2}
    z = 1.12
    cx, cy = COV.punch_centre(face, z)
    assert cx < 0.5 and cx >= 0.5 / z - 1e-3                                          # moved towards the face, never out of the frame
    assert 0.5 / z - 1e-3 <= cy <= 1 - 0.5 / z + 1e-3
    assert 0.33 <= COV.eye_line_of(face, z, (cx, cy)) <= 0.40
    assert COV.punch_centre(None, z) == (0.5, 0.5)                                    # no face: the middle
    x, y = COV.crop_offsets(1920, 1080, z, (cx, cy))
    assert 0 <= x <= int(1920 * z) // 2 * 2 - 1920 and 0 <= y <= int(1080 * z) // 2 * 2 - 1080 and x % 2 == 0 and y % 2 == 0
    assert COV.punch_size(None, portrait=True) == 1.10 and COV.punch_size(None) == 1.12 and COV.punch_size(9) == 1.25 and COV.punch_size("x") == 1.12


def test_face_in_reads_the_median_face_of_a_stretch():
    ser = {"t": [0, 1, 2, 3], "face_h": [0.2, 0.2, 0.25, 0.0], "face_cx": [0.3, 0.32, 0.31, 0.0], "face_cy": [0.4, 0.4, 0.42, 0.0]}
    f = COV.face_in(ser, 0.0, 2.5)
    assert f and 0.29 < f["cx"] < 0.33 and f["h"] >= 0.2
    assert COV.face_in({"t": [0], "face_h": [0.2]}, 0, 1) is None and COV.face_in(None, 0, 1) is None   # an older cache: no centres


def test_punch_ins_use_the_face_and_only_the_chosen_joins():
    mk = lambda out, ss: {"v": {"path": "/a.mp4", "ss": ss, "w": 1920, "h": 1080}, "a": {"path": "/a.mp4", "ss": ss}, "d": 4.0, "out": out, "item": {}}   # noqa: E731
    parts = [mk(0.0, 0.0), mk(4.0, 10.0), mk(8.0, 20.0), mk(12.0, 30.0)]
    face = lambda path, t0, t1: {"cx": 0.3, "cy": 0.35, "h": 0.2}   # noqa: E731
    LP.punch_ins(parts, 1.12, faces=face, only_at=[8.0])
    assert [bool(p.get("zoom")) for p in parts] == [False, False, True, False]
    assert parts[2]["zoom_xy"][0] < 0.5 and parts[2]["item"]["zoom_xy"] == parts[2]["zoom_xy"]
    parts2 = [mk(0.0, 0.0), mk(4.0, 10.0), mk(8.0, 20.0), mk(12.0, 30.0)]
    LP.punch_ins(parts2, 1.12)                                                         # today's rule: every second jump cut, on the middle
    assert [bool(p.get("zoom")) for p in parts2] == [False, True, False, True] and "zoom_xy" not in parts2[1]


def test_the_punch_in_filter_renders_the_full_frame_even_with_a_face_at_the_edge(tmp_path):
    src = tmp_path / "s.mp4"
    subprocess.run([FF, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=10:duration=1", "-pix_fmt", "yuv420p", str(src)], check=True)
    for face in ({"cx": 0.02, "cy": 0.1, "h": 0.2}, {"cx": 0.98, "cy": 0.95, "h": 0.2}, None):
        out = tmp_path / "o.mp4"
        vf = COV.zoom_filter(1920, 1080, 1.12, COV.punch_centre(face, 1.12) if face else None)
        subprocess.run([FF, "-y", "-v", "error", "-i", str(src), "-vf", vf, "-pix_fmt", "yuv420p", str(out)], check=True)
        info = ffmpeg.probe(out)
        s = next(x for x in info["streams"] if x["codec_type"] == "video")
        assert (s["width"], s["height"]) == (1920, 1080)


def test_movement_points_find_where_the_speaker_walked():
    ser = {"t": [0, 1, 2, 3, 4], "face_cx": [0.5, 0.5, 0.72, 0.73, 0.73]}
    assert COV.movement_points(ser, 0, 4) == [2]
    assert COV.movement_points({"t": [0, 1], "face_cx": [0.5, 0.52]}, 0, 1) == [] and COV.movement_points(None, 0, 1) == []


# --- max hold -----------------------------------------------------------------------------------------------------------------
def test_max_hold_comes_from_the_style_and_prefers_an_angle_switch():
    assert COV.max_hold(5) == 5.0 and COV.max_hold(30) == 30.0 and COV.max_hold(None) == 12.0 and COV.max_hold("x") == 12.0
    g = {"angles": [{"id": "A", "files": [{"path": "/a", "t0": 0.0, "dur": 100.0}]}, {"id": "B", "files": [{"path": "/b", "t0": 0.0, "dur": 100.0}]}],
         "recorders": [], "dur": 100.0}
    cuts = [{"start": 0.0, "end": 100.0, "angle": "A"}]
    cut, cuts2 = LP.keep_moving([], cuts, [(0.0, 40.0)], _pool(3), None, g, 40.0, limit=COV.max_hold(10), choose=lambda t, free: None)
    assert any(c["angle"] == "B" for c in cuts2) and cut == []                         # a camera change, and no unrelated B-roll forced in
    # no other camera and nothing matching: it stays on the speaker and the report says so
    report = []
    cut, _ = LP.keep_moving([], [], [(0.0, 40.0)], _pool(3), None, None, 40.0, limit=10.0, choose=lambda t, free: None, report=report)
    assert cut == [] and report and "stays on the speaker" in report[0]


# --- multicam switching --------------------------------------------------------------------------------------------------------
def _speech():
    words, t = [], 0.0
    for s in range(14):
        who = "guest" if (s // 5) % 2 == 0 else "host"
        for k in range(5):
            words.append({"text": f"w{k}" + ("." if k == 4 else ""), "start": round(t, 2), "end": round(t + 0.4, 2), "who": who})
            t += 0.5
        t += 0.7 if s % 3 == 0 else 0.2
    return words


def test_switch_points_are_sentence_ends_speaker_changes_and_thought_ends():
    words = _speech()
    pts = SW.switch_points(words, {"A": 0.9, "B": 0.1, "Wide": 0.5}, {"Wide"})
    assert pts and all(p["why"] in ("sentence_end", "speaker_change", "thought_end") for p in pts)
    assert all(p["allowed"] == ["A", "Wide"] for p in pts)                              # B never shows the speaker, the wide always may
    assert not any(SW.inside_a_word(p["t"], words) for p in pts)
    assert any(p["why"] == "thought_end" for p in pts)


def test_switches_only_at_allowed_points_hold_at_least_five_seconds_and_follow_the_speaker():
    words = _speech()
    pts = SW.switch_points(words, {"A": 0.9, "B": 0.1}, None)
    end = words[-1]["end"]
    # a director that ignores every rule: switches every 2 s, at odd times, to an angle that never shows the speaker
    segs = [{"start": float(k * 2), "end": float(k * 2 + 2), "angle": "AB"[k % 2]} for k in range(int(end // 2))]
    out = SW.apply(segs, pts, words, 5.0)
    assert out[0]["start"] == 0.0
    allowed_t = [p["t"] for p in pts]
    for a, b in zip(out, out[1:]):
        assert any(abs(b["start"] - t) < 0.01 for t in allowed_t)                         # a switch only at an allowed point
        assert not SW.inside_a_word(b["start"], words)
    assert all(s["end"] - s["start"] >= 5.0 - 1e-6 for s in out[:-1])                    # held at least five seconds
    assert all(s["angle"] == "A" for s in out)                                           # B never shows the speaker: never used


def test_a_reaction_cut_is_short_and_only_during_a_pause():
    words = [{"text": "one.", "start": 0.0, "end": 0.5}, {"text": "two", "start": 0.6, "end": 1.0}, {"text": "three.", "start": 11.0, "end": 11.5},
             {"text": "four", "start": 11.6, "end": 12.0}]
    energy = [(t / 2, 1.0) for t in range(0, 30)] + [(1.5, 9.0)]
    rc = SW.reaction_cuts(words, sorted(energy), "B", "A")
    assert len(rc) == 1 and 1.5 <= rc[0]["end"] - rc[0]["start"] <= 3.0 and rc[0]["start"] >= 1.0 and rc[0]["angle"] == "B"
    assert SW.reaction_cuts(words, [], "B", "A") == []                                     # no face samples: no reaction cuts


# --- wiring (render path) ------------------------------------------------------------------------------------------------------
def test_a_long_run_on_one_picture_is_broken_by_a_punch_in_on_a_join_before_any_b_roll():
    dev = _devices(prefer=("punch", "angle", "broll"))
    cut, _ = LP.keep_moving([], [], [(0.0, 20.0), (30.0, 50.0)], _pool(3), None, None, 40.0, limit=8.0, choose=lambda t, free: None, jumps=[30.0],
                            devices=dev)
    assert cut == [] and dev.get("punch_at") == [20.0]                                  # (the join at program second 20; no unrelated B-roll was forced in)


def test_insert_reactions_splits_the_speakers_segment_and_never_leaves_a_sliver():
    cuts = [{"start": 0.0, "end": 30.0, "angle": "A"}, {"start": 30.0, "end": 40.0, "angle": "B"}]
    out = SW.insert_reactions(cuts, [{"start": 10.0, "end": 12.0, "angle": "B", "reaction": True},
                                      {"start": 29.5, "end": 31.5, "angle": "B", "reaction": True},       # runs over a segment edge: left out
                                      {"start": 0.4, "end": 2.0, "angle": "B", "reaction": True}])          # leaves under a second before it: left out
    assert [(c["angle"], c["start"], c["end"]) for c in out] == [("A", 0.0, 10.0), ("B", 10.0, 12.0), ("A", 12.0, 30.0), ("B", 30.0, 40.0)]


def test_angle_ratios_read_the_footage_facts_and_the_wide_is_always_allowed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from clipfarm.clipper import footage as FT
    recs = {"/a": {"faces": {"on_camera_speaker_ratio": 0.9, "largest_ratio_median": 0.5}},
            "/b": {"faces": {"on_camera_speaker_ratio": 0.1, "largest_ratio_median": 0.5}},
            "/w": {"faces": {"on_camera_speaker_ratio": 0.0, "largest_ratio_median": 0.07}}}
    monkeypatch.setattr(FT, "load_cached", lambda d, p, level="fast": recs.get(str(p)))
    g = {"angles": [{"id": k.strip("/").upper(), "files": [{"path": k, "t0": 0.0, "dur": 50.0}]} for k in ("/a", "/b", "/w")]}
    ratios, wide = LP.LocalRunner.angle_ratios(SimpleNamespace(), g, SimpleNamespace(dir=tmp_path))
    assert ratios == {"A": 0.9, "B": 0.1, "W": 0.0} and wide == {"W"}
    pts = SW.switch_points([{"text": "Done.", "start": 0.0, "end": 0.4}, {"text": "Next", "start": 1.5, "end": 1.9}], ratios, wide)
    assert pts and pts[0]["allowed"] == ["A", "W"]                       # B does not show the speaker: not allowed; the wide always is


def test_an_angle_switch_at_a_jump_cut_moves_to_a_nearby_movement():
    g = {"angles": [{"id": "A", "files": [{"path": "/a", "t0": 0.0, "dur": 100.0}]}, {"id": "B", "files": [{"path": "/b", "t0": 0.0, "dur": 100.0}]}],
         "recorders": [], "dur": 100.0}
    dev = _devices(prefer=("angle", "broll", "punch"))
    _, cuts2 = LP.keep_moving([], [{"start": 0.0, "end": 100.0, "angle": "A"}], [(0.0, 50.0)], [], None, g, 50.0, jumps=[20.0], limit=1000,
                              devices=dev, moves=[20.7, 45.0])
    assert any(c["angle"] == "B" and abs(c["start"] - 20.7) < 0.01 for c in cuts2)
