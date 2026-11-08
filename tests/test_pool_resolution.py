"""WP1 (docs/plans/editor-improvement-plan.md): a B-roll pool scene is placed by its FILE seconds, not by a virtual second that can drift.

The defect (2026-11-07): "C0020.MP4 at 3.50 s for 2.60 s (the file is 3.503 s)" - a scene whose virtual start landed at the end of a 3.5 s file
after the shared timeline moved under it. Every pool scene now carries sid / file / fstart / fend; resolve_pool works the virtual times out
again from the files as they are now. Pure tests: no ffmpeg, no AI.
"""

from __future__ import annotations

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import notes as NT


def _file(name: str, start: float, dur: float, **kw) -> dict:
    return {"name": name, "path": f"/x/{name}", "start": start, "end": start + dur, "dur": dur, "w": 1920, "h": 1080, "fps": 25.0,
            "audio": False, "broll": True, **kw}


def _scene(file: str, fstart: float, fend: float, start: float, end: float, sid: str | None = None, **kw) -> dict:
    return {"file": file, "fstart": fstart, "fend": fend, "start": start, "end": end, "what": "a shot", "quality": 7.0, **({"sid": sid} if sid else {}), **kw}


def test_a_pool_scene_is_placed_by_its_file_seconds_even_when_the_timeline_moves():
    a, b = _file("A.MP4", 0.0, 100.0), _file("B.MP4", 100.0, 50.0)
    pool = [_scene("B.MP4", 1.0, 3.0, 101.0, 103.0, "s0000")]
    got, notes = LP.resolve_pool(pool, [a, b])
    assert got[0]["start"] == 101.0 and not notes
    a2, b2 = _file("A.MP4", 0.0, 110.0), _file("B.MP4", 110.0, 50.0)         # file A got 10 s longer: everything after it moved
    got, notes = LP.resolve_pool(pool, [a2, b2])
    assert got[0]["start"] == b2["start"] + 1.0 == 111.0 and got[0]["end"] == 113.0 and not notes
    assert got[0]["sid"] == "s0000" and got[0]["file"] == "B.MP4"            # (the scene is still the same scene)


def test_a_scene_past_the_end_of_its_file_is_dropped_with_a_note():
    f = _file("C0020.MP4", 100.0, 3.503)
    pool = [_scene("C0020.MP4", 3.5, 6.1, 103.5, 106.1, "s0041"), _scene("C0020.MP4", 0.0, 3.4, 100.0, 103.4, "s0042")]
    got, notes = LP.resolve_pool(pool, [f])
    assert [s["sid"] for s in got] == ["s0042"]
    assert got[0]["end"] <= f["end"] - 0.05 + 1e-9                           # the one that fits is held back from the very last frame
    assert len(notes) == 1 and "C0020.MP4" in notes[0] and "s0041" in notes[0]
    assert LP.check_pool_inside_files(got, [f]) == []


def test_a_scene_from_a_missing_file_is_dropped_not_rendered():
    f = _file("A.MP4", 0.0, 30.0)
    pool = [_scene("GONE.MP4", 0.0, 4.0, 0.0, 4.0, "s0000"), _scene("A.MP4", 2.0, 5.0, 2.0, 5.0, "s0001")]
    got, notes = LP.resolve_pool(pool, [f])
    assert [s["sid"] for s in got] == ["s0001"]
    assert len(notes) == 1 and "GONE.MP4" in notes[0] and "left out" in notes[0]
    assert LP.check_pool_inside_files(pool, [f])                              # the check names the stray scene too


def test_photos_pass_through_and_every_scene_gets_a_stable_sid():
    f = _file("A.MP4", 0.0, 30.0)
    photo = {"file": "p.jpg", "photo": "/x/p.jpg", "start": 5e6, "end": 5e6 + 2.5, "what": "(still photo) a street", "quality": 6.0}
    pool = LP.stamp_pool([{"file": "A.MP4", "start": 4.0, "end": 8.0, "what": "x", "quality": 7.0}, photo], [f])
    assert [s["sid"] for s in pool] == ["s0000", "s0001"]
    assert pool[0]["fstart"] == 4.0 and pool[0]["fend"] == 8.0 and pool[1]["fstart"] == 0.0 and pool[1]["fend"] == 2.5
    again = LP.stamp_pool(pool + [{"file": "A.MP4", "start": 9.0, "end": 12.0, "what": "y", "quality": 7.0}], [f])
    assert [s["sid"] for s in again] == ["s0000", "s0001", "s0002"]          # what has an id keeps it; a new scene is numbered on
    got, notes = LP.resolve_pool(again, [f])
    assert len(got) == 3 and not notes and got[1]["photo"] == "/x/p.jpg"


def test_an_older_plan_without_file_seconds_is_stamped_from_where_its_scenes_lie():
    f = _file("C0020.MP4", 100.0, 3.503)
    old = [{"file": "C0020.MP4", "start": 103.5, "end": 106.1, "what": "late scene", "quality": 7.0},
           {"file": "C0020.MP4", "start": 100.0, "end": 103.0, "what": "good scene", "quality": 7.0},
           {"file": "C0020.MP4", "start": 100.0, "end": 103.0, "what": "dull scene", "quality": 2.0}]
    pool, notes = LP.usable_pool(old, [f])
    assert [s["what"] for s in pool] == ["good scene"]                       # the dull one is below the quality bar, the late one is outside the file
    assert len(notes) == 1 and "C0020.MP4" in notes[0]
    assert pool[0]["sid"] == "s0001"                                         # ids are given on the whole pool, so they do not move with the filter


def test_a_saved_placement_is_matched_by_sid_when_the_pool_is_reordered():
    pool = [{"sid": "s0003", "start": 0, "end": 4}, {"sid": "s0001", "start": 0, "end": 4}]
    placed = [{"at": 2.0, "length": 2.0, "scene": 0, "sid": "s0001"}, {"at": 9.0, "length": 2.0, "scene": 1, "sid": "s0009"},
              {"at": 20.0, "length": 2.0, "scene": 1}]
    got = LP.rebind_by_sid(placed, pool)
    assert [(c["at"], c["scene"]) for c in got] == [(2.0, 1), (20.0, 1)]       # s0001 moved to number 1; s0009 is gone; one without sid keeps its number


def test_build_parts_never_silently_keeps_the_speaker_for_a_short_clip():
    talk = {"name": "talk.mp4", "path": "/x/talk.mp4", "start": 0.0, "end": 60.0, "dur": 60.0, "w": 1920, "h": 1080, "fps": 25.0, "audio": True, "broll": False}
    short = _file("C0020.MP4", 100.0, 3.503)
    files = [talk, short]
    NT.begin()
    parts = LP.build_parts(files, None, [], [(0.0, 20.0)], [{"at": 2.0, "length": 2.6, "src": 100.5, "scene": 0}])      # 3.0 s left of picture, 2.6 asked: fits
    assert [round(p["d"], 2) for p in parts if p["v"].get("cut")] == [2.6]
    NT.take()
    NT.begin()
    parts = LP.build_parts(files, None, [], [(0.0, 20.0)], [{"at": 2.0, "length": 4.0, "src": 100.5, "scene": 0}])      # 4 s asked, 2.95 s left: shortened, not skipped
    cut = [p for p in parts if p["v"].get("cut")]
    assert round(sum(p["d"] for p in cut), 2) == 2.95 and all(p["v"]["ss"] + p["d"] <= short["dur"] - 0.05 + 1e-6 for p in cut)
    assert any("shortened" in n["what"] for n in NT.take())
    NT.begin()
    parts = LP.build_parts(files, None, [], [(0.0, 20.0)], [{"at": 5.0, "length": 3.0, "src": 103.0, "scene": 0}])      # 0.45 s left: dropped, and said so
    assert not [p for p in parts if p["v"].get("cut")]
    said = NT.take()
    assert any("dropped at 5.0 s" in n["what"] and n["start"] == 5.0 for n in said)
    NT.begin()
    LP.build_parts(files, None, [], [(0.0, 20.0)], [{"at": 8.0, "length": 3.0, "src": 900.0, "scene": 0}])               # no clip at all there: dropped, and said so
    assert any("dropped at 8.0 s" in n["what"] for n in NT.take())
    NT.begin()                                                                                                           # the joined-upload path keeps the old, silent behaviour
    parts = LP.build_parts(files, None, [], [(0.0, 20.0)], [{"at": 5.0, "length": 3.0, "src": 103.0, "scene": 0}], strict=False)
    assert not [p for p in parts if p["v"].get("cut")] and NT.take() == []
