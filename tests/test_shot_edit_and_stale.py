"""B-roll strip (2026-11-06): Replace / Edit / "Don't use this" only change the plan and mark the video out of date; Re-render is the person's.
Edit trims or slides one shot inside its own file without a render."""
import json
from fractions import Fraction
from pathlib import Path

import pytest

from clipfarm.clipper import localproj as LP
from clipfarm.core import ffmpeg
from test_replace_shot import _fcp_broll, _latest, _rgb_at, made  # noqa: F401 - the fixture: a real project, made once


def _plan(made):
    return json.loads((made["folder"] / ".aivideo" / "plan.json").read_text())


def _rc(made):
    return json.loads((made["folder"] / ".aivideo" / "render_cache.json").read_text())


def _view(made):
    return made["projects"].view(made["projects"].get(made["proj"]["id"]))


def _fcp_clip(folder: Path):
    """(source start, duration) in seconds of the first B-roll-lane clip of the newest Final Cut project."""
    import xml.etree.ElementTree as ET
    root = ET.parse(max((folder / "Final Cut").rglob("*.fcpxml"), key=lambda p: p.stat().st_mtime)).getroot()
    fr = lambda t: float(Fraction(str(t or "0s").rstrip("s")))              # noqa: E731
    c = next(c for c in root.iter("asset-clip") if c.get("lane"))
    return fr(c.get("start")), fr(c.get("duration"))


def test_replace_waits_for_rerender_and_discard_goes_back(made, monkeypatch):
    r, pid, folder, tmp, job = made["runner"], made["proj"]["id"], made["folder"], made["tmp"], made["proj"]["job"]
    started = []
    monkeypatch.setattr(r, "start", lambda p, kind, **kw: started.append((p, kind)))             # (nothing may start by itself)
    v1 = _latest(folder)
    rc0 = _rc(made)
    green = r.shot_choices(pid, 0)["choices"][0]["choice"]
    res = r.stage_replace_shot(pid, 0, green)
    assert res["staged"] and res["pending"] == 1 and started == []
    assert made["store"].get(job)["status"] == "done" and _latest(folder) == v1                  # the old video stays, playable
    assert _plan(made)["pending_render"] and len(_plan(made)["pending_changes"]) == 1
    v = _view(made)
    assert v["pending_render"] and v["pending_changes"][0]["kind"] == "replaced"
    sh = r.shots(pid)
    assert sh["pending"] == 1 and sh["shots"][0]["changed"] and "green" in sh["shots"][0]["what"]
    r.resume()                                                                                    # (a restart does not render it by itself)
    assert started == []
    # a second change on the same shot: still one waiting change for it
    r.stage_replace_shot(pid, 0, "speaker")
    assert [c["kind"] for c in _plan(made)["pending_changes"]] == ["removed"] and r.shots(pid)["shots"] == []
    # "Discard changes": back to the last render's B-roll
    assert r.discard_pending(pid)["pending"] == 0
    assert _rc(made)["final"]["cutaways"] == rc0["final"]["cutaways"]
    assert not _plan(made).get("pending_render") and not _plan(made).get("pending_changes")
    with pytest.raises(ValueError, match="no waiting changes"):
        r.discard_pending(pid)
    # stage again and Re-render ONCE: the render has the change and clears the flag
    monkeypatch.undo()
    r.stage_replace_shot(pid, 0, green)
    r.rerender(pid)
    v2 = _latest(folder)
    assert v2 != v1
    g = _rgb_at(v2, 5.0, tmp)
    assert g[1] > 120 and g[0] < 60                                                              # green trees now
    assert not _plan(made).get("pending_render") and not _plan(made).get("pending_changes") and _fcp_broll(folder) == ["green trees"]
    assert made["store"].get(job)["status"] == "done"


def test_trim_and_slide_a_shot_without_rendering(made, monkeypatch):
    r, pid, folder, tmp, job = made["runner"], made["proj"]["id"], made["folder"], made["tmp"], made["proj"]["job"]
    started = []
    monkeypatch.setattr(r, "start", lambda p, kind, **kw: started.append((p, kind)))
    info = r.shot_edit_info(pid, 0)
    assert info["fps"] == 30.0 and 0 <= info["in"] < info["out"] <= info["dur"] and info["max_length"] >= 2.5
    assert info["stills"] and all((made["store"].dir(job) / n).exists() for n in info["stills"])
    assert abs((info["out"] - info["in"]) - 2.5) < 0.1
    v1 = _latest(folder)
    d1 = ffmpeg.duration(v1)                                                                      # (the older version goes to the Trash when v2 is made)
    # cut the first 3 frames off, and 3 frames off the end: starts 0.1 s later, stops 0.1 s earlier, same moment of the video
    a, b = info["in"] + 0.1, info["out"] - 0.1
    res = r.trim_shot(pid, 0, a, b)
    assert not res["clamped"] and abs(res["in"] - a) < 0.04 and abs(res["length"] - (b - a)) < 0.04 and res["pending"] == 1
    assert started == [] and _latest(folder) == v1 and made["store"].get(job)["status"] == "done"
    c = _rc(made)["final"]["cutaways"][0]
    assert c["why"] == "trimmed by you" and abs(c["at"] - info["at"]) < 1e-6 and abs(c["length"] - (b - a)) < 0.04
    sh = r.shots(pid)
    assert sh["shots"][0]["changed"] and abs(sh["shots"][0]["length"] - (b - a)) < 0.04
    # a slide: the same length, three frames earlier; whole frames
    s2 = r.trim_shot(pid, 0, res["in"] - 0.1, res["out"] - 0.1)
    assert abs((s2["out"] - s2["in"]) - res["length"]) < 0.04 and abs(s2["in"] * 30 - round(s2["in"] * 30)) < 1e-6 and s2["pending"] == 1
    # never negative, never past the file, never under half a second, never into the next shot / past the end
    low = r.trim_shot(pid, 0, -5.0, 1.0)
    assert low["in"] == 0.0 and low["out"] - low["in"] >= 0.5
    high = r.trim_shot(pid, 0, info["dur"] - 0.2, info["dur"] + 9.0)
    assert high["out"] <= info["dur"] + 1e-6 and high["out"] - high["in"] >= 0.5 and high["clamped"]
    tiny = r.trim_shot(pid, 0, 1.0, 1.1)
    assert tiny["out"] - tiny["in"] >= 0.5 - 1e-6 and tiny["clamped"]
    long = r.trim_shot(pid, 0, 0.0, 60.0)
    assert long["out"] - long["in"] <= info["max_length"] + 0.04 and long["clamped"]
    with pytest.raises(ValueError, match="start and the end"):
        r.trim_shot(pid, 0, "x", None)
    with pytest.raises(ValueError, match="isn't in the video"):
        r.trim_shot(pid, 7, 1, 2)
    # the one re-render uses the trimmed range: the picture (and the Final Cut clip) follow
    final = r.trim_shot(pid, 0, 1.0, 3.0)
    monkeypatch.undo()
    r.rerender(pid)
    v2 = _latest(folder)
    assert v2 != v1 and abs(ffmpeg.duration(v2) - d1) < 0.1
    st_, du = _fcp_clip(folder)
    assert abs(st_ - final["in"]) < 0.05 and abs(du - final["length"]) < 0.05
    from clipfarm.clipper import fcpxml as FX                                                      # (Final Cut's own format check: xmllint --dtdvalid FCPXML 1.11)
    if Path("/Applications/Motion.app").exists():
        assert FX.validate(max((folder / "Final Cut").rglob("*.fcpxml"), key=lambda q: q.stat().st_mtime)) == []
    assert not _plan(made).get("pending_changes")


def test_the_routes_and_a_busy_project(made, monkeypatch):
    from clipfarm.clipper import studio_web as W
    r, pid = made["runner"], made["proj"]["id"]
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = r
    started = []
    monkeypatch.setattr(r, "start", lambda p, kind, **kw: started.append((p, kind)))
    code, body = api.get(["long", pid, "shots", "0", "edit"])
    assert code == 200 and body["stills"] and body["fps"]
    code, body = api.get(["long", pid, "shots", "9", "edit"])
    assert code == 409 and "isn't in the video" in body["error"]
    code, body = api.post(["long", pid, "shots", "0", "trim"], {"in": body.get("in", 0) if False else 1.0, "out": 3.0})
    assert code == 200 and body["pending"] == 1 and started == []
    code, body = api.post(["long", pid, "pending", "discard"], {})
    assert code == 200 and body["pending"] == 0
    code, body = api.post(["long", pid, "pending", "discard"], {})
    assert code == 409
    r.running = pid                                                                               # (this video is being made: nothing can change under it)
    with pytest.raises(ValueError):
        api.post(["long", pid, "shots", "0"], {"choice": "speaker"})
    code, body = api.post(["long", pid, "shots", "0", "trim"], {"in": 1.0, "out": 3.0})
    assert code == 409
    r.running = None


def test_a_trimmed_shot_is_not_stretched_back_by_the_gap_closing():
    """no_flash_gaps grows shots to close a short gap; a shot trimmed by hand keeps its range and is never dropped."""
    pool = [{"start": 0.0, "end": 10.0, "file": "a"}, {"start": 0.0, "end": 10.0, "file": "b"}]
    auto = [{"at": 2.0, "length": 2.0, "src": 3.0, "scene": 0, "why": "x"}, {"at": 5.0, "length": 2.0, "src": 3.0, "scene": 1, "why": "x"}]
    grown = LP.no_flash_gaps(auto, pool, 20.0)
    assert grown[0]["length"] > 2.0 or grown[1]["length"] > 2.0                                   # (an untouched shot is grown to close the 1 s gap)
    hand = [{**auto[0], "why": "trimmed by you"}, {**auto[1], "why": "trimmed by you"}]
    kept = LP.no_flash_gaps(hand, pool, 20.0)
    assert [(c["at"], c["src"], c["length"]) for c in kept] == [(2.0, 3.0, 2.0), (5.0, 3.0, 2.0)]
