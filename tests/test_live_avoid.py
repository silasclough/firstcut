"""The live view's thumbnails: "Don't use this" on a B-roll shot (avoid.py), and what the engine does with it.

2026-11-06: "I'm making a long form video and I'm seeing the shots come in, and I want to be able to click on the little thumbnails ... I should
be able to click on a B-roll and say don't use it and pick something else." Unit tests for the list, then a real make and real re-renders."""

import json
from pathlib import Path

import pytest
from PIL import Image

from clipfarm.clipper import avoid as AV
from test_replace_shot import _latest, _rgb_at, made  # noqa: F401 - the fixture: a real project, made once


class _St:
    def __init__(self):
        self.d = {}

    def get(self, name, default=None):
        return self.d.get(name, default)

    def put(self, name, data):
        self.d[name] = json.loads(json.dumps(data))


# ---------------------------------------------------------------------------------------------------------------- the list
def test_a_shot_is_remembered_once_and_matched_by_file_and_stretch():
    st = _St()
    AV.record(st, {"name": "Library - Athletics/C5211.MP4", "start": 10, "end": 16, "what": "a sprint"})
    AV.record(st, {"name": "Library - Athletics/C5211.MP4", "start": 11, "end": 15})              # the same shot again
    assert len(AV.load(st)) == 1
    av = AV.load(st)
    assert AV.is_avoided(av, "Library - Athletics/C5211.MP4", 12, 18)                            # overlapping stretch
    assert not AV.is_avoided(av, "Library - Athletics/C5211.MP4", 16.2, 20)                      # another part of the same file
    assert not AV.is_avoided(av, "Library - Athletics/C9999.MP4", 10, 16)                        # another file
    AV.record(st, {"name": "Library - Photo/a.jpg", "start": 0, "end": 0, "photo": True})
    assert AV.is_avoided(AV.load(st), "Library - Photo/a.jpg", 0, 0, True)                       # a photo: by file
    with pytest.raises(ValueError):
        AV.record(st, {"start": 1})
    AV.forget(st, {"name": "Library - Athletics/C5211.MP4", "start": 10, "end": 16})
    assert not AV.is_avoided(AV.load(st), "Library - Athletics/C5211.MP4", 10, 16)


def test_the_library_search_leaves_ruled_out_shots_out():
    st = _St()
    AV.record(st, {"name": "Library - A/b.mp4", "start": 4, "end": 9})
    found = [{"rel": "A/b.mp4", "start": 5, "end": 8, "kind": "video"}, {"rel": "A/b.mp4", "start": 20, "end": 25, "kind": "video"},
             {"rel": "A/c.mp4", "start": 5, "end": 8, "kind": "video"}]
    assert [s["start"] for s in AV.drop_found(AV.load(st), found)] == [20, 5]


def test_an_avoided_cutaway_is_swapped_for_the_best_unused_shot_that_fits_or_taken_out():
    pool = [{"file": "r.mov", "start": 0, "end": 6, "quality": 8, "what": "red"},
            {"file": "g.mov", "start": 10, "end": 16, "quality": 9, "what": "green"},
            {"file": "b.mov", "start": 20, "end": 21, "quality": 9, "what": "blue (1 s: too short)"},
            {"file": "p.jpg", "start": 30, "end": 32, "quality": 7, "what": "a photo", "photo": "/x/p.jpg"}]
    files = [{"name": "r.mov", "start": 0}, {"name": "g.mov", "start": 10}, {"name": "b.mov", "start": 20}]
    cut = [{"at": 4.0, "scene": 0, "length": 2.5, "what": "red"}]
    bad = AV.bad_pool([{"name": "r.mov", "start": 0, "end": 6}], pool, files)
    assert bad == {0}
    out, did = AV.swap(cut, pool, bad)
    assert did == {"swapped": 1, "removed": 0} and out[0]["scene"] == 1 and out[0]["at"] == 4.0 and out[0]["length"] == 2.5     # green: the other video
    bad = AV.bad_pool([{"name": "r.mov", "start": 0, "end": 6}, {"name": "g.mov", "start": 0, "end": 6}], pool, files)
    out, did = AV.swap(cut, pool, bad)
    assert out == [] and did == {"swapped": 0, "removed": 1}                                     # blue is 1 s long, a photo isn't a video: taken out
    assert AV.swap(cut, pool, set()) == (cut, {"swapped": 0, "removed": 0})


# ---------------------------------------------------------------------------------------------------------- a real project
def _setup(made, monkeypatch):  # noqa: F811
    r = made["runner"]
    monkeypatch.setattr(r, "start", lambda pid, kind, **kw: getattr(r, kind)(pid, **kw))         # (no background thread in a test)
    pid = made["proj"]["id"]
    placed = json.loads((made["folder"] / ".aivideo" / "render_cache.json").read_text())["final"]["cutaways"][0]
    return r, pid, {"name": placed["file"], "start": 0, "end": 6, "what": placed["what"]}


def test_ruling_out_the_placed_shot_swaps_it_and_renders_again(made, monkeypatch, tmp_path):  # noqa: F811
    r, pid, shot = _setup(made, monkeypatch)
    assert "red" in shot["what"]
    v1 = _latest(made["folder"])
    res = r.avoid_shot(pid, shot)
    assert res["result"] == "swapped" and "next best" in res["message"]
    assert made["store"].get(made["proj"]["job"])["status"] == "done"
    v2 = _latest(made["folder"])
    assert v2 != v1
    c = _rgb_at(v2, 5.0, tmp_path)
    assert not (c[0] > 180 and c[1] < 60)                                                         # no red wall at 5 s any more
    rc = json.loads((made["folder"] / ".aivideo" / "render_cache.json").read_text())
    assert "red" not in rc["final"]["cutaways"][0]["what"] and len(AV.load(made["LP"].State(made["folder"]))) == 1
    assert r.avoided(pid)[0]["name"] == shot["name"]
    # and the Replace sheet never offers it again
    assert all("red" not in c["what"] for c in r.shot_choices(pid, 0)["choices"])
    assert all("red" not in x["what"] for x in r.shot_search(pid, 0, "red wall")["results"])


def test_a_ruled_out_shot_stays_out_when_the_video_renders_again(made, monkeypatch):  # noqa: F811
    r, pid, shot = _setup(made, monkeypatch)
    st = made["LP"].State(made["folder"])
    AV.record(st, shot)                                                                           # (recorded only: nothing rendered yet)
    r.rerender(pid)
    rc = json.loads((made["folder"] / ".aivideo" / "render_cache.json").read_text())
    assert all("red" not in c["what"] for c in rc["final"]["cutaways"])                           # a saved placement is changed too


def test_with_nothing_to_swap_to_the_shot_is_taken_out(made, monkeypatch):  # noqa: F811
    r, pid, shot = _setup(made, monkeypatch)
    for n in ("green trees.mov", "blue sea.mov"):
        (made["folder"] / "B-roll" / n).unlink()
    for n in ("green trees.mov", "blue sea.mov"):
        AV.record(made["LP"].State(made["folder"]), {"name": f"B-roll/{n}", "start": 0, "end": 6})
    res = r.avoid_shot(pid, shot)
    assert res["result"] in ("removed", "swapped")
    if res["result"] == "removed":
        assert r.shots(pid)["shots"] == [] and "Taken out" in res["message"]


def test_while_the_video_is_still_being_made_it_says_which_it_is(made, monkeypatch):  # noqa: F811
    r, pid, shot = _setup(made, monkeypatch)
    job = made["proj"]["job"]
    r.running = pid
    made["store"].update(job, status="finding", progress=0.3)
    early = r.avoid_shot(pid, {**shot, "start": 1})
    assert early["result"] == "excluded" and "won't be used" in early["message"].lower()
    made["store"].update(job, status="rendering", progress=0.8)
    late = r.avoid_shot(pid, {**shot, "start": 2, "end": 3})
    assert late["result"] == "next" and "next time" in late["message"]
    assert len(r.avoided(pid)) == 1                                                               # (the three overlap: one shot)
    r.running = None
    made["store"].update(job, status="done")


def test_changing_your_mind_and_the_routes(made, monkeypatch):  # noqa: F811
    from clipfarm.clipper import studio_web as W
    r, pid, shot = _setup(made, monkeypatch)
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = r
    code, body = api.post(["long", pid, "avoid"], {"shot": shot})
    assert code == 200 and body["result"] == "swapped"
    code, body = api.get(["long", pid, "avoid"])
    assert code == 200 and [x["name"] for x in body["shots"]] == [shot["name"]]
    code, body = api.post(["long", pid, "avoid"], {"shot": shot, "undo": True})
    assert code == 200 and body["result"] == "undone" and api.get(["long", pid, "avoid"])[1]["shots"] == []
    with pytest.raises(ValueError):
        api.post(["long", pid, "avoid"], {"shot": {}})
    assert api.get(["long", "nope", "avoid"])[0] == 404


# -------------------------------------------------------------------------------------------------- the live view's events
def test_library_shots_reach_the_live_view_with_what_the_player_needs(made, monkeypatch, tmp_path):  # noqa: F811
    """(They never did: the event was sent with kind=..., which is the event's own name, and the error was swallowed.)"""
    LP, r = made["LP"], made["runner"]
    thumb = tmp_path / "t.jpg"
    Image.new("RGB", (64, 36), (10, 20, 30)).save(thumb)
    shot = {"id": "abc123", "path": str(made["folder"] / "talk.mov"), "rel": "Athletics/C5211.MP4", "kind": "video", "k": 2, "start": 1.0, "end": 4.0,
            "dur": 6.0, "step": 2.0, "frames": 3, "what": "a sprint", "query": "running", "quality": 7, "size": 1, "mtime": 1, "w": 320, "h": 240,
            "fps": 30.0, "audio": True}

    class Cat:
        def enabled(self):
            return True

        def shots_for(self, *a, **k):
            return [dict(shot), {**shot, "id": "def456", "rel": "Athletics/Other.MP4", "start": 9.0, "end": 12.0}]

        def thumb(self, i, k):
            return thumb

    r.catalog = Cat()
    sent = []
    monkeypatch.setattr(LP.live, "emit", lambda store, job, kind, **d: sent.append((kind, d)))
    monkeypatch.setattr(LP.live, "later", lambda fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(LP.live, "frame", lambda *a, **k: "f1.jpg")
    r.live_job = made["proj"]["job"]
    st = LP.State(made["folder"])
    AV.record(st, {"name": "Library - Athletics/Other.MP4", "start": 8, "end": 13})              # this one was ruled out earlier
    r.add_library([], [], {}, None, st, "", lambda f, m: None)
    ev = [d for kind, d in sent if kind == "libshot"]
    assert len(ev) == 1 and ev[0]["id"] == "abc123" and ev[0]["start"] == 1.0 and ev[0]["end"] == 4.0
    assert ev[0]["dur"] == 6.0 and ev[0]["step"] == 2.0 and ev[0]["frames"] == 3 and ev[0]["name"] == "Library - Athletics/C5211.MP4"
    assert ev[0]["media"] == "video" and ev[0]["f"] == "f1.jpg"
    assert [x["name"] for x in json.loads(Path(st.dir / "library.json").read_text())["files"]] == ["Library - Athletics/C5211.MP4"]


def test_placed_broll_events_carry_the_file_name_and_stretch(made, monkeypatch):  # noqa: F811
    LP, r = made["LP"], made["runner"]
    sent = []
    monkeypatch.setattr(LP.live, "emit", lambda store, job, kind, **d: sent.append((kind, d)))
    monkeypatch.setattr(LP.live, "later", lambda fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(LP.live, "frame", lambda *a, **k: "f2.jpg")
    r.live_job = made["proj"]["job"]
    files = [{"name": "B-roll/x.mov", "path": str(made["folder"] / "talk.mov"), "start": 100.0}]
    pool = [{"file": "B-roll/x.mov", "start": 102.0, "end": 106.0, "what": "a wall"}]
    r._live_broll([{"scene": 0, "at": 4.0, "length": 2.0}], pool, files)
    ev = [d for kind, d in sent if kind == "broll"][0]
    assert ev["name"] == "B-roll/x.mov" and ev["start"] == 2.0 and ev["end"] == 6.0 and ev["photo"] is False and ev["lib"] is None


def test_the_page_has_the_clickable_thumbnails():
    html = (Path(__file__).resolve().parent.parent / "clipfarm" / "clipper" / "web" / "studio.html").read_text()
    for needle in ('data-lv="${what || "frame"}"', "function liveClick", "function liveAvoid", "Don't use this", "function liveBig",
                   "openPlayer(-1, {s, tile", "/api/app/long/${pid}/avoid", "Won't use"):
        assert needle in html, needle
