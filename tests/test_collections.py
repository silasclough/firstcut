"""Collections: named lists of references into the footage catalogue (shotcollections.py), their routes in the studio API, and what they
send to Final Cut. The catalogue is a real small one (fake picture model), the studio API is built without its background thread."""

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import catalog as C
from clipfarm.clipper import shotcollections as SC
from clipfarm.clipper import studio_web as W

from test_catalog import FakeEmbedder, _cfg, _photo, _video  # noqa: F401  (the neighbour's fakes)

FID = "a" * 40
FID2 = "b" * 40


def item(fid=FID, start=10.0, end=15.0, **kw):
    return {"id": fid, "kind": "video", "start": start, "end": end, "k": 5, "rel": "B-roll/Quad/clip.mp4", "what": "two students laughing", "dur": 60, **kw}


@pytest.fixture
def col(tmp_path):
    return SC.Collections(tmp_path / "data" / "collections.json")


# ------------------------------------------------------------------------------------------------ names
def test_names_are_trimmed_unique_ignoring_case_and_limited(col):
    c = col.create("  Student   interaction ")
    assert c["name"] == "Student interaction"
    with pytest.raises(ValueError, match="already a collection"):
        col.create("student INTERACTION")
    for bad in ("", "   ", None):
        with pytest.raises(ValueError, match="give the collection a name"):
            col.create(bad)
    with pytest.raises(ValueError, match="under 60"):
        col.create("x" * 61)
    assert col.create("x" * 60)["name"] == "x" * 60


def test_rename_keeps_the_items_and_cannot_take_another_name(col):
    a, b = col.create("Quad"), col.create("Library")
    col.add(a["id"], [item()])
    r = col.rename(a["id"], "Campus quad")
    assert r["name"] == "Campus quad" and len(r["items"]) == 1
    assert col.rename(a["id"], "campus QUAD")["name"] == "campus QUAD"          # (its own name again, other letters: fine)
    with pytest.raises(ValueError, match="already a collection"):
        col.rename(b["id"], "campus quad")
    with pytest.raises(KeyError):
        col.rename("c000000", "x")


def test_delete_forgets_the_list_only(col):
    a = col.create("Quad")
    col.add(a["id"], [item()])
    col.delete(a["id"])
    assert col.list() == []
    with pytest.raises(KeyError):
        col.get(a["id"])
    with pytest.raises(KeyError):
        col.delete(a["id"])


# ------------------------------------------------------------------------------------------------ items
def test_the_same_shot_twice_is_one_item_and_not_an_error(col):
    a = col.create("Quad")
    r = col.add(a["id"], [item(), item(start=20, end=26)])
    assert len(r["added"]) == 2 and r["duplicates"] == 0
    r = col.add(a["id"], [item(), item(fid=FID2)])
    assert len(r["added"]) == 1 and r["duplicates"] == 1
    assert len(col.get(a["id"])["items"]) == 3
    assert col.holding([item(), item(fid=FID2), item(fid="c" * 40)]) == {a["id"]: 2}


def test_a_photo_has_no_moment_and_is_kept_once(col):
    a = col.create("Photos")
    r = col.add(a["id"], [{"id": FID, "kind": "photo", "start": 3, "end": 9, "rel": "Photos/lawn.jpg"}, {"id": FID, "kind": "photo"}])
    assert len(r["added"]) == 1 and r["duplicates"] == 1
    assert col.get(a["id"])["items"][0]["start"] == 0 and col.get(a["id"])["items"][0]["end"] == 0


def test_bad_items_are_refused_and_change_nothing(col):
    a = col.create("Quad")
    for bad in ([], None, [{"id": "../../etc", "kind": "video", "start": 0, "end": 3}], [{"id": FID, "kind": "video", "start": 5, "end": 5}],
                [{"id": FID, "kind": "audio", "start": 0, "end": 3}], [{"id": FID, "kind": "video", "start": "x", "end": 3}],
                [item(), {"id": "zz", "kind": "video", "start": 0, "end": 3}], ["not an object"],
                [{"id": FID, "kind": "video", "start": float("nan"), "end": 3}]):
        with pytest.raises(ValueError):
            col.add(a["id"], bad)
    assert col.get(a["id"])["items"] == []


def test_remove_note_and_move(col):
    a = col.create("Quad")
    col.add(a["id"], [item(start=1, end=5), item(start=6, end=9), item(start=10, end=14)])
    col.move(a["id"], item(start=10, end=14), 0)
    assert [i["start"] for i in col.get(a["id"])["items"]] == [10, 1, 6]
    col.set_note(a["id"], item(start=1, end=5), "  nice\nlight ")
    assert next(i for i in col.get(a["id"])["items"] if i["start"] == 1)["note"] == "nice light"
    assert col.remove(a["id"], [item(start=1, end=5), item(start=99, end=100)]) == 1
    assert len(col.get(a["id"])["items"]) == 2
    with pytest.raises(KeyError):
        col.set_note(a["id"], item(start=1, end=5), "gone")


def test_the_list_shows_the_one_used_last_first(tmp_path):
    t = [100.0]
    col = SC.Collections(tmp_path / "c.json", now=lambda: t[0])
    a = col.create("Alpha")
    t[0] = 200.0
    b = col.create("Beta")
    assert [c["name"] for c in col.list()] == ["Beta", "Alpha"]
    t[0] = 300.0
    col.add(a["id"], [item()])
    assert [c["name"] for c in col.list()] == ["Alpha", "Beta"]


# ------------------------------------------------------------------------------------------------ the file
def test_it_survives_a_restart_and_a_damaged_file_starts_empty(tmp_path):
    p = tmp_path / "data" / "collections.json"
    a = SC.Collections(p)
    c = a.create("Quad")
    a.add(c["id"], [item()])
    b = SC.Collections(p)                                                       # a new run of the app
    assert [x["name"] for x in b.list()] == ["Quad"] and len(b.get(c["id"])["items"]) == 1
    assert not list(p.parent.glob("*.part"))                                    # written by rename: no temp file is left
    p.write_text("{not json")
    assert SC.Collections(p).list() == []
    p.write_text(json.dumps({"collections": [{"id": "../x", "name": "evil"}, {"id": "c123456", "name": "ok", "items": [{"id": "zz"}, item()]}]}))
    got = SC.Collections(p).list()
    assert [x["name"] for x in got] == ["ok"] and len(got[0]["items"]) == 1     # (a bad id or item is dropped, the rest is kept)


def test_a_failed_write_leaves_the_old_file_whole(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    a = SC.Collections(p)
    a.create("Quad")
    before = p.read_text()
    monkeypatch.setattr(SC.os, "fsync", lambda fd: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        a.create("Library")
    assert p.read_text() == before


def test_threads_adding_at_once_lose_nothing(col):
    c = col.create("Quad")

    def work(n):
        for k in range(20):
            col.add(c["id"], [item(start=n * 100 + k, end=n * 100 + k + 3)])
    ts = [threading.Thread(target=work, args=(n,)) for n in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(col.get(c["id"])["items"]) == 100
    assert len(SC.Collections(col.path).get(c["id"])["items"]) == 100


# ------------------------------------------------------------------------------------------------ with a real small catalogue
@pytest.fixture
def lib(tmp_path):
    root = tmp_path / "Drive" / "Video"
    _video(root / "B-roll" / "Athletics" / "red_gym.mp4", "red")
    _video(root / "B-roll" / "Student Life" / "blue_sky.mp4", "blue", secs=12)
    _photo(root / "Photos" / "green_lawn.jpg", (0, 255, 0))
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    return cat, root


def _api(tmp_path, cat):
    a = W.StudioAPI.__new__(W.StudioAPI)
    a.root = tmp_path / "VB home"
    a.runner = SimpleNamespace(library=lambda: cat)
    return a


def _shot(cat, rel_end):
    rec = next(r for r in cat.records() if r["rel"].endswith(rel_end))
    kind = rec["kind"]
    return {"id": rec["id"], "kind": kind, "start": 0 if kind == "photo" else 1.0, "end": 0 if kind == "photo" else 5.0, "k": 0,
            "rel": rec["rel"], "what": "a thing", "dur": rec.get("dur") or 0}


def test_the_routes_make_rename_add_view_and_delete(tmp_path, lib):
    cat, _ = lib
    api = _api(tmp_path, cat)
    red, blue, lawn = _shot(cat, "red_gym.mp4"), _shot(cat, "blue_sky.mp4"), _shot(cat, "green_lawn.jpg")
    code, r = api.post(["collections"], {"name": "Student interaction", "items": [red]})            # create and add in one go
    assert code == 200 and r["collection"]["count"] == 1 and len(r["added"]) == 1
    cid = r["collection"]["id"]
    assert api.post(["collections"], {"name": "student interaction"})[0] == 400                   # the name is taken
    assert api.post(["collections"], {"name": "Bad", "items": [{"id": "../x"}]})[0] == 400
    assert [c["name"] for c in api.get(["collections"])[1]["collections"]] == ["Student interaction"]      # (and "Bad" was not left behind)
    code, r = api.post(["collections", cid, "add"], {"items": [blue, lawn, red]})
    assert code == 200 and len(r["added"]) == 2 and r["duplicates"] == 1 and r["collection"]["count"] == 3
    code, v = api.get(["collections", cid])
    assert [s["kind"] for s in v["shots"]] == ["video", "video", "photo"]
    s0 = v["shots"][0]
    assert s0["path"].endswith("red_gym.mp4") and not s0["missing"] and s0["step"] > 0 and s0["frames"] >= 3 and s0["dur"] > 5       # what the player takes
    assert all(not s["missing"] for s in v["shots"]) and v["collection"]["missing"] == 0
    card = api.get(["collections"])[1]["collections"][0]
    assert card["count"] == 3 and len(card["covers"]) == 3 and card["covers"][0]["id"] == red["id"]
    assert cat.thumb(card["covers"][0]["id"], card["covers"][0]["k"]).exists()
    assert api.post(["collections", cid, "rename"], {"name": "Campus life"})[1]["collection"]["name"] == "Campus life"
    assert api.post(["collections", cid, "remove"], {"items": [red]})[1]["removed"] == 1
    assert api.post(["collections", "holding"], {"items": [blue, red]})[1]["holding"] == {cid: 1}
    assert api.post(["collections", cid, "delete"], {})[0] == 200
    assert api.get(["collections"])[1]["collections"] == []
    assert api.get(["collections", cid])[0] == 404 and api.post(["collections", "nope!", "add"], {"items": [red]})[0] == 404
    assert api.post(["collections", "c0000000", "rename"], {"name": "x"})[0] == 404
    assert (Path(cat.records().__next__()["path"]).exists())                                        # the footage itself was never touched


def test_a_file_that_left_the_catalogue_shows_as_missing_and_survives_a_prune(tmp_path, lib):
    cat, root = lib
    api = _api(tmp_path, cat)
    red, blue = _shot(cat, "red_gym.mp4"), _shot(cat, "blue_sky.mp4")
    cid = api.post(["collections"], {"name": "Quad", "items": [red, blue]})[1]["collection"]["id"]
    (root / "B-roll" / "Athletics" / "red_gym.mp4").unlink()
    assert cat.index()["forgotten"] == 1                                                            # the catalogue's prune forgets the file...
    code, v = api.get(["collections", cid])
    assert v["collection"]["count"] == 2 and v["collection"]["missing"] == 1                        # ...the collection keeps the reference
    gone = next(s for s in v["shots"] if s["missing"])
    assert gone["rel"].endswith("red_gym.mp4") and gone["what"] == "a thing" and gone["path"] == ""
    assert [c["id"] for c in api.get(["collections"])[1]["collections"][0]["covers"]] == [blue["id"]]    # (no cover from a missing one)
    assert api.post(["collections", cid, "remove"], {"items": [red]})[1]["removed"] == 1            # and it can be taken out


def test_no_catalogue_at_all_lists_everything_as_missing_instead_of_failing(tmp_path):
    api = _api(tmp_path, None)
    api.runner = SimpleNamespace(library=lambda: None)
    cid = api.post(["collections"], {"name": "Quad", "items": [item()]})[1]["collection"]["id"]
    v = api.get(["collections", cid])[1]
    assert v["library"] is False and v["shots"][0]["missing"] and v["collection"]["covers"] == []


def test_searching_inside_a_collection_only_returns_its_own_shots(tmp_path, lib):
    cat, _ = lib
    api = _api(tmp_path, cat)
    red, blue, lawn = _shot(cat, "red_gym.mp4"), _shot(cat, "blue_sky.mp4"), _shot(cat, "green_lawn.jpg")
    a = api.post(["collections"], {"name": "Mixed", "items": [blue, lawn]})[1]["collection"]["id"]      # (no red in it)
    got = api.post(["collections", a, "view"], {"q": "a blue sky"})[1]["shots"]
    assert got and got[0]["id"] == blue["id"] and all(s["id"] != red["id"] for s in got)
    assert [s["id"] for s in api.post(["collections", a, "view"], {"q": ""})[1]["shots"]] == [blue["id"], lawn["id"]]
    # a word in the description or note counts too (a shot whose file ranks low still shows when its words are typed)
    api.collections().set_note(a, lawn, "graduation lawn")
    assert lawn["id"] in [s["id"] for s in api.post(["collections", a, "view"], {"q": "graduation"})[1]["shots"]]
    cat.embedder = lambda: (_ for _ in ()).throw(RuntimeError("no picture model"))                  # (without the model the words alone filter)
    assert lawn["id"] in [s["id"] for s in api.post(["collections", a, "view"], {"q": "graduation"})[1]["shots"]]


def test_a_file_that_clearly_is_not_what_you_typed_is_left_out(tmp_path, lib, monkeypatch):
    cat, _ = lib
    api = _api(tmp_path, cat)
    blue, lawn = _shot(cat, "blue_sky.mp4"), _shot(cat, "green_lawn.jpg")
    a = api.post(["collections"], {"name": "Mixed", "items": [blue, lawn]})[1]["collection"]["id"]
    real = cat.search
    monkeypatch.setattr(cat, "search", lambda *x, **k: [{**s, "chance": 0.0005 if s["id"] == lawn["id"] else 0.2} for s in real(*x, **k)])
    got = [s["id"] for s in api.post(["collections", a, "view"], {"q": "a blue sky"})[1]["shots"]]
    assert got == [blue["id"]]                                                                       # (the lawn's odds are under the floor)
    api.collections().set_note(a, lawn, "blue banner on the lawn")                                   # ...unless its own words say otherwise
    assert lawn["id"] in [s["id"] for s in api.post(["collections", a, "view"], {"q": "blue"})[1]["shots"]]


def test_final_cut_gets_an_event_named_after_the_collection(tmp_path, lib):
    cat, _ = lib
    api = _api(tmp_path, cat)
    red, blue = _shot(cat, "red_gym.mp4"), _shot(cat, "blue_sky.mp4")
    cid = api.post(["collections"], {"name": 'Student "interaction" & more', "items": [red, blue]})[1]["collection"]["id"]
    v = api.get(["collections", cid])[1]
    shots = [{**cat.load(s["id"]), "start": s["start"], "end": s["end"], "what": s["what"]} for s in v["shots"]]
    xml = cat.fcpxml(shots, v["collection"]["name"])
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml.split("<!DOCTYPE fcpxml>")[1])                  # (well formed, whatever the name holds)
    assert root.find("library/event").get("name") == 'Student "interaction" & more' and xml.count("<asset-clip") == 2
    p = tmp_path / "x.fcpxml"
    p.write_text(xml)
    from clipfarm.clipper import fcpxml as FX
    if Path("/Applications/Motion.app").exists():
        assert FX.validate(p) == []


def test_the_module_name_cannot_shadow_the_standard_library():
    """clipfarm/clipper holds scripts that are run by path (dictate_worker.py): a file called collections.py next to them would be
    imported instead of the standard library's."""
    assert not (Path(SC.__file__).parent / "collections.py").exists()


# ------------------------------------------------------------------------------------------------ the page and the server agree on a shot's key
import shutil      # noqa: E402
import subprocess  # noqa: E402

NODE = shutil.which("node")


@pytest.mark.skipif(not NODE, reason="node is not installed")
def test_the_page_builds_the_same_item_and_key_as_the_server_so_undo_finds_what_was_added():
    """Undo sends back the items the page built, matched against the keys the server answered with: any difference in rounding or in how a
    photo's moment is written would make Undo silently do nothing. The page's own collItem/collKey are run by node."""
    src = (Path(__file__).resolve().parents[1] / "clipfarm/clipper/web/studio.html").read_text()
    a = src.index("const collItem = s =>")
    b = src.index("const COLL_ICON")
    cases = [{"id": FID, "kind": "video", "start": 12.3456, "end": 17.5, "step": 2, "k": 9, "rel": "a/b.mp4", "what": "x", "dur": 60},
             {"id": FID2, "kind": "video", "start": 0, "end": 3.999, "step": 0, "k": 4, "rel": "c.mp4", "dur": 10},
             {"id": FID, "kind": "photo", "start": 5, "end": 6, "k": 0, "rel": "p.jpg"}]
    js = src[a:b] + "\nconst cases = " + json.dumps(cases) + ";\nconsole.log(JSON.stringify(cases.map(c => { const it = collItem(c); return [it, collKey(it)]; })));"
    r = subprocess.run([NODE, "-e", js], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    for (it, key), case in zip(json.loads(r.stdout), cases):
        stored = SC.clean_item(it)                                              # what the server keeps of the page's item
        assert SC.item_key(stored) == key, (it, key)
        assert it["k"] == (case["k"] if case["kind"] == "photo" or not case["step"] else round(case["start"] / case["step"]))
