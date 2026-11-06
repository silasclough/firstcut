"""The Replace sheet's search box (2026-11-06: "in the option to replace the shot of a B-roll shot, there should be a box where you can search
for other clips"): this project's B-roll by its words, the footage library by how it looks; a library shot that isn't in the project
joins it, and only that shot changes."""

import json
import subprocess

import pytest
from PIL import Image

from test_replace_shot import FF, _clip, _fcp_broll, _latest, _rgb_at, made  # noqa: F401 - the fixture


class Catalog:
    """A stand-in footage library: a yellow field (video) and a magenta poster (photo) outside the project, plus a file the project has."""

    def __init__(self, tmp, folder):
        lib = tmp / "Library" / "Campus"
        lib.mkdir(parents=True)
        _clip(lib / "yellow field.mov", 6, color="yellow")
        Image.new("RGB", (480, 270), (230, 0, 230)).save(lib / "poster.jpg")
        common = {"w": 480, "h": 270, "fps": 30.0, "audio": False, "tc": 0.0, "tc_df": False}
        self.recs = {
            "lib0yellow": {"id": "lib0yellow", "path": str(lib / "yellow field.mov"), "rel": "Campus/yellow field.mov", "kind": "video", "dur": 6.0,
                           "size": (lib / "yellow field.mov").stat().st_size, "mtime": 1.0, **common,
                           "scenes": [{"start": 0.0, "end": 6.0, "what": "a yellow field of flowers", "quality": 8}]},
            "lib0poster": {"id": "lib0poster", "path": str(lib / "poster.jpg"), "rel": "Campus/poster.jpg", "kind": "photo", "dur": 0.0,
                           "size": 1, "mtime": 1.0, **common, "scenes": [{"start": 0.0, "end": 0.0, "what": "a bright magenta poster", "quality": 8}]},
            "lib0green": {"id": "lib0green", "path": str(folder / "B-roll" / "green trees.mov"), "rel": "Park day/B-roll/green trees.mov",
                          "kind": "video", "dur": 6.0, "size": 1, "mtime": 1.0, **common, "scenes": [{"start": 0, "end": 6, "what": "green trees", "quality": 8}]},
        }
        self.asked = []

    def search(self, q, n=40, per_file=1, **kw):
        self.asked.append(q)
        out = []
        for i, r in self.recs.items():
            out.append({"id": i, "path": r["path"], "rel": r["rel"], "kind": r["kind"], "k": 1, "start": 0.0, "end": 5.0 if r["kind"] == "video" else 0.0,
                        "dur": r["dur"], "score": 0.3, "what": r["scenes"][0]["what"]})
        return out[:n]

    def load(self, i):
        return self.recs.get(i)


@pytest.fixture
def lib(made, monkeypatch):
    cat = Catalog(made["tmp"], made["folder"])
    monkeypatch.setattr(made["runner"], "library", lambda: cat)
    return cat


def test_the_search_finds_this_projects_shots_by_their_words_and_the_library_by_its_pictures(made, lib):
    r, pid = made["runner"], made["proj"]["id"]
    res = r.shot_search(pid, 0, "green trees in the park")["results"]
    mine = [x for x in res if x["where"] == "project"]
    assert mine and "green" in mine[0]["what"] and mine[0]["pic"].startswith(f"/media/{made['proj']['job']}/shot_")
    libr = [x for x in res if x["where"] == "library"]
    assert {x["what"] for x in libr} == {"a yellow field of flowers", "a bright magenta poster"}      # (the project's own file isn't offered twice)
    y = next(x for x in libr if "yellow" in x["what"])
    assert y["choice"] == "lib:lib0yellow:0.00:5.00" and y["folder"] == "Campus" and y["pic"] == "/api/library/thumb/lib0yellow/1"
    assert y["length"] == 2.5 and lib.asked == ["green trees in the park"]
    assert r.shot_search(pid, 0, "   ")["results"] == []
    with pytest.raises(ValueError, match="isn't in the video"):
        r.shot_search(pid, 5, "trees")


def test_a_library_shot_joins_the_project_and_only_that_shot_changes(made, lib):
    r, pid, folder, tmp = made["runner"], made["proj"]["id"], made["folder"], made["tmp"]
    talk_before = _rgb_at(_latest(folder), 9.0, tmp)                                  # (a re-render writes over the same file)
    rc0 = json.loads((folder / ".aivideo" / "render_cache.json").read_text())
    plan0 = json.loads((folder / ".aivideo" / "plan.json").read_text())
    r.check_replace(pid, 0, "lib:lib0yellow:0.00:5.00")                                  # fine: says nothing
    r.replace_shot(pid, 0, "lib:lib0yellow:0.00:5.00")
    assert made["store"].get(made["proj"]["job"])["status"] == "done"
    v2 = _latest(folder)
    y = _rgb_at(v2, 5.0, tmp)
    assert y[0] > 180 and y[1] > 180 and y[2] < 90, y                                  # the yellow field is on screen where the wall was
    assert abs(talk_before[0] - _rgb_at(v2, 9.0, tmp)[0]) < 12                          # the talking around it is untouched
    files = json.loads((folder / ".aivideo" / "files.json").read_text())
    f = next(x for x in files if x["name"] == "Library - Campus/yellow field.mov")
    assert f["library"] and f["broll"] and f["start"] >= 2_000_000.0
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["pool"][:-1] == plan0["pool"] and plan["pool"][-1]["file"] == f["name"]   # appended: every other shot keeps its number
    rc = json.loads((folder / ".aivideo" / "render_cache.json").read_text())
    assert rc["sig"] == rc0["sig"] and rc["planned"] == rc0["planned"] and rc["pinned"] and rc["pool_n"] == len(plan0["pool"])
    assert [round(c["at"], 1) for c in rc["final"]["cutaways"]] == [round(c["at"], 1) for c in rc0["final"]["cutaways"]]
    assert any("yellow field" in n for n in _fcp_broll(folder))                          # Final Cut gets the library clip
    r.rerender(pid)                                                                     # another render keeps it (the pool only grew)
    assert _rgb_at(_latest(folder), 5.0, tmp)[2] < 90
    assert json.loads((folder / ".aivideo" / "render_cache.json").read_text())["final"]["cutaways"][0]["file"] == f["name"]
    # the same library shot again: no second copy in the pool
    r.replace_shot(pid, 0, "lib:lib0yellow:0.00:5.00")
    assert len(json.loads((folder / ".aivideo" / "plan.json").read_text())["pool"]) == len(plan["pool"])


def test_a_library_photo_and_refusals(made, lib):
    r, pid, folder, tmp = made["runner"], made["proj"]["id"], made["folder"], made["tmp"]
    r.replace_shot(pid, 0, "lib:lib0poster:0.00:0.00")
    m = _rgb_at(_latest(folder), 5.0, tmp)
    assert m[0] > 150 and m[2] > 150 and m[1] < 80, m                                  # the magenta poster, as an animated photo
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["pool"][-1]["photo"].endswith("poster.jpg") and plan["pool"][-1]["start"] >= 2_500_000.0 or plan["pool"][-1]["photo"]
    with pytest.raises(ValueError, match="pick one"):
        r.check_replace(pid, 0, "lib:not valid")
    with pytest.raises(ValueError, match="isn't there any more"):
        r.check_replace(pid, 0, "lib:nosuchid:0.00:2.00")
    (tmp / "Library" / "Campus" / "yellow field.mov").unlink()
    with pytest.raises(ValueError, match="isn't there any more"):
        r.check_replace(pid, 0, "lib:lib0yellow:0.00:5.00")


def test_the_search_route(made, lib):
    from clipfarm.clipper import studio_web as W
    api = W.StudioAPI.__new__(W.StudioAPI)
    api.runner = made["runner"]
    out = api.shot_search(made["proj"]["id"], 0, "yellow")
    assert any(x["where"] == "library" for x in out["results"])
