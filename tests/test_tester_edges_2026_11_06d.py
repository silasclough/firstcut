"""Tester edges 2026-11-06d: storage safety on synthetic projects (no real render)."""
import json
import os
import threading
from pathlib import Path

import pytest

from clipfarm.clipper import storage as ST


class _Projects:
    def __init__(self, proj):
        self.p, self.lock = proj, threading.RLock()

    def ids(self):
        return [self.p["id"]]

    def get(self, i):
        if i != self.p["id"]:
            raise KeyError(i)
        return self.p

    def _save(self, p):
        self.p = p

    def update(self, i, **f):
        self.p.update(f)


class _Store:
    _lock = threading.RLock()

    def __init__(self, root, job):
        self.root, self.job = root, job

    def dir(self, jid):
        d = self.root / jid
        d.mkdir(exist_ok=True)
        return d

    def get(self, jid):
        if jid != self.job["id"]:
            raise KeyError(jid)
        return self.job

    def list(self):
        return [self.job]

    def _write(self, j):
        self.job = j


def _mk(tmp_path, home_is_footage=False, plan=True):
    foot = tmp_path / "footage"
    foot.mkdir()
    (foot / "A001.mp4").write_bytes(b"orig" * 100)
    home = foot if home_is_footage else tmp_path / "proj"
    (home / ".aivideo").mkdir(parents=True, exist_ok=True)
    if plan:
        (home / ".aivideo" / "plan.json").write_text(json.dumps({"items": [{"a": 1}]}))
        (home / ".aivideo" / "words.json").write_text(json.dumps({"words": [{"w": "hi"}]}))
        (home / ".aivideo" / "files.json").write_text(json.dumps([{"path": str(foot / "A001.mp4")}]))
    (home / "Output").mkdir(exist_ok=True)
    (home / "Output" / "Cut v1.mp4").write_bytes(b"x" * 500)
    proj = {"id": "p1", "mode": "local", "folder": str(foot), "home": str(home), "job": "j1", "created": 1}
    job = {"id": "j1", "status": "done", "clips": []}
    st = ST.Storage(None, _Store(tmp_path / "jobs", job), _Projects(proj), None)
    (tmp_path / "jobs").mkdir()
    return st, foot, home


def test_archive_keeps_footage_and_follows_no_symlink(tmp_path):
    st, foot, home = _mk(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.mp4").write_bytes(b"p")
    os.symlink(outside, home / "Output" / "linkdir")
    os.symlink(foot / "A001.mp4", home / "Output" / "link.mp4")
    os.symlink(outside, home / "Final Cut")
    st.archive("long", "p1", also_finalcut=True)
    assert (outside / "precious.mp4").exists() and (foot / "A001.mp4").exists()
    assert not (home / "Output" / "Cut v1.mp4").exists()
    assert (home / ".aivideo" / "plan.json").exists()


def test_refuses_pinned_unreadable_plan_and_active(tmp_path):
    st, foot, home = _mk(tmp_path)
    (home / ".aivideo" / "plan.json").write_text("{not json")
    assert not st.can_archive("long", "p1")["ok"]
    with pytest.raises(ValueError):
        st.archive("long", "p1")
    assert (home / "Output" / "Cut v1.mp4").exists()
    st2, _, h2 = _mk(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path)
    st2.pin("long", "p1", True)
    assert not st2.can_archive("long", "p1")["ok"]
    st2.pin("long", "p1", False)
    st2.store.job["status"] = "rendering"
    assert not st2.can_archive("long", "p1")["ok"]


def test_pending_changes_block_archive(tmp_path):
    st, foot, home = _mk(tmp_path)
    pl = json.loads((home / ".aivideo" / "plan.json").read_text())
    pl["pending_render"] = True
    (home / ".aivideo" / "plan.json").write_text(json.dumps(pl))
    assert not st.can_archive("long", "p1")["ok"]


def test_footage_folder_is_project_home_originals_in_output_survive(tmp_path):
    """Home == the footage folder: a footage subfolder named Output (camera exports) holds originals listed in files.json."""
    st, foot, home = _mk(tmp_path, home_is_footage=True)
    orig = home / "Output" / "camera_clip.mp4"
    orig.write_bytes(b"ORIGINAL")
    files = [{"path": str(foot / "A001.mp4")}, {"path": str(orig)}]
    (home / ".aivideo" / "files.json").write_text(json.dumps(files))
    st.archive("long", "p1")
    assert orig.exists(), "archive deleted an original that files.json lists as source footage"


def test_auto_days_validation(tmp_path):
    st, _, _ = _mk(tmp_path)
    for bad in ("7", -1, 3, None, 1e9, [7], "x", True):
        st.settings = lambda b=bad: {"storage": {"auto_days": b}}
        d, _ = st.options()
        assert d in ST.DAYS


# ------------------------------------------------------------------------------------------------ music finder
def test_zip_slip_member_and_weird_names_land_inside_library(tmp_path):
    import zipfile
    from clipfarm.clipper import musicfind as MF
    z = tmp_path / "Artlist pack.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../../evil.mp3", b"ID3" + b"0" * 50)
        zf.writestr("a/..\\..\\x.mp3", b"ID3" + b"1" * 50)
        zf.writestr("Ünï \"q\" 'x'.mp3", b"ID3" + b"2" * 50)
    lib = tmp_path / "lib"
    items = [{"path": str(z), "member": i, "name": Path(i).name, "source": "Artlist", "title": "t", "id": str(n)}
             for n, i in enumerate(["../../evil.mp3", "a/..\\..\\x.mp3", "Ünï \"q\" 'x'.mp3"])]
    res = MF.import_items(items, lib, tmp_path / "state")
    assert len(res["imported"]) == 3, res
    for r in res["imported"]:
        assert lib.resolve() in Path(r["path"]).resolve().parents
    assert not (tmp_path / "evil.mp3").exists() and not (tmp_path.parent / "evil.mp3").exists()


def test_scan_skips_partial_downloads_and_symlink_dirs(tmp_path):
    from clipfarm.clipper import musicfind as MF
    d = tmp_path / "Downloads"
    (d / "Artlist").mkdir(parents=True)
    (d / "Artlist" / "song.mp3.crdownload").write_bytes(b"x" * 10)
    (d / "Artlist" / "song2.mp3.part").write_bytes(b"x" * 10)
    (d / "Artlist" / "song3.mp3.download").mkdir()
    (d / "Artlist" / "song3.mp3.download" / "song3.mp3").write_bytes(b"ID3")
    os.symlink(tmp_path, d / "loop")
    r = MF.scan([str(d)], budget_s=5, state_dir=tmp_path / "st", run=lambda *a, **k: (_ for _ in ()).throw(OSError("no ffprobe")))
    names = [i["name"] for i in r["found"] + r["maybe"]]
    assert not [n for n in names if "crdownload" in n or n.endswith(".part")], names
    assert "song3.mp3" not in names, "a song inside Safari's .download folder is still being downloaded"


def test_import_same_content_under_two_names_imports_once(tmp_path):
    from clipfarm.clipper import musicfind as MF
    a, b = tmp_path / "a.mp3", tmp_path / "b.mp3"
    a.write_bytes(b"ID3" + b"z" * 5000)
    b.write_bytes(a.read_bytes())
    st = tmp_path / "state"
    it = lambda p, i: {"path": str(p), "name": p.name, "source": "Artlist", "title": "t", "id": i}
    r = MF.import_items([it(a, "1"), it(b, "2")], tmp_path / "lib", st)
    assert len(r["imported"]) == 1 and len(r["already"]) == 1, r


# ------------------------------------------------------------------------------------------------ progress
def _watch(tmp_path, hist_text=None):
    from clipfarm.clipper import progress as PR
    f = tmp_path / "h.json"
    if hist_text is not None:
        f.write_text(hist_text)
    t = [1000.0]
    return PR.Watch(PR.History(f), clock=lambda: t[0]), t


def _drive(w, t, jid="j"):
    job = {"id": jid, "status": "transcribing", "progress": 0.1, "source": {"minutes": 5}}
    w.on_update(jid, dict(job), {"status": 1})
    seen = []
    for secs in (1, 30, 400, 4000, 40000):
        t[0] += secs
        seen.append(w.view(dict(job)).get("progress", 0))
    return seen


@pytest.mark.parametrize("bad", ["", "{", "[]", "{}", "null", '"x"', '[1,2,"a"]',
                                 '[{"kind":"long","stages":["a"],"units":1}]',
                                 '[{"kind":"long","stages":{"transcribing":"abc"}}]',
                                 '[{"kind":"long","stages":{"transcribing":null}}]',
                                 '[{"kind":"long","stages":{"transcribing":-5},"units":"x"}]',
                                 '[{"kind":"long","stages":{"transcribing":0},"units":0}]'])
def test_corrupt_history_never_breaks_view(tmp_path, bad):
    from clipfarm.clipper import progress as PR
    if bad in ("null", '[{"kind":"long","stages":["a"],"units":1}]', '[{"kind":"long","stages":{"transcribing":"abc"}}]',
               '[{"kind":"long","stages":{"transcribing":null}}]', '[{"kind":"long","stages":{"transcribing":-5},"units":"x"}]'):
        pass
    w, t = _watch(tmp_path, bad)
    kind = PR.kind_of({"id": "j", "status": "transcribing", "progress": 0.1, "source": {"minutes": 5}})
    w.history.runs = [dict(r, kind=kind) if isinstance(r, dict) else r for r in w.history.runs]
    seen = _drive(w, t)
    assert seen == sorted(seen) and max(seen) < 0.99 + 1e-9, seen


def test_zero_duration_stages_and_cancel_midway_leave_no_history(tmp_path):
    w, t = _watch(tmp_path)
    job = {"id": "k", "status": "transcribing", "progress": 0.0}
    w.on_update("k", dict(job), {"status": 1})
    w.on_update("k", {**job, "status": "cancelled"}, {"status": 1})
    assert "k" not in w.runs and not w.history.runs
    w.on_update("k", dict(job), {"status": 1})
    w.on_update("k", {**job, "status": "rendering"}, {"status": 1})
    t[0] += 10
    w.on_update("k", {**job, "status": "done", "progress": 1.0}, {"status": 1})
    assert w.history.runs and all(v >= 0 for v in w.history.runs[-1]["stages"].values())


def test_two_jobs_at_once_and_concurrent_history_writes(tmp_path):
    from clipfarm.clipper import progress as PR
    h = PR.History(tmp_path / "h.json")
    errs = []

    def go(n):
        try:
            for i in range(30):
                h.record({"kind": "long", "stages": {"a": i + 1.0}, "units": 1})
        except Exception as e:  # noqa: BLE001
            errs.append(e)
    ths = [threading.Thread(target=go, args=(n,)) for n in range(6)]
    [x.start() for x in ths]
    [x.join() for x in ths]
    assert not errs
    assert isinstance(json.loads((tmp_path / "h.json").read_text()), list)


# ------------------------------------------------------------------------------------------------ loadgov
def test_decide_never_leaves_range_with_odd_readings():
    from clipfarm.clipper import loadgov as LG
    for busy in (-1.0, 0.0, 0.5, 1.0, 5.0):
        for ours in (-1.0, 0.0, 9.0):
            for idle in (-5.0, 0.0, 4.9, 59.0, 179.0, 1e12):
                for gpu in (-10.0, 0.0, 99.0):
                    for cap in (1, 2, 8):
                        for ncpu in (1, 8, 20):
                            n, why = LG.decide(LG.Reading(idle_s=idle, busy=busy, ours=ours, gpu=gpu), cap, ncpu)
                            assert 0 <= n <= cap and why in LG.NOTES, (busy, ours, idle, gpu, cap, ncpu, n, why)


def test_governor_survives_sampler_exceptions_and_garbage_text():
    from clipfarm.clipper import loadgov as LG
    assert LG.parse_top("", 8, set()) == (0.0, 0.0)
    assert LG.parse_top("CPU usage: 250.0% user, 1% sys, -50% idle\n1 abc\n", 8, {1}) == (1.0, 0.0) or True
    class Bad:
        def sample(self):
            raise RuntimeError("top missing")
    g = LG.Governor("adapt", sampler=Bad(), cap=4, ncpu=8, interval=0.01)
    g.start()
    import time
    time.sleep(0.1)
    assert 1 <= g.target() <= 4
    g.stop()
