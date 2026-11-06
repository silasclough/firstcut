"""Tester probes for release 0.3.0 (parallel catalogue pass, camera motion, finished_video, long form, dictation). Plain tests pass today;
xfail(strict=True) marks a bug found."""
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from test_catalog import FakeEmbedder, _cfg, _video  # noqa: F401

from clipfarm.clipper import cameramotion as CM
from clipfarm.clipper import catalog as C


# ------------------------------------------------------------------ 1. catalogue
def _drive(tmp_path, n=6):
    root = tmp_path / "Drive" / "Video"
    for k in range(n):
        _video(root / f"clip{k}.mp4", "red", secs=3)
    return root


def test_parallel_index_counts_and_status_progress(tmp_path):
    root = _drive(tmp_path, 7)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    seen = []
    r = cat.index(progress=lambda k, n, w: seen.append((k, n)))
    assert (r["new"], r["failed"], r["left"]) == (7, 0, 0)
    assert all(n == 7 for _, n in seen) and [k for k, _ in seen] == sorted(k for k, _ in seen)
    assert len(list(cat.records())) == 7
    assert (cat.dir / "index.json").exists()


def test_one_unreadable_file_counts_failed_others_done(tmp_path):
    root = _drive(tmp_path, 4)
    (root / "bad.mp4").write_bytes(b"not a video" * 100)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    r = cat.index()
    assert (r["new"], r["failed"], r["left"]) == (4, 1, 0)
    assert "stopped" not in r


def test_picture_model_raising_mid_run_halts_no_deadlock(tmp_path):
    root = _drive(tmp_path, 8)
    calls = {"n": 0}

    class Flaky(FakeEmbedder):
        def images(self, ims):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("model died")
            return super().images(ims)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=Flaky())
    out = {}
    t = threading.Thread(target=lambda: out.update(r=cat.index()), daemon=True)
    t.start(); t.join(60)
    assert not t.is_alive(), "deadlock"
    r = out["r"]
    assert "picture model" in r["stopped"]
    assert r["new"] >= 1 and r["new"] + r["failed"] + r["left"] == 8
    assert r["failed"] == 0                                         # model failure is not the file's fault


def test_drive_going_away_mid_run_halts_cleanly(tmp_path):
    root = _drive(tmp_path, 8)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    real = cat._probe

    def probe(p):
        if p.name == "clip3.mp4":
            shutil.rmtree(root)
        return real(p)
    cat._probe = probe
    r = cat.index()
    assert "drive went away" in r.get("stopped", "")
    assert r["failed"] == 0


def test_deadline_in_the_past_does_nothing_but_still_compiles(tmp_path):
    root = _drive(tmp_path, 3)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    r = cat.index(deadline=time.time() - 1)
    assert r["new"] == 0 and r["left"] == 3


def test_stop_set_mid_run_leaves_rest(tmp_path):
    root = _drive(tmp_path, 8)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    n = {"k": 0}
    real = cat.index_one

    def one(f):
        n["k"] += 1
        if n["k"] == 2:
            cat.stop.set()
        return real(f)
    cat.index_one = one
    r = cat.index()
    assert 2 <= r["new"] < 8 and r["left"] == 8 - r["new"]


def test_exception_escaping_index_one_does_not_lose_the_run(tmp_path):
    root = _drive(tmp_path, 4)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    real = cat.index_one

    def one(f):
        if f["rel"].endswith("clip1.mp4"):
            raise OSError("disk full")                              # e.g. save() failing in the error path
        return real(f)
    cat.index_one = one
    r = cat.index()                                                 # must not raise
    assert r["new"] == 3


def test_workers_config_is_clamped(tmp_path):
    root = _drive(tmp_path, 2)
    for w in (0, -3, 99):
        cat = C.Catalog(_cfg(tmp_path, [root], workers=w), embedder=FakeEmbedder())
        assert cat.index()["new"] in (2, 0)


def test_finished_video_rules():
    P = Path
    yes = ["Final/a.mp4", "Project/Exports/a.mp4", "x/Delivered/a.mp4", "Project/final.mp4", "a/Interview V2.mp4", "a/rough cut.mp4",
           "a/master.mov", "a/talk_final_v3.mp4", "Rendered/x.mp4"]
    no = ["Raw/final.mp4", "Footage/Final cut test.mp4", "B-roll/master shot.mp4", "C0018.mp4", "Interview/clip.mp4",
          "Finalist/clip.mp4", "Finale.mp4"]
    for r in yes:
        assert C.finished_video(P(r)), r
    for r in no:
        assert not C.finished_video(P(r)), r


def test_a_cam_folder_is_camera():
    assert not C.finished_video(Path("A-cam/V2.mp4"))


def test_finished_video_dir_beats_camera_dir():
    assert C.finished_video(Path("Raw/Exports/a.mp4"))


def test_preview_same_range_twice_and_concurrent_makes_one(tmp_path):
    root = _drive(tmp_path, 1)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    i = next(cat.records())["id"]
    res = []
    ts = [threading.Thread(target=lambda: res.append(cat.preview(i, 0, 2))) for _ in range(4)]
    [t.start() for t in ts]; [t.join(60) for t in ts]
    assert len(set(res)) == 1 and res[0] and res[0].exists()
    assert not list((cat.dir / "previews").glob("*.part.mp4"))
    assert cat.preview(i, 99, 100) is not None                      # past the end clamps
    assert cat.preview("nope", 0, 1) is None


# ------------------------------------------------------------------ camera motion
def _heat_with_motion_file(tmp_path, content):
    root = _drive(tmp_path, 1)
    cat = C.Catalog(_cfg(tmp_path, [root]), embedder=FakeEmbedder())
    cat.index()
    rec = next(cat.records())
    p = cat.dir / "motion" / f"{rec['id']}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    if content is None:
        p.unlink(missing_ok=True)
    else:
        p.write_bytes(content)
    rec["motion"] = CM.VERSION
    cat.save(rec)
    return cat, rec["id"]


@pytest.mark.parametrize("content", [b"", b"garbage", b"\x93NUMPY\x01\x00" + b"x" * 20])
def test_heat_survives_corrupt_motion_file(tmp_path, content):
    cat, i = _heat_with_motion_file(tmp_path, content)
    h = cat.heat(i)
    assert h["heat"]


def test_heat_survives_wrong_shape_motion_file(tmp_path):
    import io
    cat, i = _heat_with_motion_file(tmp_path, b"")
    b = io.BytesIO(); np.save(b, np.zeros((10, 3), np.float32))
    (cat.dir / "motion" / f"{i}.npy").write_bytes(b.getvalue())
    assert cat.heat(i)["heat"]
    b = io.BytesIO(); np.save(b, np.zeros(10, np.float32))
    (cat.dir / "motion" / f"{i}.npy").write_bytes(b.getvalue())
    assert cat.heat(i)["heat"]


def test_heat_missing_motion_file_goes_pending_not_error(tmp_path, monkeypatch):
    monkeypatch.setattr(CM, "extract", lambda *a, **k: np.zeros((0, 7), np.float32))
    cat, i = _heat_with_motion_file(tmp_path, None)
    h = cat.heat(i)
    assert h["heat"] and h.get("motion") == "pending"
    for _ in range(50):
        if not cat._motion_jobs:
            break
        time.sleep(0.1)
    assert cat.load(i)["motion"] is None
    assert "motion" not in cat.heat(i)


def test_forget_removes_motion_file(tmp_path):
    cat, i = _heat_with_motion_file(tmp_path, b"")
    (cat.dir / "motion" / f"{i}.npy").write_bytes(b"x")
    cat.forget(i)
    assert not (cat.dir / "motion" / f"{i}.npy").exists()


def test_cameramotion_degenerate_inputs():
    assert CM.analyze([], np.zeros((0, 0, 0), np.uint8)).shape == (0, 7)
    assert CM.analyze([0, 1], np.zeros((2, 10, 10), np.uint8)).shape == (0, 7)
    assert CM.read_frames(b"", b"")[0] == []
    assert CM.read_frames(b"abc", "pts_time:0 pts_time:1 s:192x108")[0] == []
    assert CM.profile(None) is None and CM.profile(np.zeros((2, 7))) is None
    assert CM.best_move(None) is None
    flat = CM.analyze([0, .5, 1, 1.5, 2], np.zeros((5, 48, 64), np.uint8))
    p = CM.profile(flat)
    assert p is not None and np.all(np.isfinite(p["cam"]))
    # duplicate / non-increasing times
    CM.profile(CM.analyze([0, 0, 0, 0], np.random.randint(0, 255, (4, 48, 64), dtype=np.uint8)))


def test_cameramotion_extract_on_audio_only_and_one_keyframe(tmp_path):
    from clipfarm.core import ffmpeg
    ff = ffmpeg.bin_path("ffmpeg")
    import subprocess
    a = tmp_path / "a.m4a"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "sine=d=2", str(a)], check=True)
    assert CM.extract(ff, a).shape == (0, 7)
    one = tmp_path / "one.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=160x90:d=2:r=25", "-c:v", "libx264", "-g", "999", str(one)], check=True)
    assert CM.extract(ff, one).shape == (0, 7)
    assert CM.extract(ff, tmp_path / "missing.mp4").shape == (0, 7)


def test_cameramotion_detects_a_pan():
    rng = np.random.RandomState(1)
    big = (rng.rand(120, 400) * 255).astype(np.uint8)
    import cv2
    big = cv2.GaussianBlur(big, (5, 5), 0)
    frames = np.stack([big[:108, 6 * k:6 * k + 192] for k in range(10)])
    m = CM.analyze([k * .5 for k in range(10)], frames)
    p = CM.profile(m)
    assert p["runs"] and CM.best_move(p, 1.0) is not None


# ------------------------------------------------------------------ 2. long form: transcript editing
def _te_world(punct=False):
    from test_transcript_edit import make_words
    from clipfarm.clipper.localproj import seg
    words = make_words()
    if punct:
        for w in words:
            w["text"] = w["text"] + ",”?!…"
    return words, seg(words), {"title": "t", "gap": 0.75, "items": [{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 3, "b": 4}]}


def test_transcript_edit_fuzz_overlapping_ops_keep_invariants():
    import random
    from clipfarm.clipper import transcript_edit as TE
    from types import SimpleNamespace
    from clipfarm.clipper.localproj import LocalRunner
    for punct in (False, True):
        words, sents, plan0 = _te_world(punct)
        rnd = random.Random(7)
        for _ in range(300):
            plan, ops = plan0, []
            for _ in range(rnd.randint(1, 6)):
                i = rnd.randrange(-3, len(words) + 3); j = rnd.randrange(-3, len(words) + 3)
                plan = (TE.cut if rnd.random() < .5 else TE.add)(plan, sents, words, i, j)
                its = plan["items"]
                for it in its:
                    if it["kind"] == "talk" and "t0" in it:
                        assert it["t0"] <= it["t1"] + 1e-6, it
                cuts = plan.get("te_cuts") or []
                assert all(a < b for a, b in cuts) and cuts == sorted(cuts)
                assert all(cuts[k][1] < cuts[k + 1][0] for k in range(len(cuts) - 1))
            iv = LocalRunner.intervals_of(SimpleNamespace(), plan, sents, words, words)
            assert all(a < b for a, b in iv) and iv == sorted(iv)
            assert all(iv[k][1] <= iv[k + 1][0] + 1e-6 for k in range(len(iv) - 1)), iv
        assert plan0["items"] == [{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 3, "b": 4}]


def test_transcript_edit_cut_then_add_same_words_restores():
    from clipfarm.clipper import transcript_edit as TE
    words, sents, plan = _te_world()
    for i, j in [(0, 0), (3, 9), (0, 11), (18, 29), (5, 5)]:
        c = TE.cut(plan, sents, words, i, j)
        back = TE.add(c, sents, words, i, j)
        assert TE.kept_flags(back, sents, words) == TE.kept_flags(plan, sents, words), (i, j)


def test_transcript_edit_empty_and_degenerate():
    from clipfarm.clipper import transcript_edit as TE
    words, sents, plan = _te_world()
    assert TE.cut(plan, sents, words, 5, 2)["items"] == plan["items"]          # reversed selection
    assert TE.apply_ops(plan, sents, words, [{"op": "cut", "t0": 500, "t1": 600}]) == plan
    assert TE.apply_ops(plan, sents, words, [{"op": "cut", "t0": 9, "t1": 3}]) == plan
    assert TE.snap_range([], 0, 1) is None
    assert TE.cut({"items": []}, [], [], 0, 0) == {"items": []}
    assert TE.paragraph_breaks([], []) == []
    assert TE.diff([], [], [], [])["cut_words"] == 0


def test_avoid_edge_inputs():
    from clipfarm.clipper import avoid as AV

    class St:
        d = {}
        def get(self, n, default=None): return self.d.get(n, default)
        def put(self, n, v): self.d[n] = v
    st = St()
    with pytest.raises(ValueError):
        AV.record(st, {"name": "   "})
    with pytest.raises(ValueError):
        AV.record(st, {"name": "x", "start": "abc"})            # (the route must turn this into a 4xx, see below)
    AV.record(st, {"name": "Library - \u00e9\U0001f3ac/\"q\".mp4", "start": 1, "end": 4, "what": "x" * 5000, "id": "i" * 500})
    got = AV.load(st)[0]
    assert len(got["what"]) == 200 and len(got["id"]) == 64
    assert AV.load(type("S", (), {"get": lambda s, n, d=None: "garbage"})()) == []
    assert AV.load(type("S", (), {"get": lambda s, n, d=None: {"shots": [1, "x", None, {"name": ""}]}})()) == []
    # swap with nothing to swap to removes the shot rather than raising
    cut, did = AV.swap([{"scene": 0, "length": 3}], [{"start": 0, "end": 5, "file": "f"}], {0})
    assert cut == [] and did == {"swapped": 0, "removed": 1}
    cut, did = AV.swap([{"scene": None, "length": 3}], [], {0})
    assert did == {"swapped": 0, "removed": 0}


def test_notes_rows_roundtrip_and_odd_characters():
    from clipfarm.clipper import notes as NT
    issues = [{"level": "fix", "start": 12.0, "end": 15.0, "what": "pop \u00b7 at \"quote\" & <b>tag</b> \U0001f600 \u65e5\u672c", "do": "cut <it>"},
              {"level": "warn", "start": None, "what": "whole thing", "do": ""},
              {"level": "weird", "start": 3, "what": "unknown level"},
              {"level": "info", "start": 7, "what": "", "do": ""}]
    rows = NT.issue_rows(issues)
    assert [r["level"] for r in rows] == ["Fix", "Look", "Info", "Info"]
    md = NT.quality_md(issues, {}, "T\u00e4tle {x} %s")
    back = NT.rows_from_md(md)
    assert [r["level"] for r in back] == [r["level"] for r in rows]
    assert back[0]["what"] == rows[0]["what"] and back[0]["do"] == "cut <it>"
    assert NT.rows_from_md(None) == [] and NT.rows_from_md("\x00\ufffd garbage\n- **Fix** broken") == []


def test_notes_multiline_what_survives_roundtrip():
    from clipfarm.clipper import notes as NT
    md = NT.quality_md([{"level": "fix", "start": 1, "what": "line one\nline two", "do": "a\nb"}], {})
    back = NT.rows_from_md(md)
    assert back and "line two" in (back[0]["what"] + back[0]["do"])


def test_notes_missing_keys_do_not_crash():
    from clipfarm.clipper import notes as NT
    NT.issue_rows([{}])


# ------------------------------------------------------------------ fcpxml titles and fit_label
TITLE_TEXTS = ['He said "wait" & <left> \'x\'', "A" * 400, "Hi", "x", "Why \U0001f600 emoji \U0001f468‍\U0001f469‍\U0001f467 日本語", "line1\nline2\ttab",
               "Supercalifragilisticexpialidocious" * 4, "]]> <![CDATA[ & &amp; &#0;", "  padded  "]


@pytest.mark.parametrize("frame", [(1920, 1080), (1080, 1920), (1080, 1080)])
def test_fcpxml_titles_valid_and_on_the_frame_for_odd_text(frame, tmp_path):
    from fractions import Fraction
    import re
    from clipfarm.clipper import fcpxml as FX
    W, H = frame
    items = [{"kind": "clip", "out": 0, "dur": 30.0, "path": "/Volumes/X/Shoot/C0001.MP4", "start": 0.0, "w": 3840, "h": 2160,
              "fps": 23.976023976023978, "audio": True, "file_dur": 600.0, "tc": 0.0}]
    for kind in ("hook", "point", "end"):
        titles = [{"at": 1.0 + k * 3, "len": 2.5, "text": t, "kind": kind, "theme": "glass"} for k, t in enumerate(TITLE_TEXTS)]
        xml = FX.build_multicam("Tal\"k & <co>", W, H, Fraction(24000, 1001), None, items, titles=titles)
        p = tmp_path / f"{kind}{W}x{H}.fcpxml"
        p.write_text(xml)
        if Path("/Applications/Motion.app").exists():
            assert FX.validate(p) == [], (kind, FX.validate(p))
        for m in re.finditer(r'<param name="Position"[^>]*value="(-?[\d.]+) (-?[\d.]+)"', xml):
            x, y = float(m.group(1)), float(m.group(2))
            if kind == "end" or W == 1920:
                assert abs(x) < W and abs(y) < H, (kind, x, y)
        assert xml.count("<title ") == len(TITLE_TEXTS)


def _quality_api(notes_info):
    from types import SimpleNamespace as NS
    from clipfarm.clipper import studio_web as SW
    ns = NS(runner=object(), projects=NS(get=lambda pid: {"notes_info": notes_info}))
    return NS(get=lambda rest: SW.StudioAPI.get(ns, rest))


def test_quality_route_report_with_invalid_utf8(tmp_path):
    md = tmp_path / "q.md"
    md.write_bytes(b"- **Fix** \xb7 00:12 \xb7 bad \xff\xfe bytes\n")
    code, body = _quality_api({"report": str(md)}).get(["long", "p", "quality"])
    assert code == 200


# ------------------------------------------------------------------ keep-cut naming
def test_name_it_edges():
    from types import SimpleNamespace
    from clipfarm.clipper import flow as FL
    sent = lambda t: SimpleNamespace(text=t)  # noqa: E731
    long_words = [sent("word " * 10)]

    def ai(data=None, exc=None):
        def cj(*a, **k):
            if exc:
                raise exc
            return SimpleNamespace(data=data)
        return SimpleNamespace(complete_json=cj)
    assert FL.name_it(ai({"title": "x"}), [], "") == ""                                     # nothing said
    assert FL.name_it(ai({"title": "x"}), [sent("too few words here")], "") == ""
    assert FL.name_it(ai(exc=RuntimeError("down")), long_words) == ""                        # AI unreachable
    assert FL.name_it(ai({"title": None}), long_words) == ""
    assert FL.name_it(ai({}), long_words) == ""
    assert FL.name_it(ai({"title": '  "Quoted"\n  Title\t— \U0001f600 '}), long_words) == "Quoted Title — \U0001f600"
    assert len(FL.name_it(ai({"title": "t" * 500}), long_words)) == 90
    assert FL.name_it(ai({"title": 12345}), long_words) == "12345"
    assert FL.name_it(ai({"title": "a/b: c"}), long_words) == "a/b: c"                      # (deliver() runs it through safe() before it is a file name)
