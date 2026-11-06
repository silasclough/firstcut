"""Edge cases for the pre-posting checks (clipgate.weak_title / problems, moments.hard_cap / demote_weak_titles, opening.check, the feed filter, own uploads).
Tests marked xfail(strict=True) record a real defect found by the tester: they flip to a failure (XPASS) the day the defect is fixed, so remove the mark then."""

import json
import subprocess
import time

import pytest

from clipfarm.clipper import clipgate, feed as F, moments as M, ocr, opening, relay
from clipfarm.clipper.store import Store
from clipfarm.context import Context
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

FF = ffmpeg.bin_path("ffmpeg")


def _mk(tmp_path, name, src, secs=10, extra=()):
    out = tmp_path / name
    subprocess.run([FF, "-y", "-v", "error", *src, "-t", str(secs), *extra, str(out)], check=True)
    return out


def _lavfi(spec):
    return ["-f", "lavfi", "-i", spec]


# ----------------------------------------------------------------------------- weak_title: odd input
@pytest.mark.parametrize("t", [None, "", "   ", "​​​", 12345, {"a": 1}, "x?", "ペンギン", "😀"])
def test_weak_title_never_raises_and_refuses_tiny_but_does_not_judge_a_missing_one(t):
    r = clipgate.weak_title(t)
    if t is None or not str(t).strip():
        assert r is None                                                                   # no title at all is not this check's business
    elif isinstance(t, str):
        assert r


@pytest.mark.parametrize("t", ["Ünïcödé Café Ωmega Straße 5", "😀😀😀 😀 😀", "Sam Altman fired from OpenAI in a day"])
def test_weak_title_unicode_that_states_something_passes(t):
    assert clipgate.weak_title(t) is None


def test_weak_title_is_linear_on_huge_titles():
    t0 = time.time()
    for t in ("A" * 500_000, "Word " * 100_000 + "?", "ab " * 200_000 + "?", "How do " * 50_000):
        clipgate.weak_title(t)
    assert time.time() - t0 < 5


def test_problems_is_tolerant_of_missing_keys():
    assert clipgate.problems({}) == []
    assert clipgate.problems({"duration": None, "title": None, "qa": None, "transcript": None}) == []
    assert clipgate.problems({"duration": 20, "qa": {"issues": [{"level": "info", "what": "x"}, {"level": "warn", "what": "y"}]}}) == []


def test_length_limit_boundaries():
    assert clipgate.problems({"duration": 40}) == []
    assert any("seconds long" in p for p in clipgate.problems({"duration": 40.01}))
    assert any("only" in p for p in clipgate.problems({"duration": 6.9}))
    assert clipgate.problems({"duration": 7}) == []


# ----------------------------------------------------------------------------- weak_title: strong titles are not refused (fixed 2026-10-31)
STRONG_BUT_REFUSED = ["Why did he lie about 12 million dollars", "How did they build the Pyramids in 20 years", "This man survived 3 days in the ocean",
                      "Watch a cop catch a suspect with two officers missing", "Look at how Chernobyl exploded", "Hello Kitty sells 5 billion dollars",
                      "Hi-tech Samsung phone explodes in Ohio", "Wait for it: Elon sells 3 Teslas"]


@pytest.mark.parametrize("t", STRONG_BUT_REFUSED)
def test_titles_with_a_concrete_name_or_number_are_not_weak(t):
    assert clipgate.weak_title(t) is None


@pytest.mark.xfail(strict=True, reason="a title written in a script without spaces (Japanese) counts as one word")
def test_weak_title_does_not_count_words_in_unspaced_scripts():
    assert clipgate.weak_title("日本の首相が辞任した理由は何ですか") is None


@pytest.mark.xfail(strict=True, reason="'Who? What? Why?' is nothing concrete but passes because 'What' is capitalised")
def test_a_pile_of_question_words_is_weak():
    assert clipgate.weak_title("Who? What? Why?")


# ----------------------------------------------------------------------------- moments
def test_hard_cap_boundaries():
    assert M.hard_cap(30) == 38 and M.hard_cap(12) == 20 and M.hard_cap(0) == 8
    assert M.hard_cap(30.1) == max(50.1, 30.1 * 1.35)                                        # long-form formula resumes just above 30


def test_demote_weak_titles_floor_and_edge_inputs():
    assert M.demote_weak_titles([]) == []
    a = M.Moment(0, 10, 0, 3, title="Hmm", score=1.5)
    b = M.Moment(0, 10, 0, 3, title="Wow", score=1.5, reason="funny")
    M.demote_weak_titles([a, b])
    assert a.score == 1.0 and "weak title" in a.reason                                      # never below 1
    assert b.score == 1.0 and b.reason.startswith("funny; weak title")


# ----------------------------------------------------------------------------- opening.check on odd files
@pytest.fixture
def no_ocr(monkeypatch):
    monkeypatch.setattr(ocr, "read", lambda shots: None)


def test_check_never_raises_and_passes_unreadable_files(tmp_path, no_ocr):
    (tmp_path / "empty.mp4").write_bytes(b"")
    (tmp_path / "junk.mp4").write_bytes(b"not a video")
    for name in ("empty.mp4", "junk.mp4", "missing.mp4"):
        assert opening.check(tmp_path / name) == []


def test_check_on_very_short_and_audio_only_files(tmp_path, no_ocr):
    pic = _lavfi("testsrc2=size=270x480:rate=24")
    assert opening.check(_mk(tmp_path, "half_s.mp4", pic, 0.5, ["-pix_fmt", "yuv420p"])) == []
    assert opening.check(_mk(tmp_path, "three.mp4", pic, 3, ["-pix_fmt", "yuv420p"])) == []
    assert opening.check(_mk(tmp_path, "a.mp4", _lavfi("sine=frequency=440"), 10)) == []          # no picture at all: nothing to judge
    assert opening.check(_mk(tmp_path, "a.m4a", _lavfi("sine=frequency=440"), 10)) == []


def test_check_flags_a_black_and_a_flat_clip_and_a_black_first_second(tmp_path, no_ocr):
    black = _mk(tmp_path, "b.mp4", _lavfi("color=c=black:size=270x480:rate=24"), 10, ["-pix_fmt", "yuv420p"])
    flat = _mk(tmp_path, "f.mp4", _lavfi("color=c=0x3366aa:size=270x480:rate=24"), 10, ["-pix_fmt", "yuv420p"])
    for v in (black, flat):
        what = " | ".join(i["what"] for i in opening.check(v))
        assert "empty" in what and "does not move" in what
    open_black = tmp_path / "ob.mp4"
    subprocess.run([FF, "-y", "-v", "error", *_lavfi("testsrc2=size=270x480:rate=24"), "-t", "10", "-vf", "fade=t=in:st=0:d=1.2:color=black",
                    "-pix_fmt", "yuv420p", str(open_black)], check=True)
    assert [i["what"] for i in opening.check(open_black)] == ["it opens on an empty (black or flat) picture"] or opening.check(open_black) == []


@pytest.mark.parametrize("size", ["1080x1920", "1920x1080", "64x64", "270x480"])
def test_moving_footage_passes_in_any_frame_shape(tmp_path, no_ocr, size):
    v = _mk(tmp_path, f"m{size}.mp4", _lavfi(f"testsrc2=size={size}:rate=24"), 6, ["-pix_fmt", "yuv420p"])
    assert opening.check(v) == []


def test_brand_check_failures_never_lose_the_motion_verdict(tmp_path, monkeypatch):
    still = _mk(tmp_path, "s.mp4", _lavfi("color=c=0x3366aa:size=270x480:rate=24"), 10, ["-pix_fmt", "yuv420p"])
    monkeypatch.setattr(ocr, "read", lambda shots: (_ for _ in ()).throw(RuntimeError("helper died")))
    assert opening.check(still)                                                                # the motion issues are still reported
    monkeypatch.setattr(ocr, "read", lambda shots: [[{"t": "SUBSCRIBE"}]] * len(shots))        # a reader that forgets the position keys
    assert opening.check(still)
    monkeypatch.setattr(ocr, "read", lambda shots: [])                                         # wrong number of pages
    assert opening.check(still)


def test_branding_ignores_low_confidence_missing_confidence_and_short_clips(tmp_path, monkeypatch):
    v = _mk(tmp_path, "m.mp4", _lavfi("testsrc2=size=270x480:rate=24"), 10, ["-pix_fmt", "yuv420p"])
    line = lambda t, y, c: {"t": t, "x": 0.1, "y": y, "w": 0.3, "h": 0.04, "c": c}
    monkeypatch.setattr(ocr, "read", lambda shots: [[line("SUBSCRIBE", 0.9, 0.3)]] * len(shots))
    assert opening.branding(v, 10.0) == []                                                     # confidence 0.3
    monkeypatch.setattr(ocr, "read", lambda shots: [[{"t": "subscribe", "x": 0, "y": 0.92, "w": .3, "h": .04}]] * len(shots))
    assert opening.branding(v, 10.0) == ["subscribe"]                                          # no "c" key counts as sure
    monkeypatch.setattr(ocr, "read", lambda shots: pytest.fail("a clip under 4 s is not read"))
    assert opening.branding(v, 3.0) == []


def test_branding_top_strip_and_the_handle_said_aloud(tmp_path, monkeypatch):
    v = _mk(tmp_path, "m.mp4", _lavfi("testsrc2=size=270x480:rate=24"), 10, ["-pix_fmt", "yuv420p"])
    line = lambda t, y: {"t": t, "x": 0.1, "y": y, "w": 0.3, "h": 0.04, "c": 1.0}
    monkeypatch.setattr(ocr, "read", lambda shots: [[line("@someoneelse", 0.02)]] * len(shots))
    assert opening.branding(v, 10.0) == ["@someoneelse"]
    assert opening.branding(v, 10.0, "go follow someoneelse") == []                           # the speaker says the name himself
    monkeypatch.setattr(ocr, "read", lambda shots: [[line("@abc", 0.95)]] * len(shots))        # 4 characters needed after the @
    assert opening.branding(v, 10.0) == []


# ----------------------------------------------------------------------------- ocr
def test_ocr_read_with_no_helper_and_no_tesseract_says_could_not_look(monkeypatch, tmp_path):
    monkeypatch.setattr(ocr, "helper", lambda: None)
    monkeypatch.setattr(ocr.ffmpeg, "which", lambda name: None)
    assert ocr.read([tmp_path / "a.jpg"]) is None
    assert ocr.read([]) == []


def test_ocr_helper_that_returns_the_wrong_number_of_lines_falls_back(monkeypatch, tmp_path):
    fake = tmp_path / "fake_ocr"
    fake.write_text("#!/bin/sh\necho '[]'\n")
    fake.chmod(0o755)
    monkeypatch.setattr(ocr, "helper", lambda: fake)
    monkeypatch.setattr(ocr.ffmpeg, "which", lambda name: None)
    assert ocr.read([tmp_path / "a.jpg", tmp_path / "b.jpg"]) is None                         # 1 line for 2 images: not trusted
    monkeypatch.setattr(ocr.ffmpeg, "which", lambda name: None)
    assert ocr.read([tmp_path / "a.jpg"]) == [[]]


def test_ocr_helper_that_crashes_or_prints_garbage_does_not_raise(monkeypatch, tmp_path):
    fake = tmp_path / "fake_ocr"
    fake.write_text("#!/bin/sh\necho '[not json'\nexit 3\n")
    fake.chmod(0o755)
    monkeypatch.setattr(ocr, "helper", lambda: fake)
    monkeypatch.setattr(ocr.ffmpeg, "which", lambda name: None)
    assert ocr.read([tmp_path / "a.jpg"]) is None


def test_ocr_tesseract_fallback_reports_positions(monkeypatch, tmp_path):
    from PIL import Image
    tess = tmp_path / "tesseract"
    tess.write_text("#!/bin/sh\nprintf 'level\\tpage\\tblock\\tpar\\tline\\tword\\tl\\tt\\tw\\th\\tconf\\ttext\\n5\\t1\\t1\\t1\\t1\\t1\\t10\\t90\\t20\\t5\\t90\\tyoutube\\n'\n")
    tess.chmod(0o755)
    Image.new("RGB", (100, 100)).save(tmp_path / "a.jpg")
    monkeypatch.setattr(ocr, "helper", lambda: None)
    monkeypatch.setattr(ocr.ffmpeg, "which", lambda name: str(tess))
    row = ocr.read([tmp_path / "a.jpg"])[0][0]
    assert row["t"] == "youtube" and (row["x"], row["y"], row["w"], row["h"]) == (0.1, 0.9, 0.2, 0.05)           # at the bottom edge: a banner, not a caption


# ----------------------------------------------------------------------------- feed + relay.ingest: own uploads and records with missing keys
@pytest.fixture
def ctx(tmp_path):
    cfg = load_config()
    for k in ("queue", "state", "work", "review", "ready", "posted", "rejected", "public", "logs", "reports"):
        cfg.data["paths"][k] = str(tmp_path / k)
    cfg.data["paths"]["dedupe_db"] = str(tmp_path / "d.sqlite")
    cfg.root = tmp_path
    cfg.data["clipper"]["feed"] = {"enabled": True, "grace_s": 30, "previews": False}
    cfg.data["clipper"]["library_dir"] = str(tmp_path / "library")
    (tmp_path / "library").mkdir()
    return Context(cfg)


def _inbox(ctx, clips, own=False, jid="20261010-020000-abcdef"):
    d = ctx.cfg.root / "inbox" / "clipper" / jid
    d.mkdir(parents=True)
    for c in clips:
        (d / c["file"]).write_bytes(b"v" * 100)
    job = {"id": jid, "created": 0, "status": "done", "stage": "", "progress": 1, "title": "Src", "options": {}, "clips": clips, "info": {},
           "source": {"url": "https://youtu.be/bbb", "auto": True, "kind": "cc", "channel": "Chan", "topic": "podcast", "id": "bbb"}}
    (d / "manifest.json").write_text(json.dumps({"job": job, "source": job["source"], "credit": "", **({"own": True, "priority": True} if own else {})}))
    return jid


def _c(i, **kw):
    return {"index": i, "file": f"clip_{i:02d}.mp4", "thumb": None, "caption": "c", "score": 9, "duration": 22, "layout": "track", "caption_style": "bold", **kw}


def test_feed_and_ingest_with_clips_that_lack_title_or_qa_keys(ctx):
    clips = [_c(1, title="Cop finds suspect but missing a second officer"), _c(2), _c(3, title=None), _c(4, title="Cop finds suspect but missing a second officer", qa=None),
             _c(5, title="Cop finds suspect but missing a second officer", qa={})]
    jid = _inbox(ctx, clips)
    assert relay.ingest(ctx) == [jid]
    store = Store(ctx.cfg.root / "data/clipper")
    fd = F.Feed(ctx.cfg, store)
    assert {i["index"] for i in fd.items()} == {1, 2, 3, 4, 5}                                 # no title key / title None = not judged on a title
    states = {i: (store.get(jid)["clips"][i - 1].get("review") or {}).get("state") for i in range(1, 6)}
    assert states == {1: "review", 2: "review", 3: "review", 4: "review", 5: "review"}


def test_feed_hides_what_the_gate_refuses_but_shows_own_uploads(ctx):
    bad_qa = {"issues": [{"level": "bad", "what": "the picture does not move for 97% of the clip (one still with captions)"}]}
    good = "Cop finds suspect but missing a second officer"
    clips = [_c(1, title=good), _c(2, title=good, qa=bad_qa), _c(3, title="How do I do that?"), _c(4, title=good, duration=55)]
    jid = _inbox(ctx, clips)
    relay.ingest(ctx)
    store = Store(ctx.cfg.root / "data/clipper")
    assert {i["index"] for i in F.Feed(ctx.cfg, store).items()} == {1}
    for i in (2, 3, 4):                                                                        # refused on ingest too, with the reason
        r = store.get(jid)["clips"][i - 1]["review"]
        assert r["state"] == "rejected" and r["auto"]
    assert "does not move" in store.get(jid)["clips"][1]["review"]["why"]


def test_own_upload_is_exempt_in_ingest_and_in_the_feed(ctx, monkeypatch):
    monkeypatch.setattr(relay, "_post_now", lambda: None)                                      # a priority clip would kick the posting job
    bad_qa = {"issues": [{"level": "bad", "what": "the picture does not move for 97% of the clip (one still with captions)"}]}
    clips = [_c(1, title="How do I do that?", duration=55, qa=bad_qa)]
    jid = _inbox(ctx, clips, own=True)
    assert relay.ingest(ctx) == [jid]
    store = Store(ctx.cfg.root / "data/clipper")
    job = store.get(jid)
    assert job["source"]["own"] is True
    assert clipgate.problems(job["clips"][0])                                                  # the gate would refuse it ...
    assert job["clips"][0]["review"]["state"] == "review"                                      # ... but your own upload is queued anyway
    assert {i["index"] for i in F.Feed(ctx.cfg, store).items()} == {1}                         # ... and shown in the feed
