"""Checks that keep a weak clip from being posted: a picture that never moves, burned-in promo from another channel, a vague title, a clip that is too long."""

import subprocess

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import clipgate, moments as M, ocr, opening
from clipfarm.core import ffmpeg

GOOD = ["Cop finds suspect but missing a second officer", "Why the comedy industry doesn't want real people", "How the Chernobyl reactor exploded",
        "A man's story changed with every question", "Chat hits 62 million just as IShowSpeed buys iPhone duo", "Who topped Spotify in 2025?",
        "100 billion neurons with 7000 connections each", "xQc says every brick matters when building a wall"]
BAD = ["How do I do that?", "How you doing?", "What just happened in this game?", "Streamer reacts to unexpected moment", "You won't believe this one", "Wow",
       "Crazy story", "Hey guys welcome back"]


@pytest.mark.parametrize("t", GOOD)
def test_good_titles_pass(t):
    assert clipgate.weak_title(t) is None


@pytest.mark.parametrize("t", BAD)
def test_vague_titles_are_refused(t):
    assert clipgate.weak_title(t)


@pytest.mark.parametrize("t", ["Why did he lie about 12 million dollars", "Hi-tech Samsung phone explodes in Ohio", "Hello Kitty sells 5 billion dollars", "Mars Landing"])
def test_titles_with_a_number_a_name_or_two_words_are_fine(t):
    assert clipgate.weak_title(t) is None


def test_gate_refuses_long_clips_and_weak_titles_but_not_a_normal_clip():
    ok = {"duration": 22, "title": GOOD[0], "transcript": "He walked in. Nobody was there."}
    assert clipgate.problems(ok) == []
    assert any("seconds long" in p for p in clipgate.problems({**ok, "duration": 55}))
    assert any("weak title" in p for p in clipgate.problems({**ok, "title": "How do I do that?"}))
    assert clipgate.problems({"duration": 22, "transcript": "He walked in. Nobody was there."}) == []        # a clip record with no title is not judged on one


def test_gate_refuses_what_the_opening_check_found():
    bad = {"duration": 22, "title": GOOD[0], "transcript": "Done.", "qa": {"issues": [{"level": "bad", "what": "the picture does not move for 97% of the clip"}]}}
    assert clipgate.problems(bad) == ["the picture does not move for 97% of the clip"]


def test_short_form_may_run_eight_seconds_over_but_long_form_keeps_its_slack():
    assert M.hard_cap(28) == 36
    assert M.hard_cap(55) == 75


def test_weak_titles_lose_score_and_good_ones_do_not():
    a, b = M.Moment(0, 10, 0, 3, title="How do I do that?", score=8), M.Moment(0, 10, 0, 3, title=GOOD[0], score=8)
    M.demote_weak_titles([a, b])
    assert a.score == 5.5 and "weak title" in a.reason and b.score == 8


def _frames(kind, n=200):
    rng = np.random.default_rng(1)
    base = rng.integers(40, 200, (opening.H, opening.W)).astype(np.float32)
    if kind == "still":
        return np.stack([base] * n)
    if kind == "moving":
        return np.stack([np.roll(base, 3 * i, axis=1) for i in range(n)])
    if kind == "black":
        return np.stack([np.zeros_like(base)] * n)
    raise ValueError(kind)


def test_motion_tells_a_slideshow_from_real_footage():
    assert opening.motion(_frames("still"))["still_share"] == 1.0
    assert opening.motion(_frames("moving"))["still_share"] == 0.0
    assert opening.motion(_frames("black"))["blank_open"] is True
    assert opening.motion(_frames("moving"))["blank_open"] is False


def test_a_still_opening_is_found_even_when_the_rest_moves():
    f = np.concatenate([_frames("still", 40), _frames("moving", 120)])
    m = opening.motion(f)
    assert m["still_open"] >= opening.STILL_OPEN and m["still_share"] < opening.STILL_CLIP


def _video(tmp_path, moving):
    if moving:
        src = ["-f", "lavfi", "-i", "testsrc2=size=270x480:rate=24"]
    else:
        im = tmp_path / "s.png"
        Image.fromarray(np.random.default_rng(2).integers(0, 255, (480, 270, 3), dtype=np.uint8)).save(im)
        src = ["-loop", "1", "-framerate", "24", "-i", str(im)]
    out = tmp_path / ("m.mp4" if moving else "s.mp4")
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *src, "-t", "10", "-pix_fmt", "yuv420p", str(out)], check=True)
    return out


def test_check_on_real_files(tmp_path):
    still, moving = _video(tmp_path, False), _video(tmp_path, True)
    assert any("does not move" in i["what"] for i in opening.check(still, brand=False))
    assert opening.check(moving, brand=False) == []
    assert opening.qa_field(still, brand=False)["issues"]


def _page(*lines):
    return [{"t": t, "x": 0.1, "y": y, "w": 0.3, "h": 0.04, "c": 1.0} for t, y in lines]


def test_burned_in_promo_is_found_at_the_edges_only(tmp_path, monkeypatch):
    v = _video(tmp_path, True)
    monkeypatch.setattr(ocr, "read", lambda shots: [_page(("SUBSCRIBE", 0.9)), _page(("hello there", 0.5)), _page(("@officialandy", 0.93)), [], [], []][: len(shots)])
    assert opening.branding(v, 10.0, "") == ["@officialandy", "subscribe"]
    monkeypatch.setattr(ocr, "read", lambda shots: [_page(("subscribe to my newsletter", 0.5))] * len(shots))       # in the middle of the picture: our own captions live there
    assert opening.branding(v, 10.0, "") == []
    monkeypatch.setattr(ocr, "read", lambda shots: [_page(("SUBSCRIBE", 0.9))] * len(shots))
    assert opening.branding(v, 10.0, "he said subscribe to the channel") == []                                  # said aloud: it is the speaker's sentence, not a banner


def test_no_text_reader_means_not_checked_rather_than_failed(tmp_path, monkeypatch):
    v = _video(tmp_path, True)
    monkeypatch.setattr(ocr, "read", lambda shots: None)
    assert opening.branding(v, 10.0, "") is None
    assert opening.check(v) == []


def test_the_swipe_feed_hides_what_the_posting_gate_would_refuse(tmp_path):
    from clipfarm.clipper import feed
    assert feed.clipgate is clipgate                                    # one rule, two places: the feed does not show a clip a like could never post


def test_hook_checks_flag_late_speech_and_no_face_as_warnings():
    from clipfarm.clipper import clipcraft as CC
    on_time = [{"text": "a", "start": 0.2, "end": 0.5}]
    late = [{"text": "a", "start": 1.4, "end": 1.7}]
    assert CC.hook_warnings(on_time, 20, 0.4, True) == []
    w = CC.hook_warnings(late, 20, 2.0, True)
    assert {i["code"] for i in w} == {"hook_late", "no_face_early"} and all(i["level"] == "warn" for i in w)
    assert [i["code"] for i in CC.hook_warnings(on_time, 20, None, True)] == ["no_face_early"]
    assert CC.hook_warnings(on_time, 20, None, False) == []                       # faces were not measured (no face model, or a layout without them): nothing to say
    assert CC.hook_warnings([], 20, 0.0, False)[0]["code"] == "hook_late"
    assert CC.hook_warnings(late, 2.0, None, True) == []                          # too short to judge
    assert CC.first_face_time([[], [], ["f"]], 4.0) == 0.5 and CC.first_face_time([[]], 4.0) is None
    # a warning is never a rejection: the gate only refuses level "bad"
    clip = {"duration": 20, "transcript": "A complete sentence.", "title": "A man robs a bank in Ohio", "qa": {"issues": w}}
    assert not [p for p in clipgate.problems(clip, None) if "face" in p or "first word" in p]
