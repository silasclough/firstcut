"""Edge cases found by the tester: crop_filter on both frame shapes, plans past the limit, check failures never raise, and false positives in the gate
."""
import subprocess
from pathlib import Path

import pytest

from clipfarm.clipper import clipgate, opening, sourcecheck as SC
from clipfarm.core import ffmpeg


@pytest.mark.parametrize("sh", [1080, 1920])
@pytest.mark.parametrize("plan", [{"top": 0.1, "bottom": 0.2}, {"top": 0.0, "bottom": 0.3}, {"top": 0.012, "bottom": 0.012}])
def test_crop_filter_keeps_even_sizes_inside_the_picture(sh, plan):
    f, h, y = SC.crop_filter(sh, plan)
    assert f and h % 2 == 0 and y % 2 == 0 and y + h <= sh and h < sh


def test_crop_filter_nothing_to_do_and_tiny_pictures():
    assert SC.crop_filter(1080, {"top": 0, "bottom": 0}) == ("", 1080, 0)
    assert SC.crop_filter(40, {"top": 0.1, "bottom": 0.2}) == ("", 40, 0)


def test_crop_filter_really_renders_without_distortion(tmp_path):
    src = tmp_path / "s.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=640x360:d=2:r=10", str(src)], check=True)
    f, h, y = SC.crop_filter(360, {"top": 0.1, "bottom": 0.2})
    out = tmp_path / "o.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vf", f.lstrip(","), str(out)], check=True)
    assert ffmpeg.probe(out)["streams"][0]["width"] == 640 and ffmpeg.probe(out)["streams"][0]["height"] == h


def test_a_plan_past_the_limit_crops_nothing():
    p = SC.crop_plan({"top": 0.1, "bottom": 0.3, "captions": False, "cap_band": None})
    assert p["top"] == 0 and p["bottom"] == 0


def test_missing_and_garbage_files_are_never_a_reason_to_refuse(tmp_path):
    bad = tmp_path / "junk.mp4"
    bad.write_bytes(b"not a video")
    assert clipgate.check_file(tmp_path / "nope.mp4")["issues"] == []
    assert clipgate.check_file(bad)["issues"] == []
    assert SC.probe_source(bad)["still"] is False
    assert SC.motion_of(bad, 0, 5)["still"] is False


@pytest.mark.parametrize("t", ["Kevin Nealon's Terrifying Johnny Carson Moment", "What happened when control rods were removed from Chernobyl", "How The Office almost died"])
def test_good_titles_pass(t):
    assert clipgate.weak_title(t) is None


@pytest.mark.parametrize("t", ["Police overwatch team saves a hostage", "Couples therapy roleplay exercise that works", "How billionaires are gaming the tax system",
                               "Why the full game of life is rigged", "A gta truck driver in Texas"])
def test_real_life_titles_with_game_words_are_not_game_footage(t):
    assert clipgate.gaming_why({"title": t}) is None


def test_three_wrapped_caption_lines_that_share_the_are_not_doubled():
    mk = lambda y, t: {"t": t, "y": y, "h": 0.04, "x": 0.2, "w": 0.6, "c": 0.9}
    assert not opening.double_captions([[mk(0.62, "THE REACTOR WAS"), mk(0.69, "THE ONLY ONE"), mk(0.76, "IN THE WORLD")]] * 3)


@pytest.mark.parametrize("t", ["New paleto heist, but the computer's broken", "GTA RP: my first day as a cop", "Valorant ace in the last round", "Full gameplay of the new level"])
def test_game_titles_are_still_refused(t):
    assert clipgate.gaming_why({"title": t})


def test_one_game_word_list_in_both_modules():
    assert SC.GAME_WORDS is clipgate.GAME_WORDS
    assert SC.gameplay_meta({"title": "Why the full game of life is rigged", "tags": ["family"]}) is None
    assert SC.gameplay_meta({"title": "A day", "tags": ["gaming"]})
    assert clipgate.gaming_why({"title": "A day"}, {"topic": "gaming"})


def test_plan_past_the_limit_turns_our_captions_off_when_the_source_has_its_own():
    f = {"promo": ["x"], "top": 0.1, "bottom": 0.0, "captions": True, "cap_band": (0.72, 0.8)}      # bottom crop would be 0.3: with the top 0.4, over the limit
    plan = SC.crop_plan(f)
    assert plan == {"top": 0.0, "bottom": 0.0, "own_captions": False}
    assert SC.crop_plan({"promo": ["x"], "top": 0.2, "bottom": 0.3, "captions": False, "cap_band": None})["own_captions"] is True
