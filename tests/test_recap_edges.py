"""Recap reels: odd inputs to the planning (no ffmpeg)."""
import pytest

from clipfarm.clipper import recap as R


def _shots(n, fps, per, files=5):
    return [{"id": i, "file": f"f{i % files}.mov", "path": "x", "start": 0.0, "end": per, "what": "", "quality": 6, "fps": fps} for i in range(n)]


@pytest.mark.parametrize("fps", [24, 25, 50, 59.94, 60, 120])
@pytest.mark.parametrize("per", [0.5, 1.2, 8.0])
@pytest.mark.parametrize("length", [0.5, 2.0, 7.9, 12.0, 30.0])
def test_lay_out_is_gapless_and_the_right_length_for_any_fps_and_scene_length(fps, per, length):
    segs = R.merge_talk(R.lay_out(length, _shots(40, fps, per)))
    t = 0.0
    for s in segs:
        assert abs(s["at"] - t) < 0.02 and s["d"] > 0
        t = s["at"] + s["d"]
        if s["kind"] == "broll":
            assert s["src_d"] <= per + 1e-6 and s["speed"] >= R.MAX_SLOW
    assert abs(t - length) < 0.03


def test_windows_with_no_words_and_tiny_videos_do_not_crash():
    assert R.windows([], 60.0, None, 3, 30.0)
    assert R.windows([], 0.0, None, 3, 30.0) == []
    assert len(R.windows([{"text": "a", "start": 0, "end": 0.4}], 5.0, None, 1, 30.0)) == 1
    assert R.windows([{"text": "a", "start": 0, "end": 0.4}], 3.0, None, 1, 30.0) == []


# (fixed: a short video makes one reel, or says why not)
def test_a_short_video_still_gets_one_window():
    assert len(R.windows([{"text": "a", "start": 0, "end": 0.4}], 8.0, None, 1, 30.0)) == 1


# (fixed: the last shot takes the missing frames)
def test_lay_out_is_exact_with_short_scenes():
    segs = R.merge_talk(R.lay_out(19.9, _shots(40, 60, 2.0)))
    assert abs(sum(s["d"] for s in segs) - 19.9) < 0.03
