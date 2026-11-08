"""Edge cases for WP7/WP8/WP9/WP11 found by the tester: hostile AI answers, faces at the border, the caption band over a long video, the clip's end.
The tests marked xfail(strict) are real defects (they fail today; when the product is fixed they flip to XPASS and strict makes the run fail, so the
marker has to be removed)."""
from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from clipfarm.clipper import brollsound as BS
from clipfarm.clipper import clipcraft as CC
from clipfarm.clipper import covers as COV
from clipfarm.clipper import placement as PL
from clipfarm.clipper import switching as SW
from clipfarm.clipper import visualnouns as VN


class _AI:
    def __init__(self, data):
        self.data = data

    def complete_json(self, *a, **k):
        if isinstance(self.data, Exception):
            raise self.data
        return NS(data=self.data)


def _sents():
    words = [{"text": w, "start": i * 0.4, "end": i * 0.4 + 0.35} for i, w in enumerate("the red car drove down the long road quickly".split())]
    return [NS(idx=0, start=0.0, end=1.5, text="the red car drove"), NS(idx=1, start=1.6, end=3.5, text="down the long road quickly")], words


# ------------------------------------------------------------------------------------------------ WP9: hostile AI answers
@pytest.mark.parametrize("answer", [
    {"sentences": [{"i": 99, "nouns": [{"phrase": "car"}]}]},                       # an index the code never offered
    {"sentences": "nope"}, {}, None, [], {"sentences": [None, 1, "x", {"i": None}, {"i": "zz"}]},
    {"sentences": [{"i": 0, "nouns": "abc"}, {"i": 1, "nouns": {"a": 1}}, {"i": 1, "tags": "hook"}]},
    {"sentences": [{"i": 0, "nouns": [None, "s", {"phrase": None}, {"phrase": "x" * 5000}, {"phrase": "unicorn"}]}]},
])
def test_visual_nouns_survive_invented_or_broken_answers(answer):
    sents, words = _sents()
    r = VN.extract(_AI(answer), sents, words, {})
    assert r["nouns"] == [] and r["tags"] in ({}, {1: []}) or all(n["word_first"] <= n["word_last"] < len(words) for n in r["nouns"])


def test_visual_nouns_phrases_are_clamped_and_word_numbers_come_from_code():
    sents, words = _sents()
    ans = {"sentences": [{"i": 0, "nouns": [{"phrase": "red car", "concrete": "nan", "kind": "bogus"}, {"phrase": "car", "concrete": 7}], "tags": ["hook", "bogus"]}]}
    r = VN.extract(_AI(ans), sents, words, {})
    assert [(n["word_first"], n["word_last"]) for n in r["nouns"]] == [(1, 2), (2, 2)]
    assert all(0.0 <= n["concrete"] <= 1.0 and n["kind"] == "object" for n in r["nouns"])
    assert r["tags"] == {0: ["hook"]}


def test_visual_nouns_with_no_sentences_asks_nothing_and_says_not_ok():
    r = VN.extract(_AI(RuntimeError("must not be called")), [], [], {})
    assert r == {"nouns": [], "tags": {}, "ok": False}


def test_a_failing_ai_is_not_cached():
    sents, words = _sents()
    c = {}
    assert VN.extract(_AI(RuntimeError("down")), sents, words, c)["ok"] is False and c == {}


def test_a_two_hour_transcript_is_asked_in_chunks_of_sixty():
    n = 2400                                                  # ~2 h of speech
    sents = [NS(idx=i, start=i * 3.0, end=i * 3.0 + 2.9, text=f"s{i} red car") for i in range(n)]
    words = [{"text": "red", "start": i * 3.0, "end": i * 3.0 + 0.4} for i in range(n)]
    calls = []

    class Many:
        def complete_json(self, task, system, prompt, schema):
            calls.append(prompt.count("\n") + 1)
            return NS(data={"sentences": []})
    VN.extract(Many(), sents, words, {})
    assert len(calls) == n // VN.CHUNK and max(calls) <= VN.CHUNK + 1


def test_an_infinite_sentence_index_does_not_raise():
    sents, words = _sents()
    VN.extract(_AI({"sentences": [{"i": float("inf")}]}), sents, words, {})


# ------------------------------------------------------------------------------------------------ WP8: rubric and the clip's edges
@pytest.mark.parametrize("bad", [float("nan"), 0, 6, -1, "x", None, "", [], {}])
def test_rubric_rejects_missing_nan_and_out_of_range_grades(bad):
    assert CC.rubric_of({"hook": bad, "flow": 3, "value": 3}) is None
    assert CC.score_of(None, 4.0) == 4.0


def test_rubric_rejects_an_infinite_grade():
    assert CC.rubric_of({"hook": float("inf"), "flow": 3, "value": 3}) is None


def test_score_scale_is_the_same_for_every_clip():
    assert CC.score_of({"hook": 3, "flow": 3, "value": 3}) == 6.0 and CC.score_of({"hook": 5, "flow": 5, "value": 5}) == 10.0
    assert CC.score_of({"hook": 1, "flow": 1, "value": 1}) == 2.0


def test_length_band_for_a_short_and_a_long_clip():
    assert "6-60 seconds" in CC.length_band(6, 60) and "24 seconds" in CC.length_band(6, 60)
    assert "6-6" in CC.length_band(6, 6)


def test_a_clip_that_ends_mid_word_ends_after_that_word_plus_the_tail():
    words = [{"text": "one", "start": 0.0, "end": 0.5}, {"text": "two", "start": 1.5, "end": 2.0}]
    (a, b), = CC.end_after_last_word([(0.0, 0.3)], words)
    assert abs(b - (0.5 + CC.TAIL_S)) < 1e-6
    assert CC.end_after_last_word([], words) == [] and CC.end_after_last_word([(0, 1)], []) == [(0, 1)]


def test_the_clip_never_ends_inside_the_next_word():
    words = [{"text": "a", "start": 0.0, "end": 0.5}, {"text": "b", "start": 0.52, "end": 1.0}]
    (_a, b), = CC.end_after_last_word([(0.0, 0.3)], words)
    assert b <= 0.52 + 1e-6


def test_trim_lead_never_moves_the_start_later_or_past_the_end():
    assert CC.trim_lead([(0, 5)], [{"text": "x", "start": 0.05, "end": 0.3}]) == [(0, 5)]
    assert CC.trim_lead([(2.0, 5)], [{"text": "x", "start": 4.9, "end": 5.3}]) == [(2.0, 5)]
    assert CC.trim_lead([(0, 5)], [{"text": "x", "start": 1.0, "end": 1.3}])[0][0] == pytest.approx(0.85)
    assert CC.trim_lead([], []) == []


def test_hook_warnings_skip_clips_under_three_seconds_and_survive_nan():
    assert CC.hook_warnings([], 2.0, None, True) == []
    assert isinstance(CC.hook_warnings([{"start": float("nan")}], 10, float("nan"), True), list)


# ------------------------------------------------------------------------------------------------ WP7: punch-ins at the edges
@pytest.mark.parametrize("cx,cy", [(0.0, 0.0), (1.0, 1.0), (0.02, 0.98), (0.5, 0.0), (float("nan"), 0.5)])
@pytest.mark.parametrize("W,H", [(1080, 1920), (1920, 1080)])
@pytest.mark.parametrize("zoom", [1.05, 1.12, 1.25])
def test_a_punch_in_on_a_face_at_the_border_stays_inside_the_picture(cx, cy, W, H, zoom):
    c = COV.punch_centre({"cx": cx, "cy": cy, "h": 0.2}, zoom)
    x, y = COV.crop_offsets(W, H, zoom, c)
    sw, sh = int(W * zoom) // 2 * 2, int(H * zoom) // 2 * 2
    assert 0 <= x <= sw - W and 0 <= y <= sh - H and x % 2 == 0 and y % 2 == 0
    assert COV.zoom_filter(W, H, zoom, c).startswith(f"scale={sw}:{sh},crop={W}:{H}:")


def test_a_face_near_the_edge_is_still_inside_the_punched_crop():
    z = 1.25
    cx, cy = COV.punch_centre({"cx": 0.95, "cy": 0.3, "h": 0.2}, z)
    assert cx + 0.5 / z <= 1.0 + 1e-9 and cx - 0.5 / z >= -1e-9 and cx - 0.5 / z <= 0.95 <= cx + 0.5 / z


def test_punch_size_is_always_inside_the_practice_range():
    for v in (None, "x", float("nan"), float("inf"), -5, 0, 9):
        assert 1.05 <= COV.punch_size(v) <= 1.25 and 1.05 <= COV.punch_size(v, portrait=True) <= 1.25


def test_no_device_fits_leaves_the_cut_and_says_so_and_broll_never_repeats():
    out, notes = COV.plan_covers([{"t": 1, "can": None}, {"t": 2, "can": {"broll": True}}, {"t": 3, "can": {"broll": True}}])
    assert [d["device"] for d in out] == ["broll"] and len(notes) == 2


# ------------------------------------------------------------------------------------------------ WP7: switching
def test_apply_with_no_segments_or_no_points_changes_nothing_harmful():
    assert SW.apply([], [], [], 3.0) == []
    segs = [{"start": 0, "end": 10, "angle": "A"}, {"start": 10, "end": 20, "angle": "B"}]
    assert [s["angle"] for s in SW.apply(segs, [], [], 3.0)] == ["A", "B"]       # (no points: the director's own boundary, as asked)


def test_a_one_second_hold_is_merged_and_a_reaction_never_cuts_in_without_a_second_of_speaker():
    segs = [{"start": 0, "end": 10, "angle": "A"}, {"start": 10, "end": 10.5, "angle": "B"}]
    assert len(SW.apply(segs, [{"t": 10, "allowed": None}], [], 3.0)) == 1
    cuts = [{"start": 0, "end": 10, "angle": "A"}]
    out = SW.insert_reactions(cuts, [{"start": 0.5, "end": 2.0, "angle": "B"}, {"start": 9.5, "end": 10.2, "angle": "B"}])
    assert out == cuts


# ------------------------------------------------------------------------------------------------ WP9: whose sound is in the video
@pytest.mark.parametrize("notes,expected", [("Keep the natural sound of the clips.", "*"), ("", set()), (None, set()), ("natural soundtrack please", set())])
def test_natural_sound_is_only_on_when_asked(notes, expected):
    assert BS.allowed_files(notes, None) == expected


def test_a_note_that_refuses_natural_sound_does_not_turn_it_on():
    assert BS.allowed_files("no natural sound please", None) == set()
    assert BS.allowed_files("don't keep the natural sound", None) == set()


def test_a_talking_broll_file_is_dropped_only_when_real_talking_footage_exists_and_never_for_exact_lists():
    talk = {"name": "talk", "start": 0, "end": 10, "audio": True, "path": "/t"}
    chat = {"name": "b", "start": 10, "end": 20, "audio": True, "path": "/b", "facts": {"role": "broll", "speech": {"ratio": 0.6}}}
    out, notes = BS.drop_broll_speech([(0, 20)], [talk, chat])
    assert out == [(0, 10.0)] and notes
    assert BS.drop_broll_speech([(0, 20)], [chat])[0] == [(0, 20)]                # a folder of only such files keeps them
    assert BS.drop_broll_speech([(0, 20)], [talk, chat], exact=True)[0] == [(0, 20)]
    assert BS.drop_broll_speech([(0, 20)], [talk, chat], keep_sound_ok="*")[0] == [(0, 20)]
    assert BS.drop_broll_speech([(0, 20)], [talk, dict(chat, facts="junk")])[0] == [(0, 20)]


def test_a_bad_graded_file_is_flagged_and_a_failing_grader_never_stops_a_render():
    class Bad:
        def grade(self, p, ss):
            return {"grade": "bad"}

    class Boom:
        def grade(self, p, ss):
            raise RuntimeError("x")
    f = [{"name": "b", "path": "/b"}]
    parts = [{"a": {"path": "/b", "ss": 1}, "out": 0, "d": 2}, {"a": {"path": "/b", "ss": 3}, "out": 2.1, "d": 2}, {"a": None}]
    assert BS.poor_sound_flags(parts, f, Bad()) == [{"start": 0.0, "end": 4.1, "file": "b", "why": "muffled or far from the microphone"}]
    assert BS.poor_sound_flags(parts, f, Boom()) == []


# ------------------------------------------------------------------------------------------------ WP11: placement
@pytest.mark.parametrize("W,H", [(1080, 1920), (1920, 1080)])
def test_a_card_never_goes_over_mouth_eyes_or_the_caption_band_and_a_huge_face_drops_it(W, H):
    cap = PL.caption_zone((int(H * 0.7), int(H * 0.8)), 0, 10)
    zs = PL.zones_from_face(0.5, 0.4, 0.2, W, H, 0, 10) + [cap]
    for kind in ("lower", "hook", "stat", "entity", "bug", "title"):
        r = PL.choose(kind, min(500, W * 0.4), min(200, H * 0.1), 0, 5, [(0, 5)], zs, None, W, H, PL.safe_box(W, H), caption_top=int(H * 0.7))
        if r:
            assert PL.hard_hit((r["x"], r["y"], r["w"], r["h"]), zs, 0, 5) == 0.0
    huge = PL.zones_from_face(0.5, 0.5, 0.95, W, H, 0, 10)
    r = PL.choose("lower", 400, 150, 0, 5, [(0, 5)], huge + [cap], None, W, H)
    assert r is None or PL.hard_hit((r["x"], r["y"], r["w"], r["h"]), huge + [cap], 0, 5) == 0.0


def test_contrast_helpers_survive_empty_and_off_frame_rects():
    f = np.full((50, 50, 3), 255, np.uint8)
    assert PL.local_contrast(f, (100, 100, 50, 50)) == 21.0
    assert PL.local_contrast(f, (0, 0, 50, 50)) < 1.2
    assert PL.plate_for(1.0)["alpha"] == 1.0 and PL.plate_for(21.0) is None
    assert PL.tone_for([]) == "dark"


def test_the_caption_band_covers_the_whole_video_even_with_many_shots():
    W, H = 1080, 1920
    shots = [(i * 3.0, i * 3.0 + 3.0) for i in range(10)]
    faces = []
    for i, (a, b) in enumerate(shots):
        if i % 2:
            faces += PL.zones_from_face(0.5, 0.8, 0.3, W, H, a, b)
    bands = PL.caption_band(shots, faces, W, H, 0.75, 200)
    assert bands[0][0] == 0.0 and bands[-1][1] == pytest.approx(30.0)
    assert all(abs(p[1] - q[0]) < 1e-6 for p, q in zip(bands, bands[1:]))
