"""Animation graphic edge cases (tester): end of video, crowding, stop-word concepts, missing words / sentences, scheduling."""
from types import SimpleNamespace

import pytest

from clipfarm.clipper import gfxplan as GP
from clipfarm.clipper import localproj as LP

from test_animation_graphics import FakeAI, item, sents, words_of  # noqa: F401


def spoken(ss):
    return " ".join(s.text for s in ss)


def test_words_missing_none_or_without_timing_give_an_untimed_diagram_not_an_error():
    ss = sents()
    for w in (None, [], [{"text": "x"}]):
        v = GP.validate(item(), spoken(ss), ss, 30.0, 3.2, w)
        assert v is None or v["timed"] == []
    v = GP.validate(item(), spoken(ss), ss, 30.0, 3.2, None)
    assert v and v["timed"] == [] and v["narration"]


def test_sentences_empty_or_bad_at_values_are_dropped_quietly():
    ss = sents()
    assert GP.validate(item(), "", [], 30.0, 3.2, None) is None
    for at in (None, "soon", -5, 1e9, "inf"):
        r = GP.validate(item(at=at), spoken(ss), ss, 30.0, 3.2, words_of(ss))
        assert r is None or 3.4 <= r["at"] < 30.0


def test_a_concept_made_only_of_stop_words_or_short_words_is_dropped():
    ss = sents()
    assert GP.validate(item(concept="the a an and or of to in on for with is are was it that this we you"), spoken(ss), ss, 30.0, 3.2, None) is None
    assert GP.validate(item(concept="it is on to go at it up so we do"), spoken(ss), ss, 30.0, 3.2, None) is None
    assert GP.validate(item(concept=None), spoken(ss), ss, 30.0, 3.2, None) is None
    assert GP.validate(item(concept=["a", "list"]), spoken(ss), ss, 30.0, 3.2, None) is None or True       # (wrong type must not raise)


def test_a_very_long_concept_is_cut_to_240_characters():
    ss = sents()
    v = GP.validate(item(concept="selector fork sleeve teeth gear " * 40), spoken(ss), ss, 30.0, 3.2, None)
    assert v is None or len(v["concept"]) <= 240


def test_an_animation_right_at_the_end_is_dropped_and_one_that_just_fits_is_cut_to_the_room_left():
    ss = sents()
    ws = words_of(ss)
    assert GP.validate(item(at=24.5), spoken(ss), ss, 28.0, 3.2, ws) is None
    v = GP.validate(item(at=6.2), spoken(ss), ss, 14.6, 3.2, ws)                              # room: 14.6 - 6.2 - 0.8 = 7.6 s
    assert v is None or v["at"] + v["len"] <= 14.6 - 0.6


def test_the_animation_never_starts_inside_the_title_card():
    ss = sents()
    v = GP.validate(item(at=0.0), spoken(ss), ss, 30.0, 3.2, words_of(ss))
    assert v is None or v["at"] >= 3.4


def test_two_animations_closer_than_14s_keep_only_the_first():
    ss = sents()
    prof = {**LP.PROFILES["youtube"], "kinds": ["animation"]}
    cards = GP.plan(FakeAI(), ss, "", prof, 150.0, words=words_of(ss))
    ans = [c for c in cards if c["kind"] == "animation"]
    assert all(b["at"] - a["at"] >= 14.0 for a, b in zip(ans, ans[1:]))
    assert len(ans) <= 3


def test_budget_is_one_per_75_seconds_up_to_three_and_at_least_one():
    ss = sents()
    prof = {**LP.PROFILES["youtube"], "kinds": ["animation"], "graphics_per_s": 1}
    seen = []

    class AI:
        def complete_json(self, task, system, prompt, schema, **kw):
            seen.append(prompt)
            return SimpleNamespace(data={"graphics": []})
    for out_len, n in ((20.0, 1), (150.0, 2), (900.0, 3)):
        GP.plan(AI(), ss, "", prof, out_len, words=words_of(ss))
        assert f"at most {n} 'animation'" in seen[-1], (out_len, seen[-1][:200])


def test_animation_not_in_the_profile_kinds_is_ignored_even_if_the_ai_returns_one():
    ss = sents()
    prof = {**LP.PROFILES["youtube"], "kinds": ["slide"]}
    assert all(c["kind"] != "animation" for c in GP.plan(FakeAI(), ss, "", prof, 150.0, words=words_of(ss)))


def anim(at, ln):
    return {"kind": "animation", "at": at, "len": ln, "concept": "c " * 6, "narration": "n", "timed": [{"text": "w", "start": 1.0}], "context": ""}


def test_schedule_keeps_animations_apart_by_the_gap_and_inside_the_video():
    out = GP.schedule([anim(10, 12), anim(15, 12)], 60.0)
    assert len(out) == 2 and out[1]["at"] >= out[0]["at"] + out[0]["len"] + GP.GAP - 0.01
    assert GP.schedule([anim(10, 12), anim(15, 12)], 24.0)[-1]["at"] + 12 <= 24.0 or len(GP.schedule([anim(10, 12), anim(15, 12)], 24.0)) == 1


def test_schedule_with_empty_list_and_zero_length_cards():
    assert GP.schedule([], 30.0) == []
    assert len(GP.schedule([anim(5, 0.0)], 30.0)) == 1


@pytest.mark.xfail(strict=True, reason="BUG (concern): schedule() delays a colliding animation by up to 4.5 s but leaves its 'timed' words relative to the OLD start (it only shifts 'times'), so the diagram is drawn 'need' seconds early against the voice; "
                   "it also shortens an earlier animation to 60% of its length while its narration still covers the full original length")
def test_a_delayed_animation_keeps_its_words_in_step_with_the_voice():
    out = GP.schedule([anim(10, 12), anim(15, 12)], 60.0)
    shift = out[1]["at"] - 15
    assert shift > 0 and out[1]["timed"][0]["start"] == pytest.approx(1.0 - shift, abs=0.01)


@pytest.mark.xfail(strict=True, reason="BUG (low): at=NaN (Python's json accepts NaN) passes max(at, hook_end+0.2) unchanged and the animation lands at 0.2 s, inside the title card")
def test_a_nan_start_time_is_dropped():
    ss = sents()
    r = GP.validate(item(at=float("nan")), spoken(ss), ss, 30.0, 3.2, words_of(ss))
    assert r is None or r["at"] >= 3.4
