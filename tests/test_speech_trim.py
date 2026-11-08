"""Speech trimming (clipper/speechtrim.py and its wiring): pauses shortened not removed, cuts outside the words, fillers by style, take dedupe, speakers."""

import random

import pytest

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import moments as M
from clipfarm.clipper import speechtrim as SPT
from clipfarm.clipper import styles_profile as SPF
from clipfarm.clipper.engine import repace


def W(text, a, b):
    return {"text": text, "start": a, "end": b}


def _gaps(iv):
    return [round(b2 - a1, 2) for (_, a1), (b2, _) in zip(iv, iv[1:])]


def test_gaps_shrink_by_a_multiplier_with_a_floor_and_sentence_gaps_keep_a_minimum():
    sty = SPF.resolve("longform_talk")
    words = [W("We", 0, .3), W("fit", .35, .6), W("every", 1.6, 1.9), W("foot", 1.95, 2.3),                     # 1.0 s inside a sentence
             W("by", 2.35, 2.5), W("hand.", 2.55, 2.9), W("It", 3.9, 4.0), W("takes", 4.05, 4.4), W("time.", 4.45, 4.8)]   # 1.0 s between sentences
    iv = repace(words, 0, 5, 0.75, style=sty)
    assert len(iv) == 3
    kept = [round(1.0 - (b2 - a1), 2) for (_, a1), (b2, _) in zip(iv, iv[1:])]          # each silence was 1.0 s: what is left of it
    assert all(0 < k < 1.0 for k in kept)                                       # shortened, never to nothing...
    in_sentence, between = kept
    assert in_sentence >= sty["pace"]["gap_shrink_to_s"] - 0.02 and in_sentence == pytest.approx(0.4, abs=0.12)    # ...by a multiplier with a floor
    assert between >= sty["pace"]["gap_min_sentence_s"] - 0.02                                                   # a sentence gap keeps its minimum
    short = [W("A", 0, .3), W("b", .8, 1.0)]                                    # 0.5 s: the speaker's rhythm, not touched
    assert repace(short, 0, 1.2, 0.75, style=sty) == [pytest.approx(x) for x in repace(short, 0, 1.2, 99.0)] or len(repace(short, 0, 1.2, 0.75, style=sty)) == 1


def test_a_question_to_answer_pause_is_kept_in_the_interview_profile():
    words = [W("Where", 0, .3), W("were", .35, .6), W("you?", .65, 1.0), W("At", 2.0, 2.2), W("home.", 2.25, 2.6)]
    iv_int = repace(words, 0, 3, 0.75, style=SPF.resolve("interview_clip"))
    iv_short = repace(words, 0, 3, 0.75, style=SPF.resolve("short_talk"))
    pause_int = sum(_gaps(iv_int)) if len(iv_int) > 1 else 1.0
    pause_short = sum(_gaps(iv_short)) if len(iv_short) > 1 else 1.0
    assert pause_int > pause_short and pause_int >= 0.9                          # the beat before an answer is kept up to a second


def test_pace_words_scale_the_numbers():
    sty = SPF.resolve("longform_talk")
    slow, fast = SPT.pace_numbers(sty, 1.1), SPT.pace_numbers(sty, 0.35)
    assert slow["over"] > fast["over"] and slow["to"] >= fast["to"] >= 0.15


def test_every_cut_is_outside_a_word_by_the_margin():
    rnd = random.Random(7)
    words, t = [], 0.0
    for _ in range(300):
        d = rnd.uniform(0.12, 0.5)
        words.append(W("w", t, t + d))
        t += d + rnd.uniform(0.2, 0.6)                                           # every gap is at least twice the margin
    margin = 0.08
    ivs = []
    for _ in range(200):
        a = rnd.uniform(0, t - 5)
        ivs.append((a, a + rnd.uniform(0.5, 4.0)))
    snapped = SPT.snap_cuts(ivs, words, margin)
    assert len(snapped) == 200
    for a, b in snapped:
        for w in words:
            assert not (w["start"] < a < w["end"]) and not (w["start"] < b < w["end"])          # never inside a word
            assert not (w["start"] - margin + 1e-9 < a < w["start"]) and not (w["end"] < b < w["end"] + margin - 1e-9)    # nor closer than the margin
    assert SPT.snap_cuts([(1.0, 2.0)], [], margin) == [(1.0, 2.0)]


def test_a_cut_in_a_gap_smaller_than_two_margins_goes_to_the_middle():
    words = [W("a", 0, 1.0), W("b", 1.1, 2.0)]
    (a, b), = SPT.snap_cuts([(1.05, 3.0)], words, 0.08)
    assert 1.0 < a < 1.1 and a == pytest.approx(1.05, abs=0.01)


def test_keep_soft_leaves_short_in_sentence_ums():
    words = [W("We", 0, .3), W("um", .4, .6), W("moved", .7, 1.0), W("in", 1.05, 1.2), W("May.", 1.25, 1.6),
             W("Um", 2.0, 2.2), W("then", 2.3, 2.6), W("the", 2.65, 2.8), W("uhh", 2.9, 3.4), W("car", 3.5, 3.8), W("broke.", 3.85, 4.3)]
    cut = [s for s in LP.filler_spans(words, "cut")]
    soft = [s for s in LP.filler_spans(words, "keep_soft")]
    assert len(cut) == 3 and LP.filler_spans(words, "keep") == []
    assert (0.4, 0.6) not in soft and (2.0, 2.2) in soft and (2.9, 3.4) in soft          # the short "um" inside a sentence stays; a sentence start and a long one go
    iv = LP.cut_fillers([(0.0, 4.4)], words, policy="keep_soft")
    assert any(a <= 0.45 and b >= 0.55 for a, b in iv)                                    # the kept "um" is still inside a kept stretch


def test_a_one_word_answer_is_never_dropped_as_a_sliver():
    words = [W("Do", 0, .2), W("you", .25, .4), W("agree?", .45, 1.0), W("Yes.", 2.3, 2.6), W("And", 4.5, 4.7), W("here's", 4.75, 5.0), W("why", 5.05, 5.4)]
    iv = LP.merge_tiny([(0.0, 1.1), (2.25, 2.65), (4.4, 5.5)], words)
    assert any(a <= 2.3 and b >= 2.6 for a, b in iv)                                      # the "Yes." is inside a kept piece
    assert len(iv) < 3                                                                    # and it joined a neighbour across the 1.15 s pause


def test_the_plan_dedupes_takes_to_the_best_one_and_lists_the_others():
    sents = [M.Sentence(0, 0, 3, "We fit every single foot by hand here"), M.Sentence(1, 4, 6, "It takes time."),
             M.Sentence(2, 7, 10, "We fit every single foot by hand here."), M.Sentence(3, 11, 13, "People notice.")]
    alt, groups = [], []
    out = LP.no_repeats([{"kind": "talk", "a": 0, "b": 3}], sents, [], "best", alt, groups)
    assert [(it["a"], it["b"]) for it in out] == [(1, 3)]                                 # the complete later take stays, the unfinished first one goes
    assert alt == [{"group": "t0", "a": 0, "b": 0}] and groups and groups[0]["id"] == "t0"
    out_first = LP.no_repeats([{"kind": "talk", "a": 0, "b": 3}], sents, [], "first")
    assert (out_first[0]["a"], out_first[-1]["b"]) == (0, 1) or all(it["a"] != 2 for it in out_first)       # policy "first" keeps the first take


def test_a_deliberate_repeat_survives_the_dedupe():
    sents = [M.Sentence(0, 0, 4, "Grace is a gift you cannot earn."), M.Sentence(1, 5, 7, "Let me say that again."), M.Sentence(2, 8, 12, "Grace is a gift you cannot earn.")]
    out = LP.no_repeats([{"kind": "talk", "a": 0, "b": 2}], sents, [], "best")
    assert sum(it["b"] - it["a"] + 1 for it in out) == 3                                  # all three sentences stay: the repeat was meant


def test_middle_items_end_on_a_finished_thought_too():
    sents = [M.Sentence(0, 0, 2, "We moved in May."), M.Sentence(1, 2.5, 4, "And then the car broke"), M.Sentence(2, 4.1, 5, "down on the highway."),
             M.Sentence(3, 20, 22, "Next we found a house."), M.Sentence(4, 22.5, 24, "It had a red door."), M.Sentence(5, 30, 31, "The end.")]
    items = LP.finish_sentence([{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 3, "b": 4}], sents, every=True)
    assert items[0]["b"] == 2                                                             # the FIRST item finished its thought as well
    one = LP.finish_sentence([{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 3, "b": 4}], sents)
    assert one[0]["b"] == 1                                                               # (the old behaviour: only the last item)
    blocked = LP.finish_sentence([{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 2, "b": 2}], sents, every=True)
    assert blocked[0]["b"] <= 1                                                           # never into what another item already says (it ends where a sentence did)


def test_an_item_never_opens_on_a_continuation():
    sents = [M.Sentence(0, 0, 2, "And that is why we left."), M.Sentence(1, 2.5, 5, "We moved in May.")]
    items = LP.skip_continuations([{"kind": "talk", "a": 0, "b": 1}], sents)
    assert items[0]["a"] == 1
    lone = LP.skip_continuations([{"kind": "talk", "a": 0, "b": 0}], sents)
    assert lone[0]["a"] == 0                                                              # a single sentence is left as it is


def test_raw_footage_clip_starts_at_the_last_take():
    sents = [M.Sentence(0, 0, 4, "The farm was founded in 1920 by two brothers."), M.Sentence(1, 5, 6, "Again."),
             M.Sentence(2, 8, 12, "The farm was founded in 1920 by two brothers."), M.Sentence(3, 13, 16, "It grew every single year after that.")]
    m = M.Moment(0, 16, 0, 3, 8.0, "x") if "title" not in M.Moment.__dataclass_fields__ else M.Moment(0, 16, 0, 3, 8.0)
    out = M.trim_crew(m, sents)
    assert out is not None and out.first == 2 and out.start == 8


def test_a_cut_never_leaves_its_file():
    words = [W("Hello", 0.0, 0.4), W("there", 0.5, 0.9)]
    (a, b), = SPT.snap_cuts([(0.2, 1.0)], words, 0.08, [(0.0, 100.0)])                    # a start inside the very first word cannot go before the file
    assert a == 0.0 and b >= 0.9
    (a, b), = SPT.snap_cuts([(100.2, 100.8)], [W("x", 100.0, 100.5), W("y", 100.6, 100.9)], 0.08, [(100.0, 200.0)])
    assert a >= 100.0
