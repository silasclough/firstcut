from clipfarm.core.captions import Word, build_timeline, clean_words, group_words, to_srt
from clipfarm.core.tts import align_to_text, estimate_word_timings


def W(text, s, e):
    return Word(text, s, e)


def test_clean_words_sorts_and_removes_overlaps():
    ws = clean_words([W("b", 1.0, 1.6), W("a", 0.0, 1.2), W("  ", 2.0, 2.1), W("c", 1.5, 1.4)])
    assert [w.text for w in ws] == ["a", "b", "c"]
    for prev, nxt in zip(ws, ws[1:]):
        assert prev.end <= nxt.start + 1e-9
    assert all(w.end > w.start for w in ws)


def test_grouping_respects_max_words_and_sentence_ends():
    ws = [W("I", 0, .1), W("found", .1, .4), W("a", .4, .5), W("sign.", .5, .9), W("It", 1.0, 1.1),
          W("said", 1.1, 1.4), W("run", 1.4, 1.7), W("now", 1.7, 2.0)]
    groups = group_words(ws, max_words=3, max_chars=40)
    assert [g.text for g in groups] == ["I found a", "sign.", "It said run", "now"]


def test_grouping_breaks_on_pauses_and_char_limit():
    ws = [W("hello", 0, .3), W("there", 1.2, 1.5), W("extraordinarily", 1.5, 2.2), W("long", 2.2, 2.4)]
    groups = group_words(ws, max_words=3, max_chars=12, pause_break=0.35)
    assert [g.text for g in groups] == ["hello", "there", "extraordinarily", "long"]


def test_timeline_is_contiguous_and_covers_total():
    ws = [W("one", .5, .8), W("two", .8, 1.1), W("three", 1.1, 1.5), W("four", 3.0, 3.4)]
    groups = group_words(ws, max_words=3)
    tl = build_timeline(groups, total=4.0)
    assert tl[0].start == 0 and tl[0].group is None          # leading silence is blank
    assert abs(tl[-1].end - 4.0) < 1e-9                      # trailing blank to the end
    for a, b in zip(tl, tl[1:]):
        assert abs(a.end - b.start) < 1e-9                   # no gaps, no overlaps
    active = [(f.group, f.active) for f in tl if f.group is not None]
    assert active == [(0, 0), (0, 1), (0, 2), (1, 0)]


def test_word_highlight_starts_exactly_at_word_start():
    ws = [W("alpha", 0.25, 0.6), W("beta", 0.6, 1.0), W("gamma", 1.0, 1.4)]
    tl = build_timeline(group_words(ws, max_words=3), total=1.4)
    starts = {(f.group, f.active): f.start for f in tl if f.group is not None}
    assert starts[(0, 0)] == 0.25 and starts[(0, 1)] == 0.6 and starts[(0, 2)] == 1.0


def test_short_pause_holds_caption_long_pause_blanks():
    ws = [W("a", 0, .3), W("b.", .3, .6), W("c", .7, 1.0), W("d.", 1.0, 1.3), W("e", 2.5, 2.8)]
    groups = group_words(ws, max_words=3)
    tl = build_timeline(groups, total=3.0, hold_gap=0.3)
    # 0.1s gap between group 0 and 1 is held (no blank); the 1.2s gap before "e" is blank
    blanks = [(round(f.start, 2), round(f.end, 2)) for f in tl if f.group is None]
    assert (1.3, 2.5) in blanks
    assert not any(0.55 < b[0] < 0.75 for b in blanks)


def test_tiny_frames_are_merged():
    ws = [W("a", 0.0, 0.01), W("b", 0.01, 0.5)]
    tl = build_timeline(group_words(ws, max_words=3), total=0.5, min_frame=1 / 30)
    assert all(f.duration >= 1 / 30 - 1e-9 for f in tl)
    assert abs(sum(f.duration for f in tl) - 0.5) < 1e-9


def test_srt_format():
    srt = to_srt(group_words([W("hi", 0.0, 0.5), W("there.", 0.5, 1.25)], max_words=3))
    assert "00:00:00,000 --> 00:00:01,250" in srt and "hi there." in srt


def test_align_restores_punctuation():
    timed = [{"text": "Hello", "start": 0, "end": .3}, {"text": "world", "start": .3, "end": .6},
             {"text": "its", "start": .7, "end": .9}, {"text": "me", "start": .9, "end": 1.0}]
    out = align_to_text("Hello, world! It's me.", timed)
    assert [w["text"] for w in out] == ["Hello,", "world!", "its", "me."]


def test_estimated_timings_monotonic_and_bounded():
    words = estimate_word_timings("This is a test. Another sentence, with commas.", 5.0)
    assert len(words) == 8
    assert words[0]["start"] >= 0 and words[-1]["end"] <= 5.0
    for a, b in zip(words, words[1:]):
        assert a["end"] <= b["start"] + 1e-6
