"""WP10: which graphic, what text, when. Pure code and a fake AI (offline): every card is found or checked against the words that were said."""

import pytest

from clipfarm.clipper import gfxcards as G
from clipfarm.clipper import gfxcontent as GC
from clipfarm.clipper import gfxdetect
from clipfarm.clipper import gfxplan as GP
from clipfarm.clipper import moments as M
from clipfarm.clipper import scripture
from clipfarm.clipper import styles_profile as SP


def talk(text, t0=10.0, dt=0.4):
    """A transcript with a timed word every 0.4 s: -> (words, sentences)."""
    words, t = [], t0
    for w in text.split():
        words.append({"text": w, "start": round(t, 3), "end": round(t + 0.3, 3)})
        t += dt
    return words, M.sentences_from_words(words)


class FakeAI:
    def __init__(self, items=None):
        self.items = items or []
        self.calls = 0

    def complete_json(self, *a, **k):
        self.calls += 1
        return type("R", (), {"data": {"graphics": self.items}})()


def style(policy="informational", **g):
    return SP.resolve("longform_talk", {"graphics": {"policy": policy, **g}})


PROFILE = {"kinds": ["lower", "stat", "quote", "slide", "chapter", "reference"], "graphics_per_s": 20, "label": "story"}
SAMPLE = ("Hello everyone. I'm Sarah Miller, the director of the clinic. Last year twenty-three percent of students dropped out. "
          "There are three reasons. First, people need hope. Second, they need a friend. Third, they need a plan. "
          "Winston Churchill said, never give in, never give in, never, never, never. Turn to John 3:16 which says for God so loved the world that he gave his only son. "
          "And that is the whole story of the year at our clinic and what we learned from it.")


# --------------------------------------------------------------------------------------------------- 1. found by code first
def test_numbers_names_lists_and_quotes_are_found_by_code_before_the_ai():
    words, sents = talk(SAMPLE)
    types = {c["type"] for c in gfxdetect.find(sents, words) if c.get("type")}
    assert {"number", "name", "list", "quote", "reference"} <= types
    ai = FakeAI([])                                                                         # the AI proposes nothing: the cards still exist
    cards = GP.plan(ai, sents, "", PROFILE, 80.0, hook_end=0.0, words=words, style=style(max_per_minute=20))
    by = {c["kind"]: c for c in cards}
    assert by["stat"]["value"] == "23%" and "dropped out" in by["stat"]["label"]            # "twenty-three percent" -> 23%
    assert by["lower"]["name"] == "Sarah Miller" and "director" in by["lower"]["role"]
    assert by["slide"]["bullets"] == ["people need hope", "they need a friend", "they need a plan"] and by["slide"]["title"] == "Three reasons"
    assert by["quote"]["who"] == "Winston Churchill" and by["quote"]["text"].startswith("never give in")
    assert by["reference"]["ref"] == "John 3:16" and by["reference"]["passage"].startswith("for God so loved")
    assert all(c.get("rule") for c in cards)                                                # every card names the rule that made it
    md = GP.check_notes(cards, sents)
    assert "rule: number 23% said at" in md and "rule: name Sarah Miller first said at" in md


def test_a_year_is_never_a_number_and_small_numbers_are_spelled_out():
    assert GC.display_number("twenty twenty-six") is None and GC.display_number("nineteen ninety-nine") is None
    assert GC.display_number("two thousand twenty") is None or GC.display_number("2020") is None
    assert GC.display_number("twenty-three", "percent") == "23%" and GC.display_number("40", "%") == "40%"
    assert GC.display_number("five", "million", True) == "$5M" and GC.display_number("1.2", "million", True) == "$1.2M"
    assert [GC.spell_small(n) for n in (1, 5, 9, 10, 23)] == ["one", "five", "nine", "10", "23"]
    words, sents = talk("In twenty twenty-six we opened. Then in 2019 we grew. It cost two thousand dollars to start.")
    nums = GC.number_candidates(words, sents)
    assert all(c["item"]["value"] not in ("46", "2,026", "2,019") for c in nums)            # the years were not turned into numbers


def test_a_name_must_be_an_introduction_and_is_shown_once():
    words, sents = talk("Hello. My name is Pastor John Davis and I'm happy to be here. I'm Happy about it. Later John Davis came back. I'm John Davis again.")
    names = GC.name_candidates(words, sents)
    assert [n["item"]["name"] for n in names] == ["John Davis"]                              # first mention only; "I'm Happy" is not a name
    assert names[0]["item"]["role"] in ("Pastor", "")


# --------------------------------------------------------------------------------------------------- 2. quotes
def test_a_quote_is_never_paraphrased_and_a_long_one_shows_only_the_spoken_clause():
    said = "He said, we hold these truths to be self evident, that all people are created equal, that they are endowed with certain rights, and that is where we begin today."
    words, sents = talk(said)
    spoken = " ".join(s.text for s in sents)
    # reworded by the AI: not said word for word -> no card
    assert GP.validate({"kind": "quote", "at": 10.5, "text": "all humans are born equal", "who": "He"}, spoken, sents, 80.0, 0.0, words) is None
    # the AI left a word out and capitalised: the card shows the transcript's own words
    v = GP.validate({"kind": "quote", "at": 10.5, "text": "That all people are created equal", "who": "He"}, spoken, sents, 80.0, 0.0, words)
    assert v and v["text"] == "that all people are created equal"
    # a quote of more than 21 words shows only the clause being spoken (split at a comma), never a smaller type
    long_text = "we hold these truths to be self evident, that all people are created equal, that they are endowed with certain rights, and that is where we begin today"
    assert len(long_text.split()) > GC.QUOTE_MAX_WORDS
    v2 = GP.validate({"kind": "quote", "at": 10.5, "text": long_text, "who": "He"}, spoken, sents, 80.0, 0.0, words)
    assert v2 and len(v2["text"].split()) <= GC.QUOTE_MAX_WORDS and v2["text"] in " ".join(w["text"] for w in words)
    assert GC.verbatim_text("we hold these truths", words) == "we hold these truths"
    assert GC.verbatim_text("we hold some truths", words) is None
    filler = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate("you could um feel the uh love".split())]
    assert GC.verbatim_text("you could feel the love", filler) == "you could feel the love"        # only fillers may go


# --------------------------------------------------------------------------------------------------- 3. scripture
def test_scripture_references_are_found_with_their_passage_and_two_john_in_prose_is_not_one():
    def refs(t):
        w, s = talk(t)
        return scripture.find(w, s)
    r = refs("Turn to John 3:16 which says for God so loved the world that he gave his only son.")
    assert r and r[0]["ref"] == "John 3:16" and r[0]["passage"].startswith("for God so loved")
    r = refs("Romans eight twenty-eight tells us that all things work together for good.")
    assert r and r[0]["ref"] == "Romans 8:28"
    r = refs("First Corinthians 13 says love is patient love is kind.")
    assert r and r[0]["ref"] == "1 Corinthians 13"
    r = refs("Psalm 23 reads The Lord is my shepherd I shall not want.")
    assert r and r[0]["ref"] == "Psalms 23"
    assert refs("I read 2 John last week and loved it.") == []                              # a book with no chapter
    assert refs("We were in John 5 times this year and Mark 3 of us came.") == []             # no cue, nothing read
    assert refs("He studied Mark 3 with the group yesterday afternoon at church.") == []
    words, sents = talk("Turn to John 3:16 which says for God so loved the world that he gave his only son.")
    spoken = " ".join(s.text for s in sents)
    v = GP.validate({"kind": "reference", "at": 11.0, "ref": "John 3:16", "passage": "for God so loved the world"}, spoken, sents, 60.0, 0.0, words)
    assert v and v["ref"] == "John 3:16" and v["passage"] and v["len"] == pytest.approx(3.0 + 0.3 * len(v["passage"].split()), abs=0.01)
    assert GP.validate({"kind": "reference", "at": 11.0, "ref": "Romans 8:28"}, spoken, sents, 60.0, 0.0, words) is None      # not said
    assert GP.validate({"kind": "reference", "at": 11.0, "ref": "John 3:16", "passage": "invented words that nobody read aloud"}, spoken, sents, 60.0, 0.0, words)["passage"] == ""


def test_a_reference_card_draws_two_lines_inside_the_frame():
    for W, H in ((1080, 1920), (1920, 1080)):
        sc = G.build_scene("reference", G.Look("glass", W, H), {"kind": "reference", "ref": "John 3:16", "passage": "for God so loved the world that he gave his only son"}, 5.0)
        assert sc is not None and sc.panel
        x, y, w, h = sc.panel
        assert x >= 0 and y >= 0 and x + w <= W and y + h <= H
        assert G.frame_at(sc, 2.0).getchannel("A").getbbox() is not None


# --------------------------------------------------------------------------------------------------- 4. durations
def test_durations_follow_the_word_count_formulas_and_timed_cards_are_never_clipped():
    stat = GC.retime({"kind": "stat", "at": 20.0, "value": "23%", "label": "of students dropped out", "len": 4.2}, style())
    assert stat["len"] == pytest.approx(GC.clamp(1.0 + 0.4 * 5, 2.0, 4.0), abs=0.01)
    q = GC.retime({"kind": "quote", "at": 20.0, "text": "never give in never give in never", "len": 2.6}, style())
    assert q["len"] == pytest.approx(GC.clamp(1.0 + 0.33 * 7, 3.0, 7.0), abs=0.01)
    assert GC.retime({"kind": "lower", "at": 5.0, "name": "A B", "role": "Director"}, style())["len"] == 4.5
    assert GC.retime({"kind": "lower", "at": 5.0, "name": "A B"}, style())["len"] == 3.0       # a name alone is shorter
    assert GC.retime({"kind": "lower", "at": 5.0, "name": "A B", "role": "Director"}, style(lower_third_s=5.5))["len"] == 5.5
    assert GC.retime({"kind": "chapter", "at": 30.0, "text": "x y"}, style())["len"] == 3.4
    listed = GC.retime({"kind": "slide", "at": 10.0, "len": 6.0, "times": [1.0, 4.0, 9.0]}, style())
    assert listed["len"] >= 9.0 + 1.5                                                           # holds after the last item is said
    assert GC.retime({"kind": "stat", "at": 58.0, "value": "9", "label": "x", "len": 4.0}, style(), 60.0)["len"] <= 60.0 - 58.0 - 0.6 + 1e-6
    # the old defect (a slow speaker, the last steps squeezed into the final second): validate and schedule keep the last step
    names = ["apple", "banana", "cherry", "damson", "elder", "fig", "grape", "honey"]
    parts = ["Two buyers."] + [f"Then {n} arrives." for n in names]
    words, t = [], 10.0
    for p in parts:
        for w in p.split():
            words.append({"text": w, "start": t, "end": t + 0.3})
            t += 0.9
        t += 1.5
    spoken = " ".join(parts)
    r = GP.validate({"kind": "twocol", "at": 10.0, "titles": ["A", "B"], "steps": [names[:4], names[4:]], "tag": {"side": 1, "text": "Out"}},
                    spoken, [M.Sentence(0, 10.0, t, spoken)], 200.0, 3.2, words)
    assert r and ("times" not in r or r["len"] >= r["times"][-1] + 1.0)
    two = [{"kind": "stat", "at": 20.0, "len": 4.2, "value": "5", "label": "x"}, {"kind": "slide", "at": 22.0, "len": 12.0, "times": [1.0, 5.0, 11.0], "title": "t", "bullets": ["a", "b", "c"]}]
    a, b = GP.schedule(two, 100)
    assert b["len"] >= b["times"][-1] + 1.0 - 1e-6                                              # shortened to make room, never below the last item + 1 s


# --------------------------------------------------------------------------------------------------- 5. timing
def test_cards_enter_three_frames_before_the_word_and_never_after_it():
    assert GC.lead_s(24.0) == pytest.approx(0.125) and GC.lead_s(30.0) == pytest.approx(0.1)
    words, sents = talk("Last year the number was twenty-three percent of all students here in the program.", t0=10.0)
    for c in GC.number_candidates(words, sents, 24.0):
        assert c["said_at"] - 0.1251 <= c["at"] <= c["said_at"] + 1e-9
    said = next(w["start"] for w in words if w["text"].startswith("twenty"))
    assert GC.entrance(said, sents, 0.0, 24.0) == pytest.approx(said - 0.125, abs=0.002)
    # never earlier than the end of the sentence before: a word that starts a sentence right after another has ended
    two = [M.Sentence(0, 0.0, 5.0, "Before."), M.Sentence(1, 5.05, 9.0, "Twenty percent of us.")]
    at = GC.entrance(5.1, two, 0.0, 24.0)
    assert 5.0 - 1e-9 <= at <= 5.1
    # exits on the next sentence end if it is close
    card = {"kind": "stat", "at": 10.0, "len": 3.5, "value": "5", "label": "x"}
    snapped = GC.snap_exit(card, [M.Sentence(0, 9.0, 13.9, "x")], None, 60.0)
    assert snapped["len"] == pytest.approx(3.9, abs=0.01)
    assert GC.snap_exit(card, [M.Sentence(0, 9.0, 20.0, "x")], None, 60.0)["len"] == 3.5     # far from a boundary: unchanged
    assert GC.snap_exit(card, [], [13.7], 60.0)["len"] == pytest.approx(3.7, abs=0.01)          # a shot change within half a second
    assert G.ENTER == pytest.approx(0.3) and G.EXIT == pytest.approx(0.23)


# --------------------------------------------------------------------------------------------------- 6. budget, gaps, redundancy
def test_at_most_six_cards_a_minute_one_at_a_time_and_no_card_that_repeats_the_caption():
    items = [{"kind": "stat", "at": 12.0 + 9 * i, "value": str(i + 2) + "%", "label": f"of group {i}"} for i in range(8)]
    text = " ".join(f"In group {i} about {i + 2} percent of group {i} said yes." for i in range(8))
    words, sents = talk(text, t0=10.0, dt=0.5)
    st = style("informational", max_per_minute=6)
    cards = GP.plan(FakeAI(items), sents, "", {"kinds": ["stat"], "graphics_per_s": 5, "label": "x"}, 60.0, hook_end=0.0, words=words, style=st)
    assert 1 <= len(cards) <= 6
    assert all(a["at"] + a["len"] + 1.5 <= b["at"] + 0.011 for a, b in zip(cards, cards[1:]))   # one at a time with the style's gap
    assert GC.cap_per_video(st, 120.0, 99) == 12 and GC.cap_per_video(None, 120.0, 7) == 7
    # a card that is just the caption's own words is dropped when captions are on
    w2, s2 = talk("We love this place so much that we all came back again and again.", t0=10.0)
    twin = {"kind": "statement", "at": 10.2, "text": "love this place"}
    assert GC.repeats_caption(twin, w2, True) and not GC.repeats_caption(twin, w2, False)
    assert not GC.repeats_caption({"kind": "stat", "at": 10.2, "value": "47%", "label": "of the moms"}, w2, True)


# --------------------------------------------------------------------------------------------------- 7. chapters
def test_chapter_cards_sit_on_outline_steps_at_least_ten_seconds_apart():
    marks = [{"n": 1, "title": "Welcome and who we are", "at": 4.0}, {"n": 2, "title": "The problem: why students leave early in the first year", "at": 20.0},
             {"n": 3, "title": "Too close", "at": 25.0}, {"n": 4, "title": "What we do about it", "at": 70.0}, {"n": 5, "title": "The very end", "at": 118.0}]
    cards = GC.chapter_cards(marks, 120.0, 10.0, hook_end=3.2)
    assert [c["text"] for c in cards] == ["Welcome and who we are", "The problem", "What we do about it"]
    assert all(b["at"] - a["at"] >= 10.0 for a, b in zip(cards, cards[1:])) and all(c["number"] and c["len"] == 3.4 for c in cards)
    assert GC.shorten_title("one two three four five six seven eight") == "one two three four five six"
    flow = [{"n": 1, "step": "Welcome"}, {"n": 2, "step": "The gospel"}]
    items = [{"kind": "talk", "a": 0, "b": 1, "step": 1}, {"kind": "talk", "a": 5, "b": 6, "step": 1}, {"kind": "talk", "a": 9, "b": 10, "step": 2}]
    got = GC.chapter_marks(flow, items, lambda a: float(a) * 2.0, lambda t: t + 1.0)
    assert got == [{"n": 1, "title": "Welcome", "at": 1.0}, {"n": 2, "title": "The gospel", "at": 19.0}]
    words, sents = talk("x " * 200, t0=0.5, dt=0.5)
    ai = FakeAI([{"kind": "chapter", "at": 50.0, "text": "An AI chapter that must be replaced"}])
    st = SP.resolve("seminar")
    plan = GP.plan(ai, sents, "", {"kinds": ["chapter", "reference", "lower"], "graphics_per_s": 20, "label": "seminar"}, 120.0, hook_end=3.2, words=words,
                   style=st, chapters=[{"n": 1, "title": "Welcome", "at": 6.0}, {"n": 2, "title": "The gospel", "at": 60.0}])
    chap = [c for c in plan if c["kind"] == "chapter"]
    assert [c["text"] for c in chap] == ["Welcome", "The gospel"]                              # the outline replaces the AI's own chapter choice


# --------------------------------------------------------------------------------------------------- 8. the genre gate
def test_the_genre_gate_gives_an_interview_only_a_name():
    interview = SP.resolve("interview_clip")
    assert interview["graphics"]["policy"] == "none"
    allk = ["lower", "stat", "quote", "slide", "chapter", "reference", "bigstat", "statement", "search"]
    assert GC.allowed_kinds(allk, interview) == ["lower"]
    assert GC.allowed_kinds(allk, SP.resolve("seminar")) == ["lower", "chapter", "reference"]
    assert "stat" in GC.allowed_kinds(allk, style("informational")) and "statement" not in GC.allowed_kinds(allk, style("informational"))
    assert GC.allowed_kinds(allk, style("all")) == allk and GC.allowed_kinds(allk, None) == allk
    words, sents = talk(SAMPLE)
    cards = GP.plan(FakeAI([{"kind": "stat", "at": 14.0, "value": "23%", "label": "of students dropped out"}]), sents, "",
                    {"kinds": ["lower", "stat", "quote", "slide", "reference"], "graphics_per_s": 20, "label": "interview"}, 80.0, hook_end=0.0, words=words, style=interview)
    assert [c["kind"] for c in cards] == ["lower"] and cards[0]["name"] == "Sarah Miller"


def test_scripture_cues_before_the_reference_and_and_it_says():
    from clipfarm.clipper import scripture as S

    def run(text):
        ws = [{"text": w, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, w in enumerate(text.split())]
        return [r["ref"] for r in S.find(ws)]
    assert run("Paul writes in 2 Corinthians 5:17 that anyone in Christ is a new creation the old has gone the new has come") == ["2 Corinthians 5:17"]
    assert run("I read John 3 verse 16 and it says for God so loved the world that he gave his one and only Son") == ["John 3:16"]
    assert run("According to Romans 8:28 all things work together for good for those who love him") == ["Romans 8:28"]
    assert run("I read 2 John last night") == [] and run("John 5 times in a row") == [] and run("Mark 3 of them came to the door yesterday afternoon") == []
    assert run("he went there in 2 Corinthians 5 weeks ago") == []


def test_finding_cards_in_a_two_hour_transcript_takes_under_a_second():
    import random
    import time
    from clipfarm.clipper import gfxcontent as G
    rnd = random.Random(1)
    vocab = "first second third okay so the we and three reasons because people God love two big points number one then also John Romans said says once 23 percent million".split()
    words = [{"text": rnd.choice(vocab), "start": i * 0.35, "end": i * 0.35 + 0.3} for i in range(20000)]
    t0 = time.time()
    G.candidates([], words)
    assert time.time() - t0 < 1.0
