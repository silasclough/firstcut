"""Take groups (clipper/takes.py): the same line said more than once, the best take, and the repeats that are meant."""

from clipfarm.clipper import moments as M
from clipfarm.clipper import takes as TK


def S(i, a, b, text):
    return M.Sentence(i, a, b, text)


def W(text, a, b):
    return {"text": text, "start": a, "end": b}


def test_consecutive_similar_sentences_form_a_take_group():
    sents = [S(0, 0, 4, "The farm was founded in 1920 by two brothers."), S(1, 5, 9, "The farm was founded in nineteen twenty by two brothers."),
             S(2, 12, 15, "Then the war came and everything changed.")]
    g = TK.groups(sents, [])
    assert len(g) == 1 and [t["a"] for t in g[0]["takes"]] == [0, 1] and g[0]["deliberate"] is False
    assert g[0]["id"] == "t0" and "farm" in g[0]["line"]


def test_a_director_cue_starts_a_new_take():
    sents = [S(0, 0, 4, "The farm was founded in 1920 by two brothers."), S(1, 5, 6, "Again."),
             S(2, 40, 44, "The farm was founded in 1920 by two brothers.")]            # 36 s later: only a cue between them lets it count
    g = TK.groups(sents, [])
    assert len(g) == 1 and [t["a"] for t in g[0]["takes"]] == [0, 2]
    assert all(t["a"] != 1 for t in g[0]["takes"])                                        # the cue itself is not a take
    far = [S(0, 0, 4, "The farm was founded in 1920 by two brothers."), S(1, 45, 49, "The farm was founded in 1920 by two brothers.")]
    assert TK.groups(far, []) == []                                                       # no cue and 41 s apart: two separate statements


def test_let_me_say_that_again_is_deliberate_and_kept():
    sents = [S(0, 0, 4, "Grace is a gift you cannot earn."), S(1, 5, 7, "Let me say that again."), S(2, 8, 12, "Grace is a gift you cannot earn.")]
    g = TK.groups(sents, [])
    assert len(g) == 1 and g[0]["deliberate"] is True and [t["a"] for t in g[0]["takes"]] == [0, 2]
    refrain = [S(i, i * 100.0, i * 100.0 + 4, "We never walk alone in this valley.") for i in range(4)]       # a refrain, 100 s apart
    g2 = TK.groups(refrain + [S(9, 500, 503, "Thank you all so much for coming tonight.")], [])
    assert len(g2) == 1 and g2[0]["deliberate"] is True and len(g2[0]["takes"]) == 4


def test_best_take_prefers_complete_clean_well_recorded_and_later():
    sents = [S(0, 0, 3, "The reactor was shut down on"), S(1, 5, 9, "The reactor was shut down on Tuesday morning."), S(2, 12, 16, "The reactor was shut down on Tuesday morning.")]
    words = [W("um", 6.0, 6.3), W("um", 6.4, 6.7)] + [W("x", 13.0, 13.1)]
    g = TK.groups(sents, words)[0]
    assert g["takes"][0]["complete"] is False and g["takes"][1]["fillers"] == 2 and g["takes"][2]["fillers"] == 0
    assert TK.best(g) == 2                                                                # complete, clean and the latest
    g["takes"][1]["fillers"] = 0                                                          # take 1 is now clean as well: sound and picture decide
    g["takes"][2]["sound"] = "bad"
    assert TK.best(g) == 1                                                                # a badly recorded take loses to an equally clean, well recorded one
    g["takes"][2]["sound"] = "good"
    g["takes"][2]["blur_ok"] = False
    assert TK.best(g) == 1                                                                # a blurred picture counts against a take


def test_policy_last_and_first_override_the_score():
    sents = [S(0, 0, 4, "The reactor was shut down on Tuesday morning."), S(1, 6, 9, "The reactor was shut down on")]
    g = TK.groups(sents, [])[0]
    assert TK.best(g) == 0 and TK.best(g, "last") == 1 and TK.best(g, "first") == 0


def test_unchosen_takes_are_listed_as_alternates():
    sents = [S(0, 0, 4, "The farm was founded in 1920 by two brothers."), S(1, 5, 9, "The farm was founded in 1920 by two brothers."),
             S(2, 10, 14, "The farm was founded in 1920 by two brothers.")]
    gs = TK.groups(sents, [])
    keep = TK.best(gs[0])
    alts = TK.alternates(gs, {gs[0]["id"]: keep})
    assert sorted(a["a"] for a in alts) == sorted(t["a"] for n, t in enumerate(gs[0]["takes"]) if n != keep) and all(a["group"] == "t0" for a in alts)
    assert TK.alternates(gs, {}) == []                                                    # a group nobody used lists nothing


def test_a_restarted_sentence_is_a_take_that_lost():
    sents = [S(0, 0, 2, "So the farm was founded um"), S(1, 3, 8, "So the farm was founded in nineteen twenty by two brothers.")]
    g = TK.groups(sents, [])[0]
    assert g["takes"][0]["restart"] is True and TK.best(g) == 1


def test_short_sentences_and_unrelated_lines_make_no_group():
    sents = [S(0, 0, 1, "Yes."), S(1, 2, 3, "Yes."), S(2, 5, 9, "The weather turned cold in November."), S(3, 10, 14, "Nobody expected the early snow.")]
    assert TK.groups(sents, []) == []
