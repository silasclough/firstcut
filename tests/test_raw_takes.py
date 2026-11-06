"""Short clips from raw footage with takes (work Mac, 2026-11-06: Tamsin Thorne Delacroix C1234 became a joke between takes and a blooper)."""
from clipfarm.clipper import moments as M


def S(i, a, b, t):
    return M.Sentence(i, a, b, t)


TAKES = [S(0, 5.0, 5.8, "Perfect."), S(1, 5.8, 6.8, "Okay."), S(2, 6.8, 7.5, "Ready?"), S(3, 7.5, 8.1, "Yes."),
         S(4, 12.3, 16.0, "At Northfield, making things isn't just a part of campus."), S(5, 16.0, 17.6, "It's the foundation."),
         S(6, 17.6, 23.4, "From the studio to classroom discussions, my education has centered on real projects and mentors."),
         S(7, 29.3, 31.5, "I like that we messed up on this first version."), S(8, 49.4, 50.0, "Go ahead."),
         S(9, 50.3, 53.5, "At Northfield, making things isn't just a part of campus."), S(10, 53.5, 54.9, "It's the foundation."),
         S(11, 54.9, 60.7, "From the studio to classroom discussions, my education has centered on real projects and mentors."),
         S(12, 60.7, 61.5, "That was a good one."), S(13, 61.5, 62.4, "And go ahead."), S(14, 62.4, 63.0, "One more.")]


def test_takes_and_crew_talk_are_recognised():
    assert M.raw_takes(TAKES)
    assert all(M.crew_talk(t) for t in ("Okay.", "Ready?", "That was a good one.", "And go ahead.", "One more time.", "Okay. Ready? Yes."))
    assert not any(M.crew_talk(t) for t in ("It's the foundation.", "I think I hit a wire.", "Okay, so here is what happened at camp."))
    talk = [S(i, i * 5.0, i * 5.0 + 4, f"Sentence number {i} about something different each time, number {i}.") for i in range(20)]
    assert not M.raw_takes(talk)                                                       # a talk is not takes


def test_crew_talk_is_trimmed_off_both_ends():
    m = M.trim_crew(M.Moment(0, 0, 8, 12), TAKES)
    assert (m.first, m.last, m.start, m.end) == (9, 11, 50.3, 60.7)
    assert M.trim_crew(M.Moment(0, 0, 0, 3), TAKES) is None


def test_the_ai_is_told_and_a_clean_take_is_not_padded():
    seen = {}

    class R:
        provider = "fake"
        data = {"clips": [{"first": 8, "last": 12, "title": "Faith is the foundation at Northfield", "score": 7}]}

    class L:
        def complete_json(self, task, system, prompt, schema, validate=None):
            seen["prompt"] = prompt
            return R()
    out = M.llm_moments(L(), TAKES, 3, 20.0, 35.0)
    assert "RAW FOOTAGE" in seen["prompt"] and "7-35 seconds" in seen["prompt"]
    assert len(out) == 1 and (out[0].first, out[0].last) == (9, 11)                     # the take, not the crew around it, not padded to 20 s
