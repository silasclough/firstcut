"""Instructions you can check (constraints.py, transcript_search.py, WP5): notes -> typed constraints with quoted sources -> anchors in the transcript and
the footage facts -> negatives as filters -> the validator -> the repair budget -> the approval decisions -> the checklist -> the post-render check."""
import json

import pytest

from clipfarm.clipper import constraints as CN
from clipfarm.clipper import moments as M
from clipfarm.clipper import transcript_search as TS

TEXTS = [
    "Welcome everyone to our summer camp week",                       # 0
    "Today we talk about forgiveness and about grace",                # 1
    "Forgiveness is a gift we give away freely",                      # 2
    "The river flooded our camp on Tuesday night",                    # 3
    "Nobody was hurt and everyone helped each other",                 # 4
    "We were wrong about the schedule and we say sorry",              # 5
    "Later that week we sang together by the fire",                   # 6
    "Let us pray for the families who lost their homes",              # 7
    "Thank you all for coming this summer",                           # 8
]


def _words(texts=TEXTS, gap=1.4):
    out, t = [], 0.0
    for tx in texts:
        for w in tx.split():
            out.append({"text": w, "start": round(t, 3), "end": round(t + 0.4, 3)})
            t += 0.45
        t += gap
    return out


class _R:
    def __init__(self, data):
        self.data = data


class FakeAI:
    """Scripted answers by task name; every prompt is kept."""

    def __init__(self, **answers):
        self.answers, self.calls, self.prompts = answers, [], []

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        self.calls.append(task)
        self.prompts.append((task, system, prompt))
        a = self.answers.get(task)
        if a is None:
            raise AssertionError(f"unexpected AI task {task}")
        return _R(a(prompt) if callable(a) else a)


@pytest.fixture
def world():
    words = _words()
    sents = M.sentences_from_words(words)
    assert len(sents) == len(TEXTS)
    return words, sents


def _file(name, start, end, **facts):
    return {"name": name, "start": start, "end": end, "dur": end - start, "facts": facts}


NOTES = ("Open with \"we were wrong about the schedule\". Keep the whole section about forgiveness. Cut the part about the flood. "
         "No B-roll while he prays. Don't use the shaky clips. Make it 1 minute. Try to keep it light.")


def _extract_answer():
    return {"constraints": [
        {"type": "order", "hard": True, "source": "Open with \"we were wrong about the schedule\"", "want": "the video opens with that line"},
        {"type": "include", "hard": True, "source": "Keep the whole section about forgiveness", "want": "all of the forgiveness section stays"},
        {"type": "exclude_words", "hard": True, "source": "Cut the part about the flood", "want": "the flood section is out"},
        {"type": "exclude_broll_window", "hard": True, "source": "No B-roll while he prays", "want": "no B-roll over the prayer"},
        {"type": "exclude_files", "hard": True, "source": "Don't use the shaky clips", "want": "no shaky clips"},
        {"type": "duration", "hard": True, "source": "Make it 1 minute", "want": "60 seconds"},
        {"type": "style", "hard": False, "source": "Try to keep it light", "want": "light tone"},
        {"type": "style", "hard": True, "source": "Always shout at the camera", "want": "invented: it is not in the notes"},       # the notes never say it
    ]}


def test_notes_become_typed_constraints_with_quoted_sources():
    ai = FakeAI(constraints_extract=_extract_answer())
    got = CN.extract(ai, NOTES)
    assert [c["type"] for c in got] == ["order", "include", "exclude_words", "exclude_broll_window", "exclude_files", "duration", "style"]
    assert all(c["source"] in NOTES.replace("“", '"') or c["source"].lower() in NOTES.lower() for c in got)         # an item whose quote is not in the notes is dropped
    assert not any("shout" in c["source"] for c in got)
    assert [c["id"] for c in got] == [f"c{i}" for i in range(1, 8)]
    assert got[0]["hard"] and got[5]["hard"] and not got[6]["hard"]
    assert ai.calls == ["constraints_extract"]
    assert CN.extract(FakeAI(), "") == [] and CN.extract(FakeAI(), "1. Welcome\n2. Thanks") == []                    # no notes, or only an outline: no call at all


def test_a_numbered_outline_is_not_listed_as_constraints():
    ai = FakeAI(constraints_extract={"constraints": []})
    CN.extract(ai, "Make it warm.\n1. Welcome\n2. The flood\n3. Thanks")
    sent = ai.prompts[0][2]
    assert "Make it warm." in sent and "The flood" not in sent


def test_a_quoted_section_is_grounded_to_word_indexes_with_a_cut_window(world):
    words, sents = world
    cons = [{"id": "c1", "type": "order", "hard": True, "source": "Open with \"we were wrong about the schedule\"", "status": "ungrounded",
             "anchor": {}, "rule": {"text": ""}, "candidates": [], "note": ""}]
    got = CN.ground(cons, words, sents, [], [], None)[0]
    assert got["status"] == "grounded" and got["rule"]["position"] == "first"
    a, b = got["anchor"]["words"]
    assert " ".join(w["text"] for w in words[a:b + 1]) == "We were wrong about the schedule"
    assert got["anchor"]["lead_silence_s"] > 1.0 and got["anchor"]["trail_silence_s"] > 0.0                  # the quiet either side: the only legal cut window
    assert TS.quote_search(words, "we were wrong about the schedule")[0]["first_word"] == a
    assert TS.quote_search(words, "the dragon ate my homework entirely") == []


def test_two_sections_about_the_same_topic_make_the_constraint_ambiguous_not_a_guess(world, monkeypatch):
    words, sents = world
    monkeypatch.setattr(TS, "FAR_S", 15.0)                                                                       # (the test transcript is short: "far apart" is 15 s here)
    ai = FakeAI(transcript_topic_span={"spans": [{"first": 1, "last": 2}, {"first": 7, "last": 7}]})          # forgiveness at the start, a prayer at the end: far apart
    cons = [{"id": "c1", "type": "include", "hard": True, "source": "Keep the section about family", "status": "ungrounded", "anchor": {},
             "rule": {"text": "the section about family"}, "candidates": [], "note": ""}]
    got = CN.ground(cons, words, sents, [], [], ai)[0]
    assert got["status"] == "ambiguous" and len(got["candidates"]) == 2 and "pick one" in got["note"]
    assert all(isinstance(c["words"][0], int) for c in got["candidates"])


def test_shaky_and_drone_are_grounded_from_footage_facts_without_the_ai(world):
    words, sents = world
    files = [_file("C0001.MP4", 0, 20, motion={"shake": 0.7}, camera={"family": "sony_c"}), _file("C0002.MP4", 20, 40, motion={"shake": 0.1}, camera={"family": "sony_c"}),
             _file("DJI_0042.MP4", 40, 60, motion={"shake": 0.1}, camera={"family": "dji"})]
    ai = FakeAI()                                                                                                 # any call would raise
    cons = [{"id": "c1", "type": "exclude_files", "hard": True, "source": "Don't use the shaky clips", "status": "ungrounded", "anchor": {}, "rule": {"text": "no shaky clips"},
             "candidates": [], "note": ""},
            {"id": "c2", "type": "order", "hard": True, "source": "Open with the drone shot", "status": "ungrounded", "anchor": {}, "rule": {"text": "the drone shot first"},
             "candidates": [], "note": ""}]
    got = CN.ground(cons, words, sents, files, [], ai)
    assert got[0]["status"] == "grounded" and got[0]["anchor"]["file_ids"] == ["C0001.MP4"]
    assert got[1]["status"] == "grounded" and got[1]["anchor"]["file_ids"] == ["DJI_0042.MP4"] and got[1]["rule"]["position"] == "first"
    assert ai.calls == []


def test_a_scene_match_answer_only_counts_for_ids_the_code_listed(world):
    words, sents = world
    pool = [{"sid": "s0001", "file": "A.MP4", "what": "children playing at the river"}, {"sid": "s0002", "file": "B.MP4", "what": "a quiet chapel"}]
    ai = FakeAI(scene_match={"sids": ["s0002", "s9999"]})
    cons = [{"id": "c1", "type": "exclude_files", "hard": True, "source": "No footage of the chapel", "status": "ungrounded", "anchor": {}, "rule": {"text": "no chapel"},
             "candidates": [], "note": ""}]
    got = CN.ground(cons, words, sents, [], pool, ai)[0]
    assert got["anchor"]["file_ids"] == ["B.MP4"] and got["anchor"]["scene_ids"] == ["s0002"]                      # the made-up id is dropped


def test_negatives_become_filters(world):
    words, sents = world
    cons = CN.ground(CN.extract(FakeAI(constraints_extract=_extract_answer()), NOTES), words, sents,
                     [_file("C0001.MP4", 0, 20, motion={"shake": 0.8}), _file("C0002.MP4", 20, 40, motion={"shake": 0.1})], [],
                     FakeAI(transcript_topic_span={"spans": [{"first": 3, "last": 3}]}))
    flt = CN.filters(cons, sents, words)
    assert flt["exclude_files"] == ["C0001.MP4"]
    assert flt["exclude_sentences"] == {3}                                                                          # the flood sentence may not be kept
    assert flt["exclude_windows"] and flt["exclude_windows"][0]["constraint"] == "c4"
    pool = [{"sid": "s1", "file": "C0001.MP4", "start": 100.0, "end": 104.0}, {"sid": "s2", "file": "C0002.MP4", "start": 25.0, "end": 29.0},
            {"sid": "s3", "file": "C0002.MP4", "start": 30.0, "end": 34.0}]
    flt["exclude_windows"] = [{"words": [0, 1], "start_s": 24.0, "end_s": 29.5, "constraint": "c4"}]
    kept = CN.apply_pool_filters(pool, flt)
    assert [s["sid"] for s in kept] == ["s3"]                                                                       # the shaky file is out, and so is the shot inside the window


def test_b_roll_over_an_excluded_window_is_dropped_at_render_time():
    plan = {"exclude_files": ["BAD.MP4"], "exclude_windows": [{"start_s": 10.0, "end_s": 20.0}]}
    pool = [{"file": "OK.MP4"}, {"file": "BAD.MP4"}]
    cut = [{"at": 2.0, "length": 3.0, "scene": 0}, {"at": 11.0, "length": 3.0, "scene": 0}, {"at": 30.0, "length": 3.0, "scene": 1}, {"at": 40.0, "length": 3.0, "scene": 0}]
    out = CN.enforce(cut, plan, pool, [(0.0, 100.0)], lambda iv, t: t)
    assert [c["at"] for c in out] == [2.0, 40.0]


def _plan(items, **kw):
    return {"items": items, "style_obj": {"pace": {"cut_margin_s": 0.08}, "broll": {"shot_s": [2.0, 5.0], "share": 0.3}}, **kw}


def _con(cid, typ, **kw):
    base = {"id": cid, "type": typ, "hard": True, "source": kw.pop("source", typ), "status": "grounded", "anchor": {}, "rule": {}, "candidates": [], "note": ""}
    base["anchor"].update(kw.pop("anchor", {}))
    base["rule"].update(kw.pop("rule", {}))
    base.update(kw)
    return base


def _codes(res):
    return {p["code"] for p in res["problems"]}


def test_check_plan_finds_each_problem_code(world):
    words, sents = world
    files = [_file("A.MP4", 0, 30), _file("B.MP4", 30, 60)]
    check = lambda plan, cons=None, **kw: CN.check_plan(plan, cons or [], files, words, kw.pop("pool", []), None, sents=sents, **kw)     # noqa: E731
    assert "UNKNOWN_ID" in _codes(check(_plan([{"kind": "talk", "a": 40, "b": 41}])))
    assert "UNKNOWN_ID" in _codes(check(_plan([{"kind": "visual", "start": 500.0, "end": 510.0}])))
    assert "END_BEFORE_START" in _codes(check(_plan([{"kind": "talk", "a": 3, "b": 2}])))
    assert "END_BEFORE_START" in _codes(check(_plan([{"kind": "visual", "start": 8.0, "end": 5.0}])))
    r = check(_plan([{"kind": "visual", "start": 2.0, "end": 41.0}]))
    assert "TRIM_BEYOND_DURATION" in _codes(r) and "A.MP4" in next(p["message"] for p in r["problems"] if p["code"] == "TRIM_BEYOND_DURATION")
    assert "REPEATED_SOURCE" in _codes(check(_plan([{"kind": "talk", "a": 1, "b": 3}, {"kind": "talk", "a": 3, "b": 4}])))
    first = _con("c1", "order", rule={"position": "first"}, anchor={"words": [0, 5], "start_s": float(words[0]["start"]), "end_s": float(words[5]["end"]), "text": "Welcome"})
    assert "ORDER_FIRST" in _codes(check(_plan([{"kind": "talk", "a": 5, "b": 5}, {"kind": "talk", "a": 0, "b": 1}]), [first]))
    assert "ORDER_FIRST" not in _codes(check(_plan([{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 5, "b": 5}]), [first]))
    last = _con("c2", "order", rule={"position": "last"}, anchor={"words": [0, 5], "start_s": float(words[0]["start"]), "end_s": float(words[5]["end"])})
    assert "ORDER_LAST" in _codes(check(_plan([{"kind": "talk", "a": 0, "b": 0}, {"kind": "talk", "a": 5, "b": 5}]), [last]))
    sec = [words.index(w) for w in words if w["text"] == "Forgiveness"]
    inc = _con("c3", "include", anchor={"spans": [[sec[0], sec[0] + 7]]})
    r = check(_plan([{"kind": "talk", "a": 5, "b": 5}]), [inc])
    assert "INCLUDE_NOT_COVERED" in _codes(r) and "Forgiveness" in next(p["message"] for p in r["problems"] if p["code"] == "INCLUDE_NOT_COVERED")
    assert "INCLUDE_NOT_COVERED" not in _codes(check(_plan([{"kind": "talk", "a": 1, "b": 2}]), [inc]))
    exw = _con("c4", "exclude_words", anchor={"spans": [[words.index(next(w for w in words if w["text"] == "flooded")) - 2, words.index(next(w for w in words if w["text"] == "flooded")) + 4]]})
    assert "INCLUDE_NOT_COVERED" in _codes(check(_plan([{"kind": "talk", "a": 3, "b": 3}]), [exw]))
    exf = _con("c5", "exclude_files", anchor={"file_ids": ["B.MP4"]})
    assert "EXCLUDED_FILE_USED" in _codes(check(_plan([{"kind": "visual", "start": 35.0, "end": 40.0}]), [exf]))
    pool = [{"sid": "s1", "file": "A.MP4", "fstart": 1.0, "fend": 5.0, "start": 1.0, "end": 5.0}]
    win = _con("c6", "exclude_broll_window", anchor={"times": [[10.0, 20.0]]})
    r = CN.check_plan(_plan([{"kind": "talk", "a": 1, "b": 1}]), [win], files, words, pool, None, sents=sents, cutaways=[{"at": 12.0, "length": 3.0, "scene": 0}], to_out=lambda t: t)
    assert "BROLL_IN_EXCLUDED_WINDOW" in _codes(r)
    gfx = _con("c7", "graphic", rule={"before_s": 3.0}, source="Put the title on screen at the start")
    assert "GRAPHIC_MISSING_AT_START" in _codes(CN.check_plan(_plan([{"kind": "talk", "a": 1, "b": 1}], gfx_cards=[{"start": 20.0}]), [gfx], files, words, [], None, sents=sents))
    mid = words[10]
    inside = {"kind": "talk", "a": 1, "b": 1, "t0": (float(mid["start"]) + float(mid["end"])) / 2, "t1": float(words[12]["end"])}
    assert "CUT_INSIDE_WORD" in _codes(check(_plan([inside])))
    tiny = {"kind": "talk", "a": 1, "b": 1, "t0": float(sents[1].start), "t1": float(sents[1].start) + 0.5}
    short = check(_plan([tiny]))
    assert "SEGMENT_TOO_SHORT" in _codes(short) and all(not p["hard"] for p in short["problems"] if p["code"] == "SEGMENT_TOO_SHORT")
    assert check(_plan([{"kind": "talk", "a": 1, "b": 2}]))["ok"]
    r = check(_plan([{"kind": "talk", "a": 1, "b": 2}]), pool=[{"sid": "s9", "file": "A.MP4", "fstart": 5.0, "fend": 400.0, "start": 5.0, "end": 400.0}])
    assert "POOL_OUTSIDE_FILES" in _codes(r)
    sh = CN.check_plan(_plan([{"kind": "talk", "a": 1, "b": 2}]), [], files, words, pool, None, sents=sents, cutaways=[{"at": 1.0, "length": 1.0, "scene": 0}])
    assert any(p["code"] == "BROLL_SHARE_OFF" and not p["hard"] for p in sh["problems"])                            # soft: it goes in the report, it never blocks


def test_duration_is_computed_by_code_and_the_fix_hint_lists_the_weakest_segments(world):
    words, sents = world
    dur = CN.ground([_con("c1", "duration", source="Make it 12 seconds", status="ungrounded")], words, sents, [], [], None)[0]
    assert dur["anchor"]["seconds"] == 12.0 and dur["rule"]["band"] == [10.8, 13.2]
    items = [{"kind": "talk", "a": 0, "b": 1, "score": 9}, {"kind": "talk", "a": 2, "b": 2, "score": 2}, {"kind": "talk", "a": 4, "b": 4, "score": 5}, {"kind": "talk", "a": 7, "b": 8, "score": 8}]
    res = CN.check_plan(_plan(items), [dur], [_file("A.MP4", 0, 90)], words, [], None, sents=sents)
    p = next(p for p in res["problems"] if p["code"] == "DURATION_OUT_OF_BAND")
    assert p["hard"] and "weakest optional passages" in p["fix_hint"]
    order = [t.split(" ")[0] for t in p["fix_hint"].split("first: ", 1)[1].split(", ")]
    assert order[0] == "i1" and order.index("i2") < order.index("i3")                                           # the lowest score first
    trimmed = CN.trim_to_band(_plan(items), sents, [dur], dur["rule"]["band"])
    assert len(trimmed["items"]) < len(items) and trimmed["items"][0] == items[0] and trimmed["items"][-1] == items[-1]
    assert CN.parse_duration("a 2-minute recap") == 120.0 and CN.parse_duration("about 90 seconds") == 90.0 and CN.parse_duration("no length") is None


def test_repair_runs_at_most_three_times_then_reports_unmet(world):
    words, sents = world
    files = [_file("A.MP4", 0, 90)]
    first = _con("c1", "order", rule={"position": "first"}, anchor={"words": [0, 5], "start_s": float(words[0]["start"]), "end_s": float(words[5]["end"]), "text": "Welcome", "times": [[0.0, 3.0]]})
    plan = _plan([{"kind": "talk", "a": 5, "b": 5}, {"kind": "talk", "a": 0, "b": 1}])
    ai = FakeAI(plan_repair={"items": [{"kind": "talk", "a": 5, "b": 5}, {"kind": "talk", "a": 0, "b": 1}], "unmet": [{"constraint": "c1", "why": "the opening line is in the middle of a take"}]})
    out = CN.finish(ai, plan, {"constraints": [first], "filters": {}, "checklist": ""}, words, sents, files, [], None, False, None)
    assert ai.calls.count("plan_repair") == 3                                                                    # never more than three
    assert out["unmet"] and out["unmet"][0]["constraint"] == "c1"
    assert out["awaiting_approval"] is True and out["approval"]["questions"]                                      # the job stops for the person
    assert next(c for c in out["constraints"] if c["id"] == "c1")["status"] == "unmet"
    auto = CN.finish(FakeAI(plan_repair=ai.answers["plan_repair"]), plan, {"constraints": [first], "filters": {}, "checklist": ""}, words, sents, files, [], None, True, None)
    assert not auto.get("awaiting_approval") and auto["unmet"]                                                    # auto-approve: it renders, the report says what is unmet


def test_a_repair_that_fixes_the_plan_stops_the_loop_and_one_that_invents_ids_is_thrown_away(world):
    words, sents = world
    files = [_file("A.MP4", 0, 90)]
    first = _con("c1", "order", rule={"position": "first"}, anchor={"words": [0, 5], "start_s": float(words[0]["start"]), "end_s": float(words[5]["end"]), "text": "Welcome"})
    bad = _plan([{"kind": "talk", "a": 5, "b": 5}, {"kind": "talk", "a": 0, "b": 1}])
    fixed = FakeAI(plan_repair={"items": [{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 5, "b": 5}]})
    out = CN.finish(fixed, bad, {"constraints": [first], "filters": {}, "checklist": ""}, words, sents, files, [], None, False, None)
    assert fixed.calls == ["plan_repair"] and [it["a"] for it in out["items"]] == [0, 5] and not out["unmet"] and not out.get("awaiting_approval")
    made_up = FakeAI(plan_repair={"items": [{"kind": "talk", "a": 0, "b": 99}, {"kind": "talk", "a": 5, "b": 5}]})
    out2 = CN.finish(made_up, bad, {"constraints": [first], "filters": {}, "checklist": ""}, words, sents, files, [], None, False, None)
    assert [it["a"] for it in out2["items"]] == [5, 0]                                                            # the invented sentence number was refused: the plan is as it was


def test_a_hard_ambiguous_constraint_stops_for_approval_and_a_soft_one_is_decided_and_reported(world):
    words, sents = world
    cand = [{"words": [4, 9], "start_s": 1.0, "end_s": 4.0, "text": "first place"}, {"words": [60, 66], "start_s": 90.0, "end_s": 93.0, "text": "second place"}]
    hard = _con("c1", "include", status="ambiguous", candidates=cand, note="“grace” could be 0:01-0:04 or 1:30-1:33: pick one.")
    soft = _con("c2", "include", status="ambiguous", candidates=cand, hard=False, note="x")
    plan = _plan([{"kind": "talk", "a": 1, "b": 2}])
    stop = CN.finish(FakeAI(), plan, {"constraints": [hard], "filters": {}, "checklist": ""}, words, sents, [_file("A.MP4", 0, 90)], [], None, True, None)
    assert stop["awaiting_approval"] and "pick one" in stop["approval"]["questions"][0]                           # even with auto-approve: a question is a question
    go = CN.finish(FakeAI(), plan, {"constraints": [soft], "filters": {}, "checklist": ""}, words, sents, [_file("A.MP4", 0, 90)], [], None, False, None)
    c = go["constraints"][0]
    assert not go.get("awaiting_approval") and c["status"] == "grounded" and c["anchor"]["words"] == [4, 9] and "I took it to be" in c["note"]
    assert "I took it to be" in CN.report(go["constraints"], go["problems"], go["unmet"])


def test_the_checklist_goes_last_in_the_brief_the_planners_read(world):
    words, sents = world
    cons = CN.ground(CN.extract(FakeAI(constraints_extract=_extract_answer()), NOTES), words, sents, [_file("C0001.MP4", 0, 20, motion={"shake": 0.8})], [],
                     FakeAI(transcript_topic_span={"spans": [{"first": 1, "last": 2}]}))
    text = CN.checklist(cons)
    assert text.startswith("INSTRUCTIONS TO SATISFY (check each before answering):")
    assert "c1 (hard) order words" in text and "c6 (hard) duration 60 s" in text
    prep = CN.prepare(FakeAI(constraints_extract=_extract_answer(), transcript_topic_span={"spans": [{"first": 1, "last": 2}]}), NOTES, None, words, sents, [_file("C0001.MP4", 0, 20)], [])
    assert prep["checklist"].startswith("INSTRUCTIONS TO SATISFY") and prep["constraints"]
    from clipfarm.clipper import localproj as LP
    import inspect
    src = inspect.getsource(LP.LocalRunner.plan)
    assert 'brief = brief + "\\n\\n" + cprep["checklist"]' in src and src.index("cprep = CN.prepare") < src.index("HL.candidates")     # (added to the brief before any planner reads it)
    assert CN.checklist([]) == ""


def test_post_render_check_finds_a_missing_include_and_a_present_exclude(world):
    words, sents = world
    inc = _con("c1", "include", anchor={"texts": ["Forgiveness is a gift we give away freely"]})
    exw = _con("c2", "exclude_words", anchor={"texts": ["The river flooded our camp on Tuesday night"]})
    dur = _con("c3", "duration", rule={"band": [50.0, 70.0]}, anchor={"seconds": 60.0})
    gfx = _con("c4", "graphic", rule={"before_s": 3.0, "text": "Summer Camp"})
    rendered_bad = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate("The river flooded our camp on Tuesday night and then we left".split())]
    got = CN.post_render_check([inc, exw, dur, gfx], rendered_bad, 90.0, "Welcome", sents=[])
    by = {x["constraint"]: x for x in got}
    assert not by["c1"]["ok"] and "MISSING" in by["c1"]["what"]
    assert not by["c2"]["ok"] and "STILL in" in by["c2"]["what"]
    assert not by["c3"]["ok"] and not by["c4"]["ok"]
    rendered_ok = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate("Forgiveness is a gift we give away freely".split())]
    ok = CN.post_render_check([inc, exw, dur, gfx], rendered_ok, 61.0, "Summer Camp week", sents=[])
    assert all(x["ok"] for x in ok)
    twice = M.sentences_from_words(_words(["We were wrong about the schedule today", "We were wrong about the schedule today", "A different line follows here for good measure"], 1.2))
    assert any("said twice" in x["what"] for x in CN.post_render_check([], [], None, None, twice))


def test_revise_constraints_merges_a_new_rule_once_and_applies_it(world):
    words, sents = world
    ai = FakeAI(constraints_extract={"constraints": [{"type": "exclude_words", "hard": True, "source": "cut the part about the flood", "want": "no flood"}]},
                transcript_topic_span={"spans": [{"first": 3, "last": 3}]})
    from clipfarm.clipper import localproj as LP
    plan = _plan([{"kind": "talk", "a": 1, "b": 4}], constraints=[])
    out, did = CN.revise_constraints(ai, plan, "please cut the part about the flood", sents, words, [_file("A.MP4", 0, 90)], LP.drop_sentences)
    assert "1 new rule" in did and out["constraints"][0]["id"] == "c1"
    kept = {i for it in out["items"] if it["kind"] == "talk" for i in range(it["a"], it["b"] + 1)}
    assert 3 not in kept and {1, 2, 4} <= kept                                                                    # the flood sentence is gone, the rest stays
    again, did2 = CN.revise_constraints(ai, out, "please cut the part about the flood", sents, words, [_file("A.MP4", 0, 90)], LP.drop_sentences)
    assert did2 == "" and len(again["constraints"]) == 1                                                          # the same rule is not added twice


def _runner(tmp_path, monkeypatch, extra=None):
    """A real local project over a folder of small test clips, with a scripted AI that also answers the instruction tasks."""
    from test_aivideo import _FakeAI, _make_folder

    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper import localproj as LP
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True, "verify_freeze": False}
    folder = _make_folder(tmp_path)
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))

    class AI(_FakeAI):
        def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
            if task in (extra or {}):
                self.calls.append(task)
                a = extra[task]
                return _R(a(prompt) if callable(a) else a)
            return super().complete_json(task, system, prompt, schema, validate=validate, images=images, **kw)
    fake = AI()
    monkeypatch.setattr(LP, "LocalAI", lambda c: fake)
    words = {"talk1.mov": [{"text": f"w{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(24)],
             "talk2.mov": [{"text": f"v{i}" + ("." if i % 5 == 4 else ""), "start": 0.5 + i * 0.5, "end": 0.5 + i * 0.5 + 0.3} for i in range(16)]}
    monkeypatch.setattr(LP.LocalRunner, "_openai_words", lambda self, f, ai: words.get(f.get("name"), []))      # (the finished video has no scripted words: the post-render check hears nothing)
    runner = LP.LocalRunner(cfg, store, projects)
    return runner, projects, store, folder, fake


def test_a_plan_without_a_render_checks_the_notes_in_code_and_an_outside_ai_can_edit_it(tmp_path, monkeypatch):
    extra = {"constraints_extract": {"constraints": [{"type": "include", "hard": True, "source": "keep \"w5 w6 w7 w8 w9\"", "want": "that line stays"},
                                                      {"type": "duration", "hard": False, "source": "about 30 seconds", "want": "30 s"}]}}
    runner, projects, store, folder, fake = _runner(tmp_path, monkeypatch, extra)
    (folder / "notes.md").write_text("Make it punchy. keep \"w5 w6 w7 w8 w9\". about 30 seconds.")
    proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": ""}, {"clips": 0})
    projects.prepare_local(proj["id"], {"dry_run": True, "library": False})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "approval" and "plan is ready" in job["stage"]
    assert not list((folder / "Output").glob("*.mp4"))                                                                  # planned, not rendered
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["dry_run"] and plan["awaiting_approval"] and [c["type"] for c in plan["constraints"]] == ["include", "duration"]
    assert plan["constraints"][0]["status"] in ("met", "unmet") and "constraints_extract" in fake.calls
    from clipfarm.clipper import planapi
    v = planapi.view(runner, proj["id"])
    assert v["items"] and all(it["id"] for it in v["items"]) and v["constraints"] and "problems" in v and "report" in v and "Your instructions" in v["report"]
    first = next(it for it in v["items"] if it["kind"] == "talk")
    ok = planapi.check(runner, proj["id"], {"items": [{"id": first["id"], "kind": "talk", "a": first["a"], "b": first["b"]}]})
    assert "problems" in ok and isinstance(ok["totals"]["length_s"], float)
    bad = planapi.check(runner, proj["id"], {"items": [{"kind": "talk", "a": 900, "b": 901}]})
    assert not bad["ok"] and any(p["code"] == "UNKNOWN_ID" for p in bad["problems"])                                    # an id the code never listed
    with pytest.raises(ValueError, match="problems that must be fixed"):
        planapi.apply(runner, proj["id"], {"items": [{"kind": "talk", "a": 900, "b": 901}]}, render=False)
    with pytest.raises(ValueError, match="list"):
        planapi.check(runner, proj["id"], {"items": "not a list"})
    hits = planapi.search(runner, proj["id"], "w5 w6 w7 w8 w9")
    assert hits["hits"] and hits["hits"][0]["lead_silence_s"] >= 0 and hits["mode"] == "quote"
    res = planapi.apply(runner, proj["id"], {"items": [{"id": first["id"], "kind": "talk", "a": first["a"], "b": first["b"]}]}, render=False)
    assert res["ok"]
    saved = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert saved["awaiting_approval"] and "dry_run" not in saved and saved["pending_note"] == "plan sent by an AI"      # saved, waiting for the render to start


def test_an_instruction_that_cannot_be_grounded_stops_for_the_person_and_an_unmet_one_is_reported(tmp_path, monkeypatch):
    extra = {"constraints_extract": {"constraints": [{"type": "include", "hard": True, "source": "keep \"words nobody ever said\"", "want": "that stays"}]},
             "plan_repair": {"items": [{"kind": "talk", "a": 0, "b": 1}], "unmet": []}}
    runner, projects, store, folder, fake = _runner(tmp_path, monkeypatch, extra)
    (folder / "notes.md").write_text("Make it punchy. keep \"words nobody ever said\".")
    proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": ""}, {"clips": 0})
    projects.prepare_local(proj["id"], {"library": False})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "done", job                                                                                  # an ungrounded constraint with nothing to check does not block the render
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["unmet"] and plan["constraints"][0]["status"] == "ungrounded" or plan["unmet"][0]["constraint"] == "c1"
    md = (folder / "Notes" / "plan.md").read_text()
    assert "Your instructions" in md and "c1" in md and "words nobody ever said" in md                                   # the report lists what could not be done
