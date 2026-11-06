"""The plan review screen's logic: editing the plan line by line before anything renders."""
from clipfarm.clipper import flow as FL
from clipfarm.clipper import moments as M

OUTLINE = ["Young couple with daughter, Lucia, moved from Hawaii to Dubuque.", "WIC referred them to Harbor Clinic.",
           'Community: people who were "cheering for us."']


def sents():
    texts = ["We moved here from Hawaii.", "Our daughter Lucia was one.", "It cost a lot.", "WIC sent us to Harbor Clinic.", "We were overwhelmed.",
             "Honestly we got diapers.", "Yeah.", "People are cheering for us.", "That meant everything."]
    return [M.Sentence(i, i * 5.0, i * 5.0 + 4.0, t) for i, t in enumerate(texts)]


def items():
    return [{"kind": "talk", "a": 0, "b": 2, "step": 1}, {"kind": "talk", "a": 3, "b": 4, "step": 2}, {"kind": "talk", "a": 7, "b": 8, "step": 3}]


def test_the_map_shows_every_line_with_its_time_and_the_unused_neighbours():
    m = FL.script_map(sents(), items(), OUTLINE)
    first = m[0]["passages"][0]
    assert [ln["sentence"] for ln in first["lines"]] == [0, 1, 2] and first["lines"][1]["start"] == 5.0
    assert first["before"] is None and first["after"] is None       # sentence 3 starts step 2's passage: both ends are taken
    assert m[1]["passages"][0]["before"] is None                     # sentence 2 is used by step 1
    assert m[1]["passages"][0]["after"]["sentence"] == 5 and m[2]["passages"][0]["before"]["sentence"] == 6


def test_names_and_quotes_are_checked_word_for_word_on_the_map():
    m = FL.script_map(sents(), items(), OUTLINE)
    assert m[0]["missing"] == [] and m[1]["missing"] == [] and m[2]["missing"] == []
    gone = FL.script_map(sents(), [{"kind": "talk", "a": 0, "b": 0, "step": 1}, {"kind": "talk", "a": 3, "b": 4, "step": 2},
                                    {"kind": "talk", "a": 8, "b": 8, "step": 3}], OUTLINE)
    assert gone[0]["missing"] == ["Lucia"]                          # said in sentence 1, which this plan no longer uses
    assert gone[2]["missing"] == ["cheering for us"]


def test_dropping_a_line_trims_splits_or_removes_a_passage():
    s = sents()
    out = FL.edit_items(s, items(), {"op": "drop", "first": 0, "last": 2, "sentence": 2})
    assert [(i["a"], i["b"]) for i in out if i["step"] == 1] == [(0, 1)]
    out = FL.edit_items(s, items(), {"op": "drop", "first": 0, "last": 2, "sentence": 1})
    assert [(i["a"], i["b"]) for i in out if i["step"] == 1] == [(0, 0), (2, 2)]
    one = [{"kind": "talk", "a": 3, "b": 3, "step": 2}]
    assert FL.edit_items(s, one, {"op": "drop", "first": 3, "last": 3, "sentence": 3}) == []


def test_extending_adds_only_a_line_nobody_else_uses():
    s = sents()
    out = FL.edit_items(s, items(), {"op": "extend", "first": 3, "last": 4, "edge": "end"})
    assert [(i["a"], i["b"]) for i in out if i["step"] == 2] == [(3, 5)]
    try:
        FL.edit_items(s, items(), {"op": "extend", "first": 3, "last": 4, "edge": "start"})
        raise AssertionError("sentence 2 belongs to step 1")
    except ValueError as e:
        assert "already used" in str(e)


def test_adding_a_found_passage_puts_it_after_the_step_and_refuses_used_lines():
    s = sents()
    out = FL.edit_items(s, items(), {"op": "add", "step": 2, "first": 5, "last": 5})
    assert [(i["a"], i["b"]) for i in out if i["step"] == 2] == [(3, 4), (5, 5)] and out.index(next(i for i in out if i["a"] == 5)) == 2
    try:
        FL.edit_items(s, items(), {"op": "add", "step": 2, "first": 4, "last": 5})
        raise AssertionError("sentence 4 is used")
    except ValueError:
        pass


def test_finding_a_line_offers_only_unused_whole_passages():
    class AI:
        def complete_json(self, task, system, prompt, schema, **kw):
            class R:
                data = {"passages": [{"first": 5, "last": 6}, {"first": 3, "last": 4}, {"first": 99, "last": 99}]}
            return R
    got = FL.find_candidates(AI(), sents(), items(), OUTLINE[1], "the diapers")
    assert [(c["first"], c["last"]) for c in got] == [(5, 6)] and got[0]["lines"][0]["text"].startswith("Honestly")


def test_a_click_on_the_review_screen_changes_the_saved_plan_and_keeps_a_backup(tmp_path):
    from clipfarm.clipper import localproj as LP
    st = LP.State(tmp_path)
    plan = {"items": items(), "awaiting_approval": True, "flow": []}
    st.put("plan.json", plan)
    runner = LP.LocalRunner.__new__(LP.LocalRunner)
    import threading
    runner.review_lock = threading.Lock()
    proj = {"notes": "Script:\n- " + "\n- ".join(OUTLINE)}
    runner._review_state = lambda pid: (proj, st, None, sents(), st.get("plan.json"))
    runner.review_edit("p", {"op": "drop", "first": 0, "last": 2, "sentence": 2})
    saved = st.get("plan.json")
    assert [(i["a"], i["b"]) for i in saved["items"] if i["step"] == 1] == [(0, 1)] and saved["flow_edited"] and saved["awaiting_approval"]
    assert saved["script_map"][0]["passages"][0]["after"]["sentence"] == 2                 # the dropped line can be put back with one click
    assert list((st.dir / "backups").glob("plan-*before-review-edits*.json"))               # the plan as it was is kept
    try:
        runner.review_edit("p", {"op": "extend", "first": 0, "last": 2, "edge": "end"})
        raise AssertionError("the passage no longer ends at 2")
    except ValueError:
        pass
