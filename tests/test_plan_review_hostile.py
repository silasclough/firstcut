"""Hostile inputs for the plan review API (WP16): empty, huge, odd and repeated requests must give plain answers, never a crash or a lost part."""
import copy
from pathlib import Path

import pytest

from clipfarm.clipper import planreview as PR
from clipfarm.clipper.localproj import State

from test_plan_review import SENTS, _plan, runner  # noqa: F401  (the fixture)


def _home(runner):
    return Path(runner.projects.get("p1")["home"])


def _put(runner, plan):
    State(_home(runner)).put("plan.json", plan)


def test_one_item_and_five_hundred_items_review_without_error(runner):
    n = len(SENTS)
    _put(runner, {"title": "t", "items": [{"id": "i0", "kind": "talk", "a": 0, "b": 1}], "constraints": [], "style": "story"})
    r = PR.review(runner, "p1")
    assert len(r["cut"]) == 1 and r["ready"] is True
    items = [{"id": f"i{k}", "kind": "talk", "a": k % n, "b": k % n} for k in range(500)]
    _put(runner, {"title": "t", "items": items, "constraints": [], "style": "story"})
    r = PR.review(runner, "p1")
    assert len(r["cut"]) == 500 and "500 parts" in r["summary"]


def test_items_out_of_range_or_malformed_do_not_crash(runner):
    items = [{"id": "x1", "kind": "talk", "a": 9999, "b": 99999}, {"id": "x2", "kind": "talk", "a": 5, "b": 2}, {"id": "x3", "kind": "talk"},
             {"id": "x4", "kind": "mystery", "a": "z"}]
    _put(runner, {"title": None, "items": items, "constraints": [], "style": "story"})
    r = PR.review(runner, "p1")
    assert isinstance(r["cut"], list)


def test_constraint_with_no_quote_no_anchor_and_twenty_questions(runner):
    cons = [{"id": f"q{k}", "type": "exclude_words", "hard": True, "status": "ambiguous", "anchor": {}, "rule": {},
             "candidates": [{"words": [k, k + 1], "start_s": 2.0 + k, "end_s": 3.0 + k, "text": f"cand {k}"}, {"words": [k + 5, k + 6], "start_s": 30.0 + k, "end_s": 31.0 + k, "text": "b"}],
             "note": ""} for k in range(20)]
    cons.append({"id": "nq", "type": "include", "hard": False, "status": "grounded"})
    p = _plan()
    p["constraints"] = cons
    _put(runner, p)
    r = PR.review(runner, "p1")
    assert len(r["questions"]) == 20 and r["ready"] is False
    for q in r["questions"]:
        assert q["choices"]
    r2 = PR.answer(runner, "p1", "q3", 0)
    assert len(r2["questions"]) == 19


def test_answer_twice_and_garbage_choices_are_refused_plainly(runner):
    PR.answer(runner, "p1", "c2", 0)
    for bad in (0, None, "x", -5, 1.5, [], {}):
        with pytest.raises(ValueError):
            PR.answer(runner, "p1", "c2", bad)


def test_negative_choice_index_is_not_a_valid_choice(runner):
    with pytest.raises(ValueError):
        PR.answer(runner, "p1", "c2", -1)       # python negative indexing must not pick "the last one" silently


def test_leave_out_twice_put_back_twice_and_unknown_ids(runner):
    PR.leave_out(runner, "p1", "i3", True)
    with pytest.raises(ValueError):
        PR.leave_out(runner, "p1", "i3", True)             # double click: the part is not in the plan any more
    items = [it["id"] for it in State(_home(runner)).get("plan.json")["items"]]
    assert items == ["i0", "i1", "i2", "i4"]
    PR.leave_out(runner, "p1", "i3", False)
    with pytest.raises(ValueError):
        PR.leave_out(runner, "p1", "i3", False)
    with pytest.raises(ValueError):
        PR.leave_out(runner, "p1", "", True)
    with pytest.raises(ValueError):
        PR.leave_out(runner, "p1", "nope", False)
    items = [it["id"] for it in State(_home(runner)).get("plan.json")["items"]]
    assert items == ["i0", "i1", "i2", "i3", "i4"] and not State(_home(runner)).get("plan.json").get("left_out_items")


def test_leaving_out_everything_that_is_not_required_never_loses_a_part(runner):
    for k in ("i0", "i2", "i3", "i4"):
        PR.leave_out(runner, "p1", k, True)
    with pytest.raises(ValueError):
        PR.leave_out(runner, "p1", "i1", True)
    plan = State(_home(runner)).get("plan.json")
    assert [i["id"] for i in plan["items"]] == ["i1"] and len(plan["left_out_items"]) == 4
    for k in ("i4", "i0", "i3", "i2"):
        PR.leave_out(runner, "p1", k, False)
    assert sorted(i["id"] for i in State(_home(runner)).get("plan.json")["items"]) == ["i0", "i1", "i2", "i3", "i4"]


def test_items_without_ids_get_stable_ids_so_leave_out_works(runner):
    p = _plan()
    for it in p["items"]:
        it.pop("id")
    _put(runner, p)
    r = PR.review(runner, "p1")
    ids = [row["id"] for row in r["cut"]]
    assert len(set(ids)) == len(ids) == 5


def test_plan_changed_between_review_and_a_later_click_is_refused_not_applied_to_the_wrong_part(runner):
    PR.review(runner, "p1")
    p = State(_home(runner)).get("plan.json")
    p["items"] = [it for it in p["items"] if it["id"] != "i3"]            # another tab or a re-plan removed it
    _put(runner, p)
    with pytest.raises(ValueError):
        PR.leave_out(runner, "p1", "i3", True)


def test_use_take_garbage(runner):
    for a, b in ((None, None), ("x", 2), (13, "y"), (13, 18), (-1, -1)):
        with pytest.raises(ValueError):
            PR.use_take(runner, "p1", "i2", a, b)
    with pytest.raises(ValueError):
        PR.use_take(runner, "p1", "i0", 13, 17)        # i0 has no such other take; the line is not its own


def test_a_plan_with_no_plan_file_says_so(runner):
    (_home(runner) / "plan.json").unlink(missing_ok=True)
    for p in list(_home(runner).rglob("plan.json")):
        p.unlink()
    with pytest.raises((ValueError, KeyError, OSError)):
        PR.review(runner, "p1")
