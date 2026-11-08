"""The planner lane of the evaluation (WP15): briefs by level, scored by the validator, offline, with the plumbing fake AI."""
import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_eval_fixture import editor_folder  # noqa: E402,F401  (the session fixture)

from eval import briefs as B  # noqa: E402
from eval import score as S  # noqa: E402
from eval import world as W  # noqa: E402

REF = Path(__file__).resolve().parent / "eval" / "reference_plans"


@pytest.fixture(scope="module")
def world(editor_folder):  # noqa: F811
    return W.load(editor_folder)


def test_levels_one_to_six_pass_with_the_fake_ai(world):
    t0 = time.time()
    rows = [S.run_brief(b, world) for b in B.levels(6)]
    for b, r in zip(B.levels(6), rows):
        assert r["grounding"] == [], (b["level"], r["grounding"])                                              # extraction + grounding found what the script says
        assert r["before"]["ok"] is False or b["level"] == 1 and r["before"]["problems"]                       # the validator saw the planner that ignores the brief
    for r in rows[:4]:
        assert r["hsr"] == 1.0, [p["message"] for p in r["after"]["problems"] if p["hard"]]                    # L1 to L4: every hard constraint met
    for r in rows[4:]:
        assert r["ssr"] >= 0.9 and r["hsr"] == 1.0
    assert [r["csl"] for r in rows] == [1, 2, 3, 4, 5, 6]                                                      # each level met all of its constraints
    assert all(r["repair_calls"] <= 3 for r in rows)
    assert time.time() - t0 < 60


def test_the_validator_sees_a_brief_the_naive_plan_ignores(world):
    r = S.run_brief(B.levels(4)[-1], world)
    codes = {p["code"] for p in r["before"]["problems"]}
    assert {"ORDER_FIRST", "EXCLUDED_FILE_USED"} <= codes and r["hsr"] == 1.0 and r["plan"]["repair_calls"] == 1


def test_compositions_and_selection_and_chain(world):
    rows = {b["composition"]: S.run_brief(b, world) for b in B.compositions()}
    assert rows["and"]["hsr"] == 1.0 and rows["and"]["csl"] == 2                                               # two constraints in one sentence
    assert rows["selection"]["constraints"][0]["anchor"]["file_ids"] == ["cam_b.mp4"]                          # "the second camera" is grounded from the file names, no model
    assert rows["chain"]["hsr"] == 1.0                                                                          # last line + a graphic at the start


def test_plan_distance_is_zero_for_the_reference_and_counts_moves_insertions_and_deletions(world):
    ref = json.loads((REF / "l2.json").read_text())
    assert S.distance(ref, ref, world) == 0.0
    extra = {"items": ref["items"] + [{"kind": "talk", "a": 3, "b": 3}]}
    assert S.distance(extra, ref, world) == 1.0                                                                 # one insertion
    assert S.distance({"items": ref["items"][:2]}, ref, world) == 1.0                                          # one deletion
    moved = {"items": [ref["items"][0], {"kind": "talk", "a": 0, "b": 1}, ref["items"][2]]}
    assert 0 < S.distance(moved, ref, world) <= 1.0                                                            # a boundary moved
    r = S.run_brief(B.levels(2)[1], world, ref_plan=ref)
    assert r["plan_distance"] is not None and r["plan_distance"] <= 2.0                                        # the repaired plan is close to the hand-made one


def test_a_brief_the_fake_cannot_ground_is_counted_as_not_met(world):
    bad = B.brief(["order_first"])
    bad = {**bad, "extract_answer": {"constraints": [{"type": "order", "hard": True, "source": "Open with the dragon scene", "want": "x"}]},
           "text": "Open with the dragon scene.", "expect": [{"status": "grounded"}]}
    r = S.run_brief(bad, world)
    assert r["hsr"] == 0.0 and r["grounding"] and r["constraints"][0]["status"] in ("ungrounded", "unmet")
