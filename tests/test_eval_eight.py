"""The eight-constraint brief on the evaluation fixture (WP5 "Done when", built by WP15): order x2, include, exclude_broll_window, exclude_files, duration, graphic,
camera. Planner lane (validator + repair, plumbing fake) and one real render of the project through the local pipeline with fake AIs, then frame inspection."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_eval_fixture import editor_folder  # noqa: E402,F401

from eval import briefs as B  # noqa: E402
from eval import e2e  # noqa: E402
from eval import frames as F  # noqa: E402
from eval import score as S  # noqa: E402
from eval import world as W  # noqa: E402
from fixtures import make_editor_folder as FX  # noqa: E402

from clipfarm.clipper import constraints as CN  # noqa: E402
from clipfarm.core import ffmpeg  # noqa: E402


@pytest.fixture(scope="module")
def world(editor_folder):  # noqa: F811
    return W.load(editor_folder)


def test_the_eight_constraint_brief_has_no_hard_problem_after_at_most_three_repairs_and_one_soft_unmet(world):
    r = S.run_brief(B.eight(), world)
    assert [c["type"] for c in r["constraints"]] == ["order", "order", "include", "exclude_broll_window", "exclude_files", "duration", "graphic", "camera"]
    assert r["grounding"] == []                                                                                  # every one was grounded to what the fixture's script says
    assert r["hsr"] == 1.0 and not [p for p in r["after"]["problems"] if p["hard"]] and r["repair_calls"] <= 3
    soft = [p for p in r["after"]["problems"] if not p["hard"] and p.get("constraint_id")]
    assert [p["constraint_id"] for p in soft] == ["c8"] and "cam_b.mp4" in soft[0]["message"]                      # the one soft unmet: the second camera is not used
    text = CN.report(r["constraints"], r["after"]["problems"], r["unmet"])
    assert "Your instructions" in text or "c1" in text


@pytest.fixture(scope="module")
def rendered(editor_folder, tmp_path_factory):  # noqa: F811
    """One real render of the seven-constraint brief (the window rule is tested on its own below)."""
    tmp = tmp_path_factory.mktemp("eight")
    mp = pytest.MonkeyPatch()
    try:
        brief = B.brief([i for i in B.eight()["ids"] if i != "broll_window"], composition="render-7")
        runner, projects, store, folder, fake = e2e.build_runner(tmp, mp, brief, editor_folder)
        proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": ""}, {"clips": 0})
        projects.prepare_local(proj["id"], {"library": False})
        runner.make(proj["id"], {})
        job = store.get(proj["job"])
        out = next((folder / "Output").glob("*.mp4"), None)
        plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
        md = (folder / "Notes" / "plan.md").read_text() if (folder / "Notes" / "plan.md").exists() else ""
        yield {"job": job, "out": out, "plan": plan, "md": md, "folder": folder, "tmp": tmp, "fake": fake}
    finally:
        mp.undo()


def test_the_brief_renders_end_to_end_with_the_instructions_met_and_listed(rendered):
    assert rendered["job"]["status"] == "done", rendered["job"]
    plan, md = rendered["plan"], rendered["md"]
    assert not [p for p in plan["problems"] if p["hard"]] and plan["repair_calls"] <= 3
    assert "Your instructions" in md and "What I checked after rendering" in md                                  # the constraint table is in Notes/plan.md
    for cid in ("c1", "c2", "c3", "c4", "c5", "c6"):
        assert f"- {cid} " in md and "done" in next(ln for ln in md.splitlines() if ln.startswith(f"- {cid} "))
    c7 = next(ln for ln in md.splitlines() if ln.startswith("- c7 "))
    assert "would like" in c7 and "not understood" in c7                                                           # the soft one: reported with its reason, not hidden
    assert "forgiveness" in md and "is in the finished video" in md                                               # the post-render check heard the include
    first = next(it for it in plan["items"] if it["kind"] == "talk")
    assert first["a"] == 5 and plan["items"][-1]["b"] == 8                                                        # opens with the schedule line, ends with the thanks
    shaky = next(c for c in plan["constraints"] if c["type"] == "exclude_files")["anchor"]["file_ids"]
    assert "broll_shaky.mp4" in shaky and "drone.mp4" not in shaky                                                 # the steady drone pan is not 'shaky' (the fixture's facts come from the real motion analysis)


def test_the_finished_video_of_the_fixture_has_no_frame_issues(rendered):
    r = F.inspect(rendered["out"], rendered["tmp"] / "frames", face_box=tuple(FX.FACE_A), first_word_s=0.5, last_word_end_s=45.0)
    assert r["issues"] == [], r["table_md"]
    assert len(r["frames"]) >= 4


def test_the_delivered_length_is_inside_the_band_the_plan_was_checked_against(rendered):
    lo, hi = 18.0, 22.0
    assert lo <= ffmpeg.duration(rendered["out"]) <= hi


def test_a_no_broll_window_brief_renders(editor_folder, tmp_path):  # noqa: F811
    mp = pytest.MonkeyPatch()
    try:
        runner, projects, store, folder, fake = e2e.build_runner(tmp_path, mp, B.brief(["order_first", "broll_window"]), editor_folder)
        proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": ""}, {"clips": 0})
        projects.prepare_local(proj["id"], {"library": False})
        runner.make(proj["id"], {})
        assert store.get(proj["job"])["status"] == "done"
    finally:
        mp.undo()


def test_a_two_sentence_include_protects_both_sentences_when_the_cut_is_trimmed(world):
    cons = [{"id": "c1", "type": "include", "hard": True, "status": "grounded", "source": "Keep the whole section about forgiveness",
             "anchor": {"spans": [[7, 22]], "times": [[world["sents"][1].start, world["sents"][2].end]]}, "rule": {}, "candidates": [], "note": ""}]
    plan = CN.tag_satisfies({"items": [{"kind": "talk", "a": i, "b": i} for i in (1, 2)]}, cons, world["sents"], world["words"])
    assert all("c1" in (it.get("satisfies") or []) for it in plan["items"])                                       # both sentences of the section are tagged
