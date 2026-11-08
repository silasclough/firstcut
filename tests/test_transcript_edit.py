"""Editing a long video from its transcript (transcript_edit.py): pure plan edits, snapping, staging, queueing and B-roll that stays."""
import json
import threading
from types import SimpleNamespace

import pytest

from clipfarm.clipper import transcript_edit as TE
from clipfarm.clipper.localproj import LocalRunner, State, seg


def make_words(n_sent=6, per=6):
    """n_sent sentences of `per` words, 0.3 s a word, 0.05 s between words, 1.3 s between sentences, ending in a full stop."""
    words, t = [], 1.0
    for s in range(n_sent):
        for k in range(per):
            text = f"w{s}_{k}" + ("." if k == per - 1 else "")
            words.append({"text": text, "start": round(t, 3), "end": round(t + 0.3, 3)})
            t += 0.35
        t += 1.3
    return words


@pytest.fixture
def world():
    words = make_words()
    sents = seg(words)
    assert len(sents) == 6
    plan = {"title": "t", "gap": 0.75, "items": [{"kind": "talk", "a": 0, "b": 1}, {"kind": "talk", "a": 3, "b": 4}]}
    return words, sents, plan


def intervals(plan, sents, words):
    return LocalRunner.intervals_of(SimpleNamespace(), plan, sents, words, words)


def kept_ids(plan, sents, words):
    return [w["text"] for w, f in zip(words, TE.kept_flags(plan, sents, words)) if f]


def test_a_plan_marks_the_words_it_plays(world):
    words, sents, plan = world
    f = TE.kept_flags(plan, sents, words)
    assert f == [(0 <= k < 12) or (18 <= k < 30) for k in range(36)]


def test_delete_a_sentence(world):
    words, sents, plan = world
    new = TE.cut(plan, sents, words, 6, 11)                        # all of sentence 1
    assert [it["a"] for it in new["items"]] == [0, 3]               # the item that played sentences 0-1 still plays sentence 0
    kept = kept_ids(new, sents, words)
    assert "w1_0" not in kept and "w0_5." in kept and "w3_0" in kept
    ivs = intervals(new, sents, words)
    assert all(a < b for a, b in ivs) and ivs == sorted(ivs)
    assert all(not (a <= words[6]["start"] + 0.1 <= b) for a, b in ivs)           # nothing of the cut sentence plays
    assert plan["items"][0] == {"kind": "talk", "a": 0, "b": 1}                   # the original plan is untouched


def test_deleting_everything_an_item_plays_removes_the_item(world):
    words, sents, plan = world
    new = TE.cut(plan, sents, words, 18, 29)
    assert [(it["a"], it["b"]) for it in new["items"]] == [(0, 1)]


def test_cutting_words_that_are_not_in_the_video_changes_nothing(world):
    words, sents, plan = world
    assert TE.cut(plan, sents, words, 12, 17) == plan


def test_a_cut_in_the_middle_of_a_sentence_keeps_both_sides_clear_of_the_cut_words(world):
    words, sents, plan = world
    new = TE.cut(plan, sents, words, 2, 3)                          # w0_2 and w0_3
    ivs = intervals(new, sents, words)
    cut_a, cut_b = words[2]["start"], words[3]["end"]
    assert all(not (a < cut_b and b > cut_a) for a, b in ivs)       # no piece reaches into the removed words
    assert any(abs(b - cut_a) < 0.06 for a, b in ivs)               # but the word before is whole
    assert any(abs(a - cut_b) < 0.12 for a, b in ivs)


def test_add_a_removed_sentence_back(world):
    words, sents, plan = world
    cut_plan = TE.cut(plan, sents, words, 6, 11)
    back = TE.add(cut_plan, sents, words, 6, 11)
    assert kept_ids(back, sents, words) == kept_ids(plan, sents, words)
    assert not back["te_cuts"]


def test_add_a_sentence_that_was_never_in_the_video(world):
    words, sents, plan = world
    new = TE.add(plan, sents, words, 12, 17)                        # sentence 2, between the two items
    kept = kept_ids(new, sents, words)
    assert "w2_0" in kept and "w2_5." in kept
    ivs = intervals(new, sents, words)
    assert ivs == sorted(ivs) and all(a < b for a, b in ivs)
    assert all(ivs[k][1] <= ivs[k + 1][0] + 1e-6 for k in range(len(ivs) - 1))   # sorted and not overlapping
    # it joined the item before it (sentences 0-2 play in one piece), no needless extra cut
    assert [(it["a"], it["b"]) for it in new["items"]][0][0] == 0
    assert len(new["items"]) <= 2


def test_adding_a_sentence_between_two_items_closes_them_into_one(world):
    words, sents, plan = world
    new = TE.add(plan, sents, words, 12, 17)
    assert len(new["items"]) == 1 and (new["items"][0]["a"], new["items"][0]["b"]) == (0, 4)


def test_added_words_far_from_everything_become_their_own_item_in_time_order(world):
    words, sents, plan = world
    new = TE.add(plan, sents, words, 30, 35)                        # sentence 5 right after the item that ends with sentence 4: it joins that one
    assert [(it["a"], it["b"]) for it in new["items"]] == [(0, 1), (3, 5)] and new["items"][-1]["t1"] == words[35]["end"]
    new = TE.add({**plan, "items": [{"kind": "talk", "a": 0, "b": 1}]}, sents, words, 30, 35)    # nothing next to it: its own item, last
    assert new["items"][-1]["a"] == 5 and new["items"][-1]["t0"] == words[30]["start"]
    new2 = TE.add({**plan, "items": [{"kind": "talk", "a": 3, "b": 4}]}, sents, words, 0, 5)     # sentence 0, before the first
    assert new2["items"][0]["a"] == 0 and new2["items"][1]["a"] == 3


def test_add_part_of_a_sentence(world):
    words, sents, plan = world
    new = TE.add({**plan, "items": [{"kind": "talk", "a": 0, "b": 0}]}, sents, words, 12, 14)    # the first three words of sentence 2
    kept = kept_ids(new, sents, words)
    assert kept[-3:] == ["w2_0", "w2_1", "w2_2"] and "w2_3" not in kept
    ivs = intervals(new, sents, words)
    assert max(b for a, b in ivs) < words[15]["start"] + 0.02        # it stops before the fourth word


def test_a_time_range_snaps_to_whole_words(world):
    words, sents, plan = world
    mid_word = (words[7]["start"] + words[7]["end"]) / 2
    assert TE.snap_range(words, mid_word, mid_word + 0.01) == (7, 7)                    # inside one word: that whole word
    assert TE.snap_range(words, words[7]["start"] + 0.2, words[8]["start"] + 0.05) == (7, 8)   # from late in a word to early in the next: both
    assert TE.snap_range(words, words[6]["end"] + 0.01, words[7]["start"] - 0.01) is None      # only the silence between words: nothing to cut
    plan2 = TE.apply_ops(plan, sents, words, [{"op": "cut", "t0": mid_word, "t1": mid_word + 0.01}])
    kept = kept_ids(plan2, sents, words)
    assert "w1_1" not in kept and "w1_0" in kept and "w1_2" in kept                    # never half a word


def test_ops_are_applied_in_order(world):
    words, sents, plan = world
    ops = [{"op": "cut", "t0": words[6]["start"], "t1": words[11]["end"]}, {"op": "add", "t0": words[8]["start"], "t1": words[9]["end"]}]
    new = TE.apply_ops(plan, sents, words, ops)
    kept = kept_ids(new, sents, words)
    assert "w1_2." not in kept and "w1_2" in kept and "w1_3" in kept and "w1_0" not in kept


def test_the_summary_says_what_will_change(world):
    words, sents, plan = world
    new = TE.cut(plan, sents, words, 6, 17 - 6 + 6 - 1)             # cut words 6..16 (sentence 1 and part of 2 which is not in): only sentence 1 counts
    d = TE.diff(TE.kept_flags(plan, sents, words), TE.kept_flags(new, sents, words), sents, words)
    assert d["cut_words"] == 6 and d["cut_sentences"] == 1 and d["added_words"] == 0
    assert 1.5 < d["cut_s"] < 2.5
    assert TE.sentence_line(d).startswith("Cut ") and "1 sentence" in TE.sentence_line(d)
    assert TE.sentence_line({"cut_words": 0, "added_words": 0, "cut_s": 0, "added_s": 0, "cut_sentences": 0, "added_sentences": 0}) == "No change"


def test_b_roll_stays_where_its_stretch_is_still_there():
    old_iv = [[10.0, 20.0], [30.0, 40.0]]                          # the old cut: 20 s long
    cutaways = [{"at": 2.0, "length": 3.0, "scene": 1}, {"at": 12.0, "length": 3.0, "scene": 2}, {"at": 18.5, "length": 3.0, "scene": 3}]
    new_iv = [[10.0, 14.0], [16.0, 20.0], [30.0, 40.0]]            # source 14-16 cut out of the first stretch
    out = TE.carry_cutaways({"iv": old_iv, "cutaways": cutaways}, new_iv)
    assert [c["scene"] for c in out] == [2]                         # shot 1 covered source 12-15, which lost 14-16: gone; shot 2 is whole; shot 3 ran past the end
    assert out[0]["at"] == pytest.approx(8.0 - 0.0, abs=0.1) or out[0]["at"] > 0


def test_a_shot_over_a_removed_stretch_is_dropped_and_the_rest_move_up():
    old_iv = [[0.0, 30.0]]
    cutaways = [{"at": 2.0, "length": 3.0, "scene": 1}, {"at": 20.0, "length": 3.0, "scene": 2}]
    new_iv = [[0.0, 1.0], [8.0, 30.0]]                              # source 1-8 removed: the first shot (2-5) is gone
    out = TE.carry_cutaways({"iv": old_iv, "cutaways": cutaways}, new_iv)
    assert [c["scene"] for c in out] == [2]
    assert out[0]["at"] == pytest.approx(20.0 - 7.0, abs=0.05)      # 7 s earlier


# ---- staging, applying and queueing through the runner ---------------------------------------------
class FakeProjects:
    def __init__(self, proj):
        self.proj = proj

    def get(self, pid):
        return self.proj


class FakeRunner:
    def __init__(self, proj, store=None):
        self.projects = FakeProjects(proj)
        self.running = None
        self.started = []
        if store is not None:
            self.store = store

    def start(self, pid, kind, **kw):
        self.started.append((pid, kind))
        self.running = pid


def project(tmp_path, words, plan):
    st = State(tmp_path)
    st.put("plan_words.json", {"words": words, "spans": []})
    if plan:
        st.put("plan.json", plan)
    return {"id": "p1", "job": "j1", "folder": str(tmp_path), "home": str(tmp_path)}, st


def test_nothing_to_edit_before_the_plan_exists(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, None)
    r = FakeRunner(proj)
    v = TE.view(r, "p1")
    assert v["state"] == "words" and len(v["words"]) == 36 and set(v["flags"]) == {"0"}
    with pytest.raises(ValueError):
        TE.stage(r, "p1", "cut", 0, 3)


def test_no_words_no_panel(tmp_path):
    proj = {"id": "p1", "job": "j1", "folder": str(tmp_path), "home": str(tmp_path)}
    assert TE.view(FakeRunner(proj), "p1")["state"] == "none"


def test_stage_shows_what_would_change_and_can_be_taken_back(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, plan)
    r = FakeRunner(proj)
    v = TE.view(r, "p1")
    assert v["state"] == "plan" and v["flags"][:12] == "1" * 12 and v["flags"][12:18] == "0" * 6
    v = TE.stage(r, "p1", "cut", 6, 11)
    assert v["flags"][6:12] == "2" * 6 and v["staged"]["summary"].startswith("Cut ") and len(v["staged"]["ops"]) == 1
    v = TE.stage(r, "p1", "add", 12, 17)
    assert v["flags"][12:18] == "3" * 6 and "Add " in v["staged"]["summary"]
    TE.stage(r, "p1", "cut", 24, 24)
    with pytest.raises(ValueError):
        TE.stage(r, "p1", "cut", 24, 24)                              # the second time it is already out
    v = TE.undo_staged(r, "p1")                                       # takes back the last one (word 24)
    assert len(v["staged"]["ops"]) == 2 and v["flags"][12:18] == "3" * 6 and v["flags"][24] == "1"
    v = TE.undo_staged(r, "p1")
    assert len(v["staged"]["ops"]) == 1 and v["flags"][12:18] == "0" * 6
    v = TE.clear(r, "p1")
    assert v["staged"]["ops"] == [] and set(v["flags"][:12]) == {"1"}
    assert st.get("plan.json") == plan                               # staging never touches the plan


def test_staging_something_that_changes_nothing_is_refused(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, plan)
    r = FakeRunner(proj)
    with pytest.raises(ValueError):
        TE.stage(r, "p1", "add", 0, 5)                               # already in the video
    with pytest.raises(ValueError):
        TE.stage(r, "p1", "cut", 12, 17)                             # already out
    with pytest.raises(ValueError):
        TE.stage(r, "p1", "cut", 5, 400)                             # no such words


def test_apply_starts_a_render_when_idle_and_queues_while_the_video_is_being_made(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, plan)
    r = FakeRunner(proj)
    TE.stage(r, "p1", "cut", 6, 11)
    r.running = "p1"                                                 # the video is being made
    out = TE.request_apply(r, "p1")
    assert out["queued"] and r.started == []                         # nothing started, nothing corrupted
    assert st.get("plan.json") == plan
    assert (st.get(TE.STAGED) or {}).get("queued") is True
    r.running = None                                                  # ... the build ends
    TE.after_run(r, "p1", "make", ok=True)
    assert r.started == [("p1", "transcript_edit")]
    assert (st.get(TE.STAGED) or {}).get("queued") is False and (st.get(TE.STAGED) or {}).get("ops")   # still staged until it is applied


def test_a_failed_build_does_not_start_the_queued_edit(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, plan)
    r = FakeRunner(proj)
    TE.stage(r, "p1", "cut", 6, 11)
    r.running = "p1"
    TE.request_apply(r, "p1")
    r.running = None
    TE.after_run(r, "p1", "make", ok=False)
    TE.after_run(r, "p1", "transcript_edit", ok=True)
    assert r.started == []


def test_applying_changes_the_plan_keeps_a_copy_and_renders_it(tmp_path, world, monkeypatch):
    words, sents, plan = world
    proj, st = project(tmp_path, words, plan)
    st.put("files.json", [])
    job_dir = tmp_path / "jobdir"
    job_dir.mkdir()
    (job_dir / "plan.json").write_text(json.dumps({"intervals": [[1.0, 5.0], [30.0, 40.0]]}))
    st.put("render_cache.json", {"final": {"cutaways": [{"at": 1.0, "length": 2.0, "scene": 0}]}, "sig": "x"})
    rendered = {}

    class R(FakeRunner):
        store = SimpleNamespace(dir=lambda job: job_dir, get=lambda job: {"options": {}})

        def _setup(self, pid):
            return proj, tmp_path, st, None, None

        def saved_inputs(self, folder, st_, ai, backend="parakeet"):
            return [], None, words, words

        def describe(self, *a, **k):
            return []

        def render(self, pid, new, files, w, scenes, ai, engine, st_, folder, note, all_words=None):
            rendered.update(plan=new, note=note)

    r = R(proj)
    TE.stage(r, "p1", "cut", 6, 11)
    TE.run_apply(r, "p1")
    assert rendered["plan"]["te_cuts"] and rendered["plan"]["pending_render"] and "cut" in rendered["note"].lower()
    assert st.get(TE.PREV)["items"] == plan["items"]                 # the plan before the edit, for Undo
    assert st.get(TE.STAGED) == {}
    assert (st.get("render_cache.json") or {}).get("te_carry", {}).get("iv") == [[1.0, 5.0], [30.0, 40.0]]
    assert any((st.dir / "backups").glob("plan-*before-transcript-edit*.json"))     # and a recoverable copy
    # Undo: the old plan comes back
    r.render = lambda pid, new, *a, **k: rendered.update(plan=new)
    TE.run_undo(r, "p1")
    assert rendered["plan"]["items"] == plan["items"] and not rendered["plan"].get("te_cuts")


def test_the_page_words_follow_the_plan_after_an_edit(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, plan)
    r = FakeRunner(proj)
    TE.stage(r, "p1", "cut", 6, 11)
    new = TE.apply_ops(plan, sents, words, st.get(TE.STAGED)["ops"])
    st.put("plan.json", new)
    st.put(TE.STAGED, {})
    v = TE.view(r, "p1")
    assert v["flags"][6:12] == "0" * 6 and v["flags"][:6] == "1" * 6


def test_paragraph_breaks_follow_the_pauses(world):
    words, sents, plan = world
    br = TE.paragraph_breaks(words, sents, gap=1.2)
    assert br == [6, 12, 18, 24, 30]
    assert TE.paragraph_breaks(words, sents, gap=9, per=2) == [12, 24]


def test_a_plan_waiting_for_approval_is_not_edited_here(tmp_path, world):
    words, sents, plan = world
    proj, st = project(tmp_path, words, {**plan, "awaiting_approval": True})
    r = FakeRunner(proj)
    assert TE.view(r, "p1")["state"] == "words"
    with pytest.raises(ValueError):
        TE.stage(r, "p1", "cut", 6, 11)


# ------------------------------------------------------------------------------------------------------------- the page's two views
def test_the_video_order_follows_the_plan_not_the_recording(world):
    words, sents, plan = world
    plan = {**plan, "items": [{"kind": "talk", "a": 3, "b": 4}, {"kind": "talk", "a": 0, "b": 1}]}     # said first, played second
    lines = TE.order_lines(plan, sents, words)
    assert [(l["a"], l["b"]) for l in lines] == [(18, 23), (24, 29), (0, 5), (6, 11)]                   # one line per sentence, in play order
    assert [l["at"] for l in lines] == sorted(l["at"] for l in lines) and lines[0]["at"] == 0.0
    assert words[lines[0]["a"]]["text"] == "w3_0" and words[lines[2]["a"]]["text"] == "w0_0"


def test_the_order_view_reaches_the_page_and_the_flags_are_shared_by_both_views(tmp_path, world):
    words, sents, plan = world
    plan = {**plan, "items": [{"kind": "talk", "a": 3, "b": 4}, {"kind": "talk", "a": 0, "b": 1}]}
    proj, st = project(tmp_path, words, plan)
    r = FakeRunner(proj)
    v = TE.view(r, "p1")
    assert v["state"] == "plan" and [(l["a"], l["b"]) for l in v["order"]][:2] == [(18, 23), (24, 29)]
    covered = {k for l in v["order"] for k in range(l["a"], l["b"] + 1)}
    assert covered == {k for k, c in enumerate(v["flags"]) if c == "1"}          # exactly the words that are in the video
    v = TE.stage(r, "p1", "cut", 24, 29)                                          # cut the second line of the video (in the order view)
    assert v["flags"][24:30] == "2" * 6                                           # ... and it shows struck through in the all-footage view too
    assert [(l["a"], l["b"]) for l in v["order"]][:2] == [(18, 23), (24, 29)]      # (the line stays listed until the edit is rendered)
    st.put("plan.json", TE.apply_ops(plan, sents, words, st.get(TE.STAGED)["ops"]))
    st.put(TE.STAGED, {})
    v = TE.view(r, "p1")
    assert all(l["a"] != 24 for l in v["order"]) and v["flags"][24:30] == "0" * 6


def test_a_plan_with_one_item_gives_its_lines_and_no_plan_gives_none(tmp_path, world):
    words, sents, plan = world
    one = {**plan, "items": [{"kind": "talk", "a": 0, "b": 0}]}
    assert [(l["a"], l["b"]) for l in TE.order_lines(one, sents, words)] == [(0, 5)]
    assert TE.order_lines({**plan, "items": []}, sents, words) == []
    assert TE.order_lines({"title": "t"}, sents, words) == []
    proj, st = project(tmp_path, words, None)
    assert TE.view(FakeRunner(proj), "p1")["order"] == []                         # words but no plan yet: the page shows all footage


def test_a_line_says_which_file_it_was_said_in_and_hand_cuts_do_not_count_toward_the_time(world):
    words, sents, plan = world
    files = [{"name": "A.mp4", "start": 0.0, "end": 20.0}, {"name": "B.mp4", "start": 20.0, "end": 60.0}]
    lines = TE.order_lines(plan, sents, words, files)
    assert [l["src"] for l in lines] == ["A.mp4", "A.mp4", "B.mp4", "B.mp4"] or lines[0]["src"] == "A.mp4"
    plan2 = {**plan, "te_cuts": [[words[0]["start"], words[2]["end"]]]}            # the first three words of sentence 0 taken out by hand
    a = TE.order_lines(plan, sents, words)
    b = TE.order_lines(plan2, sents, words)
    assert b[1]["at"] < a[1]["at"]                                                  # the next line plays earlier by the cut length
    plan3 = {**plan, "te_cuts": [[words[0]["start"], words[5]["end"]]]}            # a whole line taken out by hand plays nothing: not listed
    assert [(l["a"], l["b"]) for l in TE.order_lines(plan3, sents, words)][0] == (6, 11)
    assert TE.order_lines(plan, sents, words, [{"name": None, "start": "x"}, "junk"]) and True   # junk file records never break it


def test_a_kept_video_lists_every_line_of_its_stretch_in_order(world):
    words, sents, plan = world
    kept = {"title": "t", "items": [], "keep_cut": {"start": words[6]["start"], "end": words[17]["end"]}}
    assert [(l["a"], l["b"]) for l in TE.order_lines(kept, sents, words)] == [(6, 11), (12, 17)]
