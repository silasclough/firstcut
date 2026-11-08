"""The plan before a long video renders and what was checked after, in plain words (WP16 of docs/plans/editor-improvement-plan.md)."""
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import planapi as PA
from clipfarm.clipper import planreview as PR
from clipfarm.clipper import styles_profile as SPF
from clipfarm.clipper.localproj import State, seg

HTML = Path(__file__).resolve().parents[1] / "clipfarm" / "clipper" / "web" / "studio.html"
LINES = ["We started the program because students kept asking where to find help.", "Honestly, nobody knew who to call when something went wrong.",
         "So we built one place for all of it.", "It took about two years, and a lot of coffee.", "The first month was chaos.", "Then something clicked.",
         "A student walked in, sat down, and said this is exactly what I needed.", "That was the moment we knew it would work."]


def _words():
    words, t = [], 2.0
    for n in range(48):
        for w in LINES[n % 8].split():
            words.append({"text": w, "start": round(t, 2), "end": round(t + 0.3, 2)})
            t += 0.34
        t += 1.3
    return words


WORDS = _words()
SENTS = seg(WORDS)
FILES = [{"name": "talk_a.mp4", "start": 0.0, "end": 330.0, "dur": 330.0, "facts": {"role": "a_roll"}},
         {"name": "C0021.MP4", "start": 330.0, "end": 390.0, "dur": 60.0, "facts": {"role": "broll"}},
         {"name": "photo1.jpg", "start": 390.0, "end": 394.0, "dur": 4.0, "facts": {"role": "photo"}}]


def _first_word(k):
    return sum(len(LINES[n % 8].split()) for n in range(k))


def _tm(i):
    return [round(SENTS[i].start, 1), round(SENTS[i].end, 1)]


def _plan(first_month_required=True):
    cons = [
        {"id": "c1", "type": "include", "hard": True, "source": "start with the part about the first month", "status": "grounded",
         "anchor": {"times": [_tm(4)], "texts": ["The first month was chaos."], "words": [_first_word(4), _first_word(5) - 1], "spans": [[_first_word(4), _first_word(5) - 1]]}, "rule": {"text": "keep"}, "candidates": [], "note": ""},
        {"id": "c2", "type": "exclude_words", "hard": True, "source": "leave out the story about the parking lot", "status": "ambiguous", "anchor": {}, "rule": {"text": "leave out"},
         "candidates": [{"words": [_first_word(8), _first_word(10) - 1], "start_s": _tm(8)[0], "end_s": _tm(9)[1], "text": "We started the program because students kept asking."},
                        {"words": [_first_word(30), _first_word(32) - 1], "start_s": _tm(30)[0], "end_s": _tm(31)[1], "text": "Honestly, nobody knew who to call"}], "note": "I found two places this could mean."},
        {"id": "c4", "type": "camera", "hard": False, "source": "use the second camera when you can", "status": "unmet", "anchor": {}, "rule": {}, "candidates": [],
         "note": "There is only one camera in this folder."},
    ]
    items = [{"id": f"i{n}", "kind": "talk", "a": a, "b": b, "satisfies": (["c1"] if a <= 4 <= b else [])} for n, (a, b) in enumerate([(0, 3), (4, 5), (6, 12), (20, 24), (30, 36)])]
    return {"title": "Our first year", "gap": 0.75, "items": items, "version": 1, "constraints": cons, "awaiting_approval": True,
            "approval": {"reason": "Some of your instructions need a look before I render.", "questions": ["I found two places this could mean."]},
            "style_obj": SPF.build("story"), "style": "story",
            "pool": [{"sid": f"s{i}", "file": "C0021.MP4", "fstart": i, "fend": i + 3, "what": "x", "quality": 6} for i in range(14)],
            "alt_takes": [{"group": "g1", "a": 13, "b": 17}], "take_groups": [{"id": "g1", "takes": [{"a": 6, "b": 12}, {"a": 13, "b": 17}]}],
            "post_check": [{"ok": True, "what": "the part about the first month is in the video"}, {"ok": False, "what": "the video is 3:48 long, not 3 minutes"}]}


@pytest.fixture()
def runner(tmp_path, monkeypatch):
    home = tmp_path / "proj"
    home.mkdir()
    st = State(home)
    st.put("plan.json", _plan())
    proj = {"id": "p1", "job": "j1", "folder": str(home), "home": str(home), "notes_info": {"issues": [
        {"level": "Look", "where": "00:41", "start": 41, "what": "B-roll stayed on the speaker at 41 s: nothing matched 'the oven'", "do": ""},
        {"level": "Look", "where": "01:12", "start": 72, "what": "A B-roll shot was dropped at 72 s: its clip ended", "do": ""},
        {"level": "Fix", "where": "02:05", "start": 125, "what": "The lower third overlaps the caption band", "do": ""}]}}
    monkeypatch.setattr(PA, "_inputs", lambda r, pid: (proj, home, State(home), None, {"status": "approval", "options": {}}, FILES, WORDS, SENTS))
    return SimpleNamespace(projects=SimpleNamespace(get=lambda pid: proj), _setup=lambda pid: (proj, home, State(home), None, None))


def test_the_plan_review_api_returns_constraints_segments_broll_and_options_in_plain_words(runner):
    r = PR.review(runner, "p1")
    assert r["summary"].startswith("I will make a ") and "5 parts" in r["summary"]
    assert [u["state"] for u in r["understood"]] == ["will do", "which one?", "could not"]
    assert r["understood"][0]["quote"] == "start with the part about the first month" and r["understood"][0]["must"] is True
    assert len(r["questions"]) == 1 and [c["id"] for c in r["questions"][0]["choices"]] == ["0", "1"] and r["ready"] is False
    assert [row["at"] for row in r["cut"]][0] == "0:00" and r["cut"][1]["required"] and "first month" in r["cut"][1]["why"]
    assert r["cut"][0]["file"] == "talk_a.mp4" and r["cut"][0]["words"].startswith("We started the program")
    assert r["cut"][2]["other_takes"] and r["cut"][2]["other_takes"][0]["a"] == 13
    assert r["broll"]["available"] == 14 and "B-roll" in r["broll"]["line"]
    assert r["footage"]["files"] == 3 and "1 person talking" in r["footage"]["line"] and "1 photo" in r["footage"]["line"] and "1 B-roll shot" in r["footage"]["line"]
    assert r["options"]["pace"] in ("slow", "natural", "tight", "fast") and {x["id"] for x in r["options"]["paces"]} == {"slow", "natural", "tight", "fast"}
    assert 5 <= len(r["style"]) + 1 <= 12 and r["title"] == "Our first year"
    for t in PR.texts(r):                                                     # no internal word reaches the page
        assert not PR.INTERNAL.search(t), t


def test_choosing_a_candidate_resolves_the_ambiguous_constraint_and_replans(runner):
    r = PR.answer(runner, "p1", "c2", 1)
    assert r["questions"] == [] and r["ready"] is True
    assert all(row["words"] and not row["words"].startswith("Honestly, nobody knew who to call") or row["kind"] != "talk" for row in r["cut"] if not row.get("left_out"))
    assert [u["state"] for u in r["understood"]][1] == "will do"
    st = State(Path(runner.projects.get("p1")["home"]))
    saved = st.get("plan.json")
    c2 = next(c for c in saved["constraints"] if c["id"] == "c2")
    assert c2["status"] == "grounded" and c2["anchor"]["times"] and not saved.get("approval", {}).get("questions")
    assert saved["awaiting_approval"] is True                                  # (the person still has to press Make)
    with pytest.raises(ValueError):
        PR.answer(runner, "p1", "c2", 7)
    with pytest.raises(ValueError):
        PR.answer(runner, "p1", "nope", 0)


def test_leaving_out_a_segment_updates_the_plan_and_the_validator_runs_again(runner):
    r = PR.leave_out(runner, "p1", "i3", True)
    assert not any(row["id"] == "i3" and not row.get("left_out") for row in r["cut"]) and any(row["id"] == "i3" and row.get("left_out") for row in r["cut"])
    assert r["summary"].count("4 parts") == 1
    st = State(Path(runner.projects.get("p1")["home"]))
    assert [it["id"] for it in st.get("plan.json")["items"]] == ["i0", "i1", "i2", "i4"]
    with pytest.raises(ValueError) as e:                                       # something that was asked for cannot be left out
        PR.leave_out(runner, "p1", "i1", True)
    assert "cannot be left out" in str(e.value) and not PR.INTERNAL.search(str(e.value))
    back = PR.leave_out(runner, "p1", "i3", False)
    assert [row["id"] for row in back["cut"] if not row.get("left_out")] == ["i0", "i1", "i2", "i3", "i4"]
    assert [it["id"] for it in State(Path(runner.projects.get("p1")["home"])).get("plan.json")["items"]] == ["i0", "i1", "i2", "i3", "i4"]


def test_a_part_that_was_left_out_stays_where_it_was_in_the_list(runner):
    r = PR.leave_out(runner, "p1", "i2", True)
    ids = [row["id"] for row in r["cut"]]
    assert ids.index("i2") == 2 and [row["id"] for row in r["cut"] if row.get("left_out")] == ["i2"]


def test_using_another_take_swaps_the_passage(runner):
    r = PR.use_take(runner, "p1", "i2", 13, 17)
    row = next(x for x in r["cut"] if x["id"] == "i2")
    assert row["words"].startswith(SENTS[13].text.split()[0])
    assert row["other_takes"] and (row["other_takes"][0]["a"], row["other_takes"][0]["b"]) == (6, 12)
    with pytest.raises(ValueError):
        PR.use_take(runner, "p1", "i2", 1, 2)


def test_the_after_render_report_lists_checks_and_dropped_shots_with_times(runner):
    c = PR.checked(runner, "p1")
    assert any(not x["ok"] and "3:48" in x["what"] for x in c["checks"]) and any(x["ok"] for x in c["checks"])
    assert c["skipped"][0]["start"] == 41 and "stayed on the speaker" in c["skipped"][0]["line"]
    assert c["dropped"][0]["start"] == 72 and "dropped" in c["dropped"][0]["line"]
    assert c["placement"][0]["start"] == 125
    assert "1 check did not pass" in c["summary"]
    for t in PR.texts(c):
        assert not PR.INTERNAL.search(t), t


def test_clip_rubric_is_in_the_clip_record_and_the_page_data():
    clip = {"rubric": {"hook": 5, "flow": 3, "value": 4, "why_hook": "Opens on the claim.", "why_flow": "Repeats a point.", "why_value": "A concrete story."},
            "qa": {"issues": [{"level": "warn", "what": "no face in the first second"}]}}
    n = PR.clip_notes(clip)
    assert (n["hook"], n["flow"], n["value"]) == (5, 3, 4) and n["why"]["hook"] == "Opens on the claim." and n["warnings"] == ["no face in the first second"]
    assert PR.clip_notes({}) == {} and PR.clip_notes({"rubric": {"hook": float("inf"), "flow": "x", "value": None}})["hook"] is None
    from clipfarm.clipper import studio_web as SW
    api = SW.StudioAPI.__new__(SW.StudioAPI) if hasattr(SW, "StudioAPI") else None
    if api is not None:
        v = api.short_view({"id": "s1", "title": "t", "status": "done", "clips": [{"index": 0, "file": "a.mp4", "thumb": "a.jpg", "duration": 20, **clip}]})
        assert v["clips"][0]["notes"]["hook"] == 5


def test_a_review_of_a_plan_with_nothing_in_it_is_still_plain(runner):
    home = Path(runner.projects.get("p1")["home"])
    State(home).put("plan.json", {"title": "", "items": [], "constraints": [], "style": "story"})
    r = PR.review(runner, "p1")
    assert r["cut"] == [] and r["understood"] == [] and r["ready"] is True and r["summary"].startswith("I will make a ")


def test_the_page_has_a_hidden_rule_one_accent_and_no_gradients():
    h = HTML.read_text()
    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important", h)
    assert not re.search(r"background(-image)?\s*:[^;{}]*gradient", h)           # (a mask on the waveform is not a decoration)
    root = re.search(r":root\s*\{(.*?)\}", h, re.S).group(1)
    assert len(re.findall(r"--accent\s*:", root)) == 1
    assert 'id="l-approve"' in h and 'id="plan-slot"' in h and "What I checked" in h and "Leave out" in h


def test_the_review_card_uses_no_internal_names_in_the_page_text():
    h = HTML.read_text()
    a, b = h.index("function reviewHTML"), h.index("function reviewMount")
    for word in ("HSR", "style_obj", "ungrounded", "unmet", "abstain", "WP5", "constraint_id"):
        assert word not in h[a:b], word


def test_plan_first_stops_a_normal_video_at_the_plan_and_pressing_make_renders_it(tmp_path, monkeypatch):
    from test_constraints import _runner
    runner, projects, store, folder, fake = _runner(tmp_path, monkeypatch, {"constraints_extract": {"constraints": []}})
    (folder / "notes.md").write_text("Make it punchy.")
    proj = projects.create_local(str(folder), "", "", {"transcriber": "openai", "music": ""}, {"clips": 0})
    projects.prepare_local(proj["id"], {"plan_first": True, "library": False})
    runner.make(proj["id"], {})
    job = store.get(proj["job"])
    assert job["status"] == "approval" and not list((folder / "Output").glob("*.mp4"))
    import json
    plan = json.loads((folder / ".aivideo" / "plan.json").read_text())
    assert plan["awaiting_approval"] and not plan.get("dry_run")                 # (not a dry run: pressing Make renders this very plan)
    r = PR.review(runner, proj["id"])
    assert r["awaiting"] and r["cut"] and r["ready"] and r["summary"].startswith("I will make a ")
    for t in PR.texts(r):
        assert not PR.INTERNAL.search(t), t
    runner.approve(proj["id"])
    assert store.get(proj["job"])["status"] == "done" and list((folder / "Output").glob("*.mp4"))
    c = PR.checked(runner, proj["id"])
    assert "summary" in c and isinstance(c["checks"], list)


def test_the_plan_first_option_comes_through_the_options_the_page_sends():
    h = HTML.read_text()
    assert 'k: "plan_first"' in h and "Show me the plan first" in h
