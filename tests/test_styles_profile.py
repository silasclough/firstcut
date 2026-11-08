"""WP4: one style object per video (clipper/styles_profile.py) and what reads it. Offline; the AI is a fake."""
import json

import pytest

from clipfarm.clipper import localproj as LP
from clipfarm.clipper import styles_profile as SPF
from clipfarm.clipper.engine import PACES


class _R:
    def __init__(self, data):
        self.data = data


class FakeAI:
    """complete_json answers by task name (like the fakes in test_graphics.py)."""
    def __init__(self, **answers):
        self.answers, self.calls = answers, []

    def complete_json(self, task, system, prompt, schema, validate=None, images=None, **kw):
        self.calls.append(task)
        a = self.answers[task]
        if isinstance(a, Exception):
            raise a
        return _R(a)


def _pool(n=12):
    return [{"start": 100.0 + i * 10, "end": 108.0 + i * 10, "quality": 7, "what": f"shot {i}", "file": f"B{i}.MP4"} for i in range(n)]


def test_every_profile_resolves_to_a_full_valid_object():
    assert set(SPF.DEFAULTS) == set(SPF.PROFILES) and len(SPF.PROFILES) == 8
    for p in SPF.PROFILES:
        obj = SPF.resolve(p)
        assert obj["profile"] == p and obj["_set"] == []
        assert SPF.validate(obj) == [], p
    assert SPF.validate({"profile": "x"})                                               # a bare object is not valid
    bad = SPF.resolve("reel", {"broll": {"share": 3}})
    assert SPF.validate(bad) == ["broll.share must be between 0 and 1"]


def test_the_profile_table_matches_the_plan():
    assert SPF.DEFAULTS["interview_clip"]["broll"]["share"] == 0.05 and SPF.DEFAULTS["interview_clip"]["music"]["policy"] == "none"
    assert SPF.DEFAULTS["seminar"]["pace"]["max_hold_s"] == 30.0 and SPF.DEFAULTS["seminar"]["music"]["policy"] == "none"
    assert SPF.DEFAULTS["short_talk"]["pace"]["max_hold_s"] == 5.0 and SPF.DEFAULTS["short_talk"]["takes"]["policy"] == "last"
    assert SPF.DEFAULTS["reel"]["broll"]["share"] == 0.8 and SPF.DEFAULTS["event_recap"]["music"]["policy"] == "throughout"
    assert SPF.from_kind("story") == "longform_talk" and SPF.from_kind("youtube", is_clip=True) == "short_talk"
    assert SPF.from_kind("business") == "longform_talk" and SPF.kind_overrides("business")["graphics"]["policy"] == "all"
    assert SPF.from_kind("youtube") == "longform_talk" and SPF.kind_overrides("youtube")["pace"]["max_hold_s"] == 10.0
    assert SPF.from_kind("interview") == "longform_talk" and SPF.from_kind("interview", is_clip=True) == "interview_clip"


def test_notes_set_only_what_they_say():
    ai = FakeAI(style_from_notes={"profile": None, "pace": "fast", "captions": None, "graphics": None, "music": "none", "broll_share": None,
                                  "chronology": None, "remove_interviewer": None, "voice_over": None, "target_seconds": None})
    got = SPF.from_notes(ai, "make it fast-paced and use no music", "story")
    assert got == {"pace": "fast", "music": "none"} and ai.calls == ["style_from_notes"]            # ONE call; only what is stated
    obj = SPF.build("story", got)
    assert obj["pace"]["name"] == "fast" and obj["music"]["policy"] == "none"
    assert obj["captions"] == SPF.DEFAULTS["longform_talk"]["captions"] and obj["broll"]["share"] == SPF.DEFAULTS["longform_talk"]["broll"]["share"]
    assert obj["_set"] == ["music.policy", "pace.name"]
    assert SPF.from_notes(ai, "   ", "story") == {} and len(ai.calls) == 1                             # no notes: no call


def test_the_answer_of_the_ai_is_cleaned_and_the_length_is_read_with_it():
    ai = FakeAI(style_from_notes={"profile": "seminar", "pace": "warp", "captions": None, "graphics": "title", "music": None, "broll_share": 7,
                                  "chronology": None, "remove_interviewer": "never", "voice_over": False, "target_seconds": 480})
    got = SPF.from_notes(ai, "a sermon, about eight minutes, keep both speakers", None)
    assert got == {"profile": "seminar", "graphics": "title", "remove_interviewer": "never", "voice_over": False, "target_seconds": 480.0}   # warp / 7 dropped
    obj = SPF.build("story", got)
    assert obj["profile"] == "seminar" and obj["length"]["target_s"] == 480.0 and obj["speakers"]["remove_interviewer"] == "never"


def test_a_sermon_is_a_seminar_profile_by_regex_when_the_ai_fails():
    ai = FakeAI(style_from_notes=RuntimeError("offline"))
    got = SPF.from_notes(ai, "A sermon from Sunday. Fast-paced, no music, keep it in order, keep both speakers.", "story")
    assert got.pop("_fallback") is True
    assert got == {"profile": "seminar", "pace": "fast", "music": "none", "chronology": "keep", "remove_interviewer": "never"}
    assert SPF.build("story", got)["profile"] == "seminar"
    assert SPF.regex_settings("a podcast about gardens")["profile"] == "interview_clip"
    assert SPF.regex_settings("an event recap with highlights")["profile"] == "event_recap"
    assert SPF.regex_settings("a plain story about my grandfather") == {}


def test_options_beat_notes_beat_points_beat_defaults():
    points = {"pace": "slow", "music": "throughout", "captions": "phrase"}
    notes = {"pace": "tight", "music": "none"}
    options = {"pace": "fast"}
    obj = SPF.build("story", notes, points, options)
    assert obj["pace"]["name"] == "fast"                    # the option
    assert obj["music"]["policy"] == "none"                 # the notes beat the points
    assert obj["captions"]["style"] == "phrase"             # the points beat the default ("none")
    assert obj["graphics"]["policy"] == SPF.DEFAULTS["longform_talk"]["graphics"]["policy"]
    # a profile the options choose (`kind` given on the page) beats the notes' profile
    assert SPF.build("promo", {"profile": "seminar"}, kind_chosen=True)["profile"] == "promo"
    assert SPF.build("promo", {"profile": "seminar"})["profile"] == "seminar"


def test_what_a_legacy_kind_adds_is_not_counted_as_stated():
    y = SPF.build("youtube")
    assert y["pace"]["name"] == "tight" and y["pace"]["max_hold_s"] == 10.0 and y["_set"] == []      # (so a render does not change pace for it)
    assert not SPF.stated(y, "pace.name")
    assert SPF.stated(SPF.build("youtube", {"pace": "slow"}), "pace.name")


def test_the_legacy_numbers_are_unchanged_by_the_style_object():
    """DEFAULTS of the profiles the code already renders hold today's numbers, so nothing changes until a person states something."""
    for kind, per in (("story", 18.0), ("business", 18.0), ("youtube", 10.0), ("promo", 6.0), ("ad", 3.0)):
        assert SPF.per_s(SPF.build(kind)) == per == float(LP.PROFILES[kind]["cut_per_s"]), kind
    for kind in ("story", "promo", "ad", "youtube", "business"):
        s = SPF.build(kind)
        assert SPF.show_s(s) == 3.2 and (s["pace"]["max_hold_s"] == 12.0 or kind == "youtube")
    assert SPF.get(SPF.build("story"), "broll.shot_s") == [1.5, 4.5] and SPF.get(SPF.build("ad"), "broll.shot_s") == [1.2, 3.2]
    assert SPF.get(SPF.build("story"), "multicam.min_hold_s") == 6.0 and SPF.get(SPF.build("story"), "multicam.baseline_hold_s") == [9.0, 12.0]
    assert LP.plan_style({"style": "ad"})["profile"] == "ad" and LP.plan_style({})["profile"] == "longform_talk"      # an old plan has no style_obj
    assert SPF.per_s(SPF.build("story", {"broll_share": 0.5})) == 6.0                    # a stated share replaces the profile's own figure (3.0 / 0.5)


def test_a_new_kind_has_a_profile_and_the_regex_finds_it():
    assert {"seminar", "interview", "recap"} <= set(LP.KINDS) and all(k in LP.PROFILES for k in LP.KINDS)

    class Down:
        def complete_json(self, *a, **k):
            raise RuntimeError("offline")
    assert LP.video_kind(Down(), "A sermon preached on Sunday") == "seminar"
    assert LP.video_kind(Down(), "a podcast with two hosts") == "interview"
    assert LP.video_kind(Down(), "a recap of the weekend") == "recap"
    assert LP.video_kind(Down(), "a plain story about my grandfather") == "story"


def test_capabilities_lists_profiles_limits_and_checks():
    c = SPF.capabilities()
    assert c["profiles"] == list(SPF.PROFILES) and c["limits"] == {"min_clip_s": 7, "max_minutes": 120}
    assert "karaoke" in c["caption_styles"] and "none" in c["graphics_policies"] and "glass" in c["graphics_themes"]
    assert c["style_schema"]["broll"]["share"] is not None and c["defaults"]["seminar"]["pace"]["max_hold_s"] == 30.0
    assert isinstance(c["checks"], list)
    json.dumps(c)                                                                         # (goes to the AI as JSON)


def test_describe_writes_the_style_as_short_plain_lines():
    s = SPF.describe(SPF.build("promo"))
    lines = s.splitlines()
    assert len(lines) == 5 and "65 %" in lines[1] and "never over the hook" in lines[1] and "phrases" in lines[2] and lines[4] == "Music throughout."
    assert "No music." in SPF.describe(SPF.build("story", {"music": "none"}))


def test_the_options_style_is_checked_and_a_partial_can_be_applied():
    new, problems = SPF.apply_partial(SPF.build("story"), {"pace": {"name": "fast"}, "broll": {"share": 0.2}})
    assert not problems and new["pace"]["name"] == "fast" and new["broll"]["share"] == 0.2 and new["broll"]["shot_s"] == [1.5, 4.5]
    assert {"pace.name", "broll.share"} <= set(new["_set"])
    same, problems = SPF.apply_partial(SPF.build("story"), {"broll": {"share": 9}})
    assert problems == ["broll.share must be between 0 and 1"] and same["broll"]["share"] == 0.30
    seminar, problems = SPF.apply_partial(new, {"profile": "seminar"})
    assert seminar["profile"] == "seminar" and seminar["pace"]["name"] == "fast" and seminar["pace"]["max_hold_s"] == 30.0     # what was stated stays
    assert SPF.flat_from_nested({"pace": {"name": "fast"}, "music": {"policy": "none"}, "broll": {"share": 0.2}}) == \
        {"pace": "fast", "music": "none", "broll_share": 0.2}


def test_keep_moving_reads_max_hold_from_the_style():
    seminar, short = SPF.build("story", {"profile": "seminar"}), SPF.resolve("short_talk")
    hold = lambda s: float(SPF.get(s, "pace.max_hold_s"))                                  # noqa: E731
    assert hold(seminar) == 30.0 and hold(short) == 5.0
    cut, _ = LP.keep_moving([], [], [(0.0, 20.0)], _pool(), None, None, 20.0, limit=hold(seminar), show=SPF.show_s(seminar))
    assert cut == []                                                                       # a seminar speaker may stay 20 s
    cut, _ = LP.keep_moving([], [], [(0.0, 6.5)], _pool(), None, None, 6.5, limit=hold(short), show=SPF.show_s(short))
    assert len(cut) >= 1 and cut[0]["at"] < 6.0                                            # a short clip changes picture inside 6 s
    cut, _ = LP.keep_moving([], [], [(0.0, 20.0)], _pool(), None, None, 20.0)               # (today's default: 12 s)
    assert len(cut) >= 1


def test_the_planners_prompts_carry_the_styles_numbers():
    from clipfarm.clipper import broll as BR
    from clipfarm.clipper import multicam as MC
    assert BR.plan_system() == BR.PLAN_SYSTEM and MC.director_system() == MC.DIRECTOR_SYSTEM            # (the defaults are the text that was there)
    t = BR.plan_system(30.0, 25.0, 2.5, 5.0)
    assert "at least every 30 seconds" in t and "about every 21 to 28 seconds" in t and "2.5 to 5 seconds" in t and "12 seconds" not in t
    d = MC.director_system(5.0, 15.0, 30.0)
    assert "every 15 to 30 seconds (never more than 30 seconds" in d and "at least 5 seconds" in d


def test_plan_cutaways_asks_with_the_styles_shot_length(monkeypatch):
    from clipfarm.clipper import broll as BR
    from clipfarm.clipper import moments as M
    seen = {}

    class AI:
        def complete_json(self, task, system, prompt, schema, **kw):
            seen["system"] = system
            return _R({"cutaways": [{"at": 6.0, "scene": 0, "length": 9.0}]})
    sents = [M.Sentence(0, 0.0, 12.0, "We show the old library on the hill today.")]
    out = BR.plan_cutaways(AI(), sents, [{"start": 0.0, "end": 8.0, "what": "library", "quality": 8, "file": "B.MP4"}], "", 60.0, 18.0,
                           cmin=2.5, cmax=5.0, hold=30.0)
    assert out and out[0]["length"] == 5.0 and "2.5 to 5 seconds" in seen["system"]                 # clamped to the style's longest shot
    out = BR.plan_cutaways(AI(), sents, [{"start": 0.0, "end": 8.0, "what": "library", "quality": 8, "file": "B.MP4"}], "", 60.0, 18.0)
    assert out and out[0]["length"] == 4.5                                                          # (no style: today's 4.5)


def test_qa_holds_the_picture_to_the_styles_limit(monkeypatch):
    from pathlib import Path

    from clipfarm.clipper import qa
    monkeypatch.setattr(qa.ffmpeg, "duration", lambda p: 60.0)
    monkeypatch.setattr(qa, "loud_and_peaks", lambda v: (-14.0, -3.0, []))
    monkeypatch.setattr(qa, "silences", lambda v: [])
    monkeypatch.setattr(qa, "cut_times", lambda v: [10.0, 40.0])                                    # one picture for 30 s
    monkeypatch.setattr(qa, "frozen", lambda v: [])
    monkeypatch.setattr(qa, "camera_health", lambda c: [])
    rule = lambda r: [i["what"] for i in r["issues"] if "the same picture" in i["what"]]            # noqa: E731
    assert "the rule is 12" in rule(qa.check(Path("x.mp4")))[0]
    assert rule(qa.check(Path("x.mp4"), max_hold=30.0)) == []                                       # a seminar may hold a picture 30 s (+2)
    assert "the rule is 5" in rule(qa.check(Path("x.mp4"), max_hold=5.0))[0]


# ----------------------------------------------------------------------------------------------------------------------------- the brief
def test_brief_for_never_cuts_a_rule_in_half_and_puts_the_style_and_notes_before_the_rules(tmp_path):
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper.engine import Engine
    from clipfarm.core.config import load_config
    from clipfarm.clipper.store import Store
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True}
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    rules = "\n".join(f"- point number {i}: " + "word " * 12 for i in range(30))
    projects.rules_path().write_text(rules + "\n")
    proj = {"notes": "A two minute recap of the weekend."}
    full = projects.brief(proj)
    cut = projects.brief_for(proj, "hook", 600, style=SPF.build("promo"))
    assert len(cut) <= 600 and cut.startswith("HOW THIS VIDEO IS EDITED:") and cut.index("two minute recap") < cut.index("point number")
    assert all(ln in full or ln in SPF.describe(SPF.build("promo")) or ln.startswith("HOW THIS") or not ln for ln in cut.splitlines())    # whole lines only
    assert cut.rstrip().endswith("word") and "point number" in cut.splitlines()[-1] and "word word" in cut.splitlines()[-1]
    assert projects.brief_for(proj, "hook", 100000) .endswith(full.splitlines()[-1])                  # nothing cut when it fits
    assert SPF.fit_brief("one\n\ntwo\n\nthree", 9) == "one\n\ntwo"                                    # at the end of a paragraph


def test_points_are_read_once_and_kept_for_the_planner(tmp_path):
    from clipfarm.clipper import aivideo as AV
    from clipfarm.clipper.engine import Engine
    from clipfarm.core.config import load_config
    from clipfarm.clipper.store import Store
    cfg = load_config()
    cfg.root = tmp_path
    cfg.data["paths"].update({"state": str(tmp_path / "state")})
    cfg.data["clipper"]["ai_videos"] = {"dir": str(tmp_path / "x"), "local": True, "data_dir": "ai", "local_projects": True}
    store = Store(tmp_path / "jobs")
    projects = AV.Projects(cfg, store, Engine(cfg, store))
    assert projects.points_style() == {} and projects.points_sig() == ""
    projects.set_points_style({"pace": "fast", "music": "none", "_fallback": True}, "abc")
    assert projects.points_style() == {"pace": "fast", "music": "none"} and projects.points_sig() == "abc"
    from clipfarm.clipper import studio as SD
    cl = SD.clean({"style_overrides": {"long": {"pace": "fast", "bogus": 1}}})
    assert cl["style_overrides"]["long"]["pace"] == "fast" and cl["style_overrides"]["long"]["music"] is None and "bogus" not in cl["style_overrides"]["long"]
    with pytest.raises(ValueError):
        SD.clean({"style_overrides": {"long": "fast"}})
    assert SD.defaults()["style_overrides"] == {"short": {}, "long": {}}


# --------------------------------------------------------------------------------------------------------- change_video: pace is a style
def test_change_video_pace_changes_the_style_and_the_gap_not_only_the_prose(tmp_path):
    class Proj:
        def get(self, pid):
            return {"folder": str(tmp_path), "id": pid}
    r = LP.LocalRunner.__new__(LP.LocalRunner)
    r.projects = Proj()
    st = LP.State(tmp_path)
    st.put("plan.json", {"style": "story", "items": [], "gap": 0.75})
    out = r.set_style("p1", {"pace": {"name": "fast"}, "music": {"policy": "none"}})
    plan = st.get("plan.json")
    assert plan["style_obj"]["pace"]["name"] == "fast" and plan["style_obj"]["music"]["policy"] == "none"
    assert plan["gap"] == PACES["fast"]["gap"] and out["describe"].splitlines()[-1] == "No music."
    got = r.style_of("p1")
    assert got["style"]["pace"]["name"] == "fast" and got["problems"] == []
    with pytest.raises(ValueError, match="between 0 and 1"):
        r.set_style("p1", {"broll": {"share": 4}})
    assert st.get("plan.json")["style_obj"]["broll"]["share"] == 0.30                       # nothing changed
    with pytest.raises(ValueError):
        LP.State(tmp_path / "empty").put("x.json", {})
        r2 = LP.LocalRunner.__new__(LP.LocalRunner)
        r2.projects = type("P", (), {"get": lambda self, pid: {"folder": str(tmp_path / "empty")}})()
        r2.set_style("p2", {"pace": {"name": "fast"}})                                         # no plan yet


def test_set_pace_returns_the_gap_of_the_pace():
    s = SPF.build("story")
    assert SPF.set_pace(s, "fast", PACES) == PACES["fast"]["gap"] and s["pace"]["name"] == "fast" and "pace.name" in s["_set"]
    assert SPF.set_pace(s, "warp", PACES) is None and s["pace"]["name"] == "fast"


def test_points_are_read_once_when_they_change_and_not_again(monkeypatch):
    import threading
    from types import SimpleNamespace

    from clipfarm.clipper import localai as LA
    from clipfarm.clipper import studio_web as SW
    ai = FakeAI(style_from_notes={"pace": "fast", "music": "none", "profile": None, "captions": None, "graphics": None, "broll_share": None,
                                  "chronology": None, "remove_interviewer": None, "voice_over": None, "target_seconds": None})
    monkeypatch.setattr(LA, "LocalAI", lambda cfg: ai)
    kept, saved = {}, []

    class Projects:
        def points_sig(self):
            return kept.get("sig", "")

        def points_style(self):
            return kept.get("settings", {})

        def set_points_style(self, settings, sig=""):
            kept.update(settings=settings, sig=sig)
    fake = SimpleNamespace(projects=Projects(), settings=SimpleNamespace(update=lambda ch: saved.append(ch)), cfg=None)

    def run(pts):
        SW.StudioAPI.sync_points_style(fake, "long", pts)
        for t in threading.enumerate():
            if t.name == "points-style":
                t.join(5)
    run(["Keep it fast-paced.", "No music."])
    assert kept["settings"] == {"pace": "fast", "music": "none"} and saved[-1] == {"style_overrides": {"long": {"pace": "fast", "music": "none"}}}
    assert ai.calls == ["style_from_notes"]
    run(["Keep it fast-paced.", "No music."])                                              # the same points: nothing is read again
    assert ai.calls == ["style_from_notes"]
    run([])                                                                                  # no points: the settings are cleared, no call
    assert kept["settings"] == {} and saved[-1] == {"style_overrides": {"long": {}}} and ai.calls == ["style_from_notes"]
