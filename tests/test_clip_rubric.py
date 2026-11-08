"""WP8: the clip rubric, length band, hook, ending and caption look of short clips."""

from __future__ import annotations

import subprocess

import pytest

from clipfarm.clipper import clipcraft as CC, editstyle, moments as M
from clipfarm.clipper.engine import Engine
from clipfarm.clipper.store import Store
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config


class _R:
    def __init__(self, data, provider="fake"):
        self.data, self.provider = data, provider


class FakeLLM:
    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def complete_json(self, task, system, prompt, schema, validate=None, **kw):
        self.calls.append((task, system, prompt))
        if validate:
            validate(self.answer)
        return _R(self.answer)


def _sents(texts, each=4.0):
    return [M.Sentence(i, i * each, i * each + each - 0.3, t) for i, t in enumerate(texts)]


TEXTS = ["He walked into the bank with a mask on.", "And that is when the alarm went off.", "Nobody inside moved for a full minute.",
         "The guard finally reached the button.", "Police were there in ninety seconds.", "They caught him two blocks away."]


def _clip(**kw):
    c = {"first": 0, "last": 3, "title": "Man robs bank, caught in ninety seconds", "hook": "a robbery with a twist", "caption": "wow", "hashtags": ["crime"],
         "hook_grade": 5, "flow": 3, "value": 4, "why_hook": "opens on the crime", "why_flow": "a little slow in the middle", "why_value": "a story with an ending",
         "score": 1, "reason": "story", "mood": "suspense"}
    c.update(kw)
    return c


# ---------------------------------------------------------------------------------------------------- the rubric
def test_the_rubric_fields_are_required_and_the_score_is_computed_by_code():
    need = M.SCHEMA["properties"]["clips"]["items"]["required"]
    assert {"hook_grade", "flow", "value", "why_hook", "why_flow", "why_value"} <= set(need) and "score" not in need
    llm = FakeLLM({"clips": [_clip()]})
    out = M.llm_moments(llm, _sents(TEXTS), 3, 8, 30)
    m = out[0]
    assert m.rubric["hook"] == 5 and m.rubric["flow"] == 3 and m.rubric["value"] == 4 and m.rubric["why_flow"].startswith("a little slow")
    assert m.score == round((5 * 0.4 + 3 * 0.3 + 4 * 0.3) * 2, 1) == 8.2                  # the model said 1; the code says 8.2
    assert 1.0 <= CC.score_of({"hook": 1, "flow": 1, "value": 1}) <= 10.0 and CC.score_of({"hook": 5, "flow": 5, "value": 5}) == 10.0


def test_a_clip_without_valid_grades_keeps_the_models_own_score():
    llm = FakeLLM({"clips": [_clip(hook_grade=9, flow=None)]})
    m = M.llm_moments(llm, _sents(TEXTS), 3, 8, 30)[0]
    assert m.rubric is None and m.score == 1.0
    assert CC.rubric_of({"hook": "x", "flow": 3, "value": 3}) is None


def test_the_prompt_carries_the_options_length_band_not_a_fixed_sweet_spot():
    llm = FakeLLM({"clips": []})
    M.llm_moments(llm, _sents(TEXTS), 3, 12, 28)
    system = llm.calls[0][1]
    assert "12-28 seconds" in system
    assert "sweet spot" not in system and "15-25" not in system and "35 seconds do clearly worse" not in system
    assert "Grade each clip 1 to 5" in system
    assert "30-50 seconds" in M.system_for(30, 50)


def test_a_new_prompt_changes_the_moments_cache_key():
    assert isinstance(M.RUBRIC_V, int) and M.RUBRIC_V >= 1


# ---------------------------------------------------------------------------------------------------- length: back to the hook
def test_a_continuation_first_sentence_grows_backwards_to_the_hook():
    s = _sents(TEXTS)
    m = M.fit_length(M.Moment(0, 0, 1, 3), s, 8, 30)                  # the AI started on "And that is when..."
    assert m.first == 0 and s[m.first].text.startswith("He walked")
    m2 = M.fit_length(M.Moment(0, 0, 2, 5), s, 8, 30)                 # a clean first sentence stays where it is
    assert m2.first == 2
    far = [M.Sentence(i, i * 40.0, i * 40.0 + 3.0, t) for i, t in enumerate(TEXTS)]
    assert M.fit_length(M.Moment(0, 0, 1, 3), far, 4, 200).first == 1   # the earlier sentence is 37 s away: not its hook


def test_the_backwards_growth_stays_inside_the_hard_cap():
    s = _sents(TEXTS, each=20.0)
    m = M.fit_length(M.Moment(0, 0, 1, 2), s, 10, 20)
    assert m.duration <= M.hard_cap(20)


# ---------------------------------------------------------------------------------------------------- re-hooks
def test_a_long_clip_gets_one_rehook_marker_from_the_list_it_was_shown():
    s = [M.Sentence(i, i * 6.0, i * 6.0 + 5.5, f"Sentence number {i} says something.") for i in range(14)]
    long_clip, short_clip = M.Moment(s[0].start, s[11].end, 0, 11), M.Moment(s[12].start, s[13].end, 12, 13)
    llm = FakeLLM({"clips": [{"clip": 0, "sentence": 6, "why": "restates the stake"}, {"clip": 1, "sentence": 13, "why": "x"}]})
    out = M.llm_rehooks(llm, [long_clip, short_clip], s)
    assert out[0].rehook_at == pytest.approx(s[6].start - s[0].start) and out[1].rehook_at is None      # the short clip was never asked about
    assert "Clip 1" not in llm.calls[0][2]
    bad = M.llm_rehooks(FakeLLM({"clips": [{"clip": 0, "sentence": 999, "why": "made up"}]}), [M.Moment(s[0].start, s[11].end, 0, 11)], s)
    assert bad[0].rehook_at is None                                                              # an id the code never listed is ignored
    boom = type("L", (), {"complete_json": lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))})()
    assert M.llm_rehooks(boom, [M.Moment(s[0].start, s[11].end, 0, 11)], s)[0].rehook_at is None


# ---------------------------------------------------------------------------------------------------- the edges
def _words(n=8, t0=0.3, step=0.4):
    return [{"text": f"w{i}", "start": t0 + i * step, "end": t0 + i * step + 0.3} for i in range(n)]


def test_the_clip_ends_a_quarter_second_after_the_last_word_and_short_clips_keep_loop_room():
    w = _words()                                                      # last word ends at 3.4
    assert CC.end_after_last_word([(0.2, 3.9)], w) == [(0.2, 3.65)]    # ran on into dead air: cut back
    assert CC.end_after_last_word([(0.2, 3.4)], w) == [(0.2, 3.65)]    # cut on the word: given its quarter second
    assert CC.end_after_last_word([(0.2, 3.5)], w, loop=True) == [(0.2, 3.8)]            # a clip that loops keeps 0.15 s of room tone more
    nxt = w + [{"text": "next", "start": 3.5, "end": 3.8}]             # the next sentence starts right after: never into it
    assert CC.end_after_last_word([(0.2, 3.45)], nxt)[0][1] == pytest.approx(3.47)
    assert CC.end_after_last_word([], w) == []
    assert CC.could_loop("Nobody expected the robbery", "Nobody saw the robbery coming") and not CC.could_loop("Hello there friend", "Goodbye for now")


def test_the_lead_is_trimmed_to_a_fraction_of_a_second():
    w = _words(t0=1.6)
    assert CC.trim_lead([(0.0, 4.0)], w, 0.15)[0][0] == pytest.approx(1.45)
    assert CC.trim_lead([(1.5, 4.0)], w, 0.15)[0][0] == pytest.approx(1.5)      # never earlier than it started
    assert CC.trim_lead([(0.0, 0.4)], w, 0.15) == [(0.0, 0.4)]


# ---------------------------------------------------------------------------------------------------- the whole render
def _engine(tmp_path):
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {})
    return Engine(cfg, store), store, job


def _source(store, job, secs=5):
    src = store.dir(job["id"]) / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=s=1280x720:r=25:d={secs}", "-f", "lavfi", "-i", f"sine=f=300:d={secs}",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    return src


def test_a_rendered_clip_ends_just_after_its_last_word_and_carries_its_rubric(tmp_path, monkeypatch):
    from clipfarm.clipper import reframe as R
    eng, store, job = _engine(tmp_path)
    src = _source(store, job)
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([[]] * 8, 8))                 # no face anywhere
    w = _words(t0=1.0)                                                # speech from 1.0 s; the last word ends at 4.1
    m = M.Moment(start=0.2, end=4.4, first=0, last=7, title="T", caption="c", score=8, rubric={"hook": 5, "flow": 4, "value": 4, "why_hook": "a claim", "why_flow": "ok", "why_value": "ok"},
                 rehook_at=2.0)
    clip = eng.render_clip(job["id"], 1, m, w, src, layout="auto", style="none", pace="natural", hook_title=False)
    assert clip["rubric"]["hook"] == 5 and clip["rehook_at"] == 2.0
    start = 1.0 - 0.12                                                # the clip starts at its first word (less the pause pad), not at the 0.2 s the moment began
    assert clip["duration"] - (4.1 - start) == pytest.approx(0.25, abs=0.06)        # the last word ends at 4.1: the last frame is a quarter second later
    assert clip["duration"] < 4.1 - 0.2                                               # and the 0.8 s of dead air before the hook is gone
    codes = {i.get("code") for i in (clip["qa"] or {}).get("issues", [])}
    assert "no_face_early" in codes and all(i["level"] == "warn" for i in clip["qa"]["issues"] if i.get("code") in ("hook_late", "no_face_early"))


def test_the_edges_can_be_switched_off(tmp_path, monkeypatch):
    from clipfarm.clipper import reframe as R
    eng, store, job = _engine(tmp_path)
    eng.cfg.data.setdefault("clipper", {})["clip_edges"] = False
    src = _source(store, job)
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([[]] * 8, 8))
    m = M.Moment(start=0.2, end=4.4, first=0, last=7, title="T", caption="c", score=8)
    clip = eng.render_clip(job["id"], 1, m, _words(t0=1.0), src, layout="fit", style="none", pace="natural", hook_title=False)
    assert clip["duration"] > 4.1 - 0.88 + 0.3 and not any(i.get("code") for i in (clip["qa"] or {}).get("issues", []))


# ---------------------------------------------------------------------------------------------------- the hook card has one home
def test_the_hook_card_is_made_by_one_replaceable_method_and_the_card_graphics_wait_for_it(tmp_path, monkeypatch):
    from clipfarm.clipper import reframe as R
    eng, store, job = _engine(tmp_path)
    seen = {}

    def card(text, png, W, H):
        seen["call"] = (text, W, H)
        from PIL import Image
        Image.new("RGBA", (400, 120), (255, 255, 255, 255)).save(png)
    monkeypatch.setattr(Engine, "_hook_card", lambda self, text, png, W, H: card(text, png, W, H))
    monkeypatch.setattr(R, "detect_faces", lambda *a, **k: ([[]] * 8, 8))
    src = _source(store, job)
    m = M.Moment(start=0.2, end=4.4, first=0, last=7, title="The title", caption="c", score=8)
    eng.render_clip(job["id"], 1, m, _words(t0=1.0), src, layout="fit", style="none", pace="natural", hook_title=True)
    assert seen["call"] == ("The title", 1080, 1920)


def test_the_title_seconds_are_the_graphics_hook_end(monkeypatch):
    from clipfarm.clipper import gfxplan
    seen = {}
    monkeypatch.setattr(gfxplan, "plan", lambda *a, **k: seen.setdefault("hook_end", k.get("hook_end")) and [])

    class E:
        cfg = load_config()

        def router_for_now(self):
            return None, "x"
    words = [{"text": "a", "start": 0.0, "end": 0.4}]
    editstyle.apply(E(), "j", __import__("pathlib").Path("/nonexistent.mp4"), words, 20.0, "youtube", 1080, 1920, 30, True, hook_end=3.2)
    assert seen["hook_end"] == 3.2


# ---------------------------------------------------------------------------------------------------- caption look by mood
def test_caption_kind_rules():
    assert CC.caption_kind({"kind": "twitch_clip"}, "step one do this") == "phrase"
    assert CC.caption_kind({"title": "The Blocks Podcast"}, "here is how to bake bread in three steps") == "phrase"
    assert CC.caption_kind({}, "here is how to cook rice: step one", None) == "karaoke"
    assert CC.caption_kind({}, "I cried when he told me", None) == "phrase"
    assert CC.caption_kind({}, "I cried when he told me", "karaoke") == "karaoke"                # the AI's answer decides when there is one
    assert CC.caption_style_for("karaoke", 0) in CC.KARAOKE and CC.caption_style_for("phrase", 3) in CC.PHRASE


def test_the_engine_picks_the_caption_look_by_mood_and_podcasts_are_phrase(tmp_path):
    eng, _store, _job = _engine(tmp_path)
    words = [{"text": t, "start": i * 0.4, "end": i * 0.4 + 0.3} for i, t in enumerate("here is how to cook rice step one".split())]
    m = M.Moment(0.0, 3.0, 0, 0)
    eng.ccfg = {**eng.ccfg, "caption_mood_ai": False}
    assert eng._caption_look({"source": {"kind": "twitch_clip"}}, m, words, 0) in CC.PHRASE
    assert eng._caption_look({"source": {"title": "Cooking tips"}}, m, words, 0) in CC.KARAOKE
    eng.ccfg = {**eng.ccfg, "caption_by_mood": False}
    assert eng._caption_look({"source": {"title": "Cooking tips"}}, m, words, 1) == "pop"            # the old rotation when the setting is off
