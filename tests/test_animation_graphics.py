"""Custom animated diagrams in talking-head videos: the planner proposes an 'animation' where the speaker explains a mechanism (checked against what is said),
and the editor lays the rendered diagram over the picture."""
import subprocess
from pathlib import Path
from types import SimpleNamespace

from clipfarm.clipper import gfxplan as GP
from clipfarm.clipper import localproj as LP
from clipfarm.clipper import moments as M
from clipfarm.clipper import polish
from clipfarm.core import ffmpeg

TALK = [
    "Hi everyone, today I want to show you something about gearboxes.",
    "A dog clutch works like this: a selector fork slides a sleeve sideways along the shaft.",
    "The teeth on the side of the sleeve then lock into matching teeth on the side of the gear.",
    "That is why racing cars can change gear so fast without any synchronising parts.",
    "Anyway, that is all for today, thanks for watching and see you next time.",
]


def sents():
    t, out = 0.0, []
    for i, x in enumerate(TALK):
        out.append(M.Sentence(i, t, t + 5.5, x))
        t += 6.0
    return out


def words_of(ss):
    ws = []
    for s in ss:
        toks = s.text.split()
        for k, w in enumerate(toks):
            ws.append({"text": w, "start": s.start + k * (5.0 / len(toks)), "end": s.start + k * (5.0 / len(toks)) + 0.25})
    return ws


def item(**kw):
    return {"kind": "animation", "at": 6.2, "concept": "A selector fork slides a sleeve sideways until the teeth on its side lock into the teeth of a gear", **kw}


def test_a_grounded_animation_is_accepted_with_its_narration_and_timing():
    ss = sents()
    v = GP.validate(item(), " ".join(s.text for s in ss), ss, 30.0, 3.2, words_of(ss))
    assert v and v["kind"] == "animation" and 7.0 <= v["len"] <= 12.0
    assert "selector fork" in v["narration"] and v["timed"] and v["timed"][0]["start"] >= -0.05
    assert "dog clutch" in v["context"] and v["concept"].startswith("A selector fork")


def test_an_animation_about_something_not_said_is_dropped():
    ss = sents()
    bad = item(concept="A nuclear reactor splits uranium atoms to heat water into steam for a turbine")
    assert GP.validate(bad, " ".join(s.text for s in ss), ss, 30.0, 3.2, words_of(ss)) is None
    assert GP.validate(item(concept="gears"), " ".join(s.text for s in ss), ss, 30.0, 3.2, words_of(ss)) is None            # too short to direct


def test_an_animation_that_would_run_past_the_end_is_dropped_or_cut_short():
    ss = sents()
    spoken = " ".join(s.text for s in ss)
    assert GP.validate(item(at=26.0), spoken, ss, 28.0, 3.2, words_of(ss)) is None


class FakeAI:
    def complete_json(self, task, system, prompt, schema, **kw):
        assert "animation" in system and "CUSTOM ANIMATED DIAGRAM" in system
        return SimpleNamespace(data={"graphics": [
            {"kind": "animation", "at": 6.2, "concept": "A selector fork slides a sleeve sideways until the teeth on its side lock into the teeth of a gear"},
            {"kind": "animation", "at": 14.0, "concept": "The matching teeth on the side of the sleeve lock into the teeth on the side of the gear"},
            {"kind": "animation", "at": 20.0, "concept": "Racing cars change gear fast because the sleeve slides onto the gear without any synchronising parts"},
            {"kind": "animation", "at": 22.0, "concept": "A selector fork slides a sleeve along the shaft so the teeth lock into the gear on the side"}]})


def test_the_plan_keeps_animations_within_their_own_small_budget():
    ss = sents()
    cards = GP.plan(FakeAI(), ss, "", LP.PROFILES["youtube"], 150.0, words=words_of(ss))
    anims = [c for c in cards if c["kind"] == "animation"]
    assert 1 <= len(anims) <= 3
    assert all(a["at"] >= 3.4 for a in anims)
    notes = GP.check_notes(cards, ss)
    assert "selector fork" in notes                                                         # the check list for a human shows what each diagram draws


def test_story_business_and_youtube_profiles_allow_animations_but_ads_do_not():
    assert all("animation" in LP.PROFILES[k]["kinds"] for k in ("story", "business", "youtube"))
    assert "animation" not in LP.PROFILES["ad"]["kinds"] and "animation" not in LP.PROFILES["promo"]["kinds"]


def test_render_animation_lays_the_clip_out_as_a_full_frame_overlay(tmp_path, monkeypatch):
    src = tmp_path / "anim.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=0x0e203a:s=640x360:r=30:d=4", "-pix_fmt", "yuv420p", str(src)], check=True)
    seen = {}

    def fake_animate(cfg, ai, tier, concept, narration, words, dur, look, work, source_text, say=print):
        seen.update({"tier": tier, "concept": concept, "dur": dur, "look": look, "source": source_text})
        return {"file": src, "tier": tier}
    from clipfarm.clipper import explainer as EX
    monkeypatch.setattr(EX, "animate", fake_animate)
    monkeypatch.setattr("clipfarm.clipper.localai.LocalAI", lambda cfg: object())
    cfg = SimpleNamespace(get=lambda k, d=None: {"explainer.animation_tier": "svg"}.get(k, d))
    out = tmp_path / "card0.mov"
    spec = {"kind": "animation", "concept": "A selector fork slides a sleeve", "narration": "x", "timed": [], "context": "the speaker's words"}
    got = polish.render_animation(cfg, spec, 3.0, out, 1280, 720, "blueprint")
    assert got == out and out.exists() and abs(ffmpeg.duration(out) - 3.0) < 0.2
    assert seen["tier"] == "svg" and seen["look"] == ("blueprint", "grid") and seen["source"] == "the speaker's words"


def test_render_animation_returns_none_when_the_diagram_cannot_be_made(tmp_path, monkeypatch):
    from clipfarm.clipper import explainer as EX
    monkeypatch.setattr(EX, "animate", lambda *a, **k: None)
    monkeypatch.setattr("clipfarm.clipper.localai.LocalAI", lambda cfg: object())
    cfg = SimpleNamespace(get=lambda k, d=None: d)
    assert polish.render_animation(cfg, {"kind": "animation", "concept": "x y z a b c"}, 3.0, tmp_path / "c.mov", 1280, 720, None) is None
    monkeypatch.setattr(EX, "animate", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no browser")))
    assert polish.render_animation(cfg, {"kind": "animation", "concept": "x y z a b c"}, 3.0, tmp_path / "c.mov", 1280, 720, None) is None
