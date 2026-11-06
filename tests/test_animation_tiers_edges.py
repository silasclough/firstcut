"""Animation tiers (tester): deterministic parts only, no Hyperframes / Chrome / AI. animate_lib geometry checks and drawing, look tokens and page CSS for every
theme x page, and explainer.animate's cache key and svg -> library fallback."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import animate_hf as AH
from clipfarm.clipper import animate_lib as AL
from clipfarm.clipper import explainer as EX
from clipfarm.clipper.graphics import THEMES

PAGES = ("dots", "grid", "rule", "stripe", "blocks", "frame", None, "nonsense")


def gear(i, x, y, r, teeth=20, **kw):
    return {"type": "gear", "id": i, "x": x, "y": y, "r": r, "teeth": teeth, **kw}


def good_spec():
    return {"elements": [gear("g1", 700, 450, 200, 24, spin_dps=60), gear("g2", 700 + 200 + 120 - 12, 450, 120, 14, mesh="g1"),
                         {"type": "label", "text": "Driver", "x": 700, "y": 780, "size": 50}]}


def test_good_description_has_no_problems_and_wrong_counts_are_reported():
    assert AL.problems(good_spec()) == []
    assert any("between 3 and" in p for p in AL.problems({"elements": []}))
    assert any("between 3 and" in p for p in AL.problems({}))
    assert any("between 3 and" in p for p in AL.problems({"elements": [good_spec()["elements"][0]] * 30}))


def _limit_message_matches_the_check():
    msg = AL.problems({"elements": []})[0]
    n = int(msg.split("and ")[1].split(" ")[0])
    spec = {"elements": [{"type": "label", "text": "a", "x": 500, "y": 500, "size": 50}] * (n + 1)}
    assert any("between 3 and" in p for p in AL.problems(spec)), "the AI is told the limit is %d but %d elements are accepted" % (n, n + 1)


@pytest.mark.xfail(strict=True, reason="BUG (minor): problems() tells the AI 'between 3 and 22 elements' but accepts up to 24")
def test_limit_message_is_honest():
    _limit_message_matches_the_check()


def test_geometry_problems_are_named():
    s = good_spec()
    s["elements"][1]["x"] = 1300                                                              # too far from the driver
    assert any("should touch" in p for p in AL.problems(s))
    s = good_spec()
    s["elements"][1]["mesh"] = "ghost"
    assert any("does not exist" in p for p in AL.problems(s))
    s = good_spec()
    s["elements"][0]["x"] = 50
    assert any("safe area" in p for p in AL.problems(s))
    s = {"elements": [{"type": "gear", "id": "a"}, {"type": "belt", "a": "x", "b": "y"}, {"type": "label", "text": "one two three four five", "x": 50, "y": 5, "size": 10}]}
    t = " | ".join(AL.problems(s))
    assert "needs x, y, r and teeth" in t and "two pulleys" in t and "1-3 words" in t and "smaller than" in t and "outside the safe area" in t


@pytest.mark.xfail(strict=True, reason="BUG: problems() raises (AttributeError/TypeError) on an element that is not an object, a gear whose x is a string, or a null element, instead of reporting it; animate_lib.make() then lets the exception out of the whole tier")
@pytest.mark.parametrize("els", [["gear", "x", "y"], [None, None, None], [gear("a", "100", 400, 100), gear("b", 500, 400, 100), gear("c", 900, 400, 100)]])
def test_problems_does_not_crash_on_malformed_elements(els):
    assert isinstance(AL.problems({"elements": els}), list)


def tokens(look=("clean", "dots")):
    return AH.look_tokens(*look)


def all_elements():
    return {"elements": [
        gear("g1", 500, 450, 200, 24, spin_dps=60, at=0, fill=True), gear("g2", 500 + 200 + 120 - 12, 450, 120, 14, mesh="g1", at=0.5),
        {"type": "pulley", "id": "p1", "x": 1200, "y": 300, "r": 100, "spin_dps": -90, "r_to": 160, "r_start": 1, "r_dur": 2},
        {"type": "pulley", "id": "p2", "x": 1600, "y": 500, "r": 100},
        {"type": "belt", "a": "p1", "b": "p2", "at": 0.2}, {"type": "belt", "a": "p1", "b": "missing"},
        {"type": "slider", "x": 300, "y": 600, "w": 300, "h": 100, "text": "Part", "move_dx": 400, "move_dy": 50, "move_start": 1, "move_dur": 2, "fill": True},
        {"type": "box", "x": 900, "y": 700, "w": 200, "h": 100, "text": "Box"},
        {"type": "arrow", "x": 300, "y": 800, "x2": 900, "y2": 800, "at": 0.5, "dur": 1}, {"type": "line", "x": 300, "y": 850, "x2": 300, "y2": 850},
        {"type": "label", "text": "Label", "x": 960, "y": 600, "size": 50, "color": "accent"}, {"type": "counter", "x": 960, "y": 200, "from_n": 0, "to_n": 1234567, "suffix": " x", "color": "dim"}]}


@pytest.mark.parametrize("look", [("clean", "dots"), ("blueprint", "grid"), ("slate", "stripe"), ("mint", "rule")])
def test_scene_frame_draws_every_element_type_for_the_whole_duration(look):
    sc = AL.Scene(all_elements(), tokens(look), look)
    for t in (0.0, 0.01, 0.5, 1.0, 2.5, 4.0, 10.0, -1.0):
        im = sc.frame(t)
        assert im.size == (1920, 1080) and im.mode == "RGB"
    # something is drawn by t=3 that is not at t=0 (a still frame would be a failed animation)
    assert sc.frame(0.0).tobytes() != sc.frame(3.0).tobytes()


@pytest.mark.parametrize("ra,rb", [(100, 100), (240, 60), (60, 240), (1, 400), (400, 1), (0.001, 0.001)])
def test_belt_geometry_with_equal_and_very_different_radii_and_overlapping_pulleys(ra, rb):
    for dist in (500, 5, 0.5, 0):
        spec = {"elements": [{"type": "pulley", "id": "a", "x": 600, "y": 400, "r": ra}, {"type": "pulley", "id": "b", "x": 600 + dist, "y": 400, "r": rb}, {"type": "belt", "a": "a", "b": "b"}]}
        sc = AL.Scene(spec, tokens(), ("clean", "dots"))
        assert sc.frame(1.0).size == (1920, 1080)


def test_belt_follows_the_growing_pulley_radius():
    spec = {"elements": [{"type": "pulley", "id": "a", "x": 600, "y": 450, "r": 80, "r_to": 240, "r_start": 0, "r_dur": 2}, {"type": "pulley", "id": "b", "x": 1300, "y": 450, "r": 80}, {"type": "belt", "a": "a", "b": "b"}]}
    sc = AL.Scene(spec, tokens(), ("clean", "dots"))
    assert sc.radius(spec["elements"][0], 0.0) == pytest.approx(80) and sc.radius(spec["elements"][0], 5.0) == pytest.approx(240)
    assert sc.frame(0.1).tobytes() != sc.frame(2.5).tobytes()


def test_meshed_gears_turn_opposite_ways_at_the_tooth_ratio_and_a_mesh_loop_ends():
    spec = good_spec()
    sc = AL.Scene(spec, tokens(), ("clean", "dots"))
    g1, g2 = spec["elements"][:2]
    a1, a2 = sc.angle(g1, 1.0), sc.angle(g2, 1.0)
    assert a1 == pytest.approx(60.0)
    assert a2 == pytest.approx(-60.0 * 24 / 14 + 180.0 / 14)
    loop = {"elements": [gear("a", 500, 450, 100, 12, mesh="b"), gear("b", 700, 450, 100, 12, mesh="a"), gear("c", 900, 450, 100, 12)]}
    assert math.isfinite(AL.Scene(loop, tokens(), ("clean", "dots")).angle(loop["elements"][0], 2.0))


@pytest.mark.parametrize("theme", sorted(THEMES) + ["nonexistent"])
@pytest.mark.parametrize("page", PAGES)
def test_look_tokens_and_page_css_for_every_theme_and_page(theme, page):
    t = AH.look_tokens(theme, page)
    for k in ("page", "ink", "dim", "dot", "accent"):
        assert len(t[k]) == 7 and t[k].startswith("#") and int(t[k][1:], 16) >= 0
    css = AH.page_css(t)
    assert css.startswith("background-color:#") and "None" not in css and "nan" not in css
    assert page not in ("grid", "rule", "stripe", "dots") or "background-image" in css


def test_look_tokens_ink_is_readable_on_the_page():
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    for theme in THEMES:
        t = AH.look_tokens(theme, "dots")
        a, b = sorted((lum(t["ink"]), lum(t["page"])))
        assert (b + 0.05) / (a + 0.05) >= 3.0, (theme, t)                                   # WCAG large-text contrast


@pytest.mark.xfail(strict=True, reason="BUG (minor): words are embedded with json.dumps inside an inline <script> without escaping '</': a spoken/typed word '</script>' ends the script and the page renders blank")
def test_assemble_survives_a_script_or_word_containing_script_end_tag(tmp_path):
    proj = AH.assemble(tmp_path / "p", "tl.to(HF.svg, {duration: 1});", "", tokens(), 3.0, [{"text": "</script><b>", "start": 0.5}, {"text": "no-start"}])
    html = (proj / "index.html").read_text()
    assert html.count("</script>") == 3                                                      # gsap, hf-lib and the one inline script: the word did not close it
    assert (proj / "gsap.min.js").exists()


# ------------------------------------------------------------------ explainer.animate
class Tier:
    def __init__(self, ok, log, name):
        self.ok, self.log, self.name = ok, log, name

    def make(self, cfg, ai, concept, narration, words, dur, look, d, source_text, say=print):
        self.log.append(self.name)
        if not self.ok:
            return None
        d.mkdir(parents=True, exist_ok=True)
        (d / "animation.mp4").write_bytes(b"x")
        return {"file": d / "animation.mp4"}


def patch_tiers(monkeypatch, svg_ok, lib_ok):
    log = []
    monkeypatch.setattr(AH, "make", Tier(svg_ok, log, "svg").make)
    monkeypatch.setattr(AL, "make", Tier(lib_ok, log, "library").make)
    return log


def run(tmp_path, tier, **kw):
    a = dict(concept="c", narration="n", dur=8.0, look=("clean", "dots"))
    a.update(kw)
    return EX.animate(SimpleNamespace(), object(), tier, a["concept"], a["narration"], [], a["dur"], a["look"], tmp_path, "src", say=lambda *_: None)


def test_auto_falls_back_from_svg_to_the_library_and_stops_at_the_first_success(tmp_path, monkeypatch):
    log = patch_tiers(monkeypatch, False, True)
    r = run(tmp_path, "auto")
    assert r["tier"] == "library" and log == ["svg", "library"]
    log.clear()
    patch = patch_tiers(monkeypatch, True, True)
    assert run(tmp_path / "x", "auto")["tier"] == "svg" and patch == ["svg"]
    patch_tiers(monkeypatch, False, False)
    assert run(tmp_path / "y", "auto") is None


def test_a_fixed_tier_never_falls_back(tmp_path, monkeypatch):
    log = patch_tiers(monkeypatch, False, True)
    assert run(tmp_path, "svg") is None and log == ["svg"]
    assert run(tmp_path, "library")["tier"] == "library"


def test_cached_clip_is_reused_only_for_the_same_concept_length_and_look(tmp_path, monkeypatch):
    log = patch_tiers(monkeypatch, True, True)
    assert "cached" not in run(tmp_path, "svg")
    assert run(tmp_path, "svg")["cached"] is True and log == ["svg"]
    run(tmp_path, "svg", dur=8.01)
    run(tmp_path, "svg", concept="other")
    run(tmp_path, "svg", look=("sky", "blocks"))
    run(tmp_path, "svg", narration="changed words")
    assert log == ["svg"] * 5


def test_unknown_tier_does_not_crash_the_build_it_is_treated_as_library(tmp_path, monkeypatch):
    log = patch_tiers(monkeypatch, True, True)
    r = run(tmp_path, "bogus")
    assert r is None or r["tier"] in ("bogus", "library")


@pytest.mark.xfail(strict=True, reason="BUG (waste): with tier 'auto' and a failing svg tier, the library clip is cached but the next call tries svg FIRST again (wiping its folder and calling the AI + Hyperframes) before it reaches the cached library clip")
def test_auto_reuses_a_cached_library_clip_without_retrying_svg(tmp_path, monkeypatch):
    log = patch_tiers(monkeypatch, False, True)
    run(tmp_path, "auto")
    log.clear()
    r = run(tmp_path, "auto")
    assert r["cached"] and log == []
