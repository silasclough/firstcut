"""Tester (2026-11-06): an ad's B-roll pool is cut at quality 6 in render() but at quality 4 in the swap strip (_shot_state), so the shot
numbers of the two disagree as soon as an ad has a shot of quality 4 or 5: the strip shows another shot's picture and a swap renders a
shot nobody chose."""
import pytest

from test_replace_shot import COLORS, _latest, _rgb_at, made  # noqa: F401 - the fixture


def test_an_ads_swap_renders_the_shot_that_was_chosen(made):  # noqa: F811
    LP, r, pid, folder = made["LP"], made["runner"], made["proj"]["id"], made["folder"]
    st = LP.State(LP.home_of(made["projects"].get(pid)))
    plan = st.get("plan.json")
    pool = plan["pool"]
    by = {next(c for c in COLORS if c in str(s.get("file"))): s for s in pool}
    pool = [{**by["green trees"], "quality": 5}, by["blue sea"], by["red wall"]]  # a usable shot (5), but under an ad's bar of 6, listed first
    st.put("plan.json", {**plan, "pool": pool, "style": "ad"})
    r.rerender(pid)
    assert made["store"].get(made["proj"]["job"])["status"] == "done"
    rc = st.get("render_cache.json")
    render_pool = [s for s in pool if s["quality"] >= 6]                          # what render() numbered the shots in
    shots = r.shots(pid)["shots"]
    assert shots, "the ad has B-roll"
    k = shots[0]["k"]
    placed = rc["final"]["cutaways"][k]
    strip_pool = r._shot_state(pid)["pool"]                                         # what the strip numbers them in
    import numpy as np
    from PIL import Image
    thumb = np.asarray(Image.open(made["store"].dir(made["proj"]["job"]) / shots[0]["thumb"]).convert("RGB"), dtype=float).mean(axis=(0, 1))
    assert thumb[0] > 150 and thumb[2] < 80, f"the red wall is on screen; the strip's picture of it is {tuple(thumb.round())}"
    # the strip must describe the same pool shot that render() placed
    assert strip_pool[placed["scene"]]["file"] == render_pool[placed["scene"]]["file"], (
        f"shot {k}: render placed {render_pool[placed['scene']]['file']}, the strip reads {strip_pool[placed['scene']]['file']}")


def test_an_ads_choices_never_offer_the_shot_already_on_screen_and_a_swap_shows_the_chosen_one(made, tmp_path):  # noqa: F811
    LP, r, pid, folder = made["LP"], made["runner"], made["proj"]["id"], made["folder"]
    st = LP.State(LP.home_of(made["projects"].get(pid)))
    plan = st.get("plan.json")
    by = {next(c for c in COLORS if c in str(s.get("file"))): s for s in plan["pool"]}
    st.put("plan.json", {**plan, "pool": [{**by["green trees"], "quality": 5}, by["blue sea"], by["red wall"]], "style": "ad"})
    r.rerender(pid)
    shot = r.shots(pid)["shots"][0]
    offered = [c["what"] for c in r.shot_choices(pid, shot["k"])["choices"]]
    assert not any("red" in w for w in offered), f"the red wall is on screen, yet offered as a replacement: {offered}"
    assert not any("green" in w for w in offered), "the green trees (quality 5) are under an ad's bar: not offered"
    pick = next(c for c in r.shot_choices(pid, shot["k"])["choices"] if "blue" in c["what"])
    r.replace_shot(pid, shot["k"], pick["choice"])
    after = r.shots(pid)["shots"][0]
    rgb = _rgb_at(_latest(folder), after["at"] + after["length"] / 2, tmp_path)
    assert rgb[2] > 100 and rgb[0] < 80, f"chose the blue sea, the video shows {rgb}"
