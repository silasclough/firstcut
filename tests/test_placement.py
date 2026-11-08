"""WP11: typed timed zones, safe boxes, slots scored per shot, caption bands, contrast, the frame check, and the placement wired into the card renderer.
Synthetic frames and zones only (no models): the face boxes are written by hand."""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from clipfarm.clipper import gfxcards as G
from clipfarm.clipper import placement as PL
from clipfarm.clipper import qa as QA


def _frame(W, H, v=60):
    return np.full((H, W, 3), v, np.uint8)


def test_safe_box_values_for_nine_sixteen_and_sixteen_nine():
    s = PL.safe_box(1080, 1920)
    assert (s["x0"], s["y0"], s["x1"], s["y1"]) == (120, 290, 888, 1248)                          # the notes' union of Meta, TikTok and Shorts
    s2 = PL.safe_box(720, 1280)
    assert abs(s2["y0"] - 193) <= 1 and abs(s2["x0"] - 80) <= 1                                    # scaled
    h = PL.safe_box(1920, 1080)
    assert (h["x0"], h["y0"], h["x1"], h["y1"]) == (96, 54, 1824, 1026)                            # SMPTE title-safe 5 %
    assert PL.safe_box(1080, 1920, "shorts")["y1"] > s["y1"]                                       # a Shorts-only export lets the bottom shrink


def test_zones_keep_their_time_and_a_walking_presenter_only_blocks_where_he_is():
    W, H = 1920, 1080
    left = PL.zones_from_face(0.2, 0.4, 0.1, W, H, 0.0, 2.0)                                       # on the left for 2 s ...
    right = PL.zones_from_face(0.8, 0.4, 0.1, W, H, 2.0, 4.0)                                      # ... then on the right
    zs = left + right
    bl = (0.0, 0.0, 0.4 * W, H)
    assert PL.covered(bl, zs, 0.0, 2.0) > 0.1 and PL.covered(bl, zs, 2.0, 4.0) == 0.0
    # a card in the left half is only blocked for the first half of the window
    whole = PL.covered(bl, zs, 0.0, 4.0)
    assert 0.05 < whole < PL.covered(bl, zs, 0.0, 2.0)
    kinds = {z["kind"] for z in left}
    assert kinds == {"face", "mouth_eyes", "body"} and all(z["t0"] == 0.0 and z["t1"] == 2.0 for z in left)


def test_a_card_goes_to_the_clear_side_and_never_over_mouth_and_eyes():
    W, H = 1920, 1080
    zs = PL.zones_from_face(0.2, 0.5, 0.12, W, H, 0.0, 6.0)                                        # the speaker is on the left
    safe = PL.safe_box(W, H)
    pick = PL.choose("lower", 600, 160, 0.0, 6.0, [(0.0, 6.0)], zs, lambda t: _frame(W, H), W, H, safe)
    assert pick is not None and pick["slot"] in ("bottom_right", "upper_right", "upper_left")      # (bottom-left is under the body zone)
    rect = (pick["x"], pick["y"], pick["w"], pick["h"])
    assert PL.hard_hit(rect, zs, 0.0, 6.0) == 0.0
    assert PL.inside(rect, safe, slack=3)
    # with the speaker on the right the lower third stays bottom-left
    zr = PL.zones_from_face(0.8, 0.5, 0.12, W, H, 0.0, 6.0)
    assert PL.choose("lower", 600, 160, 0.0, 6.0, [(0.0, 6.0)], zr, None, W, H, safe)["slot"] == "bottom_left"


def test_hysteresis_keeps_the_slot_unless_another_is_a_quarter_better_and_never_moves_inside_a_shot():
    W, H = 1920, 1080
    safe = PL.safe_box(W, H)
    # a little busy at bottom_left (a small prop), nothing elsewhere: but not a quarter better than staying put
    prop = [PL.zone("text", (110, 880, 160, 940), 0.0, 9.0, 0.35)]
    first = PL.choose("lower", 600, 160, 0.0, 4.0, [(0.0, 4.0)], [], None, W, H, safe)
    assert first["slot"] == "bottom_left"
    stay = PL.choose("lower", 600, 160, 4.0, 8.0, [(4.0, 8.0)], prop, None, W, H, safe, prev_slot="bottom_left")
    assert stay["slot"] == "bottom_left"                                                         # the prop costs a little, not a quarter of the card
    busy = [PL.zone("text", (100, 780, 800, 1020), 4.0, 8.0, 0.8)]                               # now bottom_left really is busy
    move = PL.choose("lower", 600, 160, 4.0, 8.0, [(4.0, 8.0)], busy, None, W, H, safe, prev_slot="bottom_left")
    assert move["slot"] != "bottom_left"
    # one choice per shot window: the result names one slot and one window, it does not change inside the window
    assert move["t0"] == 4.0 and move["t1"] == 8.0 and "slot" in move


def test_a_card_that_fits_nowhere_is_shortened_before_it_is_dropped():
    W, H = 1920, 1080
    safe = PL.safe_box(W, H)
    # the speaker's mouth covers the whole centre of the frame for the first 3 s of a 6 s card window: only the last 3 s are clear
    zs = [PL.zone("mouth_eyes", (0, 0, W, H), 0.0, 3.0)] + [PL.zone("face", (0, 0, W, H), 0.0, 3.0)]
    pick = PL.choose("stat", 700, 400, 0.0, 6.0, [(0.0, 3.0), (3.0, 6.0)], zs, None, W, H, safe)
    assert pick is not None and pick["shortened"] and pick["t0"] >= 3.0 - 1e-6 and pick["t1"] == 6.0
    # covered for the whole window and no clear stretch of the minimum length: dropped
    allz = [PL.zone("mouth_eyes", (0, 0, W, H), 0.0, 6.0)]
    assert PL.choose("stat", 700, 400, 0.0, 6.0, [(0.0, 6.0)], allz, None, W, H, safe) is None


def test_the_caption_band_moves_to_the_top_for_a_shot_with_a_low_face_and_stays_two_seconds():
    W, H = 1080, 1920
    low = PL.zones_from_face(0.5, 0.62, 0.3, W, H, 0.0, 4.0)                                        # a face low in the frame (chin well below 0.57 H)
    high = PL.zones_from_face(0.5, 0.25, 0.25, W, H, 4.0, 8.0)
    bands = PL.caption_band([(0.0, 4.0), (4.0, 8.0)], low + high, W, H, 0.72, 420)
    safe = PL.safe_box(W, H)
    assert bands[0][2] < H * 0.4 and bands[0][2] >= safe["y0"] - 30                                  # moved to the top of the safe area
    assert bands[1][2] == int(H * 0.72 - 420 / 2)                                                    # a high face: the usual place
    # a second shot that would move again within two seconds stays where it was
    quick = PL.caption_band([(0.0, 1.0), (1.0, 2.0)], PL.zones_from_face(0.5, 0.62, 0.3, W, H, 0.0, 1.0), W, H, 0.72, 420)
    assert len({y for *_t, y in quick}) == 1
    assert len(PL.caption_band([(0.0, 5.0)], [], W, H, 0.72, 420)) == 1


def test_a_bright_frame_gets_a_plate_until_contrast_is_four_and_a_half_to_one():
    white = _frame(400, 300, 250)
    ratio = PL.local_contrast(white, (50, 50, 200, 100), (255, 255, 255))
    assert ratio < 1.5
    plate = PL.plate_for(ratio, (255, 255, 255))
    assert plate and plate["alpha"] == 1.0                                                           # white words on a white sky: a solid plate
    mid = np.full((300, 400, 3), 170, np.uint8)
    p2 = PL.plate_for(PL.local_contrast(mid, (50, 50, 200, 100)), (255, 255, 255))
    assert p2 is not None and p2["alpha"] in (0.0, 0.35)                                             # a middling picture: a stroke or a 35 % plate
    dark = _frame(400, 300, 20)
    assert PL.plate_for(PL.local_contrast(dark, (50, 50, 200, 100)), (255, 255, 255)) is None
    # where contrast is lowest: a half-white, half-dark rectangle is judged on its worst part
    half = _frame(400, 300, 20)
    half[:, 200:] = 250
    assert PL.local_contrast(half, (0, 0, 400, 300), (255, 255, 255)) < 1.5
    assert PL.tone_for([white]) == "light" and PL.tone_for([dark]) == "dark"


def test_the_integral_image_cost_matches_a_direct_average():
    W, H = 640, 360
    zs = [PL.zone("mouth_eyes", (100, 100, 300, 250), 0.0, 5.0)]
    m, ii = PL.intrusiveness((H, W), zs, 1.0, None)
    rect = (80, 80, 160, 120)
    direct = float(m[10:25, 10:30].mean())
    assert abs(PL.rect_cost(ii, rect, (H, W)) - direct) < 0.05
    assert PL.rect_cost(ii, (400, 200, 100, 100), (H, W)) == 0.0                                     # elsewhere: nothing in the way
    sal = PL.saliency([_frame(W, H, 100), _frame(W, H, 100)])
    assert sal.shape == (H, W) and float(sal.max()) <= 1.0


def test_slots_follow_the_kind_and_the_orientation():
    safe_p, safe_l = PL.safe_box(1080, 1920), PL.safe_box(1920, 1080)
    names_p = [s["name"] for s in PL.slots("lower", 1080, 1920, safe_p)]
    assert names_p[:3] == ["bottom_left", "bottom_center", "bottom_right"] and "top_left" in names_p
    names_l = [s["name"] for s in PL.slots("lower", 1920, 1080, safe_l)]
    assert names_l == ["bottom_left", "bottom_right", "upper_left", "upper_right"]
    assert PL.slots("bug", 1080, 1920, safe_p)[0]["name"] == "top_left" and PL.slots("bug", 1920, 1080, safe_l)[0]["name"] == "top_right"
    # with captions on, the bottom row sits above them
    above = PL.slots("lower", 1920, 1080, safe_l, caption_top=700)[0]
    assert above["y"] <= 700 - 0.02 * 1080 + 1


# ------------------------------------------------------------------------------------------------------------------ the renderer
def test_the_lower_third_sits_above_the_captions_not_at_the_top():
    for W, H, cap_c in ((1920, 1080, 0.84), (1080, 1920, 0.72)):
        look = G.Look("glass", W, H)
        cap_top = cap_c * H - 66 * (H / 1080)
        look.avoid = [(0.0, cap_top - 8, float(W), cap_c * H + 70)]
        look.caption_top = cap_top
        look.safe = PL.safe_box(W, H)
        sc = G.build_scene("hook", look, {"kind": "hook", "text": "Find your fit"}, 3.0)
        x, y, w, h = sc.panel
        assert y + h <= cap_top - 0.015 * H, (W, H, sc.panel)                                         # just above the words, not up at 8 % of the height
        assert y > 0.3 * H                                                                            # and not the top of the frame
        assert x < W / 2                                                                              # bottom-LEFT, as designed
        assert y + h <= look.safe["y1"] + 2


def test_a_lower_third_without_captions_keeps_its_old_place():
    look = G.Look("glass", 1920, 1080)
    sc = G.build_scene("hook", look, {"kind": "hook", "text": "Find your fit"}, 3.0)
    assert sc.panel[1] + sc.panel[3] <= 1080 * (1 - 0.22) + 1 and sc.panel[0] < 1920 / 2


def test_a_presenter_who_walks_does_not_block_a_card_for_the_whole_window():
    W, H = 1920, 1080
    look = G.Look("glass", W, H)
    # the speaker is at the bottom-left for the first 0.8 s of a 6 s card, then gone: the union of the zones blocks bottom-left, the timed zones do not
    z = PL.zones_from_face(0.15, 0.7, 0.14, W, H, 0.0, 0.5)
    look.avoid = [(min(b[0] for b in [q["box"] for q in z if q["kind"] == "face"]), 600.0, 500.0, 1000.0)]
    look.timed, look.seconds = z, 6.0
    sc = G.build_scene("hook", look, {"kind": "hook", "text": "Find your fit"}, 6.0)
    assert sc is not None and sc.panel[0] < W / 2 and sc.panel[1] > 0.5 * H                          # still bottom-left: the walk-through was short


def test_the_style_preview_shows_the_same_spot_as_the_render(tmp_path):
    from clipfarm.clipper import studio as SD
    from clipfarm.clipper import studio_web as SW
    import inspect
    src = inspect.getsource(SW.StudioAPI.gfx_preview) if hasattr(SW, "StudioAPI") else inspect.getsource(SW)
    assert "look.caption_top" in src and "safe_box" in src                                           # the picker draws with the caption zone and the safe box


# ------------------------------------------------------------------------------------------------------------------ the bug and the frame check
def test_the_bug_sits_inside_the_safe_box_and_moves_away_from_a_lower_third(tmp_path):
    from clipfarm.clipper import foldergfx as F
    logo = tmp_path / "logo.png"
    im = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((0, 0, 399, 199), fill=(200, 30, 30, 255))
    im.save(logo)
    for W, H, side_expected in ((1080, 1920, "left"), (1920, 1080, "right")):
        out = F.bug_frame(str(logo), W, H, tmp_path / f"bug{W}.png")
        a = np.asarray(Image.open(out).convert("RGBA"))[..., 3]
        ys, xs = np.nonzero(a)
        safe = PL.safe_box(W, H)
        assert xs.min() >= safe["x0"] and xs.max() <= safe["x1"] and ys.min() >= safe["y0"] and ys.max() <= safe["y1"]
        assert (xs.mean() < W / 2) == (side_expected == "left")
        other = F.bug_frame(str(logo), W, H, tmp_path / f"bugo{W}.png", side="left" if side_expected == "right" else "right")
        ao = np.asarray(Image.open(other).convert("RGBA"))[..., 3]
        assert (np.nonzero(ao)[1].mean() < W / 2) != (side_expected == "left")
    # 9:16: 8-10 % of the width; 16:9: about 6 % of the height
    a = np.asarray(Image.open(tmp_path / "bug1080.png").convert("RGBA"))[..., 3]
    assert 0.07 * 1080 <= (np.nonzero(a)[1].max() - np.nonzero(a)[1].min()) <= 0.11 * 1080


def test_frame_check_finds_a_card_over_a_face_and_a_caption_under_a_card():
    W, H = 1920, 1080
    safe = PL.safe_box(W, H)
    zs = PL.zones_from_face(0.5, 0.5, 0.14, W, H, 0.0, 10.0)
    over = {"at": 2.0, "len": 4.0, "kind": "stat", "rect": (800, 400, 400, 250), "full": False}
    fine = {"at": 12.0, "len": 3.0, "kind": "lower", "rect": (130, 700, 500, 140), "full": False}
    caps = [{"t0": 0.0, "t1": 20.0, "rect": (0.0, 880.0, float(W), 140.0)}]
    issues = QA.frame_check([over, fine], caps, zs, safe, W, H)
    codes = [i["code"] for i in issues]
    assert "graphic_over_face" in codes
    assert not any(i["code"] == "graphic_over_face" and i["start"] == 12.0 for i in issues)
    # a card sharing the caption rows
    clash = {"at": 5.0, "len": 3.0, "kind": "stat", "rect": (130, 850, 500, 150), "full": False}
    assert "card_over_caption" in [i["code"] for i in QA.frame_check([clash], caps, [], safe, W, H)]
    assert "caption_under_card" in [i["code"] for i in QA.frame_check([clash], caps, [], safe, W, H, captions_on_top=False)]
    # a full-screen card with the captions underneath it
    full = {"at": 8.0, "len": 4.0, "kind": "fullslide", "rect": None, "full": True}
    assert "caption_under_card" in [i["code"] for i in QA.frame_check([full], caps, [], safe, W, H, captions_on_top=False)]
    assert not QA.frame_check([full], caps, [], safe, W, H, captions_on_top=True)
    # outside the safe box and at the edge
    edge = {"at": 1.0, "len": 2.0, "kind": "lower", "rect": (5, 40, 300, 100), "full": False}
    got = [i["code"] for i in QA.frame_check([edge], [], [], safe, W, H)]
    assert "text_outside_safe_box" in got and "card_touches_edge" in got
    # contrast: white words on a white picture
    bright = {"at": 1.0, "len": 2.0, "kind": "stat", "rect": (300, 300, 300, 100), "full": False, "ink": (255, 255, 255)}
    got = QA.frame_check([bright], [], [], safe, W, H, frame_fn=lambda t: _frame(W, H, 250))
    assert "low_contrast" in [i["code"] for i in got]


# ------------------------------------------------------------------------------------------------------------------ captions on top of cards
def test_captions_stay_visible_under_a_full_screen_card_and_are_hidden_under_a_statement_card():
    from clipfarm.clipper import polish
    assert {"statement", "kinetic"} <= polish.CAPTIONS_HIDDEN_UNDER and "fullslide" not in polish.CAPTIONS_HIDDEN_UNDER


# ------------------------------------------------------------------------------------------------------------------ wired into the render
def test_make_placer_ranks_the_scenes_own_anchors_by_what_is_under_them():
    W, H = 1080, 1920
    face = PL.zones_from_face(0.30, 0.72, 0.16, W, H, 0.0, 6.0)               # the speaker stands bottom-left
    placer = PL.make_placer("lower", 0.0, 4.0, [(0.0, 4.0)], face, None, W, H, PL.safe_box(W, H), caption_top=0.78 * H)
    left, right = (130.0, 1300.0), (700.0, 1300.0)
    got = placer(320.0, 120.0, [left, right])                                   # the design's first choice (left) has a face under it
    assert got is not None and got[0] > W / 2                                   # the clear side wins
    clear = PL.make_placer("lower", 0.0, 4.0, [(0.0, 4.0)], [], None, W, H, PL.safe_box(W, H))
    assert clear(320.0, 120.0, [(130.0, 1300.0), (700.0, 1300.0)]) == (130.0, 1300.0)   # nothing in the way: the design's own choice stays


def test_the_card_renderer_uses_the_placer_when_one_is_given(tmp_path):
    from clipfarm.clipper import gfxcards
    seen = {}

    def placer(w, h, prefer):
        seen["prefer"] = prefer
        return (600.0, 700.0)
    r = gfxcards.render(None, {"kind": "stat", "value": "42", "label": "answers", "len": 2.0}, 1080, 1920, tmp_path / "c.mov", fps=25.0, placer=placer)
    assert r and seen["prefer"]                                                 # asked, with the scene's own anchors
    import numpy as np
    from clipfarm.clipper import gfxcards as G
    look = G.Look(None, 1080, 1920)
    look.chooser = placer
    assert look.place(200, 100, [(10, 10), (500, 500)]) == (600, 700)


def test_the_hook_title_sits_inside_the_safe_box_on_both_shapes():
    from clipfarm.clipper.engine import Engine
    for W, H in ((1080, 1920), (1920, 1080)):
        sb = PL.safe_box(W, H)
        xexpr, y = Engine._hook_xy(W, H)
        assert sb["y0"] <= y <= 0.2 * H
        cx = int(xexpr.split("-")[0])
        assert abs(cx - (sb["x0"] + sb["x1"]) / 2) <= 1


def test_words_stay_readable_under_a_full_screen_card_on_a_real_render(tmp_path):
    """A white 540x960 picture, a light full-screen slide over it, bold captions: the caption band is darker than the page while the card is up
    (the plate), the words are still drawn, and the caption band keeps its place."""
    import subprocess
    from clipfarm.clipper import polish
    from clipfarm.clipper.engine import Engine
    from clipfarm.clipper.store import Store
    from clipfarm.core import ffmpeg
    from clipfarm.core.config import load_config
    cfg = load_config()
    cfg.root = tmp_path
    for k in ("work", "logs", "state", "queue"):
        cfg.data["paths"][k] = str(tmp_path / k)
    store = Store(tmp_path / "clipper")
    job = store.create({"filename": "x.mp4"}, {"mode": "polish", "caption_style": "bold", "music": "", "fcp": False, "pace": "natural", "graphics": "all",
                                              "gfx_cards": [{"kind": "fullslide", "title": "Three things", "bullets": ["Faith", "Hope", "Love"], "at": 3.4, "len": 3.0}]})
    src = store.dir(job["id"]) / "source.mp4"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=white:s=540x960:r=25:d=10", "-f", "lavfi", "-i", "sine=f=300:d=10",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(src)], check=True)
    words = [{"text": w, "start": 0.3 + i * 0.5, "end": 0.7 + i * 0.5} for i, w in enumerate("this is a test of the words under a full screen card and more".split())]
    clip = polish.run(Engine(cfg, store), job["id"], src, words)
    f = store.dir(job["id"]) / clip["file"]

    def frame(t):
        png = tmp_path / f"f{t}.png"
        subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", str(t), "-i", str(f), "-frames:v", "1", str(png)], check=True)
        return np.asarray(Image.open(png).convert("RGB")).astype(float)
    under, after = frame(4.6), frame(7.0)
    band = (slice(600, 780), slice(0, 540))
    assert under[band].mean() < after[band].mean() - 25                         # the plate darkened the band under the card
    assert (under[band].max(axis=2) > 240).any()                                # ... and the white words are still there
    assert under[100:300].mean() > 200                                          # the card's own page is up above the captions
