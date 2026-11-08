"""Graphics upgrades (2026-11-06, from the research on Hyperframes-like tools): motion blur on moving graphics and photo moves, animated
full-screen kinds drawn with Skia (drawchart, count, kinetic; Pillow cousins without skia), and small Lottie accents rendered locally."""
import json
import subprocess

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import gfxcards as G
from clipfarm.clipper import gfxplan as GP
from clipfarm.clipper import lottiefx as LF
from clipfarm.clipper import skiagfx as SK
from clipfarm.core import ffmpeg


# ------------------------------------------------------------------------------------------------ motion blur
def _lower(W=1920, H=1080):
    look = G.Look("broadcast", W, H)
    return G.build_scene("lower", look, {"kind": "lower", "name": "Dr. Ruth Jones", "role": "Professor of Biblical Studies"}, 4.0)


def test_blur_only_while_something_moves():
    sc = _lower()
    assert G.moving(sc, 0.15) and not G.moving(sc, 2.0) and G.moving(sc, 3.9)
    still = G.frame_blurred(sc, 2.0)
    assert still.tobytes() == G.frame_at(sc, 2.0).tobytes()                       # a still frame is exactly the plain frame (free)


def test_a_moving_frame_gets_a_soft_trail():
    sc = _lower()
    plain, blurred = np.asarray(G.frame_at(sc, 0.15))[..., 3], np.asarray(G.frame_blurred(sc, 0.15))[..., 3]   # (still sliding in, nearly opaque: the slide-in is 0.3 s now)
    soft = lambda a: int(((a > 8) & (a < 247)).sum())
    assert soft(blurred) > soft(plain) * 1.2                                        # more half-transparent edge pixels: the trail
    assert abs(int((blurred > 8).sum()) - int((plain > 8).sum())) < 0.4 * int((plain > 8).sum())   # same thing, not a smear


def test_the_render_can_switch_blur_off(tmp_path):
    from types import SimpleNamespace
    cfg = SimpleNamespace(get=lambda k, d=None: False if k == "clipper.ai_videos.motion_blur" else d)
    r = G.render(cfg, {"kind": "lower", "name": "A B", "role": "C", "len": 1.0}, 640, 360, tmp_path / "x.mov")
    assert r and (tmp_path / "x.mov").stat().st_size > 0


def test_photo_moves_keep_their_length_with_blur(tmp_path):
    from clipfarm.clipper import photos as PH
    pic = tmp_path / "p.jpg"
    Image.fromarray((np.random.default_rng(1).random((720, 1280, 3)) * 255).astype(np.uint8)).save(pic)
    counts = []
    for mb in (False, True):
        out = PH.bake(str(pic), "pan_right", 1.0, 640, 360, 30.0, tmp_path / f"o{mb}.mp4", motion_blur=mb)
        r = subprocess.run([ffmpeg.bin_path("ffprobe"), "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames",
                            "-of", "csv=p=0", str(out)], capture_output=True, text=True)
        counts.append(int(r.stdout.strip()))
    assert counts[0] == counts[1] and counts[0] >= 28


# ------------------------------------------------------------------------------------------------ skia kinds
skia_only = pytest.mark.skipif(not SK.available(), reason="skia-python not installed")


def _ink_box(kind, spec, W, H, t):
    """The bounding box of what the card draws (its page left out), at time t."""
    caught = {}
    real = G._fullscreen
    try:
        G._fullscreen = lambda look, sp, secs, draw: caught.setdefault("draw", draw) and None or real(look, sp, secs, draw)
        look = G.Look("broadcast", W, H)
        assert SK.SCENES[kind](look, spec, 6.0) is not None, kind
    finally:
        G._fullscreen = real
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    from PIL import ImageDraw
    caught["draw"](ImageDraw.Draw(lay), lay, t, (255, 255, 255), (180, 190, 210), (255, 210, 63))
    return lay.getchannel("A").getbbox()


@skia_only
@pytest.mark.parametrize("W,H", [(1920, 1080), (1080, 1920)])
def test_skia_kinds_stay_inside_the_frame(W, H):
    specs = {"drawchart": {"chart": {"type": "line", "title": "Students enrolled", "labels": ["2019", "2020", "2021", "2022", "2023", "2024"],
                                     "values": ["210", "245", "238", "301", "355", "412"]}},
             "count": {"value": "92%", "label": "of graduates are working in ministry within a year of finishing their degree here"},
             "kinetic": {"text": "Faith isn't just part of campus here. It's the very foundation.", "emphasis": "the very foundation"}}
    for kind, sp in specs.items():
        box = _ink_box(kind, sp, W, H, 5.5)
        assert box, kind
        m = int(min(W, H) * 0.02)
        assert box[0] >= m and box[1] >= m and box[2] <= W - m and box[3] <= H - m, (kind, box)


@skia_only
def test_the_count_lands_on_the_exact_number_and_a_long_line_shrinks():
    look = G.Look("broadcast", 1920, 1080)
    assert SK.scene_count(look, {"value": "nothing"}, 4.0) is None
    assert SK.scene_kinetic(look, {"text": "one"}, 4.0) is None                    # (one word is not a line)
    long = SK.scene_kinetic(look, {"text": "This is a much longer line that has to shrink to fit"}, 6.0)
    assert long is not None


def test_without_skia_every_kind_falls_back_to_its_pillow_cousin(monkeypatch):
    monkeypatch.setattr(SK, "available", lambda: False)
    look = G.Look("broadcast", 1920, 1080)
    assert G.SCENES["kinetic"](look, {"text": "Faith is the foundation of everything here", "emphasis": "the foundation", "times": [0.3 * i for i in range(7)]}, 5.0) is not None
    assert G.SCENES["count"](look, {"value": "92%", "label": "of graduates"}, 5.0) is not None
    assert G.SCENES["drawchart"](look, {"chart": {"type": "bar", "title": "T", "labels": ["A", "B"], "values": ["1", "2"], "unit": ""}}, 7.0) is not None


# ------------------------------------------------------------------------------------------------ lottie
@pytest.mark.parametrize("name", LF.ACCENTS)
def test_accents_have_alpha_and_the_right_length(name):
    fr = LF.frames(name, 160, 90, (255, 210, 63), 24)
    assert len(fr) == 24 and all(f.mode == "RGBA" and f.size == (160, 90) for f in fr)
    assert fr[-1].getchannel("A").getextrema()[1] == 255                          # drawn by the end
    if name != "ping":
        assert fr[0].getchannel("A").getextrema()[1] == 0                         # ...and draws itself on (empty at the start)


def test_without_rlottie_the_accents_are_drawn_plainly(monkeypatch):
    monkeypatch.setattr(LF, "available", lambda: False)
    LF._CACHE.clear()
    fr = LF.frames("check", 120, 120, (255, 255, 255), 12)
    assert len(fr) == 12 and fr[-1].getchannel("A").getextrema()[1] > 0
    LF._CACHE.clear()


@pytest.mark.skipif(not LF.available(), reason="rlottie not installed")
def test_a_lottie_file_renders_at_the_videos_rate(tmp_path):
    p = tmp_path / "a.json"
    p.write_text(json.dumps(LF.doc("arrow", 200, 100, (255, 0, 0), 30)))           # 30 frames at 30 fps = 1 s
    fr = LF.render_file(p, 200, 100, fps=25.0)
    assert len(fr) == 25 and fr[-1].getchannel("A").getextrema()[1] == 255


# ------------------------------------------------------------------------------------------------ the planner
def test_the_planner_offers_the_animated_kinds_where_their_cousins_are_allowed():
    import re
    assert all(k in GP.SCHEMA["properties"]["graphics"]["items"]["properties"]["kind"]["enum"] for k in ("drawchart", "count", "kinetic"))
    assert GP.FULL_OF["kinetic"] == "statement" and "drawchart" in GP.FULL
    assert re.search(r"'drawchart'.*'count'.*'kinetic'", GP.SYSTEM, re.S)
