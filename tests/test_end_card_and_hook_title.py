"""Two graphics bugs from a real long form (a keep-cut "Add B-roll" project):

A. A logo card ("Northfield University" on a shield) landed over students' faces and its name was cut off ("nmaus Universi").
   Causes: the face check saw only the nearest person (it looked at 480 px and ignored faces under 3 % of the width, and in a wide shot every
   student is about 2 %), and the name under a logo / flag / photo was drawn on one line however long it was.
B. In Final Cut the hook title was a basic dark-on-white title off to the side. It is now placed on the panel the app drew, wrapped like the app's
   (a Basic Title does not wrap by itself), white text, and the exact look of the app goes in as a clip with the editable title switched off.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from clipfarm.clipper import facezones, gfxcards
from clipfarm.clipper import fcpxml as FX
from clipfarm.clipper import polish

FRAMES = {"16:9": (1920, 1080), "9:16": (1080, 1920), "1:1": (1080, 1080)}


def _logo(tmp_path) -> str:
    img = Image.new("RGBA", (400, 400), (0, 0, 0, 0))
    ImageDraw.Draw(img).polygon([(40, 40), (360, 40), (360, 220), (200, 370), (40, 220)], fill=(0, 70, 150, 255))
    p = tmp_path / "logo.png"
    img.save(p)
    return str(p)


def _ink_box(img: Image.Image):
    return img.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()


# ---------------------------------------------------------------------------- A: the name under a logo is never cut off
@pytest.mark.parametrize("name", ["Northfield University", "The Northfield University of Northern Iowa", "Supercalifragilisticexpialidocious Foundation"])
@pytest.mark.parametrize("frame", list(FRAMES))
def test_the_name_under_a_logo_fits_inside_its_card(tmp_path, frame, name):
    W, H = FRAMES[frame]
    look = gfxcards.Look(None, W, H)
    tile = gfxcards._tile(look, {"type": "logo", "file": _logo(tmp_path), "name": name}, look.px(300), look.px(22))
    assert tile is not None
    layer, card_w, card_h, m = tile
    lab_top = m + look.px(22) + look.px(300)                                   # (the label starts under the picture)
    label = layer.crop((m, lab_top, m + card_w, layer.height)).copy()
    # draw only the label: the panel behind it is part of the layer, so measure the label layer itself
    lab = gfxcards.fit_label(look.case(name), look, card_w - 10, max(look.px(20), int(look.px(300) * 0.15)))
    box = _ink_box(lab)
    assert box is not None
    assert box[0] >= 1 and box[2] <= lab.width - 1, (name, box, lab.width)       # the first and last letters are on the card
    assert lab.width <= card_w                                                   # and the label is no wider than the card
    assert label.width == card_w


def test_a_logo_tile_is_inside_the_frame_with_margins_at_three_shapes(tmp_path):
    for frame, (W, H) in FRAMES.items():
        look = gfxcards.Look(None, W, H)
        spec = {"kind": "entity", "items": [{"type": "logo", "file": _logo(tmp_path), "name": "Northfield University", "at_rel": 0.0}]}
        sc = gfxcards.build_scene("entity", look, spec, 3.0)
        assert sc is not None, frame
        fr = gfxcards.frame_at(sc, 1.5)
        box = _ink_box(fr)
        assert box is not None
        m = 0.02 * min(W, H)
        assert box[0] >= m and box[1] >= m and box[2] <= W - m and box[3] <= H - m, (frame, box)       # never cropped by the edge


# ---------------------------------------------------------------------------- A: it stays off the faces (the real cause)
def _faces(W, H, n=6):
    """Small heads spread over the upper middle of a wide shot (zones as the face check returns them, 0..1)."""
    out = []
    for k in range(n):
        cx = 0.40 + 0.04 * k
        out.append((cx - 0.02, 0.10 + 0.01 * (k % 2), cx + 0.02, 0.30))
    return out


@pytest.mark.parametrize("frame", list(FRAMES))
def test_a_logo_card_is_placed_clear_of_every_face(tmp_path, frame):
    W, H = FRAMES[frame]
    look = gfxcards.Look(None, W, H)
    zones = _faces(W, H)
    look.avoid = [(a * W, b * H, c * W, d * H) for a, b, c, d in zones]
    spec = {"kind": "entity", "items": [{"type": "logo", "file": _logo(tmp_path), "name": "Northfield University", "at_rel": 0.0}]}
    sc = gfxcards.build_scene("entity", look, spec, 3.0)
    if sc is None:                                           # no room that leaves every face clear: dropped, never put over a face
        return
    box = _ink_box(gfxcards.frame_at(sc, 1.5))
    for a, b, c, d in look.avoid:
        overlap = not (box[2] <= a or box[0] >= c or box[3] <= b or box[1] >= d)
        assert not overlap, (frame, box, (a, b, c, d))


def test_the_face_check_sees_small_faces_in_a_wide_shot(monkeypatch, tmp_path):
    """facezones asks the detector for faces down to 1.2 % of the width at 960 px (it used 3 % at 480 px, so a group of students was invisible)."""
    seen = {}

    def fake(src, start, dur, models_dir, rate, width=640, mouths=False, min_w=0.03):
        seen.update(width=width, min_w=min_w)
        return [[]], 1
    from clipfarm.clipper import reframe
    monkeypatch.setattr(reframe, "detect_faces", fake)
    facezones.zones(tmp_path / "x.mp4", 0.0, 2.0, tmp_path, 16 / 9)
    assert seen["width"] >= 960 and seen["min_w"] <= 0.015


def test_detect_faces_keeps_the_old_default_for_framing():
    import inspect

    from clipfarm.clipper import reframe
    assert inspect.signature(reframe.detect_faces).parameters["min_w"].default == 0.03


# ---------------------------------------------------------------------------- B: the Final Cut title is where the app draws it
def test_the_hook_title_is_wrapped_like_the_app_and_centred_on_its_text():
    W, H = 1920, 1080
    g = FX.graphic_title("Why did Maple Hall's extra space sit empty?", "hook", W, H, theme="glass")
    assert len(g["lines"]) >= 2 and "\n" in g["text"]                      # the app wraps at about half the frame: a Basic Title would not
    assert g["color"].startswith(("1.000 1.000 1.000", "0.9"))             # a dark glass look: white words
    x, y = g["pos"]
    assert x < 0 and y < 0                                                   # bottom left
    assert -W / 2 < x < 0 and -H / 2 < y < -H * 0.1
    look = gfxcards.Look("glass", W, H)
    longest = max(len(ln) for ln in g["lines"])
    assert (x + W / 2) - (look.px(38) + 0) > 0 and longest > 0
    # the block of text (centre +- half its width) is on the frame
    mt = gfxcards.lower_metrics(look, {"text": "Why did Maple Hall's extra space sit empty?"}, "hook")
    assert x - mt["text_w"] / 2 >= -W / 2 and x + mt["text_w"] / 2 <= W / 2


def test_the_title_follows_the_panel_the_renderer_really_drew():
    W, H = 1920, 1080
    default = FX.graphic_title("Find Your Fit", "hook", W, H, theme="glass")
    moved = FX.graphic_title("Find Your Fit", "hook", W, H, rect=(1200, 100, 600, 160), theme="glass")      # (it moved to a clear spot at the top right)
    assert moved["pos"][0] > default["pos"][0] and moved["pos"][1] > default["pos"][1]
    assert moved["pos"][1] > 0                                                                         # up in the frame (y is up in Final Cut)


def test_the_panel_of_a_lower_third_is_reported_for_any_look():
    for theme in ("glass", "clean", "slate", "bold"):
        look = gfxcards.Look(theme, 1920, 1080)
        sc = gfxcards.build_scene("hook", look, {"kind": "hook", "text": "Find your fit"}, 3.0)
        assert sc is not None and sc.panel is not None, theme                     # (a flat look has no glass rectangle: the panel is kept apart)
        x, y, w, h = sc.panel
        assert 0 <= x and 0 <= y and x + w <= 1920 and y + h <= 1080


def test_the_metrics_helper_gives_the_scene_its_own_numbers():
    look = gfxcards.Look("glass", 1920, 1080)
    text = "A fairly long headline that has to go on two lines in the card"
    mt = gfxcards.lower_metrics(look, {"text": text}, "hook")
    sc = gfxcards.build_scene("hook", look, {"kind": "hook", "text": text}, 3.0)
    layer_h = mt["line_h"] * len(mt["lines"]) + 2 * 2 + 6 + int(mt["size"] * 0.3)
    px, py, pw, ph = gfxcards.lower_panel_at(look, mt["text_w"], layer_h)
    assert sc.panel == (px, py, pw, ph)                                          # the default spot is the one the scene uses without faces


# ---------------------------------------------------------------------------- B: what the FCPXML says
def _xml(titles, overlays=None):
    items = [{"kind": "clip", "out": 0, "dur": 8.0, "path": "/Volumes/X/Shoot/C0001.MP4", "start": 0.0, "w": 3840, "h": 2160,
              "fps": 23.976023976023978, "audio": True, "file_dur": 600.0, "tc": 0.0}]
    return FX.build_multicam("Talk", 1920, 1080, Fraction(24000, 1001), None, items, titles=titles, overlays=overlays)


def _dtd_errors(xml, tmp_path):
    p = tmp_path / "t.fcpxml"
    p.write_text(xml)
    return FX.validate(p) if (FX.DTD_DIR.exists() or Path("/Applications/Motion.app").exists()) else []


def test_the_hook_title_in_the_fcpxml_is_white_wrapped_and_not_a_black_box(tmp_path):
    xml = _xml([{"at": 0.0, "len": 3.2, "text": "Why did Maple Hall's extra space sit empty?", "kind": "hook", "theme": "glass"}])
    m = re.search(r'<title [^>]*name="Hook title"[^>]*>.*?</title>', xml, re.S)
    assert m, xml[:400]
    t = m.group(0)
    assert "Arial Black" not in t and 'fontColor="0.07 0.07 0.07 1"' not in t and 'backgroundColor="1 1 1 1"' not in t
    assert "Helvetica Neue" in t and 'fontColor="1.000 1.000 1.000 1"' in t
    assert "\n" in re.search(r"<text-style ref=\"ts\d+\">(.*?)</text-style>", t, re.S).group(1)            # wrapped by us
    pos = re.search(r'value="(-?[\d.]+) (-?[\d.]+)"', t)
    x, y = float(pos.group(1)), float(pos.group(2))
    assert -960 < x < 0 and -540 < y < 0
    assert _dtd_errors(xml, tmp_path) == []


def test_a_double_quote_and_newline_in_a_title_stay_valid(tmp_path):
    xml = _xml([{"at": 0.0, "len": 3.0, "text": 'He said "wait" & <left>', "kind": "hook", "theme": "glass"}])
    assert _dtd_errors(xml, tmp_path) == []


def test_a_title_that_also_exists_as_a_clip_is_switched_off(tmp_path):
    xml = _xml([{"at": 0.0, "len": 3.2, "text": "Find your fit", "kind": "hook", "theme": "glass", "off": True},
                {"at": 5.0, "len": 2.0, "text": "Key point", "kind": "point", "theme": "glass"}])
    hook = re.search(r'<title [^>]*name="Hook title[^"]*"[^>]*>', xml).group(0)
    assert 'enabled="0"' in hook and "editable copy" in hook
    key = re.search(r'<title [^>]*name="Key point"[^>]*>', xml).group(0)
    assert "enabled" not in key                                                       # no clip of its own: this title IS the graphic
    assert _dtd_errors(xml, tmp_path) == []


def test_fcp_titles_marks_the_graphics_that_have_a_clip_and_keeps_their_panel():
    opts = {"graphics": "all", "hook_text": "Find your fit", "gfx_theme": "glass", "callouts": [{"at": 12.0, "len": 2.0, "text": "Two points"}]}
    plain = polish.fcp_titles(opts, "T")
    assert [t.get("off") for t in plain] == [False, False] and plain[0]["theme"] == "glass"
    rect = (115, 700, 900, 160)
    out = polish.fcp_titles(opts, "T", {"hook": {"rect": rect}, 12.0: {"rect": None}})
    assert out[0]["off"] and out[0]["rect"] == rect and out[1]["off"] and out[1]["rect"] is None
