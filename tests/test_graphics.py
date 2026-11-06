import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from clipfarm.clipper import gfxcards as G
from clipfarm.clipper import gfxplan as GP
from clipfarm.clipper import moments as M
from clipfarm.clipper import polish
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

SENTS = [M.Sentence(0, 0.0, 6.0, "Hi, my name is Zahara Nakia and I am a mom of two."),
         M.Sentence(1, 6.0, 14.0, "In the first year our visits went from 4 a month to 18 a month."),
         M.Sentence(2, 14.0, 24.0, "We got three things: birthing classes, the baby boutique and financial classes."),
         M.Sentence(3, 24.0, 32.0, "The way this place impacted me the most was you could feel the love."),
         M.Sentence(4, 32.0, 40.0, "About 22 percent of the moms had no support at home.")]
SPOKEN = " ".join(s.text for s in SENTS)


def v(item):
    return GP.validate(item, SPOKEN, SENTS, 60.0, 3.2)


def test_only_what_is_said_becomes_a_graphic():
    assert v({"kind": "stat", "at": 33.0, "value": "22%", "label": "of moms had no support at home"})["value"] == "22%"
    assert v({"kind": "stat", "at": 33.0, "value": "45%", "label": "invented"}) is None                                   # a number nobody said
    ch = v({"kind": "chart", "at": 8.0, "title": "Visits a month", "chart_type": "column", "labels": ["Start", "After a year"], "values": ["4", "18"]})
    assert ch and ch["chart"]["values"] == ["4", "18"] and ch["chart"]["type"] == "column"
    assert v({"kind": "chart", "at": 8.0, "labels": ["A", "B"], "values": ["4", "99"]}) is None                         # only one of the two numbers was said
    assert v({"kind": "quote", "at": 25.0, "text": "you could feel the love", "who": "Zahara"})["text"] == "you could feel the love"
    assert v({"kind": "quote", "at": 25.0, "text": "this program saved my whole family from ruin"}) is None              # not said
    assert v({"kind": "lower", "at": 4.0, "name": "Zahara Nakia", "role": "Mom"})["name"] == "Zahara Nakia"
    assert v({"kind": "lower", "at": 4.0, "name": "Jane Smith", "role": "CEO"}) is None                                  # never said
    sl = v({"kind": "slide", "at": 15.0, "title": "What we got", "bullets": ["Birthing classes", "Baby boutique", "Financial classes", "A private jet"]})
    assert sl and sl["bullets"] == ["Birthing classes", "Baby boutique", "Financial classes"]                          # the unsaid bullet is dropped
    assert v({"kind": "slide", "at": 15.0, "title": "T", "bullets": ["Birthing classes", "A private jet", "A yacht"]}) is None   # fewer than three left


def test_graphics_are_placed_after_the_title_card_and_inside_the_video():
    assert v({"kind": "stat", "at": 0.5, "value": "22%", "label": "x"}) is None or v({"kind": "stat", "at": 0.5, "value": "22%", "label": "x"})["at"] >= 3.2
    assert v({"kind": "stat", "at": 59.0, "value": "22%", "label": "x"}) is None                                         # no room before the end


class FakeAI:
    def __init__(self, items):
        self.items = items

    def complete_json(self, *a, **k):
        return type("R", (), {"data": {"graphics": self.items}})()


def test_the_plan_respects_the_kind_of_video_and_never_overlaps():
    items = [{"kind": "lower", "at": 4.0, "name": "Zahara Nakia", "role": "Mom"}, {"kind": "stat", "at": 5.0, "value": "22%", "label": "no support"},
             {"kind": "quote", "at": 25.0, "text": "you could feel the love", "who": "Zahara"}, {"kind": "stat", "at": 33.0, "value": "22%", "label": "no support at home"}]
    story = {"kinds": ["lower", "quote"], "graphics_per_s": 60, "label": "story"}
    got = GP.plan(FakeAI(items), SENTS, "", story, 60.0)
    assert [g["kind"] for g in got] == ["lower"]                                                                         # one per 60 s, and only its kinds
    youtube = {"kinds": ["stat", "quote", "lower"], "graphics_per_s": 15, "label": "youtube"}
    got = GP.plan(FakeAI(items), SENTS, "", youtube, 60.0)
    assert len(got) >= 2 and all(a["at"] + a["len"] + GP.GAP <= b["at"] + 0.011 for a, b in zip(got, got[1:]))


# ------------------------------------------------------------------------------------------------ the renderer
def test_every_card_draws_with_glass_and_animates():
    specs = [{"kind": "hook", "text": "How one heartbeat turned fear into hope", "len": 4}, {"kind": "lower", "name": "Zahara Nakia", "role": "Mom", "len": 4},
             {"kind": "stat", "value": "22%", "label": "of moms had no support", "len": 4}, {"kind": "quote", "text": "you could feel the love", "who": "Z", "len": 5},
             {"kind": "slide", "title": "What we got", "bullets": ["Birthing classes", "Baby boutique", "Financial classes"], "len": 7},
             {"kind": "chart", "chart": {"type": "donut", "title": "Help", "labels": ["A", "B"], "values": ["60", "40"], "unit": "%"}, "len": 6}]
    for W, H in ((1920, 1080), (1080, 1920)):
        for sp in specs:
            sc = G.build_scene(sp["kind"], G.Look("glass", W, H), sp, sp["len"])
            assert sc.glass and all(g["w"] > 0 and g["h"] > 0 for g in sc.glass)
            early, mid = G.frame_at(sc, 0.05), G.frame_at(sc, min(2.5, sp["len"] - 1))
            assert mid.getchannel("A").getbbox() is not None                                                             # something is on screen
            assert early.getchannel("A").getbbox() is None or early.getchannel("A").getextrema()[1] < mid.getchannel("A").getextrema()[1] + 1
            assert G.frame_at(sc, sp["len"] - 0.01).getchannel("A").getextrema()[1] < 40                                  # faded out at the end


def test_a_card_is_one_transparent_movie(tmp_path):
    cfg = load_config()
    r = G.render(cfg, {"kind": "stat", "value": "22%", "label": "of moms had no support at home", "len": 2.0}, 960, 540, tmp_path / "c.mov", "glass")
    assert r and r["file"].exists() and 1.8 < ffmpeg.duration(r["file"]) < 2.3
    st = ffmpeg.video_stream(ffmpeg.probe(r["file"]))
    assert st["width"] == 960 and st["codec_name"] == "qtrle" and r["glass"][0]["w"] > 100
    assert G.render(cfg, {"kind": "nope", "len": 2}, 960, 540, tmp_path / "d.mov") is None                              # an unknown card is skipped, not fatal


def test_the_picture_behind_a_glass_panel_is_blurred_only_while_it_shows(tmp_path):
    ff = ffmpeg.bin_path("ffmpeg")
    src = tmp_path / "s.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=gray:s=640x360:r=24:d=6,noise=alls=70:allf=t+u", "-pix_fmt", "yuv420p", str(src)], check=True)
    r = {"x": 100, "y": 100, "w": 300, "h": 160, "r": 30, "dim": 0.0}
    mask = polish.write_mask(tmp_path / "m.png", 300, 160, 30)
    lines, cur = polish.glass_filters("base", 0, 2.0, 2.0, r, 1, 640, 360, 24.0)
    g = ["[0:v]format=yuv420p[base]"] + lines + [f"[{cur}]format=yuv420p[out]"]
    out = tmp_path / "o.mp4"
    subprocess.run([ff, "-y", "-v", "error", "-i", str(src), "-loop", "1", "-t", "2.6", "-i", str(mask), "-filter_complex", ";".join(g), "-map", "[out]", "-t", "6",
                    "-pix_fmt", "yuv420p", str(out)], check=True)
    def sharp(t):
        p = subprocess.run([ff, "-v", "error", "-ss", str(t), "-i", str(out), "-frames:v", "1", "-vf", "crop=300:160:100:100,format=gray,edgedetect=low=0.1:high=0.3,signalstats", "-f", "null", "-"],
                           capture_output=True, text=True)
        return p.stderr
    assert ffmpeg.duration(out) > 5.5                                                                                     # the program is not cut short by the chain
    full = subprocess.run([ff, "-v", "error", "-ss", "3.0", "-i", str(out), "-frames:v", "1", "-vf", "crop=300:160:100:100,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    before = subprocess.run([ff, "-v", "error", "-ss", "0.5", "-i", str(out), "-frames:v", "1", "-vf", "crop=300:160:100:100,format=gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    import numpy as np
    a, b = np.frombuffer(full, dtype=np.uint8).astype(float), np.frombuffer(before, dtype=np.uint8).astype(float)
    assert np.abs(np.diff(a)).mean() < 0.6 * np.abs(np.diff(b)).mean()                                                   # smoother inside the window: it is blurred


def test_graphic_clips_in_the_final_cut_project_are_valid(tmp_path):
    from clipfarm.clipper import fcpxml as FX
    if not FX.DTD_DIR.exists():
        pytest.skip("Final Cut Pro not installed here - DTD check skipped")
    group = {"angles": [{"id": "Cam A", "files": [{"name": "a1", "path": "/Volumes/X/Shoot/C0001.mp4", "t0": 0.0, "dur": 200.0, "w": 3840, "h": 2160, "fps": 24.0}]}],
             "recorders": [], "dur": 200.0}
    items = [{"kind": "mc", "out": 0, "dur": 12.0, "t": 10.0, "video": "Cam A", "audio": "Cam A"}, {"kind": "mc", "out": 12.0, "dur": 8.0, "t": 40.0, "video": "Cam A", "audio": "Cam A"}]
    ov = [{"at": 3.5, "len": 4.0, "path": "/Volumes/X/Proj/Graphics/graphic_01.mov", "w": 1920, "h": 1080, "fps": 30.0},
          {"at": 14.0, "len": 5.0, "path": "/Volumes/X/Proj/Graphics/graphic_02.mov", "w": 1920, "h": 1080, "fps": 30.0}]
    xml = FX.build_multicam("T", 1920, 1080, Fraction(24000, 1001), group, items, overlays=ov)
    p = tmp_path / "g.fcpxml"
    p.write_text(xml)
    assert xml.count('name="Graphic"') == 2 and "graphic_01.mov" in xml
    assert FX.validate(p) == []


def test_each_kind_of_video_has_its_own_style():
    from clipfarm.clipper import localproj as LP
    assert set(LP.KINDS) >= {"story", "business", "youtube", "promo", "ad"}
    assert LP.PROFILES["youtube"]["punch_ins"] and not LP.PROFILES["business"]["punch_ins"] and not LP.PROFILES["story"]["punch_ins"]    # no fast punch-ins for a business
    assert "chart" in LP.PROFILES["business"]["kinds"] and "chart" not in LP.PROFILES["story"]["kinds"]
    assert LP.PROFILES["youtube"]["graphics_per_s"] < LP.PROFILES["business"]["graphics_per_s"] < LP.PROFILES["story"]["graphics_per_s"]      # YouTube has the most
    assert LP.PROFILES["youtube"]["cut_per_s"] < LP.PROFILES["business"]["cut_per_s"]                                                     # and cuts the picture more often
    assert LP.video_kind(None, "", "business") == "business" and LP.video_kind(None, "", "youtube") == "youtube"


def test_the_graphics_choice_reaches_the_editor_and_the_page_offers_it():
    from clipfarm.clipper import aivideo
    html = (Path(aivideo.__file__).parent / "web" / "ai.html").read_text()
    assert html.count('value="all"') >= 2 and 'value="business"' in html and 'value="youtube"' in html


def test_the_look_is_matched_to_the_footage_and_the_kind_of_video():
    from clipfarm.clipper import graphics as GR
    for kind in ("business", "story", "youtube", "promo", "ad"):
        assert GR.match_theme(0.7, kind, "p") in GR.LOOKS["bright"][kind] and GR.match_theme(0.2, kind, "p") in GR.LOOKS["dark"][kind]
    assert GR.match_theme(0.7, "business", "p") in ("glass_light", "clean", "pastel_orange", "pastel_green", "pastel_blue", "pastel_pink") and GR.match_theme(0.2, "business", "p") in ("glass", "slate", "pastel_blue", "pastel_green")     # calm, light or dark
    assert "bold" in GR.LOOKS["dark"]["youtube"] and "bold" not in GR.LOOKS["dark"]["business"] and "bold" not in GR.LOOKS["bright"]["business"]    # loud only for YouTube
    assert GR.match_theme(0.7, "story", "same") == GR.match_theme(0.7, "story", "same")                                                        # stable for one video
    assert all(t in GR.THEMES for fam in GR.LOOKS.values() for ts in fam.values() for t in ts)


def test_every_look_draws_every_card_in_both_orientations_with_readable_ink():
    import numpy as np
    spec = {"kind": "stat", "value": "22%", "label": "of moms had no support", "len": 4}
    for theme in ("glass", "glass_light", "clean", "slate", "bold", "mono", "minimal"):
        for W, H in ((1920, 1080), (1080, 1920)):
            for kind, sp in (("stat", spec), ("quote", {"kind": "quote", "text": "you could feel the love", "who": "Z", "len": 5}),
                             ("slide", {"kind": "slide", "title": "T", "bullets": ["one", "two", "three"], "len": 7}),
                             ("lower", {"kind": "lower", "name": "Zahara", "role": "Mom", "len": 5})):
                look = G.Look(theme, W, H)
                sc = G.build_scene(kind, look, sp, sp["len"])
                fr = np.asarray(G.frame_at(sc, min(3.0, sp["len"] - 1.2)))
                assert fr[..., 3].max() > 200, (theme, W, H, kind)                                                            # something opaque is drawn
                assert all(g["tone"] == look.tone for g in sc.glass)
                if not look.glass:
                    assert all(g["dim"] for g in sc.glass)                                                                    # flat looks only veil the picture, never blur a panel
    bold = G.Look("bold", 1080, 1920)
    assert abs(sum(bold.fg) - sum(bold.accent)) > 100                                                                         # ink and card are not the same colour


# ------------------------------------------------------------------------------------------------ real pictures of named things
class _Pic:
    def __init__(self, tmp_path):
        from PIL import Image
        self.dir = tmp_path
        for n, mode in (("flag.png", (200, 30, 60, 255)), ("mark.png", (255, 255, 255, 255)), ("photo.jpg", (60, 90, 140, 255))):
            Image.new("RGBA", (400, 260), mode).convert("RGB" if n.endswith("jpg") else "RGBA").save(tmp_path / n)

    def resolve(self, cfg, item):
        table = {"japan": ("flag", "flag.png", None, "", False), "openai": ("logo", "mark.png", "#10A37F", "", True),
                 "anthropic": ("logo", "mark.png", "#191919", "", True), "sam altman": ("photo", "photo.jpg", None, "V / Wikimedia Commons (CC BY 2.0)", False)}
        hit = table.get(item["name"].lower())
        if not hit:
            return None
        t, f, color, credit, mark = hit
        return {"file": str(self.dir / f), "type": t, "name": item["name"], "credit": credit, "color": color, "mark": mark}


def _words(text, start=0.0, step=0.4):
    return [{"text": w, "start": round(start + i * step, 2), "end": round(start + i * step + 0.3, 2)} for i, w in enumerate(text.split())]


ENT_SENTS = [M.Sentence(0, 4.0, 14.0, "Japan is far away but OpenAI and Anthropic both opened offices there."),
             M.Sentence(1, 14.0, 24.0, "Then Sam Altman visited and Narnia was mentioned too.")]


def test_named_things_get_their_real_picture_when_they_are_said(tmp_path, monkeypatch):
    from clipfarm.clipper import entities as EN
    monkeypatch.setattr(EN, "resolve", _Pic(tmp_path).resolve)
    words = _words("Japan is far away but OpenAI and Anthropic both opened offices there.", 4.0) + _words("Then Sam Altman visited and Narnia was mentioned too.", 14.0)
    spoken = " ".join(s.text for s in ENT_SENTS)
    item = {"kind": "entity", "items": [{"name": "Japan", "type": "country", "at": 4.0}, {"name": "OpenAI", "type": "company", "at": 6.0},
                                        {"name": "Anthropic", "type": "company", "at": 6.9}, {"name": "Narnia", "type": "place", "at": 18.0},
                                        {"name": "Gotham", "type": "place", "at": 8.0}]}
    card = GP.entity_card(object(), item, spoken, words, ENT_SENTS, 40.0, 3.2)
    names = [i["name"] for i in card["items"]]
    assert names == ["Japan", "OpenAI", "Anthropic"]                       # Narnia has no picture, Gotham is never said: neither is shown
    rel = [i["at_rel"] for i in card["items"]]
    assert rel == sorted(rel) and rel[0] == pytest_approx(0.35, 0.02) and rel[1] > 1.5                  # each pops in at the moment it is spoken
    assert card["len"] > rel[-1] + 2.5 and card["at"] < 4.0 + 0.01
    card2 = GP.entity_card(object(), {"items": [{"name": "Sam Altman", "type": "person", "at": 14.5}]}, spoken, words, ENT_SENTS, 40.0, 3.2)
    assert card2["items"][0]["credit"].endswith("(CC BY 2.0)")                                           # a photo's credit travels with it


def pytest_approx(v, tol):
    class A:
        def __eq__(self, other):
            return abs(other - v) <= tol
    return A()


def test_the_planner_only_offers_pictures_when_it_can_look_them_up(tmp_path, monkeypatch):
    from clipfarm.clipper import entities as EN
    monkeypatch.setattr(EN, "resolve", _Pic(tmp_path).resolve)
    ai = FakeAI([{"kind": "entity", "at": 4.0, "items": [{"name": "Japan", "type": "country", "at": 4.0}]}])
    words = _words("Japan is far away but OpenAI and Anthropic both opened offices there.", 4.0)
    profile = {"kinds": ["entity"], "graphics_per_s": 30, "entities_per_s": 10, "label": "youtube"}
    assert [g["kind"] for g in GP.plan(ai, ENT_SENTS, "", profile, 30.0, words=words, cfg=object())] == ["entity"]
    assert GP.plan(ai, ENT_SENTS, "", profile, 30.0, words=words, cfg=None) == []                        # no way to look pictures up: no entity graphics


def test_the_picture_tiles_draw_in_every_look(tmp_path):
    p = _Pic(tmp_path)
    items = [{**p.resolve(None, {"name": n}), "at_rel": a} for n, a in (("Japan", 0.1), ("OpenAI", 0.9), ("Sam Altman", 1.7))]
    spec = {"kind": "entity", "len": 6, "items": items}
    import numpy as np
    for theme in ("glass", "glass_light", "clean", "slate", "bold"):
        for W, H in ((1920, 1080), (1080, 1920)):
            sc = G.build_scene("entity", G.Look(theme, W, H), spec, 6)
            early, late = np.asarray(G.frame_at(sc, 0.9))[..., 3], np.asarray(G.frame_at(sc, 4.0))[..., 3]
            assert late.sum() > early.sum() > 0                                                            # the third tile has not appeared yet at 0.9 s
            glass = [g for g in sc.glass if g["w"] > 0]
            assert (len(glass) == 3) == G.Look(theme, W, H).glass and all(g["t0"] >= 0 for g in glass)  # glass tiles start when their picture does


# ------------------------------------------------------------------------------------------------ never cover a face
def _hits(sc, look, zone_px):
    """Does anything drawn in the card touch the zone? (alpha of a frame where everything has appeared)"""
    import numpy as np
    fr = np.asarray(G.frame_at(sc, min(3.2, sc.seconds - 1.0)))[..., 3]
    x0, y0, x1, y1 = [int(v) for v in zone_px]
    return bool((fr[y0:y1, x0:x1] > 40).any())                                        # (a card's soft shadow is not 'covering')


def test_no_graphic_ever_covers_a_face():
    specs = [{"kind": "hook", "text": "How one heartbeat turned fear into hope", "len": 6}, {"kind": "lower", "name": "Zahara Nakia", "role": "Mom", "len": 6},
             {"kind": "stat", "value": "22%", "label": "of moms had no support", "len": 6},
             {"kind": "quote", "text": "you could feel the love of Christ whenever you walked in", "who": "Z", "len": 7},
             {"kind": "slide", "title": "What we got", "bullets": ["Birthing classes", "Baby boutique", "Financial classes"], "len": 9},
             {"kind": "chart", "chart": {"type": "column", "title": "Visits", "labels": ["A", "B", "C"], "values": ["4", "7", "12"]}, "len": 7}]
    faces_norm = {"centre": (0.36, 0.08, 0.64, 0.62), "left": (0.05, 0.10, 0.30, 0.65), "right": (0.70, 0.10, 0.95, 0.65), "low": (0.30, 0.45, 0.70, 0.98)}
    dropped = placed = 0
    for W, H in ((1920, 1080), (1080, 1920)):
        for name, z in faces_norm.items():
            for theme in ("glass", "clean", "bold"):
                for sp in specs:
                    look = G.Look(theme, W, H)
                    look.avoid = [(z[0] * W, z[1] * H, z[2] * W, z[3] * H)]
                    sc = G.build_scene(sp["kind"], look, sp, sp["len"])
                    if sc is None:
                        dropped += 1                                                       # no room beside this face: dropped, not laid over it
                        continue
                    placed += 1
                    assert not _hits(sc, look, look.avoid[0]), (W, H, name, theme, sp["kind"])
                    assert all(g["dim"] == 0 or g["w"] < W or sp["kind"] == "quote" for g in sc.glass)     # a picture with a person in it is never blurred or darkened whole (except behind a quote, by request)
    assert placed > dropped > -1 and placed > 100                                           # most graphics still find a place (smaller, or at the side)


def test_a_graphic_keeps_its_usual_place_when_nobody_is_in_the_way():
    look = G.Look("glass", 1920, 1080)
    sc = G.build_scene("lower", look, {"kind": "lower", "name": "Zahara", "role": "Mom", "len": 5}, 5)
    xs = [l["x"] for l in sc.layers]
    assert min(xs) < 1920 * 0.2 and sc.glass                                                # bottom-left, as before
    look.avoid = [(1920 * 0.02, 1080 * 0.55, 1920 * 0.45, 1080 * 0.98)]                    # a face where the lower third would go
    moved = G.build_scene("lower", look, {"kind": "lower", "name": "Zahara", "role": "Mom", "len": 5}, 5)
    assert moved is not None and min(l["x"] for l in moved.layers) > 1920 * 0.4              # it goes to the free side instead


def test_head_zones_have_room_for_hair_and_chin():
    from clipfarm.clipper import facezones
    from clipfarm.clipper.reframe import Face
    z = facezones.zone_of(Face(0.0, 0.5, 0.4, 0.12, 0.9), 16 / 9)
    assert z[0] < 0.5 - 0.12 / 2 and z[2] > 0.5 + 0.12 / 2 and z[1] < 0.4 - 0.12 * 16 / 9 / 2 and z[3] > 0.4 + 0.12 * 16 / 9 / 2


# ------------------------------------------------------------------------------------------------ new looks and new kinds of graphic
NEW_SENTS = [M.Sentence(0, 4.0, 14.0, "The Mac mini has the M5 Pro, a fifteen core CPU, a sixteen core GPU and up to 64 GB of memory."),
             M.Sentence(1, 14.0, 26.0, "It starts at $1,699 and tops out at $2,699, so about a thousand dollars between them."),
             M.Sentence(2, 26.0, 40.0, "On September 22 the post went viral, and on September 23 DigitalOcean answered.")]
NEW_SPOKEN = " ".join(s.text for s in NEW_SENTS)


def nv(item):
    return GP.validate(item, NEW_SPOKEN, NEW_SENTS, 60.0, 3.2)


def test_chips_compare_and_timeline_only_say_what_was_said():
    ch = nv({"kind": "chips", "at": 5.0, "bullets": ["M5 Pro", "fifteen core CPU", "sixteen core GPU", "a flying car"]})
    assert ch["chips"] == ["M5 Pro", "fifteen core CPU", "sixteen core GPU"]                                   # the unsaid chip is dropped
    assert nv({"kind": "chips", "at": 5.0, "bullets": ["a flying car", "a jetpack"]}) is None
    cm = nv({"kind": "compare", "at": 15.0, "labels": ["Starts at", "Tops out at"], "values": ["$1,699", "$2,699"]})
    assert cm and cm["values"] == ["$1,699", "$2,699"]
    assert nv({"kind": "compare", "at": 15.0, "labels": ["A", "B"], "values": ["$1,699", "$9,999"]}) is None   # a price nobody said
    tl = nv({"kind": "timeline", "at": 28.0, "events": [{"date": "September 22", "label": "the post went viral"}, {"date": "September 23", "label": "DigitalOcean answered"},
                                                       {"date": "October 5", "label": "everything changed"}]})
    assert tl and [e["date"] for e in tl["events"]] == ["September 22", "September 23"]                       # October 5 was never said


def test_a_hub_and_a_versus_need_their_pictures_and_their_words(tmp_path, monkeypatch):
    from clipfarm.clipper import entities as EN
    monkeypatch.setattr(EN, "resolve", _Pic(tmp_path).resolve)
    spoken = "OpenAI works with Anthropic and Japan runs a program but Gotham does not."
    words = _words(spoken, 5.0)
    sents = [M.Sentence(0, 5.0, 17.0, spoken)]
    hub = GP.entity_card(object(), {"kind": "hub", "items": [{"name": "OpenAI", "type": "company", "at": 5.0}, {"name": "Anthropic", "type": "company", "at": 6.5},
                                                              {"name": "Japan", "type": "country", "at": 8.5}]}, spoken, words, sents, 40.0, 3.2)
    assert hub["kind"] == "hub" and hub["items"][0]["name"] == "OpenAI" and hub["items"][0]["at_rel"] == 0.0 and len(hub["items"]) == 3
    assert GP.entity_card(object(), {"kind": "hub", "items": [{"name": "OpenAI", "type": "company", "at": 5.0}, {"name": "Gotham", "type": "place", "at": 9.0}]},
                          spoken, words, sents, 40.0, 3.2) is None                                              # not enough pictures for a hub
    vs = GP.entity_card(object(), {"kind": "entity", "joiner": "vs", "items": [{"name": "OpenAI", "type": "company", "at": 5.0}, {"name": "Anthropic", "type": "company", "at": 6.5}]},
                        spoken, words, sents, 40.0, 3.2)
    assert vs["kind"] == "entity" and vs["joiner"] == "vs"


def test_every_new_card_draws_in_every_look_and_both_orientations(tmp_path):
    import numpy as np
    p = _Pic(tmp_path)
    ents = [{**p.resolve(None, {"name": n}), "at_rel": a} for n, a in (("OpenAI", 0.0), ("Anthropic", 1.0), ("Japan", 2.0), ("Sam Altman", 3.0))]
    cards = {"hub": {"kind": "hub", "len": 9, "items": ents}, "vs": {"kind": "entity", "len": 6, "joiner": "vs", "items": ents[:2]},
             "chips": {"kind": "chips", "len": 7, "chips": ["M5 Pro", "15-core CPU", "16-core GPU"]},
             "compare": {"kind": "compare", "len": 7, "labels": ["Starts", "Tops out"], "values": ["$1,699", "$2,699"]},
             "timeline": {"kind": "timeline", "len": 8, "events": [{"date": "Sep 22", "label": "went viral"}, {"date": "Sep 23", "label": "they answered"}]}}
    for theme in ("glass", "glass_light", "clean", "slate", "bold", "mono", "minimal", "pastel_orange", "pastel_green", "pastel_blue", "pastel_pink", "board"):
        for W, H in ((1920, 1080), (1080, 1920)):
            for name, sp in cards.items():
                sc = G.build_scene(sp["kind"], G.Look(theme, W, H), sp, sp["len"])
                assert sc is not None, (theme, W, H, name)
                fr = np.asarray(G.frame_at(sc, min(5.5, sp["len"] - 1.2)))
                assert fr[..., 3].max() > 200, (theme, W, H, name)


def test_the_pastel_looks_are_one_colour_and_the_board_look_is_monospaced():
    for t in ("pastel_orange", "pastel_green", "pastel_blue", "pastel_pink"):
        look = G.Look(t, 1920, 1080)
        assert look.mode == "pastel" and look.light and not look.glass and len(look.palette) >= 5
        hues = {max(c) - min(c) > 8 for c in look.palette}                                                   # every chart colour is a tint or shade of the one accent
        assert hues == {True}
    from clipfarm.clipper import graphics as GR
    assert all(t in GR.THEMES for t in ("board", "pastel_orange", "pastel_green", "pastel_blue", "pastel_pink"))
    assert any("pastel" in t for t in GR.LOOKS["bright"]["business"]) and "board" in GR.LOOKS["dark"]["youtube"]
    a = G.build_scene("stat", G.Look("board", 1920, 1080), {"kind": "stat", "value": "$200", "label": "a month", "len": 5}, 5)
    assert a is not None and not a.glass                                                                     # the board look's number is a tilted sticker


def test_a_quote_card_comes_up_when_the_quoted_words_begin_not_when_the_sentence_does():
    lead = "After a very long and difficult year of everything going wrong I finally sat down and understood that"
    quote = "every single heartbeat is a reason to keep going"
    text = f"{lead} {quote}."
    words, t = [], 10.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3})
        t += 0.35
    sents = [M.Sentence(0, 10.0, t, text)]
    begins = next(w["start"] for w in words if w["text"] == "every")
    v = GP.validate({"kind": "quote", "at": 10.0, "text": quote, "who": "Zahara"}, text, sents, 90.0, 3.0, words)
    assert v and abs(v["at"] - (begins - 0.1)) < 0.05 and v["at"] > 12.0                              # (it used to be 10.2: the start of the sentence)
    look = G.Look("slate", 1920, 1080)
    look.avoid = [(1500, 100, 1800, 600)]
    sc = G.build_scene("quote", look, {"kind": "quote", "text": quote, "who": "Zahara", "len": 5}, 5)
    assert any(g["dim"] > 0 and g["w"] == 1920 for g in sc.glass)                                   # and the picture behind it goes soft, even with a person in it


def test_a_stat_is_just_the_number_and_the_words_go_in_the_label():
    sents = [M.Sentence(0, 4.0, 14.0, "It outputs 10 ready-to-go clips for Instagram and YouTube shorts.")]
    v = GP.validate({"kind": "stat", "at": 5.0, "value": "10 ready-to-go", "label": "from one YouTube link"}, sents[0].text, sents, 60.0, 3.0)
    assert v and v["value"] == "10" and v["label"].startswith("ready-to-go")
    assert GP.validate({"kind": "stat", "at": 5.0, "value": "twenty percent growth overall", "label": "x"}, sents[0].text, sents, 60.0, 3.0) is None


def test_big_text_keeps_the_tails_of_its_letters():
    f = G.font("heavy", 120)
    img = G.text_layer(["ready-to-go"], f, (255, 255, 255), line_h=1.0)
    import numpy as np
    a = np.asarray(img)[..., 3]
    assert a[-4:].max() == 0 and a[:-int(120 * 0.2)].max() > 0                                      # nothing is cut off at the bottom edge


def test_an_emphasis_card_reveals_each_word_as_it_is_spoken_and_underlines_the_key_phrase():
    import numpy as np
    text = "Some of my lowest viewed videos generated the most revenue."
    words, t = [], 20.0
    for w in ("Honestly " + text).split():
        words.append({"text": w, "start": t, "end": t + 0.3})
        t += 0.42
    sents = [M.Sentence(0, 20.0, t, "Honestly " + text)]
    v = GP.validate({"kind": "statement", "at": 20.4, "text": text, "emphasis": "generated the most revenue"}, "Honestly " + text, sents, 90.0, 3.0, words)
    assert v and v["emphasis"] == "generated the most revenue" and len(v["times"]) == len(text.split()) and v["times"] == sorted(v["times"])
    first_word_at = 20.0 + 0.42
    assert abs(v["at"] + v["times"][0] - first_word_at) < 0.02                                         # the first word shows exactly when it is said
    assert GP.validate({"kind": "statement", "at": 20.4, "text": "Some of my lowest viewed videos made nothing at all"}, "x", sents, 90.0, 3.0, words) is None   # not said
    assert GP.validate({"kind": "statement", "at": 20.4, "text": text}, "x", sents, 90.0, 3.0, None) is None                                     # no word timing: no card
    for theme in ("clean", "slate", "board", "pastel_blue"):
        for W, H in ((1920, 1080), (1080, 1920)):
            sc = G.build_scene("statement", G.Look(theme, W, H), v, v["len"])
            early, late = np.asarray(G.frame_at(sc, 0.9)).astype(int), np.asarray(G.frame_at(sc, 4.5)).astype(int)
            assert early[..., 3].min() > 200 and np.abs(early - late).sum() > 0                          # it fills the frame, and the picture changes as words arrive


def test_hub_tiles_never_overlap_even_squeezed_beside_a_face(tmp_path):
    p = _Pic(tmp_path)
    ents = [{**p.resolve(None, {"name": n}), "at_rel": a} for n, a in (("OpenAI", 0.0), ("Anthropic", 1.0), ("Japan", 2.0), ("Sam Altman", 3.0))]
    spec = {"kind": "hub", "len": 9, "items": ents}
    for W, H, face in ((1920, 1080, (0.30, 0.05, 0.72, 0.95)), (1080, 1920, (0.2, 0.15, 0.8, 0.62)), (1920, 1080, None)):
        look = G.Look("glass_light", W, H)
        if face:
            look.avoid = [(face[0] * W, face[1] * H, face[2] * W, face[3] * H)]
        sc = G.build_scene("hub", look, spec, 9)
        if sc is None:
            continue                                                                              # no room: dropped, not piled up
        rects = [(L["x"] + 98, L["y"] + 98, L["x"] + L["img"].width - 98, L["y"] + L["img"].height - 98) for L in sc.layers if L["img"].width > 120]     # (the card, without its soft shadow)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                assert min(a[2], b[2]) - max(a[0], b[0]) <= 6 or min(a[3], b[3]) - max(a[1], b[1]) <= 6, (W, H, face, a, b)


def test_the_name_under_a_logo_is_centred_on_its_card(tmp_path):
    import numpy as np
    p = _Pic(tmp_path)
    ent = p.resolve(None, {"name": "OpenAI"})
    look = G.Look("clean", 1920, 1080)
    tile, w, h, m = G._tile(look, ent, 170)
    a = np.asarray(tile)[..., 3].astype(int)
    lab = np.asarray(tile)[int(m + 22 + 170 + 12):int(m + h - 14), :, :3].astype(int).sum(axis=2)       # the strip holding the name (dark ink on the light card)
    cols = np.where((lab < 300).any(axis=0))[0]
    assert len(cols) and abs((cols.min() + cols.max()) / 2 - (m + w / 2)) <= 6, (cols.min(), cols.max(), m + w / 2)


def test_colliding_graphics_are_shortened_or_nudged_not_dropped():
    cards = [{"kind": "lower", "at": 7.0, "len": 4.5}, {"kind": "stat", "at": 9.5, "len": 4.2}, {"kind": "quote", "at": 20.0, "len": 6.7},
             {"kind": "chips", "at": 23.0, "len": 5.8}, {"kind": "statement", "at": 24.0, "len": 5.0}]
    out = GP.schedule(cards, 90.0)
    kinds = [c["kind"] for c in out]
    assert kinds[:4] == ["lower", "stat", "quote", "chips"]                       # the stat only needed the lower third a little shorter; chips start after the quote
    assert "statement" not in kinds                                              # a speech-timed card is never moved: no room means no card
    for a, b in zip(out, out[1:]):
        assert a["at"] + a["len"] + GP.GAP <= b["at"] + 0.011
    assert out[2]["len"] == 6.7                                                   # (the quote kept its full reading time)


def test_the_glass_blur_works_on_a_small_copy_in_a_4k_frame():
    lines, label = polish.glass_filters("base", 0, 5.0, 4.0, {"x": 0, "y": 0, "w": 3840, "h": 2160, "r": 0, "dim": 0.3, "tone": "light"}, None, 3840, 2160, 30.0)
    joined = ";".join(lines)
    assert "scale=960:540:flags=area" in joined and "scale=3840:2160:flags=bilinear" in joined and label == "gf0"
    small = ";".join(polish.glass_filters("base", 0, 5.0, 4.0, {"x": 10, "y": 10, "w": 600, "h": 300, "r": 20, "dim": 0.0}, None, 1280, 720, 30.0)[0])
    assert "scale=" not in small.split("gblur")[0].split("crop=")[-1]                      # (a small frame is blurred as it is)


def test_flags_that_do_not_fit_in_a_row_beside_a_big_face_stack_in_the_side_column(tmp_path):
    p = _Pic(tmp_path)
    ents = [{**p.resolve(None, {"name": "Japan"}), "name": n, "at_rel": i} for i, n in enumerate(("Japan", "China", "Mongolia"))]
    look = G.Look("pastel_blue", 1920, 1080)
    face = (0.27, 0.0, 0.77, 0.94)
    look.avoid = [(face[0] * 1920, face[1] * 1080, face[2] * 1920, face[3] * 1080), (0, 0.67 * 1080, 1920, 1080)]
    sc = G.build_scene("entity", look, {"kind": "entity", "items": ents, "len": 6}, 6)
    assert sc is not None and len(sc.layers) == 3
    xs = {L["x"] for L in sc.layers}
    assert len(xs) == 1 and max(L["x"] + L["img"].width for L in sc.layers) < 0.27 * 1920 + 40       # one column, left of the face


def test_the_transcript_scan_finds_ranges_figures_dates_and_lists():
    from clipfarm.clipper import gfxdetect as D
    assert D.number_of("2,000") == 2000 and D.number_of("one") == 1 and D.number_of("twenty five") == 25 and D.number_of("two thousand") == 2000 and D.number_of("banana") is None
    def sents(*texts):
        out, t = [], 0.0
        for i, x in enumerate(texts):
            out.append(M.Sentence(i, t, t + 4.0, x)); t += 5.0
        return out
    f = D.find(sents("The numbers could be as high as 2,000, and the numbers could be as low as one."), [])
    c = [o for o in f if o["kind"] == "compare"]
    assert len(c) == 1 and c[0]["item"]["values"] == ["2,000", "1"] and c[0]["item"]["labels"] == ["As high as", "As low as"]
    assert [o["kind"] for o in D.find(sents("Happiness goes up to 72% when you sleep."), [])] == ["stat"]
    assert [o["kind"] for o in D.find(sents("In 2019 I started. In 2021 I quit. In 2024 I came back."), [])] == ["timeline"]
    assert "slide" in [o["kind"] for o in D.find(sents("First, it opens with a hook.", "Second, it cuts dead air."), [])]
    assert D.find(sents("I went to the store and bought some bread."), []) == []


def test_a_range_the_ai_forgot_is_added_by_code_and_still_checked_against_the_words():
    text = "The numbers could be as high as 2000, and the numbers could be as low as one."
    words, t = [], 10.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3}); t += 0.4
    sents = [M.Sentence(0, 10.0, t, text)]
    ai = type("AI", (), {"complete_json": lambda self, *a, **k: type("R", (), {"data": {"graphics": []}})()})()
    out = GP.plan(ai, sents, "", {"kinds": ["compare", "stat"], "graphics_per_s": 20}, 60.0, 3.2, words)
    assert [c["kind"] for c in out] == ["compare"] and out[0]["values"] == ["2000", "1"]


FULL_SPECS = {"fullslide": {"kind": "fullslide", "title": "A good short video does four things", "bullets": ["Opens with a hook", "Gets rid of dead air", "Adds captions", "Ends on one point"], "len": 8, "at": 3.0},
              "bigstat": {"kind": "bigstat", "value": "47%", "label": "of my viewers leave in the first five seconds", "len": 5, "at": 3.0},
              "chapter": {"kind": "chapter", "text": "How it actually works", "number": "02", "len": 4, "at": 3.0},
              "fullquote": {"kind": "fullquote", "text": "The first cut is for you, the second cut is for them", "who": "My old editor", "len": 6, "at": 3.0},
              "statement": {"kind": "statement", "text": "Some of my lowest viewed videos generated the most revenue.", "emphasis": "most revenue", "len": 6, "times": [0.4 + 0.4 * i for i in range(10)], "at": 3.0}}


def test_full_screen_cards_draw_everywhere_and_arrive_with_a_transition():
    import numpy as np
    for name, sp in FULL_SPECS.items():
        for theme in ("clean", "slate", "bold", "board", "pastel_blue", "glass"):
            for W, H in ((1920, 1080), (1080, 1920)):
                for style in G.TRANSITIONS:
                    sc = G.build_scene(name, G.Look(theme, W, H), {**sp, "style": style}, sp["len"])
                    assert sc is not None, (name, theme, W, H)
                    start, mid, end = (np.asarray(G.frame_at(sc, t))[..., 3] for t in (0.0, 0.25, sp["len"] * 0.6))
                    assert start.max() == 0                                                  # nothing at the first instant
                    assert 0 < mid.mean() < end.mean()                                          # part of the page (or a fading page) while the transition runs, all of it after
                    assert (end > 200).mean() > 0.97                                         # and it is a whole opaque page
                    last = np.asarray(G.frame_at(sc, sp["len"] - 0.15))[..., 3]
                    assert last.mean() < end.mean()                                              # it leaves the same way


def test_the_planner_checks_full_screen_cards_like_their_small_cousins_and_spaces_them_out():
    sents = [M.Sentence(0, 10.0, 22.0, "It outputs 10 ready-to-go clips and the first thing it does is open with a hook, second it cuts dead air, third it adds captions."),
             M.Sentence(1, 24.0, 40.0, "Forty-seven percent of my viewers leave in the first five seconds. So the second part of this video is how it actually works.")]
    spoken = " ".join(s.text for s in sents)
    v = GP.validate({"kind": "fullslide", "at": 11.0, "title": "What it does", "bullets": ["open with a hook", "cuts dead air", "adds captions"]}, spoken, sents, 90.0, 3.2)
    assert v and v["kind"] == "fullslide" and len(v["bullets"]) == 3
    assert GP.validate({"kind": "fullslide", "at": 11.0, "title": "What", "bullets": ["fly to the moon", "eat lunch", "win a prize"]}, spoken, sents, 90.0, 3.2) is None
    assert GP.validate({"kind": "bigstat", "at": 25.0, "value": "47%", "label": "of viewers leave"}, spoken, sents, 90.0, 3.2)["kind"] == "bigstat"
    assert GP.validate({"kind": "bigstat", "at": 25.0, "value": "99%", "label": "x"}, spoken, sents, 90.0, 3.2) is None
    ch = GP.validate({"kind": "chapter", "at": 30.0, "text": "How it actually works", "number": "2"}, spoken, sents, 90.0, 3.2)
    assert ch and ch["number"] == "02"
    assert GP.validate({"kind": "chapter", "at": 30.0, "text": "Unicorn Rainbow Factory"}, spoken, sents, 90.0, 3.2) is None
    ai = type("AI", (), {"complete_json": lambda self, *a, **k: type("R", (), {"data": {"graphics": [
        {"kind": "fullslide", "at": 11.0, "title": "What it does", "bullets": ["open with a hook", "cuts dead air", "adds captions"]},
        {"kind": "bigstat", "at": 25.0, "value": "47%", "label": "of viewers leave"}]}})()})()
    out = GP.plan(ai, sents, "", {"kinds": ["fullslide", "bigstat"], "graphics_per_s": 5}, 90.0, 3.2, [])
    assert [c["kind"] for c in out] == ["fullslide", "bigstat"] and out[0]["style"] != out[1]["style"] and all(c["style"] in G.TRANSITIONS for c in out)


NEW_FULL = {"search": {"kind": "search", "text": "Who qualifies for the program?", "answer": "First-time buyers", "number": "02", "len": 6, "at": 3.0},
            "twocol": {"kind": "twocol", "title": "Same income. Different order.", "titles": ["Buyer A", "Buyer B"], "steps": [["Finds a condo", "Signs a contract", "Then applies"], ["Applies first", "Gets approved", "Then shops"]],
                       "tag": {"side": 0, "text": "Ruled out"}, "len": 8, "at": 3.0},
            "ledger": {"kind": "ledger", "title": "The condo math", "rows": [{"label": "Purchase price", "value": "$400,000"}, {"label": "Down payment", "value": "$14,000"}, {"label": "Closing costs", "value": "$8,000"}],
                       "total": {"label": "Cash needed", "value": "$22,000"}, "len": 8, "at": 3.0},
            "recap": {"kind": "recap", "bullets": ["Check eligibility first", "Apply before you sign", "Keep the paperwork ready"], "len": 7, "at": 3.0}}


def test_the_new_full_screen_cards_draw_in_every_look_and_shape():
    import numpy as np
    for name, sp in NEW_FULL.items():
        for theme in ("clean", "slate", "bold", "board", "pastel_green", "glass", "mono", "minimal"):
            for W, H in ((1920, 1080), (1080, 1920), (1280, 720)):
                sc = G.build_scene(name, G.Look(theme, W, H), sp, sp["len"])
                assert sc is not None, (name, theme, W, H)
                end = np.asarray(G.frame_at(sc, sp["len"] * 0.8))[..., 3]
                assert (end > 200).mean() > 0.97 and np.asarray(G.frame_at(sc, 0.0))[..., 3].max() == 0


def test_the_search_question_is_typed_and_the_answer_waits_for_the_end_of_the_typing():
    import numpy as np
    sc = G.build_scene("search", G.Look("clean", 1920, 1080), NEW_FULL["search"], 6)
    a, b, c = (np.asarray(G.frame_at(sc, t).convert("RGB")).astype(int) for t in (1.0, 2.0, 5.0))
    assert np.abs(a - b).sum() > 0 and np.abs(b - c).sum() > 0                                 # the picture keeps changing: typing, then the answer
    ink = lambda f: (f.sum(-1) < 150).sum()
    assert ink(a) < ink(b) < ink(c)                                                            # more dark text at each step


def test_new_full_screen_cards_are_proved_from_the_words_and_timed_to_them():
    text = ("Who qualifies for the program? First-time buyers do. Buyer A finds a condo, then signs a contract, then applies. Buyer B applies first, gets approved, then shops. "
            "The purchase price is 400,000 dollars, the down payment is 14,000 dollars and closing costs are 8,000 dollars, so you need 22,000 dollars in cash.")
    words, t = [], 10.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3}); t += 0.4
    sents = [M.Sentence(0, 10.0, t, text)]
    sp = lambda **k: GP.validate(k, text, sents, 120.0, 3.2, words)
    s = sp(kind="search", at=10.0, text="Who qualifies for the program?", answer="First-time buyers")
    assert s and s["answer"] == "First-time buyers" and s["text"].endswith("?")
    assert sp(kind="search", at=10.0, text="What is the airspeed of a swallow?") is None
    assert sp(kind="search", at=10.0, text="Who qualifies for the program?", answer="Martians only")["answer"] == ""        # an answer that was not said is dropped, the question stays
    tc = sp(kind="twocol", at=14.0, titles=["Buyer A", "Buyer B"], steps=[["finds a condo", "signs a contract", "then applies"], ["applies first", "gets approved", "then shops"]], tag={"side": 0, "text": "Ruled out"})
    assert tc and tc["tag"] == {"side": 0, "text": "Ruled out"} and len(tc["times"]) == 7 and tc["times"] == sorted(tc["times"])
    assert sp(kind="twocol", at=14.0, titles=["A", "B"], steps=[["fly to mars", "eat lunch"], ["win a prize", "go home"]]) is None
    lg = sp(kind="ledger", at=20.0, rows=[{"label": "purchase price", "value": "$400,000"}, {"label": "down payment", "value": "$14,000"}, {"label": "closing costs", "value": "$8,000"}], total={"label": "Cash needed", "value": "$22,000"})
    assert lg and len(lg["rows"]) == 3 and lg["total"]["value"] == "$22,000"
    lg2 = sp(kind="ledger", at=20.0, rows=[{"label": "purchase price", "value": "$400,000"}, {"label": "down payment", "value": "$14,000"}], total={"label": "Cash", "value": "$99,999"})
    assert lg2 and lg2["total"] == {}                                                                              # a total nobody said is not shown
    assert sp(kind="ledger", at=20.0, rows=[{"label": "purchase price", "value": "$400,000"}, {"label": "yacht", "value": "$5,000,000"}]) is None
    assert sp(kind="recap", at=10.0, bullets=["check eligibility", "apply first"]) is None                         # the bottom line belongs at the end of a video


def test_a_recap_belongs_at_the_end_and_a_slide_lights_up_each_point_as_it_is_said():
    text = "There are three steps. First open an account. Second add the documents. Third wait for approval. So the bottom line is simple: apply first, then shop."
    words, t = [], 80.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3}); t += 0.45
    sents = [M.Sentence(0, 80.0, t, text)]
    sl = GP.validate({"kind": "fullslide", "at": 80.0, "title": "Three steps", "bullets": ["open an account", "add the documents", "wait for approval"]}, text, sents, 100.0, 3.2, words)
    assert sl and len(sl["times"]) == 3 and sl["times"] == sorted(sl["times"]) and sl["times"][0] < sl["times"][1] < sl["times"][2]
    assert abs((sl["at"] + sl["times"][1]) - next(w["start"] for w in words if w["text"].startswith("documents"))) < 1.0
    rc = GP.validate({"kind": "recap", "at": 85.0, "bullets": ["apply first", "then shop"]}, text, sents, 100.0, 3.2, words)
    assert rc and rc["kind"] == "recap" and rc["title"] == "The bottom line" and len(rc["bullets"]) == 2
    sc = G.build_scene("fullslide", G.Look("clean", 1920, 1080), sl, sl["len"])
    assert sc is not None


def test_the_graphics_check_list_names_every_card_and_the_sentence_behind_it():
    cards = [{"kind": "stat", "at": 12.0, "value": "47%", "label": "of viewers leave"}, {"kind": "ledger", "at": 75.0, "title": "Math", "rows": [{"label": "Price", "value": "$400,000"}], "total": {"label": "Cash", "value": "$22,000"}}]
    sents = [M.Sentence(0, 10.0, 15.0, "Forty-seven percent of viewers leave."), M.Sentence(1, 74.0, 80.0, "The price is 400,000 dollars so you need 22,000 in cash.")]
    md = GP.check_notes(cards, sents)
    assert "0:12  stat" in md and "47%" in md and "Forty-seven percent of viewers leave." in md and "1:15  ledger" in md and "$22,000" in md and "Graphics to check" in md
    assert "no graphics" in GP.check_notes([], sents)


def test_a_stat_panel_pops_up_on_the_word_that_says_the_number():
    text = "I looked at everything for a long while and then I realised that 47 percent of my viewers leave early."
    words, t = [], 10.0
    for w in text.split():
        words.append({"text": w, "start": t, "end": t + 0.3}); t += 0.4
    sents = [M.Sentence(0, 10.0, t, text)]
    v = GP.validate({"kind": "stat", "at": 10.2, "value": "47%", "label": "of viewers leave early"}, text, sents, 60.0, 3.2, words)
    said = next(w["start"] for w in words if w["text"] == "47")
    assert v and abs(v["at"] - (said - 0.25)) < 0.02                                         # (not at the start of the sentence, 5 s too early)
