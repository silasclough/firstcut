"""Real captions in the Final Cut project, and the picture-in-picture / split-screen B-roll layouts (2026-11-06, from the research list).

Captions: the words go into the .fcpxml as REAL captions (iTT role), always, so an editor fixes a word in Final Cut instead of re-rendering;
the burned-in captions of the MP4 (titles) stay as they were. Layouts: pip = the speaker in a small rounded window top right over a long
B-roll run; split = (a vertical video) the speaker on top, the B-roll below, the face kept in the half frame. Both are chosen sparingly and
stay ordinary, editable clips in Final Cut."""
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipfarm.clipper import fcpxml as FX
from clipfarm.clipper import localproj as LP
from clipfarm.core import ffmpeg
from clipfarm.core.config import load_config

DTD = Path("/Applications/Motion.app/Contents/Frameworks/Interchange.framework/Versions/A/Resources/FCPXMLv1_11.dtd")


def _valid(xml: str, tmp_path: Path) -> None:
    p = tmp_path / "p.fcpxml"
    p.write_text(xml)
    if DTD.exists():
        r = subprocess.run(["xmllint", "--noout", "--dtdvalid", str(DTD), str(p)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[:800]


def _clip(path: Path, color: str, w: int, h: int, secs: float) -> Path:
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r=24:d={secs}",
                    "-f", "lavfi", "-i", f"sine=f=220:d={secs}", "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", str(path)], check=True)
    return path


def _file(p: Path, start: float, broll: bool, w: int, h: int) -> dict:
    d = ffmpeg.duration(p)
    return {"name": p.name, "path": str(p), "start": start, "end": start + d, "dur": d, "w": w, "h": h, "fps": 24.0, "audio": True, "broll": broll}


# ---------------------------------------------------------------------------------------------- captions
def test_the_words_go_to_final_cut_as_real_captions_that_never_overlap(tmp_path):
    subs = [(0.5, 2.2, "Thanks for coming"), (2.0, 3.5, "tonight & welcome"), (3.5, 3.55, "x"), (5.0, 6.0, "  ")]
    spans = FX.caption_spans(subs)
    assert spans == [(0.5, 2.0, "Thanks for coming"), (2.0, 3.5, "tonight & welcome")]          # cut back, a sliver and a blank dropped
    items = [{"kind": "clip", "out": 0.0, "dur": 8.0, "path": "/x/talk.mov", "start": 10.0, "w": 1920, "h": 1080, "fps": 24.0, "audio": True}]
    xml = FX.build_multicam("Talk v2", 1920, 1080, Fraction(24), None, items, subs=subs)
    assert xml.count("<caption ") == 2 and f'role="{FX.CAPTION_ROLE}"' in xml and "tonight &amp; welcome" in xml
    assert 'name="Caption"' not in xml                                                       # (no burned-in titles were asked for)
    _valid(xml, tmp_path)
    simple = FX.build("Talk", FX.Media("/x/talk.mov", 20.0, width=1920, height=1080, fps=24.0), [(0, 8)], 1920, 1080, Fraction(24), subs=subs)
    assert simple.count("<caption ") == 2
    _valid(simple, tmp_path)
    assert FX.srt(subs).startswith("1\n00:00:00,500 --> 00:00:02,000\nThanks for coming\n")


# ---------------------------------------------------------------------------------------------- choosing layouts
def test_layouts_are_chosen_sparingly_and_only_where_the_face_matters():
    caws = lambda: [{"at": float(k * 10), "length": L} for k, L in enumerate([7.0, 8.0, 9.0, 6.5, 2.0])]
    got = LP.choose_layouts(caws(), 1920, 1080, "story")
    assert [c.get("layout") for c in got] == ["pip", None, None, "pip", None]                # one in three long ones; never a short one
    assert all("layout" not in c for c in LP.choose_layouts(caws(), 1920, 1080, "ad"))       # an ad: never
    assert all("layout" not in c for c in LP.choose_layouts(caws(), 1920, 1080, "story", on=False))
    assert [c.get("layout") for c in LP.choose_layouts([{"at": 0.0, "length": 3.5}], 1080, 1920, "business")] == ["split"]
    covered = [{"at": 0.0, "length": 9.0, "why": "nobody was filmed here", "must": True}]      # no speaker to show there
    assert "layout" not in LP.choose_layouts(covered, 1920, 1080, "story")[0]
    again = LP.choose_layouts(got, 1920, 1080, "ad")                                          # re-choosing clears the old choice
    assert all("layout" not in c for c in again)


def test_slicing_a_piece_moves_the_speakers_own_picture_too():
    p = {"d": 4.0, "out": 10.0, "v": {"path": "/b.mov", "ss": 2.0, "cut": True}, "v0": {"path": "/talk.mov", "ss": 30.0},
         "a": {"path": "/talk.mov", "ss": 30.0}, "item": {"kind": "clip", "start": 30.0}}
    q = LP._slice(p, 1.5, 2.0)
    assert q["v"]["ss"] == 3.5 and q["v0"]["ss"] == 31.5 and q["a"]["ss"] == 31.5


# ---------------------------------------------------------------------------------------------- rendering them
def _render(tmp_path, W, H, broll_color, length):
    talk = _clip(tmp_path / "talk.mp4", "red", W, H, 24)
    broll = _clip(tmp_path / "broll.mp4", broll_color, W, H, 20)
    files = [_file(talk, 0.0, False, W, H), _file(broll, 40.0, True, W, H)]
    caws = [{"at": 1.0, "length": length, "src": 41.0, "file": "broll.mp4", "what": "a hall", "why": "taste"}]
    r = LP.LocalRunner(load_config(), None, None)
    st = LP.State(tmp_path / "proj")
    st.dir.mkdir(parents=True, exist_ok=True)
    rough, ln, parts = r.assemble(files, [(2.0, 20.0)], caws, None, [], st, lambda f: None, even=False, style="story")
    assert abs(ln - 18.0) < 0.2
    frame = tmp_path / "f.png"
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-ss", "4", "-i", str(rough), "-frames:v", "1", str(frame)], check=True)
    return np.asarray(Image.open(frame).convert("RGB")).astype(int), parts, files, caws


def _is(px, color):
    want = {"red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 128, 0)}[color]
    return all(abs(a - b) < 60 for a, b in zip(px, want))


def test_pip_renders_the_speaker_in_a_window_and_final_cut_gets_editable_clips(tmp_path):
    img, parts, files, caws = _render(tmp_path, 640, 360, "blue", 8.0)
    H, W = img.shape[:2]
    m = int(H * FX.PIP_MARGIN)
    pw, ph = int(W * FX.PIP_SCALE) // 2 * 2, int(H * FX.PIP_SCALE) // 2 * 2
    assert caws[0]["layout"] == "pip"
    assert _is(img[H // 2, W // 3], "blue")                                                   # the B-roll fills the frame
    assert _is(img[m + ph // 2, W - m - pw // 2], "red")                                      # the speaker in the window, top right
    assert not _is(img[m + 1, W - m - pw + 1], "red")                                         # a rounded corner (not a square one)
    caw = LP.cutaways_of_parts(parts, files)
    assert [c.get("layout") for c in caw] == ["pip"]
    xml = FX.build_multicam("Talk", W, H, Fraction(24), None, LP.fcp_items(parts), cutaways=caw, subs=[(0.0, 1.0, "hi")])
    assert xml.count('name="Speaker (pip)"') >= 1 and 'scale="0.3 0.3"' in xml
    _valid(xml, tmp_path)


def test_split_puts_the_speaker_on_top_and_the_broll_below_in_a_vertical_video(tmp_path):
    img, parts, files, caws = _render(tmp_path, 360, 640, "green", 4.0)
    H, W = img.shape[:2]
    assert caws[0]["layout"] == "split"
    assert _is(img[H // 4, W // 2], "red") and _is(img[3 * H // 4, W // 2], "green")
    xml = FX.build_multicam("Talk", W, H, Fraction(24), None, LP.fcp_items(parts), cutaways=LP.cutaways_of_parts(parts, files))
    assert 'name="Speaker (split)"' in xml and '<trim-rect top="25" bottom="25"/>' in xml and 'position="0 -25"' in xml
    _valid(xml, tmp_path)


def test_a_layout_that_fails_falls_back_to_the_broll_full_screen(tmp_path, monkeypatch):
    monkeypatch.setattr(LP, "layout_graph", lambda *a, **k: "this is not a filter graph")
    img, parts, files, caws = _render(tmp_path, 640, 360, "blue", 8.0)
    H, W = img.shape[:2]
    m = int(H * FX.PIP_MARGIN)
    assert _is(img[m + 20, W - m - 40], "blue")                                               # no window: the plain shot
