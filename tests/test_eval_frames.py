"""Frame inspection of a finished render (WP15): planted defects are found, a clean clip is clean."""
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_eval_fixture import editor_folder  # noqa: E402,F401

from eval import frames as F  # noqa: E402
from fixtures import make_editor_folder as FX  # noqa: E402

from clipfarm.core import ffmpeg  # noqa: E402


def _clip(path: Path, folder: Path, planted: bool) -> None:
    """10 s of moving picture with the fixture's face oval on it; planted: a magenta card drawn over the face from 3 s to 6 s and a flat full-screen slide from 7 s to 9 s."""
    face = folder / "_f.png"
    FX._face_png(face, FX.FACE_A)
    card = "drawbox=x=100:y=40:w=140:h=70:color=magenta:t=fill:enable='between(t,3,6)',drawbox=x=0:y=0:w=320:h=180:color=0x203040:t=fill:enable='between(t,7,9)'," if planted else ""
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=10", "-i", str(face), "-filter_complex",
                    f"[0][1]overlay=0:0,{card}format=yuv420p[v]", "-map", "[v]", "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28", str(path)], check=True, timeout=120)


CARDS = [{"at": 3.0, "len": 3.0, "kind": "stat", "rect": (100, 40, 140, 70), "full": False}, {"at": 7.0, "len": 2.0, "kind": "slide", "rect": (0, 0, 320, 180), "full": True}]
CAPS = [{"t0": 3.5, "t1": 5.0, "rect": (110, 90, 100, 30)}, {"t0": 7.5, "t1": 8.5, "rect": (60, 120, 200, 30)}]


def test_frame_inspection_finds_planted_defects(editor_folder, tmp_path):  # noqa: F811
    clip = tmp_path / "planted.mp4"
    _clip(clip, tmp_path, planted=True)
    r = F.inspect(clip, tmp_path / "fr", face_box=FX.FACE_A, cards=CARDS, caption_groups=CAPS, captions_on_top=False, cutaways=[{"at": 4.0, "length": 2.0}],
                  protected=[(3.5, 5.0)], first_word_s=2.4, last_word_end_s=8.0, hook_speech_by_s=1.0)
    codes = {i["code"] for i in r["issues"]}
    assert {"face_hidden", "graphic_over_face", "caption_under_card", "broll_in_protected", "late_first_word", "late_last_frame"} <= codes, codes
    hidden = next(i for i in r["issues"] if i["code"] == "face_hidden")
    assert hidden["frame"].endswith(".png") and (tmp_path / "fr" / hidden["frame"]).exists()                    # a person can open the frame
    assert "face_hidden" in r["table_md"] and "| t | why | file |" in r["table_md"]
    assert r["duration"] == pytest.approx(10.0, abs=0.2) and len(r["frames"]) >= 6


def test_a_clean_clip_has_no_frame_issues(editor_folder, tmp_path):  # noqa: F811
    clip = tmp_path / "clean.mp4"
    _clip(clip, tmp_path, planted=False)
    r = F.inspect(clip, tmp_path / "fr", face_box=FX.FACE_A, cards=[], caption_groups=[], first_word_s=0.5, last_word_end_s=9.8, hook_speech_by_s=1.0)
    assert r["issues"] == [], r["issues"]
    assert "No issues." in r["table_md"]


def test_the_face_check_reads_pixels_not_the_plan(editor_folder, tmp_path):  # noqa: F811
    """A card the plan does not know about (no entry in `cards`) still shows up when it is in the picture and the planned cards cover it: face_visible_share."""
    clip = tmp_path / "p.mp4"
    _clip(clip, tmp_path, planted=True)
    shot_in = F.grab(clip, 4.5, tmp_path / "a.png")
    shot_out = F.grab(clip, 1.0, tmp_path / "b.png")
    assert F.face_visible_share(shot_in, FX.FACE_A) < 0.3 and F.face_visible_share(shot_out, FX.FACE_A) > 0.9
