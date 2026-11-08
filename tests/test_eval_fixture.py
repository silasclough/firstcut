"""The shared evaluation fixture folder (WP15): synthetic footage with known facts, built once per session."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures import make_editor_folder as FX  # noqa: E402

from clipfarm.clipper import localproj as LP  # noqa: E402
from clipfarm.core import ffmpeg  # noqa: E402


@pytest.fixture(scope="session")
def editor_folder(tmp_path_factory):
    return FX.build(tmp_path_factory.mktemp("editor_folder"))


def test_the_fixture_folder_builds_and_scans_with_expected_roles(editor_folder):
    names = {p.name for p in editor_folder.iterdir()}
    assert set(FX.FILES) <= names and {"photo_01.jpg", "logo.png", "rec.wav", "notes.md", "facts.json", "talk_a.words.json"} <= names
    assert not any(n.startswith("_") for n in names)                                                          # scratch files are removed
    facts = FX.facts(editor_folder)
    assert [f["name"] for f in facts["files"]] == FX.FILES
    for f in facts["files"]:                                                                                  # the facts say what ffmpeg sees
        assert ffmpeg.duration(editor_folder / f["name"]) == pytest.approx(f["dur"], abs=0.3)
    by = {f["name"]: f for f in facts["files"]}
    assert by["talk_a.mp4"]["facts"]["role"] == "a_roll" and by["cam_b.mp4"]["facts"]["role"] == "a_roll" and by["drone.mp4"]["facts"]["role"] == "broll"
    assert by["broll_shaky.mp4"]["facts"]["motion"]["shake"] > by["broll_kitchen.mp4"]["facts"]["motion"]["shake"]
    assert by["cam_b.mp4"]["facts"]["fps"] == 50 and by["broll_kitchen.mp4"]["facts"]["fps"] == 60
    words = json.loads((editor_folder / "talk_a.words.json").read_text())
    assert " ".join(w["text"] for w in words[:8]).startswith("Welcome everyone to our summer camp week.")          # the known script
    second = json.loads((editor_folder / "talk_b.words.json").read_text())
    assert len(second) == len(words) + 4                                                                      # one restart: four extra words
    scanned = LP.scan(editor_folder)
    assert {f["name"] for f in scanned} >= set(FX.FILES)                                                      # the real scan reads every clip
    assert next(f for f in scanned if f["name"] == "talk_a.mp4")["audio"] and not next(f for f in scanned if f["name"] == "drone.mp4")["audio"]


def test_the_face_is_where_the_facts_say_it_is(editor_folder):
    from eval import frames as F
    shot = editor_folder.parent / "_face_probe.png"
    fr = F.grab(editor_folder / "talk_a.mp4", 2.0, shot)
    assert F.face_visible_share(fr, FX.FACE_A) > 0.9                                                          # skin where the box is
    assert F.face_visible_share(fr, (0.0, 0.0, 0.2, 0.2)) < 0.2                                               # and not in the corner


def test_the_generated_speech_bursts_sit_at_the_word_times(editor_folder):
    import numpy as np
    wav = editor_folder / "rec.wav"
    import wave
    with wave.open(str(wav)) as w:
        sr, data = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float)
    words = json.loads((editor_folder / "talk_a.words.json").read_text())
    w0 = words[0]
    inside = np.abs(data[int((w0["start"] + 0.36 + 0.1) * sr):int((w0["end"] + 0.36 - 0.1) * sr)]).mean()
    quiet = np.abs(data[int((w0["start"] + 0.36 - 0.4) * sr):int((w0["start"] + 0.36 - 0.05) * sr)]).mean()
    assert inside > 20 * max(quiet, 1.0)                                                                      # the recorder track is the same bursts, 0.36 s later
