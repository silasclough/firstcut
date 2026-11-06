"""Voice cleanup (voiceclean.py): DeepFilterNet on noisy or far-away speech only, MP4 only, cached, and nothing at all when it isn't
installed. The real program is replaced by a small script that copies its input (the real one is downloaded by Setup)."""
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipfarm.clipper import voiceclean as VC
from clipfarm.core import ffmpeg


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    (h / "runtime" / "bin").mkdir(parents=True)
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(h))
    VC._setting.update(at=0.0, val={})
    VC._flags.clear()
    return h


def fake_program(home: Path) -> Path:
    exe = home / "runtime" / "bin" / "deep-filter"
    exe.write_text('#!/bin/sh\nif [ "$1" = "--help" ]; then echo "  -a, --atten-lim-db <DB>\n  -D, --compensate-delay\n  -o, --output-dir <DIR>"; exit 0; fi\n'
                   'out=""; prev=""; for a in "$@"; do if [ "$prev" = "--output-dir" ]; then out="$a"; fi; prev="$a"; last="$a"; done\n'
                   'echo "$@" >> "$(dirname "$0")/calls.txt"\ncp "$last" "$out/"\n')
    exe.chmod(0o755)
    return exe


def settings(home: Path, long="auto", short="auto"):
    (home / "settings.json").write_text(json.dumps({"defaults": {"long": {"voices": long}, "short": {"voices": short}}}))
    VC._setting.update(at=0.0, val={})


def tone(path: Path, seconds=6.0) -> Path:
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-y", "-f", "lavfi", "-i", f"sine=f=300:d={seconds}", "-ar", "48000", str(path)],
                   check=True)
    return path


def test_only_noisy_speech_is_cleaned_and_the_setting_decides(home):
    assert not VC.available() and not VC.wanted("bad")                                     # not installed: never
    fake_program(home)
    settings(home)
    assert VC.available() and VC.mode("long") == "auto"
    assert VC.wanted("bad") and VC.wanted("weak") and not VC.wanted("good")
    settings(home, long="always", short="never")
    assert VC.wanted("good", "long") and not VC.wanted("bad", "short")
    settings(home, long="nonsense")
    assert VC.mode("long") == "auto"


def test_a_cleaned_copy_is_made_once_with_a_gentle_cap(home, tmp_path):
    exe = fake_program(home)
    src = tone(tmp_path / "talk.wav", 8.0)
    wav, start = VC.cleaned_range(src, 3.0, 5.0, tmp_path / "clean")
    assert wav.exists() and start == pytest.approx(2.0)                                     # (a second of margin)
    calls = (exe.parent / "calls.txt").read_text().splitlines()
    assert len(calls) == 1 and "--atten-lim-db 12" in calls[0] and "--compensate-delay" in calls[0]
    again, _ = VC.cleaned_range(src, 3.0, 5.0, tmp_path / "clean")
    assert again == wav and len((exe.parent / "calls.txt").read_text().splitlines()) == 1   # cached
    VC.cleaned_range(src, 3.0, 5.0, tmp_path / "clean", strength="strong")
    assert "--atten-lim-db 24" in (exe.parent / "calls.txt").read_text().splitlines()[-1]


def test_a_failing_program_is_a_plain_error(home, tmp_path):
    exe = home / "runtime" / "bin" / "deep-filter"
    exe.write_text("#!/bin/sh\nexit 3\n")
    exe.chmod(0o755)
    with pytest.raises(VC.VoiceCleanError):
        VC.cleaned_range(tone(tmp_path / "a.wav"), 1.0, 2.0, tmp_path / "clean")


def _runner(grade_for):
    from clipfarm.clipper import localproj as LP
    g = SimpleNamespace(span=lambda path, a, b: {"grade": grade_for(path)})
    return LP, SimpleNamespace(cfg=None, grades=lambda st: g, save_grades=lambda st: None)


def test_long_videos_clean_only_the_noisy_recording_and_final_cut_keeps_the_original(home, tmp_path):
    fake_program(home)
    settings(home)
    far, near = tone(tmp_path / "far.wav", 12.0), tone(tmp_path / "near.wav", 12.0)
    LP, me = _runner(lambda p: "bad" if "far" in p else "good")
    st = SimpleNamespace(dir=tmp_path / "state", put=lambda name, v: (tmp_path / name).write_text(json.dumps(v)))
    st.dir.mkdir()
    parts = [{"d": 2.0, "a": {"path": str(far), "ss": 1.0}}, {"d": 2.0, "a": {"path": str(far), "ss": 3.0}},
             {"d": 2.0, "a": {"path": str(near), "ss": 1.0}}]
    out = LP.LocalRunner.clean_voices(me, parts, st)
    assert out[0]["a"]["path"] == str(far) and out[0]["a"]["clean_path"].endswith(".wav")   # the original stays for Final Cut
    assert out[0]["a"]["clean_path"] == out[1]["a"]["clean_path"]                            # one cleaned copy for the stretch
    assert out[0]["a"]["clean_ss"] == pytest.approx(1.0) and out[1]["a"]["clean_ss"] == pytest.approx(3.0)
    assert "clean_path" not in out[2]["a"]                                                 # good sound: untouched
    assert json.loads((tmp_path / "voice_clean.json").read_text())["cleaned"][0]["grade"] == "bad"


def test_nothing_changes_when_it_isnt_installed_or_set_to_never(home, tmp_path):
    LP, me = _runner(lambda p: "bad")
    parts = [{"d": 2.0, "a": {"path": str(tone(tmp_path / "x.wav")), "ss": 1.0}}]
    st = SimpleNamespace(dir=tmp_path, put=lambda *a: None)
    assert LP.LocalRunner.clean_voices(me, [dict(p) for p in parts], st) == parts           # not installed
    fake_program(home)
    settings(home, long="never")
    assert LP.LocalRunner.clean_voices(me, [dict(p) for p in parts], st) == parts


def test_short_clips_read_the_cleaned_copy(home, tmp_path, monkeypatch):
    from clipfarm.clipper import engine as E
    from clipfarm.clipper import soundcheck
    fake_program(home)
    settings(home)
    src = tone(tmp_path / "src.wav", 10.0)
    monkeypatch.setattr(soundcheck.Grades, "span", lambda self, path, a, b: {"grade": "weak"})
    me = SimpleNamespace(cfg=None)
    inputs, graph = ["-ss", "2.000", "-t", "5", "-i", str(src)], ["[0:v]trim=start=0:end=4[v0]", "[0:a]atrim=start=0:end=4[a0]"]
    i2, g2 = E.Engine.clean_voice(me, tmp_path, src, 2.0, 5.0, inputs, graph)
    assert i2[:6] == inputs and i2[-1].endswith(".wav") and i2[-3] == "1.000"              # (the cleaned copy starts a second earlier)
    assert g2 == ["[0:v]trim=start=0:end=4[v0]", "[1:a]atrim=start=0:end=4[a0]"]
    monkeypatch.setattr(soundcheck.Grades, "span", lambda self, path, a, b: {"grade": "good"})
    assert E.Engine.clean_voice(me, tmp_path / "other", src, 2.0, 5.0, inputs, graph) == (inputs, graph)


def test_the_setting_and_the_setup_step(tmp_path, monkeypatch):
    from clipfarm.clipper import studio as SD
    monkeypatch.setenv("VIDEOBUILDER_HOME", str(tmp_path))
    s = SD.Settings(tmp_path)
    assert s.get()["defaults"]["long"]["voices"] == "auto" and s.get()["defaults"]["short"]["voices"] == "auto"
    assert s.update({"defaults": {"long": {"voices": "always"}}})["defaults"]["long"]["voices"] == "always"
    assert s.update({"defaults": {"long": {"voices": "loud"}}})["defaults"]["long"]["voices"] == "auto"
    assert SD.st_gb("voices") == SD.VOICE_GB
