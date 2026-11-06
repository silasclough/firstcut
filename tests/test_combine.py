import json
import subprocess

import pytest

from clipfarm.clipper import combine as C
from clipfarm.core import ffmpeg


def clip(path, w=240, h=320, seconds=2, color="red", created=None, meta=None, audio=True):
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"color=c={color}:s={w}x{h}:r=30"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=330:sample_rate=48000"]
    cmd += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    if created:
        cmd += ["-metadata", f"creation_time={created}"]
    for k, v in (meta or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    cmd += ["-movflags", "use_metadata_tags", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def test_same_device_files_join_without_reencoding_in_time_order(tmp_path):
    a = clip(tmp_path / "part_000__b.mp4", created="2026-10-09T12:00:05Z", meta={"model": "Ray-Ban Meta Smart Glasses"})
    b = clip(tmp_path / "part_001__a.mp4", seconds=3, created="2026-10-09T11:00:00Z")
    out = tmp_path / "source.mp4"
    meta = C.combine([a, b], out)
    assert meta["mode"] == "copy"
    assert [f["name"] for f in meta["files"]] == ["a.mp4", "b.mp4"]                 # recorded earlier goes first
    assert meta["files"][1]["start"] == pytest.approx(3.0, abs=0.1)
    assert meta["files"][1]["glasses"] and not meta["files"][0]["glasses"]
    assert ffmpeg.duration(out) == pytest.approx(5.0, abs=0.2)
    assert json.loads((tmp_path / "parts.json").read_text())["files"][0]["name"] == "a.mp4"


def test_mixed_files_are_normalised(tmp_path):
    a = clip(tmp_path / "part_000__x.mp4", 240, 320)
    b = clip(tmp_path / "part_001__y.mp4", 320, 180, color="blue", audio=False)       # landscape, silent
    out = tmp_path / "source.mp4"
    meta = C.combine([a, b], out)
    vs = ffmpeg.video_stream(ffmpeg.probe(out))
    assert meta["mode"] == "reencode" and (vs["width"], vs["height"]) == (240, 320)
    assert ffmpeg.audio_stream(ffmpeg.probe(out)) is not None
    assert ffmpeg.duration(out) == pytest.approx(4.0, abs=0.25)


def test_a_single_file_is_taken_as_it_is_not_copied_through_a_join(tmp_path):
    a = clip(tmp_path / "part_000__only.mp4", seconds=3)
    size = a.stat().st_size
    out = tmp_path / "source.mp4"
    seen = []
    meta = C.combine([a], out, progress=seen.append)
    assert meta["mode"] == "single" and seen == ["Preparing the video"]               # not "Joining 1 videos"
    assert not a.exists() and out.stat().st_size == size                               # moved, byte for byte: no second copy, no re-wrapping
    assert [f["name"] for f in meta["files"]] == ["only.mp4"] and meta["files"][0]["end"] == pytest.approx(3.0, abs=0.1)


def test_boundary_note():
    files = [{"name": "a.mp4", "start": 0, "end": 60}, {"name": "b.mp4", "start": 60, "end": 130}]
    note = C.boundary_note(files)
    assert "2 separate recordings" in note and "1:00.0 (b.mp4)" in note
    assert C.boundary_note(files, 70, 130) == "" and C.boundary_note(files[:1]) == ""
