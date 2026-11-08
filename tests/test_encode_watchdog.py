"""The picture encode watchdog: a hardware encode that stalls is killed and redone once in software, with a plain note; the software encode failing too raises."""
import os
import stat
import time

import pytest

from clipfarm.core import ffmpeg


def _fake_ffmpeg(tmp_path, body: str):
    p = tmp_path / "ffmpeg"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    return str(p)


def test_a_process_that_makes_no_progress_is_killed_by_the_watchdog(tmp_path):
    fake = _fake_ffmpeg(tmp_path, "sleep 60\n")
    t0 = time.monotonic()
    with pytest.raises(ffmpeg.FFmpegError, match="stalled"):
        ffmpeg.run_watched([fake, "-y", "x.mp4"], stall_s=2.0, what="the picture")
    assert time.monotonic() - t0 < 20


def test_a_stalled_hardware_encode_is_redone_once_in_software_with_a_note(tmp_path, monkeypatch):
    calls, notes = [], []

    def fake_watched(cmd, stall_s=60.0, timeout=None, what="ffmpeg"):
        calls.append((cmd, stall_s, what))
        if len(calls) == 1:
            raise ffmpeg.FFmpegError(f"{what} stalled (no progress for {stall_s:.0f}s)")
    monkeypatch.setattr(ffmpeg, "run_watched", fake_watched)
    monkeypatch.setattr(ffmpeg, "pick_encoder", lambda cfg: cfg.get("encoder") or "h264_videotoolbox")
    note = ffmpeg.encode_with_fallback(lambda vc: ["ffmpeg", vc.get("encoder", "hw")], {"encoder": "h264_videotoolbox"}, stall_s=120, what="the picture",
                                       on_note=notes.append)
    assert [c[0][1] for c in calls] == ["h264_videotoolbox", "libx264"]            # hardware first, then software, once
    assert calls[0][1] == 120 and "software" in calls[1][2]
    assert note == "the hardware encoder stalled; used the software encoder" and notes == [note]


def test_software_failing_too_raises_and_a_software_job_is_not_retried(tmp_path, monkeypatch):
    seen = []

    def always_fail(cmd, **kw):
        seen.append(cmd)
        raise ffmpeg.FFmpegError("the picture stalled (no progress for 120s)")
    monkeypatch.setattr(ffmpeg, "run_watched", always_fail)
    monkeypatch.setattr(ffmpeg, "pick_encoder", lambda cfg: cfg.get("encoder") or "h264_videotoolbox")
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.encode_with_fallback(lambda vc: ["ffmpeg", vc.get("encoder", "hw")], {"encoder": "h264_videotoolbox"}, stall_s=5)
    assert len(seen) == 2
    seen.clear()
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.encode_with_fallback(lambda vc: ["ffmpeg"], {"encoder": "libx264"}, stall_s=5)
    assert len(seen) == 2                                                             # software stalled: ONE more try, then it raises
    seen.clear()

    def exits_badly(cmd, **kw):
        seen.append(cmd)
        raise ffmpeg.FFmpegError("the picture exited 1:\nbad filter graph")
    monkeypatch.setattr(ffmpeg, "run_watched", exits_badly)
    with pytest.raises(ffmpeg.FFmpegError):
        ffmpeg.encode_with_fallback(lambda vc: ["ffmpeg"], {"encoder": "libx264"}, stall_s=5)
    assert len(seen) == 1                                                             # a real error is never retried


def test_a_healthy_encode_has_no_note(monkeypatch):
    monkeypatch.setattr(ffmpeg, "run_watched", lambda cmd, **kw: None)
    assert ffmpeg.encode_with_fallback(lambda vc: ["ffmpeg"], {"encoder": "h264_videotoolbox"}) == ""


def test_a_stalled_ffmpeg_that_keeps_printing_the_same_progress_block_is_still_killed(tmp_path):
    """The real stall: 0% CPU, but ffmpeg keeps printing an identical progress block. Comparing each line only with the previous (different-key) line made every
    repeat look like movement, so the watchdog never fired and a render waited forever."""
    body = ('i=0\nwhile [ $i -lt 400 ]; do\n'
            ' echo "frame=100"; echo "total_size=5767216"; echo "out_time_us=3000000"; echo "progress=continue"\n'
            ' sleep 0.3; i=$((i+1))\ndone\n')
    fake = _fake_ffmpeg(tmp_path, body)
    t0 = time.monotonic()
    with pytest.raises(ffmpeg.FFmpegError, match="stalled"):
        ffmpeg.run_watched([fake, "-y", "x.mp4"], stall_s=3.0, what="the picture")
    assert time.monotonic() - t0 < 30


def test_an_ffmpeg_that_keeps_making_progress_is_left_alone(tmp_path):
    body = ('i=1\nwhile [ $i -le 12 ]; do\n echo "frame=$i"; echo "total_size=$((i*1000))"; echo "out_time_us=$((i*100000))"; echo "progress=continue"\n sleep 0.4; i=$((i+1))\ndone\n'
            'echo "progress=end"\n')
    fake = _fake_ffmpeg(tmp_path, body)
    proc = ffmpeg.run_watched([fake, "-y", "x.mp4"], stall_s=3.0, what="the picture")
    assert proc.returncode == 0
