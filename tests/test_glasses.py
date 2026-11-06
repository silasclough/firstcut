import subprocess

import pytest

from clipfarm.clipper import glasses as G
from clipfarm.clipper import moments as M
from clipfarm.core import ffmpeg


def make_clip(path, w, h, seconds=4, meta=None):
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=30",
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", str(seconds),
           "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest"]
    for k, v in (meta or {}).items():
        cmd += ["-metadata", f"{k}={v}"]
    cmd += ["-movflags", "use_metadata_tags", str(path)]
    subprocess.run(cmd, check=True, timeout=120)
    return path


def test_detect_from_metadata(tmp_path):
    p = make_clip(tmp_path / "g.mp4", 360, 480, meta={"make": "Meta AI", "model": "Ray-Ban Meta Smart Glasses"})
    d = G.detect(p)
    assert d["glasses"] and d["shape_match"] and "Meta" in d["evidence"] and d["dims"] == "360x480"


def test_shape_only_is_not_auto_glasses(tmp_path):
    p = make_clip(tmp_path / "s.mp4", 360, 480)                     # stripped metadata (e.g. phone upload)
    d = G.detect(p)
    assert not d["glasses"] and d["shape_match"]
    p2 = make_clip(tmp_path / "l.mp4", 640, 360, meta={"make": "Sony"})
    d2 = G.detect(p2)
    assert not d2["glasses"] and not d2["shape_match"]


def test_contact_sheets_and_signals(tmp_path):
    p = make_clip(tmp_path / "g.mp4", 360, 480, seconds=6)
    sheets = G.contact_sheets(p, 0, 6, step=1.5)
    assert len(sheets) == 1 and sheets[0][0][:2] == b"\xff\xd8" and sheets[0][1][:2] == [0.0, 1.5]
    sig = G.visual_signals(p)
    assert len(sig["motion"]) >= 10 and all(0 <= b <= 1 for b in sig["bright"])
    assert "shaky" in G.signal_notes(sig, None, 0, 6)


def test_fit_time_never_cuts_words():
    words = [{"text": "hey", "start": 9.8, "end": 10.3}, {"text": "look", "start": 19.7, "end": 20.4}]
    m = G.fit_time(M.Moment(10.0, 20.0, -1, -1), 8, 40, 60, words)
    assert m.start == pytest.approx(9.8) and m.end == pytest.approx(20.45)
    m = G.fit_time(M.Moment(57.0, 59.0, -1, -1), 8, 40, 60, [])        # too short at the end: grows backwards
    assert m.end == 60 and m.duration >= 8
    m = G.fit_time(M.Moment(0, 90, -1, -1), 8, 40, 120, [])
    assert m.duration == 40


def test_heuristic_fallback_prefers_loud_steady_seconds():
    rate = 2.0
    secs = 60
    sig = {"rate": rate, "motion": [0.03] * int(secs * rate), "sharp": [1.0] * int(secs * rate),
           "bright": [0.5] * int(secs * rate)}
    for k in range(int(10 * rate), int(14 * rate)):          # whip pan at 10-14 s
        sig["motion"][k] = 0.4
    import numpy as np
    loud = np.zeros(secs)
    loud[40:44] = 3.0                                        # something happens at 40 s
    best = M.pick(G.heuristic_moments(sig, loud, [], secs, 8, 40), 1)[0]
    assert best.start <= 40 < best.end and not (best.start <= 11 < best.end) and best.title == ""
