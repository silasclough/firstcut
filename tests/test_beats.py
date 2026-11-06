import subprocess

import numpy as np

from clipfarm.clipper import beats as B
from clipfarm.core import ffmpeg


def _click_track(path, bpm=120, seconds=40, first=0.35):
    sr = 22050
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    period = 60.0 / bpm
    t = first
    while t < seconds - 0.1:
        i = int(t * sr)
        n = int(0.03 * sr)
        y[i:i + n] += (np.sin(2 * np.pi * 900 * np.arange(n) / sr) * np.exp(-np.arange(n) / (0.006 * sr))).astype(np.float32)
        t += period
    raw = path.with_suffix(".f32")
    raw.write_bytes(y.tobytes())
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", str(raw), str(path)], check=True)


def test_beat_grid_found_on_a_click_track(tmp_path):
    f = tmp_path / "clicks.wav"
    _click_track(f, bpm=120, first=0.35)
    r = B.analyze(f)
    assert r is not None and abs(r["bpm"] - 120) < 2.5
    d = (r["first"] - 0.35) % 0.5
    assert min(d, 0.5 - d) < 0.06                                                     # beats fall on the clicks


def test_a_track_without_a_pulse_gives_no_grid(tmp_path):
    f = tmp_path / "noise.wav"
    rng = np.random.default_rng(1)
    raw = tmp_path / "n.f32"
    raw.write_bytes((rng.standard_normal(22050 * 20) * 0.0).astype(np.float32).tobytes())
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "f32le", "-ar", "22050", "-ac", "1", "-i", str(raw), str(f)], check=True)
    assert B.analyze(f) is None


def test_cuts_snap_to_beats_without_clipping_words():
    beats = B.grid(0.5, 0.0, 60)
    words = [{"text": "a", "start": 0.0, "end": 1.0}, {"text": "b", "start": 1.9, "end": 2.6}, {"text": "c", "start": 4.0, "end": 4.6}]
    # stretches: 0.0-1.2 (cut at 1.2 -> beat at 1.0 would clip nothing: word ends 1.0), then 1.8-3.0, then 3.9-4.7
    out, n = B.snap_intervals([(0.0, 1.2), (1.8, 3.0), (3.9, 4.7)], words, beats)
    cuts = np.cumsum([b - a for a, b in out])[:-1]
    assert n >= 1 and all(abs(c / 0.5 - round(c / 0.5)) < 0.05 for c in cuts[:1])
    assert out[0][1] >= 1.0 + 0.05 - 1e-6                                              # the word that ends at 1.0 is not cut
    # a cut that would need to trim into speech is left alone
    out2, n2 = B.snap_intervals([(0.0, 1.05), (3.0, 4.7)], [{"text": "x", "start": 0.0, "end": 1.0}], B.grid(0.5, 0.0, 60))
    assert out2[0][1] >= 1.05 - 1e-6
