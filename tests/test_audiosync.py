import subprocess

import numpy as np

from clipfarm.clipper import audiosync as S
from clipfarm.core import ffmpeg

SR = 16000


def _speechy(seconds, seed):
    """Noise bursts with a speech-like on/off rhythm (the same 'voice' that two microphones would both hear)."""
    rng = np.random.default_rng(seed)
    y = np.zeros(int(seconds * SR), dtype=np.float32)
    t = 0.0
    while t < seconds - 0.5:
        d = rng.uniform(0.15, 0.7)
        i, j = int(t * SR), int(min(seconds, t + d) * SR)
        y[i:j] = rng.standard_normal(j - i).astype(np.float32) * rng.uniform(0.2, 0.6)
        t += d + rng.uniform(0.05, 0.5)
    return y


def _write(path, y, extra=None):
    raw = path.with_suffix(".f32")
    raw.write_bytes(y.astype(np.float32).tobytes())
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", str(raw), str(path)], check=True)


def test_a_clip_is_found_inside_a_longer_recording_to_within_a_few_ms(tmp_path):
    voice = _speechy(120, 3)
    rec = tmp_path / "recorder.wav"
    _write(rec, voice)
    start = 41.3                                                     # the camera started 41.3 s into the recording
    cam = voice[int(start * SR):int((start + 60) * SR)]
    cam = np.convolve(cam, np.ones(5) / 5, mode="same") + np.random.default_rng(9).standard_normal(len(cam)).astype(np.float32) * 0.05   # a different mic
    vid = tmp_path / "clip.mp4"
    cam_wav = tmp_path / "cam.wav"
    _write(cam_wav, cam)
    subprocess.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=black:size=160x120:rate=10:duration=60", "-i", str(cam_wav),
                    "-c:v", "libx264", "-c:a", "aac", "-shortest", str(vid)], check=True, timeout=120)
    env = S.envelope(S.decode(rec))
    m = S.match(vid, env, rec, 60.0)
    assert m is not None and abs(m["offset"] - start) < 0.03, m
    # a recording of something else does not match
    other = tmp_path / "other.wav"
    _write(other, _speechy(120, 77))
    assert S.match(vid, S.envelope(S.decode(other)), other, 60.0) is None
