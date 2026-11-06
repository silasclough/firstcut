"""Line up a separate audio recording (a recorder / lav mic) with the video it belongs to - automatically, like a clap sync.

Drop the recorder's audio files anywhere in the project folder. For every video the editor finds where, inside each recording,
that video's own (camera) sound occurs:

    1. coarse   compare the loudness patterns (a 100 Hz energy curve) of the camera audio and the recording - speech has a pattern
                you can find even when the two microphones sound nothing alike
    2. fine     refine to a few milliseconds by cross-correlating a loud 8 s stretch (GCC-PHAT, which ignores differences in tone)

Result per video: recording + offset (recording time = video time + offset) and a confidence. The editor then uses the clean
recording as the sound for that video (and for its transcript) instead of the camera microphone. Each video is matched on its
own, so clock drift between the recorder and the camera never builds up.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.audiosync")
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".aif", ".aiff", ".ogg", ".opus"}
SR = 8000
QUIET_POWER = 1e-4                         # 90th-percentile frame power below this (about -40 dBFS) counts as a very quiet recording
TARGET_POWER = 5e-3                        # ...and is raised to this (about -23 dBFS)
HOP = 80                                   # 10 ms frames -> a 100 Hz energy curve
MIN_Z = 7.0                                # how far the best match must stand out from all the others
HEAD_S = 120.0                             # only the first two minutes of each video are compared (cheap on huge files)


def decode(path, start: float = 0.0, seconds: float | None = None, sr: int = SR) -> np.ndarray:
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{max(0.0, start):.3f}"]
    if seconds:
        cmd += ["-t", f"{seconds:.3f}"]
    cmd += ["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, timeout=3600).stdout, dtype=np.float32)


def envelope(y: np.ndarray) -> np.ndarray:
    n = len(y) // HOP
    if n < 20:
        return np.zeros(0)
    p = (y[:n * HOP].reshape(n, HOP) ** 2).mean(axis=1)
    loud = float(np.percentile(p, 90))
    if 0 < loud < QUIET_POWER:                                 # a camera whose microphone was barely connected (-60 dB and less): bring it up to a normal
        p = p * (TARGET_POWER / loud)                          # speech level first, or the curve is flat noise and never matches the other camera
    e = np.log1p(200.0 * p)
    k = 100                                                    # remove the slow background level (1 s)
    e = e - np.convolve(e, np.ones(k) / k, mode="same")
    return e - e.mean()


def coarse(env_a: np.ndarray, env_v: np.ndarray) -> tuple[int, float] | None:
    """-> (frame lag where the video starts inside the recording, z-score). Only positions where the recording fully covers the clip."""
    la, lv = len(env_a), len(env_v)
    if lv < 50 or la < lv:
        return None
    n = 1 << int(np.ceil(np.log2(la + lv)))
    c = np.fft.irfft(np.fft.rfft(env_a, n) * np.conj(np.fft.rfft(env_v, n)), n)[:la - lv + 1]
    sq = np.concatenate([[0.0], np.cumsum(env_a ** 2)])
    energy = np.sqrt(np.maximum(sq[lv:lv + la - lv + 1] - sq[:la - lv + 1], 1e-9)) * (np.linalg.norm(env_v) + 1e-9)
    ncc = c / energy                                           # normalised, so a long quiet stretch can't win by size
    k = int(np.argmax(ncc))
    rest = np.delete(ncc, slice(max(0, k - 30), k + 30))
    z = (ncc[k] - rest.mean()) / (rest.std() + 1e-9) if len(rest) > 10 else 0.0
    return k, float(z)


def fine(video, audio, v_t0: float, a_guess: float, seconds: float = 8.0, sr: int = 16000) -> float | None:
    """Refine: a loud stretch of the video at v_t0, found in the recording near a_guess. -> offset (recording time - video time)."""
    v = decode(video, v_t0, seconds, sr)
    pad = 1.0
    a = decode(audio, a_guess - pad, seconds + 2 * pad, sr)
    if len(v) < sr * 2 or len(a) < len(v):
        return None
    n = 1 << int(np.ceil(np.log2(len(a) + len(v))))
    r = np.fft.rfft(a, n) * np.conj(np.fft.rfft(v, n))
    cc = np.fft.irfft(r / (np.abs(r) + 1e-9), n)[:len(a) - len(v) + 1]
    s = int(np.argmax(cc))
    if cc[s] < 5 * (cc.std() + 1e-9):
        return None
    return (a_guess - pad + s / sr) - v_t0


def match(video, audio_env: np.ndarray, audio_path, v_dur: float) -> dict | None:
    """Where in this recording does this video's sound occur? -> {"offset", "score"} or None."""
    seconds = min(v_dur, HEAD_S)
    v = decode(video, 0.0, seconds)
    ev = envelope(v)
    got = coarse(audio_env, ev)
    if not got or got[1] < MIN_Z:
        return None
    lag, z = got
    guess = lag * HOP / SR                                     # where the video's first second falls in the recording
    # the loudest 8 s of what we compared
    win = int(8 * 100)
    best, best_i = -1.0, 0
    for i in range(0, max(1, len(ev) - win), 100):
        s = float(np.abs(ev[i:i + win]).sum())
        if s > best:
            best, best_i = s, i
    t0 = best_i / 100.0
    off = fine(video, audio_path, t0, guess + t0)
    if off is None:
        off = guess                                            # coarse alone is accurate to ~10 ms
    return {"offset": round(float(off), 4), "score": round(z, 1)}


def sync_all(files: list[dict], audios: list[dict], cache: dict, progress=None) -> dict:
    """Match every video with sound to the recordings. `cache` maps video name -> result (kept between runs)."""
    envs: dict[str, np.ndarray] = {}
    for n, f in enumerate(files):
        if not f["audio"] or f["dur"] < 5:
            continue
        key = f["name"]
        sig = f"{f['size']}:{f['mtime']}:" + ",".join(f"{a['name']}:{a['size']}" for a in audios)
        if cache.get(key, {}).get("sig") == sig:
            continue
        if progress:
            progress(n / max(1, len(files)), f"Syncing the recorder audio to {f['name']}")
        best = None
        for a in audios:
            if a["path"] not in envs:
                envs[a["path"]] = envelope(decode(a["path"]))
            try:
                m = match(f["path"], envs[a["path"]], a["path"], f["dur"])
            except Exception as e:  # noqa: BLE001 - an odd file just doesn't match
                log.warning(f"sync {f['name']} vs {a['name']} failed: {e}")
                continue
            if m and (best is None or m["score"] > best["score"]):
                best = {**m, "audio": a["path"], "audio_name": a["name"]}
        cache[key] = {"sig": sig, "result": best}
    return cache
