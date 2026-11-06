"""Beat detection and "cut to the beat" for a music track (numpy only, no extra installs).

    analyze(path)            -> {"bpm", "period", "first", "beats": [seconds...]}   the track's beat grid
    snap_intervals(...)      nudge the end of every kept stretch so each cut lands on a beat of the music

The tracker is simple on purpose: a spectral-flux onset curve, its autocorrelation gives the tempo, and the phase that
puts the most onset energy on the grid gives where the beats fall. Good for the steady pop / lo-fi / cinematic beds a
video uses; a track with no pulse returns None and the cuts are left alone.
"""

from __future__ import annotations

import subprocess

import numpy as np

from ..core import ffmpeg

SR = 22050
HOP = 512
NFFT = 1024


def _decode(path, start: float = 0.0, seconds: float = 150.0) -> np.ndarray:
    raw = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{start:.2f}", "-t", f"{seconds:.2f}", "-i", str(path), "-vn",
                          "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"], capture_output=True, timeout=300).stdout
    return np.frombuffer(raw, dtype=np.float32)


def onset_envelope(y: np.ndarray) -> np.ndarray:
    if len(y) < NFFT * 4:
        return np.zeros(0)
    win = np.hanning(NFFT).astype(np.float32)
    n = 1 + (len(y) - NFFT) // HOP
    frames = np.lib.stride_tricks.as_strided(y, shape=(n, NFFT), strides=(y.strides[0] * HOP, y.strides[0]))
    mag = np.abs(np.fft.rfft(frames * win, axis=1))
    mag = np.log1p(10 * mag)
    flux = np.maximum(0.0, np.diff(mag, axis=0)).sum(axis=1)
    flux = np.concatenate([[0.0], flux])
    flux -= np.convolve(flux, np.ones(43) / 43, mode="same")                # remove the slow average
    return np.maximum(flux, 0.0)


def analyze(path, start: float = 0.0, seconds: float = 150.0) -> dict | None:
    y = _decode(path, start, seconds)
    env = onset_envelope(y)
    if len(env) < 200 or env.max() <= 0:
        return None
    fps = SR / HOP
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    lo, hi = int(fps * 60 / 180), int(fps * 60 / 60)                         # 60..180 BPM
    seg = ac[lo:hi + 1].copy()
    if seg.size < 3 or seg.max() <= 0:
        return None
    lags = np.arange(lo, hi + 1)
    bpm_axis = 60.0 * fps / lags
    weight = np.exp(-0.5 * (np.log2(bpm_axis / 115.0) / 0.9) ** 2)           # people hear 80-160 BPM as "the beat"
    best = int(np.argmax(seg * weight))
    lag = lags[best]
    if 0 < best < len(seg) - 1:                                               # refine between frames
        a, b, c = seg[best - 1], seg[best], seg[best + 1]
        denom = a - 2 * b + c
        lag = lag + (0.5 * (a - c) / denom if denom != 0 else 0.0)
    period = lag / fps
    if seg[best] / (ac[0] + 1e-9) < 0.03:                                    # no real pulse
        return None
    best_phase, best_score = 0.0, -1.0
    for ph in np.arange(0, period, 1.0 / fps):
        idx = np.round((ph + np.arange(0, len(env) / fps - ph, period)) * fps).astype(int)
        idx = idx[idx < len(env)]
        score = env[idx].mean() if len(idx) else 0.0
        if score > best_score:
            best_phase, best_score = float(ph), float(score)
    total = len(y) / SR
    beats = [round(start + best_phase + k * period, 3) for k in range(int((total - best_phase) / period) + 1)]
    return {"bpm": round(60.0 / period, 1), "period": round(period, 4), "first": round(start + best_phase, 3), "beats": beats}


def grid(period: float, offset: float, length: float) -> list[float]:
    """Beat times on the OUTPUT timeline, if the song starts playing at output second 0 (offset = where its first beat falls)."""
    t = offset % period
    out = []
    while t <= length + period:
        out.append(round(t, 3))
        t += period
    return out


def snap_intervals(intervals: list[tuple[float, float]], words: list[dict], beat_times: list[float],
                   max_shift: float = 0.45, min_len: float = 0.8) -> tuple[list[tuple[float, float]], int]:
    """Move each cut (the end of every kept stretch except the last) to the nearest beat by growing or trimming the stretch.
    Never trims into a spoken word and never grows over the start of the next word. Returns (intervals, how many were snapped)."""
    out = [list(iv) for iv in intervals]
    if not beat_times or len(out) < 2:
        return [tuple(iv) for iv in out], 0
    beats = np.array(beat_times)
    shift_total, snapped = 0.0, 0
    for i in range(len(out) - 1):
        a, b = out[i]
        cum = sum(y - x for x, y in out[:i + 1]) + shift_total          # where this cut falls on the output timeline now
        inside = [w for w in words if a <= w["start"] and w["end"] <= b + 0.01]
        last_end = max([w["end"] for w in inside] or [a])
        nxt = [w["start"] for w in words if w["start"] >= b - 0.01]
        next_start = nxt[0] if nxt else float("inf")
        for j in np.argsort(np.abs(beats - cum)):                       # nearest beat first that doesn't hurt the speech
            delta = float(beats[j]) - cum
            if abs(delta) > max_shift:
                break
            if abs(delta) < 0.02:
                break                                                     # already on the beat
            new_b = b + delta
            if delta < 0 and (new_b < last_end + 0.05 or new_b - a < min_len):
                continue                                                  # would trim into a word
            if delta > 0 and new_b > next_start - 0.05:
                continue                                                  # would run into the next word
            out[i][1] = new_b
            shift_total += delta
            snapped += 1
            break
    return [tuple(iv) for iv in out], snapped
