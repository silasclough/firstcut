"""The sound that goes with a full-screen card: a soft whoosh as it arrives and a shorter one as it leaves.

Made from filtered noise in numpy (nothing to download or license): a band-pass sweep that rises as the card comes in and falls as it goes out, shaped by a
smooth envelope, with a low soft thud where a card lands (push, iris). Quiet on purpose: about -22 dBFS peak, so it sits under the voice. One wav the length
of the video holds every whoosh at its moment; it is mixed into the finished sound.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

SR = 48000
PEAK = 0.08                                   # about -22 dBFS before the final mix


def _sweep(n: int, f0: float, f1: float, seed: int) -> np.ndarray:
    """Noise through a band-pass whose centre glides from f0 to f1 Hz (two one-pole low-passes, the second subtracted from the first)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float64)
    fc = np.geomspace(max(40.0, f0), max(41.0, f1), n)
    hi = 1 - np.exp(-2 * np.pi * fc / SR)
    lo = 1 - np.exp(-2 * np.pi * (fc * 0.4) / SR)
    a = b = 0.0
    out = np.empty(n)
    for i in range(n):
        a += hi[i] * (x[i] - a)
        b += lo[i] * (x[i] - b)
        out[i] = a - b
    return out


def whoosh(kind: str, direction: str, dur: float, seed: int = 1) -> np.ndarray:
    """One whoosh of `dur` seconds. kind = the card's transition (wipe, iris, push, zoom); direction = 'in' (rising) or 'out' (falling, shorter and quieter)."""
    n = int(SR * dur)
    lo, hi = {"wipe": (350.0, 5200.0), "iris": (250.0, 3200.0), "push": (180.0, 2600.0), "zoom": (500.0, 6500.0)}.get(kind, (350.0, 4800.0))
    sig = _sweep(n, lo, hi, seed) if direction == "in" else _sweep(n, hi * 0.8, lo * 1.4, seed + 7)
    t = np.linspace(0.0, 1.0, n)
    env = np.sin(np.pi * t ** (0.8 if direction == "in" else 1.25)) ** 2              # swells and fades; the 'in' peaks late (the card lands), the 'out' early
    sig = sig * env
    if kind in ("push", "iris") and direction == "in":                                 # a low thud where it lands
        tt = np.arange(int(SR * 0.22)) / SR
        thud = np.sin(2 * np.pi * (70 - 25 * tt / 0.22) * tt) * np.exp(-tt * 18)
        k = min(n, len(thud)) 
        sig[n - k:] += thud[:k] * 1.6 * max(1e-9, np.abs(sig).max())
    peak = float(np.abs(sig).max()) or 1.0
    return sig / peak * (PEAK if direction == "in" else PEAK * 0.7)


def track(events: list[dict], total: float, path: Path) -> Path | None:
    """events = [{"at", "len", "style"}] (a full-screen card's start, length and transition) -> a stereo wav of `total` seconds with the whooshes in place."""
    if not events or total <= 0:
        return None
    buf = np.zeros(int(SR * total) + SR)

    def put(sig: np.ndarray, start: float):
        i = max(0, int(start * SR))
        j = min(len(buf), i + len(sig))
        if j > i:
            buf[i:j] += sig[:j - i]
    for k, e in enumerate(events):
        style = str(e.get("style") or "wipe")
        at, ln = float(e["at"]), float(e["len"])
        put(whoosh(style, "in", 0.55, seed=11 + k), at - 0.05)                              # the sweep ends as the page lands (the transition takes 0.5 s)
        put(whoosh(style, "out", 0.45, seed=71 + k), at + ln - 0.45)
    buf = np.clip(buf[:int(SR * total)], -0.5, 0.5)
    pcm = (np.stack([buf, buf], axis=1) * 32767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path
