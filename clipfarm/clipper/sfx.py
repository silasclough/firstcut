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


VARIANTS = ("wipe", "iris", "push", "zoom")                 # the whoosh sounds; the same one is never used for two cards in a row
WINDOW_S = 30.0


def choose(events: list[dict], policy: dict | None = None) -> list[dict]:
    """The cards that get a sound, from the style's `sfx` policy: at most `max_per_30s` in any 30 seconds (0 = none), never the same whoosh twice in a row,
    and at least one silent full-screen card in any 30 seconds that has two or more. The rest stay silent. A card with no `style` is a wipe."""
    pol = policy if isinstance(policy, dict) else {}
    try:
        mx = int(pol.get("max_per_30s", 2))
    except (TypeError, ValueError):
        mx = 2
    if mx <= 0:
        return []
    out: list[dict] = []
    seen: list[tuple[float, bool]] = []                                      # (time, sounded) of the cards so far
    last = None
    for e in sorted((e for e in events if isinstance(e, dict) and "at" in e), key=lambda e: float(e["at"])):
        try:
            at = float(e["at"])
        except (TypeError, ValueError):
            continue
        recent = [(t, ok) for t, ok in seen if at - t < WINDOW_S]
        sounded = sum(1 for _t, ok in recent if ok)
        give = sounded < mx and not (recent and all(ok for _t, ok in recent))      # (a window never has every card sounded)
        seen.append((at, give))
        if not give:
            continue
        style = str(e.get("style") or "wipe")
        if style == last:                                                    # never the same sound twice in a row
            style = VARIANTS[(VARIANTS.index(style) + 1) % len(VARIANTS)] if style in VARIANTS else VARIANTS[0]
        last = style
        out.append({**e, "style": style})
    return out


def track(events: list[dict], total: float, path: Path, policy: dict | None = None) -> Path | None:
    """events = [{"at", "len", "style"}] (a full-screen card's start, length and transition) -> a stereo wav of `total` seconds with the whooshes in place
    (only for the cards the style's sfx policy allows: see choose())."""
    events = choose(events, policy) if policy is not None else events
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
