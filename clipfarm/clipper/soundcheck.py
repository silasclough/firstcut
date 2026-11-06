"""How good the sound of a stretch of recording is: clear enough to put in the video, or muffled, far away, buried in noise.

Editors hear it at once (the user, 2026-11-05: "it started on some clips with horrible audio quality"); the planner only reads the
words, so it happily opened a video on a camera's own mic from across the room. This measures what an editor hears:
    snr      how far the speech stands above the room (dB). A close mic: 30+; across a hall: under 15.
    bright   how far the speech rises above the room at 1.5-5 kHz (dB), where consonants live: a muffled mic (under clothes, in a
             pocket, far away in a boomy room) loses them first. Measured against the room's own hiss, which a plain "share of the
             high frequencies" mistakes for clarity.
    level    how loud the speech is (dBFS, before any levelling). Very quiet sound lifted up brings its noise with it.
    grade    "good" | "weak" | "bad" (and why, in plain words) from those three.

    measure(path, start, seconds) -> {"snr", "bright", "level", "speech", "grade", "why"}
"""

from __future__ import annotations

import numpy as np

from . import audiosync as AS

SR = 16000
N = 512                                   # 32 ms frames
BANDS = ((80, 300), (300, 1500), (1500, 5000))


def measure(path, start: float = 0.0, seconds: float | None = None) -> dict:
    y = AS.decode(path, start, seconds, sr=SR)
    return judge(y)


def judge(y: np.ndarray) -> dict:
    n = len(y) // N
    if n < 30:
        return {"snr": 0.0, "bright": 0.0, "level": -99.0, "speech": 0.0, "grade": "bad", "why": "no sound"}
    fr = y[:n * N].reshape(n, N).astype(np.float64)
    db = 10.0 * np.log10((fr ** 2).mean(axis=1) + 1e-12)
    floor, top = float(np.percentile(db, 10)), float(np.percentile(db, 90))
    snr = top - floor
    sp = db > floor + max(6.0, 0.35 * snr)                    # the frames with someone talking
    if sp.sum() < 10:
        return {"snr": round(snr, 1), "bright": 0.0, "level": round(top, 1), "speech": 0.0, "grade": "bad", "why": "nobody talking"}
    quiet = db < floor + 3.0
    if quiet.sum() < 5:
        quiet = db <= np.percentile(db, 20)
    hz = np.fft.rfftfreq(N, 1 / SR)
    hi = (hz >= BANDS[2][0]) & (hz < BANDS[2][1])
    win = np.hanning(N)

    def band(rows):                                       # mean high-band energy per frame (in chunks: a long file has many frames)
        tot, k = 0.0, 0
        for i in range(0, len(rows), 4096):
            sp_ = np.abs(np.fft.rfft(rows[i:i + 4096] * win, axis=1)) ** 2
            tot += float(sp_[:, hi].sum())
            k += len(sp_)
        return tot / max(1, k)
    bright = 10.0 * np.log10(band(fr[sp]) / (band(fr[quiet]) + 1e-12) + 1e-12)
    level = float(np.percentile(db[sp], 75))
    why = []
    if snr < 18:
        why.append("far from the mic or noisy")
    if bright < 12:
        why.append("muffled")
    if level < -38:
        why.append("very quiet")
    grade = "bad" if len(why) >= 2 or snr < 12 or bright < 6 else "weak" if why else "good"
    return {"snr": round(snr, 1), "bright": round(float(bright), 1), "level": round(level, 1), "speech": round(float(sp.mean()), 2),
            "grade": grade, "why": ", ".join(why)}


def score(r: dict) -> float:
    """One number to compare recordings by (higher = cleaner): speech over the room, consonants over the hiss, not too quiet."""
    lvl = float(r.get("level", -99.0))
    return round(float(r.get("snr", 0.0)) + 0.5 * min(float(r.get("bright", 0.0)), 35.0) + min(0.0, lvl + 38.0)
                 - {"good": 0.0, "weak": 6.0, "bad": 15.0}.get(r.get("grade", "bad"), 15.0), 2)


def rate_file(path, dur: float, windows: int = 3, seconds: float = 30.0) -> dict:
    """A whole recording, judged from a few windows across it (cheap on a two-hour recorder file): the average score and the worst grade
    of the windows that have someone talking."""
    import math
    dur = float(dur or 0.0)
    if dur <= 0:
        return {"score": 0.0, "grade": "bad"}
    starts = [max(0.0, min(dur - seconds, dur * f)) for f in ((0.5,) if dur <= seconds * 1.5 else [(k + 1) / (windows + 1) for k in range(windows)])]
    rs = [measure(path, s, min(seconds, dur)) for s in starts]
    talk = [r for r in rs if r.get("speech", 0) >= 0.15 and r.get("why") != "nobody talking"] or rs
    sc = sum(score(r) for r in talk) / len(talk)
    order = {"good": 0, "weak": 1, "bad": 2}
    grade = sorted((r["grade"] for r in talk), key=lambda g: order.get(g, 2))[len(talk) // 2]      # the middle one
    return {"score": round(sc if math.isfinite(sc) else 0.0, 2), "grade": grade}


class Grades:
    """Grades of WINDOW-second windows of recordings, cached in a dict (saved by the caller, e.g. the project's sound_grades.json):
    grade(path, local) -> {"grade", "snr", "bright", "level"} for the window around that moment of that file."""
    WINDOW = 10.0

    def __init__(self, data: dict | None = None, measure_fn=None):
        import threading
        self.data = data if data is not None else {}
        self.measure = measure_fn or measure
        self.lock = threading.Lock()
        self.dirty = False

    def grade(self, path: str, local: float) -> dict:
        w = max(0, int(float(local) // self.WINDOW))
        key = f"{path}|{w}"
        with self.lock:
            hit = self.data.get(key)
        if hit is not None:
            return hit
        unread = {"grade": "good", "snr": 0.0, "bright": 0.0, "level": -99.0, "why": "unread"}
        try:
            r = self.measure(path, w * self.WINDOW, self.WINDOW)
            r = {k: (float(r[k]) if k in ("snr", "bright", "level") else r[k]) for k in ("grade", "snr", "bright", "level", "why") if k in r}
        except Exception:  # noqa: BLE001 - unreadable: say nothing about it (never a reason to drop a line)
            return unread
        if r.get("why") == "no sound" and r.get("level", 0) <= -98:          # nothing decoded (unplugged, gone): a read failure, not
                                                                         # silence - say nothing about it, and don't remember it
            return unread
        with self.lock:
            self.data[key] = r
            self.dirty = True
        return r

    def span(self, path: str, a: float, b: float) -> dict:
        """The grade of a stretch: its windows, the most common grade (ties: the worse), the median level."""
        order = {"good": 0, "weak": 1, "bad": 2}
        rs = [self.grade(path, t) for t in _window_points(a, b, self.WINDOW)]
        rs = [r for r in rs if r.get("why") != "unread"] or rs
        counts: dict[str, int] = {}
        for r in rs:
            counts[r["grade"]] = counts.get(r["grade"], 0) + 1
        g = max(counts, key=lambda k: (counts[k], order.get(k, 2)))
        lv = sorted(float(r.get("level", -99.0)) for r in rs)
        return {"grade": g, "level": lv[len(lv) // 2], "why": next((r.get("why", "") for r in rs if r["grade"] == g and r.get("why")), "")}


def _window_points(a: float, b: float, w: float) -> list[float]:
    a, b = float(a), max(float(a), float(b))
    k0, k1 = int(a // w), int(max(a, b - 1e-6) // w)
    return [(k + 0.5) * w if k0 < k < k1 else min(max(a, k * w), b) for k in range(k0, k1 + 1)]
