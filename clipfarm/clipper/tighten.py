"""Cut points that land where the person really stops and starts talking. The transcript's word times are only close (a word can
end 0.2 s after its stamp, and the pause around it is a guess); the sound itself is exact. For every kept stretch this looks at the
audio the viewer will hear: a stretch that would end in the middle of a word is let run to the end of it, and dead air at either end
is trimmed to a short natural breath. It also keeps the interviewer out (their spans are cut away, with a little margin)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..core.log import get_logger
from . import audiosync as AS

log = get_logger("clipper.tighten")
FRAME = 0.01                       # the energy curve has one value per 10 ms
QUIET_RUN = 9                      # this many quiet frames in a row (90 ms) = the word is over
BREATH_BEFORE, BREATH_AFTER = 0.10, 0.14


class Meter:
    """Loudness curves (10 ms, in dB) of the audio files, cached next to the project's own work files."""

    def __init__(self, cache_dir: Path | None = None):
        self.dir = cache_dir
        self.mem: dict[str, tuple[np.ndarray, float]] = {}

    def curve(self, path: str) -> tuple[np.ndarray, float]:
        if path in self.mem:
            return self.mem[path]
        f = None
        if self.dir:
            try:
                st = Path(path).stat()
                f = self.dir / ("env_" + hashlib.sha1(f"{path}:{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()[:16] + ".npy")
                if f.exists():
                    e = np.load(f)
                    return self.mem.setdefault(path, (e, _threshold(e)))
            except OSError:
                f = None
        y = AS.decode(path, sr=AS.SR)
        n = len(y) // AS.HOP
        e = (10.0 * np.log10((y[:n * AS.HOP].reshape(n, AS.HOP) ** 2).mean(axis=1) + 1e-10)).astype(np.float32) if n else np.zeros(0, np.float32)
        if f is not None:
            try:
                np.save(f, e)
            except OSError:
                pass
        return self.mem.setdefault(path, (e, _threshold(e)))


def _threshold(e: np.ndarray) -> float:
    """Speech is anything clearly above the room: the quiet 15% of the recording sets the floor."""
    if len(e) < 50:
        return 0.0
    floor = float(np.percentile(e, 15))
    top = float(np.percentile(e, 95))
    return floor + max(6.0, min(14.0, 0.3 * (top - floor)))


def _loud(e: np.ndarray, thr: float, i: int) -> bool:
    lo, hi = max(0, i - 1), min(len(e), i + 2)
    return hi > lo and float(e[lo:hi].max()) > thr


def end_of(e: np.ndarray, thr: float, loc: float) -> float | None:
    """Where the sound that is playing at `loc` really ends (seconds in the file); a stretch that stops mid-word runs on to the word's end, and
    a stretch that stops in silence is pulled back to the last sound before it."""
    i = int(loc / FRAME)
    if i >= len(e):
        return None
    if _loud(e, thr, i):
        j, quiet = i, 0
        while j < min(len(e) - 1, i + 80):
            j += 1
            quiet = 0 if _loud(e, thr, j) else quiet + 1
            if quiet >= QUIET_RUN:
                return (j - quiet + 1) * FRAME
        return None                                                  # it just keeps going: leave the cut where it is
    for j in range(i, max(0, i - 80), -1):
        if _loud(e, thr, j):
            return (j + 1) * FRAME
    return None


def start_of(e: np.ndarray, thr: float, loc: float) -> float | None:
    i = int(loc / FRAME)
    if i >= len(e):
        return None
    if _loud(e, thr, i):
        j = i
        while j > max(0, i - 40) and _loud(e, thr, j - 1):
            j -= 1
        return j * FRAME
    for j in range(i, min(len(e) - 3, i + 70)):
        if all(_loud(e, thr, j + k) for k in range(3)):
            return j * FRAME
    return None


def refine(intervals: list[tuple[float, float]], source_at, meter: Meter, host: list[tuple[float, float]] | None = None,
           bounds: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]:
    """`source_at(t)` -> (audio path, seconds into that file) for master second t, or None. `bounds` (one (lo, hi) per interval): the
    sound may move a cut, but never before lo or after hi - the words said just before and after (a sound that runs straight on into
    the next word must not pull the cut into it)."""
    out: list[tuple[float, float]] = []
    for k, (a, b) in enumerate(intervals):
        lo, hi = bounds[k] if bounds and k < len(bounds) else (-1e9, 1e9)
        na, nb = a, b
        try:
            s = source_at(a + 0.02)
            if s:
                e, thr = meter.curve(s[0])
                r = start_of(e, thr, s[1])
                if r is not None:
                    na = a + (r - s[1]) - BREATH_BEFORE
                    na = min(max(na, a - 0.5), a + 0.7)
                    prev = max((sb for sa, sb in host or [] if sb < a + 0.35), default=None)       # never back into the interviewer's last words
                    if prev is not None and prev > na - 0.04:
                        na = max(a, prev + 0.04)
            s = source_at(max(a, b - 0.02))
            if s:
                e, thr = meter.curve(s[0])
                r = end_of(e, thr, s[1])
                if r is not None:
                    nb = b + (r - s[1]) + BREATH_AFTER
                    nb = min(max(nb, b - 0.7), b + 0.9)
                    nxt = min((sa for sa, sb in host or [] if sa > b - 0.35), default=None)         # nor on into the interviewer's next question
                    if nxt is not None and nb > nxt - 0.04:
                        nb = max(min(b, nxt - 0.04), na + 0.2)
        except Exception as ex:  # noqa: BLE001 - an unreadable file only means this stretch keeps its original cut points
            log.warning(f"couldn't measure the sound near {a:.1f}s: {ex}")
        na, nb = max(na, lo), min(nb, hi)
        if nb - na <= 0.15:                                          # the limits left nothing: the transcript's own cut points, held inside them
            na, nb = max(a, lo), min(b, hi)
        if nb - na > 0.15:
            out.append((round(max(0.0, na), 3), round(nb, 3)))
    merged: list[tuple[float, float]] = []                           # pieces that now touch or overlap are one piece (never play a moment twice)
    for a, b in out:
        if merged and a <= merged[-1][1] + 0.02 and a >= merged[-1][0]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return merged


def without(intervals: list[tuple[float, float]], spans: list[tuple[float, float]], margin: float = 0.18, min_len: float = 0.5) -> list[tuple[float, float]]:
    """The intervals minus the interviewer's spans (each widened by `margin`); what is left shorter than `min_len` is dropped."""
    out = list(intervals)
    for sa, sb in spans:
        sa, sb = sa - margin, sb + margin
        nxt = []
        for a, b in out:
            if b <= sa or a >= sb:
                nxt.append((a, b))
                continue
            if a < sa:
                nxt.append((a, sa))
            if b > sb:
                nxt.append((sb, b))
        out = nxt
    return [(a, b) for a, b in out if b - a >= min_len]


def zero_cross(path, t: float, window_s: float = 0.002, sr: int = 48000) -> float:
    """The time (s) of the zero crossing of the sound in `path` nearest to `t`, within +-window_s (t itself when there is none or the sound can't be read).
    A cut that starts on a zero crossing, under a few-millisecond fade, makes no click."""
    import subprocess

    from ..core import ffmpeg
    t0 = max(0.0, float(t) - window_s)
    try:
        p = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{t0:.6f}", "-t", f"{2 * window_s + 0.001:.6f}", "-i", str(path), "-vn", "-ac", "1",
                            "-ar", str(sr), "-f", "f32le", "-"], capture_output=True, timeout=60)
    except Exception:  # noqa: BLE001 - the cut simply stays where it was planned
        return float(t)
    x = np.frombuffer(p.stdout[: len(p.stdout) // 4 * 4], dtype="<f4")
    if x.size < 3 or not np.isfinite(x).all():
        return float(t)
    centre = (float(t) - t0) * sr
    idx = np.nonzero(np.signbit(x[:-1]) != np.signbit(x[1:]))[0]                    # the sample just before each sign change
    if idx.size == 0:
        return float(t)
    k = int(idx[np.argmin(np.abs(idx + 0.5 - centre))])
    k += int(abs(x[k + 1]) < abs(x[k]))                                             # the nearer of the two samples around the crossing
    return max(0.0, t0 + k / sr)

