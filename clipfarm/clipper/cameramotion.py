"""Camera motion of a shot: where the camera makes a smooth, purposeful move (push-in, pan, tilt, slide), as opposed to where it shakes
while the operator settles, or sits still. It feeds the Find footage heat map and "Best part" (catalog.heat).

How: tiny grayscale frames (192 px wide) come out of the catalogue's keyframe pass (a second output of the same ffmpeg call: nothing is
decoded twice), and each pair of neighbouring frames gives ONE global camera movement: feature points are tracked and a similarity
transform (shift, rotation, zoom) is fitted with RANSAC, so people or cars moving through the picture are outliers and do not count.
A push-in shows as a steady zoom, a pan as a steady shift. The keyframes are about half a second apart, so what separates a move from
a shake is not the size of the movement but whether the NEXT movements keep going the same way (sustained direction = a move, random
direction = hands settling). Scored per moment: 0.45 for a camera that sits still (neutral), up to 1 for a smooth move, down to 0 for shake.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

VERSION = 1                       # bumped when the maths changes: older records get their motion again, lazily
FRAME_W = 192                     # width of the grayscale frames the analysis reads
NEUTRAL = 0.45                    # a locked-off camera: not penalised, no bonus
MIN_DT = 0.05
WINDOW = 5                        # samples (about 2.5 s at keyframe spacing) over which "does it keep going the same way" is judged

# columns of the motion array: time of the later frame, centre shift as a fraction of the width (dx, dy), rotation (radians), log zoom,
# how much of the picture agreed (0-1), seconds since the earlier frame
T, DX, DY, ROT, LOGS, CONF, DT = range(7)


# ----------------------------------------------------------------------------------------------------------- getting the frames
def second_output(w: int = FRAME_W) -> list[str]:
    """ffmpeg arguments for one more output (after the stills' own output): tiny gray frames on stdout, one per decoded keyframe,
    their times printed by showinfo on stderr."""
    return ["-an", "-fps_mode", "passthrough", "-vf", f"scale={w}:-2:flags=area,format=gray,showinfo", "-f", "rawvideo", "pipe:1"]


def read_frames(stdout: bytes, stderr: bytes | str) -> tuple[list[float], np.ndarray]:
    """(times, frames [n, h, w]) from `second_output`'s stdout and the stderr of the same ffmpeg run."""
    err = stderr.decode("utf-8", "replace") if isinstance(stderr, (bytes, bytearray)) else str(stderr)
    times = [float(m) for m in re.findall(r"pts_time:\s*(-?[\d.]+)", err)]
    sz = re.search(r"\bs:(\d+)x(\d+)", err)
    if not times or not sz:
        return [], np.zeros((0, 0, 0), np.uint8)
    w, h = int(sz.group(1)), int(sz.group(2))
    n = min(len(times), len(stdout) // (w * h))
    if n < 2:
        return [], np.zeros((0, 0, 0), np.uint8)
    arr = np.frombuffer(stdout[:n * w * h], np.uint8).reshape(n, h, w)
    return times[:n], arr


# --------------------------------------------------------------------------------------------------------------- the analysis
def _pair(a: np.ndarray, b: np.ndarray, cv2) -> tuple[float, float, float, float, float] | None:
    """(dx, dy as a fraction of the width, rotation, log zoom, agreement 0-1) taking a to b, or None when nothing could be matched."""
    h, w = a.shape
    pts = cv2.goodFeaturesToTrack(a, maxCorners=250, qualityLevel=0.01, minDistance=4)
    src = dst = None
    if pts is not None and len(pts) >= 12:
        nxt, ok, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)
        keep = ok.reshape(-1) == 1
        if keep.sum() >= 12:
            src, dst = pts[keep].reshape(-1, 2), nxt[keep].reshape(-1, 2)
    total = len(pts) if pts is not None else 0
    M = inl = None
    if src is not None:
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=400)
    agree = float(inl.sum()) / max(total, 1) if inl is not None else 0.0
    if M is None or agree < 0.3:                                    # a fast move can break the tracker: match features instead
        try:
            orb = cv2.ORB_create(500)
            ka, da = orb.detectAndCompute(a, None)
            kb, db = orb.detectAndCompute(b, None)
            if da is not None and db is not None and len(ka) >= 12 and len(kb) >= 12:
                ms = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db)
                if len(ms) >= 12:
                    s2 = np.float32([ka[m.queryIdx].pt for m in ms])
                    d2 = np.float32([kb[m.trainIdx].pt for m in ms])
                    M2, inl2 = cv2.estimateAffinePartial2D(s2, d2, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=400)
                    if M2 is not None:
                        ag2 = float(inl2.sum()) / len(ms)
                        if ag2 > agree:
                            M, agree = M2, ag2
        except cv2.error:
            pass
    if M is None:
        return None
    A = M[:, :2]
    s = float(np.sqrt(abs(np.linalg.det(A)))) or 1.0
    rot = float(np.arctan2(M[1, 0], M[0, 0]))
    c = np.array([w / 2.0, h / 2.0])
    shift = A @ c + M[:, 2] - c                                      # where the picture's middle went (zoom about the middle: no shift)
    return float(shift[0] / w), float(shift[1] / w), rot, float(np.log(max(s, 1e-3))), min(1.0, agree)


def analyze(times, frames) -> np.ndarray:
    """Motion array [n, 7] (see the column names above; row k = frame k-1 -> frame k; row 0 has no motion), or an empty array."""
    import cv2
    n = len(times)
    out = np.zeros((n, 7), np.float32)
    if n < 3:
        return np.zeros((0, 7), np.float32)
    out[:, T] = times
    for k in range(1, n):
        dt = max(MIN_DT, float(times[k]) - float(times[k - 1]))
        out[k, DT] = dt
        r = _pair(np.ascontiguousarray(frames[k - 1]), np.ascontiguousarray(frames[k]), cv2)
        if r is not None:
            out[k, DX:CONF + 1] = r
    return out


# ---------------------------------------------------------------------------------------------------------------- the scoring
def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(np.pad(x, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def _step(x, lo, hi):
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def profile(m: np.ndarray) -> dict | None:
    """Per sample (the times in m[:, T]): move 0-1 (a smooth, sustained camera movement), shake 0-1 (movement that keeps changing
    direction), cam 0-1 (0.45 still), plus the runs of smooth movement as [(start, end, strength)]. None without enough data."""
    if m is None or len(m) < 4:
        return None
    n = len(m)
    dt = m[:, DT].clip(MIN_DT)
    ok = (m[:, CONF] >= 0.3).astype(np.float64)                      # (a pair that could not be matched counts as nothing: neutral)
    # velocity in "widths per second": shift, shift, zoom and rotation weighted so a slow push-in and a slow pan weigh the same
    vec = np.stack([m[:, DX] / dt, m[:, DY] / dt, 0.6 * m[:, LOGS] / dt, 0.4 * m[:, ROT] / dt], axis=1) * ok[:, None]
    speed = np.linalg.norm(vec, axis=1)
    half = WINDOW // 2
    cons = np.zeros(n)
    mean_sp = np.zeros(n)
    have = np.zeros(n)
    for k in range(n):
        a, b = max(1, k - half), min(n, k + half + 1)
        w = dt[a:b] * ok[a:b]
        if b - a < 2 or w.sum() <= 0:
            continue
        tot = np.linalg.norm((vec[a:b] * w[:, None]).sum(axis=0))
        path = (speed[a:b] * w).sum()
        cons[k] = tot / path if path > 1e-9 else 0.0
        mean_sp[k] = path / w.sum()
        have[k] = 1.0
    strength = _step(mean_sp, 0.008, 0.028)                          # enough movement to be a move at all
    move = strength * _step(cons, 0.55, 0.85) * have
    shake = _step(mean_sp, 0.006, 0.026) * (1.0 - _step(cons, 0.35, 0.7)) * have
    move, shake = _smooth(move), _smooth(shake)
    cam = np.clip(NEUTRAL + 0.55 * move - 0.45 * shake, 0.0, 1.0)
    t = m[:, T].astype(np.float64)
    runs, a = [], None                                               # the stretches of smooth movement
    for k in range(n):
        if move[k] >= 0.5 and a is None:
            a = k
        if (move[k] < 0.5 or k == n - 1) and a is not None:
            b = k if move[k] < 0.5 else k
            while a < b and speed[a] < 0.008:                        # (the window spreads a run a little early: begin where the camera really starts)
                a += 1
            while b > a and speed[b] < 0.008:
                b -= 1
            if b > a:
                runs.append((float(t[a] - dt[a]), float(t[b]), float(move[a:b + 1].sum() * np.mean(dt[a:b + 1]))))
            a = None
    return {"t": t, "move": move, "shake": shake, "cam": cam, "speed": mean_sp, "cons": cons, "runs": runs}


def best_move(prof: dict | None, min_s: float = 1.2) -> tuple[float, float] | None:
    """(start, end) in seconds of the strongest smooth movement (most movement x time), or None if the shot has none."""
    if not prof:
        return None
    runs = [r for r in prof["runs"] if r[1] - r[0] >= min_s]
    if not runs:
        return None
    a, b, _ = max(runs, key=lambda r: r[2])
    return a, b


# ------------------------------------------------------------------------------------------------------- files (older records)
def save(path: Path, m: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, m.astype(np.float32))
    tmp.replace(path)


def load(path: Path) -> np.ndarray | None:
    try:
        a = np.load(path)
        return a if a.ndim == 2 and a.shape[1] >= 7 else None            # (a truncated or odd file is no motion data)
    except (OSError, ValueError, EOFError, IndexError):
        return None


def extract(ffmpeg_bin: str, video: str | Path, prefix: list[str] | None = None, timeout: float = 900) -> np.ndarray:
    """For a file catalogued before motion was kept: one keyframes-only pass that writes the tiny frames only (the same decode the
    catalogue did, so it costs about what indexing the file cost). `prefix` is the nice/background wrapper."""
    r = subprocess.run([*(prefix or []), ffmpeg_bin, "-v", "info", "-nostats", "-skip_frame", "nokey", "-i", str(video), *second_output()],
                       capture_output=True, timeout=timeout)
    times, frames = read_frames(r.stdout, r.stderr)
    return analyze(times, frames) if len(times) else np.zeros((0, 7), np.float32)
