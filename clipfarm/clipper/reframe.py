"""Vertical reframing: follow the main face with a 9:16 crop, cutting like a camera operator.

Frames are sampled at `rate` fps (downscaled) and run through OpenCV's YuNet face detector.
The crop centre only moves when the subject leaves a dead zone, and never more often than
`min_shot_s` - so the result looks like deliberate cuts between speakers, not a jittery pan.
Without faces (gameplay, slides, b-roll) the clip is fitted over a blurred copy of itself.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.reframe")
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")


@dataclass
class Face:
    t: float          # seconds from the start of the analysed span
    cx: float         # centre x, 0..1 of the source width
    cy: float
    w: float          # width, 0..1 of the source width
    score: float
    act: float = 0.0  # how much the mouth moved since the previous sample (lip movement = this face is talking)
    track: int = -1   # which person this is (faces are followed from sample to sample)


def yunet_path(models_dir: Path) -> Path | None:
    p = Path(models_dir) / "face_detection_yunet_2023mar.onnx"
    if p.exists():
        return p
    try:
        import requests
        r = requests.get(YUNET_URL, timeout=60)
        r.raise_for_status()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        return p
    except Exception as e:  # noqa: BLE001 - reframing degrades to the blurred fit
        log.warning(f"face model unavailable ({e}); clips will use the blurred-fit layout")
        return None


def mouth_motion(prev: np.ndarray | None, cur: np.ndarray, box: tuple[float, float, float, float]) -> float:
    """Lip movement of one face between two frames: change in the mouth area, minus the change in the eyes / forehead
    (head and camera movement shift both; talking moves only the mouth). Frames are grayscale float arrays."""
    if prev is None:
        return 0.0
    x, y, w, h = box
    H, W = cur.shape

    def patch(a, fx0, fx1, fy0, fy1):
        x0, x1 = int(max(0, x + w * fx0)), int(min(W, x + w * fx1))
        y0, y1 = int(max(0, y + h * fy0)), int(min(H, y + h * fy1))
        return a[y0:y1, x0:x1] if x1 - x0 > 3 and y1 - y0 > 3 else None
    m1, m0 = patch(cur, 0.25, 0.75, 0.62, 0.95), patch(prev, 0.25, 0.75, 0.62, 0.95)
    u1, u0 = patch(cur, 0.15, 0.85, 0.10, 0.45), patch(prev, 0.15, 0.85, 0.10, 0.45)
    if m1 is None or m0 is None or m1.shape != m0.shape:
        return 0.0
    mouth = float(np.abs(m1 - m0).mean())
    upper = float(np.abs(u1 - u0).mean()) if (u1 is not None and u0 is not None and u1.shape == u0.shape) else 0.0
    return max(0.0, mouth - 1.0 * upper)


def detect_faces(src: Path, start: float, dur: float, models_dir: Path, rate: float = 2.0,
                 width: int = 640, mouths: bool = False, min_w: float = 0.03) -> tuple[list[list[Face]], int]:
    """Returns (faces per sample, number of samples). With mouths=True every face also gets `act` (lip movement) and a
    `track` id, which is what lets the framing follow whoever is TALKING (use rate >= 5 for that)."""
    try:
        import cv2
    except ImportError:
        return [], 0
    model = yunet_path(models_dir)
    if model is None:
        return [], 0
    info = ffmpeg.probe(src)
    vs = ffmpeg.video_stream(info) or {}
    from .glasses import _display_dims
    sw, sh = _display_dims(vs)
    sw, sh = sw or 1920, sh or 1080
    w = min(width, sw) // 2 * 2
    h = int(round(sh * w / sw)) // 2 * 2
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src),
           "-vf", f"fps={rate},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=900).stdout
    n = len(raw) // (w * h * 3)
    det = cv2.FaceDetectorYN.create(str(model), "", (w, h), 0.7, 0.3, 50)
    out: list[list[Face]] = []
    prev_gray = None
    tracks: dict[int, tuple[float, int]] = {}          # track id -> (last cx, last sample it was seen in)
    next_track = 0
    for k in range(n):
        frame = np.frombuffer(raw[k * w * h * 3:(k + 1) * w * h * 3], dtype=np.uint8).reshape(h, w, 3)
        _, faces = det.detect(frame)
        gray = frame.astype(np.float32).mean(axis=2) if mouths else None
        row = []
        for f in (faces if faces is not None else []):
            x, y, fw, fh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            if fw / w < min_w:                    # ignore tiny background faces (graphics ask for every face: a group of students in a wide shot)
                continue
            face = Face(k / rate, (x + fw / 2) / w, (y + fh / 2) / h, fw / w, score)
            if mouths:
                face.act = mouth_motion(prev_gray, gray, (x, y, fw, fh))
            row.append(face)
        if mouths:
            used: set[int] = set()
            for face in sorted(row, key=lambda f: -f.w):
                # the same person if close to where someone was a moment ago (a face that blinks out for ~1 s keeps its id)
                cand = [(abs(cx - face.cx), tid) for tid, (cx, seen) in tracks.items()
                        if tid not in used and abs(cx - face.cx) < 0.15 and k - seen <= int(rate)]
                if cand:
                    face.track = min(cand)[1]
                else:
                    face.track, next_track = next_track, next_track + 1
                used.add(face.track)
                tracks[face.track] = (face.cx, k)
            prev_gray = gray
        out.append(row)
    return out, n


def crop_path(samples: list[list[Face]], rate: float, crop_frac: float, dead_zone: float = 0.10,
              min_shot_s: float = 1.6) -> list[tuple[float, float]]:
    """Piecewise-constant crop centres [(t_from, cx 0..1)] following the biggest face, with hysteresis."""
    half = crop_frac / 2
    cur = None
    path: list[tuple[float, float]] = []
    last_cut = -1e9
    for k, faces in enumerate(samples):
        t = k / rate
        if not faces:
            continue
        # prefer the face closest to the current framing if it's still a big one; else the biggest
        biggest = max(faces, key=lambda f: f.w * f.score)
        target = biggest
        if cur is not None:
            near = min(faces, key=lambda f: abs(f.cx - cur))
            if abs(near.cx - cur) < dead_zone and near.w >= 0.6 * biggest.w:
                target = near
        want = min(1 - half, max(half, target.cx))
        if cur is None:
            cur = want
            path.append((0.0, cur))
            last_cut = t
        elif abs(want - cur) > dead_zone and t - last_cut >= min_shot_s:
            cur = want
            path.append((t, cur))
            last_cut = t
    return path


def group_path(samples: list[list[Face]], rate: float, crop_frac: float, dead_zone: float = 0.08,
               min_shot_s: float = 1.6) -> list[tuple[float, float]]:
    """Like crop_path, but for a WIDE window: it is centred on the whole group of faces in view (not just the biggest),
    so the person talking and the person listening are both in frame; it moves only when the group moves."""
    half = crop_frac / 2
    cur = None
    path: list[tuple[float, float]] = []
    last_cut = -1e9
    for k, faces in enumerate(samples):
        t = k / rate
        if not faces:
            continue
        tw = sum(f.w for f in faces)
        want = min(1 - half, max(half, sum(f.cx * f.w for f in faces) / tw))
        if cur is None:
            cur = want
            path.append((0.0, cur))
            last_cut = t
        elif abs(want - cur) > dead_zone and t - last_cut >= min_shot_s:
            cur = want
            path.append((t, cur))
            last_cut = t
    return path


def too_wide_spans(samples: list[list[Face]], rate: float, crop_frac: float, min_s: float = 1.0) -> list[tuple[float, float]]:
    """Stretches where the faces are spread wider than the window (people far apart): show the whole frame there."""
    flags = []
    for faces in samples:
        if len(faces) < 2:
            flags.append(False)
            continue
        lo = min(f.cx - f.w / 2 for f in faces)
        hi = max(f.cx + f.w / 2 for f in faces)
        flags.append(hi - lo > crop_frac * 0.92)
    spans, run = [], None
    for k, fl in enumerate(flags + [False]):
        if fl and run is None:
            run = k
        elif not fl and run is not None:
            if (k - run) / rate >= min_s:
                spans.append((run / rate, k / rate))
            run = None
    return spans


def typical_face_width(samples: list[list[Face]]) -> float:
    """Median (over samples with faces) of the biggest face's width as a fraction of the frame - small = a facecam
    in the corner of gameplay / a screen share, large = people on camera."""
    ws = sorted(max(f.w for f in s) for s in samples if s)
    return ws[len(ws) // 2] if ws else 0.0


def audio_envelope(src: Path, start: float, dur: float, rate: float) -> np.ndarray | None:
    """Loudness of the speech over time, one value per face sample (same rate). Lip movement only counts as 'talking' when
    it follows this - a listener nodding or a shaky camera moves the mouth area too, but not in time with the voice."""
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-vn",
           "-ac", "1", "-ar", "8000", "-f", "s16le", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    except Exception:  # noqa: BLE001
        return None
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if a.size < 8000:
        return None
    hop = int(8000 / rate)
    n = a.size // hop
    env = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    return env if env.std() > 1e-3 else None


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 6 or a.std() < 1e-6 or b.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def speaker_path(samples: list[list[Face]], rate: float, crop_frac: float, group_pull: float = 0.3,
                 dead_zone: float = 0.08, min_shot_s: float = 3.0, smooth_s: float = 1.4,
                 env: np.ndarray | None = None) -> list[tuple[float, float]] | None:
    """Window centres that follow whoever is talking. Per person, lip movement over a ~2.5 s window is compared with the
    loudness of the speech (`env`): the person whose mouth moves in time with the voice is the speaker (without audio:
    the one whose mouth moves most). The last speaker stays until someone else clearly takes over; cuts are at least
    `min_shot_s` apart. The window sits on the speaker, pulled a little toward the rest of the group so a nearby listener
    stays in frame. None if no one's lips moved (no usable signal) - the caller then falls back to the group framing."""
    n = len(samples)
    if n == 0:
        return None
    win = max(1, int(round(smooth_s * rate)))
    acts: dict[int, list[float]] = {}
    for k, faces in enumerate(samples):
        for f in faces:
            acts.setdefault(f.track, [0.0] * n)[k] = f.act
    if not acts or max((max(v) for v in acts.values()), default=0.0) <= 0.0:
        return None
    longw = max(6, int(round(2.5 * rate)))
    use_env = env is not None and len(env) >= n
    if use_env:
        e = np.asarray(env[:n], dtype=np.float32)
        sm = {}
        for tid, v in acts.items():
            arr = np.asarray(v, dtype=np.float32)
            sm[tid] = [(_corr(arr[max(0, k - longw + 1):k + 1], e[max(0, k - longw + 1):k + 1]) if arr[max(0, k - longw + 1):k + 1].any() else 0.0)
                       for k in range(n)]
        margin = 0.12
    else:
        sm = {tid: [sum(v[max(0, k - win + 1):k + 1]) / win for k in range(n)] for tid, v in acts.items()}
        margin = None
    half = crop_frac / 2
    speaker, cur = None, None
    path: list[tuple[float, float]] = []
    last_cut = -1e9
    for k, faces in enumerate(samples):
        if not faces:
            continue
        t = k / rate
        here = {f.track: f for f in faces}
        best = max(here, key=lambda tid: sm.get(tid, [0.0] * n)[k])
        if speaker is None or speaker not in here:
            speaker = best
        elif best != speaker and (sm[best][k] > sm.get(speaker, [0.0] * n)[k] + margin if margin is not None
                                  else sm[best][k] > 1.35 * sm.get(speaker, [0.0] * n)[k] + 0.05):
            speaker = best                                        # someone else clearly took over
        sp = here[speaker]
        others = [f for f in faces if f.track != speaker]
        group = (sum(f.cx * f.w for f in others) / sum(f.w for f in others)) if others else sp.cx
        want = min(1 - half, max(half, (1 - group_pull) * sp.cx + group_pull * group))
        if cur is None:
            cur = want
            path.append((0.0, cur))
            last_cut = t
        elif abs(want - cur) > dead_zone and t - last_cut >= min_shot_s:
            cur = want
            path.append((t, cur))
            last_cut = t
    if path and len(path) > 1 + (n / rate) / 5.0:
        return None          # it would cut between people more than every ~5 s: that's noise, not turns - keep both in frame
    return path or None


def face_coverage(samples: list[list[Face]]) -> float:
    return sum(1 for s in samples if s) / len(samples) if samples else 0.0


def x_expression(path: list[tuple[float, float]], src_w: int, crop_w: int, remap) -> str:
    """ffmpeg crop x expression over output time t (wrap it in single quotes inside a filtergraph).
    `remap` maps source-span seconds -> output seconds."""
    if not path:
        return str((src_w - crop_w) // 2)
    xs = [(remap(t), int(round(min(src_w - crop_w, max(0, cx * src_w - crop_w / 2))))) for t, cx in path]
    xs = [p for p in xs if p[0] is not None]
    if not xs:
        return str((src_w - crop_w) // 2)
    ease = min(0.27, 8 / 30.0)                                       # a small move (under 10 % of the frame) glides over 8 frames; a big one is a cut and stays a jump
    n = len(xs)
    expr = str(xs[-1][1])
    for i in range(n - 1, 0, -1):                                    # from the last keyframe back: expr is what is valid from keyframe i on
        t_i, x_i = xs[i]
        x_prev = xs[i - 1][1]
        nxt = xs[i + 1][0] if i + 1 < n else None
        dd = ease if 0 < abs(x_i - x_prev) < 0.10 * src_w else 0.0
        if nxt is not None:
            dd = min(dd, max(0.0, nxt - t_i))
        tail = expr                                                  # (already "x_i until the next keyframe, then ...")
        if dd > 0.02:
            u = f"clip((t-{t_i:.3f})/{dd:.3f},0,1)"
            ramp = f"{x_prev}+({x_i}-{x_prev})*{u}*{u}*(3-2*{u})"
            expr = f"if(lt(t,{t_i:.3f}),{x_prev},if(lt(t,{t_i + dd:.3f}),{ramp},{tail}))"
        else:
            expr = f"if(lt(t,{t_i:.3f}),{x_prev},{tail})"
    return expr


def faceless_spans(samples: list[list[Face]], rate: float, min_s: float = 1.0) -> list[tuple[float, float]]:
    """Runs of samples with no face (slides, b-roll, title cards) lasting >= min_s, in span seconds."""
    spans, run = [], None
    for k, faces in enumerate(samples + [[Face(0, 0, 0, 0, 0)]]):      # sentinel closes a trailing run
        if not faces and run is None:
            run = k
        elif faces and run is not None:
            if (k - run) / rate >= min_s:
                spans.append((run / rate, k / rate))
            run = None
    return spans


def action_path(src: Path, start: float, dur: float, crop_frac: float, rate: float = 3.0, width: int = 160, dead_zone: float = 0.10,
                min_shot_s: float = 2.0) -> dict | None:
    """Where the ACTION is in footage without faces (cooking from above, hands, a craft, a screen): the centre of what moves, over time.
    -> {"path": [(t, cx 0..1)], "active": share of samples with motion, "spread": typical horizontal spread of the motion (0..1), "concentrated": bool}
    or None when nothing moves / the frames cannot be read. A clip is "concentrated" when the motion stays in a part of the frame a crop of
    `crop_frac` of the width can hold, which is when cropping in beats showing the whole frame small over a blurred copy of itself."""
    h = 90
    cmd = [ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}", "-i", str(src), "-vf", f"fps={rate},scale={width}:{h},format=gray",
           "-f", "rawvideo", "-"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=600).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None
    n = len(raw) // (width * h)
    if n < 4:
        return None
    frames = np.frombuffer(raw[: n * width * h], dtype=np.uint8).reshape(n, h, width).astype(np.int16)
    cols = np.arange(width) / (width - 1)
    cxs: list[float | None] = []
    spreads: list[float] = []
    for k in range(1, n):
        m = (np.abs(frames[k] - frames[k - 1]) > 18)
        if m.sum() < 0.002 * m.size:                              # nothing moved
            cxs.append(None)
            continue
        wcol = m.sum(axis=0).astype(float)
        c = float((wcol * cols).sum() / wcol.sum())
        cxs.append(c)
        spreads.append(float(np.sqrt((wcol * (cols - c) ** 2).sum() / wcol.sum())))
    active = sum(1 for c in cxs if c is not None) / len(cxs)
    if active < 0.4 or not spreads:
        return None
    half = crop_frac / 2
    cur, last_cut, path = None, -1e9, []
    for k, c in enumerate(cxs):
        if c is None:
            continue
        t = (k + 1) / rate
        want = min(1 - half, max(half, c))
        if cur is None:
            cur, last_cut = want, t
            path.append((0.0, cur))
        elif abs(want - cur) > dead_zone and t - last_cut >= min_shot_s:
            cur, last_cut = want, t
            path.append((t, cur))
    spread = float(np.median(spreads))
    return {"path": path, "active": round(active, 2), "spread": round(spread, 3), "concentrated": spread * 2.2 <= crop_frac}


def facecam_box(samples: list[list[Face]], sw: int, sh: int, aspect: float, spread: float = 3.4) -> dict | None:
    """Where the streamer's face-cam is (gameplay with the cam in a corner): the median position of the biggest face, widened to a crop of `spread` face-widths with the
    proportions of the panel it will fill (aspect = width / height). -> {"x","y","w","h"} in source pixels (even numbers) or None when no face is seen often enough."""
    big = [max(s_, key=lambda f: f.w * f.score) for s_ in samples if s_]
    if len(big) < max(3, len(samples) // 4):
        return None
    med = lambda v: sorted(v)[len(v) // 2]  # noqa: E731
    cx, cy, fw = med([f.cx for f in big]), med([f.cy for f in big]), med([f.w for f in big])
    w = min(float(sw), max(0.10 * sw, spread * fw * sw))
    h = w / aspect
    if h > sh:
        h, w = float(sh), sh * aspect
    x = min(sw - w, max(0.0, cx * sw - w / 2))
    y = min(sh - h, max(0.0, cy * sh - h * 0.46))             # the face a little above the middle: room for the shoulders
    ev = lambda v: int(v) // 2 * 2  # noqa: E731
    return {"x": ev(x), "y": ev(y), "w": ev(w), "h": ev(h), "cx": cx}
