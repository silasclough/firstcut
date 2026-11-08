"""Hiding jump cuts (WP7): which device covers each cut, where a punch-in is centred, when something must change.

A jump cut is a cut inside one continuous shot (a pause or a slip taken out while the person stays put): the picture twitches. Editors hide it
with one of three devices - a punch-in (the picture scaled up a little, framed on the face), a switch to another camera angle that shows the
same speaker, or a B-roll shot that matches what is being said. The devices take turns (never the same one twice in a row) so the video does not
look like it has one trick. When none fits, the jump cut STAYS and the report says so: an unrelated picture over a serious passage is worse
than a visible cut. Pure functions: the plan decides here, the render does the pixels.
"""

from __future__ import annotations

import math

DEVICES = ("punch", "angle", "broll")
EYE_LINE = (0.33, 0.40)                 # where the eyes belong in the frame, top to bottom (the vertical framing guidance for faces)
EYE_ABOVE_CENTRE = 0.15                 # the face detector's box centre is about the nose: the eyes are this share of the box height above it
PUNCH_LANDSCAPE, PUNCH_PORTRAIT = 1.12, 1.10      # (proposal; the practice range is 8-18 %)
MOVE_SHARE = 0.15                       # a face that moves more than this share of the frame width in a second: the speaker walked or the camera re-aimed


def punch_size(style_value, portrait: bool = False) -> float:
    """The punch-in scale from the style (pace.punch_in), else the default for the frame's shape; always inside 1.05-1.25."""
    try:
        z = float(style_value)
    except (TypeError, ValueError):
        z = PUNCH_PORTRAIT if portrait else PUNCH_LANDSCAPE
    if not math.isfinite(z):
        z = PUNCH_PORTRAIT if portrait else PUNCH_LANDSCAPE
    return round(min(1.25, max(1.05, z)), 3)


def max_hold(style_value, default: float = 12.0) -> float:
    """pace.max_hold_s from the style (short_talk 5, long form 12-15, seminar 30); never absurd."""
    try:
        v = float(style_value)
    except (TypeError, ValueError):
        return default
    return min(60.0, max(2.0, v)) if math.isfinite(v) else default


# ---------------------------------------------------------------------------------------------------------------------- faces
def face_in(series: dict | None, t0: float, t1: float) -> dict | None:
    """The median face {cx, cy, h} (frame fractions) of a stretch of one file, from the facts' `series` (face_cx/face_cy/face_h per sample), or None
    when no face was measured there (an older cache has no centres: None, and the caller frames on the middle)."""
    if not series:
        return None
    ts, hs, xs, ys = (series.get(k) or [] for k in ("t", "face_h", "face_cx", "face_cy"))
    n = min(len(ts), len(hs), len(xs), len(ys))
    num = lambda v: isinstance(v, (int, float)) and v == v                    # noqa: E731 (a missing sample is None; NaN is not a number here)
    rows = [(xs[i], ys[i], hs[i]) for i in range(n) if num(ts[i]) and num(hs[i]) and num(xs[i]) and num(ys[i]) and t0 - 0.5 <= ts[i] <= t1 + 0.5 and hs[i] > 0.02]
    if not rows:
        return None
    mid = lambda k: sorted(r[k] for r in rows)[len(rows) // 2]          # noqa: E731
    return {"cx": float(mid(0)), "cy": float(mid(1)), "h": float(mid(2))}


def punch_centre(face: dict | None, zoom: float) -> tuple[float, float]:
    """Where the punched-in picture is centred (frame fractions): on the face, clamped so the crop stays inside the frame, and (when it can) with the
    eye line at 33-40 % of the crop's height. No face -> the middle."""
    z = max(1.0, float(zoom))
    half = 0.5 / z                                                      # half of the crop, as a share of the frame
    if not face:
        return 0.5, 0.5
    cx = min(1.0 - half, max(half, float(face["cx"])))
    eye = float(face["cy"]) - EYE_ABOVE_CENTRE * float(face.get("h") or 0.0)
    target = sum(EYE_LINE) / 2                                          # the middle of the 33-40 % band
    top = eye - target / z                                              # the crop's top edge such that the eye sits at `target` of the crop height
    cy = top + half
    cy = min(1.0 - half, max(half, cy))
    return round(cx, 4), round(cy, 4)


def eye_line_of(face: dict | None, zoom: float, centre: tuple[float, float]) -> float | None:
    """Where the eyes land in the punched-in frame (0 top - 1 bottom), or None without a face."""
    if not face:
        return None
    z = max(1.0, float(zoom))
    eye = float(face["cy"]) - EYE_ABOVE_CENTRE * float(face.get("h") or 0.0)
    return round((eye - (centre[1] - 0.5 / z)) * z, 3)


def crop_offsets(W: int, H: int, zoom: float, centre: tuple[float, float]) -> tuple[int, int]:
    """Pixel offset (x, y) of the W x H crop inside the picture scaled to W*zoom x H*zoom, centred on `centre`; always inside the picture and even."""
    sw, sh = int(W * zoom) // 2 * 2, int(H * zoom) // 2 * 2
    x = int(round(centre[0] * sw - W / 2))
    y = int(round(centre[1] * sh - H / 2))
    x = max(0, min(sw - W, x)) // 2 * 2
    y = max(0, min(sh - H, y)) // 2 * 2
    return x, y


def zoom_filter(W: int, H: int, zoom: float, centre: tuple[float, float] | None = None) -> str:
    """The ffmpeg chain for a punch-in: scale up, crop W x H around `centre` (the middle when None)."""
    sw, sh = int(W * zoom) // 2 * 2, int(H * zoom) // 2 * 2
    if not centre:
        return f"scale={sw}:{sh},crop={W}:{H}"
    x, y = crop_offsets(W, H, zoom, centre)
    return f"scale={sw}:{sh},crop={W}:{H}:{x}:{y}"


def movement_points(series: dict | None, t0: float, t1: float, share: float = MOVE_SHARE) -> list[float]:
    """Seconds (in the file) inside [t0, t1] where the face moved by more than `share` of the width within a second: a good moment to cut or
    cover ("cut on movement"). Needs the facts' face_cx samples."""
    if not series:
        return []
    ts, xs = series.get("t") or [], series.get("face_cx") or []
    n = min(len(ts), len(xs))
    out: list[float] = []
    for i in range(1, n):
        if not (t0 <= ts[i] <= t1) or ts[i] - ts[i - 1] <= 0 or ts[i] - ts[i - 1] > 1.6:
            continue
        if abs(xs[i] - xs[i - 1]) / (ts[i] - ts[i - 1]) > share:
            out.append(round(float(ts[i]), 2))
    return out


# ---------------------------------------------------------------------------------------------------------------------- the devices
def choose_device(last: str | None, can: dict[str, bool], prefer: tuple[str, ...] = DEVICES) -> str | None:
    """The next device for a jump cut: the first of `prefer` that is possible and is not the one used last (a different device each time); when only
    the last one is possible it may repeat rather than leave the cut bare - except broll, which never repeats back to back."""
    for d in prefer:
        if can.get(d) and d != last:
            return d
    if last and can.get(last) and last != "broll":
        return last
    return None


def plan_covers(jumps: list[dict], *, last: str | None = None) -> tuple[list[dict], list[str]]:
    """jumps: [{"t": program second, "can": {"punch": bool, "angle": bool, "broll": bool}, "prefer"?: [...]}]  -> (decisions, notes).
    A decision is {"t", "device"}; a jump no device fits is left alone and noted ("nothing fits at 41.2 s: the cut stays")."""
    out: list[dict] = []
    notes: list[str] = []
    for j in jumps:
        pref = tuple(j.get("prefer") or DEVICES)
        d = choose_device(last, j.get("can") or {}, pref)
        if d is None:
            notes.append(f"a jump cut at {float(j['t']):.1f} s was left as it is: nothing fitted over it, so it stays on the speaker")
            continue
        out.append({"t": float(j["t"]), "device": d})
        last = d
    return out, notes


def blocked(t0: float, t1: float, spans: list[tuple[float, float]]) -> bool:
    """True if [t0, t1] overlaps any protected or excluded span (program seconds): no B-roll there."""
    return any(a < t1 and b > t0 for a, b in spans or [])
