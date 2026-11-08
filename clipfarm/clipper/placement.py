"""Where a graphic or a caption may go: typed, timed zones, platform safe boxes, a cheap saliency term, fixed anchor slots scored per shot.

Learned from rendered frames (a card over a speaker's mouth, a caption hidden under a full-screen card, a lower third at the top of the frame because
the caption zone had taken its spot): the picture decides. Pure code (numpy only): nothing here draws, so it is cheap to test with synthetic frames.

    zone       {"kind": face | mouth_eyes | body | caption | text | ui | title, "box": (x0, y0, x1, y1) in pixels, "t0", "t1", "weight"}
    safe_box   the part of the frame the platforms' own buttons and the TV title-safe area leave free
    slots      the fixed places a kind of graphic may take, in priority order
    choose     the best slot for a graphic over a time window: per shot, with hysteresis, shortened before dropped
    caption_band   where the captions sit in each shot (moved to the top of the safe box when a face is low in the frame)
    contrast_*  WCAG-style contrast under a rectangle and the plate it needs
"""

from __future__ import annotations

from typing import Callable

import numpy as np

# The weights of the intrusiveness map (proposal; the structure is text + face + saliency)
W_MOUTH, W_FACE, W_BODY, W_CAPTION, W_TEXT, W_UI, W_SALIENCY = 1.0, 0.7, 0.4, 1.0, 0.8, 0.9, 0.3
HARD = ("mouth_eyes", "caption", "subject")          # a card never goes over these ("subject": the blob that moves, where no face could be measured)
COST_LIMIT = 0.6                                      # above this a slot is "too busy": try a smaller card, a shorter one, then drop it
HYSTERESIS = 0.25                                     # another slot must be a quarter better to take the card away from the last one
NEAR = 0.05                                           # weight of the distance from the speaker (near is good, on top is not)
MOVE_PENALTY = 0.25
CONTRAST_MIN = 4.5                                    # WCAG AA for text


# ------------------------------------------------------------------------------------------------------------------ safe boxes
def safe_box(W: int, H: int, target: str = "any") -> dict:
    """The area free of platform UI. 9:16: margins top 290, bottom 672, left 120, right 192 at 1080x1920 (the union of Meta's, TikTok's and Shorts'
    figures), scaled to the frame. 16:9 (and any landscape frame): SMPTE title-safe, 5 % each side (96 / 54 px at 1080p). `target` "shorts" lets the
    bottom shrink to 440 px (proposal: there is no official organic Shorts figure)."""
    W, H = int(W), int(H)
    if H > W:
        bottom = 440 if target == "shorts" else 672
        return {"x0": round(120 * W / 1080), "y0": round(290 * H / 1920), "x1": round(W - 192 * W / 1080), "y1": round(H - bottom * H / 1920)}
    return {"x0": round(W * 0.05), "y0": round(H * 0.05), "x1": round(W * 0.95), "y1": round(H * 0.95)}


def inside(rect: tuple, box: dict, slack: float = 0.0) -> bool:
    x, y, w, h = rect
    return x >= box["x0"] - slack and y >= box["y0"] - slack and x + w <= box["x1"] + slack and y + h <= box["y1"] + slack


# ------------------------------------------------------------------------------------------------------------------ zones
def zone(kind: str, box: tuple, t0: float = 0.0, t1: float = 1e9, weight: float | None = None) -> dict:
    w = {"mouth_eyes": W_MOUTH, "face": W_FACE, "subject": W_FACE, "body": W_BODY, "caption": W_CAPTION, "text": W_TEXT, "ui": W_UI, "title": W_CAPTION}.get(kind, 0.5)
    return {"kind": kind, "box": tuple(float(v) for v in box), "t0": float(t0), "t1": float(t1), "weight": float(w if weight is None else weight)}


def zones_from_face(cx: float, cy: float, fw: float, W: int, H: int, t0: float, t1: float) -> list[dict]:
    """One detected face (centre and width in 0..1 of the frame) -> its zones: the face (dilated x1.3 sideways, hairline to chin), the mouth-and-eyes
    (the inner 60 %: hard), and the body below (1.5 face-heights under the chin, lighter)."""
    aspect = W / max(1.0, H)
    fh = fw * aspect * 1.25                                              # a face is about 1.25 times taller than wide
    x0, x1 = (cx - fw * 0.65) * W, (cx + fw * 0.65) * W
    y0, y1 = (cy - fh * 0.5) * H, (cy + fh * 0.5) * H
    inner = (cx * W - fw * W * 0.3, cy * H - fh * H * 0.3, cx * W + fw * W * 0.3, cy * H + fh * H * 0.3)
    body = (x0 - fw * W * 0.2, y1, x1 + fw * W * 0.2, min(H, y1 + 1.5 * fh * H))
    return [zone("face", (x0, y0, x1, y1), t0, t1), zone("mouth_eyes", inner, t0, t1), zone("body", body, t0, t1)]


def caption_zone(band: tuple, t0: float = 0.0, t1: float = 1e9) -> dict:
    """band = (y_top, y_bottom) in pixels; full width."""
    return zone("caption", (0, band[0], 1e6, band[1]), t0, t1)


def ui_zones(W: int, H: int, safe: dict) -> list[dict]:
    """Everything outside the safe box (the platforms' buttons and the TV border) as four strips."""
    return [zone("ui", (0, 0, W, safe["y0"])), zone("ui", (0, safe["y1"], W, H)),
            zone("ui", (0, 0, safe["x0"], H)), zone("ui", (safe["x1"], 0, W, H))]


def _overlap(rect: tuple, box: tuple) -> float:
    x, y, w, h = rect
    ix = max(0.0, min(x + w, box[2]) - max(x, box[0]))
    iy = max(0.0, min(y + h, box[3]) - max(y, box[1]))
    return ix * iy


def covered(rect: tuple, zones: list[dict], t0: float, t1: float, step: float = 0.5) -> float:
    """The time-averaged weighted share of rect (x, y, w, h) covered by the zones active in [t0, t1]. 0 = nothing in the way, 1 = all of it. A presenter
    who walks through the window only counts for the part of the window they are there."""
    x, y, w, h = rect
    area = max(1.0, w * h)
    n = max(1, int(round((t1 - t0) / step)))
    ts = [t0 + (i + 0.5) * (t1 - t0) / n for i in range(n)]
    acc = 0.0
    for t in ts:
        c = 0.0
        for z in zones:
            if z["t0"] <= t < z["t1"]:
                c += z["weight"] * _overlap(rect, z["box"]) / area
        acc += min(1.0, c)
    return acc / len(ts)


def hard_hit(rect: tuple, zones: list[dict], t0: float, t1: float, share: float = 0.0, step: float = 0.5) -> float:
    """The share of the window in which the rectangle touches a hard zone (mouth and eyes, captions)."""
    n = max(1, int(round((t1 - t0) / step)))
    hits = 0
    for i in range(n):
        t = t0 + (i + 0.5) * (t1 - t0) / n
        if any(z["kind"] in HARD and z["t0"] <= t < z["t1"] and _overlap(rect, z["box"]) > share for z in zones):
            hits += 1
    return hits / n


# ------------------------------------------------------------------------------------------------------------------ intrusiveness map
def saliency(frames: list[np.ndarray]) -> np.ndarray:
    """0..1 map (at the frames' own size) of where the eye goes: edge density plus how much moves between the sampled frames."""
    g = [np.asarray(f, np.float32).mean(axis=2) if f.ndim == 3 else np.asarray(f, np.float32) for f in frames]
    edges = np.zeros_like(g[0])
    for a in g:
        gy, gx = np.gradient(a)
        edges += np.hypot(gx, gy)
    edges /= max(1, len(g))
    motion = np.zeros_like(g[0])
    for a, b in zip(g, g[1:]):
        motion += np.abs(b - a)
    if len(g) > 1:
        motion /= len(g) - 1
    sm = edges / (edges.max() + 1e-6) * 0.7 + motion / (motion.max() + 1e-6) * 0.3
    return np.clip(sm, 0.0, 1.0)


SUBJECT_MAX_AREA = 0.35                              # a "subject" blob bigger than this share of the frame is a busy picture, not a person: no zone


def subject_zone(frames: list[np.ndarray], W: int, H: int, t0: float, t1: float, max_area: float = SUBJECT_MAX_AREA) -> dict | None:
    """A conservative fallback when no face could be measured (a small or distant person, no face model): a box around the strongest edges-and-motion blob
    of the sampled frames (their top 3 %), grown 12 %, in pixels of a W x H frame; None when nothing stands out or the blob is most of the picture."""
    frames = [f for f in (frames or []) if f is not None and getattr(f, "ndim", 0) >= 2 and f.size]
    if not frames or W <= 0 or H <= 0:
        return None
    shape = frames[0].shape[:2]
    frames = [f for f in frames if f.shape[:2] == shape]
    sal = saliency(frames)
    h, w = sal.shape
    thr = max(0.35, float(np.percentile(sal, 97)))
    ys, xs = np.nonzero(sal >= thr)
    if len(xs) < max(6, 0.0015 * h * w):
        return None
    x0, x1 = np.percentile(xs, [3, 97])
    y0, y1 = np.percentile(ys, [3, 97])
    gx, gy = 0.12 * max(2.0, x1 - x0), 0.12 * max(2.0, y1 - y0)
    x0, x1, y0, y1 = max(0.0, x0 - gx), min(w - 1.0, x1 + gx), max(0.0, y0 - gy), min(h - 1.0, y1 + gy)
    if (x1 - x0) * (y1 - y0) > max_area * w * h:
        return None
    return zone("subject", (x0 / w * W, y0 / h * H, x1 / w * W, y1 / h * H), t0, t1)


def intrusiveness(shape: tuple, zones: list[dict], t: float, sal: np.ndarray | None = None, scale: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """The map I = sum of weighted zones active at t + 0.3 * saliency, at 1/scale resolution, clipped to 0..1, and its integral image (so the cost of any
    rectangle is four lookups). shape = (H, W) in pixels."""
    H, W = shape
    h, w = max(1, H // scale), max(1, W // scale)
    m = np.zeros((h, w), np.float32)
    for z in zones:
        if z["t0"] <= t < z["t1"]:
            x0, y0, x1, y1 = z["box"]
            a, b, c, d = int(max(0, x0) // scale), int(max(0, y0) // scale), int(min(W, x1) // scale + 1), int(min(H, y1) // scale + 1)
            if c > a and d > b:
                m[b:d, a:c] += z["weight"]
    if sal is not None and sal.size:
        s = np.asarray(sal, np.float32)
        ys = np.linspace(0, s.shape[0] - 1, h).astype(int)
        xs = np.linspace(0, s.shape[1] - 1, w).astype(int)
        m += W_SALIENCY * s[np.ix_(ys, xs)]
    m = np.clip(m, 0.0, 1.0)
    ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return m, ii


def rect_cost(ii: np.ndarray, rect: tuple, shape: tuple, scale: int = 8) -> float:
    """The mean intrusiveness under rect (x, y, w, h) from the integral image."""
    H, W = shape
    x, y, w, h = rect
    a, b = int(max(0, x) // scale), int(max(0, y) // scale)
    c, d = int(min(W, x + w) // scale), int(min(H, y + h) // scale)
    c, d = min(max(c, a + 1), ii.shape[1] - 1), min(max(d, b + 1), ii.shape[0] - 1)
    a, b = min(a, c - 1), min(b, d - 1)
    s = ii[d, c] - ii[b, c] - ii[d, a] + ii[b, a]
    return float(s / max(1, (c - a) * (d - b)))


# ------------------------------------------------------------------------------------------------------------------ slots
def slots(kind: str, W: int, H: int, safe: dict, caption_top: float | None = None) -> list[dict]:
    """The fixed anchors a kind of graphic may take, in priority order. A slot is {"name", "ax": left|center|right, "ay": top|mid|bottom, "x", "y"} where
    (x, y) is the anchor point in pixels inside the safe box (a rectangle is placed so that its `ax`/`ay` edge sits on it: see rect_for)."""
    portrait = H > W
    sx0, sy0, sx1, sy1 = safe["x0"], safe["y0"], safe["x1"], safe["y1"]
    mx, my = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    pad = max(8, int(min(W, H) * 0.012))
    bottom = sy1 if caption_top is None else min(sy1, caption_top - 0.02 * H)
    out: list[dict] = []

    def add(name, ax, ay, x, y):
        out.append({"name": name, "ax": ax, "ay": ay, "x": float(x), "y": float(y)})

    if kind in ("lower", "hook", "title", "point", "reference"):
        if portrait:
            add("bottom_left", "left", "bottom", sx0 + pad, bottom)
            add("bottom_center", "center", "bottom", mx, bottom)
            add("bottom_right", "right", "bottom", sx1 - pad, bottom)
            add("mid_left", "left", "mid", sx0 + pad, my)
            add("mid_right", "right", "mid", sx1 - pad, my)
            add("top_left", "left", "top", sx0 + pad, sy0 + pad)
        else:
            add("bottom_left", "left", "bottom", sx0 + pad, bottom)
            add("bottom_right", "right", "bottom", sx1 - pad, bottom)
            add("upper_left", "left", "top", sx0 + pad, sy0 + pad)
            add("upper_right", "right", "top", sx1 - pad, sy0 + pad)
    elif kind in ("stat", "quote", "chart", "slide", "ledger", "timeline"):
        add("center", "center", "mid", mx, my)
        add("upper_center", "center", "top", mx, sy0 + pad + (sy1 - sy0) * 0.04)
        add("upper_left", "left", "top", sx0 + pad, sy0 + pad)
        add("upper_right", "right", "top", sx1 - pad, sy0 + pad)
    elif kind in ("entity", "chips"):
        add("top_center", "center", "top", mx, sy0 + pad)
        add("side_left", "left", "mid", sx0 + pad, my)
        add("side_right", "right", "mid", sx1 - pad, my)
    elif kind == "bug":
        if portrait:
            add("top_left", "left", "top", sx0 + pad, sy0 + pad)
            add("top_right", "right", "top", sx1 - pad, sy0 + pad)
        else:
            add("top_right", "right", "top", sx1 - pad, sy0 + pad)
            add("top_left", "left", "top", sx0 + pad, sy0 + pad)
    else:
        add("center", "center", "mid", mx, my)
        add("upper_center", "center", "top", mx, sy0 + pad)
    return out


def rect_for(slot: dict, w: float, h: float) -> tuple[float, float, float, float]:
    x = slot["x"] - (0 if slot["ax"] == "left" else w if slot["ax"] == "right" else w / 2)
    y = slot["y"] - (0 if slot["ay"] == "top" else h if slot["ay"] == "bottom" else h / 2)
    return (x, y, float(w), float(h))


def _speaker_centre(zones: list[dict], t0: float, t1: float) -> tuple[float, float] | None:
    pts = [((z["box"][0] + z["box"][2]) / 2, (z["box"][1] + z["box"][3]) / 2) for z in zones if z["kind"] == "face" and z["t0"] < t1 and z["t1"] > t0]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def choose(kind: str, card_w: float, card_h: float, t0: float, t1: float, shots: list[tuple[float, float]], zones: list[dict], frames_fn: Callable | None,
           W: int, H: int, safe: dict | None = None, prev_slot: str | None = None, hysteresis: float = HYSTERESIS, caption_top: float | None = None,
           min_len: float = 1.5, cands: list[dict] | None = None) -> dict | None:
    """The slot for a w x h card shown in [t0, t1]. Per shot (not per frame): for each shot overlapping the window sample up to 3 frames (frames_fn(t) ->
    RGB array or None), cost(slot) = mean intrusiveness under the card + 0.05 x the distance from the speaker's face + 0.25 if it is not the previous slot.
    The previous slot is kept unless another beats it by `hysteresis` (relative). A card never goes over mouth and eyes or the captions. Over COST_LIMIT:
    the longest sub-window that is clear enough (not shorter than min_len), else None (the caller drops the card, with a note).
    -> {"slot", "x", "y", "w", "h", "t0", "t1", "cost", "shortened"} (pixels, window in the video's seconds) or None."""
    try:
        card_w, card_h, t0, t1 = float(card_w), float(card_h), float(t0), float(t1)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (card_w, card_h, t0, t1)) or card_w <= 0 or card_h <= 0 or t1 - t0 <= 1e-3 or W <= 0 or H <= 0:
        return None                                                                # (no size, no window: nothing to place)
    if card_w > W or card_h > H:
        return None                                                                # (bigger than the frame: nothing sensible to place)
    safe = safe or safe_box(W, H)
    shrunk = False
    sw, sh = safe["x1"] - safe["x0"], safe["y1"] - safe["y0"]
    if sw > 0 and sh > 0 and (card_w > sw or card_h > sh):
        k = min(sw / card_w, sh / card_h)
        if k < 0.5:
            return None                                                            # (too big to shrink into the safe box without making it unreadable)
        card_w, card_h, shrunk = card_w * k, card_h * k, True
    cands = list(cands) if cands else slots(kind, W, H, safe, caption_top)        # (cands: the scene's own anchors, to be ranked by what is under them)
    cands = [c for c in cands if _in_frame(rect_for(c, card_w, card_h), W, H)]    # (never a slot that puts the card off the picture)
    if not cands:
        return None
    if frames_fn is not None and not any(z["kind"] in ("face", "mouth_eyes", "subject") and z["t0"] < t1 and z["t1"] > t0 for z in zones):
        try:                                                                       # no face measured: keep clear of whatever stands out in the picture
            frs = [frames_fn(t0 + (i + 0.5) * (t1 - t0) / 3) for i in range(3)]
            sz = subject_zone([f for f in frs if f is not None], W, H, t0, t1)
            if sz is not None:
                zones = [*zones, sz]
        except Exception:  # noqa: BLE001 - no picture, no fallback zone
            pass
    shots = [(max(a, t0), min(b, t1)) for a, b in (shots or [(t0, t1)]) if min(b, t1) - max(a, t0) > 1e-3] or [(t0, t1)]
    centre = _speaker_centre(zones, t0, t1)
    diag = float(np.hypot(W, H))
    cache: dict[float, tuple] = {}

    def maps(t: float):
        key = round(t, 2)
        if key not in cache:
            fr = frames_fn(t) if frames_fn else None
            sal = saliency([fr]) if fr is not None else None
            cache[key] = intrusiveness((H, W), zones, t, sal)
        return cache[key]

    def sub_cost(rect: tuple, a: float, b: float) -> float:
        n = 3 if b - a > 1.0 else 1
        cs = []
        for i in range(n):
            t = a + (i + 0.5) * (b - a) / n
            cs.append(rect_cost(maps(t)[1], rect, (H, W)))
        return float(np.mean(cs))

    def slot_cost(slot: dict, a_b: list[tuple[float, float]]) -> tuple[float, tuple, float]:
        rect = rect_for(slot, card_w, card_h)
        tot, wsum = 0.0, 0.0
        for a, b in a_b:
            ln = max(1e-3, b - a)
            tot += sub_cost(rect, a, b) * ln
            wsum += ln
        c = tot / max(1e-3, wsum)
        if centre is not None:
            c += NEAR * min(1.0, float(np.hypot(rect[0] + rect[2] / 2 - centre[0], rect[1] + rect[3] / 2 - centre[1])) / diag)
        return c, rect, hard_hit(rect, zones, t0, t1)

    scored = []
    for rank, s in enumerate(cands):
        c, rect, hh = slot_cost(s, shots)
        scored.append((c + 0.01 * rank, s, rect, hh))                  # (slots are in priority order: the first wins a tie)
    slack = max(2.0, 0.003 * W)
    ok = [x for x in scored if x[3] == 0.0 and inside(x[2], safe, slack=slack)]
    if not ok:                                                         # none clear inside the safe box: a shorter stretch inside it before a place outside it
        in_box = [c for c in cands if inside(rect_for(c, card_w, card_h), safe, slack=slack)]
        short = _shorten(kind, in_box, card_w, card_h, t0, t1, zones, shots, sub_cost, min_len, centre, diag) if in_box else None
        if short is not None:
            return {**short, "shrunk": shrunk}
        ok = [x for x in scored if x[3] == 0.0]
    if not ok:
        short = _shorten(kind, cands, card_w, card_h, t0, t1, zones, shots, sub_cost, min_len, centre, diag)
        return {**short, "shrunk": shrunk} if short else None
    ok.sort(key=lambda x: x[0])
    best = ok[0]
    if prev_slot is not None:
        prev = next((x for x in ok if x[1]["name"] == prev_slot), None)
        if prev is not None and best[1]["name"] != prev_slot and prev[0] - best[0] < hysteresis * max(prev[0], 0.1):
            best = prev                                               # (not a quarter better: stay)
    c, s, rect, _hh = best
    if c > COST_LIMIT:
        short = _shorten(kind, cands, card_w, card_h, t0, t1, zones, shots, sub_cost, min_len, centre, diag)
        if short is not None:
            return {**short, "shrunk": shrunk}
        return None
    return {"slot": s["name"], "x": rect[0], "y": rect[1], "w": card_w, "h": card_h, "t0": t0, "t1": t1, "cost": round(float(c), 3), "shortened": False, "shrunk": shrunk}


def _in_frame(rect: tuple, W: int, H: int, slack: float = 1.5) -> bool:
    x, y, w, h = rect
    return x >= -slack and y >= -slack and x + w <= W + slack and y + h <= H + slack


def _shorten(kind, cands, card_w, card_h, t0, t1, zones, shots, sub_cost, min_len, centre, diag) -> dict | None:
    """The longest stretch of [t0, t1] (cut at shot edges and half-second steps) in which some slot is clear enough."""
    marks = sorted({round(t0, 3), round(t1, 3), *[round(a, 3) for a, _b in shots], *[round(b, 3) for _a, b in shots],
                    *[round(t0 + k * 0.5, 3) for k in range(1, int((t1 - t0) / 0.5))]})
    marks = [m for m in marks if t0 - 1e-6 <= m <= t1 + 1e-6]
    best = None
    for i, a in enumerate(marks):
        for b in reversed(marks[i + 1:]):
            if b - a < min_len or (best and b - a <= best["t1"] - best["t0"]):
                continue
            for s in cands:
                rect = rect_for(s, card_w, card_h)
                if hard_hit(rect, zones, a, b) > 0.0:
                    continue
                c = sub_cost(rect, a, b)
                if c <= COST_LIMIT:
                    best = {"slot": s["name"], "x": rect[0], "y": rect[1], "w": card_w, "h": card_h, "t0": a, "t1": b, "cost": round(float(c), 3), "shortened": True}
                    break
    return best


# ------------------------------------------------------------------------------------------------------------------ captions
MAX_BANDS = 24                                    # caption stretches in one render (each is one overlay branch in the ffmpeg graph)


def caption_band(shots: list[tuple[float, float]], faces: list[dict], W: int, H: int, default_center: float, box_h: int, safe: dict | None = None,
                 min_gap: float = 2.0) -> list[tuple[float, float, int]]:
    """[(t0, t1, y_top)] for the caption box in every stretch: the default place, or the top of the safe box in a shot in which a face is where the
    captions would be (the chin below 0.57 of the height, or any face box touching the band). Never moves inside a shot; no two moves within `min_gap`
    seconds of each other (a move is dropped, the band stays where it was). faces: zones of kind "face" with times."""
    safe = safe or safe_box(W, H)
    default_y = int(H * default_center - box_h / 2)
    top_y = int(safe["y0"] + 0.01 * H)
    out: list[tuple[float, float, int]] = []
    for a, b in shots:
        fs = [z for z in faces if z["kind"] == "face" and z["t0"] < b and z["t1"] > a]
        low = any(z["box"][3] > 0.57 * H for z in fs)
        touch = any(_overlap((0, default_y, W, box_h), z["box"]) > 0 for z in fs)
        y = top_y if (low or touch) and fs else default_y
        if out and out[-1][2] != y and a - out[-1][0] < min_gap:
            y = out[-1][2]
        if out and out[-1][2] == y and abs(out[-1][1] - a) < 1e-3:
            out[-1] = (out[-1][0], b, y)
        else:
            out.append((float(a), float(b), int(y)))
    if len(out) > MAX_BANDS and min_gap < 120.0:                                   # (too many moves for one ffmpeg graph: move less often, never drop the end)
        return caption_band(shots, faces, W, H, default_center, box_h, safe, min_gap * 2)
    if len(out) > MAX_BANDS:
        return [(out[0][0], out[-1][1], default_y)]
    return out


# ------------------------------------------------------------------------------------------------------------------ contrast
def _lin(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb) -> float:
    r, g, b = (float(v) for v in _lin(np.asarray(rgb, np.float32)[:3]))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(l1: float, l2: float) -> float:
    a, b = max(l1, l2), min(l1, l2)
    return (a + 0.05) / (b + 0.05)


def local_contrast(frame: np.ndarray, rect: tuple, fg=(255, 255, 255)) -> float:
    """The worst WCAG contrast between the text colour `fg` and the picture under rect (x, y, w, h): measured at the brightest and the darkest 5 %
    of the pixels, where it is lowest (a mean would hide a white sky behind white words)."""
    x, y, w, h = (int(v) for v in rect)
    H, W = frame.shape[:2]
    crop = frame[max(0, y):min(H, y + h), max(0, x):min(W, x + w)]
    if crop.size == 0:
        return 21.0
    lum = (0.2126 * _lin(crop[..., 0]) + 0.7152 * _lin(crop[..., 1]) + 0.0722 * _lin(crop[..., 2])).ravel()
    lum.sort()
    k = max(1, int(len(lum) * 0.05))
    lf = luminance(fg)
    return float(min(contrast_ratio(lf, float(lum[:k].mean())), contrast_ratio(lf, float(lum[-k:].mean()))))


def plate_for(ratio: float, fg=(255, 255, 255)) -> dict | None:
    """What a card's text needs to be readable at `ratio`: nothing at 4.5:1 and above; else a 3 px stroke, then a 35 % plate (black behind light text,
    white behind dark text), then a solid plate. Proposal values (WCAG 4.5:1; 30-40 % black plate is the working heuristic)."""
    if ratio >= CONTRAST_MIN:
        return None
    dark_ink = luminance(fg) < 0.4
    plate = (255, 255, 255) if dark_ink else (0, 0, 0)
    if ratio >= 3.0:
        return {"stroke": 3, "plate": None, "alpha": 0.0, "ink_dark": dark_ink}
    if ratio >= 1.6:
        return {"stroke": 3, "plate": plate, "alpha": 0.35, "pad": 10, "ink_dark": dark_ink}
    return {"stroke": 3, "plate": plate, "alpha": 1.0, "pad": 10, "ink_dark": dark_ink}


def tone_for(frames: list[np.ndarray]) -> str:
    """"light" when the picture under a card is bright on average (dark ink and a dark look read best), else "dark"."""
    if not frames:
        return "dark"
    m = float(np.mean([np.asarray(f, np.float32).mean() for f in frames]))
    return "light" if m > 150 else "dark"


# ------------------------------------------------------------------------------------------------------------------ using it from the render
def frame_grabber(src, width: int = 270) -> Callable:
    """frames_fn(t) -> a small RGB frame of the picture at t seconds (None when it can't be read). Cached by tenth of a second."""
    import subprocess
    from ..core import ffmpeg
    cache: dict[float, np.ndarray | None] = {}

    def grab(t: float):
        key = round(float(t), 1)
        if key in cache:
            return cache[key]
        fr = None
        try:
            p = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{max(0.0, key):.2f}", "-i", str(src), "-frames:v", "1", "-vf", f"scale={width}:-2:flags=area",
                                "-f", "image2pipe", "-vcodec", "bmp", "-"], capture_output=True, timeout=30)
            if p.stdout:
                import io
                from PIL import Image
                fr = np.asarray(Image.open(io.BytesIO(p.stdout)).convert("RGB"))
        except Exception:  # noqa: BLE001 - no picture means no saliency term, not a failure
            fr = None
        cache[key] = fr
        return fr
    return grab


def make_placer(kind: str, at: float, secs: float, shots: list[tuple[float, float]], zones: list[dict], frames_fn: Callable | None, W: int, H: int,
                safe: dict | None = None, caption_top: float | None = None) -> Callable:
    """-> placer(w, h, prefer) for gfxcards.Look: ranks the scene's own anchors (prefer = [(x, y) top-left, pixels of the output frame]) by how little
    they cover (faces, captions, platform buttons, busy picture) in every shot the card is up in, and returns the best (x, y), or None when none is clear
    (the Look then does its own search). The first anchor is the design's own choice: another one has to be a quarter better to replace it."""
    safe = safe or safe_box(W, H)

    def placer(w: float, h: float, prefer: list[tuple[float, float]]):
        cands = [{"name": f"p{i}", "ax": "left", "ay": "top", "x": float(x), "y": float(y)} for i, (x, y) in enumerate(prefer)]
        r = choose(kind, w, h, at, at + secs, shots, zones, frames_fn, W, H, safe, prev_slot="p0" if cands else None, caption_top=caption_top, cands=cands)
        if not r or r.get("shortened") or r.get("shrunk"):
            return None
        return (r["x"], r["y"])
    return placer

