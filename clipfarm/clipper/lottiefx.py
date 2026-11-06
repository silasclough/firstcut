"""Small animated accents (Lottie), rendered on this Mac with rlottie (rlottie-python, LGPL, loaded as a library).

The accents are our own: tiny Lottie documents written here in code (an arrow that draws itself, a check mark, an underline, an
attention ping), so nothing third-party is shipped. A .json / .lottie file someone has (e.g. from LottieFiles, used under its licence)
renders the same way. Without rlottie the same accents are drawn with Pillow (a plain fallback; the graphic is never lost).

    ACCENTS                              arrow | check | underline | ping
    doc(name, w, h, color, frames)       -> a Lottie document (dict)
    frames(name, w, h, color, n)         -> [PIL RGBA images], n frames (rlottie, else Pillow)
    render_file(path, w, h, fps, secs)   -> [PIL RGBA images] of any .json / .lottie file
    available()                          -> rlottie can be used
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw

from ..core.log import get_logger

log = get_logger("clipper.lottiefx")

ACCENTS = ("arrow", "check", "underline", "ping")


def available() -> bool:
    try:
        import rlottie_python  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - the library or its native part is missing: the Pillow drawing takes over
        return False


# ------------------------------------------------------------------------------------------------ authoring
def _k(v):
    return {"a": 0, "k": v}


def _ease_key(t, s):
    return {"t": t, "s": s, "o": {"x": [0.33], "y": [0.0]}, "i": {"x": [0.2], "y": [1.0]}}


def _trim(n: int, start: int = 0) -> dict:
    """A trim path that draws the stroke on from 0 to 100% over frames [start, start + 0.6 n]."""
    end = max(start + 2, int(start + n * 0.6))
    return {"ty": "tm", "s": _k(0), "o": _k(0), "m": 1, "e": {"a": 1, "k": [_ease_key(start, [0]), {"t": end, "s": [100]}]}}


def _path(points: list[tuple[float, float]], closed: bool = False) -> dict:
    z = [[0, 0]] * len(points)
    return {"ty": "sh", "ks": _k({"i": z, "o": z, "v": [list(p) for p in points], "c": closed})}


def _stroke(color, width: float) -> dict:
    return {"ty": "st", "c": _k([color[0] / 255, color[1] / 255, color[2] / 255, 1]), "o": _k(100), "w": _k(width), "lc": 2, "lj": 2}


def _tr() -> dict:
    return {"ty": "tr", "p": _k([0, 0]), "a": _k([0, 0]), "s": _k([100, 100]), "r": _k(0), "o": _k(100)}


def _layer(shapes: list[dict], n: int, ind: int = 1, ks: dict | None = None) -> dict:
    return {"ddd": 0, "ind": ind, "ty": 4, "nm": f"l{ind}", "sr": 1, "ip": 0, "op": n, "st": 0, "bm": 0, "ao": 0,
            "ks": ks or {"o": _k(100), "r": _k(0), "p": _k([0, 0, 0]), "a": _k([0, 0, 0]), "s": _k([100, 100, 100])},
            "shapes": [{"ty": "gr", "it": shapes + [_tr()]}]}


def doc(name: str, w: int, h: int, color=(255, 210, 63), n: int = 30) -> dict:
    """One of our accents as a Lottie document of w x h pixels and n frames (30 fps)."""
    u = min(w, h)
    sw = max(4.0, u * 0.07)
    if name == "arrow":                     # a curved shaft drawing itself towards the right, then the head
        pts = [(w * 0.08, h * 0.75), (w * 0.35, h * 0.55), (w * 0.62, h * 0.48), (w * 0.86, h * 0.42)]
        head = [(w * 0.70, h * 0.25), (w * 0.88, h * 0.42), (w * 0.72, h * 0.62)]
        layers = [_layer([_path(pts), _stroke(color, sw), _trim(n)], n, 1),
                  _layer([_path(head), _stroke(color, sw), _trim(max(4, n // 3), int(n * 0.5))], n, 2)]
    elif name == "check":                   # a tick inside a circle that draws round first
        r = u * 0.4
        cx, cy = w / 2, h / 2
        circle = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (i * 2 * math.pi / 40 - math.pi / 2 for i in range(41))]
        tick = [(cx - r * 0.45, cy + r * 0.02), (cx - r * 0.1, cy + r * 0.36), (cx + r * 0.5, cy - r * 0.32)]
        layers = [_layer([_path(circle), _stroke(color, sw * 0.7), _trim(n)], n, 1),
                  _layer([_path(tick), _stroke(color, sw), _trim(max(4, n // 2), int(n * 0.4))], n, 2)]
    elif name == "underline":               # a slightly curved hand-drawn underline
        pts = [(w * 0.02, h * 0.62), (w * 0.3, h * 0.48), (w * 0.62, h * 0.44), (w * 0.98, h * 0.4)]
        layers = [_layer([_path(pts), _stroke(color, max(4.0, h * 0.28)), _trim(n)], n, 1)]
    elif name == "ping":                    # a dot and a ring that grows and fades: "look here"
        r = u * 0.12
        cx, cy = w / 2, h / 2
        ring = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in (i * 2 * math.pi / 40 for i in range(41))]
        grow = {"o": {"a": 1, "k": [_ease_key(0, [100]), {"t": n - 1, "s": [0]}]}, "r": _k(0), "p": _k([cx, cy, 0]), "a": _k([cx, cy, 0]),
                "s": {"a": 1, "k": [_ease_key(0, [60, 60, 100]), {"t": n - 1, "s": [330, 330, 100]}]}}
        dot = [(cx + r * 0.55 * math.cos(a), cy + r * 0.55 * math.sin(a)) for a in (i * 2 * math.pi / 30 for i in range(31))]
        layers = [_layer([_path(ring, True), _stroke(color, sw * 0.5)], n, 1, grow),
                  _layer([_path(dot, True), {"ty": "fl", "c": _k([color[0] / 255, color[1] / 255, color[2] / 255, 1]), "o": _k(100), "r": 1}], n, 2)]
    else:
        raise ValueError(f"no accent called {name}")
    return {"v": "5.7.4", "fr": 30, "ip": 0, "op": n, "w": int(w), "h": int(h), "nm": name, "ddd": 0, "assets": [], "layers": layers}


# ------------------------------------------------------------------------------------------------ rendering
def _rlottie(data: str, w: int, h: int, n: int) -> list[Image.Image]:
    from rlottie_python import LottieAnimation
    anim = LottieAnimation.from_data(data)
    total = max(1, int(anim.lottie_animation_get_totalframe()))
    out = []
    for i in range(n):
        f = min(total - 1, int(round(i * (total - 1) / max(1, n - 1)))) if n > 1 else 0
        out.append(anim.render_pillow_frame(frame_num=f, width=w, height=h).convert("RGBA"))
    return out


def _pillow(name: str, w: int, h: int, color, n: int) -> list[Image.Image]:
    """The same accents, drawn plainly with Pillow (no rlottie): a stroke that grows along the same path."""
    d = doc(name, w, h, color, n)
    frames = []
    for i in range(n):
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        for lay in d["layers"]:
            it = lay["shapes"][0]["it"]
            pts = [tuple(p) for p in it[0]["ks"]["k"]["v"]]
            st = next((x for x in it if x["ty"] in ("st", "fl")), None)
            width = int(st["w"]["k"]) if st and st["ty"] == "st" else 0
            tm = next((x for x in it if x["ty"] == "tm"), None)
            frac = 1.0
            if tm:
                k = tm["e"]["k"]
                t0, t1 = k[0]["t"], k[1]["t"]
                frac = max(0.0, min(1.0, (i - t0) / max(1, t1 - t0)))
                frac = 1 - (1 - frac) ** 3
            if frac <= 0:
                continue
            seg = _partial(pts, frac)
            col = (*color, 255)
            if st and st["ty"] == "fl":
                dr.polygon(pts, fill=col)
            elif len(seg) > 1:
                dr.line(seg, fill=col, width=max(1, width), joint="curve")
        frames.append(im)
    return frames


def _partial(pts, frac: float):
    """The first `frac` of a polyline (by length)."""
    lens = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    total = sum(lens) or 1.0
    want, out = frac * total, [pts[0]]
    for (a, b), L in zip(zip(pts, pts[1:]), lens):
        if want >= L:
            out.append(b)
            want -= L
        else:
            k = want / L if L else 0.0
            out.append((a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k))
            break
    return out


_CACHE: dict = {}


def frames(name: str, w: int, h: int, color=(255, 210, 63), n: int = 30) -> list[Image.Image]:
    """n RGBA frames of the accent at w x h (cached)."""
    w, h = max(8, int(w)), max(8, int(h))
    key = (name, w, h, tuple(color), n)
    if key in _CACHE:
        return _CACHE[key]
    out = None
    if available():
        try:
            out = _rlottie(json.dumps(doc(name, w, h, color, n)), w, h, n)
        except Exception as e:  # noqa: BLE001 - drawn with Pillow instead
            log.warning(f"lottie {name} failed in rlottie ({e}): drawn plainly")
    if out is None:
        out = _pillow(name, w, h, color, n)
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = out
    return out


def render_file(path, w: int, h: int, fps: float = 30.0, seconds: float | None = None) -> list[Image.Image]:
    """Any .json or .lottie (zip) animation as RGBA frames at `fps` (its own length unless `seconds` is given). Needs rlottie."""
    p = Path(path)
    if p.suffix.lower() == ".lottie":
        with zipfile.ZipFile(p) as z:
            name = next((x for x in z.namelist() if x.startswith("animations/") and x.endswith(".json")), None)
            if name is None:
                raise ValueError(f"{p.name} has no animation in it")
            data = z.read(name).decode()
    else:
        data = p.read_text()
    d = json.loads(data)
    own = (float(d.get("op", 0)) - float(d.get("ip", 0))) / float(d.get("fr") or 30)
    n = max(1, int(round((seconds if seconds else own) * fps)))
    return _rlottie(data, int(w), int(h), n)


def to_png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO()
    img.save(b, "PNG")
    return b.getvalue()
