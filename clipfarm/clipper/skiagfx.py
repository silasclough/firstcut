"""Animated graphics drawn with Skia (skia-python, BSD-3): smooth paths, gradients and springy type, without Hyperframes (node + Chrome).

Three full-screen kinds, on the same page, colours, fonts and transitions as the Pillow cards (gfxcards._fullscreen), so they sit in a
video like any other card. With skia missing they fall back to their Pillow cousins (gfxcards picks: drawchart -> chart, count ->
bigstat, kinetic -> statement):

    drawchart   a line or bar chart that draws itself; the values count up; an arrow (lottiefx) points at the highest one
    count       one number counting up; a percentage fills a ring as it counts; a ping when it lands
    kinetic     a short punchline revealed word by word with a spring; the key words in the accent colour, underlined (lottiefx)

    available() -> skia can be used
    scene_drawchart / scene_count / scene_kinetic (look, spec, seconds, wf) -> gfxcards.Scene | None
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
from PIL import Image

from . import gfxcards as G
from . import lottiefx as LF

_TF: dict = {}
_KEEP: list = []                    # (skia variation objects must outlive the clones made from them)


def available() -> bool:
    try:
        import skia  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


WEIGHT = {"regular": 400, "medium": 520, "bold": 700, "heavy": 860}


def typeface(weight: str):
    import skia
    if weight in _TF:
        return _TF[weight]
    path = next((p for p in G.FONT_FILES.get(weight, G.FONT_FILES["regular"]) if Path(p).exists()), None)
    tf = skia.Typeface.MakeFromFile(path) if path else skia.Typeface.MakeDefault()
    if path == G.SYSTEM_SF and tf is not None:                                   # the system font is variable: pick the weight
        try:
            VP = skia.FontArguments.VariationPosition
            tag = (ord("w") << 24) | (ord("g") << 16) | (ord("h") << 8) | ord("t")
            coords = VP.Coordinates([VP.Coordinate(tag, float(WEIGHT.get(weight, 400)))])
            pos = VP(coords)
            args = skia.FontArguments()
            args.setVariationDesignPosition(pos)
            _KEEP.extend([coords, pos, args])
            tf = tf.makeClone(args) or tf
        except Exception:  # noqa: BLE001 - the default weight then
            pass
    _TF[weight] = tf
    return tf


def _font(weight: str, size: float):
    import skia
    f = skia.Font(typeface(weight), float(size))
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    return f


def _paint(rgb, a: float = 1.0, stroke: float = 0.0, cap_round: bool = True):
    import skia
    p = skia.Paint(AntiAlias=True, Color=skia.Color(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(255 * max(0.0, min(1.0, a)))))
    if stroke:
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(float(stroke))
        if cap_round:
            p.setStrokeCap(skia.Paint.kRound_Cap)
            p.setStrokeJoin(skia.Paint.kRound_Join)
    return p


def _surface(W: int, H: int):
    import skia
    s = skia.Surface(int(W), int(H))
    s.getCanvas().clear(skia.Color4f(0, 0, 0, 0))
    return s


def _image(surface) -> Image.Image:
    import skia
    a = surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
    return Image.fromarray(np.ascontiguousarray(a), "RGBA")


def spring(t: float, k: float = 9.0, damp: float = 0.42) -> float:
    """0 -> 1 with one soft overshoot (a damped spring): the motion of a word landing."""
    if t <= 0:
        return 0.0
    if t >= 1.6:
        return 1.0
    return 1.0 - math.exp(-damp * k * t) * math.cos(k * t * 1.15)


def _ease(t: float) -> float:
    return G.ease_out(max(0.0, min(1.0, t)))


def _accent(rgbtuple, x: int, y: int, name: str, w: int, h: int, t: float, start: float, dur: float = 0.7):
    """A lottie accent frame (or None before it starts)."""
    if t < start:
        return None
    fr = LF.frames(name, w, h, tuple(rgbtuple), 24)
    i = min(len(fr) - 1, int((t - start) / dur * (len(fr) - 1)))
    return fr[i], x, y


# ------------------------------------------------------------------------------------------------ drawchart
def scene_drawchart(look: G.Look, spec: dict, seconds: float, wf: float = 1.0) -> G.Scene | None:
    ch = spec.get("chart") or spec
    labels = [G._clean(x, 18) for x in (ch.get("labels") or [])]
    nums = [G._num(v) for v in (ch.get("values") or [])]
    pairs = [(lb, n) for lb, n in zip(labels, nums) if lb and n]
    if len(pairs) < 2:
        return None
    pairs = pairs[:8]
    kind = "line" if (ch.get("type") == "line" or ch.get("chart_type") == "line") and len(pairs) >= 3 else "bar"
    unit = str(ch.get("unit") or "")[:4]
    title = G._clean(ch.get("title") or spec.get("title") or "", 60)
    W, H = look.W, look.H
    mx = int(W * 0.09)
    tf, tl, tsize, th = G._title_block(look, title, mx, W) if title else (None, [], 0, 0)
    y_title = int(H * 0.09)
    top = (y_title + th + look.px(70)) if title else int(H * 0.16)
    bottom = int(H * (0.80 if look.portrait else 0.78))
    left, right = mx + look.px(10), W - mx
    vals = [n[0] for _l, n in pairs]
    vmax = max(vals) if max(vals) > 0 else 1.0
    vmin = min(0.0, min(vals))
    span = (vmax - vmin) or 1.0
    n = len(pairs)
    draw_t = min(2.2, max(1.2, seconds * 0.35))                                       # the chart draws itself over this long
    lab_size = look.px(36 if look.portrait else 30)
    val_size = look.px(44 if look.portrait else 38)
    peak = max(range(n), key=lambda i: vals[i])

    def ypos(v: float) -> float:
        return bottom - (v - vmin) / span * (bottom - top)

    def fmt(i: int, frac: float) -> str:
        v, dec, suf = pairs[i][1]
        shown = vals[i] * frac
        s = f"{shown:,.{dec}f}" if abs(vals[i]) >= 1000 else f"{shown:.{dec}f}"
        return s + (suf or unit)

    def draw(dr, lay, t, ink, dim, acc):
        import skia
        if title:
            G._draw_title(look, dr, t, tf, tl, tsize, mx, y_title, ink, acc)
        s = _surface(W, H)
        c = s.getCanvas()
        c.drawLine(left, bottom, right, bottom, _paint(dim, 0.6 * _ease(t / 0.4), max(2, look.px(3))))
        lf, vf = _font("bold", lab_size), _font("heavy", val_size)
        k = _ease((t - 0.3) / draw_t)                                                  # how much of the chart is drawn
        if kind == "line":
            xs = [left + (right - left) * (i / (n - 1)) for i in range(n)]
            ys = [ypos(v) for v in vals]
            path = skia.Path()
            path.moveTo(xs[0], ys[0])
            for i in range(1, n):                                                       # a smooth curve through the points
                cx = (xs[i - 1] + xs[i]) / 2
                path.cubicTo(cx, ys[i - 1], cx, ys[i], xs[i], ys[i])
            meas = skia.PathMeasure(path, False)
            L = meas.getLength()
            part = skia.Path()
            meas.getSegment(0, L * k, part, True)
            if k > 0.02:                                                                # the area under it, fading in as it draws
                area = skia.Path(part)
                end = part.getPoint(part.countPoints() - 1)
                area.lineTo(end.x(), bottom)
                area.lineTo(xs[0], bottom)
                area.close()
                fill = skia.Paint(AntiAlias=True)
                fill.setShader(skia.GradientShader.MakeLinear(
                    [skia.Point(0, top), skia.Point(0, bottom)],
                    [skia.Color(*acc, int(110 * k)), skia.Color(*acc, 0)]))
                c.drawPath(area, fill)
            c.drawPath(part, _paint(acc, 1.0, max(5, look.px(9))))
            for i in range(n):
                reach = (xs[i] - left) / max(1, right - left)
                a = _ease((k - reach) * 6 + 0.05) if k < 1 else 1.0
                if a <= 0:
                    continue
                r = look.px(14) * (0.6 + 0.4 * spring(min(1.6, (k - reach) * 4)))
                c.drawCircle(xs[i], ys[i], r, _paint(acc, a))
                c.drawCircle(xs[i], ys[i], r * 0.45, _paint(ink if look.light else (255, 255, 255), a))
                txt = fmt(i, min(1.0, a))
                tw = vf.measureText(txt)
                c.drawString(txt, xs[i] - tw / 2, ys[i] - look.px(30), vf, _paint(ink, a))
                lw = lf.measureText(pairs[i][0])
                c.drawString(pairs[i][0], xs[i] - lw / 2, bottom + look.px(46), lf, _paint(dim, a))
            px, py = xs[peak], ys[peak]
        else:
            gap = (right - left) / n
            bw = gap * 0.62
            for i in range(n):
                start = 0.3 + i * min(0.22, draw_t / (n + 1))
                g = spring(max(0.0, (t - start) / 0.9))
                if g <= 0:
                    continue
                x0 = left + gap * i + (gap - bw) / 2
                y1 = ypos(vals[i] * min(1.08, g))
                y0 = ypos(0.0)
                rr = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(x0, min(y0, y1), x0 + bw, max(y0, y1)), look.px(10), look.px(10))
                p = skia.Paint(AntiAlias=True)
                col = acc if i == peak else G.mix(acc, (ink if look.light else (255, 255, 255)), 0.35) if not look.light else G.mix(acc, (255, 255, 255), 0.35)
                p.setShader(skia.GradientShader.MakeLinear([skia.Point(0, y1), skia.Point(0, y0)],
                                                           [skia.Color(*col, 255), skia.Color(*G.mix(col, (0, 0, 0), 0.25), 255)]))
                c.drawRRect(rr, p)
                txt = fmt(i, min(1.0, g))
                tw = vf.measureText(txt)
                c.drawString(txt, x0 + bw / 2 - tw / 2, y1 - look.px(18), vf, _paint(ink, min(1.0, g)))
                lw = lf.measureText(pairs[i][0])
                c.drawString(pairs[i][0], x0 + bw / 2 - lw / 2, bottom + look.px(46), lf, _paint(dim, min(1.0, g)))
            px, py = left + gap * peak + gap / 2, ypos(vals[peak]) - look.px(70)
        lay.alpha_composite(_image(s))
        aw, ah = look.px(240), look.px(140)                                            # a line chart: the arrow points at its highest value
        acc_fr = _accent(acc, int(px - aw - look.px(40)), int(py - ah - look.px(30)), "arrow", aw, ah, t, 0.3 + draw_t + 0.2) if kind == "line" else None
        if acc_fr and n >= 3:
            im, x, y = acc_fr
            lay.alpha_composite(im, (max(0, min(W - aw, x)), max(int(top - look.px(20)), min(H - ah, y))))     # (never over the title)
    return G._fullscreen(look, spec, seconds, draw)


# ------------------------------------------------------------------------------------------------ count
def scene_count(look: G.Look, spec: dict, seconds: float, wf: float = 1.0) -> G.Scene | None:
    value = str(spec.get("value") or "").strip()[:14]
    label = " ".join(str(spec.get("label") or "").split())[:90]
    num = G._num(value)
    if not value or not num:
        return None
    target, dec, suf = num
    prefix = re.match(r"^\s*([^\d\-]*)", value).group(1)
    pct = "%" in value and 0 < target <= 100
    W, H = look.W, look.H
    R = look.px(250 if look.portrait else 230)
    cx, cy = W / 2, H * (0.42 if look.portrait else 0.43)
    big = look.px(170 if pct else 230)
    land = 0.4 + 1.3

    def draw(dr, lay, t, ink, dim, acc):
        import skia
        s = _surface(W, H)
        c = s.getCanvas()
        k = _ease((t - 0.4) / 1.3)
        shown = target * k
        txt = f"{prefix}{shown:,.{dec}f}{suf}" if abs(target) >= 1000 else f"{prefix}{shown:.{dec}f}{suf}"
        f = _font("heavy", big)
        while f.measureText(f"{prefix}{target:,.{dec}f}{suf}") > (R * 1.6 if pct else W * 0.86) and f.getSize() > look.px(60):
            f = _font("heavy", f.getSize() * 0.92)
        if pct:
            rect = skia.Rect.MakeLTRB(cx - R, cy - R, cx + R, cy + R)
            c.drawArc(rect, 0, 360, False, _paint(dim, 0.25, look.px(34)))
            c.drawArc(rect, -90, 360 * (target / 100.0) * k, False, _paint(acc, 1.0, look.px(34)))
        beat = 1.0 + 0.06 * math.sin(math.pi * max(0.0, min(1.0, (t - land) / 0.3)))
        f2 = _font("heavy", f.getSize() * beat)
        tw = f2.measureText(txt)
        c.drawString(txt, cx - tw / 2, cy + f2.getSize() * 0.36, f2, _paint(ink))
        if not pct:                                                                     # (no ring: an accent bar grows under the number)
            bw = min(W * 0.5, look.px(520)) * _ease((t - 0.9) / 0.6)
            c.drawRoundRect(skia.Rect.MakeLTRB(cx - bw / 2, cy + f.getSize() * 0.62, cx + bw / 2, cy + f.getSize() * 0.62 + look.px(10)),
                            look.px(5), look.px(5), _paint(acc))
        lf = _font("bold", look.px(58 if look.portrait else 46))
        y = cy + (R if pct else f.getSize() * 0.62) + look.px(90)
        words, line, lines = label.split(), "", []
        for w in words:                                                                 # wrap the label to 80% of the width, at most 3 lines
            trial = (line + " " + w).strip()
            if lf.measureText(trial) > W * 0.8 and line:
                lines.append(line)
                line = w
            else:
                line = trial
        if line:
            lines.append(line)
        a = _ease((t - 1.1) / 0.4)
        for ln in lines[:3]:
            lw = lf.measureText(ln)
            c.drawString(ln, cx - lw / 2, y, lf, _paint(G._lerp(dim, ink, a), max(0.0, a)))
            y += lf.getSize() * 1.22
        lay.alpha_composite(_image(s))
        pw = look.px(260)                                                               # a ping as the number lands
        acc_fr = _accent(acc, int(cx - pw / 2), int(cy - pw / 2), "ping", pw, pw, t, land, 0.8)
        if acc_fr and t < land + 0.8 and not pct:
            im, x, y0 = acc_fr
            lay.alpha_composite(im, (x, y0))
    return G._fullscreen(look, spec, seconds, draw)


# ------------------------------------------------------------------------------------------------ kinetic
def scene_kinetic(look: G.Look, spec: dict, seconds: float, wf: float = 1.0) -> G.Scene | None:
    text = " ".join(str(spec.get("text") or "").split())
    words = text.split()
    if not 2 <= len(words) <= 12:
        return None
    emph = set(w.lower().strip(".,!?;:'\"") for w in str(spec.get("emphasis") or "").split())
    times = list(spec.get("times") or [])
    if len(times) != len(words):
        times = [0.35 + i * min(0.22, (seconds * 0.45) / len(words)) for i in range(len(words))]
    W, H = look.W, look.H
    maxw = W * 0.84
    size = look.px(150 if look.portrait else 130)
    text_case = (lambda s: s.upper()) if look.upper else (lambda s: s)

    def layout(sz):
        f = _font("heavy", sz)
        space = f.measureText(" ")
        lines, cur, cw = [], [], 0.0
        for i, w in enumerate(words):
            ww = f.measureText(text_case(w))
            if cur and cw + space + ww > maxw:
                lines.append((cur, cw))
                cur, cw = [], 0.0
            cur.append((i, w, ww))
            cw += (space if len(cur) > 1 else 0) + ww
        if cur:
            lines.append((cur, cw))
        return f, space, lines
    f, space, lines = layout(size)
    while (len(lines) > 4 or max(cw for _c, cw in lines) > maxw) and size > look.px(48):
        size *= 0.9
        f, space, lines = layout(size)
    lh = size * 1.12
    y0 = H / 2 - lh * len(lines) / 2 + size * 0.8
    spots = []                                                                          # (word index, x, baseline y, width)
    for li, (cur, cw) in enumerate(lines):
        x = (W - cw) / 2
        for i, w, ww in cur:
            spots.append((i, x, y0 + li * lh, ww))
            x += ww + space
    em_idx = [i for i, w in enumerate(words) if w.lower().strip(".,!?;:'\"") in emph]
    unders = []                                                                         # an underline under the key words on each line they are on
    for yline in sorted({sp[2] for sp in spots if sp[0] in em_idx}):
        same = [sp for sp in spots if sp[0] in em_idx and abs(sp[2] - yline) < 1]
        x0, x1 = min(sp[1] for sp in same), max(sp[1] + sp[3] for sp in same)
        unders.append((int(x0 - look.px(8)), int(yline + size * 0.12), int(x1 - x0 + look.px(16)), int(max(look.px(24), size * 0.22)),
                       times[max(sp[0] for sp in same)] + 0.25))

    def draw(dr, lay, t, ink, dim, acc):
        s = _surface(W, H)
        c = s.getCanvas()
        for i, x, y, ww in spots:
            p = spring((t - times[i]) / 0.55)
            if p <= 0:
                continue
            a = min(1.0, _ease((t - times[i]) / 0.18))
            sc = 0.82 + 0.18 * p
            dy = (1.0 - p) * size * 0.45
            c.save()
            c.translate(x + ww / 2, y - size * 0.35 + dy)
            c.scale(sc, sc)
            col = acc if i in em_idx else ink
            c.drawString(text_case(words[i]), -ww / 2, size * 0.35, f, _paint(col, a))
            c.restore()
        lay.alpha_composite(_image(s))
        for ux, uy, uw, uh, ut in unders:
            fr = _accent(acc, ux, uy, "underline", uw, uh, t, ut, 0.5)
            if fr:
                lay.alpha_composite(fr[0], (max(0, ux), max(0, uy)))
    return G._fullscreen(look, spec, seconds, draw)


SCENES = {"drawchart": scene_drawchart, "count": scene_count, "kinetic": scene_kinetic}
