"""Animated on-screen graphics, drawn natively (Pillow + ffmpeg): slides, charts, stat call-outs, pull quotes, lower thirds, the title card and
key-point cards, in a "liquid glass" look. No browser, no Node, nothing to install: it works on every machine that has the editor.

Every card is ONE short transparent video (QuickTime Animation with alpha) that the editor lays over the picture at the right moment, and
that Final Cut gets as a connected clip. The glass is real in the finished video: the picture behind each panel is blurred and brightened in
the same rounded shape (the renderer returns the panel rectangles, polish.glass_chain does the blur), and the card itself carries the tint,
the soft shadow, the bright edge and the highlight. (Final Cut gets the card without the blur - the tint and edge are in the clip.)

    render(cfg, spec, W, H, out, theme) -> {"file": out, "glass": [{x, y, w, h, r, dim}]} or None
    spec = {"kind": "hook|point|lower|stat|quote|slide|chart", "len": seconds, ...the kind's data}
"""

from __future__ import annotations

import math
import re
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.gfxcards")
FPS = 30
ENTER, EXIT = 0.45, 0.35

SYSTEM_SF = "/System/Library/Fonts/SFCompact.ttf"            # on every Mac (a variable font): the same look where the Display cuts aren't installed
FONT_FILES = {
    "regular": ["/Library/Fonts/SF-Compact-Display-Regular.otf", SYSTEM_SF, "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"],
    "medium": ["/Library/Fonts/SF-Compact-Display-Medium.otf", SYSTEM_SF, "/Library/Fonts/SF-Compact-Display-Regular.otf", "/System/Library/Fonts/Supplemental/Arial.ttf"],
    "bold": ["/Library/Fonts/SF-Compact-Display-Bold.otf", SYSTEM_SF, "/Library/Fonts/SF-Compact-Display-Semibold.otf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
             "/Library/Fonts/Arial Bold.ttf"],
    "heavy": ["/Library/Fonts/SF-Compact-Display-Heavy.otf", SYSTEM_SF, "/Library/Fonts/SF-Compact-Display-Black.otf", "/System/Library/Fonts/Supplemental/Arial Black.ttf",
              "/Library/Fonts/Arial Bold.ttf"],
}
SYSTEM_SF_WEIGHT = {"regular": "Regular", "medium": "Medium", "bold": "Bold", "heavy": "Heavy"}
_FONTS: dict = {}
_CTX = threading.local()                                    # the look being built: a monospaced look asks every font() for Menlo
MONO_FILE = "/System/Library/Fonts/Menlo.ttc"


def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    mono = bool(getattr(_CTX, "mono", False)) and Path(MONO_FILE).exists()
    key = (weight, size, mono)
    if key not in _FONTS:
        if mono:
            _FONTS[key] = ImageFont.truetype(MONO_FILE, size, index=1 if weight in ("bold", "heavy") else 0)
        else:
            path = next((p for p in FONT_FILES.get(weight, FONT_FILES["regular"]) if Path(p).exists()), None)
            f = ImageFont.truetype(path, size) if path else ImageFont.load_default(size=size)
            if path == SYSTEM_SF:
                try:
                    f.set_variation_by_name(SYSTEM_SF_WEIGHT.get(weight, "Regular"))
                except (OSError, ValueError):
                    pass
            _FONTS[key] = f
    return _FONTS[key]


def rgb(hexcolor: str, default=(255, 255, 255)) -> tuple[int, int, int]:
    m = re.match(r"^#?([0-9a-f]{6})$", str(hexcolor or "").strip(), re.I)
    return tuple(int(m.group(1)[i:i + 2], 16) for i in (0, 2, 4)) if m else default


def ease_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 1 - (1 - t) ** 3


def ease_back(t: float, k: float = 1.9) -> float:
    """Ease-out with an overshoot: it flies past 1.0 and settles back (the 'pop')."""
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    t -= 1
    return 1 + (k + 1) * t ** 3 + k * t ** 2


def ease_in(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t ** 3


def ease_io(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 3 * t * t - 2 * t ** 3


# ------------------------------------------------------------------------------------------------ the look
MODES = {"glass": "glass_dark", "glass_light": "glass_light", "clean": "solid_light", "slate": "solid_dark", "bold": "bold", "mono": "mono", "minimal": "minimal",
         "pastel_orange": "pastel", "pastel_green": "pastel", "pastel_blue": "pastel", "pastel_pink": "pastel", "board": "board",
         "blueprint": "solid_dark", "sky": "solid_light", "mint": "solid_light", "broadcast": "bold", "bold_white": "bold", "glass_amber": "glass_dark", "pastel_yellow": "pastel"}


def mix(a, b, t: float) -> tuple[int, int, int]:
    """a blended toward b by t (0..1)."""
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))


class Look:
    """One video, one look (matched to its footage and kind, see graphics.match_theme):
    glass_dark / glass_light  frosted glass over a blurred picture (dark and airy)      solid_light / solid_dark  flat modern cards
    bold  a loud accent-coloured card with a hard shadow (YouTube)      mono  a terminal card      minimal  just the type, no card"""

    def __init__(self, theme: str | None, W: int, H: int):
        from .graphics import THEMES, _theme
        T = _theme(theme)
        self.name = theme if theme in THEMES else "clean"
        self.page_c = tuple(T["page"]) if T.get("page") else None                         # a look with its own page colours (explainer looks)
        self.ink_c, self.dim_c, self.dot_c = (tuple(T[k]) if T.get(k) else None for k in ("ink", "dim", "dot"))
        self.mode = MODES.get(self.name, "glass_dark")
        self.glass = self.mode in ("glass_dark", "glass_light")
        self.light = self.mode in ("glass_light", "solid_light", "pastel")
        self.mono = self.mode == "board"
        self.tone = "light" if self.light else "dark"
        self.accent = rgb(T["accent"])
        self.upper = T["case"] == "uppercase" or self.mode == "bold"
        self.W, self.H = W, H
        self.portrait = H >= W
        self.u = min(W, H) / 1080.0                       # one design unit = a pixel of a 1080-wide frame
        lum = (0.299 * self.accent[0] + 0.587 * self.accent[1] + 0.114 * self.accent[2]) / 255
        if self.mode == "bold":                           # the card IS the accent colour: ink on it is black or white, whichever reads
            self.fg = (12, 12, 14) if lum > 0.6 else (255, 255, 255)
            self.sub = self.fg
            self.deco = self.fg
        elif self.mode == "pastel":                       # one colour, light and soft: a tinted card, dark shade of the colour for the ink
            self.tint = mix(self.accent, (255, 255, 255), 0.9)
            self.edge = mix(self.accent, (255, 255, 255), 0.55)
            self.fg, self.sub, self.deco = mix(self.accent, (0, 0, 0), 0.68), mix(self.accent, (0, 0, 0), 0.45), self.accent
        elif self.mode == "board":
            self.fg, self.sub, self.deco = (255, 255, 255), (200, 200, 200), self.accent
        elif self.light:
            self.fg, self.sub, self.deco = (16, 20, 30), (72, 84, 108), self.accent
        elif self.mode == "mono":
            self.fg, self.sub, self.deco = rgb("#e8fff1"), (150, 200, 170), self.accent
        else:
            self.fg, self.sub, self.deco = (255, 255, 255), (222, 230, 244), self.accent
        self.solid = {"mono": rgb("#0b0f14"), "solid_dark": (18, 20, 26), "solid_light": (253, 253, 255), "board": (20, 20, 20)}.get(self.mode, rgb("#0b0f14"))
        if self.mode == "pastel":                         # charts in shades of the one colour
            self.palette = [self.accent, mix(self.accent, (255, 255, 255), 0.45), mix(self.accent, (0, 0, 0), 0.35), mix(self.accent, (255, 255, 255), 0.7),
                            mix(self.accent, (0, 0, 0), 0.55), mix(self.accent, (255, 255, 255), 0.25)]
        elif self.mode == "board":
            self.palette = [(255, 210, 63), (225, 80, 70), (255, 255, 255), (96, 165, 250), (110, 220, 160), (255, 150, 60)]
        else:
            self.palette = PALETTE

    avoid: list = []                                  # head zones to keep clear, in pixels of this frame: (x0, y0, x1, y1)

    def px(self, n: float) -> int:
        return max(1, int(round(n * self.u)))

    def case(self, s: str) -> str:
        return s.upper() if self.upper else s

    def clear(self, x: float, y: float, w: float, h: float, pad: float | None = None) -> bool:
        """True if the rectangle is inside the frame and touches no head zone."""
        pad = self.px(22) if pad is None else pad
        if x < pad * 0.3 or y < pad * 0.3 or x + w > self.W - pad * 0.3 or y + h > self.H - pad * 0.3:
            return False
        return all(x + w + pad <= a or x - pad >= c or y + h + pad <= b or y - pad >= d for a, b, c, d in self.avoid)

    def place(self, w: int, h: int, prefer: list[tuple[int, int]], y_max: float = 1.0) -> tuple[int, int] | None:
        """The first preferred top-left position where a w x h card is clear of every face; else the clear spot nearest to the first choice (a scan of the
        frame, not below y_max of the height). None: there is no room for it without covering someone."""
        for x, y in prefer:
            if self.clear(x, y, w, h):
                return int(x), int(y)
        if not self.avoid:
            return None if not prefer else (int(prefer[0][0]), int(prefer[0][1]))
        fx, fy = prefer[0] if prefer else (self.W / 2, self.H / 2)
        cands = [(abs(x - fx) + abs(y - fy), x, y) for x in range(0, max(1, self.W - w + 1), max(8, self.W // 24))
                 for y in range(0, max(1, int(self.H * y_max) - h + 1), max(8, self.H // 20))]
        for _d, x, y in sorted(cands):
            if self.clear(x, y, w, h):
                return int(x), int(y)
        return None


def _wrap(text: str, f: ImageFont.FreeTypeFont, max_w: int, max_lines: int = 6) -> list[str]:
    d = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    lines, cur = [], ""
    for w in str(text).split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" ,.;") + "…"
    return lines


def text_layer(lines: list[str], f: ImageFont.FreeTypeFont, fill, align: str = "left", line_h: float = 1.2, shadow: int = 0, width: int | None = None) -> Image.Image:
    d = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    lh = int(f.size * line_h)
    w = width or max((int(d.textlength(ln, font=f)) for ln in lines), default=1) + 2 * shadow + 4
    img = Image.new("RGBA", (w, lh * max(1, len(lines)) + 2 * shadow + 6 + int(f.size * 0.3)), (0, 0, 0, 0))      # (room under the last line: the tails of g, y, p, q)
    dr = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=f)
        x = {"left": 0, "center": (w - tw) / 2, "right": w - tw}[align] + shadow
        y = i * lh + shadow
        if shadow:
            dr.text((x + shadow * 0.4, y + shadow * 0.8), ln, font=f, fill=(0, 0, 0, 120))
        dr.text((x, y), ln, font=f, fill=(*fill[:3], 255))
    return img


def panel(w: int, h: int, r: int, look: Look, pad: int | None = None) -> tuple[Image.Image, int]:
    """The card body with its shadow around it, in the video's look. -> (layer, margin): the body sits at (margin, margin) inside the layer."""
    m = pad if pad is not None else look.px(40)
    m = max(m, look.px(26) * 3 + look.px(14) + 4) if look.mode not in ("board", "bold") else max(m, look.px(24))      # (room for the whole soft shadow: a Gaussian blur reaches about three sigma, and a smaller margin cuts it off in a hard line)
    img = Image.new("RGBA", (w + 2 * m, h + 2 * m), (0, 0, 0, 0))
    if look.mode == "minimal":                                                       # no card: a soft dark glow behind the type so it reads on any picture
        glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow).rounded_rectangle((m, m, m + w, m + h), radius=r, fill=(0, 0, 0, 120))
        img.alpha_composite(glow.filter(ImageFilter.GaussianBlur(look.px(42))))
        return img, m
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    body = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    def shadow(alpha: int, blur: int, dy: int, dx: int = 0, color=(0, 0, 0)):
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(sh).rounded_rectangle((m + dx, m + dy, m + w + dx, m + h + dy), radius=r, fill=(*color, alpha))
        img.alpha_composite(sh.filter(ImageFilter.GaussianBlur(blur)) if blur else sh)

    def ring(width: int, fn_alpha):
        rg = Image.new("L", (w, h), 0)
        ImageDraw.Draw(rg).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, outline=255, width=width)
        diag = np.add.outer(np.linspace(0, 1, h), np.linspace(0, 1, w)) / 2
        e = np.zeros((h, w, 4), dtype=np.uint8)
        e[..., :3] = 255
        e[..., 3] = fn_alpha(diag).astype(np.uint8)
        ei = Image.fromarray(e, "RGBA")
        ei.putalpha(ImageChops.multiply(ei.getchannel("A"), rg))
        return ei

    def vgrad(top: int, bottom: int, color=(255, 255, 255)):
        g = np.zeros((h, w, 4), dtype=np.uint8)
        g[..., :3] = color
        g[..., 3] = np.linspace(top, bottom, h).astype(np.uint8)[:, None]
        return Image.fromarray(g, "RGBA")

    def highlight(alpha: int):
        sp = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(sp).ellipse((-w * 0.25, -h * 0.9, w * 0.75, h * 0.35), fill=(255, 255, 255, alpha))
        return sp.filter(ImageFilter.GaussianBlur(look.px(70)))
    mode = look.mode
    if mode == "board":
        r = min(r, look.px(16))
        mask = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    if mode == "glass_dark":
        shadow(105, look.px(22), look.px(12))
        body.alpha_composite(Image.new("RGBA", (w, h), (12, 18, 32, 92)))              # a dark glass tint (the picture behind is blurred under it)
        body.alpha_composite(vgrad(58, 14))
        body.alpha_composite(highlight(34))
        body.alpha_composite(ring(max(2, look.px(2.2)), lambda d: 190 - 150 * d))       # the edge catches light top-left, fades bottom-right
    elif mode == "glass_light":
        shadow(54, look.px(26), look.px(14))
        body.alpha_composite(Image.new("RGBA", (w, h), (255, 255, 255, 96)))           # frosted white glass
        body.alpha_composite(vgrad(86, 30))
        body.alpha_composite(highlight(70))
        body.alpha_composite(ring(max(2, look.px(2.4)), lambda d: 235 - 110 * d))
    elif mode == "solid_light":
        shadow(62, look.px(26), look.px(14))
        body.alpha_composite(Image.new("RGBA", (w, h), (*look.solid, 247)))
        body.alpha_composite(ring(max(1, look.px(1.6)), lambda d: np.full_like(d, 26)))
    elif mode == "solid_dark":
        shadow(120, look.px(26), look.px(14))
        body.alpha_composite(Image.new("RGBA", (w, h), (*look.solid, 246)))
        body.alpha_composite(vgrad(14, 0))
        body.alpha_composite(ring(max(1, look.px(1.6)), lambda d: 52 - 30 * d))
    elif mode == "pastel":
        shadow(70, look.px(26), look.px(14), 0, mix(look.accent, (0, 0, 0), 0.3))     # a soft shadow in the colour itself
        body.alpha_composite(Image.new("RGBA", (w, h), (*look.tint, 252)))
        body.alpha_composite(vgrad(120, 0))
        ImageDraw.Draw(body).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, outline=(*look.edge, 255), width=max(3, look.px(3.5)))
    elif mode == "board":
        shadow(255, 0, look.px(9), look.px(9))                                        # a hard offset shadow, like a sticker on a board
        body.alpha_composite(Image.new("RGBA", (w, h), (*look.solid, 255)))
        ImageDraw.Draw(body).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, outline=(255, 255, 255, 255), width=max(3, look.px(4)))
    elif mode == "bold":
        shadow(255, 0, look.px(11), look.px(11))                                      # a hard offset shadow
        body.alpha_composite(Image.new("RGBA", (w, h), (look.accent[0], look.accent[1], look.accent[2], 255)))
        ImageDraw.Draw(body).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, outline=(12, 12, 14, 255), width=max(3, look.px(5)))
    else:                                                                              # mono
        shadow(130, look.px(18), look.px(10))
        body.alpha_composite(Image.new("RGBA", (w, h), (*look.solid, 240)))
        ImageDraw.Draw(body).rectangle((0, 0, look.px(10), h), fill=(*look.deco, 255))
    body.putalpha(ImageChops.multiply(body.getchannel("A"), mask))
    img.alpha_composite(body, (m, m))
    return img, m


# ------------------------------------------------------------------------------------------------ the scenes
class Scene:
    """What a card is made of: layers (RGBA image + position + when it appears), and the glass rectangles to blur behind."""

    def __init__(self, look: Look, seconds: float):
        self.look, self.seconds = look, seconds
        self.layers: list[dict] = []
        self.glass: list[dict] = []
        self.custom = False                                # a full-screen card does its own transition in and out (frame_at then leaves its frames as drawn)
        self.draw_fns: list = []                          # per-frame drawing (charts, counters): fn(t) -> Image | None placed at (x, y)

    def add(self, img: Image.Image, x: int, y: int, at: float = 0.0, rise: float = 0.0, dur: float = ENTER, slide_x: float = 0.0, pop: float = 1.0,
            exit_x: float = 0.0):
        """pop < 1: the layer grows from that scale with an overshoot. exit_x: where it slides off to (pixels) in the last moments."""
        self.layers.append({"img": img, "x": x, "y": y, "at": at, "rise": rise, "dur": dur, "sx": slide_x, "pop": pop, "exit_x": exit_x})

    def dyn(self, fn, x: int, y: int, exit_x: float = 0.0):
        self.draw_fns.append((fn, x, y, exit_x))


def _fit_size(base: int, text: str, f_weight: str, max_w: int, max_lines: int, min_size: int) -> int:
    s = base
    while s > min_size:
        if len(_wrap(text, font(f_weight, s), max_w, 99)) <= max_lines:
            return s
        s = int(s * 0.92)
    return min_size


def scene_lower(look: Look, spec: dict, seconds: float, kind: str = "lower", wf: float = 1.0) -> Scene | None:
    """Bottom-left: a name + role, the title card (the hook) or a key point."""
    sc = Scene(look, seconds)
    hook = kind == "hook"
    main = look.case(" ".join(str(spec.get("text") or spec.get("name") or "").split())[:(90 if hook else 70)])
    sub = " ".join(str(spec.get("role") or "").split())[:60]
    max_w = int(look.W * (0.84 if look.portrait else 0.52) * wf)
    size = _fit_size(look.px(64 if look.portrait else 52) if hook else look.px(52 if look.portrait else 42), main, "bold", max_w - look.px(80), 3 if hook else 2, look.px(26))
    f1 = font("heavy" if hook else "bold", size)
    lines = _wrap(main, f1, max_w - look.px(80), 3)
    t1 = text_layer(lines, f1, look.fg, line_h=1.12, shadow=2)
    parts = [t1]
    if sub:
        t2 = text_layer(_wrap(sub, font("regular", int(size * 0.62)), max_w - look.px(80), 2), font("regular", int(size * 0.62)), look.sub, line_h=1.2, shadow=1)
        parts.append(t2)
    gap = look.px(10)
    tw = max(p.width for p in parts)
    th = sum(p.height for p in parts) + gap * (len(parts) - 1)
    pw, ph = tw + look.px(76), th + look.px(44)
    body, m = panel(pw, ph, min(ph // 2, look.px(36)), look)
    lx = int(look.W * (0.06 if not look.portrait else 0.07))
    ly = int(look.H * (1 - (0.22 if not look.portrait else 0.33))) - ph
    top = int(look.H * 0.08)
    pos = look.place(pw, ph, [(lx, ly), (look.W - pw - lx, ly), (lx, top), (look.W - pw - lx, top)])      # bottom-left, else the first spot no face is in
    if pos is None:
        return None
    x, y = pos[0] - m, pos[1] - m
    right = pos[0] > look.W / 2
    away = (look.W - pos[0] + look.px(40)) if right else -(x + body.width + look.px(40))               # it leaves to the side it came from
    side = look.px(160) if right else -look.px(160)
    sc.add(body, x, y, 0.0, slide_x=side, pop=0.9, dur=0.55, exit_x=away)
    yy = m + look.px(22)
    for k_, p in enumerate(parts):
        sc.add(p, x + m + look.px(38), y + yy, 0.14 + 0.08 * k_, slide_x=side * 0.56, pop=0.95, dur=0.5, exit_x=away)
        yy += p.height + gap
    sc.glass.append({"x": x + m, "y": y + m, "w": pw, "h": ph, "r": min(ph // 2, look.px(36)), "dim": 0.0})
    return sc


def scene_sticker(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """The board look's number: a tilted accent-coloured sticker, black type, that slaps on and counts up (a price tag)."""
    sc = Scene(look, seconds)
    value = str(spec.get("value") or "").strip()[:14]
    label = look.case(" ".join(str(spec.get("label") or "").split())[:60])
    big = font("heavy", int(look.px(190 if look.portrait else 150) * (0.65 + 0.35 * wf)))
    small = font("bold", int(look.px(40 if look.portrait else 34) * (0.8 + 0.2 * wf)))
    ink = (12, 12, 14)
    lab = text_layer(_wrap(label, small, int(look.W * 0.55 * wf), 2), small, ink, align="center", line_h=1.15)
    d = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    pw = int(max(d.textlength(value, font=big), lab.width) + look.px(110))
    ph = big.size + lab.height + look.px(84)
    m = look.px(30)
    angle = -4.0
    base = Image.new("RGBA", (pw + 2 * m, ph + 2 * m), (0, 0, 0, 0))
    r = look.px(14)
    ImageDraw.Draw(base).rounded_rectangle((m + look.px(9), m + look.px(9), m + pw + look.px(9), m + ph + look.px(9)), radius=r, fill=(0, 0, 0, 255))
    ImageDraw.Draw(base).rounded_rectangle((m, m, m + pw, m + ph), radius=r, fill=(*look.accent, 255), outline=ink + (255,), width=max(3, look.px(4)))
    m_ = re.match(r"^([^\d\-]*)(-?[\d.,]+)(.*)$", value)
    expand = base.rotate(angle, expand=True, resample=Image.BICUBIC)
    cx_ = int(look.W * (0.5 if look.portrait else 0.5))
    pos = look.place(expand.width, expand.height, [((look.W - expand.width) // 2, int(look.H * 0.40)), (look.px(40), int(look.H * 0.40)),
                                                    (look.W - expand.width - look.px(40), int(look.H * 0.40))])
    if pos is None:
        return None
    sc.add(expand, pos[0], pos[1], 0.0, pop=0.5, dur=0.55, slide_x=look.px(60))
    sc.glass = []
    cw_, ch_ = expand.size
    txt_w = pw

    def counter(t: float):
        e = ease_out((t - 0.2) / 0.85)
        shown = value
        if m_:
            try:
                raw = m_.group(2).replace(",", "")
                target = float(raw)
                dec = len(raw.split(".")[1]) if "." in raw else 0
                shown = f"{m_.group(1)}{target * e:,.{dec}f}{m_.group(3)}"
            except ValueError:
                pass
        canvas = Image.new("RGBA", (cw_, ch_), (0, 0, 0, 0))
        inner = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        inner.alpha_composite(text_layer([shown], big, ink, align="center", line_h=1.0, width=pw), (0, look.px(20)))
        inner.alpha_composite(lab, ((pw - lab.width) // 2, look.px(20) + big.size + look.px(10)))
        rot = inner.rotate(angle, expand=True, resample=Image.BICUBIC)
        canvas.alpha_composite(rot, ((cw_ - rot.width) // 2, (ch_ - rot.height) // 2))
        return canvas
    sc.dyn(counter, pos[0], pos[1])
    return sc


def scene_stat(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    if look.mode == "board":
        return scene_sticker(look, spec, seconds, wf)
    sc = Scene(look, seconds)
    value = str(spec.get("value") or "").strip()[:14]
    label = " ".join(str(spec.get("label") or "").split())[:80]
    pw = int(look.W * (0.62 if not look.portrait else 0.80) * wf)
    big = font("heavy", look.px(210 if look.portrait else 170) if wf > 0.9 else int(look.px(170) * wf))
    while big.size > look.px(80) and ImageDraw.Draw(Image.new("RGBA", (4, 4))).textlength(value, font=big) > pw - look.px(90):
        big = font("heavy", int(big.size * 0.9))
    small = font("medium", look.px(46 if look.portrait else 38))
    lab_lines = _wrap(look.case(label), small, pw - look.px(90), 3)
    lab = text_layer(lab_lines, small, look.sub, align="center", line_h=1.2, shadow=1, width=pw - look.px(60))
    ink = text_layer([value], big, look.fg, align="center", line_h=1.0, shadow=3, width=pw - look.px(40)).getbbox()   # where the number's ink ends
    under = max(big.size + look.px(16), (ink[3] if ink else big.size) + look.px(14))                                   # (any font: the bar clears it)
    ph = look.px(30) + under + look.px(26) + lab.height + look.px(38)
    body, m = panel(pw, ph, look.px(54), look)
    cy_ = int(look.H * (0.50 if look.portrait else 0.46)) - ph // 2
    pos = look.place(pw, ph, [((look.W - pw) // 2, cy_), (look.px(40), cy_), (look.W - pw - look.px(40), cy_)])
    if pos is None:
        return None
    x, y = pos[0] - m, pos[1] - m
    sc.add(body, x, y, 0.0, pop=0.84, dur=0.6)
    m_ = re.match(r"^([^\d\-]*)(-?[\d.,]+)(.*)$", value)
    cnt_w = pw - look.px(40)

    def counter(t: float):
        e = ease_out((t - 0.2) / 0.85)                                              # a quick climb from zero
        if m_:
            raw = m_.group(2).replace(",", "")
            try:
                target = float(raw)
            except ValueError:
                target = None
        else:
            target = None
        if target is None:
            shown = value
        else:
            cur = target * e
            dec = len(raw.split(".")[1]) if "." in raw else 0
            shown = f"{m_.group(1)}{cur:,.{dec}f}{m_.group(3)}"
        img = text_layer([shown], big, look.fg, align="center", line_h=1.0, shadow=3, width=cnt_w)
        pulse = 1.0 + 0.09 * math.sin(math.pi * min(1.0, max(0.0, (t - 1.0) / 0.35)))        # a small beat when it lands
        if pulse > 1.004:
            big_img = _scaled(img, pulse)
            canvas_ = Image.new("RGBA", img.size, (0, 0, 0, 0))
            canvas_.alpha_composite(big_img.crop(((big_img.width - img.width) // 2, (big_img.height - img.height) // 2,
                                                  (big_img.width - img.width) // 2 + img.width, (big_img.height - img.height) // 2 + img.height)))
            return canvas_
        return img
    sc.dyn(counter, x + m + look.px(20), y + m + look.px(30))
    acc = Image.new("RGBA", (look.px(120), look.px(8)), (*look.deco, 255))
    sc.add(acc, x + m + (pw - acc.width) // 2, y + m + look.px(30) + under, 0.9, pop=0.2, dur=0.45)
    sc.add(lab, x + m + look.px(30), y + m + look.px(30) + under + look.px(26), 0.5, rise=look.px(18), dur=0.5)
    sc.glass.append({"x": x + m, "y": y + m, "w": pw, "h": ph, "r": look.px(54), "dim": 0.0})
    return sc


def scene_quote(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    sc = Scene(look, seconds)
    text = " ".join(str(spec.get("text") or "").split())[:160]
    who = " ".join(str(spec.get("who") or "").split())[:50]
    pw = int(look.W * (0.70 if not look.portrait else 0.86) * wf)
    size = _fit_size(look.px(74 if look.portrait else 60), text, "bold", pw - look.px(130), 5, look.px(34))
    f = font("bold", size)
    lines = _wrap(text, f, pw - look.px(130), 6)
    tl = text_layer(lines, f, look.fg, line_h=1.18, shadow=2)
    parts = [tl]
    if who:
        parts.append(text_layer([look.case("— " + who)], font("medium", int(size * 0.55)), look.deco, line_h=1.2, shadow=1))
    qf = font("heavy", look.px(170))
    ph = sum(p.height for p in parts) + look.px(190) + (look.px(18) if who else 0)
    body, m = panel(pw, ph, look.px(48), look)
    cy_ = int(look.H * (0.52 if look.portrait else 0.5)) - ph // 2
    pos = look.place(pw, ph, [((look.W - pw) // 2, cy_), (look.px(40), cy_), (look.W - pw - look.px(40), cy_)])
    if pos is None:
        return None
    x, y = pos[0] - m, pos[1] - m
    sc.add(body, x, y, 0.0, pop=0.88, dur=0.55)
    sc.glass.append({"x": 0, "y": 0, "w": look.W, "h": look.H, "r": 0, "dim": 0.28})                  # the whole picture goes soft behind the quote, so the card stands apart from the video
    q = text_layer(["“"], qf, look.accent, line_h=1.0)
    sc.add(q, x + m + look.px(40), y + m - look.px(6), 0.15, pop=0.3, dur=0.5)
    yy = look.px(128)
    for p in parts:
        sc.add(p, x + m + look.px(66), y + m + yy, 0.3, rise=look.px(22), dur=0.55)
        yy += p.height + look.px(18)
    sc.glass.append({"x": x + m, "y": y + m, "w": pw, "h": ph, "r": look.px(48), "dim": 0.0})
    return sc


def scene_slide(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A PowerPoint-style slide: a title and up to five bullets that appear one after another, on a blurred, dimmed picture."""
    sc = Scene(look, seconds)
    title = look.case(" ".join(str(spec.get("title") or "").split())[:70])
    bullets = [" ".join(str(b).split())[:80] for b in (spec.get("bullets") or [])][:5]
    pw = int(look.W * (0.74 if not look.portrait else 0.88) * wf)
    tsize = _fit_size(look.px(80 if look.portrait else 64), title, "heavy", pw - look.px(120), 2, look.px(36))
    tf = font("heavy", tsize)
    tl = text_layer(_wrap(title, tf, pw - look.px(120), 2), tf, look.fg, line_h=1.1, shadow=2)
    bsize = _fit_size(look.px(54 if look.portrait else 42), max(bullets, key=len) if bullets else "", "medium", pw - look.px(220), 2, look.px(28))
    bf = font("medium", bsize)
    items = []
    for b in bullets:
        items.append(text_layer(_wrap(b, bf, pw - look.px(220), 2), bf, look.fg, line_h=1.18, shadow=1))
    gap = look.px(26)
    ph = tl.height + look.px(48) + sum(i.height for i in items) + gap * max(0, len(items) - 1) + look.px(100)
    ph = min(ph, int(look.H * 0.78))
    body, m = panel(pw, ph, look.px(50), look)
    pos = look.place(pw, ph, [((look.W - pw) // 2, (look.H - ph) // 2), (look.px(40), (look.H - ph) // 2), (look.W - pw - look.px(40), (look.H - ph) // 2)])
    if pos is None:
        return None
    x, y = pos[0] - m, pos[1] - m
    sc.add(body, x, y, 0.05, pop=0.9, dur=0.6)
    sc.add(tl, x + m + look.px(60), y + m + look.px(46), 0.25, rise=look.px(22), dur=0.5)
    bar = Image.new("RGBA", (look.px(110), look.px(7)), (*look.deco, 255))
    sc.add(bar, x + m + look.px(62), y + m + look.px(46) + tl.height + look.px(10), 0.4, pop=0.1, dur=0.5)
    yy = look.px(46) + tl.height + look.px(48)
    span = max(0.5, min(0.7, (seconds - 1.6) / max(1, len(items))))
    for k, it in enumerate(items):
        dot = Image.new("RGBA", (look.px(22), look.px(22)), (0, 0, 0, 0))
        ImageDraw.Draw(dot).ellipse((0, 0, dot.width - 1, dot.height - 1), fill=(*look.deco, 255))
        at = 0.65 + k * span
        sc.add(dot, x + m + look.px(62), y + m + yy + int(bsize * 0.34), at, pop=0.1, dur=0.45)
        sc.add(it, x + m + look.px(110), y + m + yy, at, slide_x=-look.px(70), pop=0.96, dur=0.5)
        yy += it.height + gap
    if not look.avoid:                                                                              # (never blur or darken a picture with someone in it)
        sc.glass.append({"x": 0, "y": 0, "w": look.W, "h": look.H, "r": 0, "dim": 0.32})             # the whole picture goes soft and dark behind the slide
    sc.glass.append({"x": x + m, "y": y + m, "w": pw, "h": ph, "r": look.px(50), "dim": 0.0})
    return sc


PALETTE_DEFAULT = [(94, 230, 200), (255, 210, 63), (255, 111, 97), (122, 162, 255), (186, 140, 255), (120, 220, 120)]


PALETTE = PALETTE_DEFAULT


def scene_chart(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    sc = Scene(look, seconds)
    ch = spec.get("chart") or spec
    kind = str(ch.get("type") or "bar").lower()
    labels = [" ".join(str(x).split())[:22] for x in (ch.get("labels") or [])][:7]
    values = []
    for v in (ch.get("values") or [])[:7]:
        try:
            values.append(float(str(v).replace(",", "").replace("%", "").replace("$", "")))
        except ValueError:
            values.append(0.0)
    n = min(len(labels), len(values))
    labels, values = labels[:n], values[:n]
    unit = str(ch.get("unit") or "")[:6]
    title = look.case(" ".join(str(ch.get("title") or spec.get("title") or "").split())[:60])
    pw = int(look.W * (0.78 if not look.portrait else 0.90) * wf)
    ph = int(look.H * (0.52 if look.portrait else 0.70) * (0.5 + 0.5 * wf))
    body, m = panel(pw, ph, look.px(50), look)
    pos = look.place(pw, ph, [((look.W - pw) // 2, (look.H - ph) // 2), (look.px(40), (look.H - ph) // 2), (look.W - pw - look.px(40), (look.H - ph) // 2)])
    if pos is None:
        return None
    x, y = pos[0] - m, pos[1] - m
    sc.add(body, x, y, 0.05, pop=0.9, dur=0.6)
    tf = font("heavy", _fit_size(look.px(60 if look.portrait else 48), title, "heavy", pw - look.px(120), 2, look.px(30)))
    tl = text_layer(_wrap(title, tf, pw - look.px(120), 2), tf, look.fg, line_h=1.1, shadow=2)
    sc.add(tl, x + m + look.px(60), y + m + look.px(40), 0.25, rise=look.px(22), dur=0.5)
    cx0, cy0 = x + m + look.px(60), y + m + look.px(40) + tl.height + look.px(40)
    cw, chh = pw - look.px(120), ph - (cy0 - y - m) - look.px(60)
    lf = font("medium", look.px(34 if look.portrait else 28))
    vf = font("bold", look.px(38 if look.portrait else 32))
    top = max(values) if values else 1.0
    top = (top if top > 0 else 1.0) * 1.22                                     # headroom: a bar overshoots its height when it pops

    sym = unit if unit in ("%", "$", "€", "£", "x", "K", "M") else ""                         # (a unit WORD like "views" belongs in the title, not on every bar)

    def fmt(v: float) -> str:
        s = f"{v:,.1f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{v:,.0f}"
        return (s + sym) if sym else s

    def compact(v: float) -> str:
        if abs(v) >= 1e6:
            return f"{v / 1e6:.1f}".rstrip("0").rstrip(".") + "M" + sym
        if abs(v) >= 1e3:
            return f"{v / 1e3:.1f}".rstrip("0").rstrip(".") + "K" + sym
        return fmt(v)

    def frame(t: float):
        img = Image.new("RGBA", (cw, chh), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        if n == 0:
            return img
        if kind == "donut":
            total = sum(max(v, 0) for v in values) or 1
            size = min(cw // 2, chh) - look.px(10)
            ss = 3
            open_ = ease_out((t - 0.35) / 1.15)                                     # the ring opens up: it sweeps round while it turns into place
            pop = 0.55 + 0.45 * ease_back((t - 0.3) / 0.8)
            ring = Image.new("RGBA", (size * ss, size * ss), (0, 0, 0, 0))
            rd = ImageDraw.Draw(ring)
            start = -90.0 - (1 - open_) * 110.0
            for i, v in enumerate(values):
                span = 360.0 * max(v, 0) / total
                sweep = span * open_
                if sweep > 0.3:
                    rd.pieslice((0, 0, size * ss - 1, size * ss - 1), start, start + sweep - 0.8, fill=(*look.palette[i % len(look.palette)], 255))
                start += sweep
            hole = Image.new("L", ring.size, 0)
            inner = 0.34 - 0.14 * (1 - open_)
            ImageDraw.Draw(hole).ellipse((size * ss * (0.5 - inner), size * ss * (0.5 - inner), size * ss * (0.5 + inner), size * ss * (0.5 + inner)), fill=255)
            ring.putalpha(ImageChops.multiply(ring.getchannel("A"), ImageChops.invert(hole)))
            ring = ring.resize((int(size * pop), int(size * pop)), Image.LANCZOS)
            img.alpha_composite(ring, (look.px(10) + (size - ring.width) // 2, (chh - ring.height) // 2))
            ly = max(0, (chh - n * look.px(62)) // 2)
            for i, (lb, v) in enumerate(zip(labels, values)):
                a = ease_out((t - 0.9 - i * 0.14) / 0.35)
                yy = ly + i * look.px(62) + int((1 - a) * look.px(18))
                dd = int(look.px(26) * ease_back((t - 0.9 - i * 0.14) / 0.35))
                d.ellipse((size + look.px(60), yy + look.px(23) - dd // 2, size + look.px(60) + dd, yy + look.px(23) + dd // 2), fill=(*look.palette[i % len(look.palette)], int(255 * a)))
                d.text((size + look.px(104), yy), f"{lb}  {fmt(v)}", font=lf, fill=(*look.fg, int(255 * min(1.0, a))))
            return img
        if kind == "line":
            pad_b = look.px(60)
            pts = []
            for i, v in enumerate(values):
                px_ = int(i * (cw - look.px(40)) / max(1, n - 1)) + look.px(20)
                py_ = int((chh - pad_b) - (v / top) * (chh - pad_b - look.px(50)))
                pts.append((px_, py_))
            prog = ease_out((t - 0.45) / 1.3) * (n - 1)
            drawn = []
            for i in range(n - 1):
                if prog >= i:
                    f_ = min(1.0, prog - i)
                    drawn.append(pts[i])
                    if f_ < 1.0:
                        drawn.append((pts[i][0] + (pts[i + 1][0] - pts[i][0]) * f_, pts[i][1] + (pts[i + 1][1] - pts[i][1]) * f_))
                        break
                    drawn.append(pts[i + 1])
            if len(drawn) >= 2:
                d.line(drawn, fill=(*look.deco, 255), width=look.px(9), joint="curve")
            for i, (px_, py_) in enumerate(pts):
                if prog >= i - 0.01:
                    d.ellipse((px_ - look.px(12), py_ - look.px(12), px_ + look.px(12), py_ + look.px(12)), fill=(*look.fg, 255), outline=(*look.deco, 255), width=look.px(5))
                    vt = fmt(values[i])
                    d.text((px_ - d.textlength(vt, font=vf) / 2, py_ - look.px(56)), vt, font=vf, fill=(*look.fg, 255))
                lt = labels[i]
                d.text((px_ - d.textlength(lt, font=lf) / 2, chh - look.px(44)), lt, font=lf, fill=(*look.sub, 255))
            return img
        horizontal = kind == "bar" and (look.portrait or n > 4)
        if horizontal:                                                    # horizontal bars (labels on the left)
            lw = int(min(cw * 0.34, max(d.textlength(s, font=lf) for s in labels) + look.px(20)))
            row = chh // n
            for i, (lb, v) in enumerate(zip(labels, values)):
                a = ease_back((t - 0.4 - i * 0.14) / 0.75, 1.5)
                bh = int(row * 0.56)
                yy = i * row + (row - bh) // 2
                d.text((0, yy + (bh - lf.size) // 2 - 2), lb, font=lf, fill=(*look.fg, 255))
                bw = max(0, int((cw - lw - look.px(150)) * (max(v, 0) / top) * a))
                d.rounded_rectangle((lw, yy, lw + max(bw, 2), yy + bh), radius=bh // 2, fill=(*look.palette[i % len(look.palette)], 255))
                if a > 0.6:
                    d.text((lw + bw + look.px(14), yy + (bh - vf.size) // 2 - 2), fmt(v), font=vf, fill=(*look.fg, int(255 * min(1, (a - 0.6) / 0.4))))
            return img
        col = cw // n
        base_y = chh - look.px(56)
        for i, (lb, v) in enumerate(zip(labels, values)):                 # vertical columns
            a = ease_back((t - 0.4 - i * 0.14) / 0.75, 1.5)                              # each column shoots up past its height and settles
            bw = int(col * 0.58)
            bh = max(0, int((base_y - look.px(60)) * (max(v, 0) / top) * a))
            xx = i * col + (col - bw) // 2
            d.rounded_rectangle((xx, base_y - bh, xx + bw, base_y), radius=look.px(16), fill=(*look.palette[i % len(look.palette)], 255))
            if a > 0.6:
                vt = fmt(v)
                if d.textlength(vt, font=vf) > col * 0.96:
                    vt = compact(v)
                d.text((xx + (bw - d.textlength(vt, font=vf)) / 2, base_y - bh - look.px(50)), vt, font=vf, fill=(*look.fg, int(255 * min(1, (a - 0.6) / 0.4))))
            lbt = lb if d.textlength(lb, font=lf) <= col * 0.98 else lb[:3]
            d.text((xx + (bw - d.textlength(lbt, font=lf)) / 2, base_y + look.px(10)), lbt, font=lf, fill=(*look.sub, 255))
        return img
    sc.dyn(frame, cx0, cy0)
    if not look.avoid:                                                                              # (never blur or darken a picture with someone in it)
        sc.glass.append({"x": 0, "y": 0, "w": look.W, "h": look.H, "r": 0, "dim": 0.32})
    sc.glass.append({"x": x + m, "y": y + m, "w": pw, "h": ph, "r": look.px(50), "dim": 0.0})
    return sc


def scene_banner(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """The compact form of a quote, slide or chart, for a tall frame where a face and the captions leave only a strip: a wide banner of text in the top band."""
    sc = Scene(look, seconds)
    kind = spec.get("kind")
    if kind == "quote":
        head, body = "", f"“{' '.join(str(spec.get('text') or '').split())[:160]}”" + (f"  — {spec.get('who')}" if spec.get("who") else "")
    elif kind == "slide":
        head, body = " ".join(str(spec.get("title") or "").split())[:60], "  ·  ".join(" ".join(str(b).split())[:40] for b in (spec.get("bullets") or [])[:4])
    else:
        ch = spec.get("chart") or {}
        head, body = " ".join(str(ch.get("title") or spec.get("title") or "").split())[:60], "  ·  ".join(f"{a} {b}" for a, b in zip(ch.get("labels") or [], ch.get("values") or []))[:140]
    if not (head or body):
        return None
    pos, pw, ph, parts = None, 0, 0, []
    for frac in ((0.92, 0.7) if look.portrait else (0.7, 0.5, 0.38, 0.3, 0.25)):             # as wide as fits: the strip above a face, else the column beside it
        pw = int(look.W * frac)
        inner = pw - look.px(70)
        hf, bf = font("heavy", look.px(40 if look.portrait else 36)), font("medium", look.px(36 if look.portrait else 32))
        parts = []
        if head:
            parts.append(text_layer(_wrap(look.case(head), hf, inner, 2), hf, look.accent if look.mode in ("board", "bold") else look.fg, line_h=1.1, shadow=1))
        if body:
            parts.append(text_layer(_wrap(body, bf, inner, 8), bf, look.fg, line_h=1.15, shadow=1))
        ph = sum(p.height for p in parts) + look.px(14) * (len(parts) - 1) + look.px(40)
        body_img, m = panel(pw, ph, look.px(28), look)
        top_y = int(look.H * 0.05)
        pos = look.place(pw, ph, [((look.W - pw) // 2, top_y), (look.px(30), top_y), (look.W - pw - look.px(30), top_y), (look.px(30), int(look.H * 0.3)),
                                  (look.W - pw - look.px(30), int(look.H * 0.3))], y_max=0.9)
        if pos is not None:
            break
    if pos is None:
        return None
    x, y = pos[0] - m, pos[1] - m
    sc.add(body_img, x, y, 0.0, rise=-look.px(30), pop=0.92, dur=0.5)
    yy = m + look.px(20)
    for k_, p in enumerate(parts):
        sc.add(p, x + m + look.px(35), y + yy, 0.12 + 0.08 * k_, rise=-look.px(14), dur=0.45)
        yy += p.height + look.px(14)
    sc.glass.append({"x": x + m, "y": y + m, "w": pw, "h": ph, "r": look.px(28), "dim": 0.0})
    return sc


def build_scene(kind: str, look: Look, spec: dict, seconds: float) -> Scene | None:
    """A finished scene: only glass looks get blurred panels (flat looks keep just the full-frame veil behind slides and charts), each tagged light / dark."""
    sc = None
    _CTX.mono = look.mono
    try:
        for wf in ((1.0, 0.84, 0.7, 0.58, 0.48, 0.38, 0.3) if look.avoid else (1.0,)):             # someone is in the picture: smaller until it fits beside them
            sc = SCENES[kind](look, spec, seconds, wf)
            if sc is not None:
                break
        if sc is None and look.avoid and kind in ("quote", "slide", "chart"):                    # no room for the full card: the same words as a banner in the top strip
            sc = scene_banner(look, {**spec, "kind": kind}, seconds)
    finally:
        _CTX.mono = False
    if sc is None:
        return None
    sc.glass = [{**g, "tone": look.tone} for g in sc.glass if look.glass or g["dim"]]
    return sc


def _tint_mark(path: str, color: tuple[int, int, int]) -> Image.Image:
    mark = Image.open(path).convert("RGBA")
    out = Image.new("RGBA", mark.size, (*color, 0))
    out.putalpha(mark.getchannel("A"))
    return out


def _lum(c: tuple[int, int, int]) -> float:
    return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255


def _cover(img: Image.Image, w: int, h: int) -> Image.Image:
    k = max(w / img.width, h / img.height)
    img = img.resize((max(w, int(img.width * k)), max(h, int(img.height * k))), Image.LANCZOS)
    return img.crop(((img.width - w) // 2, (img.height - h) // 2, (img.width - w) // 2 + w, (img.height - h) // 2 + h))


def _rounded(img: Image.Image, r: int) -> Image.Image:
    m = Image.new("L", img.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, img.width - 1, img.height - 1), radius=r, fill=255)
    out = img.convert("RGBA")
    out.putalpha(ImageChops.multiply(out.getchannel("A"), m))
    return out


def _tile(look: Look, it: dict, th: int, pad: int | None = None) -> tuple[Image.Image, int, int, int] | None:
    """One picture of a named thing on its own card: a logo mark, a flag or a photo, with its name underneath. -> (layer, card width, card height, margin)."""
    pad = look.px(22) if pad is None else pad
    try:
        img = Image.open(it["file"]).convert("RGBA")
    except (OSError, KeyError):
        return None
    w = th if it.get("type") == "logo" else int(th * (1.5 if it.get("type") == "flag" else 1.38))
    lf = font("bold", max(look.px(20), int(th * 0.15)))
    label = look.case(" ".join(str(it.get("name") or "").split())[:26])
    lab = text_layer([label], lf, look.fg, align="center", line_h=1.1, shadow=1, width=w + 2 * pad - 10)
    ph = th + 2 * pad + lab.height + look.px(10)
    body, m = panel(w + 2 * pad, ph, look.px(34), look)
    tile = body.copy()
    if it.get("type") == "logo":
        if it.get("mark"):
            hexc = rgb(it.get("color") or "#ffffff")
            lum = _lum(hexc)
            if look.mode == "pastel":
                mark_col = look.fg                                                       # one colour only: the mark takes the card's ink
            else:
                mark_col = hexc if (lum > 0.25 if not look.light else lum < 0.72) else (look.fg if look.light else (255, 255, 255))
            mark = _tint_mark(it["file"], mark_col)
            mark.thumbnail((int(w * 0.72), int(th * 0.72)), Image.LANCZOS)
            tile.alpha_composite(mark, (m + pad + (w - mark.width) // 2, m + pad + (th - mark.height) // 2))
        else:                                                                           # a picture logo: on a white tile so its own colours read
            box = _rounded(Image.new("RGBA", (w, th), (250, 250, 252, 255)), look.px(24))
            img.thumbnail((int(w * 0.8), int(th * 0.8)), Image.LANCZOS)
            box.alpha_composite(img, ((w - img.width) // 2, (th - img.height) // 2))
            tile.alpha_composite(box, (m + pad, m + pad))
    else:
        pic = _rounded(_cover(img, w, th), look.px(22 if it.get("type") == "photo" else 12))
        if it.get("type") == "flag":
            ImageDraw.Draw(pic).rounded_rectangle((0, 0, pic.width - 1, pic.height - 1), radius=look.px(12), outline=(255, 255, 255, 90), width=2)
        tile.alpha_composite(pic, (m + pad, m + pad))
    tile.alpha_composite(lab, (m + 5, m + pad + th + look.px(8)))                       # (the label box is as wide as the card, so it starts at the card's left edge)
    return tile, w + 2 * pad, ph, m


def _badge(look: Look, text: str, size: int) -> Image.Image:
    """A round badge ("VS") in the card's accent colour."""
    sc_ = 3
    d = size * sc_
    img = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    ImageDraw.Draw(img).ellipse((0, 0, d - 1, d - 1), fill=(*look.accent, 255), outline=(255, 255, 255, 255) if look.mode == "board" else None, width=sc_ * 3)
    img = img.resize((size, size), Image.LANCZOS)
    f = font("heavy", int(size * 0.42))
    ink = (12, 12, 14) if _lum(look.accent) > 0.55 else (255, 255, 255)
    t = text_layer([text], f, ink, align="center", line_h=1.0, width=size)
    img.alpha_composite(t, (0, (size - int(f.size * 1.0)) // 2 - 2))
    return img


def scene_entity(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """The real picture of something being talked about, popping up as it is named: a flag, a company's logo, a photo of a person or place.
    Several names said in a row each get a tile of their own that pops in at its word (a logo wall); joiner "vs" puts a VS badge between two. Top of the
    frame, clear of the speaker's face and of the captions."""
    sc = Scene(look, seconds)
    items = list(spec.get("items") or [])[:6]
    if not items:
        return sc
    n = len(items)
    vs = spec.get("joiner") == "vs" and n == 2
    th = int(look.px(300 if not look.portrait else 250) * wf)
    gap = look.px(26) if not vs else look.px(110)
    pad = look.px(22)

    def width(it):
        return th if it.get("type") == "logo" else int(th * (1.5 if it.get("type") == "flag" else 1.38))
    maxw = int(look.W * (0.90 if look.portrait else 0.80))
    while th > look.px(120) and sum(width(i) + 2 * pad for i in items) + gap * (n - 1) > maxw:
        th = int(th * 0.92)
    tiles = []
    for it in items:
        t = _tile(look, it, th, pad)
        if t:
            tiles.append((it, *t))
    if not tiles:
        return sc
    total = sum(t[2] for t in tiles) + gap * (len(tiles) - 1)
    row_h = max(t[3] for t in tiles)
    y0 = int(look.H * (0.075 if look.portrait else 0.085))
    right, centre, left = look.W - total - int(look.W * 0.06), (look.W - total) // 2, int(look.W * 0.06)
    order = [right, centre, left] if (len(tiles) == 1 and not look.portrait) else [centre, right, left]       # one picture: at the right, over the empty side
    pos = look.place(total, row_h, [(x_, y0) for x_ in order], y_max=0.6)                                  # (never down where the captions are)
    if pos is None and look.avoid and not vs:                                                              # no room for a row beside a big face: one tile above the next, in a side column
        gap2 = look.px(16)
        th2 = th
        while th2 >= look.px(70):
            col = [(it, *t) for it in items if (t := _tile(look, it, th2, pad))]
            if not col:
                return None
            cw_, chh_ = max(c[2] for c in col), sum(c[3] for c in col) + gap2 * (len(col) - 1)
            cpos = look.place(cw_, chh_, [(int(look.W * 0.04), y0), (look.W - cw_ - int(look.W * 0.04), y0)], y_max=0.66)
            if cpos is not None:
                yy = cpos[1]
                for it, tile, bw, bh, m in col:
                    at = max(0.0, float(it.get("at_rel") or 0.0))
                    xx = cpos[0] + (cw_ - bw) // 2
                    sc.add(tile, xx - m, yy - m, at, pop=0.45, dur=0.6)
                    sc.glass.append({"x": xx, "y": yy, "w": bw, "h": bh, "r": look.px(34), "dim": 0.0, "t0": at})
                    yy += bh + gap2
                return sc
            th2 = int(th2 * 0.9)
        return None
    if pos is None:
        return None
    x0, y0 = pos
    x = x0
    for k_, (it, tile, bw, bh, m) in enumerate(tiles):
        at = max(0.0, float(it.get("at_rel") or 0.0))
        sc.add(tile, x - m, y0 - m, at, pop=0.45, dur=0.6)
        sc.glass.append({"x": x, "y": y0, "w": bw, "h": bh, "r": look.px(34), "dim": 0.0, "t0": at})
        if vs and k_ == 0:
            bsz = look.px(96)
            at2 = max(0.0, float(tiles[1][0].get("at_rel") or 0.0))
            sc.add(_badge(look, "VS", bsz), x + bw + (gap - bsz) // 2, y0 + (row_h - bsz) // 2, at2, pop=0.2, dur=0.55)
        x += bw + gap
    return sc


def scene_hub(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A hub-and-spoke diagram: the main thing in the middle, the things connected to it around it on lines that draw out to each as it is named."""
    sc = Scene(look, seconds)
    items = list(spec.get("items") or [])[:6]
    if len(items) < 3:
        return None
    centre, nodes = items[0], items[1:]
    rw = max(int(look.W * (0.60 if not look.portrait else 0.90) * wf), int(look.W * (0.26 if not look.portrait else 0.6)))                 # (never smaller than a column the tiles can still fit in)
    rh = max(int(look.H * (0.78 if not look.portrait else 0.46) * wf), int(look.H * (0.52 if not look.portrait else 0.3)))
    pos = look.place(rw, rh, [((look.W - rw) // 2, (look.H - rh) // 2), (look.W - rw - look.px(30), (look.H - rh) // 2), (look.px(30), (look.H - rh) // 2)])
    if pos is None:
        return None
    cx, cy = pos[0] + rw // 2, pos[1] + rh // 2
    k = len(nodes)
    th = int(look.px(170 if not look.portrait else 140) * wf)
    gap = look.px(14)
    placed, big = [], None
    for _try in range(14):                                                                 # the tiles shrink until none touches another (never a pile of overlapping logos)
        big = _tile(look, centre, int(th * 1.35))
        if big is None:
            return None
        placed = []
        for i, it in enumerate(nodes):
            ang = -math.pi / 2 + 2 * math.pi * i / k + (0.0 if k % 2 else math.pi / k)        # evenly round the centre
            t = _tile(look, it, th)
            if t is None:
                continue
            placed.append((it, t, cx + int(math.cos(ang) * (rw / 2 - t[1] / 2)), cy + int(math.sin(ang) * (rh / 2 - t[2] / 2))))
        boxes = [(cx - big[1] / 2, cy - big[2] / 2, cx + big[1] / 2, cy + big[2] / 2)] + [(nx - t[1] / 2, ny - t[2] / 2, nx + t[1] / 2, ny + t[2] / 2) for _i, t, nx, ny in placed]
        clash = any(boxes[i][0] - gap < boxes[j][2] and boxes[j][0] - gap < boxes[i][2] and boxes[i][1] - gap < boxes[j][3] and boxes[j][1] - gap < boxes[i][3]
                    for i in range(len(boxes)) for j in range(i + 1, len(boxes)))
        if not clash or th < look.px(60):
            break
        th = int(th * 0.9)
    if clash:
        return None
    if not placed:
        return None
    lines_w = max(3, look.px(5))

    def lines(t: float):
        img = Image.new("RGBA", (look.W, look.H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for it, tl, nx, ny in placed:
            at = float(it.get("at_rel") or 0.0)
            p = ease_out((t - at - 0.15) / 0.6)
            if p <= 0:
                continue
            dx, dy = nx - cx, ny - cy
            dist = math.hypot(dx, dy) or 1.0
            start = (cx + dx / dist * big[2] * 0.5, cy + dy / dist * big[3] * 0.5)
            end = (nx - dx / dist * tl[1] * 0.42, ny - dy / dist * tl[2] * 0.42)
            e = (start[0] + (end[0] - start[0]) * p, start[1] + (end[1] - start[1]) * p)
            d.line([start, e], fill=(*look.deco, 255), width=lines_w)
        return img
    sc.dyn(lines, 0, 0)
    sc.add(big[0], cx - big[1] // 2 - big[3], cy - big[2] // 2 - big[3], 0.0, pop=0.4, dur=0.6)
    sc.glass.append({"x": cx - big[1] // 2, "y": cy - big[2] // 2, "w": big[1], "h": big[2], "r": look.px(34), "dim": 0.0, "t0": 0.0})
    for it, tl, nx, ny in placed:
        at = max(0.0, float(it.get("at_rel") or 0.0))
        sc.add(tl[0], nx - tl[1] // 2 - tl[3], ny - tl[2] // 2 - tl[3], at + 0.2, pop=0.4, dur=0.6)
        sc.glass.append({"x": nx - tl[1] // 2, "y": ny - tl[2] // 2, "w": tl[1], "h": tl[2], "r": look.px(34), "dim": 0.0, "t0": at + 0.2})
    return sc


def scene_chips(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """Short facts as a stack of pills (spec lists: "M5 PRO", "15-CORE CPU"), popping in one after another."""
    sc = Scene(look, seconds)
    chips = [" ".join(str(c).split())[:34] for c in (spec.get("chips") or spec.get("bullets") or [])][:6]
    if not chips:
        return None
    f = font("bold", int(look.px(40 if not look.portrait else 38) * (0.7 + 0.3 * wf)))
    pad_x, pad_y = look.px(34), look.px(16)
    texts = [text_layer([look.case(c)], f, look.fg, align="center", line_h=1.05, shadow=1) for c in chips]
    cw = max(t.width for t in texts) + 2 * pad_x
    ch = texts[0].height + 2 * pad_y
    gap = look.px(16)
    total_h = len(chips) * ch + (len(chips) - 1) * gap
    prefs = [(look.W - cw - int(look.W * 0.06), int(look.H * 0.14)), (int(look.W * 0.06), int(look.H * 0.14)), ((look.W - cw) // 2, int(look.H * 0.10))]
    pos = look.place(cw, total_h, prefs, y_max=0.72)
    if pos is None:
        return None
    right = pos[0] > look.W / 2
    for i, (c, t) in enumerate(zip(chips, texts)):
        body, m = panel(cw, ch, ch // 2, look, pad=look.px(26))
        yy = pos[1] + i * (ch + gap)
        at = 0.15 + 0.38 * i
        sc.add(body, pos[0] - m, yy - m, at, slide_x=look.px(120) if right else -look.px(120), pop=0.6, dur=0.5)
        sc.add(t, pos[0] + (cw - t.width) // 2, yy + (ch - t.height) // 2 + 1, at + 0.08, pop=0.8, dur=0.4, slide_x=look.px(120) if right else -look.px(120))
        sc.glass.append({"x": pos[0], "y": yy, "w": cw, "h": ch, "r": ch // 2, "dim": 0.0, "t0": at})
    return sc


def _money(v: float, unit: str, prefix: str) -> str:
    s = f"{abs(v):,.2f}".rstrip("0").rstrip(".") if abs(v) < 1000 else f"{abs(v):,.0f}"
    return f"{'-' if v < 0 else ''}{prefix}{s}{unit}"


def scene_compare(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """Two numbers set against each other on a line - a price range, before / after - with the difference landing in the middle."""
    sc = Scene(look, seconds)
    labels = [" ".join(str(x).split())[:18] for x in (spec.get("labels") or [])][:2]
    raw = [str(v) for v in (spec.get("values") or [])][:2]
    if len(labels) < 2 or len(raw) < 2:
        return None
    pre = "$" if any("$" in r for r in raw) else ""
    unit = "%" if any("%" in r for r in raw) else str(spec.get("unit") or "")
    nums = []
    for r in raw:
        try:
            nums.append(float(re.sub(r"[^\d.\-]", "", r)))
        except ValueError:
            nums.append(None)
    shown = [r.strip() for r in raw]
    delta = spec.get("note") or (("+" if nums[1] - nums[0] >= 0 else "") + _money(nums[1] - nums[0], unit, pre) if None not in nums else "")
    f_val = font("heavy", int(look.px(88 if not look.portrait else 78) * (0.7 + 0.3 * wf)))
    f_lab = font("bold", int(look.px(34) * (0.8 + 0.2 * wf)))
    tiles = []
    for lb, vt in zip(labels, shown):
        v = text_layer([vt], f_val, look.fg, align="center", line_h=1.0, shadow=2)
        l = text_layer([look.case(lb)], f_lab, look.sub, align="center", line_h=1.1, shadow=1)
        w = max(v.width, l.width) + look.px(70)
        h = v.height + l.height + look.px(54)
        body, m = panel(w, h, look.px(26), look)
        t = body.copy()
        t.alpha_composite(v, (m + (w - v.width) // 2, m + look.px(18)))
        t.alpha_composite(l, (m + (w - l.width) // 2, m + look.px(18) + v.height + look.px(8)))
        tiles.append((t, w, h, m))
    if look.portrait or (look.avoid and wf < 1.0):                                       # (beside a face in a wide frame: stacked, in the free column)
        gap_len = int(look.H * 0.17 * wf)
        bw, bh = max(tiles[0][1], tiles[1][1]), tiles[0][2] + tiles[1][2] + gap_len
        pos = look.place(bw, bh, [((look.W - bw) // 2, int(look.H * 0.2)), (look.px(30), int(look.H * 0.2)), (look.W - bw - look.px(30), int(look.H * 0.2))], y_max=0.75)
        if pos is None:
            return None
        cx = pos[0] + bw // 2
        y1, y2 = pos[1], pos[1] + tiles[0][2] + gap_len
        sc.add(tiles[0][0], cx - tiles[0][1] // 2 - tiles[0][3], y1 - tiles[0][3], 0.0, pop=0.5, dur=0.55)
        sc.add(tiles[1][0], cx - tiles[1][1] // 2 - tiles[1][3], y2 - tiles[1][3], 0.5, pop=0.5, dur=0.55)
        sc.glass.append({"x": cx - tiles[0][1] // 2, "y": y1, "w": tiles[0][1], "h": tiles[0][2], "r": look.px(26), "dim": 0.0, "t0": 0.0})
        sc.glass.append({"x": cx - tiles[1][1] // 2, "y": y2, "w": tiles[1][1], "h": tiles[1][2], "r": look.px(26), "dim": 0.0, "t0": 0.5})
        a, b = (cx, y1 + tiles[0][2]), (cx, y2)
    else:
        gap_len = int(look.W * 0.30 * wf)
        bw, bh = tiles[0][1] + tiles[1][1] + gap_len, max(tiles[0][2], tiles[1][2])
        y0 = int(look.H * 0.58)
        pos = look.place(bw, bh, [((look.W - bw) // 2, y0), (look.px(40), y0), (look.W - bw - look.px(40), y0), ((look.W - bw) // 2, int(look.H * 0.12))], y_max=0.92)
        if pos is None:
            return None
        x1, x2 = pos[0], pos[0] + tiles[0][1] + gap_len
        y = pos[1] + (bh - tiles[0][2]) // 2
        sc.add(tiles[0][0], x1 - tiles[0][3], y - tiles[0][3], 0.0, slide_x=-look.px(100), pop=0.5, dur=0.55)
        sc.add(tiles[1][0], x2 - tiles[1][3], pos[1] + (bh - tiles[1][2]) // 2 - tiles[1][3], 0.5, slide_x=look.px(100), pop=0.5, dur=0.55)
        sc.glass.append({"x": x1, "y": y, "w": tiles[0][1], "h": tiles[0][2], "r": look.px(26), "dim": 0.0, "t0": 0.0})
        sc.glass.append({"x": x2, "y": pos[1] + (bh - tiles[1][2]) // 2, "w": tiles[1][1], "h": tiles[1][2], "r": look.px(26), "dim": 0.0, "t0": 0.5})
        a, b = (x1 + tiles[0][1], pos[1] + bh // 2), (x2, pos[1] + bh // 2)
    lw = max(4, look.px(7))

    pill_w = (text_layer([delta], font("heavy", int(look.px(46) * (0.8 + 0.2 * wf))), look.fg, line_h=1.0).width + look.px(46)) if delta else 0

    def line(t: float):
        img = Image.new("RGBA", (look.W, look.H), (0, 0, 0, 0))
        p = ease_out((t - 0.55) / 0.8)
        if p > 0:
            d_ = ImageDraw.Draw(img)
            ln = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            ux, uy = (b[0] - a[0]) / ln, (b[1] - a[1]) / ln
            half = pill_w / 2 + look.px(14) if pill_w and not look.portrait else (look.px(46) if pill_w else 0)
            mid = ln / 2
            for s0, s1 in ((0.0, mid - half), (mid + half, ln)):                    # the line stops short of the pill on both sides
                e0, e1 = s0, min(s1, ln * p)
                if e1 > e0:
                    d_.line([(a[0] + ux * e0, a[1] + uy * e0), (a[0] + ux * e1, a[1] + uy * e1)], fill=(*look.deco, 255), width=lw)
        return img
    sc.dyn(line, 0, 0)
    if delta:
        d = text_layer([delta], font("heavy", int(look.px(46) * (0.8 + 0.2 * wf))), (12, 12, 14) if _lum(look.accent) > 0.55 or look.mode in ("board",) else (255, 255, 255),
                       align="center", line_h=1.0)
        pill = Image.new("RGBA", (d.width + look.px(46), d.height + look.px(22)), (0, 0, 0, 0))
        ImageDraw.Draw(pill).rounded_rectangle((0, 0, pill.width - 1, pill.height - 1), radius=pill.height // 2, fill=(*look.accent, 255))
        pill.alpha_composite(d, ((pill.width - d.width) // 2, look.px(10)))
        sc.add(pill, int((a[0] + b[0]) / 2 - pill.width / 2), int((a[1] + b[1]) / 2 - pill.height / 2), 1.1, pop=0.2, dur=0.55)
    return sc


def scene_timeline(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """Dates along a line, each with what happened, appearing one after another."""
    sc = Scene(look, seconds)
    ev = [{"date": " ".join(str(e.get("date", "")).split())[:24], "label": " ".join(str(e.get("label", "")).split())[:60]} for e in (spec.get("events") or [])][:4]
    if len(ev) < 2:
        return None
    n = len(ev)
    f_d = font("heavy", int(look.px(46) * (0.75 + 0.25 * wf)))
    f_l = font("bold", int(look.px(42 if not look.portrait else 40) * (0.75 + 0.25 * wf)))
    column = (not look.portrait) and bool(look.avoid) and wf < 0.7                  # beside a big face in a wide frame: the vertical timeline, in the free side column
    if look.portrait or column:
        w_all = int(look.W * (0.84 if look.portrait else 0.28) * (wf if look.portrait else 1.0))
        avail = w_all - look.px(70) - look.px(16)                                        # what is left of the column after the line and its dots
        dcol = (12, 12, 14) if look.mode in ("board", "bold") and _lum(look.accent) > 0.55 else look.fg
        d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
        rows = []
        for k_ in (1.0, 0.88, 0.78, 0.68, 0.6, 0.52):                                    # the biggest type in which every date and every word fits the column
            fd, fl = font("heavy", max(12, int(f_d.size * k_))), font("bold", max(11, int(f_l.size * k_)))
            rows, ok = [], True
            for e in ev:
                dtxt = look.case(e["date"])
                lines = [dtxt] if d0.textlength(dtxt, font=fd) + look.px(48) <= avail else dtxt.rsplit(" ", 1)
                if any(d0.textlength(x, font=fd) + look.px(48) > avail for x in lines):
                    ok = False
                wrapped = _wrap(e["label"], fl, avail, 4)
                if any(d0.textlength(x, font=fl) > avail for x in wrapped) or " ".join(wrapped) != e["label"]:
                    ok = False
                rows.append((text_layer(lines, fd, dcol, align="center", line_h=1.0), text_layer(wrapped, fl, look.fg, line_h=1.15, shadow=1)))
            if ok:
                break
        else:
            return None                                                                  # too narrow to read: no card rather than a clipped one
        row_h = [d.height + look.px(24) + look.px(10) + lb.height + look.px(34) for d, lb in rows]       # (date pill, then its label under it)
        total_h = sum(row_h)
        pos = look.place(w_all, total_h, [((look.W - w_all) // 2, int(look.H * 0.12)), (look.px(30), int(look.H * 0.12)), (look.W - w_all - look.px(30), int(look.H * 0.12))] if look.portrait
                         else [(look.px(30), int(look.H * 0.06)), (look.W - w_all - look.px(30), int(look.H * 0.06))], y_max=0.75 if look.portrait else 0.68)
        if pos is None:
            return None
        x0, y = pos
        line_x = x0 + look.px(26)
        sc.dyn(lambda t, lx=line_x, y0=y, th=total_h: _vline(look, lx, y0, th, t), 0, 0)
        for i, ((dt, lb), rh) in enumerate(zip(rows, row_h)):
            at = 0.3 + 0.65 * i
            dot = Image.new("RGBA", (look.px(30), look.px(30)), (0, 0, 0, 0))
            ImageDraw.Draw(dot).ellipse((0, 0, dot.width - 1, dot.height - 1), fill=(*look.deco, 255), outline=(255, 255, 255, 255), width=3)
            sc.add(dot, line_x - dot.width // 2, y + look.px(12), at, pop=0.1, dur=0.5)
            pill = _pill(look, dt)
            sc.add(pill, x0 + look.px(70), y, at, slide_x=-look.px(60), pop=0.7, dur=0.5)
            sc.add(lb, x0 + look.px(70), y + pill.height + look.px(10), at + 0.1, rise=look.px(12), dur=0.5)
            y += rh
        return sc
    w_all = int(look.W * 0.78 * wf)
    cols = [w_all // n] * n
    dcol = (12, 12, 14) if look.mode in ("board", "bold") and _lum(look.accent) > 0.55 else look.fg
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    heads = []
    for k_ in (1.0, 0.88, 0.76, 0.66, 0.56, 0.48):                                      # the biggest type in which every date and every word fits its column
        fd, fl = font("heavy", max(12, int(f_d.size * k_))), font("bold", max(11, int(f_l.size * k_)))
        heads, ok = [], True
        for e in ev:
            dtxt = look.case(e["date"])
            lines = [dtxt] if d0.textlength(dtxt, font=fd) + look.px(48) <= cols[0] - look.px(14) else dtxt.rsplit(" ", 1)          # "March 2024" -> "March" / "2024"
            if any(d0.textlength(x, font=fd) + look.px(48) > cols[0] - look.px(14) for x in lines):
                ok = False
            wrapped = _wrap(e["label"], fl, cols[0] - look.px(40), 4)
            if any(d0.textlength(x, font=fl) > cols[0] - look.px(30) for x in wrapped) or " ".join(wrapped).replace("…", "") != e["label"].replace("…", ""):
                ok = False
            heads.append((text_layer(lines, fd, dcol, align="center", line_h=1.0), text_layer(wrapped, fl, look.fg, align="center", line_h=1.15, shadow=1, width=cols[0] - look.px(30))))
        if ok:
            break
    pill_h = max(_pill(look, dt).height for dt, _ in heads)
    total_h = pill_h + look.px(90) + max(lb.height for _, lb in heads)
    y0 = int(look.H * 0.56)
    pos = look.place(w_all, total_h, [((look.W - w_all) // 2, y0), (look.px(40), y0), (look.W - w_all - look.px(40), y0), ((look.W - w_all) // 2, int(look.H * 0.12))], y_max=0.95)
    if pos is None:
        return None
    x0, y0 = pos
    line_y = y0 + pill_h + look.px(26)
    sc.dyn(lambda t: _hline(look, x0, line_y, w_all, t), 0, 0)
    for i, (dt, lb) in enumerate(heads):
        cx = x0 + cols[0] * i + cols[0] // 2
        at = 0.3 + 0.65 * i
        dot = Image.new("RGBA", (look.px(30), look.px(30)), (0, 0, 0, 0))
        ImageDraw.Draw(dot).ellipse((0, 0, dot.width - 1, dot.height - 1), fill=(*look.deco, 255), outline=(255, 255, 255, 255), width=3)
        sc.add(dot, cx - dot.width // 2, line_y - dot.height // 2, at, pop=0.1, dur=0.5)
        pill = _pill(look, dt)
        sc.add(pill, cx - pill.width // 2, y0 - look.px(6), at, rise=-look.px(20), pop=0.7, dur=0.5)
        sc.add(lb, cx - lb.width // 2, line_y + look.px(34), at + 0.1, rise=look.px(14), dur=0.5)
    return sc


def _pill(look: Look, text_img: Image.Image) -> Image.Image:
    pill = Image.new("RGBA", (text_img.width + look.px(48), text_img.height + look.px(24)), (0, 0, 0, 0))
    fill = (*look.accent, 255)
    ImageDraw.Draw(pill).rounded_rectangle((0, 0, pill.width - 1, pill.height - 1), radius=pill.height // 2, fill=fill,
                                           outline=(255, 255, 255, 255) if look.mode == "board" else None, width=3)
    pill.alpha_composite(text_img, ((pill.width - text_img.width) // 2, look.px(11)))
    return pill


def _hline(look: Look, x: int, y: int, w: int, t: float) -> Image.Image:
    img = Image.new("RGBA", (look.W, look.H), (0, 0, 0, 0))
    p = ease_out((t - 0.1) / (0.5 + 0.55 * 4))
    ImageDraw.Draw(img).line([(x, y), (x + w * min(1.0, p * 1.15), y)], fill=(*look.deco, 255), width=max(4, look.px(7)))
    return img


def _vline(look: Look, x: int, y: int, h: int, t: float) -> Image.Image:
    img = Image.new("RGBA", (look.W, look.H), (0, 0, 0, 0))
    p = ease_out((t - 0.1) / 2.2)
    ImageDraw.Draw(img).line([(x, y), (x, y + h * min(1.0, p * 1.1))], fill=(*look.deco, 255), width=max(4, look.px(7)))
    return img


def scene_statement(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """An emphasis card: the picture is replaced by a quiet dotted page and ONE sentence, each word going from grey to black as it is spoken, the key phrase
    underlined with a hand-drawn line. The speaker's voice carries on underneath, so it lands exactly on the words. spec: text, times (seconds, per word),
    emphasis (the words to underline)."""
    text = " ".join(str(spec.get("text") or "").split())
    words = text.split()
    if not 3 <= len(words) <= 24:
        return None
    times = [float(t) for t in (spec.get("times") or [])]
    if len(times) != len(words):
        times = [0.35 + (max(1.0, seconds - 1.6) * i / max(1, len(words))) for i in range(len(words))]
    emph = {i for i in range(len(words))} if False else set()
    em = [w.lower().strip(".,!?;:\"'") for w in str(spec.get("emphasis") or "").split()]
    toks = [w.lower().strip(".,!?;:\"'") for w in words]
    for i in range(len(toks) - len(em) + 1):
        if em and toks[i:i + len(em)] == em:
            emph = set(range(i, i + len(em)))
            break
    dark = not look.light
    bg_c, dot_c = ((22, 22, 25), (58, 58, 64)) if dark else ((239, 239, 237), (205, 205, 202))
    ink, dim = ((244, 244, 244), (92, 92, 98)) if dark else ((18, 18, 18), (175, 175, 172))
    W, H = look.W, look.H
    base = Image.new("RGB", (W, H), bg_c)
    bd = ImageDraw.Draw(base)
    step = look.px(30)
    for yy in range(step // 2, H, step):
        for xx in range(step // 2, W, step):
            bd.ellipse((xx - 1.5, yy - 1.5, xx + 1.5, yy + 1.5), fill=dot_c)
    max_w = int(W * (0.84 if look.portrait else 0.82))
    size = look.px(104 if look.portrait else 96)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    while True:                                                                           # the biggest type that keeps it to a few lines
        f = font("bold", size)
        lines, cur = [], []
        for i, w in enumerate(words):
            trial = " ".join(words[j] for j in cur + [i])
            if cur and d0.textlength(trial, font=f) > max_w:
                lines.append(cur)
                cur = [i]
            else:
                cur.append(i)
        lines.append(cur)
        if (len(lines) <= (6 if look.portrait else 3)) or size <= look.px(36):
            break
        size = int(size * 0.92)
    lh = int(size * 1.32)
    top = (H - lh * len(lines)) // 2
    place = {}                                                                            # word index -> (x, y, width)
    space = d0.textlength(" ", font=f)
    for r, ln in enumerate(lines):
        total = sum(d0.textlength(words[i], font=f) for i in ln) + space * (len(ln) - 1)
        x = (W - total) / 2
        for i in ln:
            w_ = d0.textlength(words[i], font=f)
            place[i] = (x, top + r * lh, w_)
            x += w_ + space
    spans = []                                                                            # the underline: one stroke per line the emphasis touches
    for r, ln in enumerate(lines):
        mine = [i for i in ln if i in emph]
        if mine:
            spans.append((place[mine[0]][0], place[mine[-1]][0] + place[mine[-1]][2], top + r * lh + int(size * 1.12), mine[0], mine[-1]))

    def frame(t: float):
        img = base.copy().convert("RGBA")
        dr = ImageDraw.Draw(img)
        for i, w in enumerate(words):
            p = max(0.0, min(1.0, (t - times[i]) / 0.22))
            col = tuple(int(dim[k] + (ink[k] - dim[k]) * p) for k in range(3))
            x, y, _w = place[i]
            dr.text((x, y), w, font=f, fill=(*col, 255))
        for x0, x1, yy, a, b in spans:
            q = max(0.0, min(1.0, (t - times[a]) / max(0.4, times[b] - times[a] + 0.35)))
            if q > 0:
                xe = x0 + (x1 - x0) * ease_out(q)
                dr.line([(x0, yy), (xe, yy + 2)], fill=(*ink, 255), width=max(2, look.px(4)))
        return img
    sc = Scene(look, seconds)
    sc.custom = True
    style = _style_of(spec)
    sc.dyn(lambda t: _transition(frame(t), t, seconds, style, look), 0, 0)
    sc.glass = []
    return sc


# ------------------------------------------------------------------------------------------------ full-screen cards and their transitions
TRANSITIONS = ("wipe", "iris", "push", "zoom")


def _style_of(spec: dict) -> str:
    """Which transition this full-screen card uses: the planner's choice, else one by where it falls (so neighbours differ)."""
    st = str(spec.get("style") or "")
    return st if st in TRANSITIONS else TRANSITIONS[int(float(spec.get("at") or 0) * 7) % len(TRANSITIONS)]


def _transition(img: Image.Image, t: float, secs: float, style: str, look: Look) -> Image.Image:
    """A full-frame card arriving over the picture (and leaving again): a wipe with a bright edge, an iris opening from the middle, a push up from below, or a zoom."""
    tin, tout = 0.5, 0.45
    if t < tin:
        p = ease_io(t / tin)
    elif t > secs - tout:
        p = 1.0 - ease_io((t - (secs - tout)) / (tout - 0.05))                      # (empty by 0.05 s before the end: the last rendered frame is empty)
    else:
        return img
    p = max(0.0, min(1.0, p))
    W, H = img.size
    if p <= 0.0:
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))
    soft = max(8, int(look.u * 40))
    band = max(6, int(look.u * 22))
    a = np.asarray(img.convert("RGBA")).astype(np.float32)
    if style == "push":
        dy = int((1.0 - p) * H)
        out = np.zeros_like(a)
        out[dy:] = a[:H - dy] if dy < H else 0
        return Image.fromarray(out.astype(np.uint8), "RGBA")
    if style == "zoom":
        k = 1.0 + 0.14 * (1.0 - p)
        big = img.resize((int(W * k), int(H * k)), Image.BICUBIC)
        out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        out.alpha_composite(big.crop(((big.width - W) // 2, (big.height - H) // 2, (big.width - W) // 2 + W, (big.height - H) // 2 + H)))
        return _alpha_mul(out, ease_out(p * 1.4))
    if style == "iris":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        dist = (p * (math.hypot(W, H) / 2 + soft)) - np.hypot(xx - W / 2, yy - H / 2)
    else:                                                                                  # wipe, left to right
        dist = (p * (W + soft) - np.arange(W, dtype=np.float32))[None, :].repeat(H, axis=0)
    m = np.clip(dist / soft, 0.0, 1.0)
    a[..., 3] *= m
    edge = np.clip(1.0 - np.abs(dist - band * 0.6) / band, 0.0, 1.0) * (1.0 if p < 0.98 else 0.0)        # the bright edge that leads the card in
    col = np.array([*look.accent, 255], dtype=np.float32)
    a = a * (1 - edge[..., None] * 0.85) + col[None, None, :] * (edge[..., None] * 0.85) * np.array([1, 1, 1, 1], dtype=np.float32)
    a[..., 3] = np.maximum(a[..., 3], edge * 255 * 0.85)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


PAGES = ("dots", "grid", "rule", "stripe", "blocks", "frame", "plain")


def _page(look: Look, bg: str | None = None) -> tuple[Image.Image, tuple, tuple, tuple]:
    """The background of a full-screen card, flat and quiet: dots, graph paper, ruled lines, hatching, big colour blocks, a frame or nothing, in the look's
    light or dark (loud looks: a flat accent colour). -> (image, ink, dim ink, pattern colour)."""
    W, H = look.W, look.H
    if look.page_c:
        bg_c, ink, dim, dot_c = look.page_c, look.ink_c, look.dim_c, look.dot_c
    elif look.mode in ("bold", "board"):
        lin = [c / 255 / 12.92 if c / 255 <= 0.03928 else ((c / 255 + 0.055) / 1.055) ** 2.4 for c in look.accent]
        L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
        dark = (1.05 / (L + 0.05)) > ((L + 0.05) / 0.0545)            # white reads better than black on it (broadcast blue; not bold red)
        bg_c, dot_c = tuple(look.accent), tuple(min(255, c + 22) for c in look.accent)
        ink, dim = ((250, 250, 250), mix(look.accent, (255, 255, 255), 0.6)) if dark else ((12, 12, 14), (60, 60, 64))
    elif look.light:
        bg_c, ink, dim, dot_c = (239, 239, 237), (18, 18, 18), (175, 175, 172), (205, 205, 202)
    else:
        bg_c, ink, dim, dot_c = (22, 22, 25), (244, 244, 244), (92, 92, 98), (58, 58, 64)
    base = Image.new("RGB", (W, H), bg_c)
    bd = ImageDraw.Draw(base)
    kind = bg if bg in PAGES else "dots"
    if kind == "dots":
        step = look.px(30)
        for yy in range(step // 2, H, step):
            for xx in range(step // 2, W, step):
                bd.ellipse((xx - 1.5, yy - 1.5, xx + 1.5, yy + 1.5), fill=dot_c)
    elif kind == "grid":                                                                # graph paper: a fine grid, every fifth line a little stronger
        step = look.px(48)
        strong = mix(dot_c, ink, 0.12)
        for i, xx in enumerate(range(0, W, step)):
            bd.line((xx, 0, xx, H), fill=strong if i % 5 == 0 else dot_c, width=2 if i % 5 == 0 else 1)
        for i, yy in enumerate(range(0, H, step)):
            bd.line((0, yy, W, yy), fill=strong if i % 5 == 0 else dot_c, width=2 if i % 5 == 0 else 1)
    elif kind == "rule":                                                                # a ruled page with one margin line
        step = look.px(66)
        for yy in range(step, H, step):
            bd.line((0, yy, W, yy), fill=dot_c, width=2)
        bd.line((int(W * 0.055), 0, int(W * 0.055), H), fill=mix(bg_c, tuple(look.accent), 0.45), width=3)
    elif kind == "stripe":                                                              # fine diagonal hatching
        step = look.px(26)
        for k in range(-H, W, step):
            bd.line((k, H, k + H, 0), fill=dot_c, width=2)
    elif kind == "blocks":                                                              # poster shapes: a large circle off one corner, a slab off the other
        tint = mix(bg_c, tuple(look.accent), 0.16 if look.light else 0.22)
        tint2 = mix(bg_c, tuple(look.accent), 0.09 if look.light else 0.12)
        bd.ellipse((int(W * 0.66), -int(H * 0.42), int(W * 1.18), int(H * 0.62)), fill=tint)
        bd.rectangle((0, int(H * 0.86), int(W * 0.5), H), fill=tint2)
    elif kind == "frame":                                                               # a single inset outline
        m = look.px(38)
        bd.rectangle((m, m, W - m, H - m), outline=ink, width=max(3, look.px(6)))
    return base, ink, dim, dot_c


def _rel_lum(c) -> float:
    def ch(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    return 0.2126 * ch(c[0]) + 0.7152 * ch(c[1]) + 0.0722 * ch(c[2])


def _accent_on(look: Look, bg, ink) -> tuple:
    """The look's accent colour, darkened (or lightened) toward the ink until it can be read on this page (contrast 3:1); on a page that IS the accent, the ink."""
    if look.mode in ("bold", "board"):
        return tuple(ink)
    acc = tuple(look.accent)
    for _ in range(12):
        hi, lo = sorted((_rel_lum(acc), _rel_lum(bg)), reverse=True)
        if (hi + 0.05) / (lo + 0.05) >= 3.0:
            break
        acc = tuple(int(acc[i] + (ink[i] - acc[i]) * 0.14) for i in range(3))
    return acc


def _fullscreen(look: Look, spec: dict, seconds: float, draw) -> Scene:
    base, ink, dim, _dot = _page(look, spec.get("bg"))
    acc = _accent_on(look, base.getpixel((2, 2)), ink)
    sc = Scene(look, seconds)
    sc.custom = True
    style = _style_of(spec)

    def frame(t: float):
        img = base.copy().convert("RGBA")
        lay = Image.new("RGBA", img.size, (0, 0, 0, 0))                              # everything is drawn on a clear layer and laid over the page: text with alpha below 255 blends, it does not punch holes
        draw(ImageDraw.Draw(lay), lay, t, ink, dim, acc)
        img.alpha_composite(lay)
        return _transition(img, t, seconds, style, look)
    sc.dyn(frame, 0, 0)
    sc.glass = []
    return sc


def _lerp(a, b, k: float):
    return tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3))


def scene_fullslide(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A whole-screen slide, PowerPoint style: a big title with an accent bar and 2-5 numbered points arriving one after another."""
    title = " ".join(str(spec.get("title") or "").split())[:70]
    bullets = [" ".join(str(b).split())[:80] for b in (spec.get("bullets") or [])][:5]
    if len(bullets) < 2 or not title:
        return None
    W, H = look.W, look.H
    mx = int(W * 0.08)
    tw = W - 2 * mx
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    k_ = 1.0
    while True:                                                                           # the biggest type with the whole slide inside the frame
        tsize = _fit_size(int(look.px(120 if look.portrait else 118) * k_), look.case(title), "heavy", tw, 2, look.px(34))
        tf = font("heavy", tsize)
        tl = _wrap(look.case(title), tf, tw, 2)
        bsize = _fit_size(int(look.px(88 if look.portrait else 84) * k_), max(bullets, key=len), "bold", tw - look.px(120), 2, look.px(26))
        bf = font("bold", bsize)
        blines = [_wrap(b, bf, tw - look.px(120), 2) for b in bullets]
        if any(d0.textlength(l_, font=bf) > tw - look.px(100) for bl in blines for l_ in bl) and k_ > 0.3:          # a word wider than the column: smaller type
            k_ *= 0.9
            continue
        lh_b = int(bsize * 1.22)
        gap = int(look.px(46) * k_)
        total = len(tl) * int(tsize * 1.15) + look.px(46) + sum(len(b) * lh_b + gap for b in blines)
        if total <= H * 0.86 or k_ < 0.45:
            break
        k_ *= 0.92
    y0 = max(int(H * 0.07), (H - total) // 2)
    n = len(bullets)
    step = max(0.2, min(max(0.7, (seconds - 2.6) / n), (seconds - 1.9) / n))                 # (every point arrives before the card starts to leave)
    at = _times_of(spec, n, seconds, 0.6, 1.9) if spec.get("times") else [0.6 + i * step for i in range(n)]       # (as each is SAID, when the planner knows when)

    def draw(dr, img, t, ink, dim, acc):
        y = y0
        for ln in tl:
            dr.text((mx, y), ln, font=tf, fill=(*ink, 255))
            y += int(tsize * 1.15)
        bar_w = int(min(tw * 0.4, tsize * 5) * ease_out((t - 0.3) / 0.6))
        dr.rectangle((mx, y + look.px(8), mx + bar_w, y + look.px(8) + max(6, look.px(9))), fill=(*acc, 255))
        y += look.px(46)
        for i, lines in enumerate(blines):
            p = ease_out((t - at[i]) / 0.5)
            col = _lerp(dim, ink, ease_out((t - at[i]) / 0.3))
            if p > 0:
                dx = int((1 - p) * look.px(60))
                r = int(bsize * 0.62)
                cx, cy = mx + r + dx, y + lh_b // 2
                dr.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(*acc, int(255 * p)))
                num = str(i + 1)
                _CTX.mono = look.mono
                nf = font("heavy", int(bsize * 0.7))
                _CTX.mono = False
                dr.text((cx - d0.textlength(num, font=nf) / 2, cy - nf.size * 0.58), num, font=nf, fill=(*((12, 12, 14) if _lum(acc) > 0.5 else (255, 255, 255)), int(255 * p)))
                for j, ln in enumerate(lines):
                    dr.text((mx + 2 * r + look.px(28) + dx, y + j * lh_b), ln, font=bf, fill=(*col, int(255 * p)))
            y += len(lines) * lh_b + gap
    return _fullscreen(look, spec, seconds, draw)


def scene_bigstat(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """One number, filling the screen: it climbs from zero, then lands with a small beat; the label sits under an accent bar."""
    value = str(spec.get("value") or "").strip()[:14]
    label = " ".join(str(spec.get("label") or "").split())[:90]
    if not value:
        return None
    W, H = look.W, look.H
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    size = look.px(330 if look.portrait else 300)
    while size > look.px(80) and d0.textlength(value, font=font("heavy", size)) > W * 0.86:
        size = int(size * 0.92)
    bigf = font("heavy", size)
    lf = font("bold", look.px(60 if look.portrait else 46))
    lab = _wrap(label, lf, int(W * 0.8), 3)
    m_ = re.match(r"^([^\d\-]*)(-?[\d.,]+)(.*)$", value)

    def draw(dr, img, t, ink, dim, acc):
        shown = value
        if m_:
            try:
                raw_orig = m_.group(2)
                if re.fullmatch(r"\d+,\d{1,2}", raw_orig):                                    # "1,5" is one and a half, not fifteen
                    raw, comma = raw_orig.replace(",", "."), False
                else:
                    raw, comma = raw_orig.replace(",", ""), "," in raw_orig                  # (thousands commas only if the figure was written with them)
                target = float(raw)
                dec = len(raw.split(".")[1]) if "." in raw else 0
                shown = f"{m_.group(1)}{target * ease_out((t - 0.4) / 1.0):{',' if comma else ''}.{dec}f}{m_.group(3)}"
                if raw != raw_orig.replace(",", "") or (raw_orig.count(",") == 1 and raw != raw_orig and not comma):
                    shown = shown.replace(".", ",")                                             # (a decimal comma stays a comma)
            except ValueError:
                pass
        pulse = 1.0 + 0.07 * math.sin(math.pi * min(1.0, max(0.0, (t - 1.45) / 0.35)))
        _CTX.mono = look.mono
        f = font("heavy", int(size * pulse))
        _CTX.mono = False
        tw_ = d0.textlength(shown, font=f)
        top = int(H * 0.5) - size // 2 - (len(lab) * int(lf.size * 1.2)) // 2
        dr.text(((W - tw_) / 2, top), shown, font=f, fill=(*ink, 255))
        by = top + int(size * 1.18)
        bw = int(min(W * 0.5, 520) * ease_out((t - 0.9) / 0.6))
        dr.rectangle(((W - bw) // 2, by, (W + bw) // 2, by + max(6, look.px(9))), fill=(*acc, 255))
        y = by + look.px(40)
        for ln in lab:
            col = _lerp(dim, ink, ease_out((t - 1.1) / 0.4))
            dr.text(((W - d0.textlength(ln, font=lf)) / 2, y), ln, font=lf, fill=(*col, 255))
            y += int(lf.size * 1.2)
    return _fullscreen(look, spec, seconds, draw)


def scene_chapter(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A section title between parts of a video: a big number or mark, then the heading, left aligned on the page."""
    text = " ".join(str(spec.get("text") or spec.get("title") or "").split())[:60]
    if not text:
        return None
    num = str(spec.get("number") or "").strip()[:3]
    W, H = look.W, look.H
    mx = int(W * 0.09)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    size = _fit_size(look.px(150 if look.portrait else 120), look.case(text), "heavy", W - 2 * mx, 3, look.px(50))
    f = font("heavy", size)
    lines = _wrap(look.case(text), f, W - 2 * mx, 3)
    while max(d0.textlength(l_, font=f) for l_ in lines) > W - 1.6 * mx and size > look.px(22):
        size = int(size * 0.9)
        f = font("heavy", size)
        lines = _wrap(look.case(text), f, W - 2 * mx, 3)
    nf = font("heavy", look.px(120 if look.portrait else 96))
    total = (nf.size + look.px(30) if num else 0) + len(lines) * int(size * 1.12)
    y0 = (H - total) // 2

    def draw(dr, img, t, ink, dim, acc):
        y = y0
        if num:
            p = ease_out((t - 0.35) / 0.5)
            dr.text((mx, y + int((1 - p) * look.px(40))), num, font=nf, fill=(*acc, int(255 * p)))
            y += nf.size + look.px(30)
        for i, ln in enumerate(lines):
            p = ease_out((t - 0.55 - 0.18 * i) / 0.5)
            dr.text((mx, y + int((1 - p) * look.px(50))), ln, font=f, fill=(*ink, int(255 * p)))
            y += int(size * 1.12)
        bw = int((W - 2 * mx) * 0.3 * ease_out((t - 0.9) / 0.7))
        dr.rectangle((mx, y + look.px(14), mx + bw, y + look.px(14) + max(6, look.px(9))), fill=(*acc, 255))
    return _fullscreen(look, spec, seconds, draw)


def scene_fullquote(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A quotation on a page of its own: a big mark, the words centred, the speaker under them."""
    text = " ".join(str(spec.get("text") or "").split())[:170]
    who = " ".join(str(spec.get("who") or "").split())[:50]
    if len(text.split()) < 3:
        return None
    W, H = look.W, look.H
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    size = _fit_size(look.px(92 if look.portrait else 84), text, "bold", int(W * 0.78), 6 if look.portrait else 4, look.px(36))
    f = font("bold", size)
    lines = _wrap(text, f, int(W * 0.78), 6)
    while max(d0.textlength(l_, font=f) for l_ in lines) > W * 0.88 and size > look.px(22):                  # one very long word: smaller type rather than cut off at the edge
        size = int(size * 0.9)
        f = font("bold", size)
        lines = _wrap(text, f, int(W * 0.78), 6)
    qf = font("heavy", look.px(300 if look.portrait else 240))
    wf_ = font("medium", int(size * 0.55))
    total = len(lines) * int(size * 1.25) + (look.px(60) if who else 0)
    y0 = (H - total) // 2 + look.px(40)

    def draw(dr, img, t, ink, dim, acc):
        pq = ease_out((t - 0.3) / 0.5)
        if spec.get("mark", True):
            dr.text((int(W * 0.08), y0 - int(qf.size * 0.78) + int((1 - pq) * look.px(30))), "\u201c", font=qf, fill=(*acc, int(255 * pq)))
        for i, ln in enumerate(lines):
            p = ease_out((t - 0.5 - 0.14 * i) / 0.5)
            dr.text(((W - d0.textlength(ln, font=f)) / 2, y0 + i * int(size * 1.25) + int((1 - p) * look.px(26))), ln, font=f, fill=(*ink, int(255 * p)))
        if who:
            p = ease_out((t - 0.6 - 0.14 * len(lines)) / 0.5)
            lab = "\u2014 " + who
            dr.text(((W - d0.textlength(lab, font=wf_)) / 2, y0 + len(lines) * int(size * 1.25) + look.px(14)), lab, font=wf_, fill=(*acc, int(255 * p)))
    return _fullscreen(look, spec, seconds, draw)


def _box_colors(look: Look, ink) -> tuple:
    """(fill, edge) for the cards drawn on a page: white on a light page, charcoal on a dark one, white with a black edge on a loud (accent) page."""
    if look.mode in ("bold", "board"):
        return (255, 255, 255), (12, 12, 14)
    return ((255, 255, 255), (212, 212, 208)) if look.light else ((40, 40, 46), (70, 70, 78))


def scene_search(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A question typed into a search box on a page of its own, then the answer line: "What is X?" -> X in big type."""
    q = " ".join(str(spec.get("text") or "").split())[:90]
    ans = " ".join(str(spec.get("answer") or "").split())[:70]
    if len(q.split()) < 2:
        return None
    W, H = look.W, look.H
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    bw = int(W * (0.86 if look.portrait else 0.62))
    bh = look.px(150 if look.portrait else 124)
    size = _fit_size(look.px(60 if look.portrait else 52), q, "medium", bw - look.px(130) - (bh - look.px(32)) - look.px(60), 1, look.px(18))
    qf = font("medium", size)
    af = font("heavy", _fit_size(look.px(96 if look.portrait else 84), ans or "x", "heavy", int(W * 0.84), 2, look.px(34)))
    alines = _wrap(look.case(ans), af, int(W * 0.84), 2) if ans else []
    tag = str(spec.get("number") or "").strip()[:2]
    t0 = 0.75
    cps = max(14.0, len(q) / max(1.0, min(2.4, seconds - 2.6)))
    t_end = t0 + len(q) / cps

    def draw(dr, img, t, ink, dim, acc):
        fill, edge = _box_colors(look, ink)
        x0, y0 = (W - bw) // 2, int(H * (0.40 if look.portrait else 0.36))
        p = ease_out((t - 0.25) / 0.45)
        if p <= 0:
            return
        dy = int((1 - p) * look.px(40))
        a = int(255 * p)
        dr.rounded_rectangle((x0, y0 + dy, x0 + bw, y0 + dy + bh), radius=bh // 2, fill=(*fill[:3], a), outline=(*edge[:3], a), width=max(2, look.px(3)))
        cx, cy = x0 + look.px(70), y0 + dy + bh // 2                                      # the magnifier
        r = look.px(20)
        dr.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(*dim, a), width=max(3, look.px(5)))
        dr.line((cx + r * 0.7, cy + r * 0.7, cx + r * 1.7, cy + r * 1.7), fill=(*dim, a), width=max(3, look.px(5)))
        n = max(0, min(len(q), int((t - t0) * cps)))
        txt = q[:n]
        tx = x0 + look.px(130)
        dr.text((tx, cy - size * 0.62), txt, font=qf, fill=(*ink, a))
        if int(t * 2.2) % 2 == 0 and t < t_end + 0.6:                                      # the blinking cursor
            cxp = tx + d0.textlength(txt, font=qf) + 4
            dr.rectangle((cxp, cy - size * 0.55, cxp + max(3, look.px(4)), cy + size * 0.55), fill=(*ink, a))
        sr = bh // 2 - look.px(16)                                                          # the send button, which lights up when the question is typed
        sx = x0 + bw - sr - look.px(18)
        on = ease_out((t - t_end) / 0.3)
        pulse = 1.0 + 0.08 * math.sin(math.pi * min(1.0, max(0.0, (t - t_end - 0.1) / 0.4)))
        rr = int(sr * pulse)
        dr.ellipse((sx - rr, cy - rr, sx + rr, cy + rr), fill=(*(_lerp(dim, acc, on)), a))
        ar = int(sr * 0.42)
        dr.line((sx - ar, cy, sx + ar, cy), fill=(255, 255, 255, a), width=max(3, look.px(5)))
        dr.line((sx + ar * 0.1, cy - ar * 0.9, sx + ar, cy), fill=(255, 255, 255, a), width=max(3, look.px(5)))
        dr.line((sx + ar * 0.1, cy + ar * 0.9, sx + ar, cy), fill=(255, 255, 255, a), width=max(3, look.px(5)))
        if tag:
            tf = font("heavy", look.px(34))
            lab = f"PART {tag}"
            dr.text((W // 2 - d0.textlength(lab, font=tf) / 2, y0 - look.px(70)), lab, font=tf, fill=(*acc, a))
        pa = ease_out((t - t_end - 0.5) / 0.5)
        if alines and pa > 0:
            yy = y0 + bh + look.px(80) + int((1 - pa) * look.px(30))
            for ln in alines:
                dr.text(((W - d0.textlength(ln, font=af)) / 2, yy), ln, font=af, fill=(*ink, int(255 * pa)))
                yy += int(af.size * 1.15)
            bwid = int(min(W * 0.4, 420) * pa)
            dr.rectangle(((W - bwid) // 2, yy + look.px(8), (W + bwid) // 2, yy + look.px(8) + max(6, look.px(9))), fill=(*acc, 255))
    return _fullscreen(look, spec, seconds, draw)


def _times_of(spec: dict, n: int, seconds: float, first: float = 0.9, last_pad: float = 1.9) -> list[float]:
    """When each of n things appears: the moments they are SAID if the planner supplied them (spec 'times'), else spread evenly across the card."""
    ts = [float(x) for x in (spec.get("times") or [])]
    if len(ts) == n:
        return [min(max(0.5, x), max(0.6, seconds - 1.2)) for x in ts]
    step = max(0.2, (seconds - last_pad - first) / max(1, n))
    return [first + i * step for i in range(n)]


def scene_twocol(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """Two paths side by side ("Buyer A / Buyer B"): their steps arrive as they are said, and a red tag lands on the one that fails (or a green one on the winner)."""
    titles = [" ".join(str(x).split())[:26] for x in (spec.get("titles") or [])][:2]
    steps = [[" ".join(str(x).split())[:46] for x in col][:4] for col in (spec.get("steps") or [])][:2]
    if len(titles) < 2 or len(steps) < 2 or min(len(c) for c in steps) < 2:
        return None
    head = " ".join(str(spec.get("title") or "").split())[:60]
    tag = spec.get("tag") if isinstance(spec.get("tag"), dict) else {}
    tside = int(tag.get("side", -1)) if str(tag.get("side", "")).lstrip("-").isdigit() else -1
    ttext = " ".join(str(tag.get("text") or "").split())[:22]
    W, H = look.W, look.H
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    stacked = look.portrait
    cw = int(W * 0.84) if stacked else int(W * 0.40)
    nrows = max(len(c) for c in steps)
    k = 1.0
    for _ in range(14):                                                                  # everything scaled down until both cards (and the heading) are inside the frame
        rh = int(look.px(78 if stacked else 90) * k)
        ch = int((look.px(120) + look.px(110)) * k) + nrows * rh
        head_h = (2 * int(look.px(84 if look.portrait else 76) * k * 1.12) + look.px(34)) if head else 0
        need = head_h + (2 * ch + look.px(40) if stacked else ch) + int(H * 0.12)
        if need <= H or k < 0.4:
            break
        k *= 0.92
    sf = font("bold", _fit_size(int(look.px(40 if stacked else 38) * k), max((x for c in steps for x in c), key=len), "bold", cw - look.px(90), 1, look.px(14)))
    hf = font("heavy", max(14, int(look.px(36) * k)))
    tf = font("heavy", _fit_size(int(look.px(84 if look.portrait else 76) * k), look.case(head) if head else "x", "heavy", int(W * 0.88), 2, look.px(24)))
    flat = [x for c in steps for x in c]
    times = _times_of(spec, len(flat) + (1 if ttext else 0), seconds, 0.9, 1.5)
    order = [(c, r) for c in range(2) for r in range(len(steps[c]))]
    when = {}
    if len(spec.get("times") or []) == len(flat) + (1 if ttext else 0):
        for k_, key in enumerate(order):
            when[key] = times[k_]
        tag_t = times[-1] if ttext else None
    else:
        base = 0.9
        step = max(0.2, (seconds - 2.8) / max(1, len(flat)))
        for k_, key in enumerate(order):
            when[key] = base + k_ * step
        tag_t = seconds - 1.9
    RED = (214, 60, 60)

    def draw(dr, img, t, ink, dim, acc):
        fill, edge = _box_colors(look, ink)
        ty = int(H * 0.07)
        if head:
            pp = ease_out((t - 0.35) / 0.5)
            hl = _wrap(look.case(head), tf, int(W * 0.88), 2)
            for i, ln in enumerate(hl):
                dr.text(((W - d0.textlength(ln, font=tf)) / 2, ty + i * int(tf.size * 1.12) + int((1 - pp) * 30)), ln, font=tf, fill=(*ink, int(255 * pp)))
            ty += len(hl) * int(tf.size * 1.12) + look.px(34)
        else:
            ty = int((H - (ch * (2 if stacked else 1) + (look.px(40) if stacked else 0))) / 2)
        for c in range(2):
            if stacked:
                x0, y0 = (W - cw) // 2, ty + c * (ch + look.px(40))
            else:
                x0, y0 = int(W * 0.06) + c * (cw + int(W * 0.08)), max(ty, int(H * 0.30))
            pc = ease_out((t - 0.5 - 0.15 * c) / 0.5)
            if pc <= 0:
                continue
            dy = int((1 - pc) * look.px(50))
            a = int(255 * pc)
            lose = (c == tside and ttext and tag_t is not None and t > tag_t and RED)
            dr.rounded_rectangle((x0, y0 + dy, x0 + cw, y0 + dy + ch), radius=look.px(26), fill=(*fill[:3], a), outline=(*(RED if lose else edge[:3]), a), width=max(2, look.px(4 if lose else 3)))
            dr.text((x0 + look.px(36), y0 + dy + int(look.px(30) * k)), look.case(titles[c]), font=hf, fill=(*acc, a))
            for r, label in enumerate(steps[c]):
                pr = ease_out((t - when[(c, r)]) / 0.4)
                if pr <= 0:
                    continue
                yy = y0 + dy + int(look.px(100) * k) + r * rh
                dr.rounded_rectangle((x0 + look.px(28), yy, x0 + cw - look.px(28), yy + rh - int(look.px(14) * k)), radius=look.px(14), fill=(*_lerp(fill[:3], dim, 0.14), int(255 * pr)))
                dr.text((x0 + look.px(48), yy + (rh - int(look.px(14) * k) - sf.size) // 2 - 2), label, font=sf, fill=(*ink, int(255 * pr)))
            if c == tside and ttext and tag_t is not None:
                pt = ease_out((t - tag_t) / 0.35)
                if pt > 0:
                    pw = int(d0.textlength("\u00d7  " + ttext, font=sf) + look.px(60))
                    py = y0 + dy + ch - int(look.px(84) * k)
                    dr.rounded_rectangle((x0 + look.px(28), py, x0 + look.px(28) + int(pw * pt), py + look.px(60)), radius=look.px(14), fill=(*RED, int(255 * pt)))
                    if pt > 0.7:
                        dr.text((x0 + look.px(54), py + look.px(13)), "\u00d7  " + ttext, font=sf, fill=(255, 255, 255, 255))
    return _fullscreen(look, spec, seconds, draw)


def scene_ledger(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """The sum, line by line: items with their amounts arriving as they are said, then the total under a bar."""
    rows = [(" ".join(str(r.get("label", "")).split())[:34], " ".join(str(r.get("value", "")).split())[:14]) for r in (spec.get("rows") or []) if isinstance(r, dict)][:5]
    rows = [r for r in rows if r[0] and r[1]]
    total = spec.get("total") if isinstance(spec.get("total"), dict) else None
    if len(rows) < 2:
        return None
    title = " ".join(str(spec.get("title") or "").split())[:60]
    tlabel = " ".join(str(total.get("label") or "Total").split())[:24] if total else ""
    tvalue = " ".join(str(total.get("value") or "").split())[:14] if total else ""
    W, H = look.W, look.H
    mx = int(W * 0.12)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    lsize = look.px(58 if look.portrait else 54)
    while lsize > look.px(22) and max(d0.textlength(r[0], font=font("bold", lsize)) + d0.textlength(r[1], font=font("bold", lsize)) for r in rows) > (W - 2 * mx) * 0.88:
        lsize = int(lsize * 0.92)                                                       # label and amount must both fit on one line
    lf = font("bold", lsize)
    rows = [(r[0] if d0.textlength(r[0], font=lf) + d0.textlength(r[1], font=lf) <= (W - 2 * mx) * 0.9 else (r[0][:max(4, int(len(r[0]) * ((W - 2 * mx) * 0.9 - d0.textlength(r[1], font=lf)) / max(1.0, d0.textlength(r[0], font=lf))) - 1)].rstrip() + "…"), r[1]) for r in rows]
    tf = font("heavy", _fit_size(look.px(88 if look.portrait else 78), look.case(title) if title else "x", "heavy", W - 2 * mx, 2, look.px(34)))
    bigf = font("heavy", look.px(104 if look.portrait else 96))
    n = len(rows) + (1 if tvalue else 0)
    times = _times_of(spec, n, seconds, 0.8, 1.6)
    rowh = int(lf.size * 1.7)
    total_h = (int(tf.size * 1.2) * (len(_wrap(look.case(title), tf, W - 2 * mx, 2)) if title else 0)) + look.px(30) + rowh * len(rows) + (bigf.size + look.px(70) if tvalue else 0)
    y0 = max(int(H * 0.08), (H - total_h) // 2)

    def draw(dr, img, t, ink, dim, acc):
        y = y0
        if title:
            for ln in _wrap(look.case(title), tf, W - 2 * mx, 2):
                dr.text((mx, y), ln, font=tf, fill=(*ink, 255))
                y += int(tf.size * 1.2)
            y += look.px(30)
        for i, (lb, val) in enumerate(rows):
            p = ease_out((t - times[i]) / 0.4)
            if p > 0:
                a = int(255 * p)
                dx = int((1 - p) * look.px(50))
                dr.text((mx + dx, y), lb, font=lf, fill=(*ink, a))
                vw = d0.textlength(val, font=lf)
                lw = d0.textlength(lb, font=lf)
                x_dots = mx + lw + look.px(24)
                while x_dots < W - mx - vw - look.px(28):                                    # the dotted leader between the item and its amount
                    dr.ellipse((x_dots, y + lf.size * 0.75, x_dots + 4, y + lf.size * 0.75 + 4), fill=(*dim, a))
                    x_dots += look.px(18)
                dr.text((W - mx - vw - dx, y), val, font=lf, fill=(*acc, a))
            y += rowh
        if tvalue:
            p = ease_out((t - times[-1]) / 0.5)
            if p > 0:
                bwid = int((W - 2 * mx) * p)
                dr.rectangle((mx, y + look.px(6), mx + bwid, y + look.px(6) + max(5, look.px(8))), fill=(*ink, 255))
                a = int(255 * p)
                dr.text((mx, y + look.px(34)), look.case(tlabel), font=lf, fill=(*ink, a))
                vw = d0.textlength(tvalue, font=bigf)
                dr.text((W - mx - vw, y + look.px(14)), tvalue, font=bigf, fill=(*acc, a))
    return _fullscreen(look, spec, seconds, draw)


def scene_recap(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """The bottom line: the full-screen slide, with a default title."""
    return scene_fullslide(look, {**spec, "title": spec.get("title") or "The bottom line"}, seconds, wf)


def _clean(x, n: int) -> str:
    return " ".join(str(x or "").split())[:n]


def _title_block(look: Look, title: str, mx: int, W: int, base_size: int = 104, lines: int = 2):
    """The slide title used by the explainer cards: biggest type that fits, an accent bar under it. -> (font, lines, size, height)."""
    tsize = _fit_size(look.px(base_size), look.case(title), "heavy", W - 2 * mx, lines, look.px(34))
    tf = font("heavy", tsize)
    tl = _wrap(look.case(title), tf, W - 2 * mx, lines)
    return tf, tl, tsize, len(tl) * int(tsize * 1.15) + look.px(30)


def _draw_title(look: Look, dr, t: float, tf, tl, tsize, mx: int, y: int, ink, acc) -> int:
    for ln in tl:
        dr.text((mx, y), ln, font=tf, fill=(*ink, 255))
        y += int(tsize * 1.15)
    bar_w = int(min(look.W * 0.3, tsize * 4.5) * ease_out((t - 0.3) / 0.6))
    dr.rectangle((mx, y + look.px(6), mx + bar_w, y + look.px(6) + max(6, look.px(9))), fill=(*acc, 255))
    return y + look.px(30)


def scene_flow(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A process, step by step: boxes arrive one after another (as each is said), arrows draw toward the next, and a small pulse then runs along the chain."""
    title = _clean(spec.get("title"), 60)
    steps = [_clean(x, 44) for x in (spec.get("steps") or []) if _clean(x, 44)][:5]
    if len(steps) < 3 or not title or look.portrait:                                      # (a row of boxes needs a wide frame: portrait falls back to a slide)
        return None
    W, H = look.W, look.H
    mx = int(W * 0.07)
    n = len(steps)
    gap = look.px(78)
    bw = (W - 2 * mx - gap * (n - 1)) // n
    bh = int(min(H * 0.44, bw * 1.2))
    pad = look.px(28)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    fsize = look.px(66)
    while True:                                                                            # type small enough that every step sits inside its box
        f = font("bold", fsize)
        lines = [_wrap(look.case(x), f, bw - 2 * pad, 4) for x in steps]
        if (all(d0.textlength(l_, font=f) <= bw - 2 * pad for ls in lines for l_ in ls) and all(len(ls) * int(fsize * 1.18) <= bh - 2 * pad - look.px(30) for ls in lines)) or fsize <= look.px(18):
            break
        fsize = int(fsize * 0.92)
    tf, tl, tsize, th = _title_block(look, title, mx, W)
    y_title = int(H * 0.09)
    by = max(y_title + th + look.px(60), (H - bh) // 2 + look.px(40))
    cy = by + bh // 2
    step = max(0.3, min(1.0, (seconds - 3.0) / n))
    at = _times_of(spec, n, seconds, 0.8, 2.2) if spec.get("times") else [0.8 + i * step for i in range(n)]
    pg = _page(look, spec.get("bg"))[0].getpixel((2, 2))

    def draw(dr, img, t, ink, dim, acc):
        _draw_title(look, dr, t, tf, tl, tsize, mx, y_title, ink, acc)
        for i in range(n - 1):                                                              # arrows first, so the boxes sit on top of their ends
            x0, x1 = mx + i * (bw + gap) + bw, mx + (i + 1) * (bw + gap)
            q = ease_out((t - at[i + 1] + 0.5) / 0.5)
            if q > 0:
                xe = int(x0 + (x1 - x0) * q)
                dr.line((x0, cy, xe, cy), fill=(*acc, 255), width=max(5, look.px(8)))
                if q > 0.97:
                    ah = look.px(20)
                    dr.polygon([(x1 - 2, cy), (x1 - 2 - ah, cy - ah // 2 - 4), (x1 - 2 - ah, cy + ah // 2 + 4)], fill=(*acc, 255))
        for i in range(n):
            x = mx + i * (bw + gap)
            p = ease_back((t - at[i]) / 0.55, 1.4)
            a = ease_out((t - at[i]) / 0.3)
            if a <= 0:
                continue
            last = i == n - 1
            dy = int((1 - min(1.0, p)) * look.px(46))
            fill = (*acc, int(255 * a)) if last else (*_lerp(pg, ink, 0.07), int(255 * a))
            txt = (12, 12, 14) if (last and _lum(acc) > 0.5) else (255, 255, 255) if last else ink
            dr.rounded_rectangle((x, by + dy, x + bw, by + bh + dy), radius=look.px(26), fill=fill, outline=(*acc, int(255 * a)), width=max(4, look.px(6)))
            _CTX.mono = look.mono
            nf = font("heavy", int(fsize * 0.8))
            _CTX.mono = False
            dr.text((x + pad, by + dy + pad * 0.6), str(i + 1), font=nf, fill=(*(txt if last else acc), int(255 * a)))
            ty = by + dy + pad * 0.6 + nf.size * 1.15
            for ln in lines[i]:
                dr.text((x + pad, ty), ln, font=f, fill=(*txt, int(255 * a)))
                ty += int(fsize * 1.18)
        t_run = t - (at[-1] + 0.9)
        if t_run > 0:                                                                       # the pulse: a dot running along the whole chain, again and again
            ph = (t_run % 2.4) / 2.4
            px_ = mx + bw // 2 + (mx + (n - 1) * (bw + gap) + bw // 2 - (mx + bw // 2)) * ease_out(ph)
            r = look.px(13)
            dr.ellipse((px_ - r, cy - r, px_ + r, cy + r), fill=(*acc, int(255 * (1 - abs(ph - 0.5) * 1.2))))
    return _fullscreen(look, spec, seconds, draw)


def _num(v) -> tuple[float, int, str] | None:
    """'1,800' / '35%' / '7.5' -> (value, decimals, suffix) or None."""
    m = re.match(r"^\s*[^\d\-]*(-?[\d][\d,]*\.?\d*)\s*(.*)$", str(v))
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw), (len(raw.split(".")[1]) if "." in raw else 0), m.group(2).strip()[:8]
    except ValueError:
        return None


def scene_bars(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """Bars that grow one after another, each with its figure counting up; the biggest is the accent colour."""
    title = _clean(spec.get("title"), 60)
    labels = [_clean(x, 26) for x in (spec.get("labels") or [])]
    vals = [_num(v) for v in (spec.get("values") or [])]
    n = min(len(labels), len(vals), 6)
    unit = _clean(spec.get("unit"), 40)
    if n < 2 or not title or any(v is None or v[0] < 0 for v in vals[:n]) or max(v[0] for v in vals[:n]) <= 0:
        return None
    labels, vals = labels[:n], vals[:n]
    W, H = look.W, look.H
    mx = int(W * 0.07)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    tf, tl, tsize, th = _title_block(look, title, mx, W)
    y_title = int(H * 0.09)
    top = y_title + th + look.px(50)
    avail = H - top - int(H * 0.09)
    rh = min(look.px(120), avail // n)
    lsize = _fit_size(look.px(48), max(labels, key=len), "bold", int(W * 0.26), 2, look.px(18))
    lf = font("bold", lsize)
    lw = int(W * 0.27)
    vf = font("heavy", int(rh * 0.5))
    bar_x = mx + lw + look.px(24)
    widest = max(d0.textlength(f"{v[0]:,.{v[1]}f}{v[2] or ''}" if "," in str(spec["values"][i]) else f"{v[0]:.{v[1]}f}{v[2] or ''}", font=vf) for i, v in enumerate(vals))
    bar_w = max(look.px(200), W - bar_x - mx - int(widest) - look.px(40))                 # the figure beside the longest bar must fit inside the frame
    big = max(v[0] for v in vals)
    big_i = [v[0] for v in vals].index(big)
    at = [0.8 + i * 0.4 for i in range(n)]

    def draw(dr, img, t, ink, dim, acc):
        _draw_title(look, dr, t, tf, tl, tsize, mx, y_title, ink, acc)
        for i in range(n):
            y = top + i * rh + (rh - int(rh * 0.62)) // 2
            h = int(rh * 0.62)
            a = ease_out((t - at[i]) / 0.3)
            g = ease_out((t - at[i]) / 1.1)
            if a <= 0:
                continue
            lines = _wrap(look.case(labels[i]), lf, lw, 2)
            ly = y + h // 2 - len(lines) * int(lsize * 1.1) // 2
            for j, ln in enumerate(lines):
                dr.text((mx, ly + j * int(lsize * 1.1)), ln, font=lf, fill=(*ink, int(255 * a)))
            length = int(bar_w * (vals[i][0] / big) * g)
            col = acc if i == big_i else _lerp(dim, ink, 0.35)
            dr.rounded_rectangle((bar_x, y, bar_x + max(length, 2), y + h), radius=look.px(10), fill=(*col, int(255 * a)))
            v, dec, suf = vals[i]
            shown = f"{v * g:,.{dec}f}" if "," in str(spec["values"][i]) else f"{v * g:.{dec}f}"
            dr.text((bar_x + max(length, 2) + look.px(18), y + h // 2 - vf.size * 0.56), f"{shown}{suf or ''}", font=vf, fill=(*ink, int(255 * a)))
        if unit:
            uf = font("medium", look.px(34))
            dr.text((bar_x, top + n * rh + look.px(10)), unit, font=uf, fill=(*dim, 255))
    return _fullscreen(look, spec, seconds, draw)


def scene_line(look: Look, spec: dict, seconds: float, wf: float = 1.0) -> Scene | None:
    """A timeline that draws itself: the line runs across, and each date pops on it with what happened under."""
    title = _clean(spec.get("title"), 60)
    ev = [(_clean(e.get("date"), 14), _clean(e.get("label"), 60)) for e in (spec.get("events") or []) if isinstance(e, dict) and e.get("date") and e.get("label")][:5]
    if len(ev) < 3 or not title:
        return None
    W, H = look.W, look.H
    mx = int(W * 0.08)
    n = len(ev)
    d0 = ImageDraw.Draw(Image.new("RGBA", (4, 4)))
    tf, tl, tsize, th = _title_block(look, title, mx, W)
    y_title = int(H * 0.09)
    cy = int(H * 0.56)
    span = W - 2 * mx
    seg = span / n
    xs = [mx + seg * (i + 0.5) for i in range(n)]
    lsize = _fit_size(look.px(58), max((l_ for _d, l_ in ev), key=len), "bold", int(seg * 0.84), 4, look.px(20))
    longest = max((w_ for _d, l_ in ev for w_ in l_.split()), key=len)
    while d0.textlength(longest, font=font("bold", lsize)) > seg * 0.84 and lsize > look.px(20):          # one long word ("Construction") must fit its own column
        lsize -= 2
    lf = font("bold", lsize)
    df = font("heavy", int(lsize * 1.75))
    dates = [d_ for d_, _l in ev]
    widest = max(d0.textlength(d_, font=df) for d_ in dates)
    if widest > seg * 0.9:                                            # a long date ("25 Dec 2021") must not run into its neighbour
        df = font("heavy", max(look.px(22), int(df.size * seg * 0.9 / widest)))
    step = max(0.3, min(1.1, (seconds - 3.0) / n))
    at = _times_of(spec, n, seconds, 0.9, 2.2) if spec.get("times") else [0.9 + i * step for i in range(n)]

    def draw(dr, img, t, ink, dim, acc):
        _draw_title(look, dr, t, tf, tl, tsize, mx, y_title, ink, acc)
        reach = ease_out((t - 0.4) / (at[-1] - 0.4 + 0.5))
        dr.line((mx, cy, mx + int(span * reach), cy), fill=(*dim, 255), width=max(5, look.px(8)))
        for i, (d_, l_) in enumerate(ev):
            a = ease_out((t - at[i]) / 0.35)
            if a <= 0:
                continue
            r = int(look.px(20) * ease_back((t - at[i]) / 0.45, 1.6))
            dr.ellipse((xs[i] - r, cy - r, xs[i] + r, cy + r), fill=(*acc, 255))
            dw = d0.textlength(d_, font=df)
            dr.text((xs[i] - dw / 2, cy - look.px(36) - df.size * 1.15 - (1 - a) * look.px(20)), d_, font=df, fill=(*ink, int(255 * a)))
            for j, ln in enumerate(_wrap(l_, lf, int(seg * 0.84), 4)):
                lw_ = d0.textlength(ln, font=lf)
                dr.text((xs[i] - lw_ / 2, cy + look.px(40) + j * int(lsize * 1.2) + (1 - a) * look.px(20)), ln, font=lf, fill=(*_lerp(dim, ink, a), int(255 * a)))
    return _fullscreen(look, spec, seconds, draw)


SCENES = {"hook": lambda l, s, sec, wf=1.0: scene_lower(l, s, sec, "hook", wf), "point": lambda l, s, sec, wf=1.0: scene_lower(l, s, sec, "point", wf),
          "lower": lambda l, s, sec, wf=1.0: scene_lower(l, s, sec, "lower", wf), "stat": scene_stat, "quote": scene_quote, "slide": scene_slide, "chart": scene_chart, "entity": scene_entity, "hub": scene_hub,
          "chips": scene_chips, "compare": scene_compare, "timeline": scene_timeline, "statement": scene_statement,
          "fullslide": scene_fullslide, "bigstat": scene_bigstat, "chapter": scene_chapter, "fullquote": scene_fullquote,
          "search": scene_search, "twocol": scene_twocol, "ledger": scene_ledger, "recap": scene_recap,
          "flow": scene_flow, "bars": scene_bars, "line": scene_line}


def _skia_or(kind: str, fallback):
    """A Skia-drawn kind (skiagfx), or its Pillow cousin when skia isn't installed or the drawing fails: the graphic is never lost."""
    def build(look, spec, seconds, wf=1.0):
        try:
            from . import skiagfx
            if skiagfx.available():
                sc = skiagfx.SCENES[kind](look, spec, seconds, wf)
                if sc is not None:
                    return sc
        except Exception as e:  # noqa: BLE001
            log.warning(f"graphic {kind}: skia drawing failed ({e}); the plain version instead")
        return fallback(look, spec, seconds, wf)
    return build


SCENES.update({"drawchart": _skia_or("drawchart", lambda l, s, sec, wf=1.0: scene_chart(l, {**s, "chart": s.get("chart") or s}, sec, wf)),
               "count": _skia_or("count", scene_bigstat), "kinetic": _skia_or("kinetic", scene_statement)})


# ------------------------------------------------------------------------------------------------ rendering
def _alpha_mul(img: Image.Image, a: float) -> Image.Image:
    if a >= 0.999:
        return img
    out = img.copy()
    out.putalpha(img.getchannel("A").point([int(i * a) for i in range(256)]))
    return out


def _scaled(img: Image.Image, k: float) -> Image.Image:
    if abs(k - 1.0) < 0.004:
        return img
    return img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.BICUBIC)


def frame_at(sc: Scene, t: float) -> Image.Image:
    look = sc.look
    canvas = Image.new("RGBA", (look.W, look.H), (0, 0, 0, 0))
    secs = sc.seconds
    q = 0.0 if sc.custom else ease_in((t - (secs - EXIT)) / EXIT)                    # 0 while it is on screen, 1 when it has gone
    if t >= secs or secs - t <= 0.001:
        return canvas

    def put(img: Image.Image, x: int, y: int):
        if x >= look.W or y >= look.H or x + img.width <= 0 or y + img.height <= 0:
            return
        sx, sy = max(0, -x), max(0, -y)
        crop = img.crop((sx, sy, min(img.width, look.W - x), min(img.height, look.H - y)))
        canvas.alpha_composite(crop, (x + sx, y + sy))
    for L in sc.layers:
        raw = (t - L["at"]) / L["dur"]
        if raw <= 0:
            continue
        p = ease_back(raw) if L["pop"] < 1.0 or L["sx"] else ease_out(raw)           # a pop overshoots; a plain reveal just settles
        alpha = min(1.0, ease_out(raw * 1.6)) * (1.0 - q)
        k = L["pop"] + (1.0 - L["pop"]) * p
        img = _scaled(_alpha_mul(L["img"], alpha), k * (1.0 - 0.06 * q))
        cx = L["x"] + L["img"].width / 2 + (1 - p) * L["sx"] + q * L["exit_x"]
        cy = L["y"] + L["img"].height / 2 + (1 - min(p, 1.0)) * L["rise"]
        put(img, int(cx - img.width / 2), int(cy - img.height / 2))
    for fn, x, y, exit_x in sc.draw_fns:
        img = fn(t)
        if img is not None:
            if sc.custom:
                put(img, int(x), y)
            else:
                put(_alpha_mul(img, min(1.0, ease_out(t / ENTER)) * (1.0 - q)), int(x + q * exit_x), y)
    return canvas


# ------------------------------------------------------------------------------------------------ motion blur
BLUR_SUB = 5                         # sub-frames averaged while something moves (a 180-degree shutter: half the frame interval)
SHUTTER = 0.5
FULL_IN, FULL_OUT = 0.5, 0.45        # (a full-screen card's own transition times: _transition)


def moving(sc: Scene, t: float) -> bool:
    """Is anything in the card moving at time t? Only then is it worth blurring (a still frame costs nothing extra)."""
    secs = sc.seconds
    if sc.custom:                                                                        # full-screen: its transition in and out
        return t < FULL_IN or t > secs - FULL_OUT
    if t > secs - EXIT:
        return True
    for L in sc.layers:
        raw = (t - L["at"]) / L["dur"]
        if 0.0 < raw < 1.0 and (L["sx"] or L["rise"] or L["pop"] < 1.0 or L["exit_x"]):
            return True
    return bool(sc.draw_fns) and t < ENTER


def frame_blurred(sc: Scene, t: float, fps: float = FPS, sub: int = BLUR_SUB) -> Image.Image:
    """frame_at with motion blur: while the card moves, `sub` moments across the shutter are averaged (alpha-weighted, so edges fade
    correctly) - a sliding lower third leaves a soft trail instead of stepping (the research note on Remotion's motion blur). Only the box
    that has something in it is blended (a lower third is a small part of the frame), and a full-screen card's short wipe uses 3."""
    if sc.custom:
        sub = min(sub, 3)
    if sub <= 1 or not moving(sc, t):
        return frame_at(sc, t)
    subs = [frame_at(sc, max(0.0, t + (k / (sub - 1) - 0.5) * SHUTTER / fps)) for k in range(sub)]
    box = None
    for im in subs:
        bb = im.getchannel("A").getbbox()
        if bb:
            box = bb if box is None else (min(box[0], bb[0]), min(box[1], bb[1]), max(box[2], bb[2]), max(box[3], bb[3]))
    out = Image.new("RGBA", subs[0].size, (0, 0, 0, 0))
    if box is None:
        return out
    acc = None
    for im in subs:
        a = np.asarray(im.crop(box), dtype=np.float32)
        a[..., :3] *= a[..., 3:4] * (1.0 / 255.0)
        acc = a if acc is None else acc + a
    acc *= 1.0 / sub
    alpha = acc[..., 3:4]
    rgb = np.where(alpha > 0.5, acc[..., :3] * 255.0 / np.maximum(alpha, 1e-3), 0.0)
    out.paste(Image.fromarray(np.clip(np.concatenate([rgb, alpha], axis=-1), 0, 255).astype(np.uint8), "RGBA"), box[:2])
    return out


def render(cfg, spec: dict, W: int, H: int, out: Path, theme: str | None = None, timeout: float = 600.0, avoid: list | None = None) -> dict | None:
    """Render one card to `out` (a transparent .mov). -> {"file", "glass"} with the glass rectangles in W x H pixels, or None on failure."""
    kind = spec.get("kind")
    fn = SCENES.get(kind)
    if fn is None:
        return None
    seconds = float(spec.get("len") or 4.0)
    base = 1080.0 / min(W, H)                                  # compose near 1080 and let ffmpeg scale up: quick on big frames
    cw, ch = int(round(W * min(1.0, base) / 2) * 2), int(round(H * min(1.0, base) / 2) * 2)
    try:
        look = Look(theme, cw, ch)
        look.avoid = [(a * cw, b * ch, c * cw, d * ch) for a, b, c, d in (avoid or [])]               # zones come as 0..1 of the frame
        sc = build_scene(kind, look, spec, seconds)
        if sc is None:
            log.info(f"graphic {kind} dropped: no place for it that leaves every face clear")
            return None
        n = max(2, int(round(seconds * FPS)))
        cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{cw}x{ch}", "-r", str(FPS), "-i", "-",
               "-vf", f"scale={W}:{H}:flags=lanczos,format=argb", "-c:v", "qtrle", "-pix_fmt", "argb", str(out)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        blur = True
        try:
            blur = bool(cfg.get("clipper.ai_videos.motion_blur", True)) if cfg is not None and hasattr(cfg, "get") else True
        except Exception:  # noqa: BLE001
            pass
        for i in range(n):
            p.stdin.write((frame_blurred(sc, i / FPS) if blur else frame_at(sc, i / FPS)).tobytes())
        p.stdin.close()
        err = p.stderr.read().decode(errors="ignore")
        if p.wait(timeout=timeout) != 0 or not out.exists():
            log.warning(f"graphic {kind} failed: {err[-300:]}")
            return None
    except Exception as e:  # noqa: BLE001 - a graphic is never worth losing a video over
        log.warning(f"graphic {kind} failed: {e}")
        return None
    k = W / cw
    return {"file": out, "glass": [{"x": int(g["x"] * k), "y": int(g["y"] * k), "w": int(g["w"] * k), "h": int(g["h"] * k), "r": int(g["r"] * k), "dim": g["dim"], "tone": g.get("tone", "dark"), "t0": g.get("t0", 0.0)}
                                   for g in sc.glass]}
