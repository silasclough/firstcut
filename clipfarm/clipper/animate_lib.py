"""Topic-specific animated diagrams, tier A: building blocks.

    The AI describes a scene as DATA (gears that mesh, pulleys and belts, sliding parts, boxes, arrows, labels, counters, each with when it appears and how it
    moves) and this module draws it with Pillow, frame by frame, in the look of the cards. No browser, no code from the AI is run: the worst a bad
    description can do is draw something wrong, and the checks here (geometry, known parts) and a vision pass catch most of that.

    make(cfg, ai, concept, narration, words, duration, look, out_dir, source_text="") -> {"file", "attempts", "notes"} | None
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw

from ..core import ffmpeg
from ..core.log import get_logger
from . import animate_hf as AH
from . import gfxcards as GC

log = get_logger("clipper.animate_lib")
W, H, FPS, SS = 1920, 1080, 30, 2
TYPES = ("gear", "pulley", "belt", "box", "slider", "arrow", "label", "counter", "line")

NUM = {"type": "number"}
SCHEMA = {"type": "object", "properties": {
    "notes": {"type": "string"},
    "elements": {"type": "array", "items": {"type": "object", "properties": {
        "type": {"type": "string", "enum": list(TYPES)}, "id": {"type": "string"},
        "x": NUM, "y": NUM, "x2": NUM, "y2": NUM, "w": NUM, "h": NUM, "r": NUM, "r_to": NUM, "r_start": NUM, "r_dur": NUM,
        "teeth": {"type": "integer"}, "text": {"type": "string"}, "size": NUM, "color": {"type": "string", "enum": ["ink", "dim", "accent"]},
        "fill": {"type": "boolean"}, "at": NUM, "dur": NUM,
        "mesh": {"type": "string"}, "spin_dps": NUM, "spin_start": NUM,
        "move_dx": NUM, "move_dy": NUM, "move_start": NUM, "move_dur": NUM,
        "a": {"type": "string"}, "b": {"type": "string"},
        "from_n": NUM, "to_n": NUM, "suffix": {"type": "string"}}, "required": ["type"]}}}, "required": ["elements"]}

DOC = """ELEMENTS (all coordinates in a 1920x1080 stage; x,y = centre unless said; sizes in px):
 gear    {id, x, y, r (tip radius, 90-260), teeth (8-40), color, fill?, at, spin_dps (degrees/second, + = clockwise) OR mesh (id of the gear that drives it: it turns the opposite
          way at dps*teeth_driver/teeth_this, and you place it so the two tips touch: distance between centres = r1 + r2 minus about 12px), spin_start}
 pulley  {id, x, y, r (60-240), color, at, spin_dps, spin_start, r_to + r_start + r_dur (the radius grows or shrinks from r to r_to, starting r_start for r_dur seconds)}
 belt    {a, b (ids of two pulleys; it follows their current radii), color, at}
 slider  {id, x, y (top-left), w, h, text?, color, fill?, at, move_dx, move_dy, move_start, move_dur (slides by that amount)}   a part that moves in a line
 box     {x, y (top-left), w, h, text, size, color, fill?, at}
 arrow   {x, y -> x2, y2, color, at, dur (it draws itself on over dur seconds)}
 line    {x, y -> x2, y2, color, at, dur}
 label   {text (max 3 words), x, y (centre of the baseline), size (>= 46), color, at}
 counter {x, y, from_n, to_n, suffix, size, color, at, dur}   a number that counts up/down (only numbers the source gives)
 color is one of ink, dim, accent. at = the second the element appears (fades in). Order matters: later elements draw on top. Up to 22 elements."""

SYSTEM = (
    "You are a motion designer making ONE short animated diagram for an explainer video from a fixed set of building blocks. You receive the CONCEPT to show, the "
    "NARRATION spoken over it with the time each word is said, the SOURCE TEXT (the only place a fact, name or number may come from) and the duration D. "
    "Return JSON {notes, elements}.\n{doc}\n"
    "RULES: (1) show the mechanism WORKING: parts turn, slide, grow or count the way the narration describes, in its order; a still arrangement is a failure; keep "
    "something moving until D. (2) Time parts to the narration: an element appears (at) when the narrator first says what it is; everything is on screen by D-0.8. "
    "(3) Labels: 1-3 words from the source, size >= 46, never overlapping another label or a shape's edge; no numbers or facts the source does not give "
    "(show a ratio by drawing a small gear and a big gear, not by writing a number). (4) Keep every shape inside x 110..1810 and y 90..900 (the bottom 140px is "
    "for captions). (5) One clear idea drawn large: the main shapes 260px+ across, centred on the stage. (6) Gears that mesh must touch tip to tip and "
    "share the same tooth size: teeth proportional to r (tooth pitch about 26px, so teeth = 2*pi*r/26 rounded). (7) 6-18 elements. "
)


def _rgb(hexs: str) -> tuple[int, int, int]:
    h = hexs.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _ease_out(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 1 - (1 - x) ** 3


def _ease_io(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return 3 * x * x - 2 * x ** 3


def problems(spec: dict) -> list[str]:
    """Geometry and sanity checks on a description, in plain words for the AI to fix. Empty = fine."""
    els = spec.get("elements") or []
    out = []
    if not 3 <= len(els) <= 24:
        out.append(f"use between 3 and 22 elements (you gave {len(els)})")
    ids = {e.get("id") for e in els if e.get("id")}
    for i, e in enumerate(els):
        t, nm = e.get("type"), e.get("id") or f"#{i} {e.get('type')}"
        if t == "gear":
            if not e.get("r") or not e.get("teeth") or e.get("x") is None or e.get("y") is None:
                out.append(f"gear {nm} needs x, y, r and teeth")
                continue
            if e.get("mesh") and e["mesh"] not in ids:
                out.append(f"gear {nm} meshes with '{e['mesh']}' which does not exist")
            if e["x"] - e["r"] < 100 or e["x"] + e["r"] > W - 100 or e["y"] - e["r"] < 80 or e["y"] + e["r"] > 900:
                out.append(f"gear {nm} reaches outside the safe area (x 100..1820, y 80..900)")
        elif t == "pulley":
            r = max(float(e.get("r") or 0), float(e.get("r_to") or 0))
            if not r or e.get("x") is None or e.get("y") is None:
                out.append(f"pulley {nm} needs x, y and r")
            elif e["x"] - r < 100 or e["x"] + r > W - 100 or e["y"] - r < 80 or e["y"] + r > 900:
                out.append(f"pulley {nm} reaches outside the safe area")
        elif t == "belt":
            if e.get("a") not in ids or e.get("b") not in ids:
                out.append(f"belt {nm} needs the ids of two pulleys that exist")
        elif t in ("box", "slider"):
            if None in (e.get("x"), e.get("y"), e.get("w"), e.get("h")):
                out.append(f"{t} {nm} needs x, y, w, h")
            else:
                dx, dy = float(e.get("move_dx") or 0), float(e.get("move_dy") or 0)
                xs = [e["x"], e["x"] + dx, e["x"] + e["w"], e["x"] + e["w"] + dx]
                ys = [e["y"], e["y"] + dy, e["y"] + e["h"], e["y"] + e["h"] + dy]
                if min(xs) < 100 or max(xs) > W - 100 or min(ys) < 80 or max(ys) > 900:
                    out.append(f"{t} {nm} (with its move) reaches outside the safe area")
        elif t == "label":
            if not e.get("text") or len(e["text"].split()) > 4:
                out.append(f"label {nm} must have 1-3 words")
            if float(e.get("size") or 0) < 44:
                out.append(f"label '{e.get('text')}' is smaller than 46px")
            if e.get("x") is None or e.get("y") is None or not 100 <= e["x"] <= W - 100 or not 90 <= e["y"] <= 930:
                out.append(f"label '{e.get('text')}' is outside the safe area")
        elif t in ("arrow", "line"):
            if None in (e.get("x"), e.get("y"), e.get("x2"), e.get("y2")):
                out.append(f"{t} {nm} needs x, y, x2, y2")
        elif t == "counter":
            if e.get("to_n") is None:
                out.append(f"counter {nm} needs to_n")
    gears = [e for e in els if e.get("type") == "gear"]
    for e in gears:                                                                   # a meshing pair must really touch
        if e.get("mesh"):
            p = next((g for g in gears if g.get("id") == e["mesh"]), None)
            if p and None not in (e.get("x"), p.get("x")):
                d = math.hypot(e["x"] - p["x"], e["y"] - p["y"])
                if abs(d - (e["r"] + p["r"] - 12)) > 40:
                    out.append(f"gear {e.get('id')} should touch gear {p.get('id')}: put the centres {e['r'] + p['r'] - 12:.0f}px apart (they are {d:.0f}px apart)")
    return out


class Scene:
    def __init__(self, spec: dict, tokens: dict, look: tuple[str, str]):
        self.els = spec["elements"]
        self.by = {e["id"]: e for e in self.els if e.get("id")}
        self.col = {k: _rgb(tokens[k]) for k in ("ink", "dim", "accent")}
        lk = GC.Look(look[0], W, H)
        base, _ink, _dim, _dot = GC._page(lk, look[1])
        self.base = base.convert("RGBA")

    # -- state of one element at time t
    def angle(self, e: dict, t: float, depth: int = 0) -> float:
        if e.get("mesh") and e["mesh"] in self.by and depth < 6:
            p = self.by[e["mesh"]]
            return -self.angle(p, t, depth + 1) * (p["teeth"] / e["teeth"]) + 180.0 / e["teeth"]
        s0 = float(e.get("spin_start") or e.get("at") or 0.0)
        return float(e.get("spin_dps") or 0.0) * max(0.0, t - s0)

    def radius(self, e: dict, t: float) -> float:
        r = float(e.get("r") or 100)
        if e.get("r_to") is not None:
            f = _ease_io((t - float(e.get("r_start") or 0)) / max(0.2, float(e.get("r_dur") or 2)))
            r = r + (float(e["r_to"]) - r) * f
        return r

    def alpha(self, e: dict, t: float) -> float:
        return _ease_out((t - float(e.get("at") or 0)) / 0.45)

    def color(self, e: dict, a: float) -> tuple:
        return (*self.col.get(e.get("color") or "ink", self.col["ink"]), int(255 * a))

    # -- drawing (on the supersampled layer: every coordinate times SS)
    def draw_gear(self, d: ImageDraw.ImageDraw, e: dict, t: float, a: float) -> None:
        n, ro = int(e["teeth"]), float(e["r"])
        ri = ro * (1 - min(0.2, 2.6 / n + 0.05))
        ang = math.radians(self.angle(e, t))
        step = 2 * math.pi / n
        pts = []
        for i in range(n):
            base = ang + i * step
            for r, off in ((ri, -0.3), (ro, -0.15), (ro, 0.15), (ri, 0.3)):
                pts.append(((e["x"] + r * math.cos(base + step * off)) * SS, (e["y"] + r * math.sin(base + step * off)) * SS))
        c = self.color(e, a)
        if e.get("fill"):
            d.polygon(pts, fill=(*c[:3], int(70 * a)))
        d.line(pts + [pts[0]], fill=c, width=8 * SS, joint="curve")
        hub = ri * 0.3
        d.ellipse(((e["x"] - hub) * SS, (e["y"] - hub) * SS, (e["x"] + hub) * SS, (e["y"] + hub) * SS), outline=c, width=7 * SS)
        for k in range(4):
            aa = ang + k * math.pi / 2 + math.pi / 4
            d.line(((e["x"] + hub * math.cos(aa)) * SS, (e["y"] + hub * math.sin(aa)) * SS, (e["x"] + ri * 0.82 * math.cos(aa)) * SS, (e["y"] + ri * 0.82 * math.sin(aa)) * SS),
                   fill=c, width=6 * SS)

    def draw_pulley(self, d, e, t, a) -> None:
        r = self.radius(e, t)
        ang = math.radians(self.angle(e, t))
        c = self.color(e, a)
        x, y = e["x"], e["y"]
        d.ellipse(((x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS), outline=c, width=8 * SS)
        d.ellipse(((x - r * 0.2) * SS, (y - r * 0.2) * SS, (x + r * 0.2) * SS, (y + r * 0.2) * SS), fill=c)
        for k in range(3):
            aa = ang + k * 2 * math.pi / 3
            d.line((x * SS, y * SS, (x + r * 0.86 * math.cos(aa)) * SS, (y + r * 0.86 * math.sin(aa)) * SS), fill=c, width=6 * SS)

    def draw_belt(self, d, e, t, a) -> None:
        pa, pb = self.by.get(e.get("a")), self.by.get(e.get("b"))
        if not pa or not pb:
            return
        ra, rb = self.radius(pa, t), self.radius(pb, t)
        dx, dy = pb["x"] - pa["x"], pb["y"] - pa["y"]
        dist = math.hypot(dx, dy)
        if dist < 1:
            return
        base = math.atan2(dy, dx)
        al = math.acos(max(-1.0, min(1.0, (ra - rb) / dist)))
        # the belt: round the far side of B (from the top tangent down to the bottom tangent), straight back, then round the far side of A
        arcB = [base + al - (2 * al) * (i / 40) for i in range(41)]                    # angles on B from (base+al) down to (base-al): the side facing away from A is through 'base'
        arcA = [base - al - (2 * (math.pi - al)) * (i / 40) for i in range(41)]       # on A from (base-al) round the far side to (base+al-2pi)
        pts = [(pb["x"] + rb * math.cos(g), pb["y"] + rb * math.sin(g)) for g in arcB] + [(pa["x"] + ra * math.cos(g), pa["y"] + ra * math.sin(g)) for g in arcA]
        pts = [(x * SS, y * SS) for x, y in pts]
        d.line(pts + [pts[0]], fill=self.color(e, a), width=9 * SS, joint="curve")

    def draw_rect(self, d, e, t, a) -> None:
        f = _ease_io((t - float(e.get("move_start") or 0)) / max(0.2, float(e.get("move_dur") or 1.0)))
        ox, oy = float(e.get("move_dx") or 0) * f, float(e.get("move_dy") or 0) * f
        x0, y0 = e["x"] + ox, e["y"] + oy
        c = self.color(e, a)
        box = (x0 * SS, y0 * SS, (x0 + e["w"]) * SS, (y0 + e["h"]) * SS)
        if e.get("fill"):
            d.rounded_rectangle(box, radius=16 * SS, fill=(*c[:3], int(70 * a)))
        d.rounded_rectangle(box, radius=16 * SS, outline=c, width=6 * SS)
        if e.get("text"):
            self.text(d, e["text"], x0 + e["w"] / 2, y0 + e["h"] / 2, float(e.get("size") or 50), c, middle=True)

    def text(self, d, s: str, x: float, y: float, size: float, fill: tuple, middle: bool = False) -> None:
        f = GC.font("bold", int(size * SS))
        w = d.textlength(s, font=f)
        d.text((x * SS - w / 2, (y - size * (0.5 if middle else 0.86)) * SS), s, font=f, fill=fill)

    def draw(self, e: dict, d, t: float) -> None:
        a = self.alpha(e, t)
        if a <= 0:
            return
        ty = e["type"]
        if ty == "gear":
            self.draw_gear(d, e, t, a)
        elif ty == "pulley":
            self.draw_pulley(d, e, t, a)
        elif ty == "belt":
            self.draw_belt(d, e, t, a)
        elif ty in ("box", "slider"):
            self.draw_rect(d, e, t, a)
        elif ty in ("arrow", "line"):
            p = _ease_io((t - float(e.get("at") or 0)) / max(0.2, float(e.get("dur") or 0.8)))
            x1, y1, x2, y2 = e["x"], e["y"], e["x"] + (e["x2"] - e["x"]) * p, e["y"] + (e["y2"] - e["y"]) * p
            c = self.color(e, 1.0)
            d.line((x1 * SS, y1 * SS, x2 * SS, y2 * SS), fill=c, width=8 * SS)
            if ty == "arrow" and p > 0.97:
                ang = math.atan2(e["y2"] - e["y"], e["x2"] - e["x"])
                hs = 34
                d.polygon([(e["x2"] * SS, e["y2"] * SS), ((e["x2"] - hs * math.cos(ang - 0.45)) * SS, (e["y2"] - hs * math.sin(ang - 0.45)) * SS),
                           ((e["x2"] - hs * math.cos(ang + 0.45)) * SS, (e["y2"] - hs * math.sin(ang + 0.45)) * SS)], fill=c)
        elif ty == "label":
            self.text(d, e["text"], e["x"], e["y"], float(e.get("size") or 52), self.color(e, a))
        elif ty == "counter":
            k = _ease_out((t - float(e.get("at") or 0)) / max(0.3, float(e.get("dur") or 1.2)))
            n = float(e.get("from_n") or 0) + (float(e["to_n"]) - float(e.get("from_n") or 0)) * k
            self.text(d, f"{n:,.0f}{e.get('suffix') or ''}", e["x"], e["y"], float(e.get("size") or 90), self.color(e, a))

    def frame(self, t: float) -> Image.Image:
        lay = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(lay)
        for e in self.els:
            self.draw(e, d, t)
        img = self.base.copy()
        img.alpha_composite(lay.resize((W, H), Image.LANCZOS))
        return img.convert("RGB")


def render(scene: Scene, duration: float, out: Path) -> bool:
    n = max(2, int(round(duration * FPS)))
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(n):
            p.stdin.write(scene.frame(i / FPS).tobytes())
        p.stdin.close()
        err = p.stderr.read().decode(errors="ignore")
        return p.wait(timeout=300) == 0 and out.exists() and not err.strip()
    except Exception as e:  # noqa: BLE001
        log.warning(f"library render failed: {e}")
        return False


def design(ai, concept: str, narration: str, words: list[dict], duration: float, source_text: str, feedback: str = "", previous: dict | None = None) -> dict:
    timed = " ".join(f"{w['text']}@{float(w['start']):.1f}" for w in words if "start" in w)[:2400]
    prompt = f"CONCEPT: {concept}\n\nNARRATION: {narration}\nTIMED WORDS: {timed}\n\nD = {duration:.2f} seconds\n\nSOURCE TEXT:\n{source_text[:9000]}\n"
    if previous and feedback:
        prompt += f"\nYOUR PREVIOUS DESCRIPTION:\n{json.dumps(previous)[:6000]}\n\nPROBLEMS TO FIX (keep what works):\n{feedback}\n"
    return ai.complete_json("flow_anim_design", SYSTEM.replace("{doc}", DOC), prompt, SCHEMA).data


def make(cfg, ai, concept: str, narration: str, words: list[dict], duration: float, look: tuple[str, str], out_dir: Path, source_text: str = "",
         max_rounds: int = 3, say=print) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens = AH.look_tokens(*look)
    final = out_dir / "animation.mp4"
    prev: dict | None = None
    feedback = ""
    t0 = time.time()
    for attempt in range(1, max_rounds + 1):
        try:
            spec = design(ai, concept, narration, words, duration, source_text, feedback, prev)
        except Exception as e:  # noqa: BLE001
            log.warning(f"animation design failed: {e}")
            return None
        (out_dir / f"design_{attempt}.json").write_text(json.dumps(spec, indent=1))
        bad = problems(spec)
        say(f"  attempt {attempt}: {len(spec.get('elements', []))} elements, {len(bad)} geometry problems ({time.time() - t0:.0f}s)")
        if bad:
            prev, feedback = spec, "the checker found:\n- " + "\n- ".join(bad)
            continue
        if not render(Scene(spec, tokens, look), duration, final):
            prev, feedback = spec, "rendering failed: check that every element has the fields its type needs"
            continue
        good, probs = AH.qa(ai, final, concept, narration, duration)
        say(f"  attempt {attempt}: look {'ok' if good else 'problems: ' + '; '.join(probs)} ({time.time() - t0:.0f}s)")
        if good or attempt == max_rounds:
            return {"file": final, "attempts": attempt, "notes": spec.get("notes", ""), "qa_ok": good, "problems": probs, "seconds": time.time() - t0}
        prev, feedback = spec, "a reviewer looking at frames of your render found:\n- " + "\n- ".join(probs)
    return None
