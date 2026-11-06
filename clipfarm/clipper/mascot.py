"""Cogsworth, the Pexode mascot: a friendly blue gear who pops up in the corner when a picture has been on screen too long and "says" the line (his mouth follows the voice).

    pick_segments(segs) -> {segment index: seconds into it when he arrives}
    render(audio, mouth_from, dur, out_dir, size=260, fps=30) -> frame pattern for ffmpeg   (PNG frames with transparency)
    composite(seg_video, frames_dir, start, out, size) -> Path                                   (the segment with him on top, bottom-right)

Drawn with Pillow (flat shapes, a dark outline so he shows on light pages and dark ones), moving: the gear turns while he talks, he bobs, blinks, one arm waves, and the mouth opens
with the loudness of the narration. Everything is deterministic (no random numbers).
"""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..core import ffmpeg

ACC, INK, DARK, WARM = (74, 168, 255), (242, 246, 251), (11, 22, 38), (255, 106, 61)
SS = 2                                                           # supersampling


def pick_segments(segs: list[dict], min_dur: float = 5.0, skip_first_s: float = 2.5, min_gap_s: float = 16.0, arrive_s: float = 1.0, max_n: int | None = None, every_s: float = 18.0) -> dict[int, float]:
    """Which segments get him: the long ones (the picture has been up for a while), never in the hook, never twice within `min_gap_s`, never over the closing card."""
    out: dict[int, float] = {}
    t0, last_end = 0.0, -1e9
    total = sum(s["dur"] for s in segs)
    cap = max_n if max_n is not None else max(1, int(total / every_s))
    for k, s in enumerate(segs):
        if s.get("beat") is not None and s["dur"] >= min_dur and t0 >= skip_first_s and t0 - last_end >= min_gap_s and len(out) < cap and s.get("kind") != "end":
            out[k] = min(arrive_s, max(0.3, s["dur"] - 3.0))
            last_end = t0 + s["dur"]
        t0 += s["dur"]
    return out


def envelope(audio: Path, dur: float, fps: int = 30, ss: float = 0.0) -> np.ndarray:
    """Mouth opening 0..1 per video frame, from the loudness of the narration."""
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{ss:.3f}", "-i", str(audio), "-t", f"{dur:.3f}", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True)
    x = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float64) / 32768.0
    n = int(dur * fps)
    if len(x) < 160 or n <= 0:
        return np.zeros(max(n, 0))
    hop = 16000 // fps
    rms = np.array([np.sqrt(np.mean(x[i * hop:(i + 1) * hop] ** 2) + 1e-12) if i * hop < len(x) else 0.0 for i in range(n)])
    hi = np.percentile(rms, 92) or 1.0
    m = np.clip(rms / hi, 0, 1)
    m = np.where(m < 0.18, 0.0, m)                                # silence: mouth shut
    k = np.array([0.25, 0.5, 0.25])
    return np.clip(np.convolve(m, k, mode="same"), 0, 1)


def _ease_out_back(p: float) -> float:
    p = min(1.0, max(0.0, p))
    c1 = 1.70158
    return 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2


def _arm(d: ImageDraw.ImageDraw, pts: list[tuple[float, float]], w: float, fill) -> None:
    d.line(pts, fill=fill, width=int(w), joint="curve")
    for x, y in (pts[0], pts[-1]):
        d.ellipse((x - w / 2, y - w / 2, x + w / 2, y + w / 2), fill=fill)


def character(t: float, mouth: float, talking: bool, size: int) -> Image.Image:
    """One frame of Cogsworth, `size` px square, transparent background (t in seconds since he arrived; mouth 0..1)."""
    S = size * SS / 400.0
    im = Image.new("RGBA", (size * SS, size * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bob = 5 * math.sin(2 * math.pi * 1.1 * t) if talking else 3 * math.sin(2 * math.pi * 0.6 * t)
    cx, cy, r = 200, 160 + bob, 104
    # legs and feet
    for lx in (138, 242):
        d.rounded_rectangle(((lx) * S, (262 + bob * 0.4) * S, (lx + 20) * S, 322 * S), radius=8 * S, fill=INK)
    for fx in (148, 252):
        d.ellipse(((fx - 26) * S, 316 * S, (fx + 26) * S, 340 * S), fill=INK)
    # arms (the left one waves while he talks)
    wave = (math.sin(2 * math.pi * 1.8 * t) * 18) if talking else 0
    _arm(d, [(98 * S, 200 * S), (70 * S, (175 + wave * 0.4) * S), (60 * S, (150 + wave) * S)], 12 * S, INK)
    _arm(d, [(302 * S, 200 * S), (332 * S, 225 * S), (338 * S, 250 * S)], 12 * S, INK)
    # the gear: ten teeth, turning while he talks
    rot = (t * 24) if talking else (t * 6)
    for k in range(10):
        a = math.radians(rot + k * 36)
        pts = []
        for dx, dy in ((-0.16 * r, -1.20 * r), (0.16 * r, -1.20 * r), (0.16 * r, -0.85 * r), (-0.16 * r, -0.85 * r)):
            pts.append(((cx + dx * math.cos(a) - dy * math.sin(a)) * S, (cy + dx * math.sin(a) + dy * math.cos(a)) * S))
        d.polygon(pts, fill=ACC)
    d.ellipse(((cx - r) * S, (cy - r) * S, (cx + r) * S, (cy + r) * S), fill=ACC)
    d.ellipse(((cx - 74) * S, (cy - 74) * S, (cx + 74) * S, (cy + 74) * S), fill=(58, 140, 220))
    # face
    blink = (t % 3.4) < 0.14
    for ex in (170, 230):
        ey = cy - 18
        if blink:
            d.line(((ex - 15) * S, ey * S, (ex + 15) * S, ey * S), fill=INK, width=int(6 * S))
        else:
            d.ellipse(((ex - 16) * S, (ey - 16) * S, (ex + 16) * S, (ey + 16) * S), fill=INK)
            d.ellipse(((ex - 9) * S, (ey - 11) * S, (ex + 5) * S, (ey + 3) * S), fill=DARK)
    my = cy + 26
    if mouth > 0.12:
        ry = 6 + 20 * mouth
        d.ellipse(((cx - 24) * S, (my - ry * 0.4) * S, (cx + 24) * S, (my + ry) * S), fill=INK if mouth < 0.2 else DARK)
        d.ellipse(((cx - 24) * S, (my - ry * 0.4) * S, (cx + 24) * S, (my + ry) * S), outline=INK, width=int(4 * S))
        if mouth > 0.45:
            d.ellipse(((cx - 13) * S, (my + ry * 0.35) * S, (cx + 13) * S, (my + ry * 0.95) * S), fill=WARM)
    else:
        d.arc(((cx - 26) * S, (my - 22) * S, (cx + 26) * S, (my + 14) * S), start=20, end=160, fill=INK, width=int(6 * S))
    # a dark outline so he reads on any page
    alpha = im.getchannel("A")
    halo = alpha.filter(ImageFilter.MaxFilter(2 * int(3.5 * SS) + 1))
    base = Image.new("RGBA", im.size, (*DARK, 0))
    base.putalpha(halo)
    base.alpha_composite(im)
    return base.resize((size, size), Image.LANCZOS)


def render(audio: Path, dur: float, out_dir: Path, size: int = 260, fps: int = 30, enter_s: float = 0.4, exit_s: float = 0.3, audio_from: float = 0.0) -> int:
    """Write frame_0000.png ... for `dur` seconds: he pops in (a small overshoot), talks to the audio, and drops out again. -> number of frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    env = envelope(audio, dur, fps, audio_from)
    n = int(dur * fps)
    for i in range(n):
        t = i / fps
        m = float(env[i]) if i < len(env) else 0.0
        talk = m > 0.12 or (i + 1 < len(env) and env[min(len(env) - 1, i + 1)] > 0.12)
        fr = character(t, m, True, size)
        sc = _ease_out_back(t / enter_s) if t < enter_s else (1 - min(1.0, (t - (dur - exit_s)) / exit_s) if t > dur - exit_s else 1.0)
        if sc < 0.999:
            w = max(2, int(size * max(sc, 0.02)))
            small = fr.resize((w, w), Image.LANCZOS)
            canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            canvas.alpha_composite(small, ((size - w) // 2, size - w))             # grows up from his feet
            fr = canvas
        fr.save(out_dir / f"frame_{i:04d}.png")
    return n


def composite(seg_video: Path, frames_dir: Path, start: float, out: Path, size: int = 260, fps: int = 30, margin: tuple[int, int] = (36, 22)) -> Path:
    """The segment with Cogsworth in its bottom-right corner from `start` to the end."""
    W, H = 1920, 1080
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(seg_video), "-itsoffset", f"{start:.3f}", "-framerate", str(fps), "-i", str(Path(frames_dir) / "frame_%04d.png"),
                "-filter_complex", f"[0:v][1:v]overlay=x={W - size - margin[0]}:y={H - size - margin[1]}:eof_action=pass:format=auto,format=yuv420p[o]", "-map", "[o]", "-map", "0:a?",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "17", "-pix_fmt", "yuv420p", "-an", str(out)], timeout=600, what="adding the mascot")
    return out
