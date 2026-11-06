"""Word-by-word burned-in captions.

Timing comes from TTS word boundaries (edge-tts) or Whisper word timestamps. Words are
grouped into short chunks (max N words / M chars, broken at pauses and sentence ends);
each word of a chunk gets its own frame with that word highlighted.

Rendering uses Pillow -> transparent PNGs -> an ffconcat list (image + duration) that
ffmpeg overlays on the video. This avoids depending on libass/freetype, which the slim
Homebrew ffmpeg build does not include.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .log import get_logger

log = get_logger("captions")

SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*$")
SOFT_BREAK = re.compile(r"[,;:—–-]+[\"')\]]*$")


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Group:
    words: list[Word]

    @property
    def start(self) -> float:
        return self.words[0].start

    @property
    def end(self) -> float:
        return self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Frame:
    start: float
    end: float
    group: int | None      # None = blank (no caption)
    active: int | None     # index of highlighted word within the group

    @property
    def duration(self) -> float:
        return self.end - self.start


def coerce_words(raw: list) -> list[Word]:
    out = []
    for w in raw:
        if isinstance(w, Word):
            out.append(w)
        elif isinstance(w, dict):
            out.append(Word(str(w["text"]), float(w["start"]), float(w["end"])))
        else:
            out.append(Word(str(w[0]), float(w[1]), float(w[2])))
    return out


def clean_words(words: list[Word], min_word_s: float = 0.12) -> list[Word]:
    """Sort, drop empties, make every word end after it starts, remove overlaps."""
    ws = sorted((Word(w.text.strip(), max(0.0, w.start), max(0.0, w.end)) for w in words if w.text.strip()),
                key=lambda w: w.start)
    for i, w in enumerate(ws):
        nxt = ws[i + 1].start if i + 1 < len(ws) else None
        if w.end <= w.start:
            w.end = w.start + min_word_s
        if nxt is not None and w.end > nxt:
            w.end = max(w.start + 0.01, nxt)
    return ws


def group_words(words: list[Word], max_words: int = 3, max_chars: int = 18,
                pause_break: float = 0.35) -> list[Group]:
    groups: list[Group] = []
    cur: list[Word] = []
    for w in words:
        if cur:
            prev = cur[-1]
            joined = len(" ".join(x.text for x in cur)) + 1 + len(w.text)
            if (len(cur) >= max_words or joined > max_chars or (w.start - prev.end) > pause_break
                    or SENTENCE_END.search(prev.text) or (SOFT_BREAK.search(prev.text) and len(cur) >= 2)):
                groups.append(Group(cur))
                cur = []
        cur.append(w)
    if cur:
        groups.append(Group(cur))
    return groups


def build_timeline(groups: list[Group], total: float, hold_gap: float = 0.3,
                   min_frame: float = 1 / 30) -> list[Frame]:
    """Contiguous, non-overlapping frames covering [0, total]; blanks fill silences.

    A group stays on screen through short pauses (< hold_gap) so captions don't flicker.
    """
    raw: list[Frame] = []
    for gi, g in enumerate(groups):
        next_start = groups[gi + 1].start if gi + 1 < len(groups) else None
        g_end = g.end
        if next_start is not None and 0 <= next_start - g_end < hold_gap:
            g_end = next_start
        for wi, w in enumerate(g.words):
            end = g.words[wi + 1].start if wi + 1 < len(g.words) else g_end
            raw.append(Frame(w.start, max(end, w.start), gi, wi))

    frames: list[Frame] = []
    t = 0.0
    for f in raw:
        start, end = max(f.start, t), min(f.end, total)
        if end <= start:
            continue
        if start - t > 1e-6:
            frames.append(Frame(t, start, None, None))
        frames.append(Frame(start, end, f.group, f.active))
        t = end
    if total - t > 1e-6:
        frames.append(Frame(t, total, None, None))

    # Merge frames shorter than one video frame into their predecessor.
    merged: list[Frame] = []
    for f in frames:
        if merged and f.duration < min_frame:
            merged[-1].end = f.end
        else:
            merged.append(f)
    if merged and len(merged) > 1 and merged[0].duration < min_frame:
        merged[1].start = merged[0].start
        merged.pop(0)
    return merged


def to_srt(groups: list[Group]) -> str:
    def ts(s: float) -> str:
        ms = int(round(s * 1000))
        h, ms = divmod(ms, 3_600_000)
        m, ms = divmod(ms, 60_000)
        sec, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{sec:02},{ms:03}"
    lines = []
    for i, g in enumerate(groups, 1):
        lines += [str(i), f"{ts(g.start)} --> {ts(g.end)}", g.text, ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------- render
def _font(cfg: dict, size: int):
    from PIL import ImageFont

    for f in cfg.get("fonts", []):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    log.warning("no configured caption font found; using Pillow default")
    return ImageFont.load_default(size=size)


def _hex(c: str, alpha: int = 255) -> tuple[int, int, int, int]:
    c = c.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), alpha


def _layout(words: list[str], cfg: dict, width: int):
    """Return (font, lines) where lines is a list of lists of (word_index, text)."""
    from PIL import Image, ImageDraw

    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    size = int(cfg.get("font_size", 92))
    max_w = width * 0.88
    indexed = list(enumerate(words))
    while size >= 48:
        font = _font(cfg, size)
        space = probe.textlength(" ", font=font)

        def line_w(items):
            return sum(probe.textlength(t, font=font) for _, t in items) + space * (len(items) - 1)

        if line_w(indexed) <= max_w:
            return font, [indexed]
        if len(indexed) > 1:  # try the most balanced two-line split
            best = min(range(1, len(indexed)), key=lambda k: abs(line_w(indexed[:k]) - line_w(indexed[k:])))
            l1, l2 = indexed[:best], indexed[best:]
            if max(line_w(l1), line_w(l2)) <= max_w:
                return font, [l1, l2]
        size -= 6
    return _font(cfg, 48), [indexed]


def render_group_image(words: list[str], active: int | None, cfg: dict, width: int, height: int):
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    if active is None or not words:
        return img
    font, lines = _layout(words, cfg, width)
    draw = ImageDraw.Draw(img)
    stroke = int(cfg.get("stroke_width", 9))
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * 1.08)
    total_h = line_h * len(lines)
    y = (height - total_h) // 2
    space = draw.textlength(" ", font=font)
    text_c, hi_c, stroke_c = _hex(cfg.get("text_color", "#FFFFFF")), _hex(cfg.get("highlight_color", "#FFE81A")), \
        _hex(cfg.get("stroke_color", "#000000"))

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    placements = []
    for line in lines:
        lw = sum(draw.textlength(t, font=font) for _, t in line) + space * (len(line) - 1)
        x = (width - lw) / 2
        for idx, t in line:
            placements.append((idx, t, x, y))
            x += draw.textlength(t, font=font) + space
        y += line_h
    for idx, t, x, yy in placements:
        sdraw.text((x + 5, yy + 8), t, font=font, fill=(0, 0, 0, 150), stroke_width=stroke, stroke_fill=(0, 0, 0, 150))
    img = Image.alpha_composite(img, shadow.filter(ImageFilter.GaussianBlur(6)))
    draw = ImageDraw.Draw(img)
    for idx, t, x, yy in placements:
        draw.text((x, yy), t, font=font, fill=hi_c if idx == active else text_c,
                  stroke_width=stroke, stroke_fill=stroke_c)
    return img


def render_captions(words_raw: list, total: float, out_dir: Path, cap_cfg: dict, video_cfg: dict) -> dict:
    """Render caption PNGs + ffconcat. Returns {"ffconcat", "srt", "y", "frames", "groups"}."""
    out_dir = Path(out_dir)
    frames_dir = out_dir / "caption_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    width = int(video_cfg.get("width", 1080))
    box_h = int(cap_cfg.get("box_height", 420))
    words = clean_words(coerce_words(words_raw), float(cap_cfg.get("min_word_s", 0.12)))
    if cap_cfg.get("uppercase", True):
        for w in words:
            w.text = w.text.upper()
    groups = group_words(words, int(cap_cfg.get("max_words", 3)), int(cap_cfg.get("max_chars", 18)),
                         float(cap_cfg.get("pause_break_s", 0.35)))
    fps = float(video_cfg.get("fps", 30))
    timeline = build_timeline(groups, total, min_frame=1.0 / fps)

    blank = frames_dir / "blank.png"
    render_group_image([], None, cap_cfg, width, box_h).save(blank)
    entries: list[tuple[Path, float]] = []
    for f in timeline:
        if f.group is None:
            entries.append((blank, f.duration))
            continue
        p = frames_dir / f"g{f.group:04d}_w{f.active:02d}.png"
        if not p.exists():
            render_group_image([w.text for w in groups[f.group].words], f.active, cap_cfg, width, box_h).save(p)
        entries.append((p, f.duration))

    ffconcat = out_dir / "captions.ffconcat"
    lines = ["ffconcat version 1.0"]
    for p, d in entries:
        lines.append(f"file '{p.resolve()}'")
        lines.append(f"duration {d:.4f}")
    if entries:
        lines.append(f"file '{entries[-1][0].resolve()}'")  # concat demuxer ignores the last duration otherwise
    ffconcat.write_text("\n".join(lines) + "\n")
    srt = out_dir / "captions.srt"
    srt.write_text(to_srt(groups))
    height = int(video_cfg.get("height", 1920))
    y = int(height * float(cap_cfg.get("center_y", 0.6)) - box_h / 2)
    return {"ffconcat": ffconcat, "srt": srt, "y": max(0, min(height - box_h, y)),
            "frames": len(timeline), "groups": len(groups)}
