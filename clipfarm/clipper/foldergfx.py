"""The graphics in a project's footage folder - a logo, a lower third, a title, an animated graphic with see-through parts - used AS
graphics, never as full-screen B-roll.

What counts as a graphic (every other picture is a photo: a B-roll still, photos.py):
  - an SVG; a PNG / WebP / TIFF with see-through parts; a flat picture (a few colours: a logo, a wordmark) that isn't a whole frame
  - anything in a folder named like graphics / logo / brand / title / overlay / lower third, or named like one ("Logo.png",
    "Lower third - Dr Smith.png", "Title.png", "End card.png")
  - a video with an alpha channel (ProRes 4444, PNG or Animation codec in a .mov, VP9 with alpha in a .webm): never footage
How each is used (place()):
  - a logo: the end card of every video (logo_card: the logo alone on a plain background, OUTRO_S seconds after the last word); a small
    logo in a corner the whole time only when the notes ask for one (bug_frame)
  - a title ("Title.png", "Intro.mov"): over the opening, instead of the made-up title card
  - a lower third ("Lower third - Dr Smith.png"): while that person talks (where their name is said; the AI when that can't be told)
  - an end / outro graphic: over the last seconds before the end card
  - any other graphic with a name: where what is said matches its name (the AI decides when the words alone don't); otherwise left out
Each overlay becomes ONE transparent full-frame clip (layer(): QuickTime Animation, alpha kept) that polish lays over the picture, kept clear
of the captions and of faces, and that Final Cut gets as a connected clip above the B-roll.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.foldergfx")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".svg", ".heic", ".heif"}
ALPHA_VIDEO_EXT = {".mov", ".webm", ".mkv"}                  # (ProRes 4444 / Animation / PNG live in .mov, VP9 alpha in .webm)
GFX_DIR = re.compile(r"(?<![a-z])graphics?\b|\bgfx\b|logo|\bbrand|\btitles?\b|overlay|lower[\s_-]*thirds?|\bl3\b|watermark|\bbugs?\b|"
                     r"end[\s_-]*(card|screen)|\bmograph\b|\bstickers?\b|\bicons?\b|\bsupers?\b|\btemplates?\b", re.I)
GFX_NAME = re.compile(r"logo|wordmark|watermark|\bbug\b|lower[\s_-]*thirds?|\bl3\b|name[\s_-]*(card|tag|strap|super)|end[\s_-]*(card|screen)|"
                      r"overlay|chyron", re.I)                                   # (a "Title.jpg" photo stays a photo: only how it looks says)
LOWER = re.compile(r"lower[\s_-]*thirds?|\bl3\b|name[\s_-]*(card|tag|strap|super)|\bsuper\b|chyron", re.I)
TITLE = re.compile(r"\b(title|intro|opener|opening)\b", re.I)
END = re.compile(r"\b(outro|closing|ending)\b|\bend\b|end[\s_-]*(card|screen)", re.I)
LOGO = re.compile(r"logo|wordmark|watermark|\bbug\b|\bbrand", re.I)
ALPHA_PIX = re.compile(r"^(yuva|gbrap|rgba|bgra|argb|abgr|ya8|ya16|ayuv|pal8)", re.I)
BUG_RE = re.compile(r"watermark|logo[\s-]*bug|\bbug\b|corner logo|logo (in|on|at) (the |a )?(top|bottom|corner)|logo (the )?whole (time|video)|"
                    r"logo throughout|logo on screen the whole|keep (the|our|my) logo (on|up)", re.I)
HONORIFIC = {"dr", "mr", "mrs", "ms", "miss", "prof", "professor", "sir", "the", "and", "of", "pastor", "rev", "coach"}
IMAGE_S = 4.5                        # a still graphic stays up this long (a title 3.5 s)
TITLE_S = 3.5
MAX_VIDEO_S = 8.0                    # an animated graphic plays at most this long
FADE = 0.35


# ------------------------------------------------------------------------------------------------ what is a graphic
def video_alpha(vs: dict | None) -> bool:
    """True for a video stream with see-through parts: ProRes 4444 with alpha (yuva444p...), PNG / Animation codec (rgba / argb), VP9 with alpha."""
    if not vs:
        return False
    if ALPHA_PIX.match(str(vs.get("pix_fmt") or "")) and str(vs.get("pix_fmt")) != "pal8":
        return True
    tags = {str(k).lower(): str(v) for k, v in (vs.get("tags") or {}).items()}
    return tags.get("alpha_mode") == "1"


def _look(path: Path) -> tuple[float, int, int, int] | None:
    """(share of see-through pixels, number of colours, width, height) of a picture, or None when it can't be opened here."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            if im.format == "JPEG":
                im.draft("RGB", (192, 192))
            im = im.convert("RGBA")
            im.thumbnail((96, 96))
            px = _pixels(im)
    except Exception:  # noqa: BLE001 - HEIC without a plugin, a broken file
        return None
    clear = float((px[:, 3] < 250).mean()) if len(px) else 0.0
    solid = px[px[:, 3] >= 40]
    colours = len({(int(r) >> 4, int(g) >> 4, int(b) >> 4) for r, g, b, _a in solid})
    return clear, colours, w, h


def _pixels(im):
    """An RGBA picture as an (n, 4) array."""
    import numpy as np
    return np.asarray(im.convert("RGBA"), dtype=np.uint8).reshape(-1, 4)


def frame_shaped(w: int, h: int) -> bool:
    """A picture the shape of a video frame (16:9 or 9:16)."""
    if not w or not h:
        return False
    r = w / h
    return min(abs(r - 16 / 9), abs(r - 9 / 16)) < 0.08


def is_graphic(path, rel_parts: tuple | list = ()) -> bool:
    """A picture meant to go OVER the video (a logo, a lower third, a title), not to be shown full screen as B-roll."""
    p = Path(path)
    if p.suffix.lower() == ".svg":
        return True
    if any(GFX_DIR.search(x) for x in list(rel_parts)[:-1]) or GFX_NAME.search(p.stem):
        return True
    look = _look(p)
    if look is None:
        return False
    clear, colours, w, h = look
    if clear > 0.02:
        return True                                           # see-through parts: made to go over something
    return colours <= 40 and not (frame_shaped(w, h) and max(w, h) >= 1280)       # flat (a logo) unless it's a full-frame slide


def role_of(name: str, rel_parts: tuple | list = ()) -> str:
    """logo / lower / title / end / other, from the file's name first, then its folder's."""
    stem = Path(name).stem
    dirs = " ".join(list(rel_parts)[:-1])
    if LOWER.search(stem) or LOWER.search(dirs):
        return "lower"
    if TITLE.search(stem):
        return "title"
    if END.search(stem):
        return "end"
    if LOGO.search(stem) or LOGO.search(dirs):
        return "logo"
    if TITLE.search(dirs):
        return "title"
    return "other"


def label_of(name: str) -> str:
    """The words of a graphic's name that say what (or who) it is: "Lower third - Dr Smith.png" -> "Dr Smith"."""
    s = re.sub(r"[_]+", " ", Path(name).stem)
    s = re.sub(LOWER.pattern + r"|\b(title|intro|outro|end card|end screen|logo|graphic|gfx|overlay|final|v\d+|copy|alpha|prores|4444)\b",
               " ", s, flags=re.I)
    s = re.sub(r"\(\s*\d*\s*\)", " ", s)
    return " ".join(s.strip(" -–—.,:;").split()).strip(" -–—.,:;")


def scan(folder: Path, skipped, cache: dict | None = None, footage: set[str] | None = None) -> list[dict]:
    """Every graphic in the folder: {name, path, rel, kind: image|video, role, label, w, h, dur, clear}. Never in the footage-free
    folders (Output, Final Cut, Music...). Videos are only probed when they could hold alpha and aren't known footage (`footage`: paths
    the footage scan already took), cached by path + size + date."""
    cache = cache if cache is not None else {}
    footage = footage or set()
    out = []
    for p in sorted(folder.rglob("*")):
        ext = p.suffix.lower()
        if p.name.startswith(".") or not p.is_file() or ext not in IMAGE_EXT | ALPHA_VIDEO_EXT or str(p) in footage:
            continue
        rel = p.relative_to(folder)
        if skipped(rel.parts[:-1]):
            continue
        st = p.stat()
        key = f"{p}:{st.st_size}:{int(st.st_mtime)}"
        if key not in cache:
            if ext in IMAGE_EXT:
                look = _look(p)
                cache[key] = {"graphic": is_graphic(p, rel.parts), "kind": "image", "w": look[2] if look else 0, "h": look[3] if look else 0,
                              "clear": round(look[0], 3) if look else 0.0}
            else:
                try:
                    info = ffmpeg.probe(p)
                    vs = ffmpeg.video_stream(info)
                    cache[key] = {"graphic": video_alpha(vs), "kind": "video", "w": int((vs or {}).get("width") or 0),
                                  "h": int((vs or {}).get("height") or 0), "dur": float(info.get("format", {}).get("duration") or 0), "clear": 1.0}
                except Exception:  # noqa: BLE001 - unreadable: not a graphic we can use
                    cache[key] = {"graphic": False}
        c = cache[key]
        if not c.get("graphic"):
            continue
        out.append({"name": p.name, "path": str(p), "rel": str(rel), "kind": c["kind"], "role": role_of(p.name, rel.parts),
                    "label": label_of(p.name), "w": c.get("w", 0), "h": c.get("h", 0), "dur": c.get("dur", 0.0), "clear": c.get("clear", 0.0)})
    return out[:40]


def logo_of(gfx: list[dict]) -> str | None:
    """The project's logo: the picture named (or kept in a folder named) logo / brand; else the ONLY unnamed see-through still (or SVG) at the
    top of the folder or in a graphics folder - a wordmark like "acme.png". Anything less sure is not taken for a logo."""
    named = [g for g in gfx if g["role"] == "logo" and g["kind"] == "image"]
    if named:
        return min(named, key=lambda g: (g["rel"].count("/"), -g["clear"], g["name"]))["path"]
    loose = [g for g in gfx if g["role"] == "other" and g["kind"] == "image" and not frame_shaped(g["w"], g["h"])
             and (g["clear"] > 0.02 or g["path"].lower().endswith(".svg"))
             and ("/" not in g["rel"] or GFX_DIR.search(str(Path(g["rel"]).parent)))]
    others = [g for g in gfx if g["kind"] == "image" and g["role"] == "other"]
    return loose[0]["path"] if len(loose) == 1 and len(others) == 1 else None


# ------------------------------------------------------------------------------------------------ where each one goes
PLACE_SYSTEM = ("You place a video's own graphics (lower thirds with a person's name, labelled graphics) in the finished video. For each "
                "graphic give the number of the sentence where it belongs: for a person's lower third, the first sentence that person says "
                "(or where they are introduced); for any other graphic, the sentence that talks about what it shows. -1 when nothing fits.")
PLACE_SCHEMA = {"type": "object", "properties": {"places": {"type": "array", "items": {"type": "object", "properties": {
    "g": {"type": "integer"}, "sentence": {"type": "integer"}}, "required": ["g", "sentence"]}}}, "required": ["places"]}


def _words(s: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9']+", s.lower()) if w not in HONORIFIC and len(w) > 1]


def _mention(label: str, sents) -> int | None:
    """The first sentence that says the graphic's name (all its words, or a person's surname), or None."""
    words = _words(label)
    if not words:
        return None
    for k, s in enumerate(sents):
        said = set(_words(s.text))
        if all(w in said for w in words) or (len(words) >= 2 and words[-1] in said and len(words[-1]) >= 4):
            return k
    return None


def _len_of(g: dict, default: float) -> float:
    if g["kind"] == "video" and g.get("dur"):
        return round(min(MAX_VIDEO_S, float(g["dur"])), 2)
    return default


def place(gfx: list[dict], sents, prog_len: float, notes: str = "", ai=None) -> dict:
    """-> {"overlays": [{path, kind, role, label, at, len, w, h}], "title": bool (a title of yours replaces the made-up one),
    "logo": path | None, "bug": bool (a corner logo the whole time: only when the notes ask)}. Times are on the finished programme
    (prog_len = the programme without the end card). Overlays never overlap: the earlier wins."""
    out: list[dict] = []
    logo = logo_of(gfx)
    title = next((g for g in gfx if g["role"] == "title"), None)
    t0 = 0.0
    if title:
        ln = min(_len_of(title, TITLE_S), max(1.0, prog_len - 0.5))
        out.append({**title, "at": 0.0, "len": ln})
        t0 = ln
    end = next((g for g in gfx if g["role"] == "end" and g["path"] != logo), None)
    if end:
        ln = _len_of(end, IMAGE_S)
        if prog_len - ln > t0 + 1.0:
            out.append({**end, "at": round(prog_len - ln, 2), "len": ln})
    sents = [s for s in sents if s.start >= t0 - 0.05 and s.end <= prog_len + 0.05] if sents else []
    todo = [g for g in gfx if g["role"] == "lower" or (g["role"] == "other" and g["path"] != logo and len(_words(g["label"])) >= 1)]
    at_of: dict[int, int] = {}
    for k, g in enumerate(todo):
        m = _mention(g["label"], sents)
        if m is not None:
            at_of[k] = m
    rest = [k for k in range(len(todo)) if k not in at_of]
    lowers_left = [k for k in rest if todo[k]["role"] == "lower"]
    if rest and sents and rest == lowers_left and sum(1 for g in todo if g["role"] == "lower") == 1:
        at_of[rest[0]] = 0                                                  # one person's lower third (never named aloud): when they first speak
    elif rest and sents and ai is not None:
        lines = "\n".join(f"[{i}] ({s.start:.0f} s) {s.text[:160]}" for i, s in enumerate(sents[:300]))
        names = "\n".join(f"G{k}: {'lower third for' if todo[k]['role'] == 'lower' else 'graphic named'} '{todo[k]['label']}'" for k in rest)
        try:
            res = ai.complete_json("gfx_place", PLACE_SYSTEM, f"The graphics:\n{names}\n\nThe finished video, sentence by sentence:\n{lines}\n\n"
                                   "Where does each graphic go?", PLACE_SCHEMA)
            for x in res.data.get("places") or []:
                k, i = int(x.get("g", -1)), int(x.get("sentence", -1))
                if k in rest and 0 <= i < len(sents):
                    at_of[k] = i
        except Exception as e:  # noqa: BLE001 - unplaced graphics are simply left out
            log.warning(f"placing your graphics failed: {e}")
        if [k for k in lowers_left if k not in at_of] and len(sents) > 1:  # lower thirds the AI couldn't place: spread over the video, in order
            left = [k for k in lowers_left if k not in at_of]
            for n, k in enumerate(left):
                want = prog_len * (n + 0.5) / len(left)
                at_of[k] = min(range(len(sents)), key=lambda i: abs(sents[i].start - want))
    for k, i in sorted(at_of.items(), key=lambda x: x[1]):
        g, s = todo[k], sents[i]
        ln = _len_of(g, IMAGE_S)
        at = max(t0, s.start + 0.3)
        if at + ln > prog_len - 0.3:
            at = max(t0, prog_len - 0.3 - ln)
        out.append({**g, "at": round(at, 2), "len": round(ln, 2)})
    out.sort(key=lambda o: o["at"])
    kept: list[dict] = []
    for o in out:
        if o["len"] < 0.8 or (kept and o["at"] < kept[-1]["at"] + kept[-1]["len"] + 0.1):
            log.info(f"graphic {o['name']} left out: no room for it at {o['at']:.1f}s")
            continue
        kept.append(o)
    for g in gfx:
        if g["path"] != logo and not any(g["path"] == o["path"] for o in kept):
            log.info(f"graphic {g['name']} ({g['role']}) not used: nothing in the video matches it")
    return {"overlays": kept, "title": any(o["role"] == "title" for o in kept), "logo": logo, "bug": bool(logo and BUG_RE.search(notes or ""))}


# ------------------------------------------------------------------------------------------------ drawing
def load_rgba(path: str, edge: int = 2400):
    """The picture as RGBA (see-through parts kept); an SVG through Quick Look, anything Pillow can't open through ffmpeg."""
    from PIL import Image, ImageOps
    p = Path(path)
    if p.suffix.lower() == ".svg":
        with tempfile.TemporaryDirectory(prefix="svg-") as tmp:
            subprocess.run(["qlmanage", "-t", "-s", str(min(edge, 1600)), "-o", tmp, str(p)], capture_output=True, timeout=60)
            got = next(Path(tmp).glob("*.png"), None)
            if got is None:
                raise RuntimeError(f"could not draw {p.name}")
            with Image.open(got) as im:
                return im.convert("RGBA")
    try:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGBA")
    except Exception:  # noqa: BLE001 - HEIC and friends
        r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", str(p), "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                           capture_output=True, timeout=120)
        from io import BytesIO
        im = Image.open(BytesIO(r.stdout)).convert("RGBA")
    if max(im.size) > edge:
        im.thumbnail((edge, edge))
    return im


def _trim(im):
    """The picture without its see-through margins."""
    box = im.getchannel("A").getbbox()
    return im.crop(box) if box else im


def _overlaps(box: tuple, zones: list) -> bool:
    x0, y0, x1, y1 = box
    return any(x0 < z[2] and z[0] < x1 and y0 < z[3] and z[1] < y1 for z in zones or [])


def box_for(role: str, gw: int, gh: int, W: int, H: int, avoid: list | None = None) -> tuple[int, int, int, int] | None:
    """(x, y, w, h) for a graphic of size gw x gh: a picture the shape of the frame covers it (it was designed in place); a lower third sits
    bottom-left above the captions (bottom-right, or smaller, when a face is there); a title or anything else in the middle (higher up
    when a face is there). avoid = zones in 0..1 of the frame (faces, the captions band)."""
    if not gw or not gh:
        return None
    if abs((gw / gh) / (W / H) - 1.0) < 0.04 and gw >= W * 0.5:
        return 0, 0, W, H
    avoid = avoid or []
    caps_top = min([z[1] for z in avoid if z[0] <= 0.01 and z[2] >= 0.99] or [0.84])     # the captions band (a full-width zone), when there is one
    faces = [z for z in avoid if not (z[0] <= 0.01 and z[2] >= 0.99)]
    mx, my = int(W * 0.06), int(H * 0.06)
    for shrink in (1.0, 0.8, 0.65):
        if role == "lower":
            s = min(W * 0.42 / gw, H * 0.17 / gh, 2.5) * shrink
            w, h = max(2, int(gw * s)), max(2, int(gh * s))
            y = int(min(H - my, caps_top * H - H * 0.02)) - h
            spots = [(mx, y), (W - mx - w, y)]
        else:
            big = role == "title"
            s = min(W * (0.6 if big else 0.5) / gw, H * (0.36 if big else 0.4) / gh, 3.0) * shrink
            w, h = max(2, int(gw * s)), max(2, int(gh * s))
            lo = int(caps_top * H - H * 0.02) - h
            spots = [((W - w) // 2, min(lo, int(H * 0.42) - h // 2)), ((W - w) // 2, my), ((W - w) // 2, lo)]
        for x, y in spots:
            y = max(my // 2, y)
            if not _overlaps((x / W, y / H, (x + w) / W, (y + h) / H), faces):
                return x, y, w, h
    x, y = spots[0]
    return x, max(my // 2, y), w, h                                          # nowhere clear: its usual place (a lower third is under a face anyway)


def layer(g: dict, secs: float, W: int, H: int, fps: float, out: Path, avoid: list | None = None) -> Path | None:
    """One graphic of the folder as a transparent full-frame clip of `secs` (QuickTime Animation, alpha kept), placed by box_for and faded in
    and out. None when it can't be drawn (the video goes without it)."""
    from PIL import Image
    secs = max(0.5, float(secs))
    ff = ffmpeg.bin_path("ffmpeg")
    try:
        if g["kind"] == "video":
            gw, gh = int(g.get("w") or 0), int(g.get("h") or 0)
            box = box_for(g["role"], gw, gh, W, H, avoid)
            if not box:
                return None
            x, y, w, h = box
            fade_out = f",fade=t=out:st={max(0.0, secs - 0.25):.3f}:d=0.25:alpha=1" if float(g.get("dur") or 0) > secs + 0.05 else ""
            fc = (f"color=c=black@0.0:s={W}x{H}:r={fps:.3f}:d={secs:.3f},format=rgba[bg];"
                  f"[0:v]fps={fps:.3f},scale={w // 2 * 2}:{h // 2 * 2},format=rgba,setpts=PTS-STARTPTS[g];"
                  f"[bg][g]overlay={x}:{y}:format=auto:eof_action=pass,format=rgba{fade_out}[o]")
            cmd = [ff, "-y", "-v", "error", "-t", f"{secs:.3f}", "-i", g["path"], "-filter_complex", fc, "-map", "[o]", "-t", f"{secs:.3f}",
                   "-an", "-c:v", "qtrle", "-pix_fmt", "argb", "-r", f"{fps:.3f}", str(out)]
        else:
            im = _trim(load_rgba(g["path"]))
            box = box_for(g["role"], im.width, im.height, W, H, avoid)
            if not box:
                return None
            x, y, w, h = box
            frame = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            frame.alpha_composite(im.resize((w, h), Image.LANCZOS), (x, y))
            png = out.with_suffix(".png")
            frame.save(png)
            fc = (f"[0:v]format=argb,fade=t=in:st=0:d={FADE}:alpha=1,fade=t=out:st={max(0.0, secs - FADE):.3f}:d={FADE}:alpha=1[o]")
            cmd = [ff, "-y", "-v", "error", "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{secs:.3f}", "-i", str(png), "-filter_complex", fc,
                   "-map", "[o]", "-an", "-c:v", "qtrle", "-pix_fmt", "argb", "-r", f"{fps:.3f}", str(out)]
        ffmpeg.run(["nice", "-n", "10", *cmd], timeout=600, what=f"laying out {g['name']}")
        return out if out.exists() else None
    except Exception as e:  # noqa: BLE001 - one graphic is never worth losing a video over
        log.warning(f"graphic {g.get('name')} could not be laid out: {e}")
        return None


def _bg_for(im) -> tuple[int, int, int]:
    """A background the logo reads on: its own background colour when it has one (no box around it), else dark for a light logo and light
    for a dark one."""
    small = im.copy()
    small.thumbnail((128, 128))
    px = [tuple(int(v) for v in p) for p in _pixels(small)]
    clear = sum(1 for p in px if p[3] < 250) / max(1, len(px))
    if clear < 0.02:                                                         # opaque: its corners are its background
        w, h = small.size
        cs = [small.getpixel((0, 0)), small.getpixel((w - 1, 0)), small.getpixel((0, h - 1)), small.getpixel((w - 1, h - 1))]
        return tuple(int(sum(c[i] for c in cs) / 4) for i in range(3))
    ink = [p for p in px if p[3] >= 128] or [(0, 0, 0, 255)]
    lum = sum(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in ink) / len(ink) / 255
    return (18, 18, 20) if lum > 0.55 else (245, 245, 243)


def logo_card(logo: str, W: int, H: int, secs: float, out: Path, fps: float = 30.0) -> Path:
    """The end card: the logo alone, centred on a plain background, fading in. An ordinary (opaque) clip, so it is simply the last shot of
    the video and of the Final Cut project."""
    from PIL import Image
    im = _trim(load_rgba(logo))
    bg = _bg_for(im)
    s = min(W * 0.4 / im.width, H * 0.3 / im.height, 4.0)
    w, h = max(2, int(im.width * s)), max(2, int(im.height * s))
    base = Image.new("RGB", (W, H), bg)
    top = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    top.alpha_composite(im.resize((w, h), Image.LANCZOS), ((W - w) // 2, (H - h) // 2))
    work = out.parent
    base.save(work / f"{out.stem}_bg.png")
    top.save(work / f"{out.stem}_logo.png")
    fc = (f"[1:v]format=rgba,fade=t=in:st=0.15:d=0.5:alpha=1[l];[0:v][l]overlay=0:0:format=auto,"
          "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p[o]")
    ffmpeg.run(["nice", "-n", "10", ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error",
                "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{secs:.3f}", "-i", str(work / f"{out.stem}_bg.png"),
                "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{secs:.3f}", "-i", str(work / f"{out.stem}_logo.png"),
                "-filter_complex", fc, "-map", "[o]", "-an", "-r", f"{fps:.3f}", "-t", f"{secs:.3f}", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "14", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
                str(out)], timeout=300, what="drawing the end card")
    return out


def bug_frame(logo: str, W: int, H: int, out: Path, side: str | None = None) -> Path:
    """A small logo in a corner, inside the safe area (placement.safe_box), see-through, on a transparent full frame. 9:16: 8-10 % of the width at 60 %
    opacity, top-left inside the platform's safe box (the top-right is where the app puts its own buttons); 16:9: about 6 % of the height, top-right
    inside title-safe. `side` ("left"/"right") overrides the corner (a lower third uses the other side)."""
    from PIL import Image

    from . import placement as PL
    safe = PL.safe_box(W, H)
    portrait = H > W
    im = _trim(load_rgba(logo))
    s = min(W * 0.09 / im.width, H * 0.5 / im.height, 2.0) if portrait else min(H * 0.06 / im.height, W * 0.2 / im.width, 2.0)
    w, h = max(2, int(im.width * s)), max(2, int(im.height * s))
    im = im.resize((w, h), Image.LANCZOS)
    im.putalpha(im.getchannel("A").point(lambda v: int(v * (0.6 if portrait else 0.7))))
    frame = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    pad = max(6, int(min(W, H) * 0.012))
    left = (side or ("left" if portrait else "right")) == "left"
    x = safe["x0"] + pad if left else safe["x1"] - pad - w
    frame.alpha_composite(im, (int(x), int(safe["y0"] + pad)))
    frame.save(out)
    return out


def still_clip(png: Path, secs: float, fps: float, out: Path) -> Path | None:
    """A transparent picture as a clip of `secs` (QuickTime Animation): what Final Cut gets for the corner logo."""
    try:
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{secs:.3f}", "-i", str(png),
                    "-vf", "format=argb", "-an", "-c:v", "qtrle", "-pix_fmt", "argb", "-r", f"{fps:.3f}", str(out)], timeout=900, what="the corner logo")
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"the corner logo clip for Final Cut failed: {e}")
        return None


def end_overlay(logo: str | None, business: str, tagline: str, secs: float, W: int, H: int, out: Path, fps: float = 30.0) -> Path | None:
    """An ad's end card drawn here (when Hyperframes isn't installed): the picture behind dimmed, the logo, the name and the call to action,
    fading in. A transparent full-frame clip."""
    from PIL import Image, ImageDraw

    from .gfxcards import font
    try:
        frame = Image.new("RGBA", (W, H), (0, 0, 0, 150))
        d = ImageDraw.Draw(frame)
        parts = []
        if logo and Path(logo).is_file():
            im = _trim(load_rgba(logo))
            s = min(W * 0.34 / im.width, H * 0.26 / im.height, 4.0)
            parts.append(("img", im.resize((max(2, int(im.width * s)), max(2, int(im.height * s))), Image.LANCZOS)))
        k = min(W, H) / 1080
        for text, weight, size, fill in ((business, "bold", int(72 * k), (255, 255, 255, 255)), (tagline, "medium", int(44 * k), (255, 255, 255, 215))):
            text = " ".join(str(text or "").split())
            if text:
                f = font(weight, size)
                while d.textlength(text, font=f) > W * 0.84 and size > 18:
                    size = int(size * 0.9)
                    f = font(weight, size)
                parts.append(("text", (text, f, fill)))
        if not parts:
            return None
        gap = int(H * 0.035)
        pad = int(min(W, H) * 0.03)
        plate = [p[0] == "img" and _bg_for(p[1])[0] > 128 for p in parts]       # a dark logo: on a light plate, or it vanishes into the dimmed picture
        heights = [(p[1].height + (2 * pad if pl else 0)) if p[0] == "img" else int(p[1][1].size * 1.2) for p, pl in zip(parts, plate)]
        y = (H - sum(heights) - gap * (len(parts) - 1)) // 2
        for (kind, v), h, pl in zip(parts, heights, plate):
            if kind == "img":
                if pl:
                    d.rounded_rectangle(((W - v.width) // 2 - pad, y, (W + v.width) // 2 + pad, y + h), radius=max(4, pad // 3), fill=(245, 245, 243, 240))
                frame.alpha_composite(v, ((W - v.width) // 2, y + (pad if pl else 0)))
            else:
                text, f, fill = v
                d.text((W // 2, y + h // 2), text, font=f, fill=fill, anchor="mm")
            y += h + gap
        png = out.with_suffix(".png")
        frame.save(png)
        ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{secs:.3f}", "-i", str(png),
                    "-vf", f"format=argb,fade=t=in:st=0:d=0.45:alpha=1", "-an", "-c:v", "qtrle", "-pix_fmt", "argb", "-r", f"{fps:.3f}", str(out)],
                   timeout=300, what="drawing the end card")
        return out
    except Exception as e:  # noqa: BLE001
        log.warning(f"the end card could not be drawn: {e}")
        return None
