"""Photos in a project's B-roll folders become cutaway scenes with motion: a slow Ken Burns move, a film-strip border, or a tilted
polaroid over a soft blurred copy of itself. The AI looks at each photo once (cached), the cutaway planner places them like any other
B-roll, and each one is baked into a short silent clip (also what Final Cut gets, so the effect shows there too)."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.photos")
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}
PHOTO_DIR = re.compile(r"b[\s_-]?roll|photo|picture|still|image|pics", re.I)
MIN_LONG_SIDE = 800                                  # (proposal) a picture outside a photo folder must be at least this big on its long side to count as a photo
SCENE_S = 4.0                                        # a photo is offered as a 4 second scene (the planner may use less)
VIRTUAL_BASE = 1_000_000.0                           # photo scenes live far past the footage on the shared timeline
EFFECTS = ["zoom_in", "polaroid", "pan_right", "zoom_out", "pan_left"]            # (no film-strip border: the creator dislikes it)


def scan(folder: Path, skipped) -> list[dict]:
    """Photos inside B-roll / photos folders (never the footage, export or Final Cut folders, never our own results). Graphics - a logo,
    a lower third, anything see-through or flat (foldergfx.is_graphic) - are not photos: they are never shown full screen."""
    from .foldergfx import is_graphic
    out = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in PHOTO_EXT or p.name.startswith(".") or not p.is_file():
            continue
        rel = p.relative_to(folder)
        if skipped(rel.parts[:-1]):
            continue
        named_dir = any(PHOTO_DIR.search(x) for x in rel.parts[:-1])
        if is_graphic(p, rel.parts):
            log.info(f"{rel} is a graphic, not a photo: it is never used as full-screen B-roll")
            continue
        if not named_dir:                                                # (a photo anywhere in the folder counts when it is a real picture: big enough, not a graphic)
            if _long_side(p) < MIN_LONG_SIDE:
                continue
        out.append({"name": p.name, "path": str(p), "size": p.stat().st_size, "mtime": int(p.stat().st_mtime)})
    return out[:60]


def _jpeg(path: str, edge: int = 640) -> bytes | None:
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-i", path, "-frames:v", "1",
                        "-vf", f"scale='min({edge},iw)':-2", "-f", "mjpeg", "-q:v", "5", "-"], capture_output=True, timeout=120)
    return r.stdout or None


SCHEMA = {"type": "object", "properties": {"photos": {"type": "array", "items": {"type": "object", "properties": {
    "n": {"type": "integer"}, "what": {"type": "string"}, "quality": {"type": "number"}}, "required": ["n", "what", "quality"]}}},
    "required": ["photos"]}


def describe(ai, photos: list[dict], cache: dict) -> list[dict]:
    """Add 'what' and 'quality' (1-10) to each photo: one cheap vision call per 8 photos, remembered in `cache` (path+mtime -> result)."""
    todo = [p for p in photos if f"{p['path']}:{p['mtime']}" not in cache]
    for i in range(0, len(todo), 8):
        chunk = todo[i:i + 8]
        imgs = [(p, _jpeg(p["path"])) for p in chunk]
        imgs = [(p, b) for p, b in imgs if b]
        if not imgs:
            continue
        try:
            res = ai.complete_json("photo_describe",
                                   "You describe photographs for a video editor who cannot see them: one concrete sentence each (subject, setting), "
                                   "and how usable it is as a still in a video from 1 to 10 (sharp, well framed, interesting = high).",
                                   "Photos, in order: " + ", ".join(f"{n}={p['name']}" for n, (p, _) in enumerate(imgs)),
                                   SCHEMA, images=[b for _, b in imgs])
            for item in res.data.get("photos") or []:
                n = int(item.get("n", -1))
                if 0 <= n < len(imgs):
                    cache[f"{imgs[n][0]['path']}:{imgs[n][0]['mtime']}"] = {"what": str(item.get("what", ""))[:200], "quality": float(item.get("quality", 6))}
        except Exception as e:  # noqa: BLE001 - an undescribed photo simply isn't offered
            log.warning(f"describing photos failed: {e}")
    return [{**p, **cache[f"{p['path']}:{p['mtime']}"]} for p in photos if f"{p['path']}:{p['mtime']}" in cache]


def scenes(photos: list[dict]) -> list[dict]:
    """The described photos as pool scenes (like B-roll scenes, with a made-up place on the timeline)."""
    out = []
    for k, p in enumerate(photos):
        sz = _size(p["path"]) if p.get("path") else None
        out.append({"start": VIRTUAL_BASE + k * 10.0, "end": VIRTUAL_BASE + k * 10.0 + SCENE_S, "what": f"(still photo) {p['what']}",
                    "quality": max(6.0, float(p["quality"])), "raw_quality": float(p["quality"]), "file": p["name"], "photo": p["path"],
                    "fstart": 0.0, "fend": SCENE_S, **({"w": sz[0], "h": sz[1]} if sz else {})})
    return out


def _film_frame(W: int, H: int, out: Path) -> Path:
    from PIL import Image, ImageDraw
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bar = int(H * 0.085)
    d.rectangle([0, 0, W, bar], fill=(12, 12, 12, 255))
    d.rectangle([0, H - bar, W, H], fill=(12, 12, 12, 255))
    d.rectangle([0, 0, int(W * 0.012), H], fill=(12, 12, 12, 255))
    d.rectangle([W - int(W * 0.012), 0, W, H], fill=(12, 12, 12, 255))
    hole_w, hole_h, step = int(bar * 0.42), int(bar * 0.42), int(bar * 0.9)
    for x in range(step // 2, W, step):
        for y in (bar // 2 - hole_h // 2, H - bar // 2 - hole_h // 2):
            d.rounded_rectangle([x, y, x + hole_w, y + hole_h], radius=3, fill=(235, 232, 220, 255))
    im.save(out)
    return out


def _long_side(path: Path) -> int:
    """The long side in pixels, from the file header only (0 when it cannot be read)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return int(max(im.size))
    except Exception:  # noqa: BLE001 - heif without a plugin, a broken file: not a photo for us
        return 0


def _size(path: str) -> tuple[int, int] | None:
    try:
        from PIL import Image, ImageOps
        with Image.open(path) as im:
            return ImageOps.exif_transpose(im).size
    except Exception:  # noqa: BLE001
        return None


BLUR_SUB = 3                         # photo moves: rendered at 3x the frame rate and blended (motion blur, and no stepping on slow moves)


def bake(path: str, effect: str, seconds: float, W: int, H: int, fps: float, out: Path, motion_blur: bool = True) -> Path:
    """Render one photo as a silent clip of `seconds` with the effect. The colours of the photo are left alone, and the picture is never
    cropped hard: a photo that is not roughly the video's shape is shown WHOLE over a soft blurred copy of itself, and every move is gentle."""
    from fractions import Fraction
    fr = Fraction(fps).limit_denominator(1001)                               # 23.976 is 24000/1001 exactly: a clip a frame off would be re-timed by Final Cut
    frs = f"{fr.numerator}/{fr.denominator}"
    n = max(2, int(round(seconds * fps)))
    blur = max(8, int(round(28 * W / 1920)))                                  # the same softness at any frame size (4K needs twice the radius)
    tv = "scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p"        # JPEG-range pictures become normal video range
    size = _size(path)
    ratio = (size[0] / size[1]) / (W / H) if size else None
    near = ratio is not None and abs(ratio - 1.0) < 0.12                     # close enough to fill the frame without losing much
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", path]
    if effect == "polaroid":
        cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-loop", "1", "-framerate", f"{fps:.3f}", "-t", f"{seconds:.3f}", "-i", path]
        fw = int(H * 0.72)
        g = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur={blur}:3,format=yuv420p,"
             f"drawbox=x=0:y=0:w={W}:h={H}:color=black@0.28:t=fill[bg];"
             f"[b]scale=-2:{fw}:force_original_aspect_ratio=decrease,pad=iw+{int(H * .035)}:ih+{int(H * .035)}+{int(H * .09)}:{int(H * .0175)}:{int(H * .0175)}:white,"
             f"format=rgba,rotate=-2.2*PI/180:ow='rotw(-2.2*PI/180)':oh='roth(-2.2*PI/180)':c=none[fg];"
             f"[bg][fg]overlay=x='(W-w)/2':y='(H-h)/2+{int(H * .03)}*(1-min(t/0.45\\,1))':format=auto:shortest=1,fps={frs},{tv}[o]")
        cmd += ["-filter_complex", g, "-map", "[o]"]
    else:
        zoom = 0.08 if near else 0.06
        sub = BLUR_SUB if motion_blur else 1
        n0, n = n, n * sub                                                    # (the move is laid out over sub x the frames...)
        frs_out, frs = frs, f"{fr.numerator * sub}/{fr.denominator}"
        z, x = {"zoom_in": (f"1+{zoom}*on/{n}", "(iw-iw/zoom)/2"), "zoom_out": (f"1+{zoom}-{zoom}*on/{n}", "(iw-iw/zoom)/2"),
                "pan_right": (f"1+{zoom}", f"(iw-iw/zoom)*on/{n}"), "pan_left": (f"1+{zoom}", f"(iw-iw/zoom)*(1-on/{n})")}.get(
            effect, (f"1+{zoom}*on/{n}", "(iw-iw/zoom)/2"))
        mb = f"tmix=frames={sub}:weights='{' '.join(['1'] * sub)}',framestep={sub}," if sub > 1 else ""   # (...then each group of sub blended into one)
        if not near:                                                          # whole photo over a blurred copy; the photo itself drifts a little
            sc = min(W / size[0], H / size[1]) if size else 1.0
            fw, fh = (int(size[0] * sc) // 2 * 2, int(size[1] * sc) // 2 * 2) if size else (W // 2 * 2, H // 2 * 2)
            g = (f"[0:v]split[a][b];[a]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},boxblur={blur}:3,format=yuv420p,"
                 f"drawbox=x=0:y=0:w={W}:h={H}:color=black@0.25:t=fill[bg];"
                 f"[b]scale={fw * 2}:{fh * 2},setsar=1,zoompan=z='{z}':x='{x}':y='(ih-ih/zoom)/2':d={n}:s={fw}x{fh}:fps={frs}[fg];"
                 f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{mb}fps={frs_out},{tv}[o]")
            cmd += ["-filter_complex", g, "-map", "[o]"]
        else:
            cover = f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},setsar=1"
            cmd += ["-vf", f"{cover},zoompan=z='{z}':x='{x}':y='(ih-ih/zoom)/2':d={n}:s={W}x{H}:fps={frs},{mb}fps={frs_out},{tv}"]
        frs = frs_out
    cmd += ["-an", "-r", frs, "-t", f"{seconds:.3f}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15", "-pix_fmt", "yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv", str(out)]
    ffmpeg.run(["nice", "-n", "12", *cmd], timeout=600, what="animating a photo")
    return out


def key(path: str, effect: str, seconds: float, W: int, H: int, fps: float = 30.0, blur: bool = True) -> str:
    extra = ([] if abs(float(fps) - 30.0) < 1e-3 else [round(float(fps), 3)]) + ([] if blur else ["nb"])        # (a clip baked at 30 fps with blur keeps its old name)
    return hashlib.sha1(json.dumps([path, effect, round(seconds, 2), W, H, *extra]).encode()).hexdigest()[:14]
