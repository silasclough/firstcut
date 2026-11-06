"""Footage shot in a log profile (Sony S-Log3, Canon C-Log3, Panasonic V-Log, DJI D-Log...) looks grey and flat until it is converted.
FirstCut converts it to normal Rec.709 in everything it renders and looks at itself: the long-form MP4, short clips, previews, stills,
the frames the catalogue searches and the pictures the AI describes. The Final Cut export is left alone (the user, 2026-11-06: "only
for the MP4 clips and not when you push it to Final Cut"): the editor grades the original log clips there.

    detect(path) -> {"profile": "slog3"|"slog2"|"clog3"|"vlog"|"dlog"|None, "gamut": ..., "how": "sidecar"|"metadata"|"look"|None}
    vf(path, cfg=None) -> "lut3d=file='…':interp=tetrahedral" (or "" for normal footage, or when Settings say "Leave as shot")
    cube(profile, gamut) -> Path of a generated 33-point .cube (cached in the app home)

How it is found, in order: Sony's sidecar XML (C0001M01.XML) or the same XML inside the MP4 (CaptureGammaEquation /
CaptureColorPrimaries: the FX30 work footage says "s-log3-cine" / "s-gamut3-cine"); vendor words in the file's tags; and last a careful
look at a few frames - only for cameras that can shoot log, and only when blacks are lifted, whites are low and colour is weak all at
once (normal footage never gets a LUT by guesswork).

The LUT: the published log curve -> scene linear, the camera gamut -> Rec.709 primaries (from the chromaticities, all D65), then a gentle
filmic roll-off (mid-grey 18 % -> 0.46 on screen, 90 % white -> about 0.86, no hard clip) and the Rec.709 transfer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path

import numpy as np

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.logcolor")

SIZE = 33
MID_OUT = 0.46                 # where 18 % grey lands on screen
NONE = {"profile": None, "gamut": None, "how": None}

_lock = threading.Lock()
_cube_lock = threading.Lock()                # (the cuts run in parallel: one LUT file is written once)
_mem: dict[str, dict] = {}

# Chromaticities (x, y) of R, G, B; all with a D65 white.
PRIMARIES = {
    "rec709": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "sgamut3cine": ((0.766, 0.275), (0.225, 0.800), (0.089, -0.087)),
    "sgamut3": ((0.730, 0.280), (0.140, 0.855), (0.100, -0.050)),        # (also S-Gamut)
    "cinemagamut": ((0.740, 0.270), (0.170, 1.140), (0.080, -0.100)),
    "vgamut": ((0.730, 0.280), (0.165, 0.840), (0.100, -0.030)),
    "dgamut": ((0.710, 0.310), (0.210, 0.880), (0.090, -0.080)),
}
D65 = (0.3127, 0.3290)
DEFAULT_GAMUT = {"slog3": "sgamut3cine", "slog2": "sgamut3", "clog3": "cinemagamut", "vlog": "vgamut", "dlog": "dgamut"}
LOG_MAKES = ("sony", "canon", "panasonic", "dji", "fujifilm", "blackmagic", "nikon", "z cam")


# ------------------------------------------------------------------------------------------------ the curves (input: 0-1 RGB as ffmpeg
# hands it to a filter, i.e. legal-range video already stretched to full: 10-bit code 64 -> 0.0, 940 -> 1.0)
def _cv(v):
    """The 10-bit code value (0-1 of 1023) of a full-range value: the S-Log3 / V-Log / D-Log formulas are written in code values."""
    return (np.asarray(v, float) * 876.0 + 64.0) / 1023.0


def slog3(v):
    x = _cv(v)
    return np.where(x >= 171.2102946929 / 1023.0, 10.0 ** ((x * 1023.0 - 420.0) / 261.5) * (0.18 + 0.01) - 0.01,
                    (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0))


def slog2(v):
    x = np.asarray(v, float)                                    # (S-Log / S-Log2 are written in full-range values)
    lin = np.where(x >= 0.030001222851889303, 10.0 ** ((x - 0.616596 - 0.03) / 0.432699) - 0.037584, (x - 0.030001222851889303) / 5.0)
    return 219.0 * lin / 155.0 * 0.9


def clog3(v):
    x = np.asarray(v, float)
    lin = np.where(x < 0.097465473, -(10.0 ** ((0.12783901 - x) / 0.36726845) - 1.0) / 14.98325,
                   np.where(x <= 0.15277891, (x - 0.12512219) / 1.9754798, (10.0 ** ((x - 0.12240537) / 0.36726845) - 1.0) / 14.98325))
    return lin * 0.9


def vlog(v):
    x = _cv(v)
    return np.where(x < 0.181, (x - 0.125) / 5.6, 10.0 ** ((x - 0.598206) / 0.241514) - 0.00873)


def dlog(v):
    x = _cv(v)
    return np.where(x <= 0.14, (x - 0.0929) / 6.025, (10.0 ** (3.89616 * x - 2.27752) - 0.0108) / 0.9892)


CURVES = {"slog3": slog3, "slog2": slog2, "clog3": clog3, "vlog": vlog, "dlog": dlog}


def _rgb_to_xyz(prim) -> np.ndarray:
    xy = np.array(prim, float)
    xyz = np.stack([xy[:, 0] / xy[:, 1], np.ones(3), (1 - xy[:, 0] - xy[:, 1]) / xy[:, 1]])     # columns: R, G, B at Y = 1
    w = np.array([D65[0] / D65[1], 1.0, (1 - D65[0] - D65[1]) / D65[1]])
    return xyz * np.linalg.solve(xyz, w)


def to_709(gamut: str) -> np.ndarray:
    return np.linalg.inv(_rgb_to_xyz(PRIMARIES["rec709"])) @ _rgb_to_xyz(PRIMARIES[gamut])


def _oetf709(L):
    L = np.clip(L, 0.0, 1.0)
    return np.where(L < 0.018, 4.5 * L, 1.099 * L ** 0.45 - 0.099)


def _shoulder(x, k):
    """A filmic curve (the widely used fit of the ACES film look): a small toe, a soft shoulder that eases into white. With the
    exposure k from _exposure: 18 % grey -> 0.46, 90 % white -> about 0.86, an S-Log3 highlight at code 800 -> 0.99, no flat clip."""
    y = np.maximum(x, 0.0) * k
    return np.clip((y * (2.51 * y + 0.03)) / (y * (2.43 * y + 0.59) + 0.14), 0.0, 1.0)


@__import__("functools").lru_cache(maxsize=1)
def _exposure() -> float:
    lo, hi = 0.05, 20.0                                         # the k that puts 18 % grey at MID_OUT
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(_oetf709(_shoulder(0.18, mid))) < MID_OUT:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def display(lin: np.ndarray) -> np.ndarray:
    """Scene-linear Rec.709 -> display Rec.709 (0-1)."""
    return np.clip(_oetf709(_shoulder(lin, _exposure())), 0.0, 1.0)


def convert(rgb: np.ndarray, profile: str, gamut: str | None = None) -> np.ndarray:
    """Log RGB (…, 3) in 0-1 -> Rec.709 display RGB in 0-1 (what the LUT holds)."""
    lin = CURVES[profile](np.asarray(rgb, float))
    lin = lin @ to_709(gamut or DEFAULT_GAMUT[profile]).T
    return display(lin)


# ------------------------------------------------------------------------------------------------ the .cube
def _home() -> Path:
    h = os.environ.get("VIDEOBUILDER_HOME")
    return Path(h) if h else Path(__file__).resolve().parents[2] / "data"


def cube(profile: str, gamut: str | None = None, folder: Path | None = None) -> Path:
    gamut = gamut if gamut in PRIMARIES else DEFAULT_GAMUT[profile]
    d = Path(folder or _home() / "luts")
    out = d / f"{profile}_{gamut}_to_rec709.cube"
    if out.exists() and out.stat().st_size > 1000:
        return out
    with _cube_lock:
        if out.exists() and out.stat().st_size > 1000:
            return out
        return _write_cube(out, profile, gamut)


def _write_cube(out: Path, profile: str, gamut: str) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    g = np.linspace(0.0, 1.0, SIZE)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")                 # (.cube: red changes fastest)
    rgb = np.stack([r.ravel(), gg.ravel(), b.ravel()], axis=1)
    o = convert(rgb, profile, gamut)
    tmp = out.with_suffix(f".{os.getpid()}.part")
    with tmp.open("w") as f:
        f.write(f'TITLE "FirstCut {profile} {gamut} to Rec.709"\nLUT_3D_SIZE {SIZE}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n')
        f.write("\n".join(f"{x:.6f} {y:.6f} {z:.6f}" for x, y, z in o) + "\n")
    tmp.replace(out)
    return out


def _esc(p: Path) -> str:
    return str(p).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


_setting = {"at": 0.0, "val": None}


def _app_setting() -> str | None:
    """The app's own setting, read fresh (every few seconds): changing it in Settings takes effect without a restart."""
    import time
    h = os.environ.get("VIDEOBUILDER_HOME")
    if not h:
        return None
    if time.time() - _setting["at"] > 5:
        try:
            v = ((json.loads((Path(h) / "settings.json").read_text()).get("defaults") or {}).get("long") or {}).get("log")
        except (OSError, ValueError, AttributeError):
            v = None
        _setting.update(at=time.time(), val=v)
    return _setting["val"]


def enabled(cfg=None) -> bool:
    """Settings › How it edits › Log footage: convert (the default) or leave as shot."""
    try:
        v = _app_setting() or (cfg.get("clipper.log_footage", "convert") if cfg is not None else "convert")
        return str(v or "convert") != "leave"
    except Exception:  # noqa: BLE001
        return True


def vf(path, cfg=None) -> str:
    """The filter that converts this file's picture ("" when it isn't log, or conversion is off). Never raises."""
    try:
        if not path or not enabled(cfg):
            return ""
        d = detect(path)
        if not d.get("profile"):
            return ""
        return f"lut3d=file='{_esc(cube(d['profile'], d.get('gamut')))}':interp=tetrahedral"
    except Exception as e:  # noqa: BLE001 - a picture without a LUT beats no picture
        log.warning(f"log footage check failed for {path}: {e}")
        return ""


def vf_prefix(path, cfg=None) -> str:
    """vf() with a trailing comma, for the front of a filter chain."""
    f = vf(path, cfg)
    return f + "," if f else ""


# ------------------------------------------------------------------------------------------------ finding it
GAMMA = {"s-log3-cine": "slog3", "s-log3": "slog3", "s-log2": "slog2", "s-log": "slog2"}
GAMUT = {"s-gamut3-cine": "sgamut3cine", "s-gamut3": "sgamut3", "s-gamut": "sgamut3", "rec709": "rec709"}


def _from_sony_xml(text: str) -> dict | None:
    m = re.search(r'name="CaptureGammaEquation"\s+value="([^"]+)"', text)
    if not m:
        return None
    prof = GAMMA.get(m.group(1).strip().lower())
    if not prof:
        return {**NONE, "how": "metadata", "gamma": m.group(1)}       # (rec709, hlg, s-cinetone: nothing to convert)
    g = re.search(r'name="CaptureColorPrimaries"\s+value="([^"]+)"', text)
    gamut = GAMUT.get(g.group(1).strip().lower(), DEFAULT_GAMUT[prof]) if g else DEFAULT_GAMUT[prof]
    return {"profile": prof, "gamut": gamut}


def _sidecar(p: Path) -> dict | None:
    for d in (p.parent, p.parent.parent):
        for name in (f"{p.stem}M01.XML", f"{p.stem}M01.xml"):
            x = d / name
            if x.is_file():
                try:
                    r = _from_sony_xml(x.read_text(errors="replace"))
                except OSError:
                    r = None
                if r:
                    return {**r, "how": r.get("how") or "sidecar"}
    return None


def _embedded(p: Path) -> tuple[dict | None, str]:
    """Sony writes the same NonRealTimeMeta XML into the MP4 (at its end, sometimes its start). -> (result, the camera maker)."""
    if p.suffix.lower() not in (".mp4", ".mov", ".mxf", ".m4v"):
        return None, ""
    make = ""
    try:
        with p.open("rb") as f:
            f.seek(0, 2)
            n = f.tell()
            for off in (max(0, n - 4_000_000), 0):
                f.seek(off)
                b = f.read(4_000_000)
                m = re.search(rb"<NonRealTimeMeta.*?</NonRealTimeMeta>", b, re.S)
                if m:
                    text = m.group(0).decode("utf-8", "replace")
                    mk = re.search(r'<Device manufacturer="([^"]*)"', text)
                    make = (mk.group(1) if mk else "").lower()
                    r = _from_sony_xml(text)
                    return ({**r, "how": r.get("how") or "metadata"} if r else None), make
    except OSError:
        pass
    return None, make


def _probe(p: Path) -> tuple[dict | None, str]:
    """Vendor words in the container / stream tags (Panasonic, Canon, DJI write some of these). -> (result, maker)."""
    try:
        info = ffmpeg.probe(p)
    except Exception:  # noqa: BLE001
        return None, ""
    tags = dict((info.get("format") or {}).get("tags") or {})
    for s in info.get("streams") or []:
        tags.update(s.get("tags") or {})
    blob = " ".join(f"{k}={v}" for k, v in tags.items()).lower()
    make = next((m for m in LOG_MAKES if m in blob), "")
    for words, prof in ((("v-log", "vlog"), "vlog"), (("c-log3", "clog3", "canon log 3"), "clog3"), (("d-log", "dlog"), "dlog"),
                        (("s-log3",), "slog3"), (("s-log2",), "slog2")):
        if any(w in blob for w in words):
            return {"profile": prof, "gamut": DEFAULT_GAMUT[prof], "how": "metadata"}, make
    return None, make


def look(frames: list[np.ndarray]) -> bool:
    """A careful look-test on RGB frames (0-1): every one lifted blacks, low whites, weak colour and low contrast at once."""
    if not frames:
        return False
    votes = 0
    for f in frames:
        y = f @ np.array([0.2126, 0.7152, 0.0722])
        p2, p98 = np.percentile(y, 2), np.percentile(y, 98)
        sat = float(np.mean(f.max(axis=-1) - f.min(axis=-1)))
        # Calibrated 2026-11-06: real FX30 S-Log3 has p2 ~0.30, p98 ~0.75, saturation ~0.05; the flattest normal clip seen (a camera
        # mic across a room, a workshop shoot C0018) p2 0.04, p98 0.80, saturation 0.10. Well apart on blacks and colour.
        if p2 > 0.12 and p98 < 0.82 and (p98 - p2) < 0.6 and sat < 0.08:
            votes += 1
    return votes == len(frames) and len(frames) >= 2


def _frames(p: Path, n: int = 3, w: int = 96, h: int = 54) -> list[np.ndarray]:
    try:
        dur = float(ffmpeg.duration(p) or 0.0)
    except Exception:  # noqa: BLE001
        return []
    out = []
    for k in range(n):
        t = dur * (k + 1) / (n + 1)
        r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-v", "error", "-ss", f"{t:.2f}", "-i", str(p), "-frames:v", "1",
                            "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=60)
        if len(r.stdout) == w * h * 3:
            out.append(np.frombuffer(r.stdout, np.uint8).reshape(h, w, 3) / 255.0)
    return out


def _cache_file() -> Path:
    return _home() / "luts" / "detected.json"


def _load() -> dict:
    try:
        return json.loads(_cache_file().read_text())
    except (OSError, ValueError):
        return {}


def _save(key: str, val: dict) -> None:
    try:
        f = _cache_file()
        f.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            d = _load()
            d[key] = val
            if len(d) > 20000:
                for k in list(d)[:5000]:
                    d.pop(k, None)
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(d))
            tmp.replace(f)
    except OSError:
        pass


def detect(path, use_look: bool = True) -> dict:
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return dict(NONE)
    key = f"{p}|{st.st_size}|{int(st.st_mtime)}"
    with _lock:
        hit = _mem.get(key)
    if hit is None:
        hit = _load().get(key)
    if hit is not None:
        with _lock:
            _mem[key] = hit
        return dict(hit)
    r = _sidecar(p)
    make = ""
    if r is None:
        r, make = _embedded(p)
    if r is None:
        r, m2 = _probe(p)
        make = make or m2
    if r is None and use_look and make in LOG_MAKES and look(_frames(p)):
        prof = {"sony": "slog3", "panasonic": "vlog", "canon": "clog3", "dji": "dlog"}.get(make, "slog3")
        r = {"profile": prof, "gamut": DEFAULT_GAMUT[prof], "how": "look"}
    r = {**NONE, **(r or {})}
    if r.get("profile"):
        log.info(f"log footage: {p.name} is {r['profile']} / {r['gamut']} ({r['how']})")
    with _lock:
        _mem[key] = r
    _save(key, r)
    return dict(r)
