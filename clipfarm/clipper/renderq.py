"""Render quality: what the pieces of a rough cut are encoded as, when the decode can be done by the hardware, and how a short decode is reported.

The delivered MP4 should have ONE lossy picture generation. The pieces (localproj.assemble) are intermediates: lossless x264 when the disk has room
(faster than the old crf 16), else crf 16 as before; ProRes only when asked for. The final encode (polish.run) is the only lossy step; its bitrate
does not depend on the length (core.ffmpeg.delivery_bitrate_mbit).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..core import ffmpeg

PRORES_MBIT_1080P = 110.0         # ProRes 422 LT at 1080p (about)
LOSSLESS_MBIT_1080P = 200.0       # x264 -qp 0 of ordinary footage (about 160 Mbit/s measured on hard footage, 1.2 GB a minute)
ROOM = 4.0                        # the disk must have this many times the estimated size free
DECODE_SMALLER = 1.5              # hardware decode only when the source is this much wider than the output (a 4K shoot made into 1080p)


def estimated_bytes(w: int | float, h: int | float, seconds: float, mbit_1080p: float) -> float:
    return mbit_1080p * 1e6 / 8 * (float(w) * float(h) / (1920.0 * 1080.0)) * max(0.0, float(seconds))


def free_bytes(path: Path | str) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return float(shutil.disk_usage(p).free)
    except OSError:
        return float("inf")


def intermediate_args(w: int, h: int, seconds: float, free: float, encoders: frozenset | set | None = None, mode: str = "auto") -> list[str]:
    """-> the video part of the encode arguments for a piece of the rough cut (the sound is always pcm_s16le).
    mode (the setting clipper.render.intermediate): "auto" = lossless x264 when the disk has room, else crf 16 as before; "prores" = ProRes 422 LT
    when this ffmpeg has it and the disk has room (measured: 6x slower than x264 and not closer to the source after the final encode, so it is
    only used when asked for); "lossless"; "crf16"."""
    enc = ffmpeg.encoders() if encoders is None else encoders
    prores = ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le"]
    lossless = ["-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", "-pix_fmt", "yuv420p"]      # (ultrafast is also FASTER than crf 16 veryfast: no rate search)
    plain = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p"]
    mode = (mode or "auto").lower()
    if mode == "crf16":
        return plain
    if mode == "prores" and "prores_ks" in enc and free > ROOM * estimated_bytes(w, h, seconds, PRORES_MBIT_1080P):
        return prores
    if mode in ("auto", "prores", "lossless") and "libx264" in enc and free > ROOM * estimated_bytes(w, h, seconds, LOSSLESS_MBIT_1080P):
        return lossless
    return plain


def hw_decode_args(src_w: int | float | None, out_w: int | float, enabled: bool = True) -> list[str]:
    """-> ["-hwaccel", "videotoolbox"] when a source much bigger than the output is cut on a Mac (the decode is the slow part of a 4K to 1080p cut)."""
    if not enabled or not src_w or float(src_w) < DECODE_SMALLER * float(out_w) or "videotoolbox" not in _hwaccels():
        return []
    return ["-hwaccel", "videotoolbox"]


def _hwaccels() -> frozenset:
    global _HW
    if _HW is None:
        import subprocess
        try:
            out = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=30).stdout
            _HW = frozenset(x.strip() for x in out.splitlines()[1:] if x.strip())
        except Exception:  # noqa: BLE001
            _HW = frozenset()
    return _HW


_HW: frozenset | None = None


UNTAGGED = {"", "unknown", "unspecified", "reserved", "gbr"}


def colour_matrix_in(info: dict) -> str:
    """The matrix to read a picture with: what its tags say, else what players assume when nothing is said (bt709 from 1280 px wide, bt601 below).
    `info` is an ffprobe result of the polish input; 'auto' lets the frame's own tags decide."""
    vs = ffmpeg.video_stream(info or {}) or {}
    space = str(vs.get("color_space") or "").lower()
    if space not in UNTAGGED:
        return "auto"
    try:
        w = int(vs.get("width") or 0)
    except (TypeError, ValueError):
        w = 0
    return "bt709" if w >= 1280 else "smpte170m"


def colour_filter(path: Path | str | None) -> str:
    """The scale stage that puts a picture on bt709/tv once (the tags that follow then tell the truth). An untagged source is read as a player
    would read it, not as ffmpeg's bt601 default, so HD footage without tags keeps its colours."""
    try:
        m = colour_matrix_in(ffmpeg.probe(path)) if path else "auto"
    except Exception:  # noqa: BLE001 - never a reason to lose a render
        m = "auto"
    return f"scale=in_color_matrix={m}:out_color_matrix=bt709:out_range=tv"
