"""Small, fast-start copy of a clip for the swipe feed (432x768, ~0.5 Mbit/s, moov atom first).

The full-quality clip is what gets posted; this one only has to load instantly on a phone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..core import ffmpeg


def make(src: Path, dst: Path, nice: bool = False) -> bool:
    part = dst.with_name(dst.stem + ".part.mp4")
    base = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-vf", "scale=432:-2",
            "-pix_fmt", "yuv420p", "-r", "30", "-g", "30"]
    audio = ["-c:a", "aac", "-b:a", "64k", "-ar", "44100", "-ac", "1", "-movflags", "+faststart", str(part)]
    for enc in (["-c:v", "libx264", "-preset", "veryfast", "-crf", "31", "-maxrate", "450k", "-bufsize", "900k"],
                ["-c:v", "h264_videotoolbox", "-b:v", "450k"]):
        cmd = (["nice", "-n", "15"] if nice else []) + base + enc + audio
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=600)
            part.replace(dst)
            return True
        except Exception:  # noqa: BLE001 - try the next encoder
            part.unlink(missing_ok=True)
    return False
