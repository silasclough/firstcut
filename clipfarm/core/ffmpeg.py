"""ffmpeg/ffprobe helpers. Captions are rendered with Pillow, so no libass/freetype needed."""

from __future__ import annotations

import functools
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .log import get_logger

log = get_logger("ffmpeg")

EXTRA_PATHS = ["/opt/homebrew/bin", "/usr/local/bin"]


class FFmpegError(RuntimeError):
    pass


def which(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in EXTRA_PATHS:
        p = Path(d) / name
        if p.exists() and os.access(p, os.X_OK):
            return str(p)
    return None


@functools.lru_cache(maxsize=None)
def bin_path(name: str) -> str:
    p = which(name)
    if not p:
        raise FFmpegError(f"{name} not found on PATH (install with: brew install ffmpeg)")
    return p


def run(cmd: list[str], timeout: float | None = None, dry_run: bool = False, what: str = "ffmpeg",
        capture_stdout: bool = False) -> subprocess.CompletedProcess:
    cmd = [str(c) for c in cmd]
    if cmd[0] in ("ffmpeg", "ffprobe"):
        cmd[0] = bin_path(cmd[0])
    printable = shlex.join(cmd)
    if dry_run:
        log.info(f"[dry-run] would run {what}: {printable[:600]}")
        return subprocess.CompletedProcess(cmd, 0, b"", b"")
    log.debug(f"run {what}: {printable[:2000]}")
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE if capture_stdout else subprocess.DEVNULL,
                              stderr=subprocess.PIPE, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise FFmpegError(f"{what} timed out after {timeout}s") from e
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace").strip().splitlines()[-15:]
        raise FFmpegError(f"{what} exited {proc.returncode}:\n" + "\n".join(tail))
    return proc


def run_watched(cmd: list[str], stall_s: float = 60.0, timeout: float | None = None, what: str = "ffmpeg") -> subprocess.CompletedProcess:
    """Like run(), but ffmpeg reports its progress and is killed if it makes none for `stall_s` seconds. ffmpeg 8's scheduler can
    deadlock (0% CPU, forever); a run that is merely slow keeps reporting progress and is left alone."""
    import select
    import threading
    import time
    cmd = [str(c) for c in cmd]
    i = next(k for k, c in enumerate(cmd) if Path(c).name in ("ffmpeg", "ffmpeg.exe"))
    cmd[i + 1:i + 1] = ["-progress", "pipe:1", "-nostats", "-nostdin"]
    log.debug(f"run {what}: {shlex.join(cmd)[:2000]}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    err: list[bytes] = []
    th = threading.Thread(target=lambda: [err.append(x) for x in iter(proc.stderr.readline, b"")], daemon=True)
    th.start()
    start = last_move = time.monotonic()
    last_val = None
    killed = ""
    while proc.poll() is None:
        r, _, _ = select.select([proc.stdout], [], [], 2.0)
        if r:
            line = proc.stdout.readline()
            if line.startswith((b"out_time_us=", b"out_time_ms=", b"frame=", b"total_size=")) and line != last_val:
                if not line.startswith(b"out_time_us=N/A") and not line.endswith(b"=0\n"):
                    last_val, last_move = line, time.monotonic()
        now = time.monotonic()
        if now - last_move > stall_s:
            killed = f"{what} stalled (no progress for {stall_s:.0f}s)"
        elif timeout and now - start > timeout:
            killed = f"{what} timed out after {timeout}s"
        if killed:
            proc.kill()
            break
    proc.wait()
    th.join(2)
    if killed:
        raise FFmpegError(killed)
    if proc.returncode != 0:
        tail = b"".join(err).decode(errors="replace").strip().splitlines()[-15:]
        raise FFmpegError(f"{what} exited {proc.returncode}:\n" + "\n".join(tail))
    return subprocess.CompletedProcess(cmd, 0, b"", b"".join(err))


def probe(path: Path | str) -> dict:
    proc = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
               what="ffprobe", capture_stdout=True, timeout=60)
    return json.loads(proc.stdout.decode() or "{}")


def duration(path: Path | str) -> float:
    info = probe(path)
    d = info.get("format", {}).get("duration")
    if d is None:
        for s in info.get("streams", []):
            if s.get("duration"):
                d = s["duration"]
                break
    if d is None:
        raise FFmpegError(f"could not read duration of {path}")
    return float(d)


def video_stream(info: dict) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)


def audio_stream(info: dict) -> dict | None:
    return next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)


@functools.lru_cache(maxsize=None)
def _list(kind: str) -> str:
    try:
        return subprocess.run([bin_path("ffmpeg"), "-hide_banner", f"-{kind}"], capture_output=True,
                              timeout=30).stdout.decode(errors="replace")
    except Exception:  # noqa: BLE001
        return ""


def has_encoder(name: str) -> bool:
    return any(line.split()[1:2] == [name] for line in _list("encoders").splitlines() if line.strip())


def has_filter(name: str) -> bool:
    return any(line.split()[1:2] == [name] for line in _list("filters").splitlines() if line.strip())


@functools.lru_cache(maxsize=None)
def videotoolbox_works() -> bool:
    if not has_encoder("h264_videotoolbox"):
        return False
    try:
        proc = subprocess.run([bin_path("ffmpeg"), "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                               "testsrc2=s=320x568:d=0.5", "-c:v", "h264_videotoolbox", "-b:v", "1M",
                               "-f", "null", "-"], capture_output=True, timeout=60)
        return proc.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def pick_encoder(video_cfg: dict) -> str:
    want = video_cfg.get("encoder", "auto")
    if want == "auto":
        return "h264_videotoolbox" if videotoolbox_works() else "libx264"
    return want


def video_encode_args(video_cfg: dict) -> list[str]:
    enc = pick_encoder(video_cfg)
    if enc == "h264_videotoolbox":
        rate = video_cfg.get("videotoolbox_bitrate", "8M")
        args = ["-c:v", "h264_videotoolbox", "-b:v", rate, "-maxrate", rate, "-profile:v", "high"]
    else:
        args = ["-c:v", "libx264", "-preset", video_cfg.get("x264_preset", "medium"),
                "-crf", str(video_cfg.get("x264_crf", 20)), "-profile:v", "high"]
    return args + ["-pix_fmt", "yuv420p", "-r", str(video_cfg.get("fps", 30)), "-g", str(int(video_cfg.get("fps", 30)) * 2)]


def readable_seconds(path: Path | str) -> float:
    """How much of a video file can really be read. Files whose header promises more than the data holds (an export that
    was cut off) report their full length but run dry part-way - this finds where."""
    try:
        proc = subprocess.run([bin_path("ffprobe"), "-v", "quiet", "-select_streams", "v:0", "-show_entries",
                               "packet=pts_time", "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=600)
    except Exception:  # noqa: BLE001
        return duration(path)
    ts = []
    for line in proc.stdout.splitlines():
        try:
            ts.append(float(line.strip().rstrip(",")))
        except ValueError:
            continue
    return max(ts) if ts else duration(path)


def software_cfg(video_cfg: dict) -> dict:
    """Same settings but forcing the software encoder (the hardware one can return an empty file when it is busy)."""
    return {**video_cfg, "encoder": "libx264"}


def has_video(path: Path | str, min_bytes: int = 20_000) -> bool:
    """True if `path` is a real video: big enough, with a video stream and a duration (catches 'successful' empty files)."""
    p = Path(path)
    try:
        if not p.exists() or p.stat().st_size < min_bytes:
            return False
        info = probe(p)
        return video_stream(info) is not None and float((info.get("format") or {}).get("duration") or 0) > 0.2
    except Exception:  # noqa: BLE001
        return False


def audio_encode_args(video_cfg: dict) -> list[str]:
    return ["-c:a", "aac", "-b:a", video_cfg.get("audio_bitrate", "192k"), "-ar", "48000", "-ac", "2"]


def validate_output(path: Path, min_s: float, max_s: float, width: int, height: int) -> list[str]:
    """Basic output checks before a video can enter review. Returns a list of problems."""
    problems: list[str] = []
    if not Path(path).exists() or Path(path).stat().st_size < 50_000:
        return [f"output missing or too small: {path}"]
    info = probe(path)
    v, a = video_stream(info), audio_stream(info)
    if not v:
        problems.append("no video stream")
    elif (int(v.get("width", 0)), int(v.get("height", 0))) != (width, height):
        problems.append(f"resolution {v.get('width')}x{v.get('height')} != {width}x{height}")
    if not a:
        problems.append("no audio stream")
    dur = float(info.get("format", {}).get("duration", 0))
    if not (min_s - 0.5 <= dur <= max_s + 1.0):
        problems.append(f"duration {dur:.1f}s outside {min_s}-{max_s}s")
    return problems
