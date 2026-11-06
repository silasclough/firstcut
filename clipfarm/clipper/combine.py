"""Several uploaded videos -> one source timeline, so clips are cut from across all of them.

Order: by each file's recorded creation_time when every file has one, else the upload order (the GUI
sends files sorted by name - camera/glasses names carry the timestamp).
Join: files from the same device (identical codec / size / frame rate / audio) are concatenated without
re-encoding (fast, lossless). Mixed files are first normalised - fitted into the first file's frame with
black bars as needed, 30 fps, 48 kHz stereo (silence where a file has no audio) - then joined.
The file boundaries are kept (parts.json) so the moment finder knows where one file ends and the next
begins, and each file's metadata is checked for Meta glasses before joining (joining drops tags).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.combine")
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v")


def _created(info: dict) -> datetime | None:
    t = (info.get("format", {}).get("tags") or {}).get("creation_time")
    for st in info.get("streams", []):
        t = t or (st.get("tags") or {}).get("creation_time")
    try:
        return datetime.fromisoformat(t.split(";")[0].strip().replace("Z", "+00:00")) if t else None    # (ffmpeg 9 can write it twice: "a;b")
    except ValueError:
        return None


def _signature(info: dict) -> tuple:
    from .glasses import _display_dims
    vs = ffmpeg.video_stream(info) or {}
    au = ffmpeg.audio_stream(info) or {}
    return (vs.get("codec_name"), vs.get("width"), vs.get("height"), vs.get("r_frame_rate"), vs.get("pix_fmt"),
            _display_dims(vs), au.get("codec_name"), au.get("sample_rate"), au.get("channels"))


def order_parts(parts: list[Path], infos: list[dict]) -> list[int]:
    times = [_created(i) for i in infos]
    if all(times) and len(set(times)) == len(times):
        return sorted(range(len(parts)), key=lambda k: times[k])
    return list(range(len(parts)))


def combine(parts: list[Path], out: Path, progress=None) -> dict:
    """Join `parts` into `out`. Returns {"files": [{name, start, end}], "mode": "copy"|"reencode"}."""
    from .glasses import _display_dims, detect
    infos = [ffmpeg.probe(p) for p in parts]
    idx = order_parts(parts, infos)
    parts, infos = [parts[k] for k in idx], [infos[k] for k in idx]
    durs = [float(i.get("format", {}).get("duration") or 0) for i in infos]
    devices = []
    for p in parts:
        try:
            devices.append(detect(p))
        except Exception as e:  # noqa: BLE001
            devices.append({"glasses": False, "error": str(e)[:120]})
    same = len({_signature(i) for i in infos}) == 1
    single = len(parts) == 1
    if single:                                              # one file is not a join: take it as it is (a rename: instant, no second copy on disk)
        if progress:
            progress("Preparing the video")
        if parts[0].resolve().parent.stat().st_dev == out.resolve().parent.stat().st_dev:
            os.replace(parts[0], out)
        else:
            shutil.move(str(parts[0]), str(out))
    elif progress:
        progress(f"Joining {len(parts)} videos" + ("" if same else " (matching their formats)"))
    if single:
        pass
    elif same:
        lst = out.with_suffix(".concat.txt")
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
        cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-map", "0", "-c", "copy", "-movflags", "+faststart", str(out)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        lst.unlink(missing_ok=True)
        if r.returncode != 0:
            log.warning(f"stream-copy join failed ({r.stderr.strip()[-200:]}); re-encoding instead")
            same = False
    if not same and not single:
        vs0 = ffmpeg.video_stream(infos[0]) or {}
        W, H = (x // 2 * 2 for x in _display_dims(vs0))
        W, H = W or 1080, H or 1920
        inputs, graph, labels = [], [], []
        for k, (p, info, dur) in enumerate(zip(parts, infos, durs)):
            inputs += ["-i", str(p)]
            graph.append(f"[{k}:v]scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
                         f"setsar=1,fps=30,format=yuv420p[v{k}]")
            if ffmpeg.audio_stream(info):
                graph.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo[a{k}]")
            else:
                graph.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={dur:.3f}[a{k}]")
            labels.append(f"[v{k}][a{k}]")
        graph.append("".join(labels) + f"concat=n={len(parts)}:v=1:a=1[v][a]")
        cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *inputs, "-filter_complex", ";".join(graph),
               "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "17",
               "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)]
        ffmpeg.run(cmd, timeout=4 * 3600, what="join videos")
    files, t = [], 0.0
    for p, dur, dev in zip(parts, durs, devices):
        files.append({"name": p.name.split("__", 1)[-1], "start": round(t, 2), "end": round(t + dur, 2),
                      "glasses": bool(dev.get("glasses")), "evidence": dev.get("evidence"),
                      "shape_match": bool(dev.get("shape_match")), "dims": dev.get("dims")})
        t += dur
    meta = {"files": files, "mode": "single" if single else ("copy" if same else "reencode")}
    out.with_name("parts.json").write_text(json.dumps(meta, indent=1))
    return meta


def boundary_note(files: list[dict], start: float = 0.0, end: float = 1e9) -> str:
    """'Joined from N files: file 2 "x.mp4" starts at 1:12, ...' for the prompts."""
    if not files or len(files) < 2:
        return ""
    inside = [f for f in files if start <= f["start"] < end and f["start"] > 0]
    if not inside:
        return ""
    fmt = lambda s: f"{int(s // 60)}:{s % 60:04.1f}"   # noqa: E731
    return (f"This footage is {len(files)} separate recordings joined in order. New recordings start at: " +
            ", ".join(f"{fmt(f['start'])} ({f['name']})" for f in inside) +
            ". A clip may run across a join when the moments belong together.")
