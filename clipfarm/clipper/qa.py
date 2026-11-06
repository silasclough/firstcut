"""A quality check on the FINISHED video, run after every render (what a careful editor does before saying "done").

Every problem we found by hand in a single week would have been caught by this: a stretch of dead audio at the end (the second camera had
no microphone), a cut that came out at 60% of the length asked for, no B-roll or camera change for the whole first half, one picture held
for far too long. It looks only at the result (the audio and the picture of the exported file), so it also catches anything a bug upstream
let through. It writes Notes/quality.md next to the video, and logs the problems.

    audio     integrated loudness; any silence in the middle of the program (not the tail); a camera that recorded almost no sound
    picture   how often the picture changes (a camera switch, a B-roll cutaway, any real cut); the longest stretch with no change;
              whether the first half is as lively as the second
    length    against the length that was asked for
    clipping  the true peak (above -1 dBTP is clipping on most players), and where it happens
    frozen    a picture that does not move for seconds (a frozen frame, or a still card)
    flashes   a shot shorter than a second

Every issue has a level (bad = fix it, warn = look at it, info), plain words, where it is in the finished video (start/end, seconds) and
what to do (notes.quality_md writes it as Notes/Quality check.md).
"""

from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.qa")
SILENCE_DB, SILENCE_S = -50.0, 1.5          # this quiet for this long, inside the program, is a dropout
MAX_HOLD_S = 12.0                           # the twelve-second rule
SCENE = 0.28                                # how different two frames must be to count as a cut
QUIET_CAMERA_DB = -55.0                     # a camera whose speech averages below this has no usable microphone
PEAK_DBTP = -1.0                            # true peak above this clips on most players
FREEZE_S = 4.0                              # a picture this still for this long is worth a look
FLASH_S = 1.0                               # a shot shorter than this is a flash


def _run(args: list[str], timeout: float = 900.0) -> str:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    return p.stderr + p.stdout


def loudness(video: Path) -> float | None:
    out = _run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(video), "-vn", "-af", "ebur128", "-f", "null", "-"])
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
    return float(m[-1]) if m else None


def loud_and_peaks(video: Path, limit: float = PEAK_DBTP) -> tuple[float | None, float | None, list[tuple[float, float]]]:
    """(integrated LUFS, true peak dBTP, [(from, to) where it clips]). The loudness and true peak come from ebur128; where it clips, from
    the sample peak of every tenth of a second (ebur128's running true peak only ever goes up, so it can't say where)."""
    out = _run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(video), "-vn", "-af", "ebur128=peak=true", "-f", "null", "-"])
    i = re.findall(r"I:\s+(-?[\d.]+) LUFS", out)
    tp = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", out.split("True peak:")[-1]) if "True peak:" in out else []
    peak = None if not tp or tp[-1] == "-inf" else float(tp[-1])
    hot: list[tuple[float, float]] = []
    if peak is not None and peak > limit:
        p = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(video), "-vn", "-af",
                            "aresample=48000,asetnsamples=n=4800,astats=metadata=1:reset=1,ametadata=mode=print:key=lavfi.astats.Overall.Peak_level:file=-",
                            "-f", "null", "-"], capture_output=True, text=True, timeout=900)
        t = 0.0
        for line in p.stdout.splitlines():
            m = re.search(r"pts_time:(-?[\d.]+)", line)
            if m:
                t = float(m.group(1))
                continue
            m = re.search(r"Peak_level=(-?[\d.]+)", line)
            if m and float(m.group(1)) > limit - 0.5:                       # (a sample peak this close to the top: the true peak is over it)
                if hot and t - hot[-1][1] <= 0.5:
                    hot[-1] = (hot[-1][0], t + 0.1)
                else:
                    hot.append((t, t + 0.1))
    return (float(i[-1]) if i else None), peak, hot


def frozen(video: Path, min_s: float = FREEZE_S) -> list[tuple[float, float]]:
    """Stretches where the picture doesn't move. (freezedetect splits one still stretch at a keyframe: the pieces are joined.)"""
    out = _run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(video), "-an", "-vf", "scale=320:-2,freezedetect=n=-60dB:d=1", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"freeze_start: (-?[\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"freeze_end: (-?[\d.]+)", out)]
    ends += [ffmpeg.duration(video)] * (len(starts) - len(ends))
    joined: list[tuple[float, float]] = []
    for a, b in zip(starts, ends):
        if joined and a - joined[-1][1] <= 0.2:
            joined[-1] = (joined[-1][0], b)
        else:
            joined.append((a, b))
    return [(a, b) for a, b in joined if b - a >= min_s]


def silences(video: Path, noise_db: float = SILENCE_DB, min_s: float = SILENCE_S) -> list[tuple[float, float]]:
    out = _run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(video), "-vn", "-af", f"silencedetect=noise={noise_db}dB:d={min_s}", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", out)]
    ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", out)]
    dur = ffmpeg.duration(video)
    ends += [dur] * (len(starts) - len(ends))                    # a silence that runs to the end of the file
    return [(max(0.0, a), b) for a, b in zip(starts, ends)]


def cut_times(video: Path, threshold: float = SCENE) -> list[float]:
    """Times (s) where the picture changes: a camera switch, a cutaway, a jump cut."""
    out = _run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-i", str(video), "-an", "-vf", f"select='gt(scene,{threshold})',showinfo", "-f", "null", "-"])
    return sorted(float(t) for t in re.findall(r"pts_time:(-?[\d.]+)", out) if float(t) > 0.2)


def camera_health(paths: dict[str, str], probe_s: float = 40.0) -> list[dict]:
    """How loud each camera's own microphone is (mean level of a stretch from the middle of the file). -> [{"name", "mean_db", "usable"}]"""
    rows = []
    for name, path in paths.items():
        try:
            dur = ffmpeg.duration(Path(path))
            out = _run([ffmpeg.bin_path("ffmpeg"), "-nostats", "-ss", f"{max(0.0, dur / 2 - probe_s / 2):.1f}", "-t", f"{probe_s}", "-i", str(path), "-vn",
                        "-af", "volumedetect", "-f", "null", "-"], timeout=300)
            m = re.search(r"mean_volume: (-?[\d.]+) dB", out)
            if m:
                rows.append({"name": name, "mean_db": float(m.group(1)), "usable": float(m.group(1)) > QUIET_CAMERA_DB})
        except Exception as e:  # noqa: BLE001 - a health check never stops a render
            log.debug(f"camera health for {name} failed: {e}")
    return rows


def check(video: Path, target_s: float | None = None, cameras: dict[str, str] | None = None) -> dict:
    dur = ffmpeg.duration(video)
    issues: list[dict] = []
    lufs, peak, hot = loud_and_peaks(video)
    gaps = [(a, b) for a, b in silences(video) if a > 1.0 and (b < dur - 1.5 or b - a >= 2.5)]       # (a short silence at the very end is a fade-out)
    for a, b in gaps:
        issues.append({"level": "bad", "what": f"dead audio from {a:.1f} s to {b:.1f} s ({b - a:.1f} s of silence in the middle of the video)",
                       "start": a, "end": b, "do": "Listen there in Final Cut: a camera without a microphone, or a gap between takes. Cover it with "
                                                   "room tone or music, or tighten the cut."})
    cuts = cut_times(video)
    marks = [0.0] + cuts + [dur]
    holds = [(y - x, x) for x, y in zip(marks, marks[1:])]
    longest, at = max(holds) if holds else (dur, 0.0)
    first = sum(1 for t in cuts if t < dur / 2)
    second = len(cuts) - first
    if longest > MAX_HOLD_S + 2.0:
        issues.append({"level": "warn", "what": f"the same picture for {longest:.0f} s starting at {at:.0f} s (the rule is {MAX_HOLD_S:.0f})",
                       "start": at, "end": at + longest, "do": "Put a B-roll shot or a camera change in that stretch (Replace… on the video's page, or in Final Cut)."})
    flashes = [(x, d) for d, x in holds if d < FLASH_S and 0.2 < x < dur - FLASH_S]
    for x, d in flashes[:6]:
        issues.append({"level": "warn", "what": f"a shot of only {d:.1f} s", "start": x, "end": x + d,
                       "do": "Check it in Final Cut: extend the shot before or after over it, or trim it out."})
    for a, b in frozen(video):
        if b - a >= FREEZE_S and not any(a <= c <= b for c in cuts):
            issues.append({"level": "warn", "what": f"the picture doesn't move for {b - a:.0f} s (a frozen frame, or a still card)", "start": a, "end": b,
                           "do": "If it isn't meant to be still, look at the source clip there: it may have stuttered or dropped frames."})
    if dur > 40 and first == 0:
        issues.append({"level": "bad", "what": "the picture never changes in the first half (no camera switch, no B-roll)", "start": 0.0, "end": dur / 2,
                       "do": "Add B-roll to the first half, or make it again with a B-roll folder."})
    elif dur > 40 and first * 2 < second and second >= 4:
        issues.append({"level": "warn", "what": f"the first half is much quieter visually than the second ({first} picture changes against {second})",
                       "start": 0.0, "end": dur / 2, "do": "A couple more cutaways early on would keep people watching."})
    if target_s and (dur < 0.8 * target_s or dur > 1.25 * target_s):
        issues.append({"level": "warn", "what": f"it is {dur:.0f} s; {target_s:.0f} s was asked for",
                       "do": "Say \"make it N minutes\" in Change something, or set the length in the options and make it again."})
    if lufs is not None and not -17.0 <= lufs <= -11.0:
        issues.append({"level": "warn", "what": f"the loudness is {lufs:.1f} LUFS (the target is about -14)",
                       "do": "Raise or lower the whole mix in Final Cut so it sits near -14 LUFS for YouTube and social."})
    if peak is not None and peak > PEAK_DBTP:
        for a, b in hot[:6] or [(None, None)]:
            issues.append({"level": "bad", "what": f"the sound clips: its true peak reaches {peak:.1f} dBTP (above {PEAK_DBTP:.0f})" + (" here" if a is not None else ""),
                           "start": a, "end": b,
                           "do": "Lower the loudest part (often the music under a loud line) or put a limiter at -1 dBTP on the mix."})
    cams = camera_health(cameras or {})
    for c in cams:
        if not c["usable"]:
            issues.append({"level": "info", "what": f"{c['name']} recorded almost no sound ({c['mean_db']:.0f} dB): its audio is not used",
                           "do": "Nothing: another recording's sound is used instead."})
    return {"duration": round(dur, 1), "lufs": lufs, "peak": peak, "silences": [(round(a, 1), round(b, 1)) for a, b in gaps], "cuts": len(cuts),
            "cuts_first_half": first, "cuts_second_half": second, "longest_hold": round(longest, 1), "cameras": cams, "issues": issues}


def report_md(qa: dict) -> str:
    bad = [i for i in qa["issues"] if i["level"] == "bad"]
    head = "Nothing wrong found." if not [i for i in qa["issues"] if i["level"] != "info"] else \
        f"{len(bad)} problem(s) and {len([i for i in qa['issues'] if i['level'] == 'warn'])} warning(s)."
    lines = ["# Quality check of the finished video", "", head, ""]
    lines += [f"- {'PROBLEM' if i['level'] == 'bad' else 'warning' if i['level'] == 'warn' else 'note'}: {i['what']}" for i in qa["issues"]]
    lines += ["", f"Length {qa['duration']} s, loudness {qa['lufs']} LUFS, {qa['cuts']} picture changes ({qa['cuts_first_half']} in the first half, "
              f"{qa['cuts_second_half']} in the second), longest unchanged picture {qa['longest_hold']} s."]
    return "\n".join(lines) + "\n"
