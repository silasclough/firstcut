"""Recap reels: a long event video becomes a few short, fast, vertical reels.

Message: "Give it the finished video, get 3 reels you can post." The talk from the main video keeps going underneath, the picture is mostly B-roll
(slowed down when it was filmed at 50/60 fps), there are captions and music, and no graphics unless the B-roll folder holds a logo.

The planning is plain arithmetic with no ffmpeg and no AI, so it can be tested exactly:

    speed_for()      playback speed of one shot (60 fps -> 0.5, so it plays as 30 fps slow motion)
    slow_this()      whether a shot plays slowed (about 75% of the shots that can be)
    windows()        the stretches of the talk the reels use, from the AI's themes or, failing that, spread over the video
    bites()          where the speaker is shown for a moment (the opening, once in the middle)
    lay_out()        the reel's timeline: speaker bites and B-roll shots, each with its source time and speed

run() (called by Engine.process for mode "recap") owns the AI calls, the music and the render.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger

log = get_logger("clipper.recap")

W, H, FPS = 1080, 1920, 30.0
TARGET_S, DEFAULT_REELS = 30.0, 3
SLOW_FROM_FPS = 48.0                 # shots filmed at this rate or higher can play as slow motion
SLOW_SHARE = 0.75                    # "the majority": about three in four of the shots that can be slowed are
MAX_SLOW = 0.25                      # never slower than quarter speed (a 120 fps shot)
HOLD = (1.5, 3.2)                    # seconds a B-roll shot is on screen: fast paced
BITE_S = 2.6                         # seconds the speaker is shown
MAX_PER_FILE = 2
IMAGES = {".png", ".jpg", ".jpeg", ".webp"}
VIDEO = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".mts", ".mxf"}


# ------------------------------------------------------------------------------------------------------ speed
def speed_for(fps: float, slow: bool = True) -> float:
    """Playback speed of a shot filmed at `fps`, so that it plays at 30 fps with every frame kept: 60 -> 0.5, 50 -> 0.6, 120 -> 0.25.
    Shots at 30 fps or lower, and shots not chosen for slow motion, play at normal speed."""
    f = float(fps or 0)
    if abs(f - round(f)) < 0.1:                                            # 59.94 is 60, 119.88 is 120
        f = float(round(f))
    if not slow or f < SLOW_FROM_FPS:
        return 1.0
    return round(max(MAX_SLOW, FPS / f), 3)


def slow_this(fps: float, seen: int) -> bool:
    """Whether the next shot filmed at `fps` plays in slow motion, given how many shots that can be slowed came before it (`seen`): about `share`
    of them are (every fourth plays at normal speed: SLOW_SHARE), so a reel is not all slow motion. Shots under 48 fps never are."""
    if float(fps or 0) < SLOW_FROM_FPS:
        return False
    return seen % 4 != 3


# ------------------------------------------------------------------------------------------------------ the talk
def sentences(words: list[dict], gap: float = 0.7) -> list[dict]:
    """The talk as sentences: [{start, end, text}]. A sentence ends at . ! ? or at a pause longer than `gap` seconds."""
    out, cur = [], []
    for w in words:
        if cur and float(w["start"]) - float(cur[-1]["end"]) > gap:
            out.append(cur)
            cur = []
        cur.append(w)
        if re.search(r"[.!?]$", str(w.get("text") or "").strip()):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return [{"start": round(float(s[0]["start"]), 2), "end": round(float(s[-1]["end"]), 2), "text": " ".join(str(w["text"]) for w in s)} for s in out]


def snap(words: list[dict], a: float, b: float) -> tuple[float, float]:
    """(a, b) moved to the edges of the words, so a reel never opens or ends in the middle of a word."""
    for w in words:
        if float(w["start"]) <= a < float(w["end"]):
            a = float(w["start"]) if a - float(w["start"]) < float(w["end"]) - a else float(w["end"])
            break
    for w in words:
        if float(w["start"]) < b <= float(w["end"]):
            b = float(w["end"]) if float(w["end"]) - b < b - float(w["start"]) else float(w["start"])
            break
    return round(a, 2), round(b, 2)


def talk_in(words: list[dict], a: float, b: float) -> float:
    """Seconds of speech between a and b."""
    return sum(max(0.0, min(b, float(w["end"])) - max(a, float(w["start"]))) for w in words)


def spread(words: list[dict], total: float, n: int, target: float) -> list[tuple[float, float]]:
    """`n` windows of about `target` seconds, one from each equal part of the video, each on the stretch with the most speech in its part.
    This is what is used when the AI is not available."""
    n = max(1, n)
    part = total / n
    out = []
    for k in range(n):
        lo, hi = k * part, (k + 1) * part
        best, best_t = lo, -1.0
        t = lo
        while t + min(target, part) <= hi + 1e-6:
            tk = talk_in(words, t, t + min(target, part))
            if tk > best_t:
                best, best_t = t, tk
            t += 2.0
        a, b = snap(words, best, best + min(target, part))
        out.append((a, max(b, a + 5.0)))
    return out


def windows(words: list[dict], total: float, wanted: list[dict] | None, n: int, target: float) -> list[dict]:
    """The stretches of the talk the reels use: [{start, end, theme}], in the order they happen, none overlapping, none under 12 s or over 1.6x the
    target. `wanted` is what the AI suggested ([{start, end, theme}]); anything wrong with it is mended or replaced by `spread()`."""
    n = max(1, int(n))
    good: list[dict] = []
    for w in wanted or []:
        try:
            a, b = snap(words, max(0.0, float(w["start"])), min(total, float(w["end"])))
        except (KeyError, TypeError, ValueError):
            continue
        if b - a > target * 1.6:
            b = snap(words, a, a + target * 1.25)[1]
        if b - a >= 12.0 and not any(a < g["end"] and b > g["start"] for g in good):
            good.append({"start": a, "end": b, "theme": " ".join(str(w.get("theme") or "").split())[:80]})
    good.sort(key=lambda g: g["start"])
    good = good[:n]
    if not good and 5.0 <= total < 12.0:                                    # a very short video: one reel from all of it
        return [{"start": 0.0, "end": round(total, 2), "theme": ""}]
    while len(good) < n:                                                    # fill what the AI did not give: the stretch with most speech that is still free
        length = min(target, max(12.0, total / n))
        best, best_t = None, -1.0
        t = 0.0
        while t + length <= total + 1e-6:
            if not any(t < g["end"] + 1.0 and t + length > g["start"] - 1.0 for g in good):
                tk = talk_in(words, t, t + length)
                if tk > best_t:
                    best, best_t = t, tk
            t += 2.0
        if best is None:
            break
        a, b = snap(words, best, best + length)
        good.append({"start": a, "end": b, "theme": ""})
        good.sort(key=lambda g: g["start"])
    return good


def bites(length: float) -> list[tuple[float, float]]:
    """Where in a reel of `length` seconds the speaker is shown: the opening, and once about 55% of the way through (a long enough reel)."""
    if length < 8.0:
        return [(0.0, min(BITE_S, length))]
    out = [(0.0, BITE_S)]
    if length >= 20.0:
        m = round(length * 0.55, 2)
        out.append((m, round(m + BITE_S, 2)))
    return out


# ------------------------------------------------------------------------------------------------------ the B-roll
def candidates(files: list[dict], scenes: dict[str, list[dict]]) -> list[dict]:
    """Every usable B-roll scene: {id, file, path, start, end, what, quality, fps}. Poor ones (quality under 4) are left out."""
    out = []
    for f in files:
        for s in scenes.get(f["name"]) or []:
            q = float(s.get("quality") or 5)
            if q < 4 or float(s["end"]) - float(s["start"]) < 1.0:
                continue
            out.append({"id": len(out), "file": f["name"], "path": f["path"], "start": float(s["start"]), "end": float(s["end"]),
                        "what": str(s.get("what") or ""), "quality": q, "fps": float(f.get("fps") or 30.0)})
    return out


def lay_out(length: float, shots: list[dict], used_files: dict[str, int] | None = None) -> list[dict]:
    """The reel's timeline for `length` seconds of talk. `shots` are B-roll scenes in the order they should play (candidates() rows, the best
    first for this theme). Returns segments: {kind: "talk", at, d} for the speaker, {kind: "broll", at, d, path, ss, src_d, speed, id, what} for B-roll.
    A shot is held HOLD seconds (fast paced); a slowed shot reads half as much footage for the same time on screen. The same file is used
    at most MAX_PER_FILE times in one reel, and no scene twice. The last shot is cut to end the reel exactly."""
    used = dict(used_files or {})
    seen_fast = 0
    segs: list[dict] = []
    clock, k = 0.0, 0
    talk = bites(length)
    queue = [(i, s) for i, s in enumerate(shots)]
    seen: set[int] = set()
    while clock < length - 0.05:
        nxt = next(((a, b) for a, b in talk if b > clock + 1e-6), None)
        if nxt and nxt[0] <= clock + 1e-6:                                  # the speaker is on screen now
            d = min(nxt[1], length) - clock
            segs.append({"kind": "talk", "at": round(clock, 3), "d": round(d, 3)})
            clock += d
            continue
        room = (nxt[0] if nxt else length) - clock
        pick = next(((i, s) for i, s in queue if i not in seen and used.get(s["file"], 0) < MAX_PER_FILE), None)
        if pick is None:                                                    # not enough B-roll: the speaker carries on
            d = (nxt[0] if nxt else length) - clock
            segs.append({"kind": "talk", "at": round(clock, 3), "d": round(d, 3)})
            clock += d
            continue
        i, s = pick
        seen.add(i)
        hold = HOLD[0] + (HOLD[1] - HOLD[0]) * ((k * 37) % 100) / 100.0     # a little variety in how long each shot stays, the same every time
        k += 1
        if room - hold < 1.0:                                               # no room for another shot after this one: this one takes the rest
            hold = room
        fast = float(s.get("fps") or 0) >= SLOW_FROM_FPS
        sp = speed_for(s.get("fps") or 30.0, slow_this(s.get("fps") or 0, seen_fast))
        seen_fast += 1 if fast else 0
        avail = float(s["end"]) - float(s["start"])
        need = hold * sp
        if need > avail:                                                    # the scene is shorter than that: hold it as long as it lasts
            hold, need = avail / sp, avail
        if hold < 0.6:
            continue
        used[s["file"]] = used.get(s["file"], 0) + 1
        segs.append({"kind": "broll", "at": round(clock, 3), "d": round(hold, 3), "path": s["path"], "ss": round(float(s["start"]), 3),
                     "src_d": round(need, 3), "speed": sp, "id": s["id"], "what": s["what"], "fps": float(s.get("fps") or 30.0)})
        clock += hold
    if segs and clock > length + 1e-3:                                      # (a scene shorter than asked can overshoot by a frame or two)
        last = segs[-1]
        cut = clock - length
        last["d"] = round(last["d"] - cut, 3)
        if last["kind"] == "broll":
            last["src_d"] = round(last["d"] * last["speed"], 3)
    segs = [s for s in segs if s["d"] >= 0.1]
    if segs and 0.0 < length - sum(x["d"] for x in segs) < 0.5:             # (a sliver too short to show is given to the last shot)
        last = segs[-1]
        last["d"] = round(length - sum(x["d"] for x in segs[:-1]), 3)
        if last["kind"] == "broll":
            last["src_d"] = round(last["d"] * last["speed"], 3)
    return segs


def merge_talk(segs: list[dict]) -> list[dict]:
    """Two speaker segments in a row become one (when the B-roll ran out)."""
    out: list[dict] = []
    for s in segs:
        if out and s["kind"] == "talk" and out[-1]["kind"] == "talk":
            out[-1] = {**out[-1], "d": round(out[-1]["d"] + s["d"], 3)}
        else:
            out.append(dict(s))
    return out


def find_logo(folder: Path | None) -> Path | None:
    """A logo picture in the B-roll folder (its name says logo): the only graphic a recap reel gets."""
    if not folder or not Path(folder).is_dir():
        return None
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() in IMAGES and "logo" in p.stem.lower() and not p.name.startswith("."):
            return p
    return None


def broll_files(folder: Path | None, skip: Path | None = None, limit: int = 400) -> list[Path]:
    """The videos under the B-roll folder, in name order (not the main video itself, not hidden files)."""
    if not folder or not Path(folder).is_dir():
        return []
    out = [p for p in sorted(Path(folder).rglob("*")) if p.is_file() and p.suffix.lower() in VIDEO and not p.name.startswith(".")
           and not (skip and p.resolve() == Path(skip).resolve())]
    return out[:limit]


# ------------------------------------------------------------------------------------------------------ the AI parts
THEMES_SYSTEM = (
    "You plan short recap reels for an event video. From the transcript (with times) pick the stretches of talk that the reels will play "
    "underneath their pictures: each is about the target length, starts and ends on a sentence, is easy to follow without the rest of the "
    "video, and has a different theme from the others (for example: the welcome, the heart of the event, the thanks and the finish). They "
    "must not overlap. Give each a short title (3-6 words) that could caption the reel.")
THEMES_SCHEMA = {"type": "object", "properties": {"reels": {"type": "array", "items": {"type": "object", "properties": {
    "start": {"type": "number"}, "end": {"type": "number"}, "theme": {"type": "string"}}, "required": ["start", "end", "theme"]}}},
    "required": ["reels"]}
SHOTS_SYSTEM = (
    "You are a video editor choosing B-roll for one short, fast recap reel. You see the scenes of the footage (number, what it shows, quality) "
    "and the reel's theme and what is being said. Pick the scenes that fit the theme and the words best and tell a small story in the order "
    "they play: open on something that grabs, keep people and reactions, end on something that feels like a finish. Never two scenes that look "
    "alike in a row. Prefer scenes of high quality. Return scene numbers in play order.")
SHOTS_SCHEMA = {"type": "object", "properties": {"scenes": {"type": "array", "items": {"type": "integer"}}}, "required": ["scenes"]}


def ask_windows(ai, words: list[dict], total: float, n: int, target: float, notes: str) -> list[dict]:
    """The AI's suggestion for the reels' stretches of talk (windows() checks and mends it); [] if it cannot answer."""
    sents = sentences(words)
    step = max(1, len(sents) // 400)
    lines = "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in sents[::step])[:60000]
    try:
        res = ai.complete_json("recap_windows", THEMES_SYSTEM, (f"The person asked: {notes}\n" if notes else "")
                               + f"Make {n} reels of about {target:.0f} seconds each from this video ({total:.0f} s long).\nTranscript:\n{lines}",
                               THEMES_SCHEMA)
        return [r for r in res.data.get("reels") or [] if isinstance(r, dict)]
    except Exception as e:  # noqa: BLE001 - spread over the video instead
        log.warning(f"recap: choosing the stretches failed ({e}); spreading them over the video")
        return []


def ask_shots(ai, cands: list[dict], theme: str, said: str, want_s: float, banned: set[int], notes: str) -> list[dict]:
    """B-roll scenes for one reel, in play order: the AI's choice first, then the best of the rest, so there are always enough."""
    pool = [c for c in cands if c["id"] not in banned]
    if not pool:
        return []
    lines = "\n".join(f"S{c['id']}: {c['what']} (quality {c['quality']:.0f}/10)" for c in pool[:300])
    order: list[int] = []
    try:
        res = ai.complete_json("recap_shots", SHOTS_SYSTEM, (f"The person asked: {notes}\n" if notes else "")
                               + f"Reel theme: {theme or '(none)'}\nWhat is said: {said[:1200]}\nAbout {want_s:.0f} seconds of B-roll are needed "
                               f"(roughly {max(6, int(want_s / 2.3))} scenes).\nScenes:\n{lines}", SHOTS_SCHEMA)
        ok = {c["id"] for c in pool}
        order = [int(i) for i in res.data.get("scenes") or [] if int(i) in ok]
    except Exception as e:  # noqa: BLE001 - the best-looking scenes
        log.warning(f"recap: choosing shots failed ({e}); taking the best-looking")
    by_id = {c["id"]: c for c in pool}
    chosen = [by_id[i] for i in dict.fromkeys(order)]
    rest = sorted((c for c in pool if c["id"] not in set(order)), key=lambda c: (-c["quality"], c["id"]))
    return chosen + rest


# ------------------------------------------------------------------------------------------------------ the render
def _vf(seg: dict) -> str:
    """Filters that make one B-roll segment fill the 1080x1920 frame at 30 fps, slowed by its speed."""
    sp = float(seg.get("speed") or 1.0)
    return (f"setpts=(PTS-STARTPTS)/{sp:.4f},fps={FPS:.0f},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,format=yuv420p")


def render_reel(engine, job_id: str, index: int, src: Path, has_audio: bool, win: dict, segs: list[dict], words: list[dict], title: str,
                logo: Path | None, track, music_ss: float, opts: dict) -> dict:
    """Render one reel to clip_NN.mp4 and record it. `segs` is lay_out()'s timeline; `win` the stretch of the talk it plays under."""
    from ..core.captions import render_captions
    from . import music as MU
    from . import polish as PL
    from .engine import CAPTION_STYLES
    cfg, d = engine.cfg, engine.store.dir(job_id)
    a, b = float(win["start"]), float(win["end"])
    length = round(b - a, 3)
    stem = f"clip_{index:02d}"
    work = d / f"{stem}_work"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    vcfg = {**(cfg.get("video", {}) or {}), "width": W, "height": H, "fps": FPS, "videotoolbox_bitrate": "8M"}
    style = opts.get("caption_style", "bold")
    style = "bold" if style in ("mix", "") else style
    ccfg = {**(cfg.get("captions", {}) or {}), **{k: v for k, v in CAPTION_STYLES.get(style, {}).items() if k != "label"}}
    ccfg["center_y"] = 0.72
    cap_words = [{**w, "start": round(float(w["start"]) - a, 3), "end": round(float(w["end"]) - a, 3)} for w in words
                 if float(w["start"]) >= a - 0.01 and float(w["end"]) <= b + 0.01]
    caps = render_captions(cap_words, length, work, ccfg, vcfg) if style != "none" and cap_words else None

    inputs: list[str] = []
    g: list[str] = []
    for k, s in enumerate(segs):
        if s["kind"] == "talk":                                           # the speaker: the main video's own picture, cropped to the middle
            inputs += ["-ss", f"{a + s['at']:.3f}", "-t", f"{s['d']:.3f}", "-i", str(src)]
            g.append(f"[{k}:v]setpts=PTS-STARTPTS,fps={FPS:.0f},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1,format=yuv420p[s{k}]")
        else:
            inputs += ["-ss", f"{s['ss']:.3f}", "-t", f"{s['src_d'] + 0.1:.3f}", "-i", s["path"]]
            g.append(f"[{k}:v]trim=duration={s['src_d']:.3f},{_vf(s)},tpad=stop_mode=clone:stop_duration=1,trim=duration={s['d']:.3f},setpts=PTS-STARTPTS[s{k}]")
    g.append("".join(f"[s{k}]" for k in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[base]")
    cur = "base"
    if caps:
        inputs += ["-f", "concat", "-safe", "0", "-i", str(caps["ffconcat"])]
        ci = inputs.count("-i") - 1
        g.append(f"[{ci}:v]format=rgba[cap];[{cur}][cap]overlay=0:{caps['y']}:format=auto:eof_action=repeat,format=yuv420p[wc]")
        cur = "wc"
    if logo:
        inputs += ["-loop", "1", "-framerate", f"{FPS:.0f}", "-t", f"{length:.3f}", "-i", str(logo)]
        li = inputs.count("-i") - 1
        g.append(f"[{li}:v]scale={int(W * 0.16)}:-1,format=rgba,colorchannelmixer=aa=0.9[logo];[{cur}][logo]overlay=W-w-48:72:format=auto,format=yuv420p[wl]")
        cur = "wl"
    g.append(f"[{cur}]fade=t=out:st={max(0.0, length - 0.5):.3f}:d=0.5,format=yuv420p[vout]")
    (work / "graph_v.txt").write_text(";\n".join(g))

    afile = None
    lufs = float(cfg.get("video.loudness_lufs", -14))
    audio_out = has_audio or track is not None
    limit = max(600.0, length * 12)
    sound_err: list[BaseException] = []
    sound_thread = None
    if audio_out:                                                        # the sound is made on its own thread while the picture renders
        afile = work / "sound.m4a"

        def mix_sound():
            try:
                voice = work / "talk.wav"
                if has_audio:
                    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{a:.3f}", "-t", f"{length:.3f}", "-i", str(src), "-vn", "-ac", "2",
                                "-ar", "48000", "-c:a", "pcm_f32le", str(voice)], timeout=900, what="the talk")
                from . import soundmix as SMX
                sm = SMX.style_music({})                                          # (the same single ducking envelope as the long videos)
                PL.make_sound(voice if has_audio else src, [(0.0, length)], False, True, "", has_audio, track.path if track else None, music_ss, length,
                              MU.levels(cfg, cap_words, length, sm), lufs, work, afile, vcfg, limit, float((cfg.get("clipper.music", {}) or {}).get("min_gap_lu", 16)),
                              sm=sm, words=cap_words)
            except BaseException as e:  # noqa: BLE001 - raised below, once the picture is made
                sound_err.append(e)
        import threading
        sound_thread = threading.Thread(target=mix_sound, name="recap-sound", daemon=True)
        sound_thread.start()
    vfile = work / "picture.mp4"
    base = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", *inputs, "-/filter_complex", str(work / "graph_v.txt"), "-map", "[vout]", "-an", "-t", f"{length:.3f}"]
    ffmpeg.encode_with_fallback(lambda vc: [*base, *ffmpeg.video_encode_args(vc), str(vfile)], vcfg,
                                stall_s=float(cfg.get("clipper.render.stall_s", ffmpeg.STALL_S)), timeout=limit, what="the picture")
    if sound_thread is not None:
        sound_thread.join()
        if sound_err:
            raise sound_err[0]
    out, part = d / f"{stem}.mp4", d / f"{stem}.part.mp4"
    mux = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-i", str(vfile)] + (["-i", str(afile)] if afile else []) + ["-map", "0:v"] \
        + (["-map", "1:a"] if afile else []) + ["-c", "copy", "-metadata", f"title={title}", "-movflags", "+faststart", str(part)]
    ffmpeg.run(mux, timeout=300, what="joining picture and sound")
    part.replace(out)
    ffmpeg.run([ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error", "-ss", f"{min(2.0, length / 3):.2f}", "-i", str(out), "-frames:v", "1", "-vf", "scale=360:-2",
                "-q:v", "4", str(d / f"{stem}.jpg")], what="thumbnail")
    if caps:
        shutil.copy2(caps["srt"], d / f"{stem}.srt")
    credit = track.credit if track is not None and getattr(track, "credit", "") else ""
    (d / f"{stem}.txt").write_text((title + ("\n\n" + credit if credit else "")).strip() + "\n")
    shutil.rmtree(work, ignore_errors=True)
    slowed = sum(1 for s in segs if s["kind"] == "broll" and s["speed"] < 1.0)
    clip = {"index": index, "file": out.name, "thumb": f"{stem}.jpg", "preview": None, "srt": f"{stem}.srt" if caps else None, "title": title, "hook": "",
            "caption": title, "score": None, "reason": f"recap reel: {win.get('theme') or 'the talk from ' + str(round(a)) + ' s'}",
            "picked_by": "recap", "source_start": round(a, 2), "source_end": round(b, 2), "duration": round(length, 2), "cuts": len(segs) - 1,
            "layout": "recap", "caption_style": style, "pace": "fast", "hook_title": False, "profile": "recap", "polish": True,
            "shots": sum(1 for s in segs if s["kind"] == "broll"), "slow_motion_shots": slowed,
            "music": ({"title": track.title, "artist": track.artist, "license": track.license, "file": track.path.name} if track is not None else None),
            "rendered_at": int(time.time())}
    engine.store.set_clip(job_id, clip)
    return clip


def scenes_of(engine, files: list[Path], d: Path, progress=None) -> tuple[list[dict], dict[str, list[dict]]]:
    """(files, scenes) for the B-roll: each file probed (its frame rate matters) and described once by the vision model (cached in recap_scenes.json).
    File names are made unique by their place in the folder."""
    import json

    from . import broll as BR
    from . import polish as PL
    from .glasses import contact_sheets
    cache_p = d / "recap_scenes.json"
    try:
        have = json.loads(cache_p.read_text())
    except (OSError, ValueError):
        have = {}
    router, _where = engine.vision_router()
    out_files: list[dict] = []
    todo: list[tuple[dict, Path, int]] = []
    for p in files:
        try:
            info = ffmpeg.probe(p)
            vs = ffmpeg.video_stream(info)
            if not vs:
                continue
            dur = ffmpeg.duration(p)
            fps = float(PL._fps(vs))
        except Exception as e:  # noqa: BLE001 - one unreadable file is skipped
            log.info(f"recap: skipping {p.name}: {e}")
            continue
        if dur < 1.0:
            continue
        name = f"{p.parent.name}/{p.name}"
        st = p.stat()
        f = {"name": name, "path": str(p), "dur": dur, "fps": fps}
        out_files.append(f)
        key = have.get(name)
        if not (key and key.get("size") == st.st_size):
            todo.append((f, p, st.st_size))

    def look(item):
        f, p, size = item
        try:
            n_sheets = BR.sheets_for(f["dur"])
            sheets = contact_sheets(p, 0.0, f["dur"], max(1.0, f["dur"] / (12 * n_sheets)))[:n_sheets]
            if not sheets:
                return f, size, []
            times = ", ".join(f"{t:.1f}" for _, ts in sheets for t in ts)
            res = router.complete_json("broll_describe", BR.SYSTEM, f"File: {f['name']} ({f['dur']:.1f} s long). {len(sheets)} contact sheet(s); frame times in seconds: {times}. "
                                       "Describe the scenes. Use these times.", BR.SCHEMA, images=[b for b, _ in sheets])
            return f, size, BR._valid_scenes(res.data.get("scenes"), 0.0, f["dur"])
        except Exception as e:  # noqa: BLE001
            log.warning(f"recap: describing {f['name']} failed: {e}")
            return f, size, []
    # several files at once when an account AI answers; one at a time on this Mac's own picture model
    workers = 1 if getattr(router, "vision_local", False) else min(4, max(1, len(todo)))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="recap-look") as ex:
        for fu in as_completed([ex.submit(look, t) for t in todo]):
            f, size, sc = fu.result()
            done += 1
            if progress:
                progress(done / max(1, len(todo)), f"Looking at the B-roll ({done} of {len(todo)})")
            if sc:
                have[f["name"]] = {"size": size, "scenes": BR.split_long(sc)}
                cache_p.write_text(json.dumps(have))
    scenes = {f["name"]: (have.get(f["name"]) or {}).get("scenes") or [] for f in out_files}
    return out_files, scenes


def run(engine, job_id: str, d: Path, src: Path, words: list[dict]) -> None:
    """Make the reels of one job (Engine.process calls this for mode "recap")."""
    from . import music as MU
    job = engine.store.get(job_id)
    opts = job["options"]
    n = max(1, min(6, int(opts.get("recap_reels") or DEFAULT_REELS)))
    target = max(10.0, min(90.0, float(opts.get("recap_len") or TARGET_S)))
    notes = str(opts.get("prompt") or "")
    broll_dir = Path(opts["broll_dir"]).expanduser() if opts.get("broll_dir") else None
    total = ffmpeg.duration(src)
    info = ffmpeg.probe(src)
    has_audio = ffmpeg.audio_stream(info) is not None
    if not words:
        raise RuntimeError("there is no speech in this video to build the reels around")
    ai, _where = engine.router_for_now()
    creative = engine.creative_router()

    engine._progress(job_id, "finding", "Choosing the parts of the talk and looking at the B-roll", 0.15)
    from concurrent.futures import ThreadPoolExecutor
    pool_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recap-themes")
    wanted_f = pool_ex.submit(ask_windows, creative, words, total, n, target, notes)       # (the talk's themes are asked while the B-roll is described)

    engine._progress(job_id, "finding", "Looking at the B-roll", 0.2)
    files = broll_files(broll_dir, skip=src)
    if not files:                                                          # no B-roll folder: the video's own picture is the B-roll
        files = [src]
    bfiles, scenes = scenes_of(engine, files, d, lambda fr, m: engine._progress(job_id, "finding", m, 0.2 + 0.25 * fr))
    cands = candidates(bfiles, scenes)
    if not cands:
        raise RuntimeError("none of the footage could be used as B-roll (it all looked blurry or unusable, or could not be read)")
    logo = find_logo(broll_dir)
    wins = windows(words, total, wanted_f.result(), n, target)
    pool_ex.shutdown(wait=False)
    if not wins:
        raise RuntimeError("this video is too short (or has too little speech) to make recap reels from: it needs at least 5 seconds of talk")

    genres = MU.genres(engine.cfg)
    used_tracks: set[str] = set()
    used_scenes: set[int] = set()
    made, skipped = 0, []
    for k, win in enumerate(wins):
        idx = k + 1
        engine._progress(job_id, "rendering", f"Making reel {idx} of {len(wins)}", 0.5 + 0.48 * k / len(wins))
        a, b = float(win["start"]), float(win["end"])
        said = " ".join(str(w["text"]) for w in words if a <= float(w["start"]) <= b)
        pool = [c for c in cands if not (c["path"] == str(src) and c["start"] < b + 1 and c["end"] > a - 1)]    # (not the stretch that is the talk)
        order = ask_shots(creative, pool, win.get("theme") or "", said, b - a, used_scenes, notes)
        segs = merge_talk(lay_out(b - a, order))
        used_scenes |= {s["id"] for s in segs if s["kind"] == "broll"}
        title = " ".join(str(win.get("theme") or "").split()) or f"{job.get('title') or src.stem} · reel {idx}"
        track, music_ss = None, 0.0
        song = str(opts.get("song") or "")
        if song:
            track = MU.song(song)
        elif str(opts.get("music") or "auto") not in ("none", "off"):
            vibe = MU.choose_vibe(creative, said, genres, title)
            track = MU.pick(engine.cfg, mood=vibe.get("mood", "upbeat"), genre=vibe.get("genre", ""), seed=f"{job_id}:{idx}", exclude=used_tracks, length_s=b - a)
        if track is not None:
            used_tracks.add(track.path.name)
            music_ss = MU.clip_start(engine.cfg, track, b - a, f"{job_id}:{idx}")
        try:
            render_reel(engine, job_id, idx, src, has_audio, win, segs, words, title, logo, track, music_ss, opts)
            made += 1
        except Exception as e:  # noqa: BLE001 - one bad reel must not sink the others
            if e.__class__.__name__ == "Cancelled":
                raise
            log.warning(f"recap reel {idx} of {job_id} failed and was skipped: {e}")
            skipped.append(idx)
    if not made:
        raise RuntimeError(f"none of the {len(wins)} reels could be rendered")
    engine.store.update(job_id, status="done", progress=1.0, stage=f"{made} recap reel{'s' if made != 1 else ''} ready" + (f" ({len(skipped)} skipped after errors)" if skipped else ""))
    engine._archive(job_id)
