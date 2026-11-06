"""Background music for Clipper: pick a licensed track that fits, mix it under the voice.

pick()        the least-used track matching a mood (and optional genre), from assets/music/ (Kevin MacLeod +
              Openverse CC BY / CC0 library, each with a sidecar JSON carrying its credit line).
choose_vibe() reads what is said (transcript) and returns {mood, genre} - for whole-video jobs.
mix_filter()  ffmpeg graph: music trimmed to length, faded, loudness-matched and ducked under speech.
duck_points() the same ducking as volume keyframes (seconds, dB) for the Final Cut Pro export.

Levels (clipper.music): music sits at speech_lufs under talking clips and is pushed down further while
someone speaks (sidechain compression); clips with little talking get a fuller bed (ambient_lufs).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from ..core import ffmpeg
from ..core.log import get_logger
from ..core.music import Track, load_library
from ..core.state import JsonState

log = get_logger("clipper.music")
MOODS = ["chill", "upbeat", "funny", "suspense", "emotional", "epic", "adventure"]


def library(cfg) -> list[Track]:
    tracks, problems = load_library(cfg.path("music"), cfg.get("music.allowed_licenses", []))
    return [t for t in tracks if not t.placeholder]


def genres(cfg) -> list[str]:
    import json
    out = set()
    for side in Path(cfg.path("music")).glob("*.json"):
        try:
            g = json.loads(side.read_text()).get("genre")
        except (OSError, ValueError):
            continue
        if g:
            out.add(g)
    return sorted(out)


def _genre_of(t: Track) -> str | None:
    import json
    try:
        return json.loads(t.path.with_suffix(".json").read_text()).get("genre")
    except (OSError, ValueError):
        return None


def own_tracks(folder: Path | None) -> list[Track]:
    """Songs you put in the project's Music/ folder: yours to use, no licence checks, no credit line needed."""
    if not folder or not Path(folder).is_dir():
        return []
    from ..core.music import AUDIO_EXT
    return [_own(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in AUDIO_EXT and not p.name.startswith(".")]


def _own(p: Path) -> Track:
    """A song file as a Track: yours (no credit), or one fetched from Openverse / Epidemic (its credit line and licence travel with it)."""
    from .musicsources import credit_for, license_for
    credit = credit_for(p)
    if credit is None:
        return Track(path=p, title=p.stem, artist="(your music)", license="your own", credit="", moods=())
    return Track(path=p, title=p.stem, artist="(online music)", license=license_for(p) or "", credit=credit, moods=())


def resolve(cfg, spec: str) -> Track | None:
    """A track from a file path: the licensed library entry if it is one (with its credit), else your own file."""
    p = Path(spec)
    for t in library(cfg):
        if t.path == p or t.path.name == p.name:
            return t
    return _own(p) if p.is_file() else None


def pick(cfg, mood: str = "", genre: str = "", seed: str = "", exclude: set[str] | None = None, length_s: float = 0.0) -> Track | None:
    """Least-used track for this mood/genre (falls back to the mood, then to anything). `exclude` = file names not to pick.
    Your music library (musiclib, clipper.music_lib.roots) comes first; the bundled CC library (assets/music) is the fallback."""
    from . import musiclib
    lib = musiclib.for_cfg(cfg)
    if lib is not None:
        try:
            t = lib.pick(" ".join(x for x in (mood, genre) if x), length_s=length_s, seed=seed, exclude=exclude)
            if t:
                return t
        except Exception as e:  # noqa: BLE001 - the bundled library still works
            log.warning(f"music library pick failed: {e}")
    tracks = [t for t in library(cfg) if t.path.name not in (exclude or set())] or library(cfg)
    if not tracks:
        log.warning("no licensed music in assets/music (./cf music openverse) - rendering without music")
        return None
    pool = [t for t in tracks if genre and _genre_of(t) == genre and (not mood or mood in t.moods)] or \
        [t for t in tracks if genre and _genre_of(t) == genre] or \
        [t for t in tracks if mood and mood in t.moods] or tracks
    usage = JsonState(cfg.path("state") / "clipper_music.json")
    counts = usage.get("used", {}) or {}
    low = min(counts.get(t.key, 0) for t in pool)
    rng = random.Random(seed or None)
    choice = rng.choice([t for t in pool if counts.get(t.key, 0) == low])
    counts[choice.key] = counts.get(choice.key, 0) + 1
    usage.set("used", counts)
    return choice


VIBE_SCHEMA = {"type": "object", "properties": {
    "mood": {"type": "string", "enum": MOODS}, "genre": {"type": "string"}, "why": {"type": "string"}},
    "required": ["mood", "genre", "why"]}


def choose_vibe(llm, transcript_text: str, available_genres: list[str], title: str = "") -> dict:
    """What is SAID decides the music: the AI reads the transcript and picks a mood and a genre."""
    excerpt = transcript_text[:6000]
    prompt = (f"Video: {title}\nTranscript (excerpt):\n{excerpt}\n\n"
              f"Pick background music that fits what is being said and its emotional tone. "
              f"Genres available: {', '.join(available_genres) or 'any'}. Return one mood and one genre from that list.")
    try:
        res = llm.complete_json("music_vibe", "You are a music supervisor for short videos.", prompt, VIBE_SCHEMA)
        mood = res.data.get("mood") if res.data.get("mood") in MOODS else "chill"
        g = res.data.get("genre") if res.data.get("genre") in available_genres else ""
        return {"mood": mood, "genre": g, "why": res.data.get("why", ""), "by": res.provider}
    except Exception as e:  # noqa: BLE001
        log.warning(f"music vibe: {e}; using chill")
        return {"mood": "chill", "genre": "", "why": "default", "by": "fallback"}


def levels(cfg, words: list[dict], out_len: float) -> float:
    """Target loudness of the music bed: quieter under talking, fuller when there's little speech."""
    mc = cfg.get("clipper.music", {}) or {}
    rate = len(words) / max(1.0, out_len)
    return float(mc.get("speech_lufs", -36)) if rate >= float(mc.get("speech_words_per_s", 0.6)) \
        else float(mc.get("ambient_lufs", -26))


def song(path: str) -> Track | None:
    """A song the person chose (dropped on a project, or from their library): used as it is, no licence checks, no credit line."""
    p = Path(str(path or "")).expanduser()
    from ..core.music import AUDIO_EXT
    if not p.is_file() or p.suffix.lower() not in AUDIO_EXT | {".aif"}:
        return None
    return _own(p)


def clip_start(cfg, track: Track, out_len: float, seed: str) -> float:
    """Where a SHORT video's music starts: just before the song's best part (its drop) when the music library knows it, so a clip opens
    on the good bit, like a stock-music site's preview; else somewhere in the first third (start_offset)."""
    from . import musiclib
    try:
        lib = musiclib.for_cfg(cfg) or musiclib.MusicLibrary(cfg)
        rec = lib.info(track.path) if out_len <= 120 else None
    except Exception:  # noqa: BLE001
        rec = None
    if rec and rec.get("dur"):
        dur, best = float(rec["dur"]), float(rec.get("best") or 0)
        if dur > out_len + 1:
            return round(max(0.0, min(best - 0.5, dur - out_len - 0.5)), 2)
        return 0.0
    return start_offset(track, out_len, seed)


def start_offset(track: Track, out_len: float, seed: str) -> float:
    """Start somewhere in the first third so the same song doesn't always open the same way."""
    try:
        dur = ffmpeg.duration(track.path)
    except Exception:  # noqa: BLE001
        return 0.0
    room = max(0.0, min(dur * 0.35, dur - out_len - 2))
    return round(random.Random(int(hashlib.sha1(seed.encode()).hexdigest()[:8], 16)).uniform(0, room), 2) if room > 1 else 0.0


LOOP_XF = 2.0                    # seconds of crossfade where a song too short for the video starts again


def loop_plan(track_dur: float, start: float, total: float, xf: float = LOOP_XF) -> list[tuple[float, float, float]]:
    """How a song fills `total` seconds of video, playing from `start`: [(video second it starts at, song second it plays from, length)].
    A song that is long enough is one piece. A shorter one plays again from its beginning, each repeat overlapping the end of the one
    before by `xf` seconds (a crossfade, not a jump). The finished video's sound and the Final Cut project both follow this plan."""
    if track_dur <= 0 or start + total <= track_dur + 1e-3:
        return [(0.0, start, total)]
    xf = max(0.0, min(xf, track_dur / 3))
    first = max(0.1, track_dur - start)
    out = [(0.0, start, first)]
    t = first
    while t < total - 1e-3:
        at = t - xf
        ln = min(track_dur, total - at)
        out.append((round(at, 4), 0.0, round(ln, 4)))
        t = at + ln
    return out


def looped(track_path, plan: list[tuple[float, float, float]], xf: float, out_wav) -> None:
    """Render a loop_plan as one WAV (each repeat crossfaded into the next) - used when the song is shorter than the video."""
    cmd = [ffmpeg.bin_path("ffmpeg"), "-y", "-v", "error"]
    for _at, src, ln in plan:
        cmd += ["-ss", f"{src:.3f}", "-t", f"{ln:.3f}", "-i", str(track_path)]
    g, cur = [], "[0:a]aresample=48000,aformat=channel_layouts=stereo[p0]"
    g.append(cur)
    for k in range(1, len(plan)):
        g.append(f"[{k}:a]aresample=48000,aformat=channel_layouts=stereo[p{k}]")
    last = "p0"
    if xf < 0.05:                                                    # no crossfade asked for: the repeats simply follow each other
        g.append("".join(f"[p{k}]" for k in range(len(plan))) + f"concat=n={len(plan)}:v=0:a=1[x]")
        last = "x"
    for k in range(1, len(plan) if xf >= 0.05 else 1):
        g.append(f"[{last}][p{k}]acrossfade=d={xf:.3f}:c1=tri:c2=tri[x{k}]")
        last = f"x{k}"
    cmd += ["-filter_complex", ";".join(g), "-map", f"[{last}]", "-c:a", "pcm_s16le", str(out_wav)]
    ffmpeg.run(cmd, timeout=1800, what="looping the song")


def mix_filter(music_idx: int, out_len: float, music_lufs: float, voice: str | None, out: str,
               final_lufs: float = -14) -> list[str]:
    """Graph lines: [music_idx:a] -> trimmed, faded, levelled bed; ducked under `voice` if given -> [out]."""
    fade_out = max(0.0, out_len - 1.5)
    g = [f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{out_len:.3f},asetpts=PTS-STARTPTS,"
         f"loudnorm=I={music_lufs}:TP=-3:LRA=11,aresample=48000,"
         f"afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out:.3f}:d=1.5[mus]"]
    if voice:
        g += [f"[{voice}]aresample=48000,aformat=channel_layouts=stereo,asplit=2[vox][sc]",
              "[mus][sc]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400:makeup=1[mduck]",
              f"[vox][mduck]amix=inputs=2:duration=first:normalize=0,"
              f"loudnorm=I={final_lufs}:TP=-1.5:LRA=11,aresample=48000[{out}]"]
    else:
        g += [f"[mus]loudnorm=I={final_lufs}:TP=-1.5:LRA=11,aresample=48000[{out}]"]
    return g


def stem_filter(music_idx: int, out_len: float, music_lufs: float, voice: str, voice_lufs: float = -16.0) -> list[str]:
    """Graph lines giving the two halves of the mix separately: [vstem] the voice at a fixed loudness, [mstem] the music bed set
    `voice_lufs - music_lufs` below it and ducked while someone talks. They are measured, adjusted, then mixed (see polish)."""
    fade_out = max(0.0, out_len - 1.5)
    return [f"[{voice}]aresample=48000,aformat=channel_layouts=stereo,loudnorm=I={voice_lufs}:TP=-2:LRA=9,aresample=48000,asplit=2[vstem][sc]",
            f"[{music_idx}:a]aresample=48000,aformat=channel_layouts=stereo,atrim=0:{out_len:.3f},asetpts=PTS-STARTPTS,"
            f"loudnorm=I={music_lufs}:TP=-3:LRA=11,aresample=48000,afade=t=in:st=0:d=0.8,afade=t=out:st={fade_out:.3f}:d=1.5[mus]",
            "[mus][sc]sidechaincompress=threshold=0.02:ratio=8:attack=20:release=400:makeup=1[mstem]"]


def integrated(path) -> float | None:
    """Integrated loudness (LUFS, EBU R128) of an audio or video file; None if it can't be measured."""
    import re
    import subprocess
    r = subprocess.run([ffmpeg.bin_path("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128", "-f", "null", "-"],
                       capture_output=True, text=True, timeout=1800)
    m = re.findall(r"I:\s+(-?\d+(?:\.\d+)?)\s+LUFS", r.stderr)
    return float(m[-1]) if m else None


def meter_gap(voice_wav, music_wav) -> float | None:
    """Integrated loudness of the voice minus that of the (ducked) music bed, in LU: how far under the voice the music sits."""
    v, m = integrated(voice_wav), integrated(music_wav)
    return None if v is None or m is None else v - m


def duck_points(words: list[dict], out_len: float, bed_db: float = -14.0, under_db: float = -24.0,
                gap_s: float = 0.8, ramp: float = 0.25) -> list[tuple[float, float]]:
    """(time s, gain dB) keyframes: music dips to under_db while words are spoken, rises in longer gaps."""
    spans: list[list[float]] = []
    for w in words:
        if spans and w["start"] - spans[-1][1] <= gap_s:
            spans[-1][1] = max(spans[-1][1], w["end"])
        else:
            spans.append([w["start"], w["end"]])
    pts = [(0.0, under_db if spans and spans[0][0] < ramp else bed_db)]
    for a, b in spans:
        a0, b1 = max(0.0, a - ramp), min(out_len, b + ramp)
        pts += [(a0, bed_db), (max(a0, a), under_db), (min(b1, b), under_db), (b1, bed_db)]
    pts.append((out_len, bed_db))
    out, last_t = [], -1.0
    for t, v in sorted(pts):
        if t > last_t + 1e-3:
            out.append((round(t, 3), v))
            last_t = t
    return out
